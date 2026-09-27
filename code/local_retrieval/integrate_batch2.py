"""Import normalized batch 2 into the single production index and LanceDB.

Run with the local-retrieval venv while the web backend is stopped. The script is
restartable: table rows are checked by video ID, and artifacts are published only
after both tables have the expected row counts. Batch-1 index files are retained
under batch-2/manifests/pre_integration for recovery.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
from pathlib import Path
import shutil
import time

import lancedb
import numpy as np
import pyarrow as pa

import config

DATA = Path(config.DATA_DIR)
BATCH = DATA / "batch 2"
MANIFEST = BATCH / "manifests"
STAGE = MANIFEST / "integration_stage"
BACKUP = MANIFEST / "pre_integration"
INVENTORY = MANIFEST / "batch2_inventory.json"
CHUNK = 512


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False)
    os.replace(tmp, path)


def inventory() -> list[dict]:
    records = json.loads(INVENTORY.read_text(encoding="utf-8"))["records"]
    ids = [r["video_id"] for r in records]
    if len(ids) != len(set(ids)) or len(ids) != 614:
        raise ValueError("Batch 2 inventory has duplicate or missing video IDs")
    return sorted(records, key=lambda item: item["video_id"])


def keyframes(record: dict) -> tuple[np.ndarray, list[dict]]:
    video_id = record["video_id"]
    vectors = np.load(BATCH / "qwen3vl_img" / f"{video_id}.npy", mmap_mode="r")
    with (BATCH / "map-keyframes-k1r" / f"{video_id}.csv").open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        rows = list(csv.DictReader(stream))
    if vectors.shape != (len(rows), config.EMBEDDING_DIM):
        raise ValueError(f"KF vector/map mismatch: {video_id}")
    frames = [int(row["frame_idx"]) for row in rows]
    if len(set(frames)) != len(frames):
        raise ValueError(f"Duplicate frame: {video_id}")
    if any(not (BATCH / "Custom_Keyframes" / video_id / f"{frame:06d}.jpg").is_file()
           for frame in frames):
        raise ValueError(f"Missing active image: {video_id}")
    times = [float(row["pts_time"]) for row in rows]
    if any(a > b for a, b in zip(times, times[1:])):
        raise ValueError(f"Nonmonotonic PTS: {video_id}")
    if len(rows) and (not np.isfinite(vectors).all()
                      or np.any(np.linalg.norm(vectors, axis=1) < 0.98)):
        raise ValueError(f"Invalid KF vector: {video_id}")
    return vectors, rows


def prepare_artifacts(records: list[dict]) -> dict:
    STAGE.mkdir(parents=True, exist_ok=True)
    source_mapping = json.loads(Path(config.QWEN_MAPPING_PATH).read_text(encoding="utf-8"))
    if any(path.split("/", 1)[0].startswith(("M", "N", "S")) for path in source_mapping):
        raise RuntimeError("Production mapping already contains batch 2; use --verify")
    source_vectors = np.load(config.QWEN_INDEX_PATH, mmap_mode="r")
    if source_vectors.shape != (len(source_mapping), config.EMBEDDING_DIM):
        raise ValueError("Production index/mapping mismatch")
    total_new = sum(r["active_keyframes"] for r in records)
    combined_path = STAGE / "usearch_qwen_index_k1r.npy"
    combined_map = STAGE / "usearch_qwen_index_mapping_k1r.json"
    if not combined_path.exists():
        target = np.lib.format.open_memmap(combined_path, mode="w+", dtype="float32",
                                           shape=(len(source_mapping) + total_new, config.EMBEDDING_DIM))
        target[:len(source_mapping)] = source_vectors
        position = len(source_mapping)
        for n, record in enumerate(records, 1):
            vectors, _ = keyframes(record)
            size = len(vectors)
            target[position:position + size] = vectors
            position += size
            if n % 50 == 0:
                log(f"Index staged: {n}/{len(records)} videos")
        target.flush()
        del target
    mapping = source_mapping[:]
    for record in records:
        _, rows = keyframes(record)
        mapping.extend(f'{record["video_id"]}/{int(row["frame_idx"]):06d}.jpg' for row in rows)
    if len(mapping) != len(source_mapping) + total_new or len(mapping) != len(set(mapping)):
        raise ValueError("Combined mapping count/uniqueness failed")
    atomic_json(combined_map, mapping)
    category_path = DATA / "video_categories.json"
    categories = json.loads(category_path.read_text(encoding="utf-8"))
    fps = json.loads(Path(config.FPS_MAPPING_PATH).read_text(encoding="utf-8"))
    for record in records:
        video_id = record["video_id"]
        if video_id in categories:
            raise ValueError(f"Category collision: {video_id}")
        meta_path = BATCH / "media-info" / f"{video_id}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        categories[video_id] = {
            "video_id": video_id,
            "title": meta.get("title", "Camera giao thông" if video_id.startswith("N") else video_id),
            "author": meta.get("author", ""),
            "category": record["category"],
            "sub_category": "",
            "display_name": {"tin_tuc": "Tin tức", "dua_xe_dap": "Đua xe đạp",
                             "camera_giao_thong": "Camera giao thông"}[record["category"]],
        }
        if record["fps_csv"] is not None:
            fps[video_id] = float(record["fps_csv"])
        else:
            # Ảnh/map active rỗng: đọc FPS trực tiếp từ MOV, không dùng mặc định 25.
            import av
            with av.open(BATCH / record["video_file"]) as container:
                rate = container.streams.video[0].average_rate
                if not rate:
                    raise ValueError(f"Cannot determine FPS: {video_id}")
                fps[video_id] = float(rate)
    atomic_json(STAGE / "video_categories.json", categories)
    atomic_json(STAGE / "video_fps_mapping.json", fps)
    return {"keyframes": total_new, "mapping": len(mapping)}


def append_keyframes(records: list[dict], table) -> int:
    existing = table.count_rows()
    expected_old = 297332
    if existing < expected_old:
        raise RuntimeError(f"Existing KF table too short: {existing}")
    schema = table.schema
    imported = 0
    for n, record in enumerate(records, 1):
        video_id = record["video_id"]
        count = table.count_rows(filter=f"video_id = '{video_id}'")
        if count == record["active_keyframes"]:
            imported += count
            continue
        if count:
            raise RuntimeError(f"Partial import {video_id}: {count}; inspect before retry")
        vectors, rows = keyframes(record)
        for start in range(0, len(rows), CHUNK):
            sub = rows[start:start + CHUNK]
            vec = np.asarray(vectors[start:start + CHUNK], dtype=np.float32)
            data = {
                "id": [f"{video_id}/{int(row['frame_idx']):06d}.jpg" for row in sub],
                "video_id": [video_id] * len(sub),
                "category": [record["category"]] * len(sub),
                "sub_category": [""] * len(sub),
                "frame_id": [int(row["frame_idx"]) for row in sub],
                "timestamp_sec": [float(row["pts_time"]) for row in sub],
                "vector": [list(vector) for vector in vec],
                "asr_text": [""] * len(sub), "asr_start": [0.0] * len(sub),
                "asr_end": [0.0] * len(sub),
            }
            if "boxes_json" in schema.names:
                data["boxes_json"] = [""] * len(sub)
            if "obj_scores" in schema.names:
                data["obj_scores"] = [[0.0] * 80 for _ in sub]
            table.add(pa.Table.from_pydict(data, schema=schema))
        imported += len(rows)
        if n % 25 == 0:
            log(f"LanceDB keyframes: {n}/{len(records)} videos, {imported:,} rows")
    return imported


def shot_data(record: dict, rows: list[dict]) -> tuple[list[dict], np.ndarray, dict[str, str]]:
    video_id = record["video_id"]
    vector_path = BATCH / "output-video-encode/video_shot_indices/vectors" / f"{video_id}_shots.npy"
    if not vector_path.exists():
        return [], np.empty((0, config.EMBEDDING_DIM), dtype=np.float32), {}
    vectors = np.load(vector_path, mmap_mode="r")
    metadata = json.loads((BATCH / "output-video-encode/video_shot_indices/metadata" /
                           f"{video_id}_shots.json").read_text(encoding="utf-8"))
    if vectors.shape != (len(metadata), config.EMBEDDING_DIM):
        raise ValueError(f"Shot vector/metadata mismatch: {video_id}")
    times = [float(row["pts_time"]) for row in rows]
    accepted = []
    selected = []
    links: dict[str, tuple[float, str]] = {}
    for shot in metadata:
        left = bisect.bisect_left(times, float(shot["start_sec"]) - 0.001)
        right = bisect.bisect_right(times, float(shot["end_sec"]) + 0.001)
        if left == right:
            continue
        position = int(shot["vector_row"])
        if not 0 <= position < len(vectors) or not np.isfinite(vectors[position]).all():
            raise ValueError(f"Invalid shot vector: {shot['shot_id']}")
        accepted.append(shot)
        selected.append(position)
        anchor = float(shot.get("anchor_pts_time") or (shot["start_sec"] + shot["end_sec"]) / 2)
        for row in rows[left:right]:
            path = f"{video_id}/{int(row['frame_idx']):06d}.jpg"
            distance = abs(float(row["pts_time"]) - anchor)
            if path not in links or distance < links[path][0]:
                links[path] = (distance, shot["shot_id"])
    return accepted, np.asarray(vectors[selected], dtype=np.float32), {
        path: pair[1] for path, pair in links.items()}


def append_shots(records: list[dict], table) -> dict:
    old_map = json.loads(Path(config.KF_SHOT_MAP_PATH).read_text(encoding="utf-8"))
    old_meta = json.loads(Path(config.SHOT_METADATA_MAP_PATH).read_text(encoding="utf-8"))
    new_count = 0
    skipped = 0
    for n, record in enumerate(records, 1):
        if not record["shot_vectors"]:
            continue
        video_id = record["video_id"]
        _, rows = keyframes(record)
        accepted, vectors, links = shot_data(record, rows)
        skipped += int(record["shot_vectors"]) - len(accepted)
        count = table.count_rows(filter=f"video_id = '{video_id}'")
        if count not in (0, len(accepted)):
            raise RuntimeError(f"Partial shot import {video_id}: {count}/{len(accepted)}")
        if count == 0 and accepted:
            for start in range(0, len(accepted), CHUNK):
                sub = accepted[start:start + CHUNK]
                vec = vectors[start:start + CHUNK]
                data = {
                    "shot_id": [s["shot_id"] for s in sub], "video_id": [video_id] * len(sub),
                    "shot_index": [int(s["shot_index"]) for s in sub],
                    "start_sec": [float(s["start_sec"]) for s in sub],
                    "end_sec": [float(s["end_sec"]) for s in sub],
                    "duration_sec": [float(s["duration_sec"]) for s in sub],
                    "start_frame": [round(float(s["start_sec"]) * float(s["fps"])) for s in sub],
                    "end_frame": [round(float(s["end_sec"]) * float(s["fps"])) for s in sub],
                    "anchor_frame_id": [int(s["anchor_frame_id"]) for s in sub],
                    "cut_file": [""] * len(sub), "fps": [float(s["fps"]) for s in sub],
                    "category": [record["category"]] * len(sub),
                    "sub_category": [""] * len(sub),
                    "vector": [list(vector) for vector in vec],
                }
                table.add(pa.Table.from_pydict(data, schema=table.schema))
        new_count += len(accepted)
        old_map.update(links)
        old_meta.update({s["shot_id"]: {
            **{key: s[key] for key in ("shot_id", "video_id", "shot_index", "start_sec",
                                            "end_sec", "duration_sec", "anchor_frame_id", "fps")},
            "start_frame": round(float(s["start_sec"]) * float(s["fps"])),
            "end_frame": round(float(s["end_sec"]) * float(s["fps"])),
            "cut_file": "",
        } for s in accepted})
        if n % 50 == 0:
            log(f"LanceDB shots: {n}/{len(records)} videos, {new_count:,} rows")
    atomic_json(STAGE / "kf_shot_map.json", old_map)
    atomic_json(STAGE / "shot_metadata_map.json", old_meta)
    return {"imported": new_count, "skipped_no_keyframe": skipped}


def publish() -> None:
    BACKUP.mkdir(parents=True, exist_ok=True)
    names = ("usearch_qwen_index_k1r.npy", "usearch_qwen_index_mapping_k1r.json",
             "video_categories.json", "video_fps_mapping.json", "kf_shot_map.json",
             "shot_metadata_map.json")
    for name in names:
        source = DATA / name
        backup = BACKUP / name
        staged = STAGE / name
        if not staged.exists():
            raise FileNotFoundError(staged)
        if not backup.exists():
            shutil.copy2(source, backup)
        os.replace(staged, source)
        log(f"Published {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    records = inventory()
    db = lancedb.connect(config.LANCEDB_PATH)
    kf = db.open_table("keyframes_k1r")
    shots = db.open_table("video_shots")
    if args.verify:
        mapping = json.loads(Path(config.QWEN_MAPPING_PATH).read_text(encoding="utf-8"))
        vectors = np.load(config.QWEN_INDEX_PATH, mmap_mode="r")
        assert len(mapping) == len(vectors) == kf.count_rows()
        assert len(mapping) == 297332 + sum(r["active_keyframes"] for r in records)
        assert len(json.loads((DATA / "video_categories.json").read_text(encoding="utf-8"))) == 1487
        log(f"Verified {len(mapping):,} KF, {shots.count_rows():,} shot, 1,487 videos")
        return
    if kf.count_rows() == 297332:
        stats = prepare_artifacts(records)
        log(f"Staged {stats['keyframes']:,} batch-2 KF vectors")
    elif not (STAGE / "usearch_qwen_index_k1r.npy").exists():
        raise RuntimeError("DB partly imported but staging index absent")
    imported_kf = append_keyframes(records, kf)
    shot_stats = append_shots(records, shots)
    expected = 297332 + sum(r["active_keyframes"] for r in records)
    if kf.count_rows() != expected or imported_kf != expected - 297332:
        raise RuntimeError("KF count check failed; artifacts not published")
    if shots.count_rows() != 111601 + shot_stats["imported"]:
        raise RuntimeError("Shot count check failed; artifacts not published")
    if "obj_scores" in kf.schema.names:
        kf.drop_columns(["obj_scores", "boxes_json"])
    publish()
    atomic_json(MANIFEST / "integration_report.json", {
        "keyframes_total": kf.count_rows(), "batch2_keyframes": imported_kf,
        "shots_total": shots.count_rows(), "batch2_shots": shot_stats["imported"],
        "shots_skipped_no_keyframe": shot_stats["skipped_no_keyframe"],
        "asr_batch2": "pending", "shot_M": "pending", "rtdetr": "disabled",
    })
    log(f"Import complete: {kf.count_rows():,} KF, {shots.count_rows():,} shot")


if __name__ == "__main__":
    main()
