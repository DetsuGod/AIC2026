"""Normalize batch 2 files without changing the original frame identity.

Run from code/local_retrieval/venv:
    python normalize_batch2.py --dry-run
    python normalize_batch2.py --apply
    python normalize_batch2.py --verify

All moves stay inside Data/batch 2. The operation is resumable and records each
completed rename in manifests/normalize_journal.jsonl. Source vectors for rows
without an image are retained under manifests/source_vectors.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

import config


BASE = Path(config.DATA_DIR) / "batch 2"
MANIFESTS = BASE / "manifests"
JOURNAL = MANIFESTS / "normalize_journal.jsonl"
M_PACKAGE = BASE / "keyframe/keyframes_M/877da08c0cb0"
S_PACKAGE = BASE / "keyframe/keyframes_S/60e9d63ec893"
SHOT_PACKAGES = (
    BASE / "embedding-shot/b2-shots-s-qwen3vl",
    BASE / "embedding-shot/traffic-shot-embedding",
)


def ensure_internal(path: Path) -> None:
    if path == BASE or BASE not in path.parents:
        raise ValueError(f"Path is outside batch 2: {path}")


def source_image_roots() -> tuple[Path, ...]:
    return (BASE / "keyframe/keyframes", M_PACKAGE / "keyframes", S_PACKAGE / "keyframes")


def planned_moves() -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    for root in source_image_roots():
        if root.is_dir():
            moves.extend((item, BASE / "Custom_Keyframes" / item.name)
                         for item in sorted(root.iterdir()) if item.is_dir())
    for root, target in (
        (BASE / "kf-qwen-embedding/qwen", BASE / "qwen3vl_img"),
        (BASE / "kf-qwen-embedding/map-keyframes", BASE / "map-keyframes-k1r"),
    ):
        if root.is_dir():
            moves.extend((item, target / item.name)
                         for item in sorted(root.iterdir()) if item.is_file())
    for package in SHOT_PACKAGES:
        for kind in ("vectors", "metadata"):
            root = package / kind
            if root.is_dir():
                moves.extend((item, BASE / "output-video-encode/video_shot_indices" / kind / item.name)
                             for item in sorted(root.iterdir()) if item.is_file())
        for kind in ("embedding.json", ".status"):
            src = package / kind
            if src.exists():
                moves.append((src, BASE / "output-video-encode/video_shot_indices/provenance" / package.name / kind))
        root = package / "map-keyframes"
        if root.is_dir():
            moves.extend((item, BASE / "output-video-encode/video_shot_indices/provenance" /
                          package.name / "map-keyframes" / item.name)
                         for item in sorted(root.iterdir()) if item.is_file())
    for item in sorted((BASE / "media-info").glob("S01_V*.json")):
        moves.append((item, item.with_name(item.name.replace("S01_V", "S01-V", 1))))
    for src in (BASE / "kf-qwen-embedding/README.md",):
        if src.is_file():
            moves.append((src, MANIFESTS / "source_embedding_README.md"))
    for src in (BASE / "keyframe/keyframes_M", BASE / "keyframe/keyframes_S"):
        # Keep extraction reports, run configs and original package maps.
        if src.is_dir():
            moves.append((src, MANIFESTS / "provenance" / src.name))
    if (BASE / "keyframe/map-keyframes").is_dir():
        moves.append((BASE / "keyframe/map-keyframes", MANIFESTS / "source_keyframe_maps_N"))
    return moves


def write_journal(action: str, source: Path, target: Path) -> None:
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as output:
        output.write(json.dumps({"action": action, "source": str(source.relative_to(BASE)),
                                 "target": str(target.relative_to(BASE))}, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def move(source: Path, target: Path) -> None:
    ensure_internal(source)
    ensure_internal(target)
    if not source.exists():
        if target.exists():
            return  # A previous run completed this move.
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
    write_journal("move", source, target)


def active_rows(video_id: str, source: bool = False) -> tuple[list[int], list[dict[str, str]], list[dict[str, object]]]:
    mapping = BASE / "map-keyframes-k1r" / f"{video_id}.csv"
    source_mapping = MANIFESTS / "source_embedding_maps" / mapping.name
    if source and source_mapping.exists():
        mapping = source_mapping
    with mapping.open(newline="", encoding="utf-8-sig") as input_file:
        rows = list(csv.DictReader(input_file))
    keep: list[int] = []
    dropped: list[dict[str, object]] = []
    image_dir = BASE / "Custom_Keyframes" / video_id
    for position, row in enumerate(rows):
        frame_id = int(row["frame_idx"])
        if (image_dir / f"{frame_id:06d}.jpg").is_file():
            keep.append(position)
        else:
            dropped.append({"video_id": video_id, "source_row": position, "frame_idx": frame_id,
                            "reason": "skipped_missing_image"})
    return keep, rows, dropped


def filter_missing_images() -> dict[str, int]:
    totals = {"source_rows": 0, "active_rows": 0, "skipped_missing_image": 0, "videos_filtered": 0}
    failures: list[dict[str, object]] = []
    for mapping in sorted((BASE / "map-keyframes-k1r").glob("*.csv")):
        video_id = mapping.stem
        keep, rows, dropped = active_rows(video_id, source=True)
        vector_file = BASE / "qwen3vl_img" / f"{video_id}.npy"
        source_vector = MANIFESTS / "source_vectors" / vector_file.name
        vectors = np.load(source_vector if source_vector.exists() else vector_file,
                          mmap_mode="r", allow_pickle=False)
        if vectors.shape != (len(rows), 2048):
            raise ValueError(f"Vector/map mismatch: {video_id}: {vectors.shape}, {len(rows)} rows")
        # Windows refuses to rename an NPY while the mmap handle is open.
        del vectors
        totals["source_rows"] += len(rows)
        totals["active_rows"] += len(keep)
        totals["skipped_missing_image"] += len(dropped)
        failures.extend(dropped)
        if not dropped:
            continue
        totals["videos_filtered"] += 1
        source_map = MANIFESTS / "source_embedding_maps" / mapping.name
        if not source_vector.exists():
            move(vector_file, source_vector)
        if not source_map.exists():
            move(mapping, source_map)
        if vector_file.exists() != mapping.exists():
            raise RuntimeError(f"Partial active output for {video_id}")
        if not vector_file.exists():
            raw = np.load(source_vector, mmap_mode="r", allow_pickle=False)
            tmp_vector = vector_file.with_suffix(".npy.tmp")
            with tmp_vector.open("wb") as output:
                np.save(output, np.asarray(raw[keep]))
            tmp_vector.replace(vector_file)
            tmp_map = mapping.with_suffix(".csv.tmp")
            with tmp_map.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                for new_index, source_index in enumerate(keep, 1):
                    row = dict(rows[source_index])
                    row["n"] = str(new_index)
                    writer.writerow(row)
            tmp_map.replace(mapping)
            write_journal("filter_missing_image", source_vector, vector_file)
    report = MANIFESTS / "skipped_missing_images.json"
    report.write_text(json.dumps({"totals": totals, "rows": failures}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return totals


def verify() -> dict[str, object]:
    images = BASE / "Custom_Keyframes"
    vectors = BASE / "qwen3vl_img"
    maps = BASE / "map-keyframes-k1r"
    media = BASE / "media-info"
    shot_root = BASE / "output-video-encode/video_shot_indices"
    video_ids = {p.stem for p in (BASE / "video").iterdir() if p.suffix.lower() in (".mp4", ".mov")}
    image_ids = {p.name for p in images.iterdir() if p.is_dir()}
    vector_ids = {p.stem for p in vectors.glob("*.npy")}
    map_ids = {p.stem for p in maps.glob("*.csv")}
    if not image_ids.issubset(video_ids) or video_ids != vector_ids or video_ids != map_ids:
        raise ValueError(f"ID mismatch: video={len(video_ids)}, images={len(image_ids)}, "
                         f"vectors={len(vector_ids)}, maps={len(map_ids)}")
    if any(media.glob("S01_V*.json")):
        raise ValueError("S media-info aliases remain")
    active_count = 0
    for video_id in sorted(video_ids):
        keep, rows, dropped = active_rows(video_id)
        if dropped or np.load(vectors / f"{video_id}.npy", mmap_mode="r", allow_pickle=False).shape[0] != len(rows):
            raise ValueError(f"Active vector/map/image mismatch: {video_id}")
        active_count += len(keep)
    shot_vectors = list((shot_root / "vectors").glob("*.npy"))
    shot_metadata = list((shot_root / "metadata").glob("*.json"))
    if {p.stem for p in shot_vectors} != {p.stem for p in shot_metadata}:
        raise ValueError("Shot vector/metadata IDs differ")
    return {"videos": len(video_ids), "videos_without_images": len(video_ids - image_ids),
            "active_keyframes": active_count,
            "shot_vector_files": len(shot_vectors), "shot_metadata_files": len(shot_metadata),
            "media_info": len(list(media.glob("*.json"))),
            "missing_shot_m": True, "missing_asr": True}


def write_inventory() -> None:
    """Record actual post-migration coverage without inventing pending data."""
    records = []
    shot_root = BASE / "output-video-encode/video_shot_indices/vectors"
    for video in sorted((BASE / "video").iterdir()):
        if video.suffix.lower() not in (".mp4", ".mov"):
            continue
        video_id = video.stem
        map_path = BASE / "map-keyframes-k1r" / f"{video_id}.csv"
        with map_path.open(newline="", encoding="utf-8-sig") as input_file:
            rows = list(csv.DictReader(input_file))
        image_dir = BASE / "Custom_Keyframes" / video_id
        shot_path = shot_root / f"{video_id}_shots.npy"
        records.append({
            "video_id": video_id,
            "video_file": str(video.relative_to(BASE)).replace("\\", "/"),
            "media_info": (BASE / "media-info" / f"{video_id}.json").is_file(),
            "category": "tin_tuc" if video_id.startswith("M") else
                        "dua_xe_dap" if video_id.startswith("S") else "camera_giao_thong",
            "fps_csv": float(rows[0]["fps"]) if rows else None,
            "active_keyframes": len(rows),
            "jpg_files": len(list(image_dir.glob("*.jpg"))) if image_dir.is_dir() else 0,
            "shot_vectors": int(np.load(shot_path, mmap_mode="r", allow_pickle=False).shape[0])
                            if shot_path.is_file() else None,
            "asr": "pending",
            "object": "disabled",
        })
    result = {"schema_version": 1, "video_count": len(records), "records": records,
              "pending": ["ASR batch 2", "video shot embedding M"],
              "media_info_missing_N": True}
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    (MANIFESTS / "batch2_inventory.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--dry-run", action="store_true")
    choice.add_argument("--apply", action="store_true")
    choice.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify()
        write_inventory()
        print(json.dumps(result, ensure_ascii=False))
        return
    actions = planned_moves()
    targets = [target for _, target in actions]
    if len(targets) != len(set(targets)):
        raise ValueError("Two source files have the same target")
    for source, target in actions:
        ensure_internal(source)
        ensure_internal(target)
        if target.exists():
            raise FileExistsError(f"Preflight collision: {target}")
    if args.dry_run:
        counts: dict[str, int] = {}
        for _, target in actions:
            key = str(target.relative_to(BASE).parts[0])
            counts[key] = counts.get(key, 0) + 1
        print(json.dumps({"moves": len(actions), "by_target_root": counts,
                          "examples": [[str(a.relative_to(BASE)), str(b.relative_to(BASE))]
                                       for a, b in actions[:12]]}, ensure_ascii=False, indent=2))
        return
    for source, target in actions:
        move(source, target)
    filtered = filter_missing_images()
    result = verify()
    write_inventory()
    print(json.dumps({"filter": filtered, "verify": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
