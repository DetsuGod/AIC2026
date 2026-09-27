"""Read-only smoke checks for the integrated batch-2 production index."""
import json
import sys
from pathlib import Path

import numpy as np

import config
from caption_engine import CaptionEngine
from search_engine import VideoSearchEngine


def main() -> None:
    engine = VideoSearchEngine()
    assert len(engine.mapping) == 476140
    assert engine.lancedb_table.count_rows() == 476140
    assert engine.shot_table.count_rows() == 186102
    assert engine.rtdetr_index is None
    assert "obj_scores" not in engine.lancedb_table.schema.names
    assert "boxes_json" not in engine.lancedb_table.schema.names

    vectors = np.load(config.QWEN_INDEX_PATH, mmap_mode="r")
    for video_id, category in (("M01_V001", "tin_tuc"),
                               ("S01-V001", "dua_xe_dap"),
                               ("N001-V001", "camera_giao_thong")):
        path = next(path for path in engine.mapping if path.startswith(video_id + "/"))
        frame = int(path.split("/", 1)[1].split(".", 1)[0])
        resolved = engine.resolve_keyframe(video_id, frame)
        assert resolved["video_file_exists"] and resolved["exact_match"]
        assert Path(config.keyframe_directory(video_id), f"{frame:06d}.jpg").is_file()
        results = engine.search(query_text="sample visual query", query_vec=np.asarray(
            vectors[engine.path_to_idx[path]], dtype=np.float32), enable_asr=False,
            video_id=video_id, category=category, top_k=5, dedup_mode="none",
            return_shots=False)
        assert results and results[0]["video_id"] == video_id, video_id
        print(video_id, "top", results[0]["frame_id"], "at", results[0]["timestamp_sec"])

    caption = CaptionEngine()
    assert caption.available and len(caption.rows) == 51650
    hits = caption.search("đạp xe đạp thi đấu trên đường nhựa", 10)
    assert hits and hits[0]["keyframe"] in engine.path_to_idx
    print("caption top", hits[0]["shot_id"], "score", round(hits[0]["score"], 4))
    cap_path = hits[0]["keyframe"]
    fused = engine.search(query_text="đạp xe đạp thi đấu trên đường nhựa",
                          query_vec=np.asarray(vectors[engine.path_to_idx[cap_path]], dtype=np.float32),
                          enable_asr=False, video_id=cap_path.split("/", 1)[0], top_k=10,
                          rrf_use_caption=True, dedup_mode="none", return_shots=False)
    assert fused and any(row["caption_available"] for row in fused)
    print("caption fusion", len(fused), "top rank", fused[0]["rank_caption"])

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web/backend"))
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    cats = client.get("/api/categories").json()
    assert cats["total_videos"] == 1487
    assert cats["taxonomy"]["camera_giao_thong"]["count"] == 298
    image = client.get("/images/N001-V001/000105.jpg")
    assert image.status_code == 200 and image.headers["content-type"].startswith("image/")
    mov = client.get("/api/video-file/N001-V001", headers={"Range": "bytes=0-63"})
    assert mov.status_code == 206 and len(mov.content) == 64
    assert client.get("/api/keyframe-boxes/N001-V001/000105").json()["boxes"] == []
    print("HTTP image + MOV Range + taxonomy PASS")

    report = json.loads((Path(config.BATCH2_DIR) / "manifests/integration_report.json").read_text(
        encoding="utf-8"))
    assert report["asr_batch2"] == "pending" and report["shot_M"] == "pending"
    print("PASS", report)


if __name__ == "__main__":
    main()
