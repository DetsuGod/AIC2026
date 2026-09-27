"""Coverage-aware lexical retrieval for selected batch-1 shot captions."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from . import config
except ImportError:
    import config


SOURCE = Path(config.DATA_DIR) / "batch 1/caption/qwen3vl_8b_ctx_L23_L26.json"
INDEX_DIR = SOURCE.parent / "retrieval_index"


def build_index() -> dict:
    """Validate shot joins and create a compact CPU TF-IDF index once."""
    captions = json.loads(SOURCE.read_text(encoding="utf-8"))
    shot_meta = json.loads(Path(config.SHOT_METADATA_MAP_PATH).read_text(encoding="utf-8"))
    kf_to_shot = json.loads(Path(config.KF_SHOT_MAP_PATH).read_text(encoding="utf-8"))
    mapping = json.loads(Path(config.QWEN_MAPPING_PATH).read_text(encoding="utf-8"))

    # Only production KFs can introduce candidates. The original map has aliases;
    # iterating production mapping avoids duplicate contributions.
    representatives: dict[str, tuple[float, str]] = {}
    for path in mapping:
        shot_id = kf_to_shot.get(path)
        if not shot_id or shot_id not in shot_meta:
            continue
        meta = shot_meta[shot_id]
        anchor = int(meta.get("anchor_frame_id") or 0)
        try:
            frame = int(path.split("/", 1)[1].split(".", 1)[0])
        except ValueError:
            continue
        distance = abs(frame - anchor)
        if shot_id not in representatives or distance < representatives[shot_id][0]:
            representatives[shot_id] = (distance, path)

    accepted = []
    mismatched = []
    for item in captions:
        shot_id = f"{item['video_id']}_shot_{int(item['shot_idx']):04d}"
        meta = shot_meta.get(shot_id)
        representative = representatives.get(shot_id)
        if not meta or not representative or not item.get("caption", "").strip():
            mismatched.append({"shot_id": shot_id, "reason": "missing_shot_or_keyframe"})
            continue
        # Caption timestamps are rounded to whole seconds in the source filename;
        # metadata retains frame-level decimals, so a <1.01s delta is expected.
        if abs(float(item["shot_start_sec"]) - float(meta["start_sec"])) > 1.01 or \
           abs(float(item["shot_end_sec"]) - float(meta["end_sec"])) > 1.01:
            mismatched.append({"shot_id": shot_id, "reason": "shot_time_mismatch"})
            continue
        accepted.append({"shot_id": shot_id, "keyframe": representative[1],
                         "caption": item["caption"]})

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                                 max_features=80000, dtype=np.float32, sublinear_tf=True)
    matrix = vectorizer.fit_transform(item["caption"] for item in accepted)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    save_npz(INDEX_DIR / "captions.npz", matrix, compressed=True)
    joblib.dump(vectorizer, INDEX_DIR / "vectorizer.joblib", compress=3)
    (INDEX_DIR / "rows.json").write_text(json.dumps(
        [{"shot_id": item["shot_id"], "keyframe": item["keyframe"]} for item in accepted],
        ensure_ascii=False), encoding="utf-8")
    report = {"source": len(captions), "indexed": len(accepted),
              "excluded": len(mismatched), "vocabulary": len(vectorizer.vocabulary_),
              "excluded_samples": mismatched[:100]}
    (INDEX_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    return report


class CaptionEngine:
    def __init__(self):
        if not (INDEX_DIR / "captions.npz").exists():
            self.available = False
            self.shot_ids = set()
            return
        self.vectorizer = joblib.load(INDEX_DIR / "vectorizer.joblib")
        self.matrix = load_npz(INDEX_DIR / "captions.npz").tocsr()
        self.rows = json.loads((INDEX_DIR / "rows.json").read_text(encoding="utf-8"))
        if self.matrix.shape[0] != len(self.rows):
            raise ValueError("Caption index row mismatch")
        self.shot_ids = {row["shot_id"] for row in self.rows}
        self.available = True

    def search(self, query: str, limit: int = 1000) -> list[dict]:
        if not self.available or not query.strip():
            return []
        vector = self.vectorizer.transform([query])
        if vector.nnz == 0:
            return []
        scores = (self.matrix @ vector.T).toarray().ravel()
        positive = np.flatnonzero(scores > 0)
        if len(positive) > limit:
            subset = np.argpartition(scores[positive], -limit)[-limit:]
            positive = positive[subset]
        positions = sorted(positive, key=lambda index: (-scores[index], self.rows[index]["shot_id"]))
        return [{**self.rows[index], "score": float(scores[index]), "rank": rank}
                for rank, index in enumerate(positions, 1)]


if __name__ == "__main__":
    print(build_index())
