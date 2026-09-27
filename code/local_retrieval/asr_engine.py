import os
import json
import re
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import config
except ImportError:
    from . import config


class ASREngine:
    def __init__(self, mapping_path: str = None):
        self.mapping_path = mapping_path or config.ASR_MAPPING_PATH
        self.fps_mapping_path = config.FPS_MAPPING_PATH
        self.segments: List[Dict[str, Any]] = []
        self.kf_to_speech: Dict[str, List[Dict[str, Any]]] = {}
        self.kf_to_seg_id: Dict[str, str] = {}
        self.video_to_segments: Dict[str, List[Dict[str, Any]]] = {}
        self.fps_mapping: Dict[str, float] = {}
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._loaded = False

    def load(self) -> bool:
        if not self._loaded:
            # 1. Nạp FPS Mapping
            if os.path.exists(self.fps_mapping_path):
                with open(self.fps_mapping_path, "r", encoding="utf-8") as f:
                    self.fps_mapping = json.load(f)

            if not os.path.exists(self.mapping_path):
                print(f"[ASREngine] ⚠️ Chưa tìm thấy file {self.mapping_path}. Kênh ASR sẽ tạm thời rỗng.")
                return False

            print(f"[ASREngine] Đang nạp ASR Mapping từ {self.mapping_path}...")
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                self.segments = json.load(f)

            # Xây dựng bảng tra ngược O(1) từ Keyframe -> Danh sách lời thoại & Segment ID
            self.kf_to_speech = {}
            self.kf_to_seg_id = {}
            self.video_to_segments = {}

            corpus = []
            for idx, seg in enumerate(self.segments):
                v_id = seg.get("video_id", "")
                st = seg.get("start", 0.0)
                et = seg.get("end", 0.0)
                seg_id = f"{v_id}_{st:.2f}_{et:.2f}_{idx}"
                seg["segment_id"] = seg_id
                text = seg.get("text", "")
                corpus.append(text)

                # Gom nhóm theo video
                if v_id not in self.video_to_segments:
                    self.video_to_segments[v_id] = []
                self.video_to_segments[v_id].append({
                    "segment_id": seg_id,
                    "start": st,
                    "end": et,
                    "text": text,
                    "matched_keyframes": seg.get("matched_keyframes", [])
                })

                kfs = seg.get("matched_keyframes", [])
                for kf in kfs:
                    if kf not in self.kf_to_speech:
                        self.kf_to_speech[kf] = []
                        self.kf_to_seg_id[kf] = seg_id
                    self.kf_to_speech[kf].append({
                        "segment_id": seg_id,
                        "text": text,
                        "start": st,
                        "end": et
                    })

            # Sắp xếp các phân đoạn trong từng video theo mốc thời gian tăng dần
            for v_id in self.video_to_segments:
                self.video_to_segments[v_id].sort(key=lambda s: s.get("start", 0.0))

            # Xây dựng ma trận TF-IDF N-gram (1-gram và 2-gram)
            print("[ASREngine] Đang xây dựng ma trận TF-IDF N-gram cho toàn bộ câu thoại...")
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, norm='l2')
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            print(f"[ASREngine] ✅ TF-IDF Index hoàn tất ({len(self.vectorizer.vocabulary_):,} từ vựng/n-gram).")

            # Kết nối LanceDB bảng asr_segments (Qwen3-VL Semantic Vector)
            self.asr_table = None
            try:
                import lancedb
                if os.path.exists(config.LANCEDB_PATH):
                    db = lancedb.connect(config.LANCEDB_PATH)
                    tbl_names = db.table_names() if hasattr(db, "table_names") else db.list_tables()
                    if "asr_segments" in tbl_names:
                        self.asr_table = db.open_table("asr_segments")
                        print(f"[ASREngine] ✅ LanceDB Table 'asr_segments' ({len(self.asr_table):,} items) nạp xong.")
            except Exception as e:
                print(f"[ASREngine] ⚠️ Không thể mở bảng asr_segments: {e}")

            self._loaded = True
            print(f"[ASREngine] ✅ Đã nạp {len(self.segments):,} đoạn thoại ASR ({len(self.kf_to_speech):,} keyframes có lời thoại).")
        return True

    def get_speech_for_keyframe(self, kf_path: str) -> Optional[Dict[str, Any]]:
        """
        Tra cứu nhanh mốc lời thoại của 1 keyframe cụ thể:
        - Lớp 1 (O(1)): Tra cứu trực tiếp trong bảng băm kf_to_speech.
        - Lớp 2 (Time Interval Fallback): Tính toán mốc giây = frame_idx / fps và tìm phân đoạn chứa mốc giây đó.
        """
        self.load()
        speech_list = self.kf_to_speech.get(kf_path)
        if speech_list and len(speech_list) > 0:
            return speech_list[0]

        # Lớp 2: Tra cứu động theo thời gian thực (Time Interval Fallback)
        parts = kf_path.split("/")
        if len(parts) == 2:
            v_id = parts[0]
            f_str = parts[1].replace(".jpg", "")
            try:
                f_idx = int(f_str)
                fps = float(self.fps_mapping.get(v_id, 25.0))
                t_sec = f_idx / fps

                v_segs = self.video_to_segments.get(v_id, [])
                for seg in v_segs:
                    if (seg["start"] - 1.0) <= t_sec <= (seg["end"] + 1.0):
                        res = {
                            "segment_id": seg["segment_id"],
                            "text": seg["text"],
                            "start": seg["start"],
                            "end": seg["end"]
                        }
                        # Cache lại vào kf_to_speech để lần sau tra cứu O(1)
                        if kf_path not in self.kf_to_speech:
                            self.kf_to_speech[kf_path] = []
                        self.kf_to_speech[kf_path].append(res)
                        self.kf_to_seg_id[kf_path] = seg["segment_id"]
                        return res
            except ValueError:
                pass

        return None

    def get_speech_for_keyframe_expanded(
        self,
        kf_path: str,
        window_n: int = 1
    ) -> Dict[str, Any]:
        """
        Tra cứu lời thoại của keyframe kèm ngữ cảnh mở rộng ±window_n cụm segment liền kề.
        Giải quyết triệt để hiện tượng trôi lệch (drift) thời gian giữa audio thoại và video frame.
        """
        self.load()
        empty_res = {
            "text": "",
            "start": 0.0,
            "end": 0.0,
            "prev_text": "",
            "next_text": "",
            "prev_texts": [],
            "next_texts": [],
            "all_context": "",
            "window_n": window_n
        }

        sp = self.get_speech_for_keyframe(kf_path)
        if not sp:
            return empty_res

        v_id = kf_path.split("/")[0] if "/" in kf_path else ""
        v_segs = self.video_to_segments.get(v_id, [])
        if not v_segs or window_n <= 0:
            cur_text = sp.get("text", "")
            return {
                "text": cur_text,
                "start": sp.get("start", 0.0),
                "end": sp.get("end", 0.0),
                "prev_text": "",
                "next_text": "",
                "prev_texts": [],
                "next_texts": [],
                "all_context": cur_text,
                "window_n": window_n
            }

        # Xác định vị trí segment hiện tại trong danh sách đã sắp xếp của video
        seg_id = sp.get("segment_id", "")
        cur_idx = -1
        for i, s in enumerate(v_segs):
            if s.get("segment_id") == seg_id or (abs(s.get("start", 0.0) - sp.get("start", 0.0)) < 0.05 and abs(s.get("end", 0.0) - sp.get("end", 0.0)) < 0.05):
                cur_idx = i
                break

        if cur_idx == -1:
            cur_text = sp.get("text", "")
            return {
                "text": cur_text,
                "start": sp.get("start", 0.0),
                "end": sp.get("end", 0.0),
                "prev_text": "",
                "next_text": "",
                "prev_texts": [],
                "next_texts": [],
                "all_context": cur_text,
                "window_n": window_n
            }

        prev_segs = v_segs[max(0, cur_idx - window_n) : cur_idx]
        next_segs = v_segs[cur_idx + 1 : min(len(v_segs), cur_idx + 1 + window_n)]

        prev_texts = [s.get("text", "").strip() for s in prev_segs if s.get("text", "").strip()]
        next_texts = [s.get("text", "").strip() for s in next_segs if s.get("text", "").strip()]

        prev_text = " ... ".join(prev_texts)
        next_text = " ... ".join(next_texts)

        cur_text = sp.get("text", "").strip()
        all_parts = []
        if prev_texts:
            all_parts.extend(prev_texts)
        if cur_text:
            all_parts.append(cur_text)
        if next_texts:
            all_parts.extend(next_texts)

        all_context = " ... ".join(all_parts)

        return {
            "text": cur_text,
            "start": sp.get("start", 0.0),
            "end": sp.get("end", 0.0),
            "prev_text": prev_text,
            "next_text": next_text,
            "prev_texts": prev_texts,
            "next_texts": next_texts,
            "all_context": all_context,
            "window_n": window_n
        }

    def get_transcript_for_keyframe(self, kf_path: str) -> str:
        """Lấy chuỗi văn bản lời thoại của keyframe."""
        sp = self.get_speech_for_keyframe(kf_path)
        if sp:
            return sp.get("text", "")
        return ""

    def get_segment_id_for_keyframe(self, kf_path: str) -> Optional[str]:
        """Lấy mã phân đoạn ASR của keyframe để lọc trùng theo scene thoại."""
        self.load()
        seg_id = self.kf_to_seg_id.get(kf_path, None)
        if seg_id:
            return seg_id
        sp = self.get_speech_for_keyframe(kf_path)
        if sp:
            return sp.get("segment_id", None)
        return None

    def search_keyframes(
        self,
        query_speech: str,
        video_id: str = "",
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm lời thoại bằng TF-IDF N-gram và ánh xạ trực tiếp sang danh sách Keyframe có điểm số score_asr chuẩn hóa [0.0, 1.0].
        """
        if not self.load() or not query_speech or not query_speech.strip():
            return []

        q = query_speech.strip().lower()
        if not q:
            return []

        # Vectorize query qua TF-IDF
        q_vec = self.vectorizer.transform([q])
        sims = (self.tfidf_matrix * q_vec.T).toarray().flatten()

        # Lọc theo video_id nếu người dùng có nhập video filter
        clean_vid = video_id.strip()
        if clean_vid:
            for idx, seg in enumerate(self.segments):
                if clean_vid not in seg.get("video_id", ""):
                    sims[idx] = 0.0

        # Lấy các segment có similarity > 0
        matching_indices = np.where(sims > 0.0)[0]
        if len(matching_indices) == 0:
            return []

        max_sim = float(np.max(sims[matching_indices]))
        if max_sim <= 0:
            return []

        # Lấy top các segment phù hợp nhất
        top_seg_indices = matching_indices[np.argsort(-sims[matching_indices])[:max(top_k * 5, 200)]]

        kf_score_map: Dict[str, Dict[str, Any]] = {}

        for idx in top_seg_indices:
            raw_sim = float(sims[idx])
            seg = self.segments[idx]
            v_id = seg.get("video_id", "")
            text = seg.get("text", "")
            text_lower = text.lower()

            # Chuẩn hóa điểm số ASR về [0.0, 1.0]
            norm_score = raw_sim / max_sim
            if q in text_lower:
                norm_score = max(norm_score, 1.0)

            score = round(min(1.0, norm_score), 4)
            if score < 0.10:
                continue

            st = seg.get("start", 0.0)
            et = seg.get("end", 0.0)
            seg_id = seg.get("segment_id", f"{v_id}_{st:.2f}_{et:.2f}_{idx}")
            matched_kfs = seg.get("matched_keyframes", [])

            for kf in matched_kfs:
                if kf not in kf_score_map or score > kf_score_map[kf]["score_asr"]:
                    parts = kf.split("/")
                    vid = parts[0]
                    fid = os.path.splitext(parts[1])[0] if len(parts) > 1 else parts[0]

                    kf_score_map[kf] = {
                        "video_id": vid,
                        "frame_id": fid,
                        "path": kf,
                        "segment_id": seg_id,
                        "score_asr": score,
                        "speech_text": text,
                        "speech_start": st,
                        "speech_end": et
                    }

        results = list(kf_score_map.values())
        results.sort(key=lambda x: x["score_asr"], reverse=True)
        return results[:top_k]

    def search_semantic(
        self,
        query_vector: np.ndarray,
        video_id: str = "",
        include_videos: Optional[List[str]] = None,
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm lời thoại ASR bằng Qwen3-VL Semantic Embedding Vector trên bảng LanceDB 'asr_segments'.
        Độ phức tạp O(1) ANN search (~1ms), ánh xạ sang Keyframe với score_asr [0.0, 1.0].
        """
        self.load()
        if self.asr_table is None or query_vector is None or len(query_vector) == 0:
            return []

        search_k = min(len(self.segments), max(top_k * 5, 200))
        q_builder = self.asr_table.search(query_vector).metric("cosine").nprobes(16).refine_factor(5)

        where_clauses = []
        if video_id and video_id.strip():
            v_clean = video_id.strip()
            if len(v_clean) < 8 or "%" in v_clean:
                where_clauses.append(f"video_id LIKE '{v_clean.replace('%', '')}%'")
            else:
                where_clauses.append(f"video_id = '{v_clean}'")

        if include_videos and len(include_videos) > 0:
            inc_clean = [f"'{v.strip()}'" for v in include_videos if v.strip()]
            if inc_clean:
                where_clauses.append(f"video_id IN ({', '.join(inc_clean)})")

        if where_clauses:
            sql_where = " AND ".join([f"({c})" for c in where_clauses])
            q_builder = q_builder.where(sql_where, prefilter=True)

        try:
            hits = q_builder.limit(search_k).to_list()
        except Exception as e:
            print(f"[ASREngine] Lỗi truy vấn LanceDB asr_segments: {e}")
            return []

        if not hits:
            return []

        kf_score_map: Dict[str, Dict[str, Any]] = {}
        for hit in hits:
            dist = float(hit.get("_distance", 0.0))
            sim = max(0.0, 1.0 - dist)
            score = round(min(1.0, sim), 4)

            # Bỏ qua các kết quả có độ tương đồng quá thấp
            if score < 0.20:
                continue

            seg_id = hit.get("segment_id", "")
            vid = hit.get("video_id", "")
            text = hit.get("text", "")
            st = float(hit.get("asr_start", 0.0))
            et = float(hit.get("asr_end", 0.0))

            kfs_raw = hit.get("matched_keyframes_json", "[]")
            try:
                matched_kfs = json.loads(kfs_raw) if isinstance(kfs_raw, str) else kfs_raw
            except Exception:
                matched_kfs = []

            for kf in matched_kfs:
                if kf not in kf_score_map or score > kf_score_map[kf]["score_asr"]:
                    parts = kf.split("/")
                    f_id = os.path.splitext(parts[1])[0] if len(parts) > 1 else parts[0]
                    kf_score_map[kf] = {
                        "video_id": vid,
                        "frame_id": f_id,
                        "path": kf,
                        "segment_id": seg_id,
                        "score_asr": score,
                        "speech_text": text,
                        "speech_start": st,
                        "speech_end": et
                    }

        results = list(kf_score_map.values())
        results.sort(key=lambda x: x["score_asr"], reverse=True)
        return results[:top_k]


# Singleton instance
_asr_instance: Optional[ASREngine] = None


def get_asr_engine() -> ASREngine:
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = ASREngine()
    return _asr_instance

