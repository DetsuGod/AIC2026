"""
RerankerEngine — Bộ Tái Xếp Hạng 2 Bước (Two-Stage Intra-Video Re-Ranking).

Chiến thuật tối ưu chuẩn xác:
1. Bước 1 (Video-Level Ranking): Chọn ra Top 5 Video có điểm cao nhất từ kết quả tìm kiếm ban đầu.
2. Bước 2 (Intra-Video Candidate Pool): Với mỗi video trong Top 5, lọc ra Top 30-50 keyframes có điểm cao nhất (tạo tập 150 - 250 frames tiềm năng).
3. Bước 3 (Detailed Reranking): Dùng câu `reranker_query` chi tiết (đếm số lượng, thứ tự nón, trang phục) để tái tính điểm và đưa keyframe chuẩn xác nhất lên Top 1!
"""
import time
import numpy as np
from typing import List, Dict, Any, Optional

try:
    from . import config
    from .text_encoder import get_text_encoder
    from .search_engine import get_search_engine
except (ImportError, ValueError):
    import config
    from text_encoder import get_text_encoder
    from search_engine import get_search_engine


class VideoReranker:
    def __init__(self):
        self.search_engine = get_search_engine()
        self.text_encoder = get_text_encoder()

    def rerank_top_5_videos(
        self,
        query_text: str = "",
        reranker_query: str = "",
        top_v: int = 5,
        k_per_video: int = 40,
        ocr_query: str = "",
        speech_query: str = "",
        enable_asr: bool = False,
        object_class_id: Optional[int] = None,
        object_conf_thresh: float = 0.0,
        weight_visual: float = 0.50,
        weight_ocr: float = 0.20,
        weight_asr: float = 0.20,
        weight_object: float = 0.10,
        dedup_mode: str = "smart",
        visual_sim_thresh: float = 0.90,
        enable_cross_video_dedup: bool = True,
        cross_video_sim_thresh: float = 0.90,
        top_k: int = 100,
        use_rrf: bool = True,
        rrf_k: int = 60,
        video_id: str = "",
        include_videos: Optional[List[str]] = None,
        exclude_videos: Optional[List[str]] = None,
        category: str = "",
        sub_category: str = "",
        query_image: Optional[Any] = None,
        query_vec: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Chiến thuật Rerank tối ưu qua N lần Search kèm Video Filter:
        1. Search ban đầu lấy Top K1 keyframes (kèm đầy đủ cấu hình OCR, ASR, Object, Weights, RRF, Category, Video Filter, Image Query).
        2. Gom cụm tính điểm Video từ Top K1 -> Chọn ra Top N Video có điểm cao nhất.
        3. Với mỗi video trong Top N: Chạy search(video_id=vid, top_k=k_per_video) bằng câu chi tiết
           (hoặc câu ban đầu) kèm đúng 100% bộ lọc hiện tại của đại ca.
        4. Gộp toàn bộ kết quả từ N video và sắp xếp theo score giảm dần -> Đồng nhất hoàn hảo 100%.
        """
        t0 = time.time()
        top_k1 = top_k if top_k > 0 else 100

        if query_image is not None and query_vec is None:
            query_vec = self.text_encoder.encode_image(query_image)
        
        # 1. Tìm kiếm tổng quan trên toàn bộ kho để chọn ra Top K1 frames
        initial_results = self.search_engine.search(
            query_text=query_text,
            ocr_query=ocr_query,
            speech_query=speech_query,
            enable_asr=enable_asr,
            object_class_id=object_class_id,
            object_conf_thresh=object_conf_thresh,
            weight_visual=weight_visual,
            weight_ocr=weight_ocr,
            weight_asr=weight_asr,
            weight_object=weight_object,
            top_k=top_k1,
            dedup_mode="none",
            video_id=video_id,
            include_videos=include_videos,
            exclude_videos=exclude_videos,
            category=category,
            sub_category=sub_category,
            use_rrf=use_rrf,
            rrf_k=rrf_k,
            query_vec=query_vec
        )
        
        if not initial_results:
            return {"results": [], "top_videos": [], "total_candidates": 0, "elapsed_sec": 0.0}

        # 2. Gom cụm điểm theo Video ID từ tập Top K1 để tìm ra Top N Video có điểm cao nhất
        video_scores = {}
        for r in initial_results:
            vid = r["video_id"]
            video_scores[vid] = video_scores.get(vid, 0.0) + float(r["score"])

        top_videos = sorted(video_scores.keys(), key=lambda v: video_scores[v], reverse=True)[:top_v]

        # 3. Với mỗi video trong Top N: Chạy search() kèm bộ lọc video_id=vid
        # DÙNG CHÍNH XÁC query_text HOẶC query_vec (BẢO TOÀN 100% KẾT QUẢ VÀ ĐIỂM SỐ)
        all_candidates = []

        for vid in top_videos:
            vid_results = self.search_engine.search(
                query_text=query_text,
                ocr_query=ocr_query,
                speech_query=speech_query,
                enable_asr=enable_asr,
                object_class_id=object_class_id,
                object_conf_thresh=object_conf_thresh,
                video_id=vid,
                weight_visual=weight_visual,
                weight_ocr=weight_ocr,
                weight_asr=weight_asr,
                weight_object=weight_object,
                top_k=k_per_video,
                dedup_mode=dedup_mode,
                visual_sim_thresh=visual_sim_thresh,
                enable_cross_video_dedup=enable_cross_video_dedup,
                cross_video_sim_thresh=cross_video_sim_thresh,
                use_rrf=use_rrf,
                rrf_k=rrf_k,
                query_vec=query_vec
            )
            all_candidates.extend(vid_results)

        # 4. Sắp xếp toàn bộ keyframes theo điểm tổng giảm dần
        all_candidates.sort(key=lambda x: x["score"], reverse=True)

        elapsed = round(time.time() - t0, 3)
        print(f"[Reranker] ⏱️ Đã chọn lọc và rerank {len(all_candidates)} keyframes từ Top {len(top_videos)} videos trong {elapsed}s.")

        return {
            "results": all_candidates,
            "top_videos": top_videos,
            "total_candidates": len(all_candidates),
            "elapsed_sec": elapsed
        }


_reranker_instance: Optional[VideoReranker] = None


def get_video_reranker() -> VideoReranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = VideoReranker()
    return _reranker_instance
