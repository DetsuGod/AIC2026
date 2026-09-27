"""
QwenVLReranker — Bộ Tái Xếp Hạng Đa Phương Thức Dựa Trên Mô Hình Thị Giác - Ngôn Ngữ Qwen-VL (GPU CUDA).

Hỗ trợ 2 chế độ:
1. Pointwise (KIS Mode): Chấm điểm từng keyframe độc lập với ngữ cảnh ASR mở rộng (±1 phân đoạn).
2. Listwise Sequence (Multi-Scene & TRAKE Mode): Chấm điểm toàn bộ chuỗi đa ảnh [Image 1, Image 2, ...]
   giúp mô hình hiểu chính xác thứ tự thời gian và tính liên tục của hành động/sự kiện.

Thiết kế:
- Lazy Loading: Chỉ nạp model lên VRAM GPU (RTX 3070 8GB) khi người dùng chủ động bấm nút kích hoạt.
- Non-blocking & Streaming friendly.
- Graceful Fallback: Không làm sập ứng dụng nếu GPU bận hoặc gặp lỗi bất thường.
"""

import os
import time
import math
import torch
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

try:
    import config
except ImportError:
    from . import config


class QwenVLReranker:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or getattr(config, "QWEN_RERANKER_MODEL", "Qwen/Qwen3-VL-Reranker-2B")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._loaded = False
        self._loading = False

    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    def get_status(self) -> Dict[str, Any]:
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
        vram_total = 0.0
        vram_used = 0.0
        if cuda_avail:
            vram_total = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1)
            vram_used = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)

        return {
            "loaded": self._loaded,
            "loading": self._loading,
            "device": self.device,
            "cuda_available": cuda_avail,
            "gpu_name": gpu_name,
            "vram_total_mb": vram_total,
            "vram_used_mb": vram_used,
            "model_name": self.model_name
        }

    def load(self) -> bool:
        """Nạp model lên VRAM GPU (chỉ chạy 1 lần khi có lệnh)."""
        if self._loaded:
            return True

        if self._loading:
            while self._loading:
                time.sleep(0.5)
            return self._loaded

        self._loading = True
        t0 = time.time()
        print(f"[QwenReranker] 🚀 Đang khởi động model {self.model_name} trên {self.device.upper()}...")

        try:
            from transformers import AutoProcessor
            
            # Nạp Processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Cấu hình precision phù hợp với RTX 3070 (8GB VRAM)
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            # Nạp mô hình Qwen3-VL Reranker chuẩn
            loaded_model = None
            try:
                from transformers import Qwen3VLForConditionalGeneration
                loaded_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
            except Exception as e1:
                try:
                    from transformers import AutoModelForImageTextToText
                    loaded_model = AutoModelForImageTextToText.from_pretrained(
                        self.model_name,
                        torch_dtype=dtype,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True
                    )
                except Exception as e2:
                    from transformers import AutoModelForCausalLM
                    loaded_model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=dtype,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True
                    )

            self.model = loaded_model
            if self.device == "cpu" and self.model is not None:
                self.model.to("cpu")

            self.model.eval()

            # Cache token IDs cho yes/no
            self._yes_id = self.processor.tokenizer.encode("yes", add_special_tokens=False)[0]
            self._no_id = self.processor.tokenizer.encode("no", add_special_tokens=False)[0]

            self._loaded = True
            print(f"[QwenReranker] ✅ Nạp thành công model {self.model_name} trên {self.device.upper()} trong {time.time() - t0:.2f}s!")
            return True

        except Exception as e:
            print(f"[QwenReranker] ⚠️ Lỗi khi nạp model {self.model_name}: {e}")
            self._loaded = False
            return False
        finally:
            self._loading = False

    def _resolve_image_path(self, kf_rel_or_abs: str) -> Optional[str]:
        """Chuyển đổi keyframe mapping path thành đường dẫn file ảnh tuyệt đối."""
        if os.path.isabs(kf_rel_or_abs) and os.path.exists(kf_rel_or_abs):
            return kf_rel_or_abs

        # Tìm trong config.KF_DIR
        candidate = os.path.join(config.KF_DIR, kf_rel_or_abs)
        if os.path.exists(candidate):
            return candidate

        # Thử thêm đuôi .jpg nếu thiếu
        if not candidate.endswith(".jpg"):
            if os.path.exists(candidate + ".jpg"):
                return candidate + ".jpg"

        return None

    def score_single(
        self,
        query: str,
        keyframe_path: str,
        asr_transcript: str = "",
        asr_prev: str = "",
        asr_next: str = ""
    ) -> float:
        """
        Pointwise Reranking cho KIS Mode:
        Đánh giá mức độ khớp giữa câu truy vấn và 1 ảnh keyframe (kèm ASR mở rộng).
        Sử dụng cấu trúc template reranker chính hãng Alibaba Qwen3-VL.
        Tính xác suất P("yes") qua phân phối softmax của logits [no, yes].
        """
        if not self.is_loaded():
            if not self.load():
                return 0.5  # Fallback nếu không nạp được GPU

        abs_img_path = self._resolve_image_path(keyframe_path)
        if not abs_img_path or not os.path.exists(abs_img_path):
            return 0.0

        try:
            image = Image.open(abs_img_path).convert("RGB")
            
            # Ghép ngữ cảnh ASR mở rộng nếu có
            asr_context_parts = []
            if asr_prev:
                asr_context_parts.append(f"Trước đó: {asr_prev}")
            if asr_transcript:
                asr_context_parts.append(f"Tại cảnh: {asr_transcript}")
            if asr_next:
                asr_context_parts.append(f"Sau đó: {asr_next}")
            
            asr_block = " | ".join(asr_context_parts)
            query_with_asr = f"{query} (Lời thoại ASR: {asr_block})" if asr_block else query

            # Format template reranker chính hãng
            messages = [
                {"role": "system", "content": "Given a search query, retrieve relevant candidates that answer the query."},
                {"role": "query", "content": [{"type": "text", "text": query_with_asr}]},
                {"role": "document", "content": [{"type": "image", "image": image}]}
            ]

            try:
                text = self.processor.apply_chat_template(
                    messages,
                    chat_template="reranker",
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]

            l_yes = logits[0, self._yes_id].item()
            l_no = logits[0, self._no_id].item()

            # Softmax giữa No và Yes
            pair_logits = torch.tensor([l_no, l_yes], dtype=torch.float32)
            prob = torch.softmax(pair_logits, dim=-1)[1].item()
            return round(float(prob), 4)

        except Exception as e:
            print(f"[QwenReranker] Lỗi khi chấm điểm frame {keyframe_path}: {e}")
            return 0.50

    def score_sequence(
        self,
        query: str,
        scenes_data: List[Dict[str, Any]],
        discriminating_features: Optional[List[str]] = None,
        rerank_hint: str = "",
        mode: str = "multi_scene"
    ) -> float:
        """
        Listwise Sequence Reranking cho Multi-Scene và TRAKE:
        Truyền đồng thời danh sách N ảnh [Ảnh 1 (Cảnh 1), Ảnh 2 (Cảnh 2), ...] theo đúng thứ tự thời gian.
        Qwen-VL sẽ đánh giá cả tính tương thích hình ảnh, ngữ cảnh lời thoại (ASR) và thứ tự tuần tự của chuỗi sự kiện.
        """
        if not self.is_loaded():
            if not self.load():
                return 0.5

        if not scenes_data:
            return 0.0

        try:
            doc_content = []
            scene_descriptions = []

            for s_idx, sc in enumerate(scenes_data):
                kf_path = sc.get("keyframe") or f"{sc.get('video_id')}/{sc.get('frame_id')}.jpg"
                abs_path = self._resolve_image_path(kf_path)
                if abs_path and os.path.exists(abs_path):
                    img = Image.open(abs_path).convert("RGB")
                    doc_content.append({"type": "image", "image": img})
                    
                    sc_label = sc.get("scene_name") or f"Event E{s_idx + 1}"
                    ts = sc.get("timestamp_sec")
                    ts_str = f" [t={ts:.1f}s]" if ts is not None else ""
                    
                    sc_vis = sc.get("visual_query", "").strip()
                    sc_asr = sc.get("asr_transcript") or sc.get("asr_all") or ""
                    sc_ocr = sc.get("ocr_query", "").strip()

                    desc_parts = [f"{sc_label}{ts_str}"]
                    if sc_vis:
                        desc_parts.append(sc_vis)
                    if sc_asr:
                        desc_parts.append(f"Thoại: {sc_asr}")
                    if sc_ocr:
                        desc_parts.append(f"Chữ: {sc_ocr}")

                    scene_descriptions.append(" - ".join(desc_parts))

            if not doc_content:
                return 0.0

            seq_desc = " -> ".join(scene_descriptions)
            full_query_parts = [
                f"Chuỗi sự kiện theo thứ tự thời gian: {seq_desc}",
                f"Bối cảnh chung: {query}"
            ]

            if discriminating_features and len(discriminating_features) > 0:
                disc_str = ", ".join(discriminating_features)
                full_query_parts.append(f"Đặc trưng phân biệt cốt lõi cần đối chiếu: {disc_str}")

            if rerank_hint and rerank_hint.strip():
                full_query_parts.append(f"Gợi ý phân tích: {rerank_hint.strip()}")

            full_query = ". ".join(full_query_parts)

            messages = [
                {"role": "system", "content": "Given a temporal sequence search query, retrieve relevant sequential candidates that fulfill the chronological order and match the discriminating details."},
                {"role": "query", "content": [{"type": "text", "text": full_query}]},
                {"role": "document", "content": doc_content}
            ]

            try:
                text = self.processor.apply_chat_template(
                    messages,
                    chat_template="reranker",
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]

            l_yes = logits[0, self._yes_id].item()
            l_no = logits[0, self._no_id].item()

            pair_logits = torch.tensor([l_no, l_yes], dtype=torch.float32)
            prob = torch.softmax(pair_logits, dim=-1)[1].item()
            return round(float(prob), 4)

        except Exception as e:
            print(f"[QwenReranker] Lỗi khi chấm điểm sequence ({mode}): {e}")
            return 0.50

    def rerank_chains(
        self,
        chains: List[Dict[str, Any]],
        global_query: str,
        discriminating_features: Optional[List[str]] = None,
        rerank_hint: str = "",
        top_k_chains: int = 5,
        on_progress: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank danh sách các chuỗi sự kiện Storyboard (TRAKE Sequences):
        - Chỉ lấy Top K chuỗi ứng viên hàng đầu do TRAKE TARS DP đề xuất để tiết kiệm VRAM & thời gian.
        - Đánh giá đa phương thức toàn diện qua Qwen-VL.
        - Sắp xếp lại theo điểm qwen_score giảm dần và cập nhật lại thứ hạng.
        """
        if not chains:
            return []

        limit_k = min(len(chains), max(1, top_k_chains))
        target_chains = chains[:limit_k]
        remaining_chains = chains[limit_k:]

        print(f"[QwenReranker] 🚀 Bắt đầu thẩm định Top {limit_k}/{len(chains)} chuỗi Storyboard TRAKE...")

        for idx, chain in enumerate(target_chains):
            scenes_data = chain.get("scenes", [])
            vid = chain.get("video_id", "Unknown")
            
            score = self.score_sequence(
                query=global_query,
                scenes_data=scenes_data,
                discriminating_features=discriminating_features,
                rerank_hint=rerank_hint,
                mode="trake_chain"
            )

            chain["qwen_score"] = score
            chain["qwen_chain_score"] = score
            chain["original_rank"] = chain.get("rank", idx + 1)

            print(f"  - Chain #{idx + 1} [{vid}]: Qwen Score = {score:.4f} (Original TARS: {chain.get('joint_score', 0):.4f})")

            if on_progress and callable(on_progress):
                try:
                    on_progress(idx + 1, limit_k, vid, score)
                except Exception as cb_err:
                    print(f"Progress callback error: {cb_err}")

        # Sắp xếp các chuỗi đã chấm điểm theo Qwen Score giảm dần
        target_chains.sort(key=lambda x: x.get("qwen_score", 0.0), reverse=True)

        # Gán qwen_score = None cho các chuỗi chưa chấm
        for c in remaining_chains:
            if "qwen_score" not in c:
                c["qwen_score"] = None
            c["original_rank"] = c.get("rank", 0)

        combined_chains = target_chains + remaining_chains

        # Cập nhật lại rank và thứ tự submission
        for r_idx, c in enumerate(combined_chains):
            c["rank"] = r_idx + 1

        print(f"[QwenReranker] ✅ Hoàn tất tái xếp hạng {limit_k} chuỗi TRAKE!")
        return combined_chains


# Singleton Instance
_qwen_reranker_instance: Optional[QwenVLReranker] = None


def get_qwen_reranker() -> QwenVLReranker:
    global _qwen_reranker_instance
    if _qwen_reranker_instance is None:
        _qwen_reranker_instance = QwenVLReranker()
    return _qwen_reranker_instance
