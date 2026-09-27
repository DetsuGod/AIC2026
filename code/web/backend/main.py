import sys
import os
import json
import asyncio
import threading
import time
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union

# Thiết lập đường dẫn động chuẩn xác
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "code", "local_retrieval"))
sys.path.append(os.path.join(PROJECT_ROOT, "query"))

import submission_generator as sg
from search_engine import get_search_engine
from text_encoder import get_text_encoder
from reranker_engine import get_video_reranker
from qwen_reranker import get_qwen_reranker
import gemini_parser
import importlib
import config
import dres_client

@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    _start_warmup_if_needed()
    _start_dres_autologin_if_configured()
    yield


app = FastAPI(
    title="AIC 2026 Multimodal Search Engine",
    lifespan=app_lifespan,
)

# Trạng thái warm-up phục vụ UI. DRES auto-login chạy ở thread riêng nên không
# làm chậm hoặc làm sai readiness của retrieval.
_readiness_lock = threading.Lock()
_compute_semaphore = asyncio.Semaphore(1)
_warmup_thread: Optional[threading.Thread] = None
_dres_login_thread: Optional[threading.Thread] = None
_system_readiness: Dict[str, Any] = {
    "state": "starting",
    "stage": "boot",
    "message": "Đang khởi động backend...",
    "progress": 0,
    "started_at": None,
    "completed_at": None,
    "elapsed_sec": 0.0,
    "error": None,
}


def _update_readiness(**changes) -> None:
    """Cập nhật readiness atomically để frontend luôn nhận snapshot nhất quán."""
    with _readiness_lock:
        _system_readiness.update(changes)


def _readiness_snapshot() -> Dict[str, Any]:
    """Trả về bản sao trạng thái warm-up, không làm lộ mutable global state."""
    with _readiness_lock:
        snapshot = dict(_system_readiness)
    started_at = snapshot.get("started_at")
    if started_at and snapshot.get("state") == "warming":
        snapshot["elapsed_sec"] = round(time.time() - started_at, 2)
    return snapshot


def _warmup_system() -> None:
    """Nạp index, model query và metadata trước truy vấn thi đấu đầu tiên."""
    started_at = time.time()
    _update_readiness(
        state="warming",
        stage="indices",
        message="Đang nạp LanceDB, ASR và các mapping...",
        progress=10,
        started_at=started_at,
        completed_at=None,
        error=None,
    )
    try:
        engine = get_search_engine()
        _update_readiness(
            stage="qwen_embedding",
            message="Đang nạp và làm nóng Qwen3-VL Embedding...",
            progress=55,
        )
        # Một inference nhỏ làm nóng model/CUDA để người dùng không phải trả chi
        # phí khởi tạo trong truy vấn thi đấu đầu tiên.
        get_text_encoder().encode("video retrieval system warmup", enrich_cinematography=False)

        _update_readiness(
            stage="retrieval_probe",
            message="Đang làm nóng đường truy vấn LanceDB...",
            progress=92,
        )
        engine.search(
            query_text="general video scene",
            enable_asr=False,
            dedup_mode="smart",
            enable_cross_video_dedup=True,
            top_k=8,
            use_rrf=True,
            rrf_use_visual=True,
            rrf_use_asr=False,
            rrf_use_shot=True,
            return_shots=True,
        )

        completed_at = time.time()
        _update_readiness(
            state="ready",
            stage="complete",
            message="Hệ thống đã sẵn sàng truy vấn.",
            progress=100,
            completed_at=completed_at,
            elapsed_sec=round(completed_at - started_at, 2),
        )
        print(f"[Readiness] Hệ thống READY sau {completed_at - started_at:.2f}s.")
    except Exception as exc:
        completed_at = time.time()
        _update_readiness(
            state="error",
            stage="failed",
            message="Warm-up thất bại. Có thể thử lại từ giao diện.",
            completed_at=completed_at,
            elapsed_sec=round(completed_at - started_at, 2),
            error=str(exc),
        )
        print(f"[Readiness] Warm-up thất bại: {exc}")


def _start_warmup_if_needed() -> bool:
    """Khởi động tối đa một warm-up thread; trả False nếu đã/đang chạy."""
    global _warmup_thread
    with _readiness_lock:
        if _warmup_thread is not None and _warmup_thread.is_alive():
            return False
        if _system_readiness.get("state") == "ready":
            return False
        _warmup_thread = threading.Thread(
            target=_warmup_system,
            name="aic-readiness-warmup",
            daemon=True,
        )
        _warmup_thread.start()
    return True


def _auto_login_dres() -> None:
    """Đăng nhập DRES từ .env mà không chặn warm-up và không làm lộ credential."""
    client = dres_client.get_dres_client()
    if not client.username or not client.password:
        print("[DRES] Bỏ qua auto-login: .env chưa có đủ DRES_USERNAME/DRES_PASSWORD.")
        return
    ok, message = client.login()
    print(f"[DRES] Auto-login {'thành công' if ok else 'thất bại'}: {message}")


def _start_dres_autologin_if_configured() -> bool:
    global _dres_login_thread
    if not getattr(config, "DRES_USERNAME", "") or not getattr(config, "DRES_PASSWORD", ""):
        return False
    if _dres_login_thread is not None and _dres_login_thread.is_alive():
        return False
    _dres_login_thread = threading.Thread(
        target=_auto_login_dres,
        name="aic-dres-auto-login",
        daemon=True,
    )
    _dres_login_thread.start()
    return True


@app.get("/api/readiness")
async def readiness_endpoint():
    return {"status": "success", "data": _readiness_snapshot()}


@app.post("/api/readiness/retry")
async def retry_readiness_endpoint():
    started = _start_warmup_if_needed()
    return {
        "status": "success",
        "started": started,
        "data": _readiness_snapshot(),
    }

# Cấp quyền CORS cho Frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route phục vụ hình ảnh keyframe kèm Smart Fallback (tránh triệt để lỗi 404 do lệch bước nhảy frame)
import bisect
from fastapi.responses import Response, FileResponse

_video_frames_cache: Dict[str, List[int]] = {}

@app.get("/images/{video_id}/{frame_file}")
async def serve_keyframe_image(video_id: str, frame_file: str):
    """
    Phục vụ ảnh Keyframe với cơ chế Smart Fallback.
    Nếu frame ID yêu cầu (do tính toán timestamp lệch 25fps) không tồn tại,
    tự động tìm và phục vụ keyframe thực tế gần nhất trong video đó bằng binary search O(log N).
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id) or not re.fullmatch(r"\d+\.jpg", frame_file, re.I):
        return Response(status_code=404)
    video_dir = config.keyframe_directory(video_id)
    target_path = os.path.join(video_dir, frame_file)
    if os.path.exists(target_path):
        return FileResponse(target_path)
    
    if not os.path.exists(video_dir):
        return Response(status_code=404)
        
    try:
        frame_stem = os.path.splitext(frame_file)[0]
        target_frame_num = int(frame_stem)
    except ValueError:
        return Response(status_code=404)

    if video_id not in _video_frames_cache:
        try:
            frames = []
            for f in os.listdir(video_dir):
                if f.lower().endswith(".jpg"):
                    try:
                        frames.append(int(os.path.splitext(f)[0]))
                    except ValueError:
                        pass
            frames.sort()
            _video_frames_cache[video_id] = frames
        except Exception:
            _video_frames_cache[video_id] = []
            
    frames = _video_frames_cache[video_id]
    if not frames:
        return Response(status_code=404)
        
    idx = bisect.bisect_left(frames, target_frame_num)
    if idx == 0:
        best_num = frames[0]
    elif idx >= len(frames):
        best_num = frames[-1]
    else:
        before = frames[idx - 1]
        after = frames[idx]
        best_num = before if (target_frame_num - before) <= (after - target_frame_num) else after
        
    fallback_path = os.path.join(video_dir, f"{best_num:06d}.jpg")
    if os.path.exists(fallback_path):
        return FileResponse(fallback_path)
    return Response(status_code=404)

if os.path.exists(config.VIDEO_DIR):
    app.mount("/videos", StaticFiles(directory=config.VIDEO_DIR), name="videos")

@app.get("/api/video-file/{video_id}")
async def serve_video_file(video_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
        return Response(status_code=404)
    path = config.video_file(video_id)
    if not os.path.isfile(path):
        return Response(status_code=404)
    # MOV traffic dùng H.264 trong ISO BMFF; trình duyệt đọc Range như MP4.
    return FileResponse(path, media_type="video/mp4")

cut_videos_dir = getattr(config, "CUT_VIDEOS_BASE", "")
if os.path.exists(cut_videos_dir):
    app.mount("/cut_videos", StaticFiles(directory=cut_videos_dir), name="cut_videos")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

class SearchRequest(BaseModel):
    query: str = ""                   # Visual Semantic query
    image_base64: Optional[Union[str, List[str]]] = None # Ảnh kèm theo nếu có (Composed Query hoặc Multi-view List)
    images_base64: Optional[List[str]] = None            # Danh sách nhiều góc ảnh của cùng đối tượng
    image_weight: float = 0.5         # Tỷ trọng giữa ảnh và chữ khi kết hợp (0.0 -> 1.0)
    ocr_query: str = ""               # Ô nhập chữ OCR chuyên biệt
    speech_query: str = ""            # Lời thoại / ASR query
    enable_asr: bool = True           # Bật / tắt tính năng ASR
    enable_object: bool = False       # Bật / tắt tính năng Object Detection
    object_class_id: Optional[int] = None # Class ID COCO (0 -> 79)
    object_conf_thresh: float = 0.0   # Ngưỡng tin cậy lọc cứng (0.0 -> 0.9)
    dedup_mode: str = "smart"         # "smart" | "visual_only" | "asr_only" | "unique" | "none"
    visual_sim_thresh: float = 0.90   # Ngưỡng tương đồng vector trong cùng video (0.80 -> 0.98)
    enable_cross_video_dedup: bool = True  # Bật/tắt khử trùng thị giác liên video (chống QC/Intro)
    cross_video_sim_thresh: float = 0.90   # Ngưỡng tương đồng liên video
    video_id: str = ""                # Lọc 1 Video ID đơn lẻ
    include_videos: Optional[List[str]] = None  # Lọc tập hợp video được chọn (Tag Include)
    exclude_videos: Optional[List[str]] = None  # Lọc tập hợp video bị loại trừ (Tag Exclude)
    category: Union[str, List[str]] = ""        # Một hoặc nhiều nhóm danh mục
    sub_category: Union[str, List[str]] = ""    # Một hoặc nhiều môn học của day_hoc
    weight_visual: float = 1.0        # Trọng số kênh hình ảnh
    weight_ocr: float = 1.0           # Trọng số kênh chữ OCR
    weight_asr: float = 1.0           # Trọng số kênh âm thanh
    weight_object: float = 0.5        # Trọng số kênh vật thể RT-DETR
    top_k: int = 60
    use_rrf: bool = True              # Bật / tắt chế độ RRF
    rrf_k: int = 60                   # Tham số k của RRF
    asr_window_n: int = 1             # Số segment liền kề mở rộng ASR (±N)
    rrf_use_visual: bool = True       # Tín hiệu KF Visual tham gia RRF
    rrf_use_asr: bool = True          # Tín hiệu ASR tham gia RRF
    rrf_use_shot: bool = True         # Tín hiệu Video Shot tham gia RRF
    rrf_use_caption: bool = False     # Caption shot chọn lọc, coverage-aware
    return_shots: bool = True         # Trả về cả danh sách Shot song song

class ImageSearchRequest(BaseModel):
    image_base64: Optional[Union[str, List[str]]] = None # Ảnh base64 hoặc danh sách ảnh multi-view
    images_base64: Optional[List[str]] = None            # Danh sách nhiều góc ảnh
    query: str = ""                  # Văn bản bổ trợ kết hợp cùng ảnh (Composed Query)
    image_weight: float = 0.5        # Tỷ trọng giữa ảnh và chữ khi kết hợp
    ocr_query: str = ""              # Câu truy vấn OCR (chữ trong ảnh)
    speech_query: str = ""            # Lời thoại / ASR query
    enable_asr: bool = True           # Bật / tắt tính năng ASR
    enable_object: bool = False       # Bật / tắt tính năng Object Detection
    object_class_id: Optional[int] = None # Class ID COCO (0 -> 79)
    object_conf_thresh: float = 0.0   # Ngưỡng tin cậy lọc cứng (0.0 -> 0.9)
    dedup_mode: str = "smart"         # "smart" | "visual_only" | "asr_only" | "unique" | "none"
    visual_sim_thresh: float = 0.90   # Ngưỡng tương đồng vector trong cùng video (0.80 -> 0.98)
    enable_cross_video_dedup: bool = True  # Bật/tắt khử trùng thị giác liên video (chống QC/Intro)
    cross_video_sim_thresh: float = 0.90   # Ngưỡng tương đồng liên video
    video_id: str = ""                # Lọc 1 Video ID đơn lẻ
    include_videos: Optional[List[str]] = None  # Lọc tập hợp video được chọn (Tag Include)
    exclude_videos: Optional[List[str]] = None  # Lọc tập hợp video bị loại trừ (Tag Exclude)
    category: Union[str, List[str]] = ""
    sub_category: Union[str, List[str]] = ""
    weight_visual: float = 1.0        # Trọng số kênh hình ảnh
    weight_ocr: float = 1.0           # Trọng số kênh chữ OCR
    weight_asr: float = 1.0           # Trọng số kênh âm thanh
    weight_object: float = 0.5        # Trọng số kênh vật thể RT-DETR
    top_k: int = 60
    use_rrf: bool = True              # Bật / tắt chế độ RRF
    rrf_k: int = 60                   # Tham số k của RRF
    asr_window_n: int = 1             # Số segment liền kề mở rộng ASR (±N)
    rrf_use_visual: bool = True       # Tín hiệu KF Visual tham gia RRF
    rrf_use_asr: bool = True          # Tín hiệu ASR tham gia RRF
    rrf_use_shot: bool = True         # Tín hiệu Video Shot tham gia RRF
    rrf_use_caption: bool = False
    return_shots: bool = True         # Trả về cả danh sách Shot song song

class RerankRequest(BaseModel):
    candidate_video_ids: List[str]
    reranker_query: str
    object_class_id: Optional[int] = None
    top_k: int = 60

class RerankTopVideosRequest(BaseModel):
    query: str = ""
    image_base64: Optional[Union[str, List[str]]] = None
    images_base64: Optional[List[str]] = None
    image_weight: float = 0.5
    reranker_query: str = ""
    top_v: int = 5
    k_per_video: int = 40
    ocr_query: str = ""
    speech_query: str = ""
    enable_asr: bool = False
    enable_object: bool = True
    object_class_id: Optional[int] = None
    object_conf_thresh: float = 0.0
    weight_visual: float = 0.50
    weight_ocr: float = 0.20
    weight_asr: float = 0.20
    weight_object: float = 0.10
    dedup_mode: str = "smart"
    visual_sim_thresh: float = 0.90
    enable_cross_video_dedup: bool = True
    cross_video_sim_thresh: float = 0.90
    video_id: str = ""
    include_videos: Optional[List[str]] = None
    exclude_videos: Optional[List[str]] = None
    category: Union[str, List[str]] = ""
    sub_category: Union[str, List[str]] = ""
    top_k: int = 100
    use_rrf: bool = True
    rrf_k: int = 60

class SceneItem(BaseModel):
    scene_idx: int = 1
    name: str = ""
    visual_query: str = ""
    ocr_query: str = ""
    speech_query: str = ""
    object_filter: Optional[Dict[str, Any]] = None

class MultiSceneSearchRequest(BaseModel):
    scenes: List[SceneItem]
    enable_visual: bool = True
    enable_ocr: bool = True
    enable_asr: bool = True
    enable_object: bool = True
    object_class_id: Optional[int] = None
    object_conf_thresh: float = 0.0
    weight_visual: float = 0.50
    weight_ocr: float = 0.20
    weight_asr: float = 0.20
    weight_object: float = 0.10
    dedup_mode: str = "smart"
    visual_sim_thresh: float = 0.90
    enable_cross_video_dedup: bool = True
    cross_video_sim_thresh: float = 0.90
    video_id: str = ""
    include_videos: Optional[List[str]] = None
    exclude_videos: Optional[List[str]] = None
    category: Union[str, List[str]] = ""
    sub_category: Union[str, List[str]] = ""
    top_k: int = 100
    top_v: int = 5
    k_per_video: int = 40
    use_rrf: bool = True
    rrf_k: int = 60
    trake_mode: bool = False
    lambda_penalty: float = 0.001
    alpha_fusion: float = 0.35
    k_paths: int = 3
    trake_top_k: int = 10
    asr_window_n: int = 1
    rrf_use_visual: bool = True
    rrf_use_asr: bool = True
    rrf_use_shot: bool = True
    rrf_use_caption: bool = False
    return_shots: bool = True
    compact_response: bool = False

class QwenRerankFrameRequest(BaseModel):
    query: str
    keyframe_path: str
    asr_transcript: str = ""
    asr_prev: str = ""
    asr_next: str = ""

class QwenRerankChainsRequest(BaseModel):
    chains: List[Dict[str, Any]]
    global_query: str = ""
    discriminating_features: List[str] = []
    rerank_hint: str = ""
    top_k_chains: int = 5
    asr_window_n: int = 1

class ParseRequest(BaseModel):
    raw_text: str
    api_key: Optional[str] = None

@app.get("/api/coco-classes")
async def get_coco_classes():
    return {"status": "success", "classes": [], "disabled": True}

@app.get("/api/keyframe-boxes/{video_id}/{frame_id}")
async def get_keyframe_boxes_endpoint(video_id: str, frame_id: str):
    return {"status": "success", "video_id": video_id, "frame_id": frame_id, "boxes": [], "disabled": True}

@app.get("/api/categories")
async def get_categories_endpoint():
    """Trả taxonomy đúng với video_categories.json và số video từng nhánh."""
    cat_path = os.path.join(config.DATA_DIR, "video_categories.json")
    categories_data = {}
    if os.path.exists(cat_path):
        with open(cat_path, "r", encoding="utf-8") as f:
            categories_data = json.load(f)

    category_counts: Dict[str, int] = {}
    subject_counts: Dict[str, int] = {}
    for item in categories_data.values():
        category = str(item.get("category", "")).strip()
        subject = str(item.get("sub_category", "")).strip()
        if category:
            category_counts[category] = category_counts.get(category, 0) + 1
        if category == "day_hoc" and subject:
            subject_counts[subject] = subject_counts.get(subject, 0) + 1

    taxonomy = {
        "am_thuc": {"label": "🍳 Ẩm Thực & Món Ngon", "sub": {}},
        "tin_tuc": {"label": "📰 Tin Tức & Phóng Sự", "sub": {}},
        "day_hoc": {
            "label": "📚 Dạy Học & Ôn Thi THPT",
            "sub": {
                "toan": "📐 Môn Toán",
                "ly": "⚡ Môn Vật Lý",
                "hoa": "🧪 Môn Hóa Học",
                "van": "📖 Môn Ngữ Văn",
                "su": "🏛️ Môn Lịch Sử",
                "dia": "🌍 Môn Địa Lý",
                "sinh": "🧬 Môn Sinh Học",
                "gdcd": "⚖️ Môn GDCD",
                "tieng_anh": "🇬🇧 Môn Tiếng Anh"
            }
        },
        "du_lich_van_hoa": {"label": "🌴 Du Lịch & Văn Hóa", "sub": {}},
        "mua_lan": {"label": "🦁 Múa Lân Sư Rồng", "sub": {}},
        "dua_xe_dap": {"label": "🚴 Đua Xe Đạp", "sub": {}},
        "camera_giao_thong": {"label": "🚦 Camera Giao Thông", "sub": {}}
    }

    for category, info in taxonomy.items():
        info["count"] = category_counts.get(category, 0)
        if category == "day_hoc":
            info["sub_counts"] = subject_counts

    return {
        "status": "success",
        "taxonomy": taxonomy,
        "total_videos": len(categories_data)
    }

@app.get("/api/video-timeline/{video_id}")
async def video_timeline_endpoint(video_id: str):
    """Lấy toàn bộ danh sách keyframe của 1 video theo thứ tự thời gian tăng dần."""
    engine = get_search_engine()
    frames = engine.get_keyframes_for_video(video_id)
    return {
        "status": "success",
        "video_id": video_id,
        "total_frames": len(frames),
        "frames": frames
    }


@app.get("/api/video-fps/{video_id}")
async def get_video_fps_endpoint(video_id: str):
    """Trả FPS thực của video để frontend đổi chính xác timestamp ↔ frame."""
    engine = get_search_engine()
    fps = engine.get_video_fps(video_id)
    return {"status": "success", "video_id": video_id, "fps": fps}


@app.get("/api/keyframe-resolve/{video_id}")
async def resolve_keyframe_endpoint(video_id: str, frame_id: Optional[str] = Query(default=None)):
    """Đối chiếu video/KF nhập tay với keyframe map và FPS thực của MP4."""
    engine = get_search_engine()
    return engine.resolve_keyframe(video_id=video_id, frame_id=frame_id)

def decode_images_from_request(req):
    """Giải mã 1 ảnh hoặc danh sách nhiều góc ảnh base64 từ request."""
    import base64
    import io
    from PIL import Image

    raw_list = []
    if hasattr(req, "images_base64") and req.images_base64:
        raw_list.extend([b for b in req.images_base64 if b and str(b).strip()])

    if hasattr(req, "image_base64") and req.image_base64:
        if isinstance(req.image_base64, list):
            raw_list.extend([b for b in req.image_base64 if b and str(b).strip()])
        elif isinstance(req.image_base64, str) and req.image_base64.strip():
            raw_list.append(req.image_base64.strip())

    if not raw_list:
        return None

    pil_images = []
    for b64 in raw_list:
        data = b64
        if "," in data:
            data = data.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(data)
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            pil_images.append(img)
        except Exception as e:
            print(f"[Warning] Không thể decode ảnh base64: {e}")

    if len(pil_images) == 0:
        return None
    if len(pil_images) == 1:
        return pil_images[0]
    return pil_images

@app.post("/api/parse-query")
async def parse_query_endpoint(request: ParseRequest):
    """Gọi Gemini AI Parser bóc tách đề bài thô thành các trường chuyên biệt."""
    print(f"\n=======================================================")
    print(f"[API /api/parse-query] 🚀 Nhận yêu cầu bóc tách từ người dùng:")
    print(f"  - Raw Query: {request.raw_text[:100]}...")
    print(f"  - Key gửi lên: {'Đã có key từ UI' if request.api_key else 'Dùng Key mặc định'}")
    
    # Reload lại module parser để nhận các cập nhật mới nhất
    try:
        importlib.reload(gemini_parser)
    except Exception as e:
        print(f"Reload parser error: {e}")

    parsed = gemini_parser.parse_query_with_gemini(raw_text=request.raw_text, api_key=request.api_key)
    
    print(f"[API /api/parse-query] ✅ Trả về kết quả từ: {parsed.get('source', 'Unknown')}")
    print(f"  - Task Type: {parsed.get('task_type', 'KIS')}")
    print(f"  - Visual: {parsed.get('visual_query', '')[:100]}...")
    print(f"  - Speech: {parsed.get('speech_query', '')[:100]}...")
    print(f"  - OCR: {parsed.get('ocr_query', '')}")
    print(f"  - Object: {parsed.get('object_filter')}")
    if parsed.get('is_multi_scene'):
        print(f"  - Multi-scene / TRAKE: {len(parsed.get('scenes', []))} phân cảnh")
    print(f"=======================================================\n")
    return {"status": "success", "data": parsed, "parsed": parsed}

@app.post("/api/rerank-top-videos")
async def rerank_top_videos_endpoint(request: RerankTopVideosRequest):
    """Chiến thuật Rerank tối ưu qua N lần Search kèm Video Filter (Hỗ trợ cả Text và Image/Multi-view Query)."""
    reranker = get_video_reranker()
    obj_class = None  # Legacy request fields cannot re-enable RT-DETR.
    obj_weight = 0.0

    query_img = decode_images_from_request(request)

    result_data = reranker.rerank_top_5_videos(
        query_text=request.query,
        reranker_query=request.reranker_query,
        top_v=request.top_v,
        k_per_video=request.k_per_video,
        ocr_query=request.ocr_query,
        speech_query=request.speech_query,
        enable_asr=request.enable_asr,
        object_class_id=obj_class,
        object_conf_thresh=request.object_conf_thresh if request.enable_object else 0.0,
        weight_visual=request.weight_visual,
        weight_ocr=request.weight_ocr,
        weight_asr=request.weight_asr,
        weight_object=obj_weight,
        dedup_mode=request.dedup_mode,
        visual_sim_thresh=request.visual_sim_thresh,
        enable_cross_video_dedup=request.enable_cross_video_dedup,
        cross_video_sim_thresh=request.cross_video_sim_thresh,
        video_id=request.video_id,
        include_videos=request.include_videos,
        exclude_videos=request.exclude_videos,
        category=request.category,
        sub_category=request.sub_category,
        top_k=request.top_k,
        use_rrf=request.use_rrf,
        rrf_k=request.rrf_k,
        query_image=query_img
    )

    results = result_data["results"]
    for r in results:
        r["image_url"] = f"http://localhost:8000/images/{r['video_id']}/{r['frame_id']}.jpg"
        if r.get("cut_file"):
            r["video_url"] = f"http://localhost:8000/cut_videos/{r['cut_file']}"
        else:
            r["video_url"] = ""

    return {
        "status": "success",
        "results": results,
        "top_videos": result_data["top_videos"],
        "total_candidates": result_data["total_candidates"],
        "elapsed_sec": result_data["elapsed_sec"]
    }

@app.post("/api/search-image")
async def search_image_endpoint(request: ImageSearchRequest):
    """Tìm kiếm hình ảnh đa phương thức (Image/Multi-view to Image / Qwen3-VL 2048D + LanceDB) hỗ trợ đầy đủ OCR, ASR, Object, RRF."""
    import time
    t0 = time.time()

    try:
        engine = get_search_engine()
        img = decode_images_from_request(request)
        if img is None:
            raise ValueError("Không tìm thấy ảnh hợp lệ trong request!")

        obj_class = None
        obj_weight = 0.0

        async with _compute_semaphore:
            res_data = await run_in_threadpool(
                engine.search_by_image,
                query_image=img,
                query_text=request.query,
                image_weight=request.image_weight,
                ocr_query=request.ocr_query,
                speech_query=request.speech_query,
                enable_asr=request.enable_asr,
                object_class_id=obj_class,
                object_conf_thresh=request.object_conf_thresh if request.enable_object else 0.0,
                dedup_mode=request.dedup_mode,
                visual_sim_thresh=request.visual_sim_thresh,
                enable_cross_video_dedup=request.enable_cross_video_dedup,
                cross_video_sim_thresh=request.cross_video_sim_thresh,
                video_id=request.video_id,
                include_videos=request.include_videos,
                exclude_videos=request.exclude_videos,
                category=request.category,
                sub_category=request.sub_category,
                weight_visual=request.weight_visual,
                weight_ocr=request.weight_ocr,
                weight_asr=request.weight_asr,
                weight_object=obj_weight,
                top_k=request.top_k,
                use_rrf=request.use_rrf,
                rrf_k=request.rrf_k,
                asr_window_n=request.asr_window_n,
                rrf_use_visual=request.rrf_use_visual,
                rrf_use_asr=request.rrf_use_asr,
                rrf_use_shot=request.rrf_use_shot,
                rrf_use_caption=request.rrf_use_caption,
                return_shots=request.return_shots,
            )

        if isinstance(res_data, dict):
            results = res_data.get("results", [])
            shot_results = res_data.get("shot_results", [])
        else:
            results = res_data
            shot_results = []

        for r in results:
            r["image_url"] = f"http://localhost:8000/images/{r['video_id']}/{r['frame_id']}.jpg"
            if r.get("cut_file"):
                r["video_url"] = f"http://localhost:8000/cut_videos/{r['cut_file']}"
            else:
                r["video_url"] = ""

        for s in shot_results:
            s["video_url"] = f"http://localhost:8000/cut_videos/{s['cut_file']}"
            s["thumbnail_url"] = f"http://localhost:8000/images/{s['video_id']}/{s['best_kf_frame_id']}.jpg"

        top_videos = []
        for r in results:
            if r["video_id"] not in top_videos:
                top_videos.append(r["video_id"])

        elapsed = round(time.time() - t0, 3)
        return {
            "status": "success",
            "results": results,
            "shot_results": shot_results,
            "top_videos": top_videos,
            "total_candidates": len(results),
            "elapsed_sec": elapsed
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "results": [],
            "top_videos": [],
            "total_candidates": 0,
            "elapsed_sec": round(time.time() - t0, 3)
        }

@app.post("/api/search-multi-scene")
async def search_multi_scene_endpoint(request: MultiSceneSearchRequest):
    """Truy vấn Đa Phân Cảnh (Multi-Scene / TRAKE Temporal Query Decomposition & Video Voting)."""
    engine = get_search_engine()
    scenes_data = [s.model_dump() if hasattr(s, "model_dump") else s.dict() for s in request.scenes]
    
    async with _compute_semaphore:
        result_data = await run_in_threadpool(
        engine.search_multi_scene,
        scenes=scenes_data,
        enable_visual=request.enable_visual,
        enable_ocr=request.enable_ocr,
        enable_asr=request.enable_asr,
        enable_object=False,
        object_class_id=None,
        object_conf_thresh=0.0,
        weight_visual=request.weight_visual,
        weight_ocr=request.weight_ocr,
        weight_asr=request.weight_asr,
        weight_object=0.0,
        dedup_mode=request.dedup_mode,
        visual_sim_thresh=request.visual_sim_thresh,
        enable_cross_video_dedup=request.enable_cross_video_dedup,
        cross_video_sim_thresh=request.cross_video_sim_thresh,
        top_k=request.top_k,
        top_v=request.top_v,
        k_per_video=request.k_per_video,
        use_rrf=request.use_rrf,
        rrf_k=request.rrf_k,
        trake_mode=request.trake_mode,
        lambda_penalty=request.lambda_penalty,
        alpha_fusion=request.alpha_fusion,
        k_paths=request.k_paths,
        trake_top_k=request.trake_top_k,
        asr_window_n=request.asr_window_n,
        video_id=request.video_id,
        include_videos=request.include_videos,
        exclude_videos=request.exclude_videos,
        category=request.category,
        sub_category=request.sub_category,
        rrf_use_visual=request.rrf_use_visual,
        rrf_use_asr=request.rrf_use_asr,
        rrf_use_shot=request.rrf_use_shot,
        rrf_use_caption=request.rrf_use_caption,
        return_shots=request.return_shots,
        compact_response=request.compact_response
        )

    for sc in result_data.get("scenes_results", []):
        for r in sc.get("results", []):
            r["image_url"] = f"http://localhost:8000/images/{r['video_id']}/{r['frame_id']}.jpg"
            if r.get("cut_file"):
                r["video_url"] = f"http://localhost:8000/cut_videos/{r['cut_file']}"
            else:
                r["video_url"] = ""
        for s in sc.get("shot_results", []):
            s["video_url"] = f"http://localhost:8000/cut_videos/{s['cut_file']}"
            s["thumbnail_url"] = f"http://localhost:8000/images/{s['video_id']}/{s['best_kf_frame_id']}.jpg"

    for r in result_data.get("combined_results", []):
        r["image_url"] = f"http://localhost:8000/images/{r['video_id']}/{r['frame_id']}.jpg"
        if r.get("cut_file"):
            r["video_url"] = f"http://localhost:8000/cut_videos/{r['cut_file']}"
        else:
            r["video_url"] = ""

    for seq in result_data.get("storyboard_sequences", []):
        for sc_item in seq.get("scenes", []):
            sc_item["image_url"] = f"http://localhost:8000/images/{sc_item['video_id']}/{sc_item['frame_id']}.jpg"
            if sc_item.get("cut_file"):
                sc_item["video_url"] = f"http://localhost:8000/cut_videos/{sc_item['cut_file']}"

    return {
        "status": "success",
        "top_videos": result_data["top_videos"],
        "storyboard_sequences": result_data.get("storyboard_sequences", []),
        "trake_submission_lines": result_data.get("trake_submission_lines", []),
        "scenes_results": result_data["scenes_results"],
        "combined_results": result_data["combined_results"],
        "total_candidates": result_data["total_candidates"],
        "elapsed_sec": result_data["elapsed_sec"]
    }

@app.get("/api/qwen-status")
async def qwen_status_endpoint():
    """Kiểm tra trạng thái nạp và thông số VRAM của model Qwen-VL Reranker."""
    reranker = get_qwen_reranker()
    return {"status": "success", "data": reranker.get_status()}

@app.post("/api/qwen-load")
async def qwen_load_endpoint():
    """Chủ động nạp mô hình Qwen-VL Reranker lên VRAM GPU."""
    reranker = get_qwen_reranker()
    success = reranker.load()
    return {"status": "success" if success else "error", "data": reranker.get_status()}

@app.post("/api/qwen-rerank-frame")
async def qwen_rerank_frame_endpoint(request: QwenRerankFrameRequest):
    """Chấm điểm pointwise một keyframe đơn lẻ với Qwen-VL."""
    reranker = get_qwen_reranker()
    score = reranker.score_single(
        query=request.query,
        keyframe_path=request.keyframe_path,
        asr_transcript=request.asr_transcript,
        asr_prev=request.asr_prev,
        asr_next=request.asr_next
    )
    return {"status": "success", "score": score, "keyframe": request.keyframe_path}

@app.post("/api/qwen-rerank-chains")
async def qwen_rerank_chains_endpoint(request: QwenRerankChainsRequest):
    """Tái xếp hạng toàn bộ danh sách các chuỗi sự kiện Storyboard (TRAKE Sequences) bằng Qwen-VL."""
    reranker = get_qwen_reranker()
    reranked = reranker.rerank_chains(
        chains=request.chains,
        global_query=request.global_query,
        discriminating_features=request.discriminating_features,
        rerank_hint=request.rerank_hint,
        top_k_chains=request.top_k_chains
    )

    # Đảm bảo các image_url có đầy đủ cho giao diện hiển thị
    for seq in reranked:
        for sc_item in seq.get("scenes", []):
            if "image_url" not in sc_item:
                v_id = sc_item.get("video_id", "")
                f_id = sc_item.get("frame_id", "")
                if v_id and f_id:
                    sc_item["image_url"] = f"http://localhost:8000/images/{v_id}/{f_id}.jpg"

    return {
        "status": "success",
        "storyboard_sequences": reranked,
        "total_evaluated": min(len(request.chains), max(1, request.top_k_chains))
    }

@app.post("/api/rerank-videos")
async def rerank_videos_endpoint(request: RerankRequest):
    """Tái xếp hạng nội bộ các video ứng viên cụ thể."""
    reranker = get_video_reranker()
    results = reranker.rerank_top_videos(
        candidate_video_ids=request.candidate_video_ids,
        reranker_query=request.reranker_query,
        object_class_id=request.object_class_id,
        top_k=request.top_k
    )
    return {"status": "success", "results": results}

@app.post("/api/search")
async def search_endpoint(request: SearchRequest):
    engine = get_search_engine()
    
    obj_class = None
    obj_weight = 0.0

    query_img = decode_images_from_request(request)

    async with _compute_semaphore:
        res_data = await run_in_threadpool(
        engine.search,
        query_text=request.query,
        ocr_query=request.ocr_query,
        speech_query=request.speech_query,
        enable_asr=request.enable_asr,
        object_class_id=obj_class,
        object_conf_thresh=request.object_conf_thresh if request.enable_object else 0.0,
        dedup_mode=request.dedup_mode,
        visual_sim_thresh=request.visual_sim_thresh,
        enable_cross_video_dedup=request.enable_cross_video_dedup,
        cross_video_sim_thresh=request.cross_video_sim_thresh,
        video_id=request.video_id,
        include_videos=request.include_videos,
        exclude_videos=request.exclude_videos,
        category=request.category,
        sub_category=request.sub_category,
        weight_visual=request.weight_visual,
        weight_ocr=request.weight_ocr,
        weight_asr=request.weight_asr,
        weight_object=obj_weight,
        top_k=request.top_k,
        use_rrf=request.use_rrf,
        rrf_k=request.rrf_k,
        asr_window_n=request.asr_window_n,
        query_image=query_img,
        image_weight=request.image_weight,
        rrf_use_visual=request.rrf_use_visual,
        rrf_use_asr=request.rrf_use_asr,
        rrf_use_shot=request.rrf_use_shot,
        rrf_use_caption=request.rrf_use_caption,
        return_shots=request.return_shots
        )

    if isinstance(res_data, dict):
        results = res_data.get("results", [])
        shot_results = res_data.get("shot_results", [])
    else:
        results = res_data
        shot_results = []

    # Gắn thêm URL hình ảnh và Video URL cho Web hiển thị
    for r in results:
        r["image_url"] = f"http://localhost:8000/images/{r['video_id']}/{r['frame_id']}.jpg"
        if r.get("cut_file"):
            r["video_url"] = f"http://localhost:8000/cut_videos/{r['cut_file']}"
        else:
            r["video_url"] = ""

    # Gắn thêm video URL và thumbnail URL cho Video Shots
    for s in shot_results:
        s["video_url"] = f"http://localhost:8000/cut_videos/{s['cut_file']}" if s.get("cut_file") else ""
        s["thumbnail_url"] = f"http://localhost:8000/images/{s['video_id']}/{s['best_kf_frame_id']}.jpg"

    return {
        "status": "success",
        "results": results,
        "shot_results": shot_results
    }

# ==============================================================================
# SUBMISSION GENERATOR & ZIP EXPORTER ENDPOINTS (AIC 2026 CONTEST SUITE)
# ==============================================================================

class SubmissionFillRequest(BaseModel):
    mode: str = "kis"                          # "kis" | "qa" | "trake"
    video_id: str = ""                         # ID video mục tiêu
    frame_a: Optional[int] = None              # Điểm bắt đầu A
    frame_b: Optional[int] = None              # Điểm kết thúc B
    second_interval: Optional[List[int]] = None # [A2, B2] (nếu có 2 khoảng)
    answer: Optional[str] = None               # Câu trả lời QA (nếu mode="qa")
    trake_event_frames: Optional[List[int]] = None # Danh sách frame cho các events TRAKE
    system_backup_items: Optional[List[Dict[str, Any]]] = None # Danh sách kết quả Search Engine bồi thêm (Risk Hedging)
    target_count: int = 100                    # Mặc định 100 dòng

class SubmissionSaveRequest(BaseModel):
    file_path: str
    lines: List[str]

class SubmissionListRequest(BaseModel):
    dir_path: str

class SubmissionProcessDirRequest(BaseModel):
    dir_path: str
    target_count: int = 100

class SubmissionZipRequest(BaseModel):
    submission_dir: str
    output_zip_path: str

@app.post("/api/submission/fill")
async def api_submission_fill(req: SubmissionFillRequest):
    """
    Sinh danh sách dòng nộp bài (100 dòng cho KIS/QA, 1 dòng chuỗi cho TRAKE) theo chiến thuật phân bổ điểm.
    """
    try:
        sec_int = tuple(req.second_interval) if (req.second_interval and len(req.second_interval) >= 2) else None
        
        lines = sg.generate_submission_lines(
            mode=req.mode,
            video_id=req.video_id,
            A=req.frame_a if req.frame_a is not None else 0,
            B=req.frame_b,
            answer=req.answer,
            second_interval=sec_int,
            system_backup_items=req.system_backup_items,
            trake_event_frames=req.trake_event_frames,
            target_count=req.target_count
        )

        return {
            "status": "success",
            "mode": req.mode,
            "total_lines": len(lines),
            "preview": lines[:5],
            "lines": lines
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/submission/save-file")
async def api_submission_save_file(req: SubmissionSaveRequest):
    """
    Lưu danh sách các dòng submission vào file CSV chỉ định.
    """
    try:
        out_path = os.path.abspath(req.file_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as fp:
            fp.write("\n".join(req.lines) + "\n")
        
        return {
            "status": "success",
            "file_path": out_path,
            "line_count": len(req.lines)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/submission/list-files")
async def api_submission_list_files(req: SubmissionListRequest):
    """
    Quét và liệt kê toàn bộ các file CSV trong thư mục submission kèm thông tin tổng quan.
    """
    try:
        dir_path = os.path.abspath(req.dir_path)
        if not os.path.exists(dir_path):
            return {"status": "success", "exists": False, "files": []}

        files_info = []
        for fname in sorted(os.listdir(dir_path)):
            if fname.lower().endswith(".csv"):
                fpath = os.path.join(dir_path, fname)
                line_count = 0
                first_line = ""
                with open(fpath, "r", encoding="utf-8-sig") as fp:
                    for idx, line in enumerate(fp):
                        if idx == 0:
                            first_line = line.strip()
                        if line.strip():
                            line_count += 1
                
                # Phân loại query type từ tên file hoặc nội dung
                qtype = "KIS"
                if "-qa" in fname.lower() or '"' in first_line:
                    qtype = "QA"
                elif "-trake" in fname.lower() or (first_line.count(",") >= 2 and "[" not in first_line):
                    qtype = "TRAKE"

                files_info.append({
                    "filename": fname,
                    "file_path": fpath,
                    "type": qtype,
                    "line_count": line_count,
                    "first_line": first_line
                })

        return {
            "status": "success",
            "exists": True,
            "dir_path": dir_path,
            "file_count": len(files_info),
            "files": files_info
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/submission/process-dir")
async def api_submission_process_dir(req: SubmissionProcessDirRequest):
    """
    Quét toàn bộ thư mục CSV thô và tự động fill toàn bộ thành 100 dòng hoàn chỉnh.
    """
    try:
        dir_path = os.path.abspath(req.dir_path)
        if not os.path.exists(dir_path):
            return {"status": "error", "message": f"Thư mục không tồn tại: {dir_path}"}

        processed = []
        errors = []
        for fname in sorted(os.listdir(dir_path)):
            if fname.lower().endswith(".csv"):
                fpath = os.path.join(dir_path, fname)
                try:
                    cnt = sg.process_submission_file(fpath, target_count=req.target_count)
                    processed.append({"filename": fname, "lines_written": cnt})
                except Exception as fe:
                    errors.append({"filename": fname, "error": str(fe)})

        return {
            "status": "success",
            "processed_count": len(processed),
            "processed": processed,
            "errors": errors
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/submission/export-zip")
async def api_submission_export_zip(req: SubmissionZipRequest):
    """
    Đóng gói thư mục submission thành file .zip chuẩn quy chế BTC.
    """
    try:
        sub_dir = os.path.abspath(req.submission_dir)
        zip_path = os.path.abspath(req.output_zip_path)
        
        out_zip, count = sg.export_submission_zip(sub_dir, zip_path)
        file_size_kb = round(os.path.getsize(out_zip) / 1024, 2)

        return {
            "status": "success",
            "zip_path": out_zip,
            "file_count": count,
            "file_size_kb": file_size_kb
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==============================================================================
# DRES LIVE CONTEST API ENDPOINTS (VÒNG CHUNG KẾT AIC 2026)
# ==============================================================================
class DresLoginRequest(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    server_url: Optional[str] = None

class DresSubmitKisRequest(BaseModel):
    video_id: str
    frame_id: Optional[Union[int, str]] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    force: bool = False

class DresSubmitQaRequest(BaseModel):
    video_id: str
    answer: str
    frame_id: Optional[Union[int, str]] = None
    time_ms: Optional[int] = None
    force: bool = False

class DresSubmitTrakeRequest(BaseModel):
    video_id: str
    frame_ids: List[Union[int, str]]
    force: bool = False

class DresStartTaskRequest(BaseModel):
    max_duration_s: int = 300

class DresSetEvalRequest(BaseModel):
    evaluation_id: str
    evaluation_name: Optional[str] = None

@app.get("/api/dres/status")
async def api_dres_status():
    """Lấy trạng thái kết nối máy chủ DRES và đợt thi hiện tại."""
    client = dres_client.get_dres_client()
    return {"status": "success", "data": client.get_status()}

@app.post("/api/dres/login")
async def api_dres_login(req: DresLoginRequest):
    """Đăng nhập DRES bằng username/password từ UI hoặc .env."""
    client = dres_client.get_dres_client()
    ok, msg = client.login(username=req.username, password=req.password, server_url=req.server_url)
    return {"status": "success" if ok else "error", "message": msg, "data": client.get_status()}

@app.post("/api/dres/refresh-eval")
async def api_dres_refresh_eval():
    """Tự động làm mới danh sách đợt thi và chọn evaluation ACTIVE."""
    client = dres_client.get_dres_client()
    ok, msg = client.refresh_active_evaluation()
    return {"status": "success" if ok else "warning", "message": msg, "data": client.get_status()}

@app.post("/api/dres/set-evaluation")
async def api_dres_set_evaluation(req: DresSetEvalRequest):
    """Chủ động cấu hình Evaluation ID thủ công nếu cần."""
    client = dres_client.get_dres_client()
    ok, message = client.set_evaluation(req.evaluation_id, req.evaluation_name)
    return {"status": "success" if ok else "error", "message": message, "data": client.get_status()}

@app.post("/api/dres/submit-kis")
async def api_dres_submit_kis(req: DresSubmitKisRequest):
    """Nộp 1-Click đáp án KIS (Textual/Video KIS) bằng frame_id hoặc millisecond (ms)."""
    client = dres_client.get_dres_client()
    res = client.submit_kis(
        video_id=req.video_id,
        frame_id=req.frame_id,
        start_ms=req.start_ms,
        end_ms=req.end_ms,
        force=req.force
    )
    return res

@app.post("/api/dres/submit-qa")
async def api_dres_submit_qa(req: DresSubmitQaRequest):
    """Nộp đáp án thể thức Q&A theo format: QA-<ANSWER>-<VIDEO_ID>-<TIME(ms)>."""
    client = dres_client.get_dres_client()
    res = client.submit_qa(
        video_id=req.video_id,
        answer=req.answer,
        frame_id=req.frame_id,
        time_ms=req.time_ms,
        force=req.force
    )
    return res

@app.post("/api/dres/submit-trake")
async def api_dres_submit_trake(req: DresSubmitTrakeRequest):
    """Nộp đáp án thể thức TRAKE theo format: TR-<VIDEO_ID>-<FRAME_ID1>,<FRAME_ID2>,..."""
    client = dres_client.get_dres_client()
    res = client.submit_trake(
        video_id=req.video_id,
        frame_ids=req.frame_ids,
        force=req.force
    )
    return res

@app.post("/api/dres/start-task")
async def api_dres_start_task(req: DresStartTaskRequest):
    """Bắt đầu câu hỏi mới: xóa cache chống nộp trùng, đặt lại bộ đếm lỗi phạt và khởi động timer."""
    client = dres_client.get_dres_client()
    res = client.start_new_task(max_duration_s=req.max_duration_s)
    return res

# Mount frontend
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

# Chạy server bằng file này (python main.py)
if __name__ == "__main__":
    import uvicorn
    print("=====================================================================")
    print("⏳ [AIC 2026] Backend đang mở; warm-up sẽ chạy nền và báo readiness trên UI.")
    print("🚀 [AIC 2026] Giao diện: http://localhost:8000")
    print("=====================================================================")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        timeout_graceful_shutdown=2,
        timeout_keep_alive=3
    )
