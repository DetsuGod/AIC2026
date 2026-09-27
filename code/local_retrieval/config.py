import os

# Đường dẫn gốc của dự án (tính từ file này lùi ra 2 cấp: code/local_retrieval -> code -> project_root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

# Tự động nạp cấu hình từ .env nếu tồn tại
env_path = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(env_path):
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k and v and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

# --- LANCEDB VECTOR DATABASE ---
LANCEDB_PATH = os.path.join(DATA_DIR, "aic_lancedb")
USE_LANCEDB = True

# --- INDEX CHÍNH (Qwen3-VL-Embedding-2B) ---
# Tự động ưu tiên nạp lát cắt K1R (297,332 frames sau khi lọc), bảo toàn bảng gốc 'keyframes' làm backup
QWEN_INDEX_K1R_PATH = os.path.join(DATA_DIR, "usearch_qwen_index_k1r.npy")
QWEN_MAPPING_K1R_PATH = os.path.join(DATA_DIR, "usearch_qwen_index_mapping_k1r.json")
RTDETR_INDEX_K1R_PATH = os.path.join(DATA_DIR, "usearch_rtdetr_index_k1r.npy")

if os.path.exists(QWEN_MAPPING_K1R_PATH) or os.path.exists(os.path.join(LANCEDB_PATH, 'keyframes_k1r.lance')):
    KEYFRAME_TABLE_NAME = "keyframes_k1r"
    QWEN_INDEX_PATH = QWEN_INDEX_K1R_PATH
    QWEN_MAPPING_PATH = QWEN_MAPPING_K1R_PATH
    RTDETR_INDEX_PATH = RTDETR_INDEX_K1R_PATH if os.path.exists(RTDETR_INDEX_K1R_PATH) else os.path.join(DATA_DIR, "usearch_rtdetr_index.npy")
else:
    KEYFRAME_TABLE_NAME = "keyframes"
    QWEN_INDEX_PATH = os.path.join(DATA_DIR, "usearch_qwen_index.npy")
    QWEN_MAPPING_PATH = os.path.join(DATA_DIR, "usearch_qwen_index_mapping.json")
    RTDETR_INDEX_PATH = os.path.join(DATA_DIR, "usearch_rtdetr_index.npy")

# Metadata Bounding Boxes tọa độ pixel của từng keyframe (420.79 MB)
RTDETR_METADATA_PATH = os.path.join(DATA_DIR, "rtdetr_metadata.json")

# RT-DETR ngừng dùng ở cả hai batch. Giữ tên legacy để đọc state cũ mà không
# kích hoạt lại kênh object.
OBJECT_SEARCH_ENABLED = False

# 80 lớp đối tượng chuẩn COCO
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Thư mục chứa ảnh keyframe (tự động phát hiện)
if os.path.exists(os.path.join(DATA_DIR, "Custom_Keyframes")):
    KF_DIR = os.path.join(DATA_DIR, "Custom_Keyframes")
elif os.path.exists(os.path.join(DATA_DIR, "batch 1", "Custom_Keyframes")):
    KF_DIR = os.path.join(DATA_DIR, "batch 1", "Custom_Keyframes")
else:
    KF_DIR = os.path.join(DATA_DIR, "Custom_Keyframes")

# Thư mục chứa video gốc .mp4 (tự động phát hiện)
if os.path.exists(os.path.join(DATA_DIR, "video")):
    VIDEO_DIR = os.path.join(DATA_DIR, "video")
elif os.path.exists(os.path.join(DATA_DIR, "batch 1", "video")):
    VIDEO_DIR = os.path.join(DATA_DIR, "batch 1", "video")
else:
    VIDEO_DIR = os.path.join(DATA_DIR, "video")

BATCH2_DIR = os.path.join(DATA_DIR, "batch 2")
BATCH2_KF_DIR = os.path.join(BATCH2_DIR, "Custom_Keyframes")
BATCH2_VIDEO_DIR = os.path.join(BATCH2_DIR, "video")


def normalize_video_id(video_id: str) -> str:
    clean = str(video_id or "").strip()
    if clean.lower().endswith((".mp4", ".mov")):
        clean = os.path.splitext(clean)[0]
    return clean.upper()


def keyframe_directory(video_id: str) -> str:
    clean = normalize_video_id(video_id)
    root = BATCH2_KF_DIR if clean.startswith(("M", "N", "S")) else KF_DIR
    return os.path.join(root, clean)


def video_file(video_id: str) -> str:
    clean = normalize_video_id(video_id)
    root = BATCH2_VIDEO_DIR if clean.startswith(("M", "N", "S")) else VIDEO_DIR
    for ext in (".mp4", ".mov"):
        candidate = os.path.join(root, clean + ext)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(root, clean + (".mov" if clean.startswith("N") else ".mp4"))

# --- MODEL ---
TEXT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
QWEN_RERANKER_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
EMBEDDING_DIM = 2048

# --- RETRIEVAL PARAMS ---
TOP_K = 100
DEFAULT_OBJECT_THRESHOLD = 0.3

# --- TRAKE PERFORMANCE GUARDRAILS ---
# Top V va Frames/Video nhan voi so event de tao ra so lan refine. Gioi han
# tap trung o config de API va retrieval engine dung chung, tranh localStorage
# cu vo tinh khoi phuc cau hinh qua lon.
TRAKE_DEFAULT_TOP_V = 10
TRAKE_MAX_TOP_V = 50
TRAKE_DEFAULT_FRAMES_PER_VIDEO = 40
TRAKE_MAX_FRAMES_PER_VIDEO = 150

# --- ASR & VIDEO METADATA ---
ASR_MAPPING_PATH = os.path.join(DATA_DIR, "asr_mapping.json")
FPS_MAPPING_PATH = os.path.join(DATA_DIR, "video_fps_mapping.json")

# --- VIDEO SHOTS & MAPPING (Qwen3-VL Video 2048D) ---
VIDEO_SHOT_TABLE = "video_shots"
KF_SHOT_MAP_PATH = os.path.join(DATA_DIR, "kf_shot_map.json")
SHOT_METADATA_MAP_PATH = os.path.join(DATA_DIR, "shot_metadata_map.json")

# Thư mục chứa video phân cảnh đã cắt (.mp4 con)
if os.path.exists(os.path.join(DATA_DIR, "batch 1", "output-video-encode", "cut_videos")):
    CUT_VIDEOS_BASE = os.path.join(DATA_DIR, "batch 1", "output-video-encode", "cut_videos")
elif os.path.exists(os.path.join(DATA_DIR, "cut_videos")):
    CUT_VIDEOS_BASE = os.path.join(DATA_DIR, "cut_videos")
else:
    CUT_VIDEOS_BASE = os.path.join(DATA_DIR, "batch 1", "output-video-encode", "cut_videos")

# --- GEMINI API KEY (MẶC ĐỊNH CHO AIC 2026) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# --- DRES CONTEST CLIENT (VÒNG CHUNG KẾT) ---
DRES_SERVER_URL = os.environ.get("DRES_SERVER_URL", "https://eventretrieval.one")
DRES_USERNAME = os.environ.get("DRES_USERNAME", "")
DRES_PASSWORD = os.environ.get("DRES_PASSWORD", "")

