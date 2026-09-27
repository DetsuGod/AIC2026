"""
verify_setup.py — AIC 2026 System Health Check
===============================================
Chạy script này để kiểm tra toàn bộ môi trường trước khi thi đấu:
    python verify_setup.py

Yêu cầu: Python >= 3.10 | Môi trường ảo tại code/local_retrieval/venv/
"""
import os
import sys
import importlib

# ─── ANSI Console Colors ──────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def print_header(title: str) -> None:
    print(f"\n{BLUE}{BOLD}{'=' * 78}{RESET}")
    print(f"{BLUE}{BOLD}  {title}{RESET}")
    print(f"{BLUE}{BOLD}{'=' * 78}{RESET}")

def print_status(name: str, status: str, detail: str = "") -> None:
    if status == "OK":
        badge = f"{GREEN}[ĐẠT - OK  ]{RESET}"
    elif status == "WARN":
        badge = f"{YELLOW}[CẢNH BÁO ]{RESET}"
    else:
        badge = f"{RED}[THIẾU/LỖI]{RESET}"
    print(f"  {badge} {BOLD}{name:<30}{RESET} {detail}")

def hint(text: str) -> None:
    print(f"         {YELLOW}👉 {text}{RESET}")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print_header("AIC 2026 — KIỂM TRA MÔI TRƯỜNG HỆ THỐNG (HEALTH CHECK)")
    print(f"  {CYAN}Kiến trúc hiện tại: Batch 1 + Batch 2 | LanceDB keyframes_k1r"
          f" | Qwen3-VL-Embedding-2B | FastAPI{RESET}")

    project_root = os.path.abspath(os.path.dirname(__file__))
    data_dir     = os.path.join(project_root, "Data")
    code_lr      = os.path.join(project_root, "code", "local_retrieval")

    # Thêm code/local_retrieval vào sys.path để import config
    sys.path.insert(0, code_lr)

    all_ok = True
    warn_count = 0

    # ─── 0. Môi trường Python & venv ──────────────────────────────────────────
    print_header("0. MÔI TRƯỜNG PYTHON & VIRTUALENV")

    venv_standard = os.path.join(code_lr, "venv")
    if sys.version_info >= (3, 10):
        print_status("Python version", "OK", f"Python {sys.version.split()[0]}")
    else:
        print_status("Python version", "WARN",
                     f"Python {sys.version.split()[0]} — khuyến nghị 3.10+")
        warn_count += 1

    if os.path.exists(venv_standard):
        print_status("Virtualenv", "OK", f"code/local_retrieval/venv/")
    else:
        print_status("Virtualenv", "WARN", "Chưa tìm thấy code/local_retrieval/venv/")
        hint("Tạo venv: cd code/local_retrieval && python -m venv venv")
        hint("Cài deps : pip install -r ../../requirements.txt")
        warn_count += 1

    # ─── 1. Thư viện Python ───────────────────────────────────────────────────
    print_header("1. THƯ VIỆN PYTHON (DEPENDENCIES)")
    REQUIRED_PKGS = [
        ("torch",                 "PyTorch"),
        ("transformers",          "HuggingFace Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
        ("lancedb",               "LanceDB Vector Engine"),
        ("pyarrow",               "Apache Arrow"),
        ("fastapi",               "FastAPI Web Backend"),
        ("uvicorn",               "Uvicorn ASGI Server"),
        ("PIL",                   "Pillow Image Library"),
        ("av",                    "PyAV Video Stream Library"),
        ("scipy",                 "SciPy"),
        ("numpy",                 "NumPy"),
        ("google.generativeai",   "Google Generative AI (Gemini) — tùy chọn"),
    ]

    missing_pkgs = []
    for mod_name, label in REQUIRED_PKGS:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "ready")
            print_status(label, "OK", f"v{ver}")
        except ImportError as e:
            optional = "tùy chọn" in label
            status = "WARN" if optional else "FAIL"
            print_status(label, status, f"Chưa cài ({e})")
            if not optional:
                missing_pkgs.append(mod_name)
                all_ok = False
            else:
                warn_count += 1
        except Exception as e:
            print_status(label, "WARN", f"Cảnh báo nạp: {e}")
            warn_count += 1

    if missing_pkgs:
        hint("Cài thiếu: pip install -r requirements.txt")

    # ─── 2. GPU ───────────────────────────────────────────────────────────────
    print_header("2. PHẦN CỨNG & GPU")
    try:
        import torch
        if torch.cuda.is_available():
            dev  = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print_status("GPU (NVIDIA CUDA)", "OK", f"{dev} ({vram:.1f} GB VRAM)")
        else:
            print_status("GPU (NVIDIA CUDA)", "WARN",
                         "Không phát hiện GPU — tự động chạy CPU (tốc độ chậm hơn)")
            hint("Hệ thống vẫn khởi động bình thường; encoder CPU chạy chậm hơn ~5–20x")
            warn_count += 1
    except Exception as e:
        print_status("GPU (NVIDIA CUDA)", "WARN", str(e))
        warn_count += 1

    # ─── 3. Dữ liệu & LanceDB ─────────────────────────────────────────────────
    print_header("3. CƠ SỞ DỮ LIỆU VECTOR (LANCEDB)")

    lancedb_path = os.path.join(data_dir, "aic_lancedb")
    if not os.path.exists(lancedb_path):
        print_status("aic_lancedb/", "FAIL",
                     f"Thư mục không tồn tại: Data/aic_lancedb/")
        hint("Nhận thư mục aic_lancedb/ (~4.4 GB) từ Trưởng nhóm → đặt vào Data/aic_lancedb/")
        all_ok = False
    else:
        try:
            import lancedb
            import warnings
            db = lancedb.connect(lancedb_path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tables = db.table_names()

            # Bảng keyframes chính — ưu tiên keyframes_k1r
            if "keyframes_k1r" in tables:
                tbl   = db.open_table("keyframes_k1r")
                count = len(tbl)
                print_status("keyframes_k1r", "OK",
                             f"{count:,} keyframes (Batch 1 + Batch 2 production)")
            elif "keyframes" in tables:
                tbl   = db.open_table("keyframes")
                count = len(tbl)
                print_status("keyframes (legacy)", "WARN",
                             f"{count:,} — khuyến nghị dùng keyframes_k1r")
                warn_count += 1
            else:
                print_status("keyframes_k1r", "FAIL",
                             "Không tìm thấy bảng 'keyframes_k1r' trong LanceDB")
                hint(f"Các bảng hiện có: {tables}")
                all_ok = False

            # Bảng video_shots
            if "video_shots" in tables:
                tbl   = db.open_table("video_shots")
                count = len(tbl)
                print_status("video_shots", "OK",
                             f"{count:,} shots (RRF signal cho tìm kiếm)")
            else:
                print_status("video_shots", "WARN",
                             "Thiếu bảng video_shots — tín hiệu shot trong RRF sẽ không hoạt động")
                warn_count += 1

            # Bảng ASR (tuỳ chọn)
            if "asr_segments" in tables:
                tbl   = db.open_table("asr_segments")
                count = len(tbl)
                print_status("asr_segments", "OK",
                             f"{count:,} đoạn thoại ASR")
            else:
                print_status("asr_segments", "WARN",
                             "Không có bảng asr_segments — tìm kiếm lời thoại LanceDB tắt")
                warn_count += 1

        except Exception as e:
            print_status("LanceDB", "FAIL", f"Lỗi kết nối: {e}")
            all_ok = False

    # ─── 4. Ảnh Keyframes ─────────────────────────────────────────────────────
    print_header("4. ẢNH KEYFRAMES & VIDEO")

    # Batch 1
    b1_kf = os.path.join(data_dir, "batch 1", "Custom_Keyframes")
    b1_kf_root = os.path.join(data_dir, "Custom_Keyframes")  # đường dẫn rút gọn
    kf_found = False
    for kf_path in [b1_kf_root, b1_kf]:
        if os.path.exists(kf_path):
            n = len([d for d in os.listdir(kf_path) if os.path.isdir(os.path.join(kf_path, d))])
            print_status("Keyframes Batch 1", "OK", f"{n} thư mục video ({kf_path})")
            kf_found = True
            break
    if not kf_found:
        print_status("Keyframes Batch 1", "WARN",
                     "Chưa có Custom_Keyframes — preview ảnh sẽ bị lỗi 404")
        hint("Nhận thư mục Custom_Keyframes/ từ Trưởng nhóm → đặt vào Data/batch 1/Custom_Keyframes/")
        warn_count += 1

    # Batch 2
    b2_kf = os.path.join(data_dir, "batch 2", "Custom_Keyframes")
    if os.path.exists(b2_kf):
        n = len([d for d in os.listdir(b2_kf) if os.path.isdir(os.path.join(b2_kf, d))])
        print_status("Keyframes Batch 2", "OK", f"{n} thư mục video (M/S/N)")
    else:
        print_status("Keyframes Batch 2", "WARN",
                     "Chưa có Data/batch 2/Custom_Keyframes/ — video M/S/N không hiện ảnh")
        hint("Đặt Keyframes nhóm M/S/N vào Data/batch 2/Custom_Keyframes/")
        warn_count += 1

    # Video batch 1
    b1_vid = os.path.join(data_dir, "batch 1", "video")
    b1_vid_root = os.path.join(data_dir, "video")
    vid_found = False
    for vp in [b1_vid_root, b1_vid]:
        if os.path.exists(vp):
            n = len([f for f in os.listdir(vp) if f.lower().endswith((".mp4", ".mov"))])
            print_status("Video Batch 1 (.mp4)", "OK", f"{n} file video")
            vid_found = True
            break
    if not vid_found:
        print_status("Video Batch 1 (.mp4)", "WARN",
                     "Chưa có video batch 1 — streaming video trong timeline sẽ không hoạt động")
        hint("Đặt video L*.mp4 vào Data/batch 1/video/ hoặc Data/video/")
        warn_count += 1

    # Video batch 2
    b2_vid = os.path.join(data_dir, "batch 2", "video")
    if os.path.exists(b2_vid):
        n_mp4 = len([f for f in os.listdir(b2_vid) if f.lower().endswith(".mp4")])
        n_mov = len([f for f in os.listdir(b2_vid) if f.lower().endswith(".mov")])
        print_status("Video Batch 2 (MP4/MOV)", "OK",
                     f"{n_mp4} .mp4 (M/S) + {n_mov} .mov (N camera giao thông)")
    else:
        print_status("Video Batch 2 (MP4/MOV)", "WARN",
                     "Chưa có Data/batch 2/video/ — video M/S/N không streaming được")
        hint("Đặt video M*.mp4, S*.mp4, N*.mov vào Data/batch 2/video/")
        warn_count += 1

    # ─── 5. File dữ liệu phụ ─────────────────────────────────────────────────
    print_header("5. FILE DỮ LIỆU BỔ TRỢ")

    AUX_FILES = [
        (os.path.join(data_dir, "asr_mapping.json"),
         "ASR Mapping (lời thoại Whisper)",
         "Tìm kiếm lời thoại TF-IDF vẫn dùng file JSON riêng"),
        (os.path.join(data_dir, "video_fps_mapping.json"),
         "Video FPS Mapping (1.487 video)",
         "FPS chính xác → timestamp ms → DRES submission"),
        (os.path.join(data_dir, "video_categories.json"),
         "Taxonomy Filter (7 danh mục)",
         "Bộ lọc danh mục: am_thuc, tin_tuc, day_hoc, du_lich_van_hoa, mua_lan, dua_xe_dap, camera_giao_thong"),
        (os.path.join(data_dir, "kf_shot_map.json"),
         "KF ↔ Shot Map (bidirectional)",
         "Ánh xạ keyframe ↔ video shot cho RRF"),
        (os.path.join(data_dir, "usearch_qwen_index_mapping_k1r.json"),
         "Qwen K1R Mapping JSON (476k entries)",
         "Ánh xạ vector index → video_id/frame_id"),
    ]

    for fpath, label, desc in AUX_FILES:
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print_status(label, "OK", f"{size_mb:.1f} MB")
        else:
            print_status(label, "WARN", f"Thiếu: {os.path.basename(fpath)}")
            hint(desc)
            warn_count += 1

    # ─── 6. Cấu hình & Secrets ────────────────────────────────────────────────
    print_header("6. CẤU HÌNH & BIẾN MÔI TRƯỜNG")

    env_path = os.path.join(project_root, ".env")
    env_example = os.path.join(project_root, ".env.example")

    if os.path.exists(env_path):
        print_status(".env file", "OK", "Tồn tại (sẽ được nạp tự động khi khởi động)")
    else:
        print_status(".env file", "WARN", "Chưa có — tính năng Gemini & DRES sẽ không kết nối được")
        hint(f"Sao chép file mẫu: cp .env.example .env  rồi điền GEMINI_API_KEY và DRES credentials")
        warn_count += 1

    # Import config để kiểm tra các giá trị runtime
    try:
        import config  # noqa: module in sys.path

        # Gemini API Key
        gemini_key = getattr(config, "GEMINI_API_KEY", "")
        if gemini_key:
            masked = gemini_key[:6] + "..." + gemini_key[-4:]
            print_status("GEMINI_API_KEY", "OK", f"Đã cấu hình ({masked})")
        else:
            print_status("GEMINI_API_KEY", "WARN",
                         "Chưa cấu hình — AI phân tích truy vấn (Gemini Parser) sẽ dùng fallback")
            hint("Điền GEMINI_API_KEY vào .env  (lấy tại aistudio.google.com)")
            warn_count += 1

        # DRES Credentials
        dres_url  = getattr(config, "DRES_SERVER_URL", "")
        dres_user = getattr(config, "DRES_USERNAME", "")
        dres_pass = getattr(config, "DRES_PASSWORD", "")
        if dres_user and dres_pass:
            print_status("DRES Credentials", "OK",
                         f"Server: {dres_url} | User: {dres_user}")
        else:
            print_status("DRES Credentials", "WARN",
                         "Chưa cấu hình — nộp bài DRES live sẽ chạy dry-run cục bộ")
            hint("Điền DRES_USERNAME, DRES_PASSWORD vào .env (nhận từ BTC AIC 2026)")
            warn_count += 1

    except ImportError as e:
        print_status("config.py", "FAIL", f"Lỗi import config: {e}")
        all_ok = False

    # ─── 7. Mã nguồn runtime ─────────────────────────────────────────────────
    print_header("7. MÃ NGUỒN RUNTIME (SOURCE FILES)")
    RUNTIME_FILES = [
        ("code/local_retrieval/config.py",        "Cấu hình tập trung"),
        ("code/local_retrieval/search_engine.py", "Hybrid Search Engine (Qwen3-VL + ASR + Shot RRF)"),
        ("code/local_retrieval/text_encoder.py",  "Qwen3-VL Text Encoder"),
        ("code/local_retrieval/gemini_parser.py", "Gemini AI Query Parser"),
        ("code/local_retrieval/asr_engine.py",    "ASR Hybrid Search (Whisper + TF-IDF)"),
        ("code/local_retrieval/dres_client.py",   "DRES Live Contest Client (Chung Kết)"),
        ("code/web/backend/main.py",              "FastAPI Backend Server"),
        ("code/web/frontend/index.html",          "Frontend Giao Diện Web"),
        ("code/web/frontend/app.js",              "Frontend Controller (TRAKE, DRES, Timeline)"),
        ("code/web/frontend/style.css",           "Dark Slate UI Styling"),
    ]
    for rel_path, desc in RUNTIME_FILES:
        full = os.path.join(project_root, rel_path)
        if os.path.exists(full):
            size_kb = os.path.getsize(full) / 1024
            print_status(os.path.basename(rel_path), "OK", f"{desc} ({size_kb:.0f} KB)")
        else:
            print_status(os.path.basename(rel_path), "FAIL", f"Thiếu file: {rel_path}")
            all_ok = False

    # ─── KẾT LUẬN ─────────────────────────────────────────────────────────────
    print_header("KẾT LUẬN & HƯỚNG DẪN TIẾP THEO")

    if all_ok and warn_count == 0:
        print(f"\n  {GREEN}{BOLD}🎉 XUẤT SẮC! TOÀN BỘ MÔI TRƯỜNG ĐẠT CHUẨN 100%.{RESET}")
    elif all_ok:
        print(f"\n  {YELLOW}{BOLD}⚡ THÀNH PHẦN LÕI ĐẠT CHUẨN — {warn_count} cảnh báo nhỏ (xem ở trên).{RESET}")
    else:
        print(f"\n  {RED}{BOLD}⚠️  PHÁT HIỆN LỖI — Cần khắc phục trước khi thi đấu:{RESET}")
        if missing_pkgs:
            print(f"\n  1. Cài thư viện thiếu:")
            print(f"     {BOLD}pip install -r requirements.txt{RESET}")
        if not os.path.exists(lancedb_path):
            print(f"\n  2. Bổ sung cơ sở dữ liệu vector:")
            print(f"     Nhận {BOLD}aic_lancedb/{RESET} từ Trưởng nhóm → đặt vào {BOLD}Data/aic_lancedb/{RESET}")

    print(f"""
  {BOLD}Khởi động hệ thống:{RESET}
  • Windows  : Nháy đúp {BOLD}run_app.bat{RESET}  (tự động mở trình duyệt)
  • Linux    : {BOLD}bash run_app.sh{RESET}
  • Thủ công : {BOLD}code/local_retrieval/venv/Scripts/python code/web/backend/main.py{RESET}
  • Truy cập : {CYAN}http://127.0.0.1:8000{RESET}

  {BOLD}Tài liệu chi tiết:{RESET}
  • Kiến trúc hệ thống  : {BOLD}PROJECT_CONTEXT.md{RESET}
  • Từ điển dữ liệu     : {BOLD}code/DATA_DICTIONARY.md{RESET}
  • Hướng dẫn triển khai: {BOLD}README_VN.md{RESET}
""")

if __name__ == "__main__":
    main()
