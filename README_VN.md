<div align="center">

# 🎬 AIC 2026: Hệ Thống Truy Vấn Video Đa Phương Thức & Tìm Kiếm Tương Tác

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20CUDA-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![LanceDB](https://img.shields.io/badge/LanceDB-476k%20Keyframes-8A2BE2)](https://lancedb.github.io/lancedb/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Hệ thống truy vấn video đa phương thức tương tác cấp độ sản phẩm (Production-grade) phục vụ **Vòng Chung Kết Cuộc thi Trí tuệ Nhân tạo TP.HCM (AIC 2026)**. Hỗ trợ toàn diện ba thể thức thi đấu: Known-Item Search (KIS), Visual Question Answering (QA), và Temporal Reasoning & Keyframe Extraction (TRAKE).*

[Tiếng Việt](README_VN.md) • [English](README.md) • [Kiến Trúc](#-tổng-quan-kiến-trúc-hệ-thống) • [Triển Khai Nhanh](#-hướng-dẫn-triển-khai-nhanh-quickstart) • [Dành Cho AI Agent](#-hướng-dẫn-dành-cho-ai-agent-triển-khai-tự-động)

</div>

---

## 🌟 Điểm Nổi Bật (Highlights)

| Thành Phần | Thông Số Kỹ Thuật & Chi Tiết Triển Khai |
|---|---|
| **Mô hình Thị giác (Visual)** | `Qwen/Qwen3-VL-Embedding-2B` — Vector siêu cầu 2048 chiều (L2-normalized) |
| **Quy mô Dữ liệu** | **1.487 videos**, **476.140 keyframes** (Batch 1 + Batch 2) |
| **Định dạng Video** | `.mp4` (Batch 1 nhóm L) và `.mp4` / `.mov` (Batch 2 nhóm M, S, N) |
| **Cơ sở Dữ liệu Vector** | LanceDB với 3 bảng: `keyframes_k1r` (476k), `video_shots` (186k), `asr_segments` (16k) |
| **Nhận dạng Giọng nói (ASR)** | Faster-Whisper Large-V3 + Chỉ mục N-gram TF-IDF (Tiếng Việt & Tiếng Anh) |
| **Chiến thuật Dung hợp (Fusion)** | RRF (Reciprocal Rank Fusion) & Weighted Fusion qua các kênh Visual / ASR / Shot / Caption |
| **Đấu nối Thi đấu (Contest)** | DRES Live Client tích hợp sẵn (KIS / QA / TRAKE) với cơ chế Chống Nộp Trùng và Dry-Run |
| **Giao diện Người dùng (UI)** | Dark Slate — Phong cách tối giản Linear / Raycast Pro Minimalist |

---

## 🏛️ Tổng Quan Kiến Trúc Hệ Thống

```
Câu truy vấn tự nhiên (Tiếng Việt / Tiếng Anh)
        │
        ▼
[Gemini AI Parser]
  Bóc tách thành phần: visual_query / speech_query / qa_question / multi-scene events
        │
        ▼
[Giai đoạn 1 — Tìm kiếm Nhanh Đa Phương Thức (Top K1 Ứng viên)]
  ├─ Thị giác: Qwen3-VL-Embedding-2B → LanceDB cosine search (476.140 keyframes)
  ├─ Lời thoại: Faster-Whisper + TF-IDF N-gram (16.609 speech segments)
  ├─ Phân cảnh: Qwen3-VL Video 2048D → LanceDB video_shots (186.102 shots)
  ├─ Chú thích: TF-IDF Shot Captions (51.650 captions — tùy chọn, mặc định tắt)
  └─ Khử trùng lặp (Dedup): Smart Intra-video & Cross-video deduplication
        │
        ▼
[Giai đoạn 2 — Tìm kiếm Thời gian TRAKE / Multi-Scene]
  ├─ Quy hoạch động (Temporal DP): Ràng buộc đơn điệu t(E1) < t(E2) < … < t(En)
  ├─ Dung hợp Alpha (Alpha-Fusion): 0.35 × GlobalScore + 0.65 × DPScore
  └─ Tìm kiếm tia đa dạng (Diverse Beam Search): Đảm bảo đa dạng ứng viên video
        │
        ▼
[Giao diện Tác chiến Web — FastAPI + Vanilla HTML5/CSS/JS]
  ├─ Nộp bài trực tiếp tới server DRES với độ trễ mili-giây
  ├─ Ctrl + Click trực tiếp trên Video Player → nộp bài ngay tại frame/ms hiện tại
  ├─ Phím tắt Alt + G → mở nhanh video và nhảy tới frame/thời gian bất kỳ
  └─ Bộ lọc Taxonomy (7 danh mục) + Trục thời gian phân cảnh Timeline Strip
```

---

## 📂 Sơ Đồ Cấu Trúc Thư Mục

```
2026-AIC/
├── Data/                                  # [KHÔNG ĐƯA LÊN GIT] Nhận từ Trưởng nhóm
│   ├── aic_lancedb/                       # CSDL Vector LanceDB (~4.4 GB)
│   │   ├── keyframes_k1r.lance/           # 476.140 keyframes × 2048D (Bảng chính)
│   │   ├── video_shots.lance/             # 186.102 video shots × 2048D
│   │   └── asr_segments.lance/            # 16.609 đoạn thoại Whisper ASR
│   ├── batch 1/
│   │   ├── Custom_Keyframes/              # Ảnh JPG trích xuất từ video nhóm L
│   │   └── video/                         # File video gốc L*.mp4
│   ├── batch 2/
│   │   ├── Custom_Keyframes/              # Ảnh JPG trích xuất từ video nhóm M/S/N
│   │   └── video/                         # Video gốc M*.mp4, S*.mp4, N*.mov
│   ├── asr_mapping.json                   # Ánh xạ đoạn thoại Whisper ASR
│   ├── video_fps_mapping.json             # Bảng FPS chuẩn của toàn bộ 1.487 video
│   ├── video_categories.json              # Phân loại chuyên đề (Taxonomy) 1.487 video
│   ├── kf_shot_map.json                   # Bản đồ ánh xạ 2 chiều Keyframe ↔ Video Shot
│   └── usearch_qwen_index_mapping_k1r.json# Ánh xạ vị trí index vector → video_id/frame_id
│
├── code/
│   ├── local_retrieval/                   # Engine tìm kiếm lõi
│   │   ├── venv/                          # ⭐ Môi trường ảo Python (KHÔNG đưa lên Git)
│   │   ├── config.py                      # Cấu hình tập trung & nạp biến môi trường .env
│   │   ├── search_engine.py               # Lõi truy vấn Hybrid đa phương thức + TRAKE DP
│   │   ├── text_encoder.py                # Qwen3-VL Text & Image Encoder
│   │   ├── gemini_parser.py               # Phân tích câu hỏi tự nhiên bằng Gemini AI
│   │   ├── asr_engine.py                  # Lõi tìm kiếm giọng nói kết hợp TF-IDF
│   │   ├── dres_client.py                 # DRES Live Client cho Vòng Chung Kết
│   │   ├── caption_engine.py              # Truy vấn TF-IDF chú thích ngữ cảnh phân cảnh
│   │   └── test_batch2_integration.py     # Script kiểm tra tích hợp dữ liệu
│   └── web/
│       ├── backend/main.py                # Server FastAPI (API, Stream video, DRES Proxy)
│       └── frontend/
│           ├── index.html                 # Giao diện Web tác chiến
│           ├── app.js                     # Trình điều khiển Frontend (TRAKE, DRES, Timeline)
│           └── style.css                  # Bảng kiểu Dark Slate phong cách tối giản
│
├── verify_setup.py                        # Script tự động kiểm tra sức khỏe hệ thống (Health Check)
├── run_app.bat                            # Khởi động 1-click cho Windows
├── run_app.sh                             # Khởi động 1-click cho Linux / macOS
├── requirements.txt                       # Danh mục thư viện Python bắt buộc
├── .env.example                           # Mẫu cấu hình biến môi trường
├── PROJECT_CONTEXT.md                     # Tài liệu thiết kế hệ thống chi tiết
└── code/DATA_DICTIONARY.md                # Từ điển dữ liệu toàn diện
```

---

## 🤖 Hướng Dẫn Dành Cho AI Agent (Triển Khai Tự Động)

> **Dành cho đồng đội sử dụng AI Coding Assistant (Google Antigravity, Cursor, Claude Code, GitHub Copilot):**  
> Sao chép toàn bộ khối lệnh dưới đây và gửi thẳng vào cửa sổ chat của Agent để tự động hóa 100% quá trình cài đặt:

```markdown
Bạn là trợ lý lập trình phụ trách triển khai hệ thống AIC 2026. Hãy đọc kỹ file README_VN.md và hoàn thành các bước sau:
1. Tạo một Virtualenv tại đường dẫn: `code/local_retrieval/venv/` sử dụng Python >= 3.10.
2. Cài đặt toàn bộ dependencies: chạy `code/local_retrieval/venv/Scripts/pip install -r requirements.txt` (trên Windows) hoặc `source code/local_retrieval/venv/bin/activate && pip install -r requirements.txt` (trên Linux).
3. Nếu chưa có file `.env`, tự động sao chép từ `.env.example`.
4. Chạy script kiểm tra: `code/local_retrieval/venv/Scripts/python verify_setup.py`.
5. Đảm bảo toàn bộ các mục chính đạt `[ĐẠT - OK]`. Sau đó khởi động server bằng `run_app.bat` hoặc lệnh `python code/web/backend/main.py` và báo lại địa chỉ truy cập `http://127.0.0.1:8000`.
```

---

## 🚀 Hướng Dẫn Triển Khai Nhanh (Quickstart)

### Bước 1: Clone mã nguồn về máy

```bash
git clone https://github.com/liem-2006/aic2026.git
cd aic2026
```

### Bước 2: Tạo môi trường ảo `venv` và cài đặt thư viện

> ⚠️ **Quy chuẩn bắt buộc:** Môi trường ảo **phải** được đặt tại **`code/local_retrieval/venv/`** để tương thích với các script khởi động `run_app.bat` và `run_app.sh`.

#### Trên hệ điều hành Windows:
```powershell
cd code\local_retrieval
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r ..\..\requirements.txt
cd ..\..
```

#### Trên hệ điều hành Linux / macOS:
```bash
cd code/local_retrieval
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r ../../requirements.txt
cd ../..
```

### Bước 3: Cấu hình biến môi trường (`.env`)

Tạo file `.env` từ file mẫu:
```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Mở file `.env` và điền thông tin:
```ini
# Khóa Gemini AI API (Phân tích câu hỏi tự nhiên — Tùy chọn)
GEMINI_API_KEY=AIzaSy...

# Cấu hình máy chủ chấm thi DRES (Vòng Chung Kết)
DRES_SERVER_URL=https://eventretrieval.one
DRES_USERNAME=team_xxx
DRES_PASSWORD=mat_khau_cua_ban
```

### Bước 4: Đặt dữ liệu vào thư mục `Data/`

Nhận từ Trưởng nhóm và đặt đúng vị trí các tài nguyên sau:
* **Tối thiểu để tìm kiếm (Vector Search):**
  * `Data/aic_lancedb/` (Chứa bảng `keyframes_k1r`, `video_shots`, `asr_segments` — ~4.4 GB)
  * `Data/video_fps_mapping.json` (Quy đổi khung hình sang mili-giây ms)
  * `Data/asr_mapping.json` (Dữ liệu giọng nói Whisper)
  * `Data/video_categories.json` (Phân loại 7 danh mục chuyên đề)
* **Để xem ảnh Keyframe xem trước (Preview):**
  * `Data/batch 1/Custom_Keyframes/` (Nhóm video L)
  * `Data/batch 2/Custom_Keyframes/` (Nhóm video M, S, N)
* **Để phát Video và xem Trục thời gian (Timeline & Streaming):**
  * `Data/batch 1/video/` (Các file `L*.mp4`)
  * `Data/batch 2/video/` (Các file `M*.mp4`, `S*.mp4`, `N*.mov`)

### Bước 5: Kiểm tra sức khỏe môi trường (Health Check)

Chạy script chẩn đoán toàn diện:
```bash
# Windows
code\local_retrieval\venv\Scripts\python.exe verify_setup.py

# Linux / macOS
code/local_retrieval/venv/bin/python verify_setup.py
```

Nếu màn hình hiển thị:
```
🎉 XUẤT SẮC! TOÀN BỘ MÔI TRƯỜNG ĐẠT CHUẨN 100%.
```
Hệ thống của bạn đã sẵn sàng cho trận đấu!

### Bước 6: Khởi động hệ thống (1-Click)

* **Windows:** Nháy đúp vào **`run_app.bat`** (tự động kích hoạt venv, chạy server và mở trình duyệt).
* **Linux / macOS:** Chạy lệnh **`bash run_app.sh`**.
* **Thủ công:**
  ```bash
  code/local_retrieval/venv/Scripts/python code/web/backend/main.py
  ```

Mở trình duyệt tại địa chỉ: **`http://127.0.0.1:8000`**.

> ⏳ *Lưu ý:* Khi khởi động lần đầu, hệ thống cần khoảng 15–30 giây để nạp chỉ mục LanceDB và ASR vào bộ nhớ đệm (Warm-up). Khi huy hiệu trên thanh điều khiển chuyển sang màu xanh **`READY`**, bạn có thể bắt đầu tác chiến.

---

## 🖥️ Tính Năng Tác Chiến Trên Giao Diện Web

### 1. Các Chế Độ Tìm Kiếm
* 🔍 **KIS Search (`SEARCH NOW`)**: Tìm kiếm phân cảnh đơn lẻ (Known-Item Search). Bấm `Enter` hoặc click nút để tìm.
* 🎬 **Multi-Scene (`MULTI-SCENE`)**: Tìm kiếm phân cảnh đa sự kiện theo thứ tự xuất hiện trong kịch bản.
* ⏱️ **TRAKE Search (`SEARCH TRAKE`)**: Tìm kiếm chuỗi sự kiện có tính ràng buộc thời gian nghiêm ngặt $t(E_1) < t(E_2) < \dots < t(E_n)$ dựa trên thuật toán Quy hoạch động đơn điệu TARS Monotonic DP kết hợp Alpha-Fusion.

### 2. Tương Tác & Nộp Bài Nhanh DRES Live
* **`Ctrl + Click` trực tiếp trên Video Player**:
  * **Chế độ KIS:** Trích xuất ngay lập tức vị trí thời gian hiện tại (tính bằng mili-giây ms hoặc frame) và gửi thẳng lên máy chủ DRES.
  * **Chế độ QA:** Mở hộp thoại nhập câu trả lời trực tiếp ngay trên khung hình hiện tại mà không bị chìm giao diện.
  * **Chế độ TRAKE:** Chọn nhanh phân cảnh cho sự kiện kế tiếp trong chuỗi sự kiện lineup (duy trì đúng thứ tự $E_1 \to E_2 \to \dots$).
* **Phím tắt `Alt + G` (Mở Nhanh Video & Frame)**:
  * Cho phép nhập bất kỳ Video ID nào và nhảy tới giây/khung hình mong muốn, ngay cả khi video đó không nằm trong danh sách Top-K tìm kiếm.
* **Xác thực Chuỗi Sự Kiện TRAKE**:
  * Các sự kiện đã chọn hiển thị rõ ràng trên thanh Lineup theo đúng thứ tự chọn.
  * Có nút quay lại (**Jump-back**) để xem lại video tại đúng thời điểm của từng sự kiện trước khi nhấn nộp bài chính thức.

### 3. Công Cụ Hỗ Trợ Tác Chiến Nâng Cao
* 🗂️ **Bộ lọc Chuyên đề (Taxonomy Filter)**: 7 danh mục chuyên biệt (`am_thuc`, `tin_tuc`, `day_hoc`, `du_lich_van_hoa`, `mua_lan`, `dua_xe_dap`, `camera_giao_thong`) giúp thu hẹp phạm vi không gian tìm kiếm ngay tức thì.
* 🎞️ **Trục thời gian phân cảnh (Timeline Inspector)**: Hiển thị toàn bộ chuỗi keyframes của video xung quanh điểm tìm kiếm, hỗ trợ xác định chính xác thời điểm bắt đầu/kết thúc sự kiện.
* 🔍 **Lightbox Phóng To Chi Tiết**: Hỗ trợ Pan & Zoom lên đến 1000% để đọc chữ, biển số xe, chi tiết đồ vật nhỏ.
* ⚖️ **Dung Hợp Linh Hoạt (RRF / Weighted)**: Chuyển đổi linh hoạt giữa thuật toán Reciprocal Rank Fusion và Dung hợp trọng số qua thanh điều khiển bên trái.
* 🛡️ **Chống Nộp Trùng & Cảnh Báo Phạt**: Tự động chặn các yêu cầu nộp trùng lặp trong thời gian ngắn và ghi nhận lịch sử nộp bài nhằm bảo vệ điểm số đội thi.

---

## ❓ Xử Lý Sự Cố Thường Gặp (Troubleshooting)

| Tình Huống | Nguyên Nhân Thường Gặp | Cách Khắc Phục |
|---|---|---|
| `DLL load failed while importing _lancedb` | Thiếu bộ thư viện C++ Runtime trên Windows | Cài đặt [Microsoft Visual C++ 2015-2022 Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) rồi khởi động lại máy |
| `WinError 10048` (Address already in use) | Một tiến trình khác (hoặc backend cũ) đang chiếm cổng 8000 | Tắt tiến trình cũ trong Task Manager hoặc đổi sang cổng khác trong file `main.py` |
| Ảnh Keyframe bị lỗi 404 hoặc không hiển thị | Sai đường dẫn hoặc thiếu thư mục ảnh | Kiểm tra lại cấu trúc: `Data/batch 1/Custom_Keyframes/{VIDEO_ID}/{FRAME}.jpg` và `Data/batch 2/Custom_Keyframes/` |
| Video nhóm N (.mov) không phát được | Trình duyệt cũ hoặc thiếu codec MOV | Sử dụng trình duyệt Chrome/Edge phiên bản mới nhất (Backend đã hỗ trợ stream HTTP Range cho MOV) |
| Tìm kiếm TRAKE bị chậm | Tham số `Top Videos` hoặc `Frames/Video` quá cao | Giảm `Top Videos` xuống 10 (mặc định) và `Frames/Video` xuống 40 trong bảng cấu hình TRAKE |
| Gemini Parser không phản hồi | Chưa điền khóa API hoặc hết quota | Điền `GEMINI_API_KEY` vào file `.env`; nếu không có khóa, hệ thống sẽ tự động dùng bộ phân tích Fallback cục bộ |
| Máy không có GPU NVIDIA (CUDA) | Môi trường máy tính văn phòng / CPU-only | Hệ thống tự động kích hoạt chế độ **CPU Fallback** (tốc độ chậm hơn 5–20 lần nhưng không gây crash hệ thống) |

---

## 👥 Đội Ngũ & Tài Liệu Tham Khảo

Phát triển phục vụ Vòng Chung Kết **Ho Chi Minh City AI Challenge (AIC 2026)**.

* *Qwen3-VL-Embedding* (arXiv:2601.04720) — Nền tảng trích xuất vector đa phương thức
* *TARS Monotonic DP for Temporal Reasoning* (AIC 2025, độ chính xác 93.15%)
* *LanceDB Embedded Vector Architecture* — Truy vấn vector thời gian thực dưới 2ms
* *Faster-Whisper Large-V3 ASR* — Nhận dạng giọng nói tiếng Việt độ chính xác cao

---

## 📄 Giấy Phép (License)

Dự án được phân phối dưới giấy phép [MIT License](LICENSE). Dữ liệu video thi đấu và các bảng vector độc quyền được quản lý theo quy định bảo mật của Ban tổ chức và **không được đưa lên kho mã nguồn công khai**.
