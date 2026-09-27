# 📚 Từ Điển Dữ Liệu & Tài Nguyên Dự Án (Data Dictionary)

Tài liệu này đóng vai trò như một bản đồ tri thức (Knowledge Base) để các Agent AI và Developer hiểu rõ mục đích, cấu trúc và vị trí của các luồng dữ liệu, file index, model và module mã nguồn trong dự án AIC 2026.

> **Cập nhật lần cuối:** 2026-09-26 (batch 2 đã tích hợp) | LanceDB production giữ **3 bảng**: `keyframes_k1r` (476.140 frames), `asr_segments` (16.609 segments), `video_shots` (186.102 shots) | Model: **Qwen3-VL-Embedding-2B (2048 dim)** + **Faster-Whisper ASR**. RT-DETR đã ngừng dùng.

## Trạng thái batch 2 production

- `Data/batch 2/video/`: 614 video (M: 304 MP4, S: 12 MP4, N: 298 MOV). `Custom_Keyframes/`, `map-keyframes-k1r/` và `qwen3vl_img/` cung cấp 178.808 keyframe active. 3.409 vector thiếu ảnh bị loại, bản nguồn và danh sách bỏ qua nằm trong `manifests/`.
- `output-video-encode/video_shot_indices/`: 86.012 shot vector nguồn S/N; 74.501 shot được nhập. 11.511 shot không có KF active trong khoảng thời gian nên loại. Không có clip shot phát trên UI.
- `manifests/batch2_inventory.json` là manifest theo video; `integration_report.json` là số nhập thực tế; `pre_integration/` giữ bản các index/map trước khi publish để khôi phục nếu cần. `code/local_retrieval/integrate_batch2.py --verify` đối chiếu tổng bảng/map/NPY.
- ASR toàn batch 2 và shot embedding nhóm M đang trích xuất, **pending**. Không giả lập dữ liệu rỗng như thể extraction đã hoàn tất. 19 video N không có JPG/KF active vẫn mở được video gốc.
- `Data/video_categories.json`: 1.487 video; M → `tin_tuc`, S → `dua_xe_dap`, N → `camera_giao_thong`. Media-info S dùng `S01-V*.json` theo đính chính BTC; N chưa có media-info.
- Backend `/images/{video_id}/{frame}.jpg` và `/api/video-file/{video_id}` dùng resolver MP4/MOV. KF batch 2 dùng `pts_time` từ CSV embedding để mở timeline và quy đổi DRES, không suy diễn chỉ bằng frame/FPS. HTTP Range cho MOV đã được kiểm thử.
- Caption batch 1: nguồn `Data/batch 1/caption/qwen3vl_8b_ctx_L23_L26.json` có 52.720 record; 51.650 record ghép hợp lệ shot+keyframe production và nằm trong `Data/batch 1/caption/retrieval_index/` (TF-IDF). 1.070 bị cách ly. Công tắc Caption mặc định tắt. RRF trung bình theo các kênh khả dụng của từng KF; chưa có benchmark chất lượng đủ để bật mặc định.

---

## 1. Dữ Liệu Keyframe & Nền Tảng (Base Data)

* **[Custom Keyframes](file:///l:/Competitions/2026-AIC/Data/batch%201/Custom_Keyframes)**
  * **Mô tả:** Thư mục chứa toàn bộ ~358,639 frames đã cắt và lọc chất lượng (Laplacian blur + duplicate filtering).
  * **Đường dẫn:** `Data/batch 1/Custom_Keyframes/{video_id}/{frame_id:06d}.jpg` (Ví dụ: `L23_V007/003349.jpg`).
  * **Mục đích:** Nguồn ảnh gốc để Backend stream hiển thị trên Web Frontend và Lightbox.

* **[K1R Map Folder](file:///l:/Competitions/2026-AIC/Data/batch%201/map-keyframes-k1r)**
  * **Mô tả:** Thư mục chứa 873 file CSV thống kê danh sách 297,332 keyframes tinh hoa được giữ lại sau đợt lọc thứ 2 (triệt tiêu 61,307 frame rác/nhòe/ticker).
  * **Định dạng CSV:** `n, pts_time, fps, frame_idx`.

* **[Video Gốc MP4](file:///l:/Competitions/2026-AIC/Data/batch%201/video)**
  * **Mô tả:** Thư mục chứa toàn bộ 873 file video gốc `.mp4` (từ `L21_V001.mp4` đến `L30_V096.mp4`).
  * **Đường dẫn:** `Data/batch 1/video/{video_id}.mp4` (Ví dụ: `L23_V007.mp4`).
  * **Mục đích:** Phục vụ streaming trực tiếp qua HTTP Range tới trình phát HTML5 Native Player trên Web.

* **[Media Info](file:///l:/Competitions/2026-AIC/Data/batch%201/media-info)**
  * **Mô tả:** Chứa thông tin metadata của video gốc (FPS, duration, resolution).

* **[Video Taxonomy](file:///l:/Competitions/2026-AIC/Data/video_categories.json)**
  * **Số records:** 1.487 video, khóa ngoài logic là `video_id`.
  * **Schema mỗi record:** `{"category": "<slug>", "sub_category": "<slug hoặc rỗng>"}`.
  * **Danh mục và số video hiện tại:** `am_thuc` 498, `tin_tuc` 460, `day_hoc` 88, `du_lich_van_hoa` 63, `mua_lan` 43, `dua_xe_dap` 37, `camera_giao_thong` 298.
  * **Subcategory:** Chỉ `day_hoc` sử dụng: `toan` 10, `ly` 10, `hoa` 11, `van` 10, `su` 10, `dia` 10, `sinh` 10, `gdcd` 7, `tieng_anh` 10.
  * **Ngữ nghĩa lọc:** Nhiều `category` kết hợp theo OR. `sub_category` chỉ thu hẹp nhánh `day_hoc`, không ảnh hưởng các category được chọn cùng lúc.
  * **Nguồn đối chiếu:** Metadata gốc trong `Data/batch 1/media-info/`; endpoint công khai taxonomy và count là `GET /api/categories`.

---

## 2. Hệ Thống Tìm Kiếm Đặc Trưng Vector (LanceDB & Qwen3-VL Multimodal Index)

* **[LanceDB Table 'keyframes_k1r' (CHÍNH - TỐC ĐỘ CAO - EMBEDDING MỚI)](file:///l:/Competitions/2026-AIC/Data/aic_lancedb)**
  * **Số entries:** 476.140 records (297.332 batch 1 + 178.808 batch 2).
  * **Vector:** Nâng cấp từ tập `Data/qwen3vl_img/*.npy` (873 video) với độ tương phản cao, giải quyết triệt để lỗi representation collapse của bản cũ. L2-normalized = 1.0000.
  * **Đặc trưng tích hợp:** Qwen3-VL 2048D, ASR mapping ở các video đã có ASR. Cột `boxes_json` và `obj_scores` đã bỏ khỏi schema active.
  * **Mục đích:** Cơ sở dữ liệu chính thức phục vụ thi đấu Vòng Chung Kết với độ trễ thấp và độ tương phản tìm kiếm vượt trội.

* **[Qwen3-VL Index K1R (NPY - MỚI)](file:///l:/Competitions/2026-AIC/Data/usearch_qwen_index_k1r.npy)**
  * **Model & Nguồn:** Ghép nối chuẩn hóa từ 873 video `Data/qwen3vl_img/*.npy` (được sinh bằng `others/kaggle_embed_qwen3vl_k1r.ipynb` với chuẩn Qwen3-VL-Embedding-2B).
  * **Shape:** `(476.140 × 2048)` — dtype `float32` (~3,90 GB thập phân).
  * **Chuẩn hóa:** **Đã chuẩn hóa L2-Normalization 100% (Unit Vectors $\|v\|_2 = 1.00000$)**.
  * **Độ tương phản:** Phân tán đều trên mặt cầu 2048D (khoảng cách trung vị ngẫu nhiên ~0.24, điểm truy vấn đạt >0.50).
  * **File backup:** `Data/usearch_qwen_index_k1r_backup_oldqwen.npy` (bản K1R cũ) và `Data/usearch_qwen_index.npy` (bản gốc 358k frames) được giữ an toàn tuyệt đối.

* **[Thư Mục Embedding Gốc Theo Từng Video (NPY float16)](file:///l:/Competitions/2026-AIC/Data/qwen3vl_img)**
  * **Số files:** 873 file `.npy` tương ứng 873 video (L21_V001 đến L30_V096).
  * **Dtype:** `float16` — Tổng dung lượng thư mục chỉ **~1.13 GB** (giảm 50% so với float32).

* **[Qwen3-VL Mapping K1R (JSON)](file:///l:/Competitions/2026-AIC/Data/usearch_qwen_index_mapping_k1r.json)**
  * **Số entries:** 476.140, cùng thứ tự với NPY và database. Bản trước tích hợp nằm ở `Data/batch 2/manifests/pre_integration/`.

* **[RT-DETR Index K1R (NPY)](file:///l:/Competitions/2026-AIC/Data/usearch_rtdetr_index_k1r.npy)**
  * **Shape:** `(297,332 × 80)` — dtype `float16` (~45.4 MB). File gốc `usearch_rtdetr_index.npy` (57.38 MB) được giữ nguyên làm backup.

---

## 3. Hệ Thống Lời Thoại Âm Thanh (Faster-Whisper Large-V3 ASR)

* **[ASR Mapping File](file:///l:/Competitions/2026-AIC/Data/asr_mapping.json)**
  * **Model sinh ra:** `Faster-Whisper Large-v3` (từ `Data/batch 1/asr/*.json` gồm 873 file).
  * **Số đoạn thoại:** 16,609 segments câu thoại chuẩn ngữ nghĩa (1,362,943 từ, liên kết tới 330,695 keyframes, chiếm 92.2% CSDL).
  * **Dung lượng:** 19.64 MB.
  * **Mục đích:** Cho phép tìm kiếm toàn văn theo lời thoại nhân vật, lọc trùng lặp theo phân đoạn thoại ASR (`seg_id`), và hiển thị transcript chi tiết trong Lightbox & Timeline.

* **[Video FPS Mapping File](file:///l:/Competitions/2026-AIC/Data/video_fps_mapping.json)**
  * **Mô tả:** Chứa FPS của 1.487 video; batch 2 lấy từ CSV đi kèm embedding hoặc probe video khi map active rỗng.
  * **Mục đích:** Đảm bảo mốc thời gian $t = \text{frame\_idx} / \text{FPS}$ chuẩn xác 100% khi tua video, nộp mili-giây lên DRES và khớp nối ASR.

---

## 4. Hệ Thống Phân Cảnh Video (Video Shots 2048D)

* **[LanceDB Table 'video_shots'](file:///l:/Competitions/2026-AIC/Data/aic_lancedb)**
  * **Số records:** 186.102 shots (111.601 batch 1 + 74.501 batch 2 S/N).
  * **Đặc trưng:** Vector chuyển động cảnh Qwen3-VL Video 2048D, mốc thời gian `start_sec`, `end_sec`, `duration_sec`, file video cắt con `cut_file`.
  * **Mapping O(1):** `kf_shot_map.json` và `shot_metadata_map.json` liên kết 2 chiều giữa Keyframe và Video Shot.
  * **Phạm vi sử dụng:** Đây là nguồn tín hiệu retrieval/RRF nội bộ. Frontend không còn chế độ phát các clip shot; Top-K, TRAKE và lightbox chỉ trình chiếu keyframe tĩnh, còn nút `Video` mở video gốc tại đúng mốc thời gian.

---

## 5. Hệ Thống Phát Hiện Vật Thể (RT-DETR Object Detection)

**Lưu ý:** Đây là dữ liệu lịch sử batch 1, không còn là dependency runtime. Engine không nạp NPY/metadata, API object trả rỗng, frontend ẩn panel/boxes; các file nguồn bên dưới được giữ để truy vết.

* **[RT-DETR Multi-hot Index (NPY)](file:///l:/Competitions/2026-AIC/Data/usearch_rtdetr_index_k1r.npy)**
  * **Shape:** `(297,332 × 80)` — dtype `float16`.
  * **Mô tả:** Confidence score cao nhất của 80 lớp đối tượng COCO Dataset cho từng frame.
  * **Mục đích:** Bộ lọc Pre-filter lọc cứng hoặc kết hợp tính điểm với Visual score.

* **[RT-DETR Metadata (JSON)](file:///l:/Competitions/2026-AIC/Data/rtdetr_metadata.json)**
  * **Kích thước:** ~420 MB (Nạp Lazy load khi mở Lightbox).
  * **Mô tả:** Danh sách Bounding Box `[x, y, w, h]`, confidence và class name cho từng keyframe.

---

## 6. Hệ Thống Nộp Bài Trực Tuyến DRES (Vòng Chung Kết)

* **[dres_client.py](file:///l:/Competitions/2026-AIC/code/local_retrieval/dres_client.py)**
  * **Chuẩn giao tiếp:** REST API v2 theo chuẩn quốc tế VBS (Video Browser Showdown) trên máy chủ DRES (`https://eventretrieval.one`).
  * **Bảo mật:** `DRES_SERVER_URL`, `DRES_USERNAME`, `DRES_PASSWORD` nằm trong `.env` đã được `.gitignore`; password không được trả về frontend hoặc lưu trong `localStorage`.
  * **Run ID / Evaluation ID:** Người dùng nhập thủ công trên Web; frontend lưu khóa `aic_2026_dres_run_id` trong `localStorage`, sau đó gọi `POST /api/dres/set-evaluation` để đồng bộ backend.
  * **Quy đổi thời gian:** Tự động quy đổi `frame_id` sang mili-giây ($ms$) theo FPS thực tế của từng video.
  * **Cơ chế chống nộp trùng:** Bộ nhớ đệm chỉ chặn request mà **backend local hiện tại** đã gửi thành công trong câu hiện tại. Nó không biết submission từ máy/đồng đội khác và được reset khi bắt đầu task mới.
  * **Ngữ nghĩa trạng thái:** `SUBMITTED`/HTTP success chỉ xác nhận máy chủ đã nhận request. Chỉ xem là đúng/sai khi payload có verdict rõ như `CORRECT`, `ACCEPTED`, `WRONG`, `REJECTED` hoặc `INCORRECT`.
  * **Hỗ trợ 3 thể thức:**
    * KIS (Textual & Video): `mediaItemName`, `start`, `end` (đơn vị ms).
    * Q&A: `QA-<ANSWER>-<VIDEO_ID>-<TIME(ms)>`.
    * TRAKE: `TR-<VIDEO_ID>-<FRAME_ID1>,<FRAME_ID2>,...`.

---

## 7. Dữ Liệu Runtime và State Cục Bộ

| Thành phần | Vị trí / khóa | Phạm vi và lưu ý |
|---|---|---|
| Secret runtime | `.env` | DRES/Gemini; không commit. Mẫu cấu hình nằm ở `.env.example`. |
| Run ID DRES | `localStorage.aic_2026_dres_run_id` | Theo browser/profile local; không tự đồng bộ sang máy hoặc tài khoản khác. |
| Control panel | `localStorage.aic_2026_control_panel_state_v3` | Trọng số, Top-K, taxonomy và trạng thái collapse của panel. |
| RRF signals | `aic_rrf_visual`, `aic_rrf_asr`, `aic_rrf_shot` | Lựa chọn tín hiệu RRF hiện hành. |
| View mode | `aic_view_mode` | `shot` hoặc `keyframe`. |
| FPS frontend cache | `_videoFpsCache` trong RAM | Nạp lười qua `GET /api/video-fps/{video_id}`; mất khi reload. |
| KF resolver cache | `_video_keyframe_ids_cache`, `_actual_video_fps_cache` trong RAM backend | Nạp lười theo video; đối chiếu KF nhập tay với K1R map và FPS MP4 qua `GET /api/keyframe-resolve/{video_id}`. |
| DRES dedup/history | RAM của `DresClient` | Theo process backend và task hiện tại; không phải dữ liệu liên máy. |
| Readiness | RAM backend + `GET /api/readiness` | `starting/warming/ready/error`, progress, stage, elapsed và error; không phải health verdict của DRES. |
| Search mode | `currentAppMode` trong frontend | `KIS` hoặc `TRAKE`; chỉ chọn thuật toán/giao diện retrieval. |
| Submission mode | `#dres-task-mode-select` trong frontend | `kis`, `qa` hoặc `trake`; chỉ quyết định hành vi nộp DRES, độc lập với search mode. |
| DRES offline dry-run | RAM/DOM frontend | Cho phép test đầy đủ thao tác chọn và nộp KIS/QA/TRAKE khi chưa đăng nhập; không gọi API, không ghi history/dedup và không tạo trạng thái “đã nộp”. |

### Quy ước TRAKE Draft

- Input hợp lệ: `E1: ... E2: ...` hoặc từ hai đoạn văn trở lên, ngăn cách bằng ít nhất một dòng trống.
- Parser tạo mảng scene gồm `scene_idx`, `name`, `visual_query`, `ocr_query`, `speech_query`, `object_filter`.
- Enter lần đầu chỉ tạo/chỉnh draft E1..EN; Enter lần hai mới chạy TRAKE search. `Shift+Enter` xuống dòng.
- Draft có thể bắt đầu từ ô Gemini Parser hoặc Visual Query. Truy vấn Visual một cảnh vẫn dùng Enter để search bình thường.
- Khi chuyển KIS → TRAKE, frontend render lại tabs và editor từ `latestParsedScenes`; draft không phụ thuộc submission mode DRES.

---

## 8. Cấu Trúc Mã Nguồn (Source Code Architecture)

```
code/
├── local_retrieval/                ← Core Python Logic (Chạy Local)
│   ├── config.py                   ← Cấu hình tập trung cho dataset K1R production
│   ├── text_encoder.py             ← Qwen3-VL Text & Composed Multimodal Encoder
│   ├── asr_engine.py               ← ASR Speech Hybrid Search (Semantic + TF-IDF)
│   ├── search_engine.py            ← Search Engine Đa Phương Thức (LanceDB keyframes_k1r)
│   ├── reranker_engine.py          ← Two-Stage Video-Level N-Search Reranker
│   ├── qwen_reranker.py            ← Qwen-VL GPU CUDA Reranker (Pointwise & Sequence)
│   ├── dres_client.py              ← DRES Live Contest Client (Vòng Chung Kết)
│   ├── build_k1r_compact_dataset.py← Script tạo lát cắt K1R & bảng keyframes_k1r
│   └── gemini_parser.py            ← Bộ phân tích đề bài thi tự động bằng Gemini AI
│
├── web/                            ← Giao Diện Người Dùng & Backend API
│   ├── backend/
│   │   └── main.py                 ← FastAPI (background warm-up/readiness, Search, Timeline, FPS, DRES APIs)
│   └── frontend/
│       ├── index.html              ← Giao diện Web Dark Slate Pro + DRES Live Bar
│       ├── style.css               ← Linear / Raycast minimalist styling
│       └── app.js                  ← Client controller, local TRAKE parser, taxonomy, Video Player, DRES
│
└── slurm/scripts/                  ← Slurm Compute Cluster Pipelines
    ├── python/                     ← Trích xuất Qwen3-VL, Faster-Whisper, RT-DETR
    └── bash/                       ← Job submission scripts (Part 1/2/3, Forward/Reverse)
```

---

## 9. Bảng Tra Cứu Trọng Số & Ngưỡng Mặc Định

| Tham số | Giá trị mặc định | Giải thích |
|---|---|---|
| `weight_visual` | `1.0` | Trọng số UI mặc định của hình ảnh Qwen3-VL trong Weighted mode |
| `weight_ocr` | `1.0` (UI ẩn) | Kênh tương thích; UI chung kết đang ẩn và vô hiệu hóa OCR |
| `weight_asr` | `1.0` | Trọng số UI mặc định của lời thoại trong Weighted mode |
| `weight_object` | `0` | Legacy API; RT-DETR đã vô hiệu hóa |
| `rrf_k` | `60` | Hằng số làm mượt khi bật RRF mode |
| `multi_channel_bonus` | không dùng | RRF trung bình theo kênh khả dụng, không cộng bonus |
| `visual_sim_thresh` | `0.90` | Ngưỡng Cosine Similarity để Smart Dedup gộp frame trùng |
| `top_k` ($K_1$) | `100` | Số lượng candidate frames quét ở vòng 1 |
| `top_v` ($N$) | `5` | Số lượng video có điểm cao nhất được chọn để Rerank |
| `k_per_video` ($M$) | `40` | Số lượng frames tối đa rút ra từ mỗi video trong Top $N$ |

### Quan hệ tham số TRAKE

- `top_k`: Candidate Pool Stage 1 cho mỗi event.
- `top_v`: giới hạn cứng số video được quét sâu; chi phí refined search là `N_events × top_v`.
- `k_paths`: số chuỗi tối đa sinh từ mỗi video.
- `trake_top_k`: số chuỗi trả về, được clamp tối đa ở `top_v × k_paths`; không được dùng để tự tăng `top_v`.
