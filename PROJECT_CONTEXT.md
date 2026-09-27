# 📖 PROJECT CONTEXT — AIC 2026 | Hệ thống Truy vấn Video Đa Phương Thức

> **Mục đích tài liệu này:** Đọc vào là biết ngay ngữ cảnh toàn bộ dự án, kiến trúc thuật toán, cơ chế Reranking, Multi-Scene, Timeline và giao diện Web mà không cần hỏi lại từ đầu.  
> **Cập nhật lần cuối:** 2026-09-26 (batch 2 đã tích hợp)
>
> **Trạng thái hiện tại:** Luồng thi đấu ưu tiên tốc độ thao tác và xác thực thủ công: Visual/ASR/Video Shot retrieval, RRF hoặc Weighted Fusion, TRAKE, bộ lọc danh mục, Video Player/Timeline và nộp DRES. OCR và các UI reranker nặng đang **ẩn mặc định**; backend tương thích vẫn được giữ để có thể bật lại sau.

---

## 1. MỤC TIÊU CUỐI CÙNG

**Batch 2 đã tích hợp (2026-09-26):** 614 video M/S/N đã chuẩn hóa vào `Data/batch 2`; 3.409 vector thiếu JPG bị loại có manifest truy vết. Production chung có **1.487 video, 476.140 keyframe và 186.102 shot** trong ba bảng LanceDB (`keyframes_k1r`, `video_shots`, `asr_segments`); riêng batch 2 đóng góp 178.808 keyframe và 74.501 shot S/N. 11.511 cửa sổ shot nguồn không có keyframe hợp lệ trong khoảng nên không nhập. ASR batch 2 và shot nhóm M **vẫn đang trích xuất**, chưa được tạo dữ liệu giả. Xem [PLAN_batch2_integration.md](others/PLAN_batch2_integration.md) và `Data/batch 2/manifests/integration_report.json`.

**Runtime mới:** resolver phục vụ ảnh/video M/S/N (.mp4 và .mov), thời gian KF batch 2 lấy từ `pts_time` map embedding; taxonomy thêm `camera_giao_thong` (298 video), M thêm vào `tin_tuc` (304), S vào `dua_xe_dap` (12). RT-DETR đã ngừng load/sử dụng và ẩn UI; schema keyframe active đã bỏ `obj_scores`/`boxes_json`. Các đoạn mô tả RT-DETR ở phần lịch sử pipeline bên dưới không còn phản ánh runtime hiện tại.

**Caption RRF:** 51.650/52.720 caption nguồn được đối chiếu shot/time và nối tới keyframe production; 1.070 không có keyframe/shot phù hợp bị loại. Công tắc Caption mặc định tắt; khi bật, TF-IDF caption có thể đưa ứng viên mới vào retrieval. RRF dùng trung bình theo kênh *có dữ liệu* ở từng candidate, không dùng bonus 1,25; caption/ASR/shot thiếu khác với có dữ liệu nhưng không hit. Cần benchmark chất lượng theo từng nhóm coverage trước khi bật Caption mặc định.

Xây dựng hệ thống **Truy vấn Video Đa Phương Thức (Multimodal Video Retrieval)** hoàn chỉnh cho cuộc thi AIC 2026:
- Người dùng nhập đề bài thô (Tiếng Việt hoặc Tiếng Anh) mô tả phân cảnh, hành động, màu sắc, chữ trên màn hình hoặc lời thoại nhân vật.
- Hệ thống phân tích, tìm kiếm và trả về danh sách **keyframe** khớp nhất kèm mốc thời gian và lời thoại ASR khi có.
- Tập production: 476.140 keyframe trên 1.487 video; độ trễ tùy truy vấn, bộ lọc và chế độ TRAKE.

```
Đề bài thô (Raw Prompt / Task Description)
    │
    ▼
[Gemini AI Parser] ──▶ Tự động bóc tách: Visual Query, OCR Text, ASR Speech, OD Class, Multi-Scene, QA Banner
    │
    ▼
[Stage 1: Multi-Modal Fast Search (Top K1)]
    ├─ Visual Embedding: Qwen3-VL-Embedding-2B (2048 dim, USearch Cosine Metric)
    ├─ Speech Search: Faster-Whisper Large-V3 + TF-IDF N-gram Index
    ├─ OCR-compatible query path: giữ ở backend, ẩn khỏi UI chung kết
    ├─ Caption lexical TF-IDF (tùy chọn; coverage 51.650 shot batch 1)
    ├─ Smart Dedup: Lọc trùng lặp ảnh + thời gian trong cùng video
    └─ Cross-Video Dedup: Triệt tiêu các đoạn intro / logo / chuyển cảnh trùng lặp qua nhiều video
    │
    ▼
[Stage 2: Temporal Multi-Scene / TRAKE (luồng chính) và Reranker (tùy chọn, UI ẩn)]
    ├─ 1. Gom điểm Top K1 frames để chọn Top N Video tiềm năng nhất.
    ├─ 2. Intra-Video Search chạy search(video_id=vid, top_k=M) cho từng video trong Top N.
    ├─ 3. Multi-Scene DP: Ràng buộc tuần tự thời gian t(Cảnh 1) < t(Cảnh 2) < ... < t(Cảnh N).
    └─ 4. Gộp toàn bộ frames sạch từ N video, sắp xếp theo Score (Bảo toàn 100% điểm số & thứ hạng).
```

---

## 2. DỮ LIỆU VÀO — CẤU TRÚC VIDEO GỐC

- **Nguồn:** Dataset AIC 2026 tải từ Kaggle, giải nén trên Slurm.
- **Định dạng:** File `.mp4` được tổ chức theo thư mục:
  ```
  Data/
  ├── Videos_L01/video/L01_V001.mp4, L01_V002.mp4...
  ├── Videos_L21_a/video/L21_V001.mp4...
  └── ...
  ```
- **Tổng cộng:** ~873 video, tổng ~358,639 keyframe sau khi cắt và lọc chất lượng.

---

## 3. GIAI ĐOẠN 1: CẮT KEYFRAME (extract_keyframes_slurm.py)

### 3.1 Chiến thuật cắt cảnh — 2 Lớp Bổ Sung Nhau
1. **Lớp 1 — PySceneDetect (ContentDetector):** Phát hiện hard cut giữa các cảnh dựa trên thay đổi nội dung pixel.
2. **Lớp 2 — Hàm toán học tự build (Chống bỏ sót cảnh dài):**
   ```python
   def get_num_keyframes(duration_sec: float) -> int:
       """N = 2 * sqrt(t) — Căn bậc hai chống bùng nổ dữ liệu"""
       return max(1, int(round(2 * math.sqrt(duration_sec))))
   ```
   Mỗi cảnh được cắt thêm $N$ keyframes phân bổ đều (`linspace`).

### 3.2 Cấu trúc Keyframe Output
```
Data/Custom_Keyframes/
├── L01_V001/
│   ├── 000150.jpg   ← Frame số 150 của video L01_V001
│   └── 003500.jpg
└── L23_V007/
    └── 003349.jpg
```
- **Mapping ngầm:** `L01_V001/001042.jpg` = frame 1042 của video L01_V001.
- **Timestamp:** $t = \frac{\text{frame\_id}}{\text{FPS}}$ (trích xuất chuẩn xác từ `Data/video_fps_mapping.json`).

---

## 4. GIAI ĐOẠN 2: LỌC KEYFRAME (filter_keyframes_slurm.py)

| Bộ lọc | Thuật toán | Ngưỡng |
|---|---|---|
| **Nhòe (Blur)** | Laplacian Energy của gradient (bỏ 15% đáy chứa subtitle) | `energy < 50` → loại |
| **Trùng lặp** | Cosine Similarity BEiT-3 + IoU đối tượng RT-DETR | `sim > 0.95 và IoU_obj > 0.8` → loại |

---

## 5. GIAI ĐOẠN 3: TRÍCH XUẤT ĐẶC TRƯNG QWEN3-VL (extract_qwen_slurm.py)

### 5.1 Model: Qwen/Qwen3-VL-Embedding-2B (arXiv 2601.04720)
- **Kiến trúc:** Visual Tokens (1536 dim) → LLM Transformer (2048 dim) → **Last Token Pooling (EOS Token)**.
- **Số chiều vector:** `2048` chiều (dtype `float32`).
- **Thư viện:** `sentence-transformers[image]` với instruct prefix chuẩn tác giả.

### 5.2 Chiến thuật 3-Part Song Song trên Slurm
- 358,639 ảnh chia 3 Part (~119,546 ảnh/Part) chạy song song trên 3 GPU.
- Checkpoint mỗi 1000 ảnh → `merge_indices.py` gộp thành `usearch_qwen_index.npy` (~2.74 GB) và `usearch_qwen_index_mapping.json`.

---

## 6. GIAI ĐOẠN 4: TRÍCH XUẤT LỜI THOẠI ASR & VIDEO FPS MAPPING
- **ASR Engine 2 Lớp (Faster-Whisper Large-V3 + TF-IDF N-gram Vectorizer):**
  - **Tập dữ liệu:** 16,609 segments tiếng Việt chất lượng cao từ Faster-Whisper Large-V3.
  - **Thuật toán Tìm kiếm Văn bản ASR:** Dùng `TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, norm='l2')`. Tự động triệt tiêu stop words/từ phổ biến và khuếch đại các thực thể hiếm (ví dụ: *"hổ"*, *"bạch tuộc"*, *"thịt nạc xay"*...). Điểm số chuẩn hóa chuẩn xác trên $[0.0, 1.0]$.
  - **Cơ chế Tra cứu 2 Lớp:**
    - Lớp 1: Bảng băm trực tiếp $O(1)$ từ Keyframe sang Speech Context.
    - Lớp 2: Tra cứu động theo mốc thời gian thực $t = \frac{\text{frame\_id}}{\text{FPS}}$ và so khớp khoảng $[t_{\text{start}}, t_{\text{end}}]$ của video để không bao giờ bị sót ASR.
  - **Video FPS Mapping:** Đã trích xuất chính xác 100% FPS cho toàn bộ 873 video vào `Data/video_fps_mapping.json` (25.0, 26.44, 29.97, 30.0...) đảm bảo ánh xạ thời gian $t = \text{frame\_idx} / \text{FPS}$ chuẩn xác 100%.

---

## 7. GIAI ĐOẠN 5: TRÍCH XUẤT VẬT THỂ RT-DETR (extract_rtdetr_slurm.py)
- **Index:** `Data/usearch_rtdetr_index.npy` (358,639 × 80 classes, Multi-hot float16).
- **Metadata Bounding Boxes:** `Data/rtdetr_metadata.json` (420 MB) lưu tọa độ x, y, w, h và class name cho từng frame.

---

## 8. KIẾN TRÚC THUẬT TOÁN LOCAL RETRIEVAL (`code/local_retrieval/`)

### 8.1 Bộ Phân Tích Đề Bài Gemini AI (`gemini_parser.py`)
- **Mô hình hỗ trợ:** Ưu tiên sử dụng `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`, `gemini-3.5-flash` trên Google AI Studio.
- **API Key:** Không hardcode trong source. Nạp từ `GEMINI_API_KEY` trong `.env`; ô nhập trên giao diện chỉ là phương án ghi đè cục bộ.
- **Cơ chế Ngắt Lỗi & Dự Phòng Thông Minh (Smart Fallback):**
  - Timeout: 10s cho mỗi lượt gọi mạng.
  - Tự động chuyển ngay sang `_fallback_smart_parser(raw_text)` nội bộ nếu không có mạng hoặc gặp lỗi xác thực.
- **Quy Tắc Chuyên Biệt Hóa 4 Kênh (Multimodal Specialization):**
  - 🎬 **`visual_query` (Bảo tồn Bối cảnh & Không gian):** 
    - Bắt buộc giữ lại không gian, bối cảnh và hoạt cảnh chung (*"Người phụ nữ dạy nấu ăn cho những người khác"*, *"Không gian lễ hội ẩm thực"*).
    - **Tuyệt đối KHÔNG nhồi nhét chữ viết, số đo, định lượng chi tiết** (*200g*, *500ml*, *số 42*) vào `visual_query` vì CLIP/SigLIP không đọc được chữ nhỏ và sẽ làm nhiễu vector thị giác.
    - Thay vào đó chỉ mô tả dưới dạng thực thể trực quan (*"tờ giấy hoặc bảng công thức món ăn"*).
  - 🔤 **`ocr_query` (Nhận diện chữ & số đo):** Chuyên trách nhận toàn bộ văn bản, tên biển hiệu, công thức, con số đo lường (*"200g thịt nạc xay"*).
  - 🎙️ **`speech_query` (Lời thoại & Thuyết minh):** Trích xuất lời nói trực tiếp hoặc chủ đề phóng sự/sự kiện (*"người phụ nữ dạy nấu ăn"*, *"lễ hội ẩm thực Nhật Bản"*).
  - 📦 **`object_filter`:** Gợi ý ID và tên lớp trong 80 lớp COCO của RT-DETR.
  - 🎯 **`reranker_query`:** Lưu trữ toàn bộ câu mô tả đầy đủ để Reranker đối chiếu sâu.
  - ❓ **`qa_question`:** Tách câu hỏi truy vấn đáp án nếu là đề thi QA.

### 8.2 Chiến Thuật Dung Hợp Đa Phương Thức (Multi-Modal Fusion)
Hệ thống hỗ trợ hai mode độc lập và cho phép đổi trực tiếp trên UI:

1. **Reciprocal Rank Fusion (RRF):**
   $$\text{RRF}(d) = \sum_{m \in \{\text{vis}, \text{ocr}, \text{asr}, \text{obj}\}} \frac{1}{k + \text{Rank}_m(d)}, \quad \text{Final Score}(d) = \frac{\text{RRF}(d)}{\max(\text{RRF}_{\text{batch}})}$$
   - Mặc định `k = 60`; trạng thái các tín hiệu Visual/ASR/Video Shot được lưu ở `localStorage`.
   - Video Shot chỉ còn là **tín hiệu retrieval** tham gia xếp hạng. UI Top-K và TRAKE luôn hiển thị keyframe tĩnh; chế độ trình chiếu/hover clip shot và player shot trong lightbox đã được loại bỏ.
2. **Weighted Fusion (Dung hợp trọng số):**
   $$\text{Final Score} = \frac{w_v \cdot S_{\text{vis}} + w_o \cdot S_{\text{ocr}} + w_a \cdot S_{\text{asr}} + w_{\text{obj}} \cdot S_{\text{obj}}}{w_v + w_o + w_a + w_{\text{obj}}}$$
3. **Cross-Video Deduplication (Lọc trùng lặp qua nhiều video):**
   - Triệt tiêu các keyframe dạng intro/logo/chuyển cảnh đài truyền hình xuất hiện lặp đi lặp lại ở nhiều video khác nhau khi câu truy vấn có độ tương đồng với đoạn intro.
   - Ngưỡng tương đồng Cosine Similarity tùy chỉnh linh hoạt từ UI ($0.80 \rightarrow 0.99$).

### 8.3 Thuật Toán Tìm Kiếm Đa Phân Cảnh (Temporal Multi-Scene Search)
- **Quy hoạch động tuần tự thời gian (`_find_chronological_sequence`):**
  - Đảm bảo chuỗi keyframe được chọn luôn thỏa mãn điều kiện nhân quả thời gian:
    $$t(\text{Scene 1}) < t(\text{Scene 2}) < \dots < t(\text{Scene N})$$
  - Thưởng thêm $+25\%$ điểm đồng thuận cho các video có đầy đủ các cảnh con xuất hiện đúng thứ tự thời gian.

### 8.4 Thuật Toán TRAKE Mode — TARS Monotonic DP (Temporal Reasoning & Keyframe Extraction)
Tham khảo nghiên cứu AIC2025 (Đạt độ chính xác 93.15%, hoàn toàn Training-Free):
1. **Công thức truy hồi Prefix-Maximum DP với khoảng phạt khoảng cách $\lambda$:**
   $$dp[s, j] = \text{Sim}[s, j] + \max_{k < j}\bigl(dp[s-1, k] - \lambda(t_j - t_k)\bigr)$$
   - Khử độ phức tạp từ $O(n m^2)$ xuống $O(nm)$ bằng cách duy trì giá trị Prefix-Max theo thời gian.
   - $\lambda \in [0.0001, 0.01]$ điều chỉnh linh hoạt từ UI, phạt các bước nhảy thời gian bất thường hoặc quá xa.
2. **Alpha-Fusion Score (Kết hợp điểm ngữ cảnh toàn cục và điểm DP):**
   $$\text{FinalScore} = \alpha \cdot \text{GlobalScore} + (1 - \alpha) \cdot \text{DPScore} \quad (\alpha = 0.35)$$
3. **Diverse Beam Search ($K\_\text{paths}$):**
   - Với mỗi video ứng viên, trích xuất $K\_\text{paths}$ chuỗi thời gian tối ưu khác nhau (không chỉ 1 chuỗi duy nhất) để người dùng có nhiều phương án lựa chọn nộp bài.
4. **Format Dòng Nộp Bài Chuẩn BTC AIC:**
   `<video_id>,<F_E1>,<F_E2>,<F_E3>,<F_E4>` (Ví dụ: `L26_V194,4700,5130,5412,5850`, frame ID dạng số nguyên không padding số 0).
5. **Tối ưu hóa Precomputed Query Vectors:**
   - Vector nhúng Qwen3-VL được mã hóa 1 lần duy nhất cho mỗi sự kiện con $E_1 \dots E_N$ và tái sử dụng cho tất cả các lượt quét intra-video, tránh forward pass lặp lại. Tổng độ trễ vẫn phụ thuộc chủ yếu vào `N_events × Top Videos` và các tín hiệu RRF đang bật.
6. **Giới hạn chi phí theo Top Videos:**
   - `Top Videos (V)` là giới hạn cứng của pha quét sâu; số lượt refined search bằng `số event × V`.
   - `Top Sequences` chỉ giới hạn số chuỗi trả ra, không được tự nâng `V`. Sức chứa tối đa là `V × K-paths`; UI/backend sẽ clamp và cảnh báo nếu yêu cầu vượt mức này.

---

## 9. GIAO DIỆN WEB LINEAR / RAYCAST PRO MINIMALIST (`code/web/`)

**Profile thi đấu hiện hành:** tối giản số nút để ưu tiên tốc độ. `OCR_FEATURE_VISIBLE = false` và `RERANK_FEATURE_VISIBLE = false`; các endpoint/module vẫn tồn tại nhưng không nằm trong luồng mặc định.

### 9.1 Bố Cục Gom Cụm 1 Phía & Mode Switcher
1. **Mode Switcher Bar (Đầu Sidebar):**
   - Nút `[🔍 SEARCH KIS]`: Chế độ tìm kiếm đơn cảnh và đa phân cảnh thông thường.
   - Nút `[⏱️ SEARCH TRAKE]`: Chế độ chuyên biệt cho bài toán chuỗi sự kiện tuần tự TRAKE.
   - Đây chỉ là **search mode**, độc lập với ba submission mode KIS/QA/TRAKE trên DRES Live Bar. Ví dụ có thể Search TRAKE rồi nộp KIS, hoặc Search KIS rồi chọn chuỗi để nộp TRAKE.
2. **Sidebar Trái Cố Định (380px):** Gom Gemini Parser, Visual, bộ lọc danh mục, ASR, Object, Smart Dedup, Candidate Pool và TRAKE Params. Bộ lọc danh mục nằm ngay dưới Visual Query.
3. **Module 5: Smart & Cross-Video Deduplication:**
   - *Intra-Video Dedup:* Ngưỡng tương đồng ảnh nội bộ video (mặc định 0.90), chế độ tự động (Visual + ASR) hoặc 1 Frame duy nhất / Video.
   - *Cross-Video Dedup:* Công tắc và thanh trượt khử trùng liên video (mặc định 0.90) nhằm triệt tiêu các cảnh logo, nhạc hiệu mở đầu, quảng cáo lặp lại giữa các video.
4. **Module 7: TRAKE Parameters Panel:**
   - Candidate Pool ($K_1$, mặc định 100), Top Videos ($V$, mặc định 10), Frames/Video ($M$, mặc định 40), Lambda ($\lambda$, mặc định 0.001), K-Paths (mặc định 3), Top Sequences (mặc định 10).
   - Ước lượng số search quét sâu: `N_events × Top Videos`; giảm Candidate Pool không làm giảm số lượt này.
5. **Sticky Action Dock:**
   - *KIS Mode:* `[ 🚀 SEARCH NOW ]`, `[ 🎬 MULTI-SCENE ]`; reranker được ẩn.
   - *TRAKE Mode:* `[ 🚀 TÌM 1 CẢNH ]`, `[ ⏱️ SEARCH TRAKE ]`; Qwen reranker được ẩn.

### 9.2 TRAKE Storyboard Matrix Workspace
- **Ma trận dòng sự kiện tuần tự:** Mỗi hàng đại diện cho 1 chuỗi ứng viên tối ưu từ một Video ID với đầy đủ các thẻ $E_1, E_2, E_3, E_4$.
- **Mũi tên chênh lệch thời gian ($\Delta t$):** Hiển thị rõ bước nhảy thời gian giữa 2 sự kiện liên tiếp (ví dụ: `➔ +15.2s`).
- **Nút Copy TRAKE Line:** 1-click sao chép trực tiếp chuỗi định dạng CSV nộp bài (`video_id,F1,F2,F3,F4`).
- **Nút Xuất CSV:** Tải trực tiếp file `.csv` chứa toàn bộ Top K chuỗi ứng viên để nộp bài lên hệ thống BTC.

### 9.3 Trải Nghiệm Xác Thực Hình Ảnh Pro (Verification Workspace)
- **Precision Keyframe Cards:** Lưới kết quả tối giản với thumbnail 16:9 sắc nét, điểm số tương phản cao, nút `📋 Copy ID` và `▶️ Video`.
- **Interactive Lightbox Inspector (Pan & Zoom tại tâm con trỏ chuột):**
  - **Lăn chuột Zoom tại điểm trỏ (Wheel Zoom at Cursor):** Chỉ chuột vào vị trí nào và lăn bánh xe chuột để phóng to/thu nhỏ ngay tại vị trí con trỏ chuột ($100\% \rightarrow 1000\%$).
  - **Drag to Pan:** Kéo chuột di chuyển khung hình khi đang zoom để quan sát mọi chi tiết nhỏ.
  - **Double Click / Nút Reset:** Đưa ảnh về kích thước chuẩn. Bounding Box RT-DETR bám dính chuẩn xác theo tỷ lệ zoom.
- **Bảng Lưới Timeline Keyframe (Vertical Scrollable Grid):**
  - Bảng lưới đa cột `minmax(160px, 1fr)` cuộn dọc duyệt qua toàn bộ keyframe của video từ 00:00 đến hết.
  - Khóa cứng tỷ lệ thumbnail 16:9, hiển thị mốc thời gian và mã frame.
  - Tự động cuộn mượt mà (`scrollIntoView`) và đánh dấu viền đỏ dạ quang (`timeline-item-active`) tại keyframe mục tiêu.
- **Phân Tầng Z-Index Chuẩn 3 Lớp:**
  - `Timeline Modal`: Tầng dưới (`z-index: 10000`).
  - `Lightbox Inspector`: Tầng trên (`z-index: 20000`). Bấm vào keyframe trong Timeline sẽ mở Lightbox đè lên trên; tắt Lightbox lập tức quay lại Timeline tại đúng vị trí cuộn.
  - `Video Player Streaming`: Tầng cao nhất (`z-index: 30000`).

### 9.4 Qwen3-VL-Reranker GPU (tùy chọn, hiện ẩn khỏi UI)

Phần này là năng lực dự phòng, không thuộc cấu hình thi đấu mặc định vì độ trễ cao.
- **Chống Trôi Lời Thoại (ASR Temporal Drift Fix):**
  - Mở rộng $\pm 1$ phân đoạn lời thoại liền kề trong cùng video (`get_speech_for_keyframe_expanded`).
  - Giai đoạn Recall lan tỏa 85% điểm số sang keyframe lân cận để đảm bảo frame đúng không bị rớt khỏi Candidate Pool khi ASR xuất hiện lệch thời gian so với hình ảnh.
- **Tùy Chọn Top X AI Rerank:**
  - Thanh chọn linh hoạt `Top X: [ 20 | 30 | 50 | All ]` và ô nhập số tùy ý ngay phía trên nút AI Rerank.
  - Cho phép người dùng chỉ thẩm định 20 hay 30 frame đầu tiên để hoàn thành trong 3-5 giây, tiết kiệm thời gian GPU tối đa.
- **Cụm 3 Tab Chuyển Đổi View Kết Quả (Results View Switcher Tabs):**
  - `📋 Kết Quả Gốc`: Danh sách 100-200 frames theo thứ hạng tìm kiếm vector/RRF ban đầu.
  - `🤖 BXH AI Rerank [count]`: Tự động tái sắp xếp toàn bộ lưới ảnh theo **điểm số AI giảm dần**, đưa các frame khớp nhất lên vị trí `#1 AI`, `#2 AI` kèm huy hiệu cúp 🏆.
  - `🟢 Chỉ Frame Khớp (AI ≥ 0.50)`: Bộ lọc thông minh chỉ hiển thị các frame đạt chuẩn điểm AI cao, ẩn toàn bộ frame rác.
- **Kích Hoạt Thủ Công & Streaming:**
  - *KIS Mode:* Nút `[🤖 AI RERANK (Qwen-VL GPU)]` stream điểm số $P(\text{"yes"})$ từng frame về giao diện theo thời gian thực qua Server-Sent Events (SSE).
  - *Multi-Scene Mode:* Nút `[🤖 AI RERANK SEQ]` tổ hợp best frame từng phân cảnh thành chuỗi đa ảnh đưa vào Qwen-VL để đánh giá toàn chuỗi và sắp xếp lại video.
  - *TRAKE Mode:* Nút `[🤖 AI RERANK ALL SEQUENCES]` thẩm định tính tuần tự và tái xếp hạng lại toàn bộ Storyboard sequences.
  - *Chỉ báo màu sắc:* 🟢 Xanh lá ($\ge 0.80$, nộp/mở rộng ngay), 🟡 Vàng ($0.50 \le \text{score} < 0.80$), 🔴 Đỏ ($< 0.50$, bỏ qua).

### 9.5 Nhập TRAKE Không Cần Gemini

- Chấp nhận cả format `E1: ... E2: ...` và các đoạn văn ngăn cách bởi ít nhất một dòng trống.
- Có thể nhập tại **Gemini Parser** hoặc **Visual Query**.
- **Enter lần 1:** nhận diện, chuyển sang TRAKE, tạo E1..EN và mở editor; chưa chạy search.
- **Enter lần 2:** chạy `performTrakeSearch()`; **Shift+Enter** chỉ xuống dòng.
- Nếu Visual Query chỉ có một sự kiện, Enter vẫn chạy tìm kiếm đơn cảnh.
- Editor hỗ trợ chèn event trước/sau, xóa và `Alt+↑/↓` để đổi thứ tự.

### 9.6 Taxonomy Filter

- Nguồn chuẩn: `Data/video_categories.json`, đối chiếu metadata trong `Data/batch 1/media-info/`.
- Bảy danh mục độc lập: `am_thuc`, `tin_tuc`, `day_hoc`, `du_lich_van_hoa`, `mua_lan`, `dua_xe_dap`, `camera_giao_thong`.
- Chọn nhiều danh mục hoặc nhiều môn bằng **Ctrl+Click**. Các danh mục kết hợp theo OR; subcategory chỉ áp dụng cho nhánh `day_hoc`.
- Chỉ `day_hoc` có subcategory: Toán, Lý, Hóa, Văn, Sử, Địa, Sinh, GDCD, Tiếng Anh.

### 9.7 Readiness, Video Player và DRES

- Backend mở HTTP trước rồi warm-up retrieval ở background. UI khóa các nút search cho tới khi `/api/readiness` trả `state = ready`; trạng thái lỗi có thể retry.
- `Alt+G` hoặc nút **Mở / Nộp hộ** mở hộp hai tab: kiểm tra video/KF ngoài Top-K và nộp hộ đồng đội. KF nhập tay được đối chiếu với keyframe map; nếu không tồn tại, UI hiển thị KF gần nhất thay vì âm thầm dùng sai ID.
- Hộp mở nhanh nhận KF dạng số hoặc `F:<frame_id>`, thời gian dạng `T:<seconds>` hoặc `mm:ss`; tab nộp hộ hỗ trợ KIS, QA và chuỗi TRAKE phân cách bằng dấu phẩy.
- Video Player dùng FPS thực qua `/api/video-fps/{video_id}`, có live badge `mm:ss`, mili-giây và frame ID.
- Ctrl+Click lên video giữ frame chính xác tính từ `currentTime × FPS`, kể cả frame nằm giữa hai KF đã trích xuất; timeline chỉ chọn được KF có trong map nên có thể thưa hơn 0.5–1 giây. KIS nộp ngay; QA mở form đáp án; TRAKE thêm/bỏ event.
- Đổi submission mode chỉ thay hành vi Ctrl+Click/nút nộp; không tự chuyển search mode. Đổi search mode cũng không thay submission mode.
- DRES tự đăng nhập từ `.env`. Password không lưu ở frontend; Run ID do người dùng nhập được lưu ở `localStorage` và chỉ đổi khi người dùng thay thủ công.
- DRES offline không khóa search, mở video, timeline, chọn keyframe/event TRAKE, mở form QA hay thao tác nộp thử. Khi người dùng bấm nộp, frontend chạy **local dry-run** (phản hồi màu vàng ngắn hạn), chỉ cảnh báo cần đăng nhập và không gửi request/ghi nhận “đã nộp”.
- Phản hồi `SUBMITTED` chỉ xác nhận DRES đã nhận request, không đồng nghĩa đáp án đúng. Màu xanh/đỏ chỉ dùng khi payload có verdict rõ `CORRECT/ACCEPTED` hoặc `WRONG/REJECTED`; lỗi HTML/non-JSON được hiển thị như lỗi giao tiếp.

---

## 10. LỆNH CHẠY HỆ THỐNG DƯỚI LOCAL

```powershell
# Chạy Backend Server (FastAPI + Static Frontend + Video Streaming)
cd l:\Competitions\2026-AIC\code\local_retrieval
.\venv\Scripts\python.exe ..\web\backend\main.py

# Truy cập giao diện Web:
# http://localhost:8000
```

- Chỉ chạy **một** backend trên port 8000. `WinError 10048` nghĩa là port đã có một process khác chiếm dụng; thường là một instance backend đang chạy sẵn.
- Dòng `Application startup complete` chưa có nghĩa retrieval đã sẵn sàng; xem badge `WARMING UP/READY` trên UI hoặc `/api/readiness`.

---

## 11. BẢO HIỂM HỆ THỐNG VÀ SAO LƯU

- **Snapshot Ổn Định v1:** [`backups/snapshot_stable_v1_20260821/`](file:///l:/Competitions/2026-AIC/backups/snapshot_stable_v1_20260821)
- **1-Click Restore:** [`backups/RESTORE_STABLE_V1.bat`](file:///l:/Competitions/2026-AIC/backups/RESTORE_STABLE_V1.bat)
- **Git Checkpoint:** `Checkpoint v1: He thong KIS va TRAKE on dinh truoc khi tich hop Qwen Reranker` (100% Offline).
- Các artifact thử nghiệm đã dọn (`test_cuts_adaptive`, `vintern_*`, `eval_cache_*`, ảnh keyframe test, dump text...) không thuộc dependency runtime. `__pycache__` được Python tự sinh lại.
- Trước thay đổi cấu trúc lớn, ưu tiên tạo Git checkpoint hoặc backup có tên rõ ràng; không khôi phục đè toàn bộ worktree khi đang có thay đổi chưa commit.

---

## 12. BÀI HỌC KINH NGHIỆM & QUY TẮC PHÒNG TRÁNH LỖI (DEV RULES)

1. **Lập Trình Phòng Thủ DOM:** Luôn kiểm tra `null` trước khi thao tác DOM (`el?.innerText`, `setStatus(text)`).
2. **Quy Tắc Cache-Busting:** Mỗi khi cập nhật `style.css` hoặc `app.js`, **bắt buộc tăng version query** trong `index.html` (ví dụ: `v20260820_trake_v1`) và dùng `Ctrl + Shift + R` khi kiểm thử.
3. **Phân Lớp Z-Index Modal:** Modals lồng nhau (Timeline $\rightarrow$ Lightbox $\rightarrow$ Video) phải có khoảng cách Z-Index rõ ràng ($10000 \rightarrow 20000 \rightarrow 30000$) để không bị chìm xuống dưới.
4. **Phân Tách Kênh Visual vs OCR:** Không nhồi nhét số đo định lượng chi tiết (*200g*, *500ml*) vào `visual_query` (vì CLIP không đọc được text nhỏ), chuyển toàn bộ sang `ocr_query`.
5. **Khóa Tỷ Lệ Khung Ảnh 16:9:** Luôn áp dụng `aspect-ratio: 16 / 9; object-fit: cover;` cho container ảnh thumbnail để tránh hiện tượng co rút hoặc méo hình trong CSS Grid.
6. **Tối ưu hóa Embedding trong Đa Phân Cảnh:** Luôn tính trước `query_vec` một lần duy nhất cho mỗi phân cảnh/sự kiện trước khi lặp qua danh sách video ứng viên để triệt tiêu thời gian forward pass thừa.
7. **Bảo mật runtime:** Secret chỉ nằm trong `.env` đã được `.gitignore`; `.env.example` chỉ chứa placeholder. Không lưu password DRES vào `localStorage`, log hoặc source.
8. **State frontend không phải dữ liệu dùng chung:** Run ID, lựa chọn RRF, trạng thái panel và view mode nằm trong `localStorage` của trình duyệt/máy hiện tại. Khi đổi browser/máy phải nhập lại.
9. **Chống nộp trùng có phạm vi local:** Cache chỉ biết các request được backend hiện tại gửi trong câu hiện tại; không biết submission từ máy hoặc đồng đội khác.
10. **Tài liệu là handoff giữa các phiên/tài khoản:** Sau thay đổi lớn phải cập nhật `PROJECT_CONTEXT.md` và `code/DATA_DICTIONARY.md`; không phụ thuộc vào memory của một chat Codex.
