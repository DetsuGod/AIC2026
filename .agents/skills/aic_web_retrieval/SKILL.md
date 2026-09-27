---
name: aic_web_retrieval
description: Định hướng phong cách lập trình cho hệ thống Truy vấn Web, Local Retrieval và Pipeline Slurm của dự án AIC 2026.
---

# Phong cách Làm việc (AIC Web & Retrieval Workflow)

Skill này chi phối toàn bộ cách thức hành xử và viết code của Agent khi thao tác tại thư mục `code/`. Mọi Agent tham gia dự án phải tuân thủ nghiêm ngặt các điều sau:

## 1. Xưng hô (Communication)
- Khi giao tiếp trong chat, luôn phải gọi người dùng (User) là **"đại ca"**. (Ví dụ: "Đã hoàn thành xong chức năng này rồi thưa đại ca", "Đại ca xem thử giao diện này đã ổn chưa?").

## 2. Tiêu chuẩn Code trên Slurm (`code/slurm/`)
- **Mục đích:** Chủ yếu dùng để bóc tách và phân tích Data khối lượng lớn.
- **Tiêu chuẩn bắt buộc:** 
  - Phải tuyệt đối bám sát cấu trúc thư mục của Slurm.
  - Phải có cơ chế **Checkpoint** để lưu và khôi phục tiến trình.
  - Mỗi dòng log in ra phải kèm theo timestamp (thời gian), tính toán ETA (thời gian ước tính hoàn thành), và tiến độ phần trăm rõ ràng để đại ca dễ dàng theo dõi.

## 3. Tiêu chuẩn Code Giao diện Web (`code/web/`)
- **Tính mở rộng (Scalability):** Hệ thống web phải được thiết kế dạng Module/Component (Ráp nối) để sau này dễ dàng cắm thêm nhiều thuật toán (CLIP, OCR, ASR...).
- **Phong cách giao diện (UI/UX - Linear / Raycast Pro Minimalist):**
  - **Bố cục Gom Cụm 1 Phía (Unified Dashboard):** Toàn bộ bảng điều khiển, bộ lọc, module nhập liệu và thanh Multi-Scene Switcher phải được gom gọn gàng, cố định tại Cột Sidebar bên trái (Width: 380px), có thanh thao tác dính đáy (Sticky Dock). Tuyệt đối không để các nút bấm hoặc form điều khiển phân tán sang các khu vực khác làm rối mắt và giảm tốc độ thao tác của mắt người.
  - **Không gian Xác Thực Tối Đa (Verification Workspace):** Toàn bộ khu vực bên phải được giải phóng 100% diện tích cho lưới Keyframe kết quả, Video Player, Timeline Strip và Lightbox để con người verify thông tin và nộp bài với tốc độ cao nhất.
  - **Ngôn ngữ Thiết Kế:** Phong cách Dark Slate tối giản hiện đại (`#0b0f19`, `#111827`, viền mỏng tinh tế `1px solid rgba(255,255,255,0.08)`, điểm nhấn sắc nét Emerald `#10b981` / Indigo `#6366f1` / Amber `#f59e0b`).
- **Tính năng (Features):** Hỗ trợ đầy đủ bộ công cụ: Multi-modal Search, RRF Fusion, Gemini AI Parser, Multi-Scene Segmented Switcher, Smart Dedup, 3 Nút Bấm Tác Chiến (`SEARCH NOW`, `MULTI-SCENE`, `RERANK`), Video Player HTML5 với tua frame và copy ID 1-click.

## 4. Tiêu chuẩn Code Thuật toán Truy vấn (`code/local_retrieval/`)
- **Single Responsibility:** Viết code chia nhỏ thành từng hàm, mỗi hàm thực hiện đúng một mục đích duy nhất.
- **Tham số hóa:** Code phải chạy được ngay lập tức ra kết quả. Các tham số (Threshold, Top K, Weight...) phải được gom lại một chỗ, chú thích chức năng thật đầy đủ để đại ca dễ dàng tinh chỉnh.
- **Đường dẫn Động (Dynamic Paths):** Tuyệt đối KHÔNG DÙNG đường dẫn cứng kiểu tuyệt đối (`C:\Users\...` hay `l:\Competitions\...`) trong code. Phải dùng `os.path.join(os.path.dirname(__file__), ...)` để đảm bảo khi copy thư mục `code/` sang máy khác, dự án vẫn chạy mượt mà không gãy link.
- **Tái lập kết quả (Reproducible):** Nếu dùng random, phải set seed. Các bước chạy phải ra cùng một kết quả ở mọi lần test.
- **Quản lý API / Cấu hình:** Mọi biến cấu hình hoặc đường dẫn API (nếu có) phải được đặt tập trung tại một file config duy nhất.

## 5. Quy chuẩn Môi trường và Kiểm thử (Environments & Testing)
- **Local Retrieval (`code/local_retrieval/`):** TẤT CẢ mọi code liên quan đến local search, web API đều phải chạy trên một môi trường ảo (virtual environment) duy nhất được khởi tạo bên trong thư mục `code/local_retrieval/`. Tuyệt đối không dùng môi trường global.
- **Test Pipeline Slurm (`code/slurm/`):** 
  - Trước khi code được ném lên Slurm để chạy thật, phải tạo một môi trường ảo (venv) ngay bên trong thư mục `code/slurm/` để test nhanh thuật toán và check lỗi cú pháp.
  - Sau khi test thành công trên môi trường giả lập, phải xuất ra một phiên bản Script hoàn chỉnh cho Slurm.
  - **Dọn dẹp:** Sau khi test xong, BẮT BUỘC phải xóa toàn bộ dữ liệu nháp, file rác, file log vừa sinh ra để tiết kiệm ổ cứng. Tuy nhiên, được phép **GIỮ LẠI môi trường ảo venv** trong `code/slurm/` để tái sử dụng cho các lần test Pipeline (ASR/OCR) sau này.
