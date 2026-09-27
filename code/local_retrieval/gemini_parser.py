"""
Gemini Query Parser — Bóc tách & Tối ưu hóa đề bài thô của cuộc thi AIC 2026 (KIS, QA, TRAKE)
Hỗ trợ:
- Single-Scene & Multi-Scene Query Decomposition (Nhận diện chuỗi phân cảnh trước-sau).
- 4 Kênh Đa Phương Thức: Visual Query (Embedding) + OCR Query + Speech Query (ASR) + Object Filter (RT-DETR COCO 80 classes).
"""
import os
import json
import re
from typing import Dict, Any, Optional, List
import requests

try:
    from . import config
except (ImportError, ValueError):
    import config

SYSTEM_INSTRUCTION = """
Bạn là Chuyên gia Tối ưu hóa Truy vấn Thị giác Đa Phương Thức (Multimodal Visual Query Optimizer) cấp cao cho cuộc thi AIC 2026 (Ho Chi Minh City AI Challenge).

### NGUYÊN TẮC CỐT LÕI: TRUNG THỰC TUYỆT ĐỐI & CHUYÊN BIỆT HÓA CÁC KÊNH (STRICT FIDELITY & MULTIMODAL SPECIALIZATION)

1. 🎯 QUY TẮC THIẾT KẾ VISUAL QUERY (DÀNH CHO CLIP / SIGLIP / QWEN-VL EMBEDDINGS):
   - **Bảo tồn trọn vẹn Bối cảnh, Không gian & Hoạt cảnh chung:**
     Khi đề bài mở đầu bằng bối cảnh không gian hoặc hoạt cảnh chung (ví dụ: "Đoạn video về một người phụ nữ dạy nấu ăn cho những người khác...", "Một lễ hội ẩm thực Nhật Bản...", "Cuộc đua xe đạp...", "Phóng sự về trường học...", "Đoạn video múa lân một con lân màu vàng đen trắng..."):
     ➔ **BẮT BUỘC GIỮ LẠI BỐI CẢNH/HOẠT CẢNH NÀY TRONG `visual_query`** (ví dụ: "Người phụ nữ dạy nấu ăn cho những người khác, một người đang cầm tờ giấy hoặc bảng công thức món ăn").
     Tuyệt đối không được vì đã đưa vào ASR/Speech mà lại tước bỏ bối cảnh không gian này khỏi `visual_query`!
   - **Tuyệt đối KHÔNG nhồi nhét chữ viết, số đo, định lượng chi tiết vào `visual_query`:**
     Mô hình thị giác CLIP/SigLIP/Qwen-VL KHÔNG ĐỌC ĐƯỢC CHỮ NHỎ hay các số đo định lượng (như "200g", "500ml", "số 42", "chữ in ABC"). Việc nhồi nhét số đo/chữ viết vào `visual_query` sẽ làm nhiễu và giảm mạnh độ chính xác tìm kiếm hình ảnh.
     ➔ Chuyển toàn bộ các chữ viết, con số, định lượng này sang kênh `ocr_query`.
     ➔ Trong `visual_query`, chỉ mô tả dưới dạng thực thể thị giác (ví dụ: "tờ giấy ghi công thức món ăn", "nguyên liệu thịt xay", "biển hiệu cổng trường").
   - **Thứ tự mô tả từ to đến nhỏ:** `[Bối cảnh / Không gian]` ➔ `[Chủ thể / Nhân vật]` ➔ `[Hành động]` ➔ `[Đồ vật / Chi tiết thị giác]`.
   - **Không tự bịa đặt chi tiết ngoài đề bài:** Chỉ mô tả những gì đề bài đề cập hoặc là sự thật vật lý tự nhiên hiển nhiên (cực quang màu xanh lá tím, tuyết màu trắng).

2. 🔤 QUY TẮC THIẾT KẾ OCR QUERY (DÀNH CHO TEXT ON SCREEN / ON PAPER):
   - Chuyên trách tìm kiếm văn bản, con số, định lượng, tên biển hiệu, công thức xuất hiện trên khung hình:
     Ví dụ: `"200g thịt nạc xay"`, `"thịt nạc xay"`, `"TOYOTA"`, `"Trường THPT"`. Nếu không có chữ để `""`.

3. 🎙️ QUY TẮC THIẾT KẾ SPEECH / ASR QUERY (HYBRID QWEN3-VL SEMANTIC + TF-IDF KEYWORD):
   - Kênh ASR của hệ thống sử dụng **mô hình nhúng ngữ nghĩa Qwen3-VL (2048-dim)** kết hợp ma trận TF-IDF N-gram để truy vấn lời thoại và thuyết minh tiếng Việt trên LanceDB.
   - **Cách thiết kế `speech_query` tối ưu:**
     + **Chủ đề thuyết minh / Phóng sự truyền hình:** Sử dụng câu tự nhiên hoặc cụm từ phản ánh đúng văn phong biên tập viên/phát thanh viên (ví dụ: `"chương trình Món ngon mỗi ngày, hướng dẫn công thức nấu ăn, chuẩn bị nguyên liệu"`, `"cuộc đua xe đạp cúp truyền hình, các tay đua về đích thứ tự nhất nhì ba"`, `"nghệ thuật Lân Sư Rồng, di sản văn hóa phi vật thể quốc gia"`).
     + **Lời thoại nhân vật & Tên riêng:** Trích xuất chính xác lời nói, danh từ riêng, thuật ngữ, sự kiện mà nhân vật hoặc MC phát biểu.
     + **Từ khóa chuyên đề:** Đưa vào các từ khóa phong phú để vừa kích hoạt vector tương đồng Qwen3-VL, vừa khớp chính xác từ vựng trên tầng TF-IDF.
   - Nếu đề bài hoàn toàn không có yếu tố âm thanh, lời thoại hoặc thuyết minh, để `""`.

4. 📦 QUY TẮC OBJECT FILTER (RT-DETR COCO 80 CLASSES):
   - Gợi ý class ID từ 80 lớp COCO (person: 0, bicycle: 1, car: 2, motorcycle: 3, chair: 56, dining table: 60, book: 73, cake: 55...). Không có để `null`.

5. ⏱️ QUY TẮC ĐẶC THÙ CHO TRAKE & MULTI-SCENE (TEMPORAL REASONING & CHAIN DISCRIMINATION):
   - Khi đề bài chứa các nhãn sự kiện tuần tự: `E1:`, `E2:`, `E3:`, `E4:`... hoặc các cụm từ *"tìm các sự kiện sau"*, *"gồm các khoảnh khắc sau"*, hoặc diễn biến thời gian qua nhiều phân cảnh:
     + Đặt `"task_type": "TRAKE"`, `"is_multi_scene": true`.
     + **Nguyên tắc Kế thừa Bối cảnh (Context Inheritance):** Trích xuất câu giới thiệu/bối cảnh chung ở đầu đề bài (ví dụ: "Đoạn video múa lân một con lân màu vàng đen trắng", "Trong đoạn video nấu ăn món nấm") và **TÍCH HỢP BỐI CẢNH NÀY VÀO TỪNG SỰ KIỆN CON `E1, E2, E3...`** trong trường `visual_query`.
     + Đặt `name` cho từng sự kiện theo format: `"E1: <Mô tả ngắn>"`, `"E2: <Mô tả ngắn>"`, v.v.
     + Trong `visual_query` cấp ngoài: Tóm tắt toàn bộ chuỗi sự kiện và bối cảnh video.

6. 🎯 QUY TẮC PHÂN TÍCH TÁI XẾP HẠNG CHO AI RERANKER (DISCRIMINATING FEATURES & RERANK HINT):
   - **`discriminating_features` (Mảng string):** Trích xuất các thực thể, danh từ riêng, số thứ tự, màu sắc hoặc đặc điểm phân biệt cốt lõi nhất để tránh nhầm lẫn giữa các chuỗi có hành động tương tự (ví dụ: `["măng tây", "chảo dầu"]` để phân biệt với cá/khoai tây; hoặc `["cột số 4", "ban giám khảo", "con rồng"]` để phân biệt với các màn biểu diễn lân khác).
   - **`anchor_event_index` (Số nguyên 0-indexed):** Chỉ định sự kiện nào trong chuỗi là điểm neo (anchor) đặc trưng nhất, dễ định vị chính xác nhất.
   - **`rerank_hint` (Chuỗi string ngắn 1-2 câu):** Gợi ý ngắn gọn chỉ dẫn mô hình Qwen-VL Reranker biết cần tập trung quan sát hoặc đối chiếu chi tiết nào trong ảnh để đưa chuỗi đúng nhất lên top.

7. 🎬 ĐƠN CẢNH (SINGLE-SCENE) VS ĐA PHÂN CẢNH (MULTI-SCENE):
   - Đơn cảnh: `is_multi_scene`: false, `scenes`: [].
   - Đa phân cảnh (KIS Multi-scene / TRAKE):
     + `is_multi_scene`: true.
     + `scenes`: Mảng các object theo thứ tự thời gian.
     + Mỗi cảnh con gồm:
       `scene_idx` (int 1, 2...), `name` (string ngắn), `visual_query` (câu mô tả chi tiết của riêng cảnh đó), `ocr_query` (string), `speech_query` (string), `object_filter` (object hoặc null).
     + `visual_query` cấp ngoài: Tóm tắt bao quát toàn bộ câu chuyện.
     + Luôn cung cấp `discriminating_features`, `anchor_event_index`, và `rerank_hint`.

8. 🎥 QUY TẮC NHẬN DIỆN & TỐI ƯU HÓA GÓC QUAY & CHUYỂN ĐỘNG ĐIỆN ẢNH (CINEMATOGRAPHY TAGS ENRICHMENT):
   - Mô hình nhúng thị giác (Qwen3-VL 2048D) được huấn luyện trên hàng trăm triệu video với các thuật ngữ điện ảnh tiếng Anh chuẩn quốc tế. Tín hiệu góc quay tiếng Việt rất dễ bị lu mờ trước các danh từ vật thể nếu không được kích hoạt đúng từ khóa chuẩn.
   - **KHI ĐỀ BÀI ĐỀ CẬP ĐẾN GÓC QUAY, HƯỚNG NHÌN HOẶC CHUYỂN ĐỘNG MÁY ẢNH BẰNG TIẾNG VIỆT:**
     ➔ **BẮT BUỘC TỰ ĐỘNG BỔ SUNG CÁC TỪ KHÓA ĐIỆN ẢNH TIẾNG ANH TƯƠNG ỨNG VÀO `visual_query`** để kích hoạt cực đại độ tương đồng vector:
     + **Góc từ trên xuống / nhìn từ trên cao / flycam / từ nóc nhà / trên không:**
       ➔ Thêm: `"top-down view, bird's-eye view, high-angle overhead shot, aerial view, looking down from above"`
     + **Góc từ dưới lên / góc thấp / nhìn từ mặt đất:**
       ➔ Thêm: `"low-angle shot, worm's-eye view, ground-level upward perspective, looking up from floor"`
     + **Cận cảnh / cực cận / soi cận đồ vật:**
       ➔ Thêm: `"close-up shot, extreme close-up, macro framing, tight focal shot"`
     + **Toàn cảnh / góc rộng / flycam phong cảnh:**
       ➔ Thêm: `"wide-angle shot, extreme wide shot, panoramic landscape view, long shot"`
     + **Lia máy ngang (trái/phải):**
       ➔ Thêm: `"panning shot, horizontal camera pan, camera panning across scene"`
     + **Nghiêng máy dọc (lên/xuống):**
       ➔ Thêm: `"tilting shot, vertical tilt, camera tilting up/down"`
     + **Máy quay bám theo đối tượng / di chuyển theo nhân vật:**
       ➔ Thêm: `"tracking shot, following camera movement, dolly shot, steadycam motion"`
     + **Góc nhìn thứ nhất (POV / qua mắt nhân vật):**
       ➔ Thêm: `"first-person point of view, POV shot, subjective camera perspective"`
     + **Góc nhìn qua vai:**
       ➔ Thêm: `"over-the-shoulder shot, OTS camera angle"`

---

### CÁC VÍ DỤ TIÊU BIỂU (GOLD STANDARD EXAMPLES):

Ví dụ 1 (TRAKE: Múa lân 4 sự kiện tuần tự E1 -> E4):
Input: "Đoạn video múa lân một con lân màu vàng đen trắng, tìm các sự kiện sau:
E1: Lân quay vòng trên cột số 4 bằng 2 chân trước rồi tiếp đất. Khoảnh khắc đầu tiên mà lân bắt đầu xoay vòng.
E2: Khoảnh khắc 4 chân hoàn toàn chạm đất đầu tiên.
E3: Khoảnh khắc đầu tiên 2 người biểu diễn lân cuối chào ban giám khảo.
E4: Sau đó lân tiến lại chào một con rồng. Khoảnh khắc đầu tiên con rồng cử động đầu."
Output JSON:
{
  "task_type": "TRAKE",
  "is_multi_scene": true,
  "scenes": [
    {
      "scene_idx": 1,
      "name": "E1: Lân bắt đầu xoay vòng",
      "visual_query": "Biểu diễn múa lân con lân màu vàng đen trắng, lân đứng bằng 2 chân trước xoay vòng trên cột số 4",
      "ocr_query": "",
      "speech_query": "múa lân",
      "object_filter": {
        "class_id": 0,
        "class_name": "person",
        "min_confidence": 0.3
      }
    },
    {
      "scene_idx": 2,
      "name": "E2: 4 chân chạm đất",
      "visual_query": "Biểu diễn múa lân con lân màu vàng đen trắng, cả 4 chân của con lân hoàn toàn tiếp xúc chạm mặt đất",
      "ocr_query": "",
      "speech_query": "múa lân",
      "object_filter": {
        "class_id": 0,
        "class_name": "person",
        "min_confidence": 0.3
      }
    },
    {
      "scene_idx": 3,
      "name": "E3: Chào ban giám khảo",
      "visual_query": "Biểu diễn múa lân, hai người biểu diễn trong trang phục múa lân cúi đầu chào ban giám khảo",
      "ocr_query": "",
      "speech_query": "múa lân, ban giám khảo",
      "object_filter": {
        "class_id": 0,
        "class_name": "person",
        "min_confidence": 0.3
      }
    },
    {
      "scene_idx": 4,
      "name": "E4: Con rồng cử động đầu",
      "visual_query": "Con lân tiến lại gần con rồng trang trí lễ hội, đầu con rồng bắt đầu cử động",
      "ocr_query": "",
      "speech_query": "múa lân, con rồng",
      "object_filter": null
    }
  ],
  "visual_query": "Biểu diễn múa lân con lân vàng đen trắng qua 4 khoảnh khắc: xoay vòng trên cột, tiếp đất 4 chân, cúi chào ban giám khảo và tiến lại chào con rồng",
  "ocr_query": "",
  "speech_query": "múa lân, ban giám khảo",
  "object_filter": {
    "class_id": 0,
    "class_name": "person",
    "min_confidence": 0.3
  },
  "reranker_query": "Đoạn video múa lân con lân màu vàng đen trắng với 4 sự kiện tuần tự: lân quay vòng trên cột số 4, 4 chân chạm đất, chào ban giám khảo, chào con rồng",
  "qa_question": "",
  "summary": "TRAKE 4 sự kiện: Kế thừa bối cảnh múa lân vàng đen trắng vào từng sự kiện E1-E4, đảm bảo tính liên kết thị giác và thứ tự thời gian nghiêm ngặt."
}

Ví dụ 2 (TRAKE: Nấu ăn măng tây 4 sự kiện):
Input: "E1: Khoảnh khắc đầu tiên bột được bỏ vào tô măng tây.
E2: Khoảnh khắc đầu tiên thấy miếng măng tây đầu tiên tiếp xúc với dầu trong chảo.
E3: Khoảnh khắc miếng măng tây đầu tiên rời khỏi chảo dầu.
E4: Khoảng khắc miếng măng tây cuối cùng rời chảo dầu và nằm hoàn toàn trên dĩa."
Output JSON:
{
  "task_type": "TRAKE",
  "is_multi_scene": true,
  "scenes": [
    {
      "scene_idx": 1,
      "name": "E1: Đổ bột vào tô măng tây",
      "visual_query": "Đầu bếp rắc đổ bột vào tô đựng các cọng măng tây xanh",
      "ocr_query": "",
      "speech_query": "măng tây",
      "object_filter": {
        "class_id": 0,
        "class_name": "person",
        "min_confidence": 0.3
      }
    },
    {
      "scene_idx": 2,
      "name": "E2: Măng tây tiếp xúc dầu",
      "visual_query": "Đầu bếp thả miếng măng tây tẩm bột chạm vào dầu sôi trong chảo chiên",
      "ocr_query": "",
      "speech_query": "chiên măng tây",
      "object_filter": null
    },
    {
      "scene_idx": 3,
      "name": "E3: Gắp măng tây khỏi chảo dầu",
      "visual_query": "Đầu bếp dùng đũa hoặc kẹp gắp miếng măng tây chiên rời khỏi chảo dầu nóng",
      "ocr_query": "",
      "speech_query": "",
      "object_filter": null
    },
    {
      "scene_idx": 4,
      "name": "E4: Măng tây đặt lên dĩa",
      "visual_query": "Miếng măng tây chiên giòn được gắp ra và đặt hoàn toàn trên dĩa thức ăn",
      "ocr_query": "",
      "speech_query": "",
      "object_filter": null
    }
  ],
  "visual_query": "Quy trình chế biến măng tây chiên: đổ bột vào tô măng tây, thả măng tây vào chảo dầu, gắp măng tây ra khỏi dầu và bày lên dĩa",
  "ocr_query": "",
  "speech_query": "măng tây",
  "object_filter": {
    "class_id": 0,
    "class_name": "person",
    "min_confidence": 0.3
  },
  "reranker_query": "Video nấu ăn măng tây: đổ bột vào tô, chiên trong chảo dầu, gắp ra khỏi dầu và đặt trên dĩa",
  "qa_question": "",
  "summary": "TRAKE 4 sự kiện tuần tự chế biến món măng tây chiên."
}

Ví dụ 3 (Đơn cảnh QA: Lớp học nấu ăn & Công thức thịt xay):
Input: "Đoạn video về một người phụ nữ dạy nấu ăn cho những người khác. Trong đoạn video có thể thấy một người đang cầm công thức món ăn với nguyên liệu chính là 200g thịt nạc xay. Hỏi tiêu đề của công thức nấu ăn (tên món ăn) này là gì?"
Output JSON:
{
  "task_type": "QA",
  "is_multi_scene": false,
  "scenes": [],
  "visual_query": "Người phụ nữ dạy nấu ăn cho những người khác, một người đang cầm tờ giấy hoặc bảng công thức món ăn",
  "ocr_query": "200g thịt nạc xay",
  "speech_query": "người phụ nữ dạy nấu ăn",
  "object_filter": {
    "class_id": 0,
    "class_name": "person",
    "min_confidence": 0.3
  },
  "reranker_query": "Người phụ nữ dạy nấu ăn cho những người khác, một người cầm công thức món ăn nguyên liệu 200g thịt nạc xay",
  "qa_question": "Tiêu đề của công thức nấu ăn (tên món ăn) này là gì?",
  "summary": "Visual Query giữ trọn bối cảnh người phụ nữ dạy nấu ăn và hành động cầm tờ giấy công thức món ăn; tách số đo 200g thịt nạc xay sang OCR; trích xuất ASR và câu hỏi QA."
}

Ví dụ 4 (Đơn cảnh KIS: Lễ hội ẩm thực & Cô bé đeo bạch tuộc):
Input: "Đoạn video về một lễ hội ẩm thực Nhật Bản lớn nhất thế giới. Hãy tìm chính xác phân cảnh một cô bé đeo một con bạch tuộc / con mực màu đỏ phía trước ngực. Trên tay cô bé có cầm một chiếc túi giấy."
Output JSON:
{
  "task_type": "KIS",
  "is_multi_scene": false,
  "scenes": [],
  "visual_query": "Không gian lễ hội ẩm thực, một cô bé đeo một con bạch tuộc hoặc con mực màu đỏ phía trước ngực, trên tay cầm một chiếc túi giấy",
  "ocr_query": "",
  "speech_query": "lễ hội ẩm thực Nhật Bản lớn nhất thế giới",
  "object_filter": {
    "class_id": 0,
    "class_name": "person",
    "min_confidence": 0.3
  },
  "reranker_query": "Cô bé đeo con bạch tuộc hoặc con mực màu đỏ phía trước ngực, tay cầm chiếc túi giấy tại lễ hội ẩm thực Nhật Bản lớn nhất thế giới",
  "qa_question": "",
  "summary": "Visual Query giữ không gian lễ hội ẩm thực cùng chi tiết cô bé đeo bạch tuộc đỏ cầm túi giấy; Speech Query chứa chủ đề lễ hội ẩm thực Nhật Bản."
}

### ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:
Chỉ trả về DUY NHẤT một khối JSON hợp lệ theo đúng schema trên.
"""


def _sanitize_visual_query(query: str) -> str:
    """Làm sạch câu truy vấn nhưng giữ trọn vẹn chi tiết thị giác và tính liền mạch của câu văn."""
    if not query:
        return ""
    text = query.strip()
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_query_with_gemini(raw_text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Gọi Gemini API bóc tách đề bài thô thành 4 kênh truy vấn chuyên biệt và các phân cảnh con.
    Sử dụng các model Gemini 3.1 Flash Lite / 3.5 Flash Lite / 3.5 Flash tốc độ cao.
    Nếu không có API key, key lỗi hoặc mạng chậm -> Tự động fallback sang Smart Parser nội bộ.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or getattr(config, "GEMINI_API_KEY", "")
    if not key or not key.strip():
        print("[GeminiParser] Không có API Key -> Sử dụng Smart Fallback Parser nội bộ.")
        return _fallback_smart_parser(raw_text)

    # Danh sách model Gemini Flash chính thức hoạt động tốt nhất cho API Key AIC 2026
    model_candidates = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.5-flash"]

    for model_name in model_candidates:
        try:
            print(f"[GeminiParser] ⏳ Đang gửi request tới Google AI Studio (Model: {model_name})...")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key.strip()}"
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": f"{SYSTEM_INSTRUCTION}\n\nĐỀ BÀI CẦN BÓC TÁCH:\n\"\"\"{raw_text}\"\"\""}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "responseMimeType": "application/json"
                }
            }

            resp = requests.post(url, json=payload, timeout=10.0)
            if resp.status_code == 200:
                res_json = resp.json()
                text_content = res_json['candidates'][0]['content']['parts'][0]['text'].strip()
                
                # Trích xuất khối JSON chuẩn xác
                json_match = re.search(r'\{.*\}', text_content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                else:
                    parsed = json.loads(text_content)
                    
                parsed["visual_query"] = _sanitize_visual_query(parsed.get("visual_query", ""))
                
                # Làm sạch từng visual_query trong scenes nếu có
                if parsed.get("is_multi_scene") and parsed.get("scenes"):
                    for sc in parsed["scenes"]:
                        sc["visual_query"] = _sanitize_visual_query(sc.get("visual_query", ""))
                else:
                    parsed["is_multi_scene"] = False
                    parsed["scenes"] = []

                # Đảm bảo các trường hỗ trợ Qwen-VL Reranker luôn có giá trị chuẩn
                if not isinstance(parsed.get("discriminating_features"), list):
                    parsed["discriminating_features"] = []
                if not isinstance(parsed.get("anchor_event_index"), int):
                    parsed["anchor_event_index"] = 0
                if not isinstance(parsed.get("rerank_hint"), str):
                    parsed["rerank_hint"] = ""

                parsed["source"] = f"Gemini AI ({model_name})"
                print(f"[GeminiParser] ✅ Google AI Studio phản hồi thành công 200 qua model {model_name}!")
                return parsed

            elif resp.status_code in [400, 401, 403]:
                # Lỗi API key không hợp lệ hoặc hết quota -> Fallback ngay lập tức
                print(f"[GeminiParser] ⚠️ Google AI Studio báo lỗi xác thực Key (Status {resp.status_code}): {resp.text[:150]}")
                break
            else:
                print(f"[GeminiParser] ⚠️ Google AI Studio báo Status {resp.status_code}: {resp.text[:150]}")

        except Exception as e:
            print(f"[GeminiParser] ❌ Lỗi kết nối / timeout tới Google AI Studio ({model_name}): {e}")

    # Nếu tất cả model đều lỗi hoặc mất mạng -> Fallback tức thì
    print(f"[GeminiParser] ⚠️ Không thể kết nối tới Google AI Studio -> Chuyển sang Smart Fallback Parser nội bộ.")
    fallback_res = _fallback_smart_parser(raw_text)
    fallback_res["source"] = "Smart Fallback Parser (Nội Bộ)"
    return fallback_res


def _fallback_smart_parser(raw_text: str) -> Dict[str, Any]:
    """Bộ bóc tách dự phòng chạy nội bộ khi không có API Key hoặc mất mạng."""
    cleaned = raw_text.strip()
    is_qa = "?" in cleaned or "Hỏi " in cleaned or "hỏi " in cleaned

    obj_filter = None
    lower_t = cleaned.lower()
    if "xe đạp" in lower_t or "bicycle" in lower_t:
        obj_filter = {"class_id": 1, "class_name": "bicycle", "min_confidence": 0.3}
    elif "ô tô" in lower_t or "xe hơi" in lower_t or "car" in lower_t:
        obj_filter = {"class_id": 2, "class_name": "car", "min_confidence": 0.3}
    elif "xe máy" in lower_t or "motorcycle" in lower_t:
        obj_filter = {"class_id": 3, "class_name": "motorcycle", "min_confidence": 0.3}
    elif "bánh" in lower_t or "cake" in lower_t:
        obj_filter = {"class_id": 55, "class_name": "cake", "min_confidence": 0.3}
    elif "người" in lower_t or "tay đua" in lower_t or "câu lạc bộ" in lower_t:
        obj_filter = {"class_id": 0, "class_name": "person", "min_confidence": 0.3}

    # Bóc OCR từ trích dẫn hoặc từ viết hoa
    quoted = re.findall(r'["\']([^"\']+)["\']', cleaned)
    caps = re.findall(r'\b[A-Z0-9]{3,}\b', cleaned)
    ocr_tokens = list(set(quoted + caps))
    ocr_query = " ".join(ocr_tokens) if ocr_tokens else ""

    # Trích xuất đặc trưng phân biệt ban đầu
    discriminating = []
    if ocr_tokens:
        discriminating.extend(ocr_tokens)
    keywords = re.findall(r'\b(?:măng tây|cá|khoai tây|áo xanh|cột số \d+|ban giám khảo|con rồng|xe đạp|tay đua)\b', lower_t)
    for kw in keywords:
        if kw not in discriminating:
            discriminating.append(kw)

    # Gọt bớt các từ râu ria đề bài
    vis = cleaned
    boilerplates = [
        r'^\s*đoạn video về\s*',
        r'^\s*tìm phân cảnh\s*',
        r'^\s*trong khung hình gồm có\s*',
        r'^\s*đoạn video cho thấy\s*',
        r'^\s*đây là phân cảnh\s*'
    ]
    for bp in boilerplates:
        vis = re.sub(bp, '', vis, flags=re.IGNORECASE)

    vis = _sanitize_visual_query(vis)

    # 1. TRAKE shorthand: mỗi đoạn cách nhau bởi ít nhất một dòng trống là một event.
    paragraph_events = [
        _sanitize_visual_query(block.replace("\n", " "))
        for block in re.split(r"\n[\t ]*\n+", cleaned)
        if block.strip()
    ]
    if len(paragraph_events) >= 2:
        scenes = []
        for idx, event_text in enumerate(paragraph_events):
            scenes.append({
                "scene_idx": idx + 1,
                "name": f"E{idx + 1}: {event_text[:30]}...",
                "visual_query": event_text,
                "ocr_query": "",
                "speech_query": "",
                "object_filter": obj_filter
            })
        return {
            "task_type": "TRAKE",
            "is_multi_scene": True,
            "scenes": scenes,
            "visual_query": "; ".join(paragraph_events),
            "ocr_query": ocr_query,
            "speech_query": "",
            "object_filter": obj_filter,
            "discriminating_features": discriminating,
            "anchor_event_index": 0,
            "rerank_hint": f"Quan sát chuỗi {len(scenes)} sự kiện tuần tự.",
            "reranker_query": cleaned,
            "qa_question": "",
            "summary": f"Bóc tách TRAKE {len(scenes)} sự kiện từ các đoạn cách dòng."
        }

    # 2. Kiểm tra TRAKE pattern E1:, E2:...
    trake_matches = list(re.finditer(r'(E\d+)\s*:\s*(.+?)(?=(?:E\d+\s*:|$))', cleaned, re.DOTALL | re.IGNORECASE))
    if len(trake_matches) >= 2:
        # Lấy bối cảnh toàn cục trước E1 (nếu có)
        first_e_pos = trake_matches[0].start()
        global_context = cleaned[:first_e_pos].strip()
        global_context = re.sub(r'tìm các sự kiện sau.*|gồm các khoảnh khắc.*', '', global_context, flags=re.IGNORECASE).strip()
        global_context = re.sub(r'^\s*đoạn video\s*(về|múa|nấu)?\s*', '', global_context, flags=re.IGNORECASE).strip()

        scenes = []
        for idx, m in enumerate(trake_matches):
            e_label = m.group(1).upper()
            e_text = _sanitize_visual_query(m.group(2))
            
            # Kế thừa bối cảnh video chung
            event_vis = f"{global_context}, {e_text}" if global_context else e_text
            scenes.append({
                "scene_idx": idx + 1,
                "name": f"{e_label}: {e_text[:30]}...",
                "visual_query": event_vis,
                "ocr_query": "",
                "speech_query": global_context if global_context else "",
                "object_filter": obj_filter
            })

        return {
            "task_type": "TRAKE",
            "is_multi_scene": True,
            "scenes": scenes,
            "visual_query": f"{global_context} với {len(scenes)} sự kiện tuần tự" if global_context else vis,
            "ocr_query": ocr_query,
            "speech_query": global_context if global_context else "",
            "object_filter": obj_filter,
            "discriminating_features": discriminating,
            "anchor_event_index": 0,
            "rerank_hint": f"Quan sát chuỗi {len(scenes)} sự kiện tuần tự để đối chiếu đặc trưng phân biệt: {', '.join(discriminating) if discriminating else 'toàn cảnh'}",
            "reranker_query": cleaned,
            "qa_question": "",
            "summary": f"Bóc tách TRAKE {len(scenes)} sự kiện tự động (Fallback Parser)."
        }

    # 3. Kiểm tra phân cảnh đa đoạn thông thường (sau đó, tiếp theo)
    scenes = []
    is_multi_scene = False
    if "phân cảnh tiếp theo" in lower_t or "sau đó" in lower_t or "tiếp theo" in lower_t:
        parts = re.split(r'phân cảnh tiếp theo|sau đó|tiếp theo', cleaned, flags=re.IGNORECASE)
        if len(parts) >= 2:
            is_multi_scene = True
            for idx, p in enumerate(parts):
                p_clean = _sanitize_visual_query(p)
                if p_clean:
                    scenes.append({
                        "scene_idx": idx + 1,
                        "name": f"Cảnh {idx + 1}",
                        "visual_query": p_clean,
                        "ocr_query": "",
                        "speech_query": "",
                        "object_filter": obj_filter
                    })

    return {
        "task_type": "QA" if is_qa else "KIS",
        "is_multi_scene": is_multi_scene,
        "scenes": scenes,
        "visual_query": vis,
        "ocr_query": ocr_query,
        "speech_query": "",
        "object_filter": obj_filter,
        "discriminating_features": discriminating,
        "anchor_event_index": 0,
        "rerank_hint": f"Đối chiếu chi tiết phân biệt: {', '.join(discriminating)}" if discriminating else "",
        "reranker_query": cleaned,
        "qa_question": cleaned if is_qa else "",
        "summary": "Bóc tách bởi Smart Fallback Parser (Offline mode)."
    }

