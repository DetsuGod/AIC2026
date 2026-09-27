"""
reencode_asr_lancedb.py — Tái mã hóa toàn bộ 16,609 phân đoạn ASR trong LanceDB
bằng Qwen/Qwen3-VL-Embedding-2B theo chuẩn Document-side (prompt_name="document", L2-norm).

Quy trình:
1. Sao lưu bảng cũ sang 'asr_segments_backup' (an toàn 100%).
2. Nạp toàn bộ 16,609 văn bản ASR từ bảng hiện tại.
3. Dùng TextEncoder.encode_documents() theo batch trên GPU RTX.
4. Ghi đè bảng 'asr_segments' với vector mới.
5. Tiến hành benchmark đánh giá khách quan giữa Bản Cũ và Bản Mới.
"""

import os
import sys
import time
import json
import numpy as np
import pyarrow as pa
import lancedb

# Nạp module nội bộ
sys.path.insert(0, os.path.dirname(__file__))
import config
from text_encoder import get_text_encoder, OFFICIAL_INSTRUCT_PROMPT, OFFICIAL_SPEECH_INSTRUCT_PROMPT

def log(msg: str):
    print(msg, flush=True)

def reencode_asr():
    log("=" * 75)
    log("🚀 BẮT ĐẦU QUY TRÌNH RE-ENCODE TOÀN BỘ KHO ASR SANG CHUẨN QWEN3-VL DOCUMENT")
    log("=" * 75)

    db_path = config.LANCEDB_PATH
    log(f"📍 Kết nối LanceDB tại: {db_path}")
    db = lancedb.connect(db_path)

    if "asr_segments" not in db.table_names():
        raise RuntimeError("Không tìm thấy bảng 'asr_segments' trong LanceDB!")

    tbl = db.open_table("asr_segments")
    total_rows = len(tbl)
    log(f"📊 Tổng số phân đoạn ASR cần xử lý: {total_rows:,} bản ghi")

    # BƯỚC 1: SAO LƯU BẢNG HIỆN TẠI (BACKUP)
    backup_table_name = "asr_segments_backup"
    log(f"\n[1/4] Sao lưu bảng 'asr_segments' sang '{backup_table_name}'...")
    old_arrow = tbl.to_arrow()
    if backup_table_name in db.table_names():
        log(f"  -> Bảng '{backup_table_name}' đã tồn tại từ trước, giữ nguyên bản gốc sơ khai.")
    else:
        db.create_table(backup_table_name, data=old_arrow)
        log(f"  -> ✅ Đã sao lưu thành công {total_rows:,} bản ghi sang '{backup_table_name}'.")

    # BƯỚC 2: TRÍCH XUẤT VĂN BẢN VÀ RE-ENCODE TRÊN GPU
    log("\n[2/4] Đang nạp toàn bộ danh sách câu thoại và kích hoạt TextEncoder...")
    df = tbl.to_pandas()
    texts = df["text"].tolist()

    encoder = get_text_encoder()
    encoder._load()

    batch_size = 64
    log(f"  -> Bắt đầu mã hóa {len(texts):,} câu (batch_size={batch_size}) qua GPU...")
    t0 = time.time()
    new_vectors = encoder.encode_documents(
        texts=texts,
        batch_size=batch_size,
        show_progress_bar=True
    )
    t_encode = time.time() - t0
    log(f"  -> ✅ Mã hóa hoàn tất trong {t_encode:.2f}s (Tốc độ: {len(texts)/t_encode:.1f} câu/giây)!")

    # BƯỚC 3: XÂY DỰNG PYARROW TABLE VÀ GHI ĐÈ LANCEDB
    log("\n[3/4] Đang đóng gói dữ liệu và cập nhật bảng 'asr_segments'...")
    dim = 2048
    vec_flat = new_vectors.astype(np.float32).reshape(-1)
    vec_pa = pa.FixedSizeListArray.from_arrays(pa.array(vec_flat, type=pa.float32()), dim)

    schema = pa.schema([
        ("id", pa.string()),
        ("segment_id", pa.string()),
        ("video_id", pa.string()),
        ("text", pa.string()),
        ("asr_start", pa.float32()),
        ("asr_end", pa.float32()),
        ("matched_keyframes_json", pa.string()),
        ("vector", pa.list_(pa.float32(), dim))
    ])

    pa_table = pa.Table.from_arrays([
        pa.array(df["id"].tolist(), type=pa.string()),
        pa.array(df["segment_id"].tolist(), type=pa.string()),
        pa.array(df["video_id"].tolist(), type=pa.string()),
        pa.array(df["text"].tolist(), type=pa.string()),
        pa.array(df["asr_start"].astype(np.float32).tolist(), type=pa.float32()),
        pa.array(df["asr_end"].astype(np.float32).tolist(), type=pa.float32()),
        pa.array(df["matched_keyframes_json"].tolist(), type=pa.string()),
        vec_pa
    ], schema=schema)

    new_tbl = db.create_table("asr_segments", data=pa_table, mode="overwrite")
    log(f"  -> ✅ Đã ghi đè thành công bảng 'asr_segments' với {len(new_tbl):,} bản ghi chuẩn.")

    # BƯỚC 4: BENCHMARK SO SÁNH KHÁCH QUAN GIỮA BẢN CŨ VÀ BẢN MỚI
    log("\n[4/4] 🔬 TIẾN HÀNH ĐỐI CHIẾU THỰC NGHIỆM GIỮA BẢN CŨ VÀ BẢN MỚI")
    log("=" * 75)

    old_tbl = db.open_table(backup_table_name)
    old_df = old_tbl.to_pandas()
    old_vecs = np.stack(old_df["vector"].to_numpy())

    # 1. So sánh phân phối hình học vector
    cosines_same_doc = np.sum(old_vecs * new_vectors, axis=1)
    log(f"📌 ĐỘ TƯƠNG ĐỒNG COSINE CỦA CÙNG MỘT VĂN BẢN (OLD vs NEW):")
    log(f"   • Mean: {float(np.mean(cosines_same_doc)):.5f}")
    log(f"   • Min:  {float(np.min(cosines_same_doc)):.5f}")
    log(f"   • Max:  {float(np.max(cosines_same_doc)):.5f}")
    log(f"   • Std:  {float(np.std(cosines_same_doc)):.5f}")

    # 2. Benchmark 5 kịch bản truy vấn thực tế
    test_queries = [
        {
            "query": "Chào mừng quý vị đến với chương trình 60 giây của đài truyền hình thành phố Hồ Chí Minh",
            "desc": "Truy vấn trùng khớp câu mở đầu chương trình thời sự"
        },
        {
            "query": "đồng bằng sông Cửu Long sụt lún gấp 20 lần so với nước biển dâng",
            "desc": "Truy vấn ngữ nghĩa về biến đổi khí hậu / sụt lún"
        },
        {
            "query": "Vận chuyển cấp tốc trái tim từ Hà Nội về Huế ghép cho bệnh nhân",
            "desc": "Truy vấn sự kiện y tế khẩn cấp / ghép tạng"
        },
        {
            "query": "Châu Âu nhiệt độ nóng như thiêu đốt cùng những đám cháy rừng",
            "desc": "Truy vấn thời sự thảm họa cháy rừng quốc tế"
        },
        {
            "query": "dự báo thời tiết khu vực nam bộ mưa dông rải rác",
            "desc": "Truy vấn chủ đề bản tin thời tiết hàng ngày"
        }
    ]

    log("\n📊 ĐỐI CHIẾU KẾT QUẢ TRUY VẤN (RETRIEVAL RETRIEVAL PERFORMANCE):")
    log("-" * 75)

    for idx, q_info in enumerate(test_queries, 1):
        q_text = q_info["query"]
        desc = q_info["desc"]

        # Cách cũ: dùng encoder.encode() (gắn Image Instruct + Cinematography) tìm trên bảng CŨ
        old_q_vec = encoder.encode(q_text, enrich_cinematography=True)
        old_hits = old_tbl.search(old_q_vec).metric("cosine").limit(3).to_list()

        # Cách mới: dùng encoder.encode_query_for_speech() (Speech Instruct) tìm trên bảng MỚI
        new_q_vec = encoder.encode_query_for_speech(q_text)
        new_hits = new_tbl.search(new_q_vec).metric("cosine").limit(3).to_list()

        log(f"\n[Test Case {idx}] \"{q_text}\"")
        log(f"Mục tiêu: {desc}")
        log(f"  • BẢN CŨ  (Query gắn Image Instruct + Bảng ASR cũ):")
        for rank, h in enumerate(old_hits, 1):
            sim = 1.0 - float(h.get("_distance", 0.0))
            txt_snippet = h.get("text", "")[:70].replace("\n", " ")
            log(f"     Rank {rank} (Score: {sim:.4f}) [{h.get('video_id')}]: {txt_snippet}...")

        log(f"  • BẢN MỚI (Query gắn Speech Instruct + Bảng ASR Document-side mới):")
        for rank, h in enumerate(new_hits, 1):
            sim = 1.0 - float(h.get("_distance", 0.0))
            txt_snippet = h.get("text", "")[:70].replace("\n", " ")
            log(f"     Rank {rank} (Score: {sim:.4f}) [{h.get('video_id')}]: {txt_snippet}...")

    log("\n" + "=" * 75)
    log("🎉 TOÀN BỘ TIẾN TRÌNH RE-ENCODE VÀ SO SÁNH HOÀN TẤT THÀNH CÔNG RỰC RỠ!")
    log("=" * 75)

if __name__ == "__main__":
    reencode_asr()
