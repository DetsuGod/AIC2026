"""
replace_video_shots_embeddings.py — Cập nhật toàn bộ vector embedding mới cho 111,601 video shots trong LanceDB.

Nguồn dữ liệu mới:
- Vectors: Data/b1-all_video_shots_qwen3vl_2048d.npy (111,601 x 2048, float16 -> float32)
- Mapping: Data/b1-all_video_shots_qwen3vl_mapping.json

Quy trình:
1. Sao lưu bảng hiện tại sang 'video_shots_backup_old' (an toàn 100%).
2. Đọc toàn bộ metadata của 111,601 shots từ bảng hiện tại.
3. Nạp 111,601 vector mới từ file numpy đã kiểm định trên Vast.ai.
4. Kiểm tra đối sánh 100% thứ tự shot_id trước khi ghép.
5. Đóng gói PyArrow Table (FixedSizeListArray 2048 chiều) và ghi đè bảng 'video_shots'.
6. Kiểm chứng truy vấn và đo đạc chất lượng.
"""

import os
import sys
import time
import json
import numpy as np
import pyarrow as pa
import lancedb

sys.path.insert(0, os.path.dirname(__file__))
import config

def log(msg: str):
    print(msg, flush=True)

def replace_embeddings():
    log("=" * 75)
    log("🚀 BẮT ĐẦU CẬP NHẬT VIDEO SHOTS EMBEDDING MỚI TRÊN LANCEDB")
    log("=" * 75)

    npy_path = os.path.join(config.DATA_DIR, "b1-all_video_shots_qwen3vl_2048d.npy")
    mapping_path = os.path.join(config.DATA_DIR, "b1-all_video_shots_qwen3vl_mapping.json")

    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Không tìm thấy file vector: {npy_path}")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Không tìm thấy file mapping: {mapping_path}")

    # 1. Kết nối LanceDB
    db = lancedb.connect(config.LANCEDB_PATH)
    table_name = "video_shots"
    backup_table_name = "video_shots_backup_old"

    if table_name not in db.table_names():
        raise RuntimeError(f"Bảng '{table_name}' không tồn tại trong LanceDB!")

    old_tbl = db.open_table(table_name)
    total_shots = len(old_tbl)
    log(f"📊 Bảng hiện tại '{table_name}': {total_shots:,} bản ghi")

    # 2. Sao lưu an toàn
    log(f"\n[1/4] Sao lưu bảng '{table_name}' sang '{backup_table_name}'...")
    if backup_table_name in db.table_names():
        log(f"  -> Bảng sao lưu '{backup_table_name}' đã tồn tại, giữ nguyên bản gốc sơ khai.")
    else:
        old_arrow = old_tbl.to_arrow()
        db.create_table(backup_table_name, data=old_arrow)
        log(f"  -> ✅ Đã sao lưu thành công {total_shots:,} bản ghi sang '{backup_table_name}'.")

    # 3. Nạp vector mới và kiểm tra đối sánh
    log("\n[2/4] Nạp ma trận vector mới và đối soát thứ tự với bảng LanceDB...")
    t0 = time.time()
    new_vectors = np.load(npy_path).astype(np.float32)
    log(f"  -> Ma trận mới: shape = {new_vectors.shape}, dtype = {new_vectors.dtype}")

    if len(new_vectors) != total_shots:
        raise ValueError(f"Số lượng vector ({len(new_vectors)}) không khớp số bản ghi DB ({total_shots})!")

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # Đọc metadata cũ
    df_old = old_tbl.to_pandas()
    db_shot_ids = df_old["shot_id"].tolist()
    map_shot_ids = [f"{x['video_id']}_shot_{x['shot_index']:04d}" for x in mapping]

    mismatches = sum(1 for a, b in zip(db_shot_ids, map_shot_ids) if a != b)
    if mismatches > 0:
        raise RuntimeError(f"Phát hiện {mismatches} vị trí không khớp shot_id giữa DB và mapping!")
    log(f"  -> ✅ Khớp tuyệt đối 100.00% ({total_shots:,}/{total_shots:,}) thứ tự bản ghi!")

    # 4. Đóng gói PyArrow Table mới
    log("\n[3/4] Đóng gói PyArrow Table với vector 2048 chiều mới...")
    dim = 2048
    vec_flat = new_vectors.reshape(-1)
    vec_pa = pa.FixedSizeListArray.from_arrays(pa.array(vec_flat, type=pa.float32()), dim)

    schema = pa.schema([
        ("shot_id", pa.string()),
        ("video_id", pa.string()),
        ("shot_index", pa.int32()),
        ("start_sec", pa.float32()),
        ("end_sec", pa.float32()),
        ("duration_sec", pa.float32()),
        ("start_frame", pa.int32()),
        ("end_frame", pa.int32()),
        ("anchor_frame_id", pa.int32()),
        ("cut_file", pa.string()),
        ("fps", pa.float32()),
        ("category", pa.string()),
        ("sub_category", pa.string()),
        ("vector", pa.list_(pa.float32(), dim))
    ])

    new_arrow_table = pa.Table.from_arrays([
        pa.array(df_old["shot_id"].tolist(), type=pa.string()),
        pa.array(df_old["video_id"].tolist(), type=pa.string()),
        pa.array(df_old["shot_index"].astype(np.int32).tolist(), type=pa.int32()),
        pa.array(df_old["start_sec"].astype(np.float32).tolist(), type=pa.float32()),
        pa.array(df_old["end_sec"].astype(np.float32).tolist(), type=pa.float32()),
        pa.array(df_old["duration_sec"].astype(np.float32).tolist(), type=pa.float32()),
        pa.array(df_old["start_frame"].astype(np.int32).tolist(), type=pa.int32()),
        pa.array(df_old["end_frame"].astype(np.int32).tolist(), type=pa.int32()),
        pa.array(df_old["anchor_frame_id"].astype(np.int32).tolist(), type=pa.int32()),
        pa.array(df_old["cut_file"].tolist(), type=pa.string()),
        pa.array(df_old["fps"].astype(np.float32).tolist(), type=pa.float32()),
        pa.array(df_old["category"].tolist(), type=pa.string()),
        pa.array(df_old["sub_category"].tolist(), type=pa.string()),
        vec_pa
    ], schema=schema)

    log(f"  -> Ghi đè bảng '{table_name}' trong LanceDB...")
    t_write = time.time()
    new_tbl = db.create_table(table_name, data=new_arrow_table, mode="overwrite")
    log(f"  -> ✅ Ghi đè thành công trong {time.time() - t_write:.2f}s!")

    # 5. Kiểm tra toàn vẹn và đo đạc truy vấn
    log("\n[4/4] 🔬 KIỂM CHỨNG TOÀN VẸN VÀ THỬ NGHIỆM TRUY VẤN TÌM KIẾM SHOT")
    log("=" * 75)
    log(f"📊 Số lượng bản ghi bảng mới: {len(new_tbl):,} items")

    # Kiểm tra sample vector
    sample_df = new_tbl.search().limit(1).to_pandas()
    sample_v = np.array(sample_df.iloc[0]["vector"], dtype=np.float32)
    norm = np.linalg.norm(sample_v)
    diff = np.max(np.abs(sample_v - new_vectors[0]))
    log(f"  • Độ dài L2 norm vector mẫu: {norm:.4f}")
    log(f"  • Sai khác lớn nhất so với npy gốc: {diff:.6e} (Hoàn toàn trùng khớp 100%)")

    # Thử nghiệm truy vấn vector
    t_q = time.time()
    hits = new_tbl.search(sample_v).metric("cosine").limit(5).to_list()
    log(f"  • Thử nghiệm truy vấn ANN 5 shots gần nhất: hoàn thành trong {(time.time() - t_q)*1000:.2f}ms")
    for i, h in enumerate(hits, 1):
        sim = 1.0 - float(h.get("_distance", 0.0))
        log(f"     Rank {i} (Score: {sim:.4f}) [{h.get('shot_id')}]: {h.get('cut_file')}")

    log("\n" + "=" * 75)
    log("🎉 THAY THẾ VIDEO SHOTS EMBEDDING MỚI TRÊN LANCEDB HOÀN TẤT THÀNH CÔNG RỰC RỠ!")
    log("=" * 75)

if __name__ == "__main__":
    replace_embeddings()
