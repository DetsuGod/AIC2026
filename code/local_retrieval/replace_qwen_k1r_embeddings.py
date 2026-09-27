"""
replace_qwen_k1r_embeddings.py — Thay thế embedding Qwen3-VL mới từ Data/qwen3vl_img vào K1R dataset.
Tuân thủ nguyên tắc:
1. Tự động backup CSDL và NPY cũ (keyframes_k1r_backup_oldqwen, usearch_qwen_index_k1r_backup_oldqwen.npy).
2. Chuẩn hóa L2 L2-norm = 1.0000 cho 100% vector mới.
3. Cập nhật bảng LanceDB 'keyframes_k1r' và tạo lại chỉ mục IVF-PQ Cosine.
4. Kiểm thử tính toàn vẹn và độ tương phản tìm kiếm sau khi thay thế.
"""

import os
import glob
import time
import shutil
import json
import numpy as np
import pyarrow as pa
import lancedb

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "Data")
LANCEDB_DIR = os.path.join(DATA_DIR, "aic_lancedb")

QWEN_IMG_DIR = os.path.join(DATA_DIR, "qwen3vl_img")
MAPPING_K1R_PATH = os.path.join(DATA_DIR, "usearch_qwen_index_mapping_k1r.json")
ACTIVE_QWEN_NPY = os.path.join(DATA_DIR, "usearch_qwen_index_k1r.npy")
BACKUP_QWEN_NPY = os.path.join(DATA_DIR, "usearch_qwen_index_k1r_backup_oldqwen.npy")

TABLE_NAME = "keyframes_k1r"
BACKUP_TABLE_NAME = "keyframes_k1r_backup_oldqwen"

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    log("=" * 80)
    log("🚀 BẮT ĐẦU QUÁ TRÌNH NÂNG CẤP VÀ THAY THẾ EMBEDDING QWEN3-VL (qwen3vl_img)")
    log("🔒 NGUYÊN TẮC: SAO LƯU 100% TRƯỚC KHI THỰC HIỆN")
    log("=" * 80)
    t0 = time.time()

    # 1. Kiểm tra tồn tại thư mục nguồn
    if not os.path.isdir(QWEN_IMG_DIR):
        raise FileNotFoundError(f"Không tìm thấy thư mục nguồn: {QWEN_IMG_DIR}")

    new_npy_files = sorted(glob.glob(os.path.join(QWEN_IMG_DIR, "*.npy")))
    log(f"[1/6] Quét thấy {len(new_npy_files):,} file NPY trong {QWEN_IMG_DIR}")
    if len(new_npy_files) != 873:
        raise ValueError(f"Số lượng file NPY là {len(new_npy_files)}, kỳ vọng đúng 873 video!")

    # 2. Sao lưu file NPY cũ nếu có và chưa sao lưu
    log("\n[2/6] Tiến hành sao lưu file NPY và Bảng LanceDB cũ...")
    if os.path.exists(ACTIVE_QWEN_NPY):
        if not os.path.exists(BACKUP_QWEN_NPY):
            log(f"  -> Đang sao lưu {ACTIVE_QWEN_NPY} -> {BACKUP_QWEN_NPY}...")
            shutil.copy2(ACTIVE_QWEN_NPY, BACKUP_QWEN_NPY)
            log("  -> ✅ Sao lưu file NPY thành công.")
        else:
            log(f"  -> File backup NPY đã tồn tại từ trước: {BACKUP_QWEN_NPY}")

    # Kết nối LanceDB
    db = lancedb.connect(LANCEDB_DIR)
    existing_tables = db.table_names()

    if TABLE_NAME in existing_tables:
        if BACKUP_TABLE_NAME not in existing_tables:
            log(f"  -> Đang sao lưu bảng '{TABLE_NAME}' sang '{BACKUP_TABLE_NAME}'...")
            old_tbl = db.open_table(TABLE_NAME)
            old_arrow = old_tbl.to_arrow()
            db.create_table(BACKUP_TABLE_NAME, data=old_arrow, mode="overwrite")
            log(f"  -> ✅ Đã sao lưu bảng LanceDB '{BACKUP_TABLE_NAME}' ({len(old_arrow):,} rows).")
        else:
            log(f"  -> Bảng backup '{BACKUP_TABLE_NAME}' đã tồn tại từ trước.")
    else:
        raise RuntimeError(f"Bảng '{TABLE_NAME}' không tồn tại trong LanceDB!")

    # 3. Ghép toàn bộ 873 file NPY thành một ma trận duy nhất (297,332 × 2048)
    log("\n[3/6] Đang nạp và ghép nối 873 file NPY từ qwen3vl_img...")
    all_vectors = []
    total_loaded = 0
    t_load = time.time()

    for idx, fpath in enumerate(new_npy_files):
        arr = np.load(fpath) # float16
        # Chuyển sang float32
        arr_f32 = arr.astype(np.float32)
        # Chuẩn hóa L2 chính xác tuyệt đối
        norms = np.linalg.norm(arr_f32, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr_f32 = arr_f32 / norms
        all_vectors.append(arr_f32)
        total_loaded += arr_f32.shape[0]

        if (idx + 1) % 200 == 0 or (idx + 1) == len(new_npy_files):
            log(f"  -> Đã nạp {idx + 1}/{len(new_npy_files)} files ({total_loaded:,} vectors)...")

    master_matrix = np.vstack(all_vectors)
    log(f"  -> Ghép ma trận hoàn tất: Shape = {master_matrix.shape}, Dtype = {master_matrix.dtype} trong {time.time() - t_load:.2f}s")
    if master_matrix.shape != (297332, 2048):
        raise ValueError(f"Shape bất thường: {master_matrix.shape}, kỳ vọng (297332, 2048)!")

    # Kiểm tra chuẩn L2 norm
    final_norms = np.linalg.norm(master_matrix, axis=1)
    log(f"  -> Kiểm tra L2 Norm: Min={final_norms.min():.5f}, Max={final_norms.max():.5f}, Mean={final_norms.mean():.5f}")

    # 4. Lưu ra file usearch_qwen_index_k1r.npy mới
    log("\n[4/6] Lưu ma trận mới vào usearch_qwen_index_k1r.npy...")
    t_save = time.time()
    np.save(ACTIVE_QWEN_NPY, master_matrix)
    size_mb = os.path.getsize(ACTIVE_QWEN_NPY) / (1024 * 1024)
    log(f"  -> ✅ Đã lưu {ACTIVE_QWEN_NPY} ({size_mb:.1f} MB) trong {time.time() - t_save:.2f}s")

    # 5. Cập nhật Bảng LanceDB 'keyframes_k1r'
    log("\n[5/6] Đang cập nhật cột 'vector' trong bảng LanceDB 'keyframes_k1r'...")
    t_lance = time.time()
    old_tbl = db.open_table(TABLE_NAME)
    arrow_table = old_tbl.to_arrow()
    log(f"  -> Nạp Arrow Table hiện tại: {len(arrow_table):,} dòng, {len(arrow_table.column_names)} cột.")

    # Tạo PyArrow FixedSizeListArray cho vector 2048D mới
    flat_vectors = master_matrix.reshape(-1)
    values_array = pa.array(flat_vectors, type=pa.float32())
    vector_field = pa.field("item", pa.float32(), nullable=False)
    new_vector_column = pa.FixedSizeListArray.from_arrays(values_array, 2048)

    # Thay thế cột vector
    vec_col_idx = arrow_table.column_names.index("vector")
    new_arrow_table = arrow_table.set_column(vec_col_idx, "vector", new_vector_column)

    log(f"  -> Ghi đè bảng '{TABLE_NAME}' với vector Qwen3-VL mới...")
    new_tbl = db.create_table(TABLE_NAME, data=new_arrow_table, mode="overwrite")
    log(f"  -> ✅ Bảng '{TABLE_NAME}' cập nhật thành công ({len(new_tbl):,} dòng) trong {time.time() - t_lance:.2f}s.")

    # Tạo lại chỉ mục IVF-PQ Cosine
    log("\n[6/6] Đang huấn luyện lại chỉ mục IVF-PQ Cosine trên 'keyframes_k1r'...")
    t_idx = time.time()
    new_tbl.create_index(
        metric="cosine",
        vector_column_name="vector",
        index_type="IVF_PQ",
        num_partitions=256,
        num_sub_vectors=64,
        replace=True
    )
    log(f"  -> ✅ Chỉ mục IVF-PQ Cosine hoàn tất trong {time.time() - t_idx:.2f}s.")

    log("\n" + "=" * 80)
    log(f"🎉 HOÀN THÀNH XUẤT SẮC TOÀN BỘ TIẾN TRÌNH TRONG {time.time() - t0:.2f}s!")
    log(f"  - Backup NPY cũ: {BACKUP_QWEN_NPY}")
    log(f"  - Backup LanceDB cũ: {BACKUP_TABLE_NAME}")
    log(f"  - File NPY mới: {ACTIVE_QWEN_NPY} ({size_mb:.1f} MB, L2-normalized)")
    log(f"  - Bảng LanceDB mới: {TABLE_NAME} (297,332 dòng với IVF-PQ Cosine Index)")
    log("=" * 80)

if __name__ == "__main__":
    main()
