import os
import sys
import time
import json
import numpy as np
import torch
import lancedb

def detect_cross_video_ads(
    db_path=r"L:\Competitions\2026-AIC\Data\aic_lancedb",
    output_dir=r"L:\Competitions\2026-AIC\Data",
    sim_threshold=0.96,
    min_video_count=2, # xuất hiện ở ít nhất 2 video_id khác nhau
    batch_size=2000
):
    print("=" * 75)
    print(f"🚀 BẮT ĐẦU QUÉT QUẢNG CÁO / INTRO / BUMPER XUYÊN VIDEO TỪ LANCEDB")
    print(f"  • Ngưỡng tương đồng (Cosine Similarity): >= {sim_threshold}")
    print(f"  • Điều kiện: Xuất hiện ở >= {min_video_count} video_id khác nhau")
    print("=" * 75)

    t0 = time.time()
    db = lancedb.connect(db_path)
    tbl = db.open_table("video_shots")
    total_shots = tbl.count_rows()
    print(f"📊 Tổng số video shots trong LanceDB: {total_shots:,}")

    # Đọc dữ liệu
    print("⏳ Đang nạp metadata và vector từ LanceDB...")
    arrow_tbl = tbl.to_arrow()
    
    shot_ids = arrow_tbl["shot_id"].to_pylist()
    video_ids = arrow_tbl["video_id"].to_pylist()
    cut_files = arrow_tbl["cut_file"].to_pylist()
    durations = arrow_tbl["duration_sec"].to_pylist()
    categories = arrow_tbl["category"].to_pylist()
    
    # Vector float16 trên GPU
    vectors_raw = arrow_tbl["vector"].combine_chunks()
    values = np.array(vectors_raw.values, dtype=np.float32)
    vectors = values.reshape((total_shots, 2048))
    del arrow_tbl, values, vectors_raw
    
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    vectors = vectors / norms
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v_tensor = torch.tensor(vectors, dtype=torch.float16, device=device)
    print(f"✅ Đã nạp {total_shots:,} vectors lên {device} ({time.time()-t0:.2f}s)!\n")

    vid_array = np.array(video_ids)
    
    # Ad detection using graph components / adjacency
    # shot_index -> set of matched shot_indices in other videos
    print(f"🔍 Đang tính toán ma trận tương đồng theo batch ({batch_size} shots/lô)...")
    
    # Adjacency list: shot_i -> set of shot_j
    adj = {i: set() for i in range(total_shots)}
    t_search = time.time()
    
    for start_idx in range(0, total_shots, batch_size):
        end_idx = min(start_idx + batch_size, total_shots)
        q_batch = v_tensor[start_idx:end_idx]
        
        # Batch cosine similarity on GPU: (B, N)
        sim_batch = torch.matmul(q_batch, v_tensor.T) # shape (B, total_shots)
        
        # Lọc nhanh các vị trí >= sim_threshold
        high_sim_mask = (sim_batch >= sim_threshold)
        high_sim_indices = torch.nonzero(high_sim_mask, as_tuple=False).cpu().numpy()
        
        for q_row, t_col in high_sim_indices:
            global_q = start_idx + q_row
            global_t = t_col
            if global_q != global_t and vid_array[global_q] != vid_array[global_t]:
                adj[global_q].add(global_t)
                adj[global_t].add(global_q)
                
        pct = (end_idx / total_shots) * 100
        print(f"  ⚡ Đã quét [{end_idx:,}/{total_shots:,}] ({pct:5.1f}%) | Thời gian: {time.time()-t_search:.1f}s")

    print(f"\n✅ Hoàn tất tính toán trong {time.time()-t_search:.2f}s!")

    # Connected Components (Gom cụm các shot quảng cáo giống nhau)
    print("🧩 Đang gom cụm (Clustering) các đoạn quảng cáo / bumper...")
    visited = set()
    clusters = []
    
    for i in range(total_shots):
        if i not in visited and len(adj[i]) > 0:
            # BFS / DFS to find component
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # Check unique video_ids in this cluster
            unique_vids = set(video_ids[idx] for idx in comp)
            if len(unique_vids) >= min_video_count:
                clusters.append({
                    "indices": comp,
                    "unique_video_count": len(unique_vids),
                    "unique_videos": sorted(list(unique_vids)),
                    "shot_count": len(comp)
                })

    # Sort clusters by frequency (số video xuất hiện giảm dần)
    clusters.sort(key=lambda c: (c["unique_video_count"], c["shot_count"]), reverse=True)
    
    # Tổng hợp danh sách blacklist
    blacklist_shot_files = []
    blacklist_shot_ids = []
    cluster_reports = []
    
    for c_id, cl in enumerate(clusters, 1):
        c_shots = []
        for idx in cl["indices"]:
            fname = os.path.basename(cut_files[idx])
            s_id = shot_ids[idx]
            blacklist_shot_files.append(fname)
            blacklist_shot_ids.append(s_id)
            c_shots.append({
                "shot_id": s_id,
                "video_id": video_ids[idx],
                "filename": fname,
                "duration_sec": round(durations[idx], 2),
                "cut_file": cut_files[idx]
            })
            
        # Chọn 1 shot đại diện (ví dụ shot đầu tiên) để user xem thử
        rep_shot = c_shots[0]
        cluster_reports.append({
            "cluster_id": c_id,
            "rep_sample_shot": rep_shot["filename"],
            "rep_duration_sec": rep_shot["duration_sec"],
            "unique_video_count": cl["unique_video_count"],
            "total_shot_count": cl["shot_count"],
            "sample_videos": cl["unique_videos"][:10],
            "shots": c_shots
        })

    # Deduplicate blacklist
    blacklist_shot_files = sorted(list(set(blacklist_shot_files)))
    blacklist_shot_ids = sorted(list(set(blacklist_shot_ids)))

    # Xuất file kết quả
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. File Blacklist tinh gọn để Notebook Vast.ai nạp thẳng vào
    blacklist_file = os.path.join(output_dir, "blacklist_ad_shots.json")
    with open(blacklist_file, "w", encoding="utf-8") as f:
        json.dump(blacklist_shot_files, f, ensure_ascii=False, indent=2)

    # 2. File Báo cáo chi tiết từng cụm quảng cáo để đại ca kiểm duyệt
    report_file = os.path.join(output_dir, "ad_clusters_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "sim_threshold": sim_threshold,
            "min_video_count": min_video_count,
            "total_clusters": len(cluster_reports),
            "total_ad_shots": len(blacklist_shot_files),
            "clusters": cluster_reports
        }, f, ensure_ascii=False, indent=2)

    # 3. File text tóm tắt trực quan để đọc nhanh
    summary_file = os.path.join(output_dir, "ad_clusters_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"BÁO CÁO CÁC CỤM QUẢNG CÁO / BUMPER TƯƠNG ĐỒNG >= {sim_threshold}\n")
        f.write(f"Tổng số cụm phát hiện: {len(cluster_reports)} | Tổng số shot quảng cáo: {len(blacklist_shot_files):,}\n")
        f.write("=" * 80 + "\n\n")
        for cl in cluster_reports:
            f.write(f"📌 [CỤM #{cl['cluster_id']:03d}] {cl['rep_sample_shot']} ({cl['rep_duration_sec']}s)\n")
            f.write(f"   • Xuất hiện ở: {cl['unique_video_count']} video khác nhau (Tổng cộng {cl['total_shot_count']} shots)\n")
            f.write(f"   • Các video tiêu biểu: {', '.join(cl['sample_videos'][:8])}...\n\n")

    print("\n" + "=" * 75)
    print("🎉 QUÉT QUẢNG CÁO HOÀN TẤT THÀNH CÔNG!")
    print(f"  • Số cụm quảng cáo/bumper tìm thấy: {len(cluster_reports)}")
    print(f"  • Tổng số shot quảng cáo phát hiện: {len(blacklist_shot_files):,} / {total_shots:,} shots ({len(blacklist_shot_files)/total_shots*100:.1f}%)")
    print(f"  • File Blacklist cho Notebook:      {blacklist_file}")
    print(f"  • File Báo cáo chi tiết kiểm duyệt: {report_file}")
    print(f"  • File Tóm tắt dễ đọc:              {summary_file}")
    print("=" * 75)

if __name__ == "__main__":
    detect_cross_video_ads()
