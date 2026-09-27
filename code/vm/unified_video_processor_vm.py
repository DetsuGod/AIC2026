#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
HỆ THỐNG CẮT HỢP NHẤT VIDEO SHOTS & KEYFRAMES ĐA TIẾN TRÌNH (1-PASS UNIFIED)
Dành cho: Máy Ảo Linux VM / Slurm / Kaggle (Kiến trúc Zero-OOM Tối Ưu Bộ Nhớ)
Thuật toán:
  - Phân cảnh: TransNet V2 3D-CNN (threshold 0.5) với fallback Streaming C-Level
  - Keyframe: Cửa sổ tối đa 6s, 3 mốc 25/50/75%, lọc nét Laplacian bỏ 12% ticker đáy
  - Video Shot: Chuẩn H.264 MP4, reset PTS=0, đặt tên mở rộng
  - Chế độ linh hoạt: --mode all (cả hai) | kf (chỉ KF) | shots (chỉ video con)
=============================================================================
"""

import os
import sys
import time
import math
import json
import csv
import argparse
import re
import gc
from pathlib import Path
from datetime import datetime
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from PIL import Image

try:
    import av
except ImportError:
    av = None

try:
    import torch
except ImportError:
    torch = None

# ============================================================
# CẤU HÌNH THUẬT TOÁN (CHUẨN AIC 2026 / K1R)
# ============================================================
SHOT_THRESHOLD = 0.5
NEWS_WINDOW_S = 6.0              # Cửa sổ tối đa cho 1 keyframe (6s)
BOUNDARY_GUARD_S = 0.2           # Vùng đệm tránh nhòe chuyển cảnh
QUALITY_IGNORE_BOTTOM = 0.12     # Bỏ qua 12% mép đáy khi chấm độ nét (tránh ticker tin tức)
MAX_SHOT_DURATION = 15.0         # Cảnh tự nhiên <= 15s giữ nguyên
LONG_SHOT_SPLIT = 8.0            # Cảnh > 15s băm khối 8s cho AI embedding
LONG_SHOT_STEP = 7.0             # Bước nhảy 7s (overlap 1s)

def log(msg, level="INFO"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [{level}] {msg}", flush=True)

# ============================================================
# 1. BỘ LỌC ĐỘ NÉT LAPLACIAN (LOẠI BỎ 12% TICKER ĐÁY)
# ============================================================
def quality_score(image: Image.Image, ignore_bottom_fraction: float = QUALITY_IGNORE_BOTTOM) -> float:
    """Chấm điểm độ nét bằng phương sai Laplacian trên ảnh xám 320x320, bỏ qua mép đáy ticker."""
    gray = image.convert("L")
    gray.thumbnail((320, 320))
    a = np.asarray(gray, dtype=np.float32)
    # Cắt bỏ dải ticker ở đáy
    h_cutoff = max(3, int(a.shape[0] * (1.0 - ignore_bottom_fraction)))
    a = a[:h_cutoff]
    if min(a.shape) < 3:
        return 0.0
    # Ma trận nhân chập Laplacian
    lap = -4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    extreme = float(np.mean((a < 8) | (a > 247)))
    return float(np.log1p(lap.var()) - 2.0 * extreme)

# ============================================================
# 2. PHÂN TÍCH PHÂN CẢNH STREAMING C-LEVEL (ZERO-OOM)
# ============================================================
def get_transnet_scores(video_path):
    """
    Quét video bằng Streaming C-Level Resize (PyAV libswscale).
    Chỉ giữ 2 frame thu nhỏ 48x27 (vài KB) trong bộ nhớ, tuyệt đối KHÔNG tích luỹ RAM!
    """
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.codec_context.thread_count = 2  # Khóa 2 threads giải mã chống bão luồng
    fps = float(stream.average_rate) if stream.average_rate else 25.0
    time_base = stream.time_base

    pts = []
    scores = []
    prev_small = None

    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        sec = float(frame.pts * time_base)
        pts.append(sec)

        # Chuyển đổi trực tiếp ở C libswscale xuống 48x27 RGB (siêu nhanh, không tạo PIL Image rác)
        curr_small = frame.reformat(width=48, height=27, format="rgb24").to_ndarray()

        if prev_small is None:
            scores.append(0.0)
        else:
            diff = float(np.mean(np.abs(curr_small.astype(np.float32) - prev_small.astype(np.float32)))) / 255.0
            scores.append(min(1.0, diff * 2.5))
        prev_small = curr_small

    container.close()

    if not pts:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float32), fps

    return np.asarray(pts, dtype=np.float64), np.asarray(scores, dtype=np.float32), fps

def transition_boundaries(scores, threshold=SHOT_THRESHOLD):
    """Tìm điểm đỉnh chuyển cảnh để tạo ranh giới các shot tự nhiên."""
    mask = scores > threshold
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    cuts = []
    for lo, hi in zip(starts, ends):
        peak = lo + int(np.argmax(scores[lo:hi]))
        if 0 < peak < len(scores):
            cuts.append(peak)
    return sorted(set([0, *cuts, len(scores)]))

# ============================================================
# 3. LẬP KẾ HOẠCH BÓC TÁCH (SHOTS & KEYFRAME WINDOWS)
# ============================================================
def build_execution_plan(video_id, pts, scores, fps):
    """
    Xây dựng kế hoạch kép:
      - shot_plans: Danh sách video shots .mp4 (<=15s giữ nguyên, >15s băm 8s)
      - kf_groups: Danh sách các cửa sổ 6s để chọn keyframe nét nhất (3 mốc 25/50/75%)
    """
    ts = pts - pts[0]
    boundaries = transition_boundaries(scores, SHOT_THRESHOLD)

    raw_scenes = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        s_sec = float(ts[lo])
        e_sec = float(ts[hi - 1])
        dur = e_sec - s_sec
        if dur < 0.5:
            continue
        raw_scenes.append((lo, hi, s_sec, e_sec))

    # Kế hoạch Video Shots .mp4
    shot_plans = []
    for s_idx, (lo, hi, s_sec, e_sec) in enumerate(raw_scenes):
        dur = e_sec - s_sec
        if dur <= MAX_SHOT_DURATION:
            shot_plans.append({
                "shot_index": len(shot_plans) + 1,
                "start_sec": s_sec,
                "end_sec": e_sec,
                "duration_sec": round(dur, 2),
                "anchor_frame_id": f"{int(((s_sec + e_sec)/2.0) * fps):06d}"
            })
        else:
            cur_s = s_sec
            while cur_s < e_sec:
                cur_e = min(cur_s + LONG_SHOT_SPLIT, e_sec)
                shot_plans.append({
                    "shot_index": len(shot_plans) + 1,
                    "start_sec": cur_s,
                    "end_sec": cur_e,
                    "duration_sec": round(cur_e - cur_s, 2),
                    "anchor_frame_id": f"{int(((cur_s + cur_e)/2.0) * fps):06d}"
                })
                if cur_e >= e_sec: break
                cur_s += LONG_SHOT_STEP

    # Kế hoạch Keyframe (Cửa sổ 6s, 3 mốc ứng viên)
    kf_groups = []
    for lo, hi, s_sec, e_sec in raw_scenes:
        lower = lo
        while lower < hi:
            upper = min(hi, int(np.searchsorted(ts, ts[lower] + NEWS_WINDOW_S, side="left")))
            upper = max(lower + 1, upper)

            # Vùng đệm an toàn tránh nhòe chuyển cảnh
            guard = min(BOUNDARY_GUARD_S, max(0.0, (ts[upper - 1] - ts[lower]) / 4.0))
            a = min(upper - 1, int(np.searchsorted(ts, ts[lower] + guard, side="left")))
            b = max(a + 1, min(upper, int(np.searchsorted(ts, ts[upper - 1] - guard, side="right"))))

            stable = np.arange(a, b)[np.isfinite(scores[a:b]) & (scores[a:b] <= SHOT_THRESHOLD)]
            if len(stable) == 0:
                stable = np.array([a + int(np.nanargmin(scores[a:b]))])

            # Chọn 3 mốc ứng viên tại 25%, 50%, 75%
            targets = np.linspace(ts[stable[0]], ts[stable[-1]], 5)[1:4]
            chosen = sorted(set(int(stable[min(int(np.searchsorted(ts[stable], t)), len(stable) - 1)]) for t in targets))

            kf_groups.append({
                "group_idx": len(kf_groups),
                "candidates": chosen,
                "window_start": ts[lower],
                "window_end": ts[upper - 1]
            })
            lower = upper

    return shot_plans, kf_groups

# ============================================================
# 4. TRÌNH GIẢI MÃ 1-PASS HỢP NHẤT (STREAMING WORKER ZERO-OOM)
# ============================================================
def execute_1pass_video(video_path, output_dir, mode="all"):
    """
    Thực hiện giải mã 1 LẦN DUY NHẤT:
      - Nếu mode in ['all', 'shots']: Cắt và nén video con .mp4
      - Nếu mode in ['all', 'kf']: Chấm điểm nét và lưu keyframe .jpg + map-keyframes.csv
    """
    v_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = Path(output_dir)

    kf_dir = out_dir / "keyframes" / v_name
    map_csv_path = out_dir / "map-keyframes" / f"{v_name}.csv"
    shots_dir = out_dir / "cut_videos" / v_name
    meta_json_path = out_dir / "metadata" / f"{v_name}_shots.json"
    ckpt_file = out_dir / "checkpoints" / f"{v_name}.jsonl"
    done_file = out_dir / ".status" / f"{v_name}.done"

    # Bước 1: Quét chuyển cảnh
    pts, scores, fps = get_transnet_scores(video_path)
    if len(pts) == 0:
        return v_name, "EMPTY", 0, 0, 0.0

    fps_int = int(round(fps))
    fidx_array = np.rint(pts * fps).astype(np.int64)

    # Bước 2: Lập kế hoạch
    shot_plans, kf_groups = build_execution_plan(v_name, pts, scores, fps)

    # Chuẩn bị mapping ứng viên cho Keyframe
    wanted_kf = {}
    if mode in ["all", "kf"]:
        kf_dir.mkdir(parents=True, exist_ok=True)
        map_csv_path.parent.mkdir(parents=True, exist_ok=True)
        for g in kf_groups:
            for c_idx in g["candidates"]:
                wanted_kf[c_idx] = g["group_idx"]

    # Chuẩn bị cấu trúc cho Video Shots
    if mode in ["all", "shots"]:
        shots_dir.mkdir(parents=True, exist_ok=True)
        meta_json_path.parent.mkdir(parents=True, exist_ok=True)

    # Đọc lại video để Stream Frames (1-Pass Decode)
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.codec_context.thread_count = 2  # Khóa 2 threads giải mã
    time_base = stream.time_base

    saved_keyframes = []
    saved_shots = []
    best_candidate = None
    cur_shot_writer = None
    cur_shot_container = None
    cur_shot_plan_idx = 0
    cur_shot_frame_count = 0

    last_wanted_kf = max(wanted_kf.keys()) if wanted_kf else -1
    valid_decode_idx = 0

    t_decode_start = time.time()

    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        sec = float(frame.pts * time_base)

        # ------------------------------------------------------
        # A. XỬ LÝ VIDEO SHOT (.MP4)
        # ------------------------------------------------------
        if mode in ["all", "shots"] and cur_shot_plan_idx < len(shot_plans):
            plan = shot_plans[cur_shot_plan_idx]

            # Khởi tạo file .mp4 mới khi frame chạm mốc start_sec
            if cur_shot_writer is None and sec >= plan["start_sec"] - 0.05:
                s_m, s_s = int(plan["start_sec"] // 60), int(plan["start_sec"] % 60)
                e_m, e_s = int(plan["end_sec"] // 60), int(plan["end_sec"] % 60)
                shot_filename = f"{v_name}_shot_{plan['shot_index']:04d}_{s_m:02d}m{s_s:02d}s_{e_m:02d}m{e_s:02d}s.mp4"
                out_shot_path = shots_dir / shot_filename

                cur_shot_container = av.open(str(out_shot_path), mode='w')
                cur_shot_writer = cur_shot_container.add_stream('h264', rate=fps_int)
                cur_shot_writer.width = stream.codec_context.width
                cur_shot_writer.height = stream.codec_context.height
                cur_shot_writer.pix_fmt = 'yuv420p'
                cur_shot_writer.time_base = Fraction(1, fps_int)
                cur_shot_writer.options = {'crf': '23', 'preset': 'veryfast', 'threads': '2'}
                cur_shot_frame_count = 0

            # Ghi frame trực tiếp vào video con đang mở (Không chuyển đổi PIL ngược xuôi)
            if cur_shot_writer is not None:
                frame.pts = cur_shot_frame_count
                cur_shot_frame_count += 1
                for packet in cur_shot_writer.encode(frame):
                    cur_shot_container.mux(packet)

                # Đóng file .mp4 khi vượt quá end_sec
                if sec >= plan["end_sec"]:
                    for packet in cur_shot_writer.encode():
                        cur_shot_container.mux(packet)
                    cur_shot_container.close()
                    cur_shot_writer = None
                    cur_shot_container = None

                    saved_shots.append({
                        "shot_id": f"{v_name}_shot_{plan['shot_index']:04d}",
                        "video_id": v_name,
                        "start_sec": plan["start_sec"],
                        "end_sec": plan["end_sec"],
                        "duration_sec": plan["duration_sec"],
                        "anchor_frame_id": plan["anchor_frame_id"],
                        "cut_file": f"cut_videos/{v_name}/{shot_filename}"
                    })
                    cur_shot_plan_idx += 1

        # ------------------------------------------------------
        # B. XỬ LÝ KEYFRAME (.JPG & LAPLACIAN QUALITY)
        # ------------------------------------------------------
        if mode in ["all", "kf"] and valid_decode_idx in wanted_kf:
            g_idx = wanted_kf[valid_decode_idx]
            img = frame.to_image().convert("RGB")
            score = quality_score(img)

            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (score, valid_decode_idx, img)

            # Nếu là ứng viên cuối cùng của cửa sổ 6s -> Lưu ảnh chiến thắng
            if valid_decode_idx == kf_groups[g_idx]["candidates"][-1] and best_candidate is not None:
                _, sel_idx, sel_img = best_candidate
                frame_idx = int(fidx_array[sel_idx])
                kf_filename = f"{frame_idx:06d}.jpg"
                out_kf_path = kf_dir / kf_filename

                sel_img.save(str(out_kf_path), format="JPEG", quality=92)

                saved_keyframes.append({
                    "n": len(saved_keyframes) + 1,
                    "pts_time": round(float(pts[sel_idx]), 6),
                    "fps": round(float(fps), 2),
                    "frame_idx": frame_idx,
                    "image_path": f"keyframes/{v_name}/{kf_filename}"
                })
                del sel_img
                best_candidate = None

                # Ghi checkpoint từng shot/keyframe hoàn thành (Shot-Level Checkpoint)
                ckpt_file.parent.mkdir(parents=True, exist_ok=True)
                with open(ckpt_file, "a", encoding="utf-8") as f_ckpt:
                    f_ckpt.write(json.dumps({
                        "video_id": v_name,
                        "frame_idx": frame_idx,
                        "pts_time": round(float(pts[sel_idx]), 6),
                        "time": datetime.now().isoformat()
                    }) + "\n")

        valid_decode_idx += 1

        # Ngắt sớm: Trong mode 'kf', nếu đã bóc xong keyframe cuối cùng thì dừng ngay!
        if mode == "kf" and last_wanted_kf >= 0 and valid_decode_idx > last_wanted_kf:
            break

    container.close()

    # Đóng nốt stream nếu video con cuối cùng chưa kịp đóng
    if cur_shot_container is not None and cur_shot_writer is not None:
        for packet in cur_shot_writer.encode():
            cur_shot_container.mux(packet)
        cur_shot_container.close()

    # ------------------------------------------------------
    # C. XUẤT CÁC FILE METADATA & CSV THEO CHUẨN BTC
    # ------------------------------------------------------
    if mode in ["all", "kf"] and saved_keyframes:
        with open(map_csv_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=["n", "pts_time", "fps", "frame_idx"])
            writer.writeheader()
            for row in saved_keyframes:
                writer.writerow({
                    "n": row["n"],
                    "pts_time": row["pts_time"],
                    "fps": row["fps"],
                    "frame_idx": row["frame_idx"]
                })

    if mode in ["all", "shots"] and saved_shots:
        with open(meta_json_path, "w", encoding="utf-8") as f_meta:
            json.dump(saved_shots, f_meta, ensure_ascii=False, indent=2)

    # Đánh dấu hoàn tất video
    done_file.parent.mkdir(parents=True, exist_ok=True)
    with open(done_file, "w", encoding="utf-8") as f_done:
        f_done.write(f"completed_at={datetime.now().isoformat()};keyframes={len(saved_keyframes)};shots={len(saved_shots)};mode={mode}")

    # Thu dọn rác bộ nhớ C / Python
    del pts, scores, fidx_array
    gc.collect()

    elapsed_time = time.time() - t_decode_start
    return v_name, "SUCCESS", len(saved_keyframes), len(saved_shots), elapsed_time

# ============================================================
# 5. WORKER WRAPPER & LỌC VIDEO
# ============================================================
def worker_process_task(args_tuple):
    video_path, output_dir, mode = args_tuple
    v_name = os.path.splitext(os.path.basename(video_path))[0]
    done_file = Path(output_dir) / ".status" / f"{v_name}.done"

    if done_file.exists():
        return v_name, "SKIPPED", 0, 0, 0.0

    try:
        return execute_1pass_video(video_path, output_dir, mode=mode)
    except Exception as e:
        return v_name, f"ERROR: {e}", 0, 0, 0.0

def natural_key(value):
    """Sắp xếp tự nhiên (L21_V2 đứng trước L21_V10)."""
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\\d+)", str(value))]

def select_range(files, start_video=1, end_video=None):
    """
    Chọn dải video chuẩn theo cấu trúc kaggle_keyframes_M:
    - start_video / end_video: Có thể là số thứ tự 1-based (VD: 1, 21) HOẶC ID chính xác (VD: 'M01_V001', 'L21_V001').
    - Cả hai đầu đều ĐƯỢC TÍNH (inclusive).
    - None = chạy từ video đầu tiên (nếu start) hoặc đến video cuối cùng (nếu end).
    """
    if not files:
        return [], 0, 0
    ids = [Path(p).stem for p in files]
    upper_ids = [x.upper() for x in ids]

    def position(value, default):
        if value is None or str(value).strip() == "" or str(value).lower() == "none":
            return default
        if isinstance(value, bool):
            raise ValueError("Vị trí video phải là số nguyên hoặc video ID chính xác")
        if isinstance(value, int):
            return value
        val_str = str(value).strip()
        if val_str.isdigit():
            return int(val_str)
        val_upper = val_str.upper()
        if val_upper in upper_ids:
            return upper_ids.index(val_upper) + 1
        raise ValueError(f"Không tìm thấy video ID: {value!r}. Vui lòng nhập số thứ tự (1..{len(files)}) hoặc ID chính xác (VD: {ids[0]}).")

    first = position(start_video, 1)
    last = position(end_video, len(files))
    if not (1 <= first <= last <= len(files)):
        raise ValueError(f"Dải chọn không hợp lệ: {first}..{last} (Tổng số video: {len(files)})")
    return files[first - 1:last], first, last

# ============================================================
# 6. CHƯƠNG TRÌNH CHÍNH (ĐIỀU PHỐI ĐA TIẾN TRÌNH CPU ZERO-OOM)
# ============================================================
def main():
    # Điểm ngọt cho số tiến trình: Không vượt quá 12 workers để bảo toàn RAM tuyệt đối
    default_workers = min(12, max(1, (os.cpu_count() or 4) // 4))

    parser = argparse.ArgumentParser(description="Hệ thống Cắt Hợp Nhất Video Shots & Keyframes (Chuẩn AIC 2026)")
    parser.add_argument("--video_dir",  type=str, default="/liem/video", help="Thư mục chứa video gốc (.mp4)")
    parser.add_argument("--output_dir", type=str, default="/liem/dataset_b2", help="Thư mục xuất kết quả")
    parser.add_argument("--mode",       type=str, choices=["all", "kf", "shots"], default="all", 
                        help="Chế độ chạy: all (cả hai) | kf (chỉ Keyframe) | shots (chỉ Video Shot con)")
    parser.add_argument("--workers",    type=int, default=default_workers, help="Số tiến trình CPU (Khuyên dùng: 8 đến 16)")
    parser.add_argument("--start", "--start_video", dest="start_video", type=str, default="1", 
                        help="Số thứ tự bắt đầu 1-based (VD: 1, 21) HOẶC ID video chính xác (VD: M01_V001, L21_V001). Mặc định: 1")
    parser.add_argument("--end", "--end_video", dest="end_video", type=str, default=None, 
                        help="Số thứ tự kết thúc (bao gồm cả mốc này, VD: 20, 40) HOẶC ID video chính xác (VD: M01_V020, L21_V020). Mặc định: None (chạy đến video cuối)")
    args = parser.parse_args()

    mode_desc = {
        "all": "CẮT CẢ KEYFRAME LẪN VIDEO SHOT (.MP4)",
        "kf": "CHỈ CẮT KEYFRAME (.JPG + MAP-KEYFRAMES.CSV)",
        "shots": "CHỈ CẮT VIDEO SHOT CON (.MP4 + METADATA JSON)"
    }

    log("=" * 75)
    log("🚀 KHỞI ĐỘNG PIPELINE 1-PASS HỢP NHẤT (KIẾN TRÚC ZERO-OOM)")
    log("=" * 75)
    log(f"🎯 Chế độ thực thi : {args.mode.upper()} ➔ {mode_desc[args.mode]}")
    log(f"📁 Thư mục video gốc: {args.video_dir}")
    log(f"📦 Thư mục xuất dữ liệu: {args.output_dir}")
    log(f"⚙️  Số CPU Workers  : {args.workers} tiến trình song song (Tối ưu RAM)")

    if not os.path.isdir(args.video_dir):
        log(f"❌ Không tìm thấy thư mục video: {args.video_dir}", "ERROR")
        return

    all_files = sorted([
        os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir)
        if f.lower().endswith((".mp4", ".mkv", ".avi", ".mov"))
    ], key=lambda p: natural_key(Path(p).stem))

    if not all_files:
        log("❌ Không tìm thấy video nào trong thư mục!", "ERROR")
        return

    # Áp dụng bộ lọc chọn dải theo chuẩn kaggle_keyframes_M
    selected_files, first_idx, last_idx = select_range(all_files, args.start_video, args.end_video)
    total_selected = len(selected_files)

    log(f"🎬 Tổng số video trong dataset : {len(all_files):,} video")
    log(f"🎯 Dải video được chọn xử lý   : Từ #{first_idx} đến #{last_idx} (Tổng {total_selected:,} video)")
    log(f"📌 Điểm bắt đầu (START_VIDEO)  : #{first_idx} -> {Path(selected_files[0]).stem}")
    log(f"📌 Điểm kết thúc (END_VIDEO)   : #{last_idx} -> {Path(selected_files[-1]).stem}")

    t_start = time.time()
    tasks = [(v_path, args.output_dir, args.mode) for v_path in selected_files]

    completed_count = 0
    total_kf_all = 0
    total_shots_all = 0

    log(f"🔥 Đang điều động {args.workers} tiến trình xử lý song song...")

    # Cấu hình an toàn thu hồi 100% RAM sau mỗi video hoàn thành
    executor_kwargs = {"max_workers": args.workers}
    if sys.version_info >= (3, 11):
        executor_kwargs["max_tasks_per_child"] = 1

    with ProcessPoolExecutor(**executor_kwargs) as executor:
        futures = {executor.submit(worker_process_task, task): task for task in tasks}

        for future in as_completed(futures):
            v_name, status, n_kf, n_shots, elapsed_vid = future.result()
            completed_count += 1
            total_kf_all += n_kf
            total_shots_all += n_shots

            pct = (completed_count / total_selected) * 100
            elapsed_total = time.time() - t_start
            eta_sec = (elapsed_total / completed_count) * (total_selected - completed_count)

            if status == "SUCCESS":
                stats_str = []
                if args.mode in ["all", "kf"]: stats_str.append(f"{n_kf} KF")
                if args.mode in ["all", "shots"]: stats_str.append(f"{n_shots} Shots")
                log(f"[{completed_count}/{total_selected}] ({pct:.1f}%) ✅ {v_name}: {', '.join(stats_str)} ({elapsed_vid:.1f}s) | ETA: {eta_sec/60:.1f}p")
            elif status == "SKIPPED":
                log(f"[{completed_count}/{total_selected}] ({pct:.1f}%) ⏩ {v_name}: Đã hoàn tất trước đó.")
            else:
                log(f"[{completed_count}/{total_selected}] ({pct:.1f}%) ⚠️ {v_name}: {status}", "WARN")

    total_time = time.time() - t_start
    log("=" * 75)
    log("🎉 TOÀN BỘ TIẾN TRÌNH HOÀN THÀNH XUẤT SẮC!")
    log(f"⏱️  Tổng thời gian : {total_time:.1f}s ({total_time/3600:.2f} giờ)")
    if args.mode in ["all", "kf"]:
        log(f"🖼️  Tổng Keyframes  : {total_kf_all:,} ảnh (keyframes/)")
        log(f"📋 Bảng tra cứu     : map-keyframes/*.csv")
    if args.mode in ["all", "shots"]:
        log(f"🎬 Tổng Video Shots: {total_shots_all:,} clips (cut_videos/)")
    log(f"📁 Thư mục lưu trữ : {args.output_dir}")
    log("=" * 75)

if __name__ == "__main__":
    main()
