import os
import sys
import time
import psutil
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import av
from datetime import datetime

# ============================================================
# CẤU HÌNH & TỐI ĐA HÓA PHẦN CỨNG (CPU & GPU)
# ============================================================
# 1. Ép PyTorch tận dụng toàn bộ số luồng CPU (AMD Ryzen 7 5800H: 8 cores, 16 threads)
NUM_THREADS = psutil.cpu_count(logical=True) or 16
torch.set_num_threads(NUM_THREADS)
try:
    torch.set_num_interop_threads(4)
except RuntimeError:
    pass

# 2. Bật tối ưu hóa Ampere Tensor Cores & cuDNN
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Các đường dẫn file test theo yêu cầu
KF_PATH = r"l:\Competitions\2026-AIC\Data\batch 1\Custom_Keyframes\L21_V001\000346.jpg"
SHOT_PATH = r"l:\Competitions\2026-AIC\Data\batch 1\output-video-encode\cut_videos\L21_V001\L21_V001_shot_0015_00m44s_00m52s.mp4"
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_RUNS = 2
BENCHMARK_RUNS = 10

def log(msg, level="INFO"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{now}] [{level}] {msg}", flush=True)

def log_sep(title=""):
    line = "=" * 70
    print(f"\n{line}", flush=True)
    if title:
        print(f">>> {title}", flush=True)
        print(f"{line}", flush=True)

def print_hardware_info():
    log_sep("THÔNG TIN PHẦN CỨNG HỆ THỐNG LOCAL")
    log(f"OS: {sys.platform} | Python: {sys.version.split()[0]}")
    log(f"CPU: {psutil.cpu_count(logical=False)} Cores, {NUM_THREADS} Logical Threads (Active PyTorch threads: {torch.get_num_threads()})")
    ram = psutil.virtual_memory()
    log(f"RAM: Total {ram.total / 1024**3:.2f} GB | Available {ram.available / 1024**3:.2f} GB")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log(f"GPU: {gpu_name} ({vram_total:.2f} GB VRAM)")
        log(f"PyTorch CUDA: {torch.__version__} | TF32: {torch.backends.cuda.matmul.allow_tf32} | BF16 Supported: {torch.cuda.is_bf16_supported()}")
    else:
        log("GPU: KHÔNG TÌM THẤY CUDA! Chạy trên CPU.", "WARN")

# ============================================================
# PART 1: BENCHMARK 1 KEYFRAME (extract_qwen_slurm.py)
# ============================================================
def benchmark_keyframe():
    log_sep("PHẦN 1: BENCHMARK 1 KEYFRAME (Theo extract_qwen_slurm.py)")
    log(f"File ảnh mẫu: {KF_PATH}")
    if not os.path.isfile(KF_PATH):
        log(f"File ảnh không tồn tại: {KF_PATH}", "ERROR")
        return None

    # Load SentenceTransformer đúng như extract_qwen_slurm.py
    from sentence_transformers import SentenceTransformer
    
    log(f"Đang nạp SentenceTransformer: {MODEL_NAME} ...")
    t_load_start = time.time()
    st_model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=DEVICE)
    t_load = time.time() - t_load_start
    log(f"✅ Nạp SentenceTransformer hoàn tất trong {t_load:.2f}s")

    # Đọc ảnh để chuẩn bị test
    t_io_start = time.time()
    img = Image.open(KF_PATH).convert("RGB")
    t_io = (time.time() - t_io_start) * 1000
    log(f"Đọc ảnh & convert RGB: {img.size} trong {t_io:.2f} ms")

    # Warmup runs
    log(f"Bắt đầu Warmup ({WARMUP_RUNS} lần) ...")
    for _ in range(WARMUP_RUNS):
        _ = st_model.encode([img], batch_size=1, show_progress_bar=False, convert_to_numpy=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark runs
    log(f"Chạy Benchmark chính thức ({BENCHMARK_RUNS} lần lặp) ...")
    torch.cuda.reset_peak_memory_stats()
    times_ms = []
    sample_vec = None

    for r in range(BENCHMARK_RUNS):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        vec = st_model.encode([img], batch_size=1, show_progress_bar=False, convert_to_numpy=True)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        dt_ms = (t1 - t0) * 1000
        times_ms.append(dt_ms)
        sample_vec = vec[0]

    mean_ms = np.mean(times_ms)
    min_ms = np.min(times_ms)
    max_ms = np.max(times_ms)
    std_ms = np.std(times_ms)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    norm_val = np.linalg.norm(sample_vec)

    log(f"Kết quả Keyframe (SentenceTransformer):")
    log(f"  Shape Vector  : {sample_vec.shape} (L2-norm: {norm_val:.4f})")
    log(f"  Thời gian TB  : {mean_ms:.2f} ms ± {std_ms:.2f} ms (Min: {min_ms:.2f} ms | Max: {max_ms:.2f} ms)")
    log(f"  Tốc độ xử lý  : {1000 / mean_ms:.2f} Keyframes / giây (~{ (1000 / mean_ms) * 3600:,.0f} Keyframes / giờ)")
    log(f"  Peak VRAM     : {peak_vram_mb:.1f} MB")

    # Giải phóng st_model để test phần 2
    del st_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms,
        "fps": 1000 / mean_ms,
        "peak_vram_mb": peak_vram_mb,
        "dim": sample_vec.shape[0]
    }

# ============================================================
# PART 2: BENCHMARK 1 SHOT VIDEO (extract_video_shots_bidirectional_slurm.py)
# ============================================================
def benchmark_shot_video():
    log_sep("PHẦN 2: BENCHMARK 1 SHOT VIDEO (Theo extract_video_shots_bidirectional_slurm.py)")
    log(f"File video shot mẫu: {SHOT_PATH}")
    if not os.path.isfile(SHOT_PATH):
        log(f"File shot không tồn tại: {SHOT_PATH}", "ERROR")
        return None

    from transformers import AutoModel, AutoProcessor
    from qwen_vl_utils import process_vision_info

    log(f"Đang nạp AutoModel & AutoProcessor: {MODEL_NAME} (dtype=FP16) ...")
    t_load_start = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    qwen_model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        attn_implementation="sdpa"
    ).to(DEVICE)
    qwen_model.eval()
    t_load = time.time() - t_load_start
    log(f"✅ Nạp AutoModel hoàn tất trong {t_load:.2f}s")

    # 1. Bóc tách thời gian giải mã video và trích mẫu frames (PyAV)
    log_sep("2.1. ĐO THỜI GIAN GIẢI MÃ VIDEO (PyAV)")
    decode_times = []
    sampled_frames = []
    
    for _ in range(5):
        t0 = time.perf_counter()
        container = av.open(SHOT_PATH)
        stream = container.streams.video[0]
        raw_frames = []
        for frame in container.decode(stream):
            if frame.pts is not None:
                raw_frames.append(frame.to_image().convert("RGB"))
        container.close()
        
        # Theo script slurm: sample_count = 6 nếu <= 5.0s, ngược lại 8
        sample_count = 8
        indices = np.linspace(0, len(raw_frames) - 1, min(sample_count, len(raw_frames)), dtype=int)
        sampled_frames = [raw_frames[i] for i in indices]
        
        t1 = time.perf_counter()
        decode_times.append((t1 - t0) * 1000)

    mean_decode_ms = np.mean(decode_times)
    log(f"Giải mã {len(raw_frames)} frames -> Trích mẫu {len(sampled_frames)} frames:")
    log(f"  Thời gian TB  : {mean_decode_ms:.2f} ms (Min: {np.min(decode_times):.2f} ms | Max: {np.max(decode_times):.2f} ms)")

    # 2. Xây dựng message và đo thời gian Vision Preprocessing
    log_sep("2.2. ĐO THỜI GIAN VISION TOKENIZER & PROCESSOR")
    batch_msgs = [[{"role": "user", "content": [{"type": "video", "video": sampled_frames, "fps": 1.5}]}]]
    
    prep_times = []
    inputs = None
    for _ in range(5):
        t0 = time.perf_counter()
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in batch_msgs]
        all_imgs, all_vids = [], []
        for m in batch_msgs:
            i_inp, v_inp = process_vision_info(m)
            if i_inp: all_imgs.extend(i_inp)
            if v_inp: all_vids.extend(v_inp)
        inputs = processor(
            text=texts,
            images=all_imgs if all_imgs else None,
            videos=all_vids if all_vids else None,
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)
        t1 = time.perf_counter()
        prep_times.append((t1 - t0) * 1000)

    mean_prep_ms = np.mean(prep_times)
    input_ids_len = inputs.input_ids.shape[1]
    log(f"Độ dài sequence sau khi tokenizer: {input_ids_len} tokens")
    log(f"  Thời gian TB  : {mean_prep_ms:.2f} ms")

    # 3. Đo thời gian GPU Forward Pass & Pooling (L2 Normalize)
    log_sep("2.3. ĐO THỜI GIAN GPU MODEL FORWARD & POOLING")
    # Warmup
    for _ in range(WARMUP_RUNS):
        with torch.no_grad():
            out = qwen_model(**inputs)
            emb = out.last_hidden_state[:, -1, :]
            _ = F.normalize(emb, p=2, dim=1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark GPU forward
    torch.cuda.reset_peak_memory_stats()
    forward_times = []
    final_vec = None

    for _ in range(BENCHMARK_RUNS):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            out = qwen_model(**inputs)
            emb = out.last_hidden_state[:, -1, :]
            emb = F.normalize(emb, p=2, dim=1)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        forward_times.append((t1 - t0) * 1000)
        final_vec = emb.cpu().to(torch.float32).numpy()[0]

    mean_forward_ms = np.mean(forward_times)
    min_forward_ms = np.min(forward_times)
    max_forward_ms = np.max(forward_times)
    std_forward_ms = np.std(forward_times)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    norm_val = np.linalg.norm(final_vec)

    log(f"Kết quả GPU Forward & Pooling:")
    log(f"  Shape Vector  : {final_vec.shape} (L2-norm: {norm_val:.4f})")
    log(f"  Thời gian TB  : {mean_forward_ms:.2f} ms ± {std_forward_ms:.2f} ms (Min: {min_forward_ms:.2f} ms | Max: {max_forward_ms:.2f} ms)")
    log(f"  Peak VRAM     : {peak_vram_mb:.1f} MB")

    # 4. Đo Full End-to-End Pipeline (Từ file MP4 trên đĩa -> Vector 2048D)
    log_sep("2.4. ĐO FULL PIPELINE END-TO-END CHO 1 SHOT VIDEO")
    e2e_times = []
    for _ in range(BENCHMARK_RUNS):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        # Đọc & Sample
        container = av.open(SHOT_PATH)
        stream = container.streams.video[0]
        raw_frames = [f.to_image().convert("RGB") for f in container.decode(stream) if f.pts is not None]
        container.close()
        indices = np.linspace(0, len(raw_frames) - 1, min(8, len(raw_frames)), dtype=int)
        s_frames = [raw_frames[i] for i in indices]
        
        # Processor
        msgs = [[{"role": "user", "content": [{"type": "video", "video": s_frames, "fps": 1.5}]}]]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in msgs]
        all_imgs, all_vids = [], []
        for m in msgs:
            i_inp, v_inp = process_vision_info(m)
            if i_inp: all_imgs.extend(i_inp)
            if v_inp: all_vids.extend(v_inp)
        inp = processor(text=texts, images=all_imgs or None, videos=all_vids or None, padding=True, return_tensors="pt").to(DEVICE)
        
        # Forward & Pool
        with torch.no_grad():
            o = qwen_model(**inp)
            e = F.normalize(o.last_hidden_state[:, -1, :], p=2, dim=1)
            v = e.cpu().to(torch.float32).numpy()
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        e2e_times.append((t1 - t0) * 1000)

    mean_e2e_ms = np.mean(e2e_times)
    min_e2e_ms = np.min(e2e_times)
    max_e2e_ms = np.max(e2e_times)
    std_e2e_ms = np.std(e2e_times)

    log(f"Tổng kết Full End-to-End (Đọc file -> 8 frames -> Tokenize -> GPU -> Vector):")
    log(f"  Thời gian TB  : {mean_e2e_ms:.2f} ms ± {std_e2e_ms:.2f} ms (tương đương {mean_e2e_ms / 1000:.3f} s / shot)")
    log(f"  Tốc độ xử lý  : {1000 / mean_e2e_ms:.2f} Shots / giây (~{ (1000 / mean_e2e_ms) * 3600:,.0f} Shots / giờ)")

    del qwen_model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mean_decode_ms": mean_decode_ms,
        "mean_prep_ms": mean_prep_ms,
        "mean_forward_ms": mean_forward_ms,
        "mean_e2e_ms": mean_e2e_ms,
        "min_e2e_ms": min_e2e_ms,
        "max_e2e_ms": max_e2e_ms,
        "std_e2e_ms": std_e2e_ms,
        "shots_per_sec": 1000 / mean_e2e_ms,
        "peak_vram_mb": peak_vram_mb,
        "dim": final_vec.shape[0]
    }

# ============================================================
# MAIN
# ============================================================
def main():
    print_hardware_info()
    
    res_kf = benchmark_keyframe()
    res_shot = benchmark_shot_video()
    
    log_sep("BẢNG TỔNG HỢP HIỆU NĂNG EMBEDDING LOCAL (RTX 3070 8GB)")
    if res_kf:
        print(f"| Giai đoạn Keyframe (1 ảnh 720p) | Thời gian (ms) | Tỷ lệ (%) |")
        print(f"| :--- | :--- | :--- |")
        print(f"| Đọc ảnh & Forward GPU (SentenceTransformers) | {res_kf['mean_ms']:.2f} ms | 100.0% |")
        print(f"  => Tốc độ Keyframe: {res_kf['fps']:.2f} KF/giây | {res_kf['fps']*3600:,.0f} KF/giờ | VRAM đỉnh: {res_kf['peak_vram_mb']:.1f} MB\n")
        
    if res_shot:
        total = res_shot['mean_e2e_ms']
        dec_pct = (res_shot['mean_decode_ms'] / total) * 100
        prep_pct = (res_shot['mean_prep_ms'] / total) * 100
        fwd_pct = (res_shot['mean_forward_ms'] / total) * 100
        other_pct = max(0, 100.0 - dec_pct - prep_pct - fwd_pct)
        print(f"| Giai đoạn Shot Video (8 frames, 8.07s) | Thời gian (ms) | Tỷ lệ (%) |")
        print(f"| :--- | :--- | :--- |")
        print(f"| 1. PyAV Giải mã & Sample 8 frames      | {res_shot['mean_decode_ms']:.2f} ms | {dec_pct:.1f}% |")
        print(f"| 2. Vision Tokenizer & Processor        | {res_shot['mean_prep_ms']:.2f} ms | {prep_pct:.1f}% |")
        print(f"| 3. GPU Forward & Last Token Pooling    | {res_shot['mean_forward_ms']:.2f} ms | {fwd_pct:.1f}% |")
        print(f"| 4. Overhead khác (I/O, tensor transfer)| {total - (res_shot['mean_decode_ms'] + res_shot['mean_prep_ms'] + res_shot['mean_forward_ms']):.2f} ms | {other_pct:.1f}% |")
        print(f"| **TỔNG CỘNG END-TO-END**               | **{total:.2f} ms** ({total/1000:.3f} s) | **100.0%** |")
        print(f"  => Tốc độ Shot Video: {res_shot['shots_per_sec']:.2f} Shot/giây | {res_shot['shots_per_sec']*3600:,.0f} Shot/giờ | VRAM đỉnh: {res_shot['peak_vram_mb']:.1f} MB\n")

if __name__ == "__main__":
    main()
