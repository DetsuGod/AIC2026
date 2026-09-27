"""
submission_generator.py — Engine Xử Lý & Sinh File Nộp Bài (Submission Generator) cho AIC 2026.

Chiến thuật phân bổ điểm số (Score Allocation & Risk Hedging):
- Top 1   (1.0đ): Điểm trung tâm (A + B) / 2
- Top 2-5 (0.8đ): 4 mốc phân bổ StepT5 = |B-A|/3 bao quát mút A, B
- Top 6-15(0.6đ): 10 mốc phân bổ StepT15 = |B-A|/9
- Top 16-100: "Bảo hiểm rủi ro" — Tự động bồi các Video tiếp theo từ Search Engine (System Ranking)
              hoặc Interval phụ (second_interval) hoặc linspace nội bộ.

Tuân thủ định dạng CSV & ZIP chuẩn BTC AIC 2026.
"""

import os
import re
import zipfile
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union


def generate_kis_frames(
    A: int,
    B: Optional[int] = None,
    target_count: int = 100,
    second_interval: Optional[Tuple[int, int]] = None
) -> List[int]:
    """
    Sinh danh sách frame theo chiến thuật phân bổ điểm số (Top 1 -> Top 100) cho 1 video.
    """
    if B is None or A == B:
        return [A + 5 * i for i in range(target_count)]

    if A > B:
        A, B = B, A

    L = B - A
    stepT5 = L / 3.0
    stepT15 = L / 9.0

    selected = []
    used = set()

    def add_val(val: float) -> int:
        cand = int(round(val))
        while cand in used:
            cand += 5
        selected.append(cand)
        used.add(cand)
        return cand

    # 1. Top 1 (1.0đ)
    top1 = int(round((A + B) / 2.0))
    if top1 == A or top1 == B:
        top1_cand = top1
        while top1_cand == A or top1_cand == B:
            top1_cand += 5
        top1 = top1_cand

    selected.append(top1)
    used.add(top1)

    # 2. Top 2 - 5 (0.8đ): Bắt buộc chứa 2 đầu mút A và B
    t2 = A
    t3 = int(round(A + stepT5))
    t4 = int(round(A + 2 * stepT5))
    t5 = B

    top2_5_candidates = [t2, t3, t4, t5]
    for idx, cand in enumerate(top2_5_candidates):
        if idx == 0:
            selected.append(A)
            used.add(A)
        elif idx == 3:
            selected.append(B)
            used.add(B)
        else:
            c = cand
            while c in used or c == B:
                c += 1 if c < B else 5
            selected.append(c)
            used.add(c)

    # 3. Top 6 - 15 (0.6đ): 10 frames chia mịn
    for k in range(10):
        add_val(A + k * stepT15)

    # 4. Top 16 - 20 (5 frames): Fill nội bộ theo linspace
    for p in np.linspace(A, B, 5):
        add_val(p)

    # 5. Top 21 - 50 (30 frames): Fill nội bộ
    for p in np.linspace(A, B, 30):
        add_val(p)

    # 6. Top 51 - 100 (50 frames): Fill second_interval (nếu có) hoặc tiếp tục [A, B]
    if second_interval and (B - A + 1) < 50:
        A2, B2 = second_interval
        if A2 > B2:
            A2, B2 = B2, A2
        for p in np.linspace(A2, B2, 50):
            add_val(p)
    else:
        for p in np.linspace(A, B, 50):
            add_val(p)

    # Bù đủ số lượng nếu cần
    while len(selected) < target_count:
        add_val(selected[-1] + 5)

    return selected[:target_count]


def sanitize_qa_answer(answer: Optional[str]) -> Optional[str]:
    """
    Chuẩn hóa câu trả lời QA theo quy chuẩn CSV của BTC AIC 2026:
    - Loại bỏ xuống dòng / ký tự rác
    - Escape double quotes: " -> ""
    - Bọc trong cặp ngoặc kép
    - Tối đa 100 ký tự
    """
    if not answer:
        return None
    ans = answer.replace("\r", " ").replace("\n", " ").strip()
    # Bỏ ngoặc kép ngoài cùng nếu người dùng đã tự gõ
    if (ans.startswith('"') and ans.endswith('"')) or (ans.startswith("'") and ans.endswith("'")):
        ans = ans[1:-1].strip()
    # Escape internal quotes
    ans = ans.replace('"', '""')
    # Giới hạn tối đa 100 ký tự theo quy định BTC
    if len(ans) > 100:
        ans = ans[:100]
    return f'"{ans}"'


def generate_submission_lines(
    mode: str,
    video_id: str,
    A: int,
    B: Optional[int] = None,
    answer: Optional[str] = None,
    second_interval: Optional[Tuple[int, int]] = None,
    system_backup_items: Optional[List[Dict[str, Any]]] = None,
    trake_event_frames: Optional[List[int]] = None,
    target_count: int = 100
) -> List[str]:
    """
    Tạo danh sách các dòng submission hoàn chỉnh cho 1 câu truy vấn (KIS, QA hoặc TRAKE).
    
    Hỗ trợ Smart System Fill (Risk Hedging):
    - Top 1 - 15: Dồn toàn lực cho Video mục tiêu [A-B]
    - Top 16 - 100: Bồi các frame/video xếp hạng tiếp theo từ system_backup_items (nếu có)
    """
    mode = mode.lower().strip()
    clean_vid = video_id.strip()
    if clean_vid.endswith(".mp4"):
        clean_vid = clean_vid[:-4]

    # --- CHẾ ĐỘ TRAKE ---
    if mode == "trake":
        if not trake_event_frames or len(trake_event_frames) < 2:
            raise ValueError("Chế độ TRAKE yêu cầu ít nhất 2 Frame ID của chuỗi sự kiện.")
        frame_strs = [str(int(f)) for f in trake_event_frames]
        return [f"{clean_vid},{','.join(frame_strs)}"]

    # --- CHẾ ĐỘ KIS & QA ---
    formatted_answer = sanitize_qa_answer(answer) if mode == "qa" else None

    # Sinh danh sách frame cho Target Video chính
    primary_frames = generate_kis_frames(A, B, target_count=target_count, second_interval=second_interval)

    # Nếu KHÔNG CÓ system_backup_items (chế độ fill thuần túy)
    if not system_backup_items or len(system_backup_items) == 0:
        lines = []
        for f in primary_frames:
            if formatted_answer:
                lines.append(f"{clean_vid},{f},{formatted_answer}")
            else:
                lines.append(f"{clean_vid},{f}")
        return lines

    # NẾU CÓ SYSTEM BACKUP ITEMS (Chế độ Tác Chiến Tích Hợp - Smart Risk Hedging)
    # 1. Dòng 1 -> 15: Lấy từ primary_frames
    lines = []
    used_pairs = set()

    for f in primary_frames[:15]:
        pair = (clean_vid, f)
        if pair not in used_pairs:
            used_pairs.add(pair)
            if formatted_answer:
                lines.append(f"{clean_vid},{f},{formatted_answer}")
            else:
                lines.append(f"{clean_vid},{f}")

    # 2. Dòng 16 -> 100: Bồi các Video khác từ kết quả Search Engine
    for item in system_backup_items:
        if len(lines) >= target_count:
            break
        item_vid = item.get("video_id", "").strip()
        if item_vid.endswith(".mp4"):
            item_vid = item_vid[:-4]
        item_fid = int(item.get("frame_id", 0))

        pair = (item_vid, item_fid)
        if pair not in used_pairs:
            used_pairs.add(pair)
            if formatted_answer:
                lines.append(f"{item_vid},{item_fid},{formatted_answer}")
            else:
                lines.append(f"{item_vid},{item_fid}")

    # 3. Nếu vẫn chưa đủ 100 dòng thì lấy tiếp từ phần còn lại của primary_frames
    idx = 15
    while len(lines) < target_count and idx < len(primary_frames):
        f = primary_frames[idx]
        idx += 1
        pair = (clean_vid, f)
        if pair not in used_pairs:
            used_pairs.add(pair)
            if formatted_answer:
                lines.append(f"{clean_vid},{f},{formatted_answer}")
            else:
                lines.append(f"{clean_vid},{f}")

    # 4. Nếu vẫn còn thiếu, bồi tịnh tiến +5
    curr_f = primary_frames[-1] if primary_frames else A
    while len(lines) < target_count:
        curr_f += 5
        pair = (clean_vid, curr_f)
        if pair not in used_pairs:
            used_pairs.add(pair)
            if formatted_answer:
                lines.append(f"{clean_vid},{curr_f},{formatted_answer}")
            else:
                lines.append(f"{clean_vid},{curr_f}")

    return lines[:target_count]


def parse_raw_submission_line(raw_line: str, target_count: int = 100) -> List[str]:
    """
    Phân tích một dòng định nghĩa thô trong file CSV và sinh 100 dòng hoàn chỉnh.
    Ví dụ hỗ trợ:
    - 'L21_V015, [25635 - 26598]'
    - 'L29_V023, [10935-10968], [24920-24975]'
    - 'L28_V020,[6750 - 6884],"Anh hùng nhược ngộ vô dung địa"'
    - 'L24_V033,15930,15990,16320,16890' (TRAKE -> giữ nguyên)
    """
    line = raw_line.strip()
    if not line:
        return []

    # 1. Tìm Video ID
    vid_m = re.match(r"^\s*([A-Za-z0-9_]+)", line)
    if not vid_m:
        raise ValueError(f"Không tìm thấy Video ID hợp lệ ở dòng: {line}")
    video = vid_m.group(1).replace(".mp4", "")
    rest = line[vid_m.end():].strip()
    if rest.startswith(","):
        rest = rest[1:].strip()

    # 2. Kiểm tra nếu là dòng TRAKE (danh sách các số thuần túy không có ngoặc vuông)
    if "[" not in rest and '"' not in rest and "'" not in rest:
        parts = [p.strip() for p in rest.split(",") if p.strip()]
        if len(parts) >= 2 and all(p.isdigit() for p in parts):
            # Đây là dòng TRAKE chuỗi sự kiện -> giữ nguyên định dạng
            return [f"{video},{','.join(parts)}"]

    # 3. Trích xuất câu trả lời Answer (nếu có dấu ngoặc kép ở cuối)
    answer = None
    last_bracket_idx = rest.rfind("]")
    if last_bracket_idx != -1:
        ans_part = rest[last_bracket_idx + 1:].strip()
        if ans_part.startswith(","):
            ans_part = ans_part[1:].strip()
        if ans_part:
            answer = ans_part
    else:
        # Kiểm tra nếu dạng video, frame, "answer"
        first_comma = rest.find(",")
        if first_comma != -1:
            ans_part = rest[first_comma + 1:].strip()
            if ans_part:
                answer = ans_part

    # 4. Trích xuất khoảng [A - B] hoặc [A] hoặc số đơn A
    bracket_matches = re.findall(r"\[(\d+)\s*-\s*(\d+)\]", rest)
    single_bracket = re.findall(r"\[(\d+)\]", rest)

    intervals = []
    is_single = False

    if bracket_matches:
        for start, end in bracket_matches:
            intervals.append((int(start), int(end)))
    elif single_bracket:
        intervals.append((int(single_bracket[0]), int(single_bracket[0])))
        is_single = True
    else:
        num_m = re.search(r"^\s*(\d+)", rest)
        if num_m:
            intervals.append((int(num_m.group(1)), int(num_m.group(1))))
            is_single = True

    if not intervals:
        raise ValueError(f"Không nhận diện được frame nào từ dòng: {line}")

    A, B = intervals[0]
    second_interval = intervals[1] if len(intervals) > 1 else None

    mode = "qa" if answer else "kis"
    return generate_submission_lines(
        mode=mode,
        video_id=video,
        A=A,
        B=None if is_single else B,
        answer=answer,
        second_interval=second_interval,
        target_count=target_count
    )


def process_submission_file(file_path: str, output_path: Optional[str] = None, target_count: int = 100) -> int:
    """
    Đọc một file CSV thô, sinh 100 dòng hoàn chỉnh và ghi ra file đích.
    """
    if output_path is None:
        output_path = file_path

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File không tồn tại: {file_path}")

    with open(file_path, "r", encoding="utf-8-sig") as fp:
        raw_lines = [l.strip() for l in fp if l.strip()]

    if not raw_lines:
        return 0

    # Phân tích file: nếu có chứa bracket hoặc chỉ có 1-2 dòng định nghĩa
    if any("[" in l for l in raw_lines) or len(raw_lines) <= 2:
        out_lines = parse_raw_submission_line(raw_lines[0], target_count=target_count)
    else:
        # Trường hợp file đã có nhiều dòng sẵn -> kiểm tra nếu là TRAKE thì giữ nguyên
        first_line = raw_lines[0]
        if first_line.count(",") >= 2 and "[" not in first_line and '"' not in first_line:
            # TRAKE sequences
            out_lines = raw_lines
        else:
            # File KIS/QA đã có nhiều dòng cũ -> lấy A=dòng đầu, B=dòng cuối refill lại
            out_lines = parse_raw_submission_line(f"{raw_lines[0].split(',')[0]}, [{raw_lines[0].split(',')[1]} - {raw_lines[-1].split(',')[1]}]", target_count=target_count)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as fp:
        fp.write("\n".join(out_lines) + "\n")

    return len(out_lines)


def export_submission_zip(submission_dir: str, output_zip_path: str) -> Tuple[str, int]:
    """
    Đóng gói thư mục submission thành file .zip theo đúng quy định BTC:
    - Bên trong file ZIP PHẢI có thư mục gốc mang tên 'submission/'
    - Chứa tất cả các file CSV bên trong
    """
    if not os.path.exists(submission_dir):
        raise FileNotFoundError(f"Thư mục không tồn tại: {submission_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(output_zip_path)), exist_ok=True)

    csv_files = [f for f in os.listdir(submission_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        raise ValueError(f"Không tìm thấy file .csv nào trong thư mục: {submission_dir}")

    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for fname in sorted(csv_files):
            fpath = os.path.join(submission_dir, fname)
            # Lưu trữ với tiền tố 'submission/' bên trong archive
            arcname = os.path.join("submission", fname)
            zipf.write(fpath, arcname=arcname)

    return output_zip_path, len(csv_files)
