"""
dres_client.py — DRES Live Contest Client (Vòng Chung Kết AI Challenge 2026).
Tuân thủ chuẩn quốc tế VBS (Video Browser Showdown) và DRES v2 REST API (https://github.com/dres-dev/DRES).

Chức năng:
1. Quản lý phiên: Login tường minh, lưu sessionId, tự động dò tìm active evaluationID sau login.
2. Quy đổi thời gian chuẩn: Chuyển đổi chính xác frame_id -> millisecond (ms) dựa trên video_fps_mapping.json.
3. 1-Click Submit KIS: Nộp đáp án chỉ với 1 click chuột trực tiếp từ bất kỳ thẻ Keyframe nào.
4. Hỗ trợ đầy đủ các thể thức thi: Textual/Video KIS, Q&A (QA-answer-video-time), TRAKE (TR-video-f1,f2,...).
5. Cơ chế chống nộp trùng (Anti-Duplicate) và đếm lỗi phạt (-10đ/lần) để bảo vệ điểm số đội thi.
"""

import os
import json
import csv
import time
import ssl
import threading
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

try:
    import config
except ImportError:
    from . import config


class DRESClient:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DRESClient, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.server_url = getattr(config, "DRES_SERVER_URL", "https://eventretrieval.one").rstrip("/")
        self.username = getattr(config, "DRES_USERNAME", "")
        self.password = getattr(config, "DRES_PASSWORD", "")

        self.session_id: Optional[str] = None
        self.user_id: Optional[str] = None
        self.user_role: Optional[str] = None
        self.active_evaluation_id: Optional[str] = None
        self.evaluation_name: Optional[str] = None
        self.available_evaluations: List[Dict[str, Any]] = []
        self.last_login_time: Optional[float] = None

        # Quản lý FPS để quy đổi Frame ID -> Millisecond (ms)
        self.fps_mapping: Dict[str, float] = {}
        self._batch2_frame_pts: Dict[str, Dict[int, float]] = {}
        self._load_fps_mapping()

        # Quản lý lịch sử nộp & chống nộp trùng trong cùng 1 câu hỏi
        self.submission_history: List[Dict[str, Any]] = []
        self.task_submitted_keys: set = set()
        self.wrong_count: int = 0
        self.task_start_time: Optional[float] = None
        self.task_max_duration_s: int = 300  # Mặc định 5 phút (300s)

        # SSL context
        self.ssl_context = ssl.create_default_context()

        self._initialized = True
        print(f"[DRES] Khởi tạo DRES Client cho máy chủ: {self.server_url}")

    def _load_fps_mapping(self):
        """Nạp từ điển FPS của các video trong kho dữ liệu."""
        fps_path = getattr(config, "FPS_MAPPING_PATH", "")
        if fps_path and os.path.exists(fps_path):
            try:
                with open(fps_path, "r", encoding="utf-8") as f:
                    self.fps_mapping = json.load(f)
                print(f"[DRES] Đã nạp bảng tra cứu FPS: {len(self.fps_mapping):,} video.")
            except Exception as e:
                print(f"[DRES] Không thể đọc FPS mapping: {e}")

    def _make_request(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None, timeout: int = 8) -> Tuple[int, Any]:
        """Gửi HTTP request an toàn đến máy chủ DRES."""
        url = f"{self.server_url}{endpoint}"
        body_bytes = None
        headers = {"Accept": "application/json"}

        if data is not None:
            body_bytes = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, context=self.ssl_context, timeout=timeout) as response:
                status_code = response.status
                res_body = response.read().decode("utf-8")
                try:
                    parsed = json.loads(res_body)
                except Exception:
                    parsed = {"raw": res_body}
                return status_code, parsed
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            try:
                err_json = json.loads(err_body)
            except Exception:
                err_json = {"error": err_body}
            return e.code, err_json
        except Exception as e:
            return 500, {"error": str(e)}

    def login(self, username: Optional[str] = None, password: Optional[str] = None, server_url: Optional[str] = None) -> Tuple[bool, str]:
        """Đăng nhập vào hệ thống DRES để lấy sessionId và cập nhật active evaluation."""
        if server_url:
            self.server_url = server_url.rstrip("/")
        if username:
            self.username = username
        if password:
            self.password = password

        if not self.username or not self.password:
            return False, "Thiếu Username hoặc Password DRES!"

        payload = {"username": self.username, "password": self.password}
        code, resp = self._make_request("/api/v2/login", method="POST", data=payload)

        if code == 200 and isinstance(resp, dict) and "sessionId" in resp:
            self.session_id = resp["sessionId"]
            self.user_id = resp.get("id")
            self.user_role = resp.get("role")
            self.last_login_time = time.time()
            # Không log session token: console có thể được stream/chụp màn hình khi thi.
            print(f"[DRES] ✅ Đăng nhập thành công tài khoản '{self.username}'.")

            # Tự động dò tìm đợt thi ACTIVE
            eval_ok, eval_msg = self.refresh_active_evaluation()
            msg = f"Đăng nhập thành công! {eval_msg}"
            return True, msg
        else:
            err = resp.get("description") or resp.get("error") or resp.get("message") or f"Lỗi HTTP {code}"
            print(f"[DRES] ❌ Đăng nhập thất bại: {err}")
            return False, f"Đăng nhập thất bại: {err}"

    def refresh_active_evaluation(self) -> Tuple[bool, str]:
        """Lấy danh sách evaluation và tự động chọn evaluation đang ACTIVE."""
        if not self.session_id:
            return False, "Chưa đăng nhập DRES. Hãy đăng nhập tường minh từ giao diện."

        endpoint = f"/api/v2/client/evaluation/list?session={self.session_id}"
        code, resp = self._make_request(endpoint, method="GET")

        if code == 200 and isinstance(resp, list):
            self.available_evaluations = [ev for ev in resp if isinstance(ev, dict)]
            if not resp:
                self.active_evaluation_id = None
                self.evaluation_name = None
                return False, "Hiện chưa có đợt thi (Evaluation) nào trên máy chủ DRES."

            # Ưu tiên tìm evaluation có status == 'ACTIVE'
            active_eval = None
            for ev in resp:
                if isinstance(ev, dict) and ev.get("status") == "ACTIVE":
                    active_eval = ev
                    break

            # Nếu không có active, lấy evaluation đầu tiên
            if not active_eval and len(resp) > 0 and isinstance(resp[0], dict):
                active_eval = resp[0]

            if active_eval:
                self.active_evaluation_id = active_eval.get("id")
                self.evaluation_name = active_eval.get("name")
                print(f"[DRES] 🎯 Tìm thấy đợt thi: '{self.evaluation_name}' (ID: {self.active_evaluation_id}) | Trạng thái: {active_eval.get('status')}")
                return True, f"Đã kết nối đợt thi: '{self.evaluation_name}' (Trạng thái: {active_eval.get('status')})"

            return False, "Không tìm thấy đợt thi hợp lệ."
        else:
            err = resp.get("description") or resp.get("error") if isinstance(resp, dict) else f"Lỗi HTTP {code}"
            return False, f"Lỗi truy vấn evaluation: {err}"

    def set_evaluation(self, evaluation_id: str, evaluation_name: Optional[str] = None) -> Tuple[bool, str]:
        """Chọn Run ID thủ công, có đối chiếu danh sách DRES khi danh sách đã được nạp."""
        clean_id = str(evaluation_id or "").strip()
        if not clean_id:
            return False, "Run ID không được để trống."

        matched = next(
            (ev for ev in self.available_evaluations if str(ev.get("id", "")).strip() == clean_id),
            None,
        )
        if self.available_evaluations and matched is None:
            return False, "Run ID không nằm trong danh sách evaluation mà tài khoản DRES hiện thấy."

        self.active_evaluation_id = clean_id
        self.evaluation_name = (
            str(evaluation_name or "").strip()
            or (str(matched.get("name", "")).strip() if matched else "")
            or "Run thủ công"
        )
        return True, f"Đã chọn Run ID: {clean_id}"

    def frame_to_time_ms(self, video_id: str, frame_id: Union[int, str]) -> int:
        """
        Quy đổi frame_id sang millisecond (ms) chuẩn xác dựa trên FPS thực tế của video.
        time_ms = round((frame_id / fps) * 1000)
        """
        clean_vid = config.normalize_video_id(video_id)

        try:
            fid = int(frame_id)
        except (ValueError, TypeError):
            fid = 0

        fps = self.fps_mapping.get(clean_vid, 25.0)
        if fps <= 0:
            fps = 25.0

        time_sec = fid / fps
        if clean_vid.startswith(("M", "N", "S")):
            if clean_vid not in self._batch2_frame_pts:
                csv_path = os.path.join(config.BATCH2_DIR, "map-keyframes-k1r", clean_vid + ".csv")
                with open(csv_path, newline="", encoding="utf-8-sig") as source:
                    self._batch2_frame_pts[clean_vid] = {
                        int(row["frame_idx"]): float(row["pts_time"]) for row in csv.DictReader(source)
                    }
            time_sec = self._batch2_frame_pts[clean_vid].get(fid, time_sec)
        time_ms = int(round(time_sec * 1000.0))
        return time_ms

    def submit_kis(
        self,
        video_id: str,
        frame_id: Optional[Union[int, str]] = None,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Nộp bài thể thức KIS (Textual KIS hoặc Video KIS).
        Nếu truyền frame_id: tự động tính ra start_ms = end_ms = time_ms.
        Nếu truyền start_ms / end_ms: nộp theo khoảng mili-giây.
        """
        clean_vid = config.normalize_video_id(video_id)

        # 1. Tính toán start_ms và end_ms
        if start_ms is None:
            if frame_id is None:
                return {"status": "error", "message": "Vui lòng cung cấp frame_id hoặc start_ms!"}
            time_ms = self.frame_to_time_ms(clean_vid, frame_id)
            start_ms = time_ms
            end_ms = time_ms
        else:
            if end_ms is None:
                end_ms = start_ms

        if start_ms > end_ms:
            start_ms, end_ms = end_ms, start_ms

        # 2. Cơ chế Chống Nộp Trùng (Anti-Duplicate Protection)
        dedup_key = f"KIS:{clean_vid}:{start_ms}:{end_ms}"
        if dedup_key in self.task_submitted_keys and not force:
            return {
                "status": "warning",
                "sub_status": "DUPLICATE",
                "outcome": "DUPLICATE_LOCAL",
                "accepted": False,
                "message": f"⚠️ Backend local đã gửi [Video {clean_vid} | Mốc {start_ms}ms] trong câu hiện tại. Yêu cầu trùng được chặn và không gửi lại DRES.",
                "video_id": clean_vid,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "wrong_count": self.wrong_count
            }

        # 3. Đảm bảo phiên đăng nhập và active evaluation
        if not self.session_id or not self.active_evaluation_id:
            ok, msg = self.refresh_active_evaluation()
            if not ok:
                return {"status": "warning", "message": f"⚠️ DRES chưa sẵn sàng: {msg}. Hãy kiểm tra kết nối mạng hoặc tài khoản!", "wrong_count": self.wrong_count}

        # 4. Đóng gói body chuẩn BTC DRES cho KIS
        body = {
            "answerSets": [{
                "answers": [{
                  "mediaItemName": clean_vid,
                  "start": str(start_ms),
                  "end": str(end_ms)
                }]
            }]
        }

        endpoint = f"/api/v2/submit/{self.active_evaluation_id}?session={self.session_id}"
        code, resp = self._make_request(endpoint, method="POST", data=body)

        # 5. Phân tích kết quả trả về từ DRES
        return self._process_submission_response(
            code=code,
            resp=resp,
            task_type="KIS",
            video_id=clean_vid,
            detail_key=dedup_key,
            start_ms=start_ms,
            end_ms=end_ms,
            frame_id=frame_id
        )

    def submit_qa(
        self,
        video_id: str,
        answer: str,
        frame_id: Optional[Union[int, str]] = None,
        time_ms: Optional[int] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Nộp bài thể thức Q&A.
        Format chuẩn BTC: text = "QA-<ANSWER>-<VIDEO_ID>-<TIME(ms)>"
        """
        clean_vid = config.normalize_video_id(video_id)

        clean_ans = str(answer).strip().replace("\r", " ").replace("\n", " ")
        if not clean_ans:
            return {"status": "error", "message": "Câu trả lời QA không được để trống!"}

        if time_ms is None:
            if frame_id is None:
                return {"status": "error", "message": "Vui lòng cung cấp frame_id hoặc time_ms cho QA!"}
            time_ms = self.frame_to_time_ms(clean_vid, frame_id)

        qa_text = f"QA-{clean_ans}-{clean_vid}-{time_ms}"
        dedup_key = f"QA:{qa_text}"

        if dedup_key in self.task_submitted_keys and not force:
            return {
                "status": "warning",
                "sub_status": "DUPLICATE",
                "outcome": "DUPLICATE_LOCAL",
                "accepted": False,
                "message": f"⚠️ Backend local đã gửi đáp án QA '{qa_text}' trong câu hiện tại. Yêu cầu trùng được chặn và không gửi lại DRES.",
                "wrong_count": self.wrong_count
            }

        if not self.session_id or not self.active_evaluation_id:
            ok, msg = self.refresh_active_evaluation()
            if not ok:
                return {"status": "warning", "message": f"⚠️ DRES chưa sẵn sàng: {msg}. Hãy kiểm tra kết nối mạng hoặc tài khoản!", "wrong_count": self.wrong_count}

        body = {
            "answerSets": [{
                "answers": [{
                    "text": qa_text
                }]
            }]
        }

        endpoint = f"/api/v2/submit/{self.active_evaluation_id}?session={self.session_id}"
        code, resp = self._make_request(endpoint, method="POST", data=body)

        return self._process_submission_response(
            code=code,
            resp=resp,
            task_type="QA",
            video_id=clean_vid,
            detail_key=dedup_key,
            qa_text=qa_text,
            time_ms=time_ms,
            frame_id=frame_id
        )

    def submit_trake(
        self,
        video_id: str,
        frame_ids: List[Union[int, str]],
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Nộp bài thể thức TRAKE.
        Format chuẩn BTC: text = "TR-<VIDEO_ID>-<FRAME_ID1>,<FRAME_ID2>,..."
        """
        clean_vid = config.normalize_video_id(video_id)

        if not frame_ids or len(frame_ids) < 2:
            return {"status": "error", "message": "TRAKE yêu cầu ít nhất 2 frame IDs!"}

        frame_str = ",".join(str(int(f)) for f in frame_ids)
        trake_text = f"TR-{clean_vid}-{frame_str}"
        dedup_key = f"TRAKE:{trake_text}"

        if dedup_key in self.task_submitted_keys and not force:
            return {
                "status": "warning",
                "sub_status": "DUPLICATE",
                "outcome": "DUPLICATE_LOCAL",
                "accepted": False,
                "message": f"⚠️ Backend local đã gửi chuỗi TRAKE '{trake_text}' trong câu hiện tại. Yêu cầu trùng được chặn và không gửi lại DRES.",
                "wrong_count": self.wrong_count
            }

        if not self.session_id or not self.active_evaluation_id:
            ok, msg = self.refresh_active_evaluation()
            if not ok:
                return {"status": "warning", "message": f"⚠️ DRES chưa sẵn sàng: {msg}. Hãy kiểm tra kết nối mạng hoặc tài khoản!", "wrong_count": self.wrong_count}

        body = {
            "answerSets": [{
                "answers": [{
                    "text": trake_text
                }]
            }]
        }

        endpoint = f"/api/v2/submit/{self.active_evaluation_id}?session={self.session_id}"
        code, resp = self._make_request(endpoint, method="POST", data=body)

        return self._process_submission_response(
            code=code,
            resp=resp,
            task_type="TRAKE",
            video_id=clean_vid,
            detail_key=dedup_key,
            trake_text=trake_text,
            frame_ids=frame_ids
        )

    def _process_submission_response(
        self,
        code: int,
        resp: Any,
        task_type: str,
        video_id: str,
        detail_key: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Phân tích phản hồi từ DRES Server, cập nhật lịch sử và tính điểm phạt."""
        now_str = datetime.now().strftime("%H:%M:%S")
        record = {
            "timestamp": now_str,
            "task_type": task_type,
            "video_id": video_id,
            "http_code": code,
            **kwargs
        }

        # OpenAPI DRES 2.x: `status` là boolean; verdict thật nằm ở `submission`.
        status_text = "UNKNOWN"
        description = ""

        if isinstance(resp, dict):
            raw_verdict = (
                resp.get("submission") or
                resp.get("submissionStatus") or
                resp.get("verdict") or
                (resp.get("status") if isinstance(resp.get("status"), str) else None)
            )
            if raw_verdict is not None:
                status_text = str(raw_verdict).strip().upper()
            elif code in (200, 202):
                status_text = "SUBMITTED"
            else:
                status_text = "ERROR"
            description = resp.get("description") or resp.get("message") or resp.get("error") or ""
        elif isinstance(resp, str):
            description = resp
            if code in (200, 202):
                status_text = "SUBMITTED"

        # Phân loại trạng thái
        is_correct = "CORRECT" in status_text or "ACCEPT" in status_text
        is_wrong = "WRONG" in status_text or "REJECT" in status_text or "INCORRECT" in status_text
        is_indeterminate = any(
            marker in status_text
            for marker in ("INDETERMINATE", "UNDECIDABLE", "UNDECIDED", "SUBMITTED")
        )
        accepted = code in (200, 202)

        if is_wrong:
            self.wrong_count += 1
            record["result"] = "WRONG"
            ui_status = "error"
            outcome = "WRONG"
            ui_msg = f"❌ DRES: Đáp án SAI! (-10 điểm phạt | Đã nộp sai {self.wrong_count} lần). {description}"
        elif is_correct:
            record["result"] = "CORRECT"
            ui_status = "success"
            outcome = "CORRECT"
            ui_msg = f"🎉 DRES: CHÚC MỪNG! ĐÁP ÁN ĐÚNG HOÀN TOÀN! {description}"
        elif accepted and (is_indeterminate or code in (200, 202)):
            record["result"] = "PENDING"
            ui_status = "info"
            outcome = "PENDING"
            ui_msg = f"📤 DRES: Server đã nhận bài nhưng chưa trả verdict đúng/sai. {description}"
        else:
            record["result"] = "FAILED"
            ui_status = "warning"
            outcome = "TECHNICAL_ERROR"
            ui_msg = f"⚠️ DRES: Lỗi kết nối (HTTP {code}): {description}. Không xác định đúng/sai."

        # Chỉ chống nộp trùng khi DRES đã thực sự nhận request.
        if accepted:
            self.task_submitted_keys.add(detail_key)
        record["outcome"] = outcome
        record["accepted"] = accepted
        record["dres_status"] = status_text
        record["ui_message"] = ui_msg
        self.submission_history.insert(0, record)

        print(
            f"[DRES] Submission {task_type} | HTTP {code} | "
            f"verdict={status_text} | accepted={accepted} | video={video_id}"
        )

        return {
            "status": ui_status,
            "outcome": outcome,
            "accepted": accepted,
            "dres_status": status_text,
            "http_code": code,
            "message": ui_msg,
            "wrong_count": self.wrong_count,
            "video_id": video_id,
            "record": record,
            "raw_response": resp
        }

    def start_new_task(self, max_duration_s: int = 300) -> Dict[str, Any]:
        """Đặt lại trạng thái cho câu hỏi mới (Xóa cache chống trùng, reset bộ đếm lỗi phạt, khởi động timer)."""
        self.task_submitted_keys.clear()
        self.wrong_count = 0
        self.task_start_time = time.time()
        self.task_max_duration_s = max_duration_s

        print(f"[DRES] 🔄 Đã khởi động câu hỏi mới. Thời gian: {max_duration_s}s.")
        return {
            "status": "success",
            "message": f"Đã bắt đầu câu hỏi mới! Thời gian tối đa: {max_duration_s // 60} phút.",
            "task_max_duration_s": max_duration_s,
            "wrong_count": 0
        }

    def get_status(self) -> Dict[str, Any]:
        """Lấy toàn bộ trạng thái kết nối DRES phục vụ hiển thị Live Header."""
        is_connected = bool(self.session_id and self.active_evaluation_id)
        
        # Tính điểm ước tính nếu nộp đúng ngay lúc này
        estimated_score = 0.0
        elapsed_s = 0.0
        remaining_s = self.task_max_duration_s

        if self.task_start_time is not None:
            elapsed_s = max(0.0, time.time() - self.task_start_time)
            remaining_s = max(0.0, float(self.task_max_duration_s) - elapsed_s)
            f_t = max(0.0, 1.0 - (elapsed_s / float(self.task_max_duration_s)))
            raw_score = 50.0 + 50.0 * f_t - (self.wrong_count * 10.0)
            estimated_score = max(0.0, round(raw_score, 1))

        return {
            "server_url": self.server_url,
            "username": self.username,
            "is_logged_in": bool(self.session_id),
            "is_connected": is_connected,
            "active_evaluation_id": self.active_evaluation_id,
            "evaluation_name": self.evaluation_name,
            "available_evaluation_count": len(self.available_evaluations),
            "wrong_count": self.wrong_count,
            "elapsed_s": round(elapsed_s, 1),
            "remaining_s": round(remaining_s, 1),
            "estimated_score": estimated_score,
            "total_submitted": len(self.task_submitted_keys),
            "history": self.submission_history[:10]
        }


# Singleton accessor
_dres_client_instance: Optional[DRESClient] = None

def get_dres_client() -> DRESClient:
    global _dres_client_instance
    if _dres_client_instance is None:
        _dres_client_instance = DRESClient()
    return _dres_client_instance
