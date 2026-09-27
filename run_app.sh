#!/bin/bash
# ===============================================================================
# AIC 2026 - Multimodal Video Retrieval System (One-Click Launcher for Linux/macOS)
# ===============================================================================

echo "==============================================================================="
echo "  🚀 KHỞI ĐỘNG HỆ THỐNG TRUY VẤN VIDEO AIC 2026 (ONE-CLICK LAUNCHER)"
echo "==============================================================================="
echo ""

# 1. Tìm môi trường ảo Python
PYTHON_EXE=""

if [ -f "code/local_retrieval/venv/bin/python" ]; then
    PYTHON_EXE="code/local_retrieval/venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON_EXE="venv/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_EXE=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_EXE="python3"
elif command -v python &>/dev/null; then
    PYTHON_EXE="python"
else
    echo "❌ [LỖI] Không tìm thấy Python hoặc môi trường ảo Virtualenv!"
    echo "Vui lòng cài đặt: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

echo "✅ [THÀNH CÔNG] Đang sử dụng Python: $PYTHON_EXE"
echo ""

# 2. Tự động mở trình duyệt sau 3 giây
(
    sleep 3
    if command -v xdg-open &>/dev/null; then
        xdg-open http://127.0.0.1:8000
    elif command -v open &>/dev/null; then
        open http://127.0.0.1:8000
    fi
) &

# 3. Khởi động Backend
echo "🌐 [INFO] Đang khởi động Backend Server tại http://127.0.0.1:8000 ..."
echo "🛑 [INFO] Nhấn Ctrl + C để dừng hệ thống."
echo "==============================================================================="
echo ""

$PYTHON_EXE code/web/backend/main.py
