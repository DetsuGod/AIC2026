@echo off
chcp 65001 > nul
title AIC 2026 - Multimodal Video Retrieval System
echo ===============================================================================
echo   🚀 KHỞI ĐỘNG HỆ THỐNG TRUY VẤN VIDEO AIC 2026 (ONE-CLICK LAUNCHER)
echo ===============================================================================
echo.

:: 1. Tìm môi trường ảo Python
set "PYTHON_EXE="

if exist "code\local_retrieval\venv\Scripts\python.exe" (
    set "PYTHON_EXE=code\local_retrieval\venv\Scripts\python.exe"
    goto :FOUND_PYTHON
)

if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
    goto :FOUND_PYTHON
)

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
    goto :FOUND_PYTHON
)

:: Nếu không thấy venv, thử dùng python hệ thống
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=python"
    goto :FOUND_PYTHON
)

echo [LỖI] Không tìm thấy Python hoặc môi trường ảo Virtualenv!
echo Vui lòng cài đặt Python hoặc tạo venv: python -m venv venv
pause
exit /b 1

:FOUND_PYTHON
echo [THÀNH CÔNG] Đang sử dụng Python: %PYTHON_EXE%
echo.

:: 2. Tự động mở trình duyệt sau 3 giây khi server sẵn sàng
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:8000"

:: 3. Khởi chạy FastAPI Backend Server
echo [INFO] Đang khởi động Backend Server tại http://127.0.0.1:8000 ...
echo [INFO] Nhấn Ctrl + C để dừng hệ thống.
echo ===============================================================================
echo.

%PYTHON_EXE% code\web\backend\main.py

pause
