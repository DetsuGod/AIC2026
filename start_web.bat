@echo off
title AIC 2026 - Multimodal Pro Search Engine
echo =====================================================================
echo  [AIC 2026] DANG KHOI DONG SERVER MULTIMODAL SEARCH ENGINE...
echo  Dia chi Web: http://localhost:8000
echo =====================================================================
cd /d "%~dp0"
"%~dp0code\local_retrieval\venv\Scripts\python.exe" "%~dp0code\web\backend\main.py"
pause
