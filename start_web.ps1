# Script khoi dong Web Server AIC 2026 cho PowerShell
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " [AIC 2026] DANG KHOI DONG SERVER MULTIMODAL SEARCH ENGINE..." -ForegroundColor Green
Write-Host " Dia chi Web: http://localhost:8000" -ForegroundColor Yellow
Write-Host "=====================================================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$pythonExe = Join-Path $scriptDir "code\local_retrieval\venv\Scripts\python.exe"
$mainPy = Join-Path $scriptDir "code\web\backend\main.py"

& $pythonExe $mainPy
