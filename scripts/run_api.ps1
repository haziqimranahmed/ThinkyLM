# ThinkyLM — Start API Server (Windows PowerShell)
# ==================================================
# Usage: .\scripts\run_api.ps1
# Optional: set THINKYLM_CHECKPOINT before running

Write-Host "ThinkyLM — FastAPI Server" -ForegroundColor Cyan

$env:TOKENIZERS_PARALLELISM = "false"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:THINKYLM_CONFIG = "configs/debug_1m.yaml"

if (-not $env:THINKYLM_CHECKPOINT) {
    Write-Host "NOTICE: THINKYLM_CHECKPOINT not set. API will run with random weights." -ForegroundColor Yellow
    Write-Host "To load a checkpoint, set: `$env:THINKYLM_CHECKPOINT = 'checkpoints/debug_1m/...'" -ForegroundColor Yellow
}

Write-Host "`nStarting server at http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Swagger docs at   http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray

uvicorn api.main:app --host 127.0.0.1 --port 8000
