# ThinkyLM — Run Tests (Windows PowerShell)
# ==========================================
# Usage: .\scripts\run_tests.ps1

Write-Host "ThinkyLM — Test Suite" -ForegroundColor Cyan

$env:TOKENIZERS_PARALLELISM = "false"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:THINKYLM_CONFIG = "configs/debug_1m.yaml"
$env:THINKYLM_CHECKPOINT = ""

python -m pytest tests/ -v --tb=short
