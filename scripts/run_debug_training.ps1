# ThinkyLM — Run Debug Training (Windows PowerShell)
# =====================================================
# Runs the safe 20-step smoke training on CPU.
# Maximum runtime: 3 minutes. Checkpoint saved automatically.
#
# Usage: .\scripts\run_debug_training.ps1

Write-Host "ThinkyLM — Debug Training (20 steps, CPU)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

$env:TOKENIZERS_PARALLELISM = "false"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"

# Step 1: Prepare data
Write-Host "`n[1/3] Preparing sample data..." -ForegroundColor Yellow
python data_pipeline/prepare_sample_data.py

# Step 2: Train tokenizer
Write-Host "`n[2/3] Training tokenizer..." -ForegroundColor Yellow
python tokenizer/train_tokenizer.py `
    --input data/sample `
    --vocab-size 4000 `
    --output tokenizer/generated

# Step 3: Run training
Write-Host "`n[3/3] Starting 20-step debug training..." -ForegroundColor Yellow
Write-Host "This will take up to 3 minutes. Press Ctrl+C to stop early." -ForegroundColor Gray
python training/pretrain.py --config configs/debug_1m.yaml

Write-Host "`nTraining complete! Check checkpoints/debug_1m/ for saved checkpoint." -ForegroundColor Green
