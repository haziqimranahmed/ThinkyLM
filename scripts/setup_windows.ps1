# ThinkyLM — Windows Setup Script
# Run in PowerShell: .\scripts\setup_windows.ps1

Write-Host "ThinkyLM Windows Setup" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan

# Check Python version
$pythonVersion = python --version 2>&1
Write-Host "Python: $pythonVersion"

# Install CPU-only PyTorch
Write-Host "`nInstalling PyTorch (CPU)..." -ForegroundColor Yellow
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
Write-Host "`nInstalling remaining dependencies..." -ForegroundColor Yellow
python -m pip install tokenizers pyyaml pydantic fastapi "uvicorn[standard]" gradio tqdm safetensors tensorboard psutil httpx pytest pytest-asyncio ruff

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. python scripts/check_hardware.py"
Write-Host "  2. python data_pipeline/prepare_sample_data.py"
Write-Host "  3. python tokenizer/train_tokenizer.py"
Write-Host "  4. python training/pretrain.py --config configs/debug_1m.yaml"
