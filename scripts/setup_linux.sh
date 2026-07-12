#!/bin/bash
# ThinkyLM — Linux Setup Script
# Usage: chmod +x scripts/setup_linux.sh && ./scripts/setup_linux.sh

set -e
echo "ThinkyLM Linux Setup"
echo "===================="

echo "Python: $(python3 --version)"

echo ""
echo "Installing PyTorch (CPU)..."
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "Installing remaining dependencies..."
python3 -m pip install tokenizers pyyaml pydantic fastapi "uvicorn[standard]" \
    gradio tqdm safetensors tensorboard psutil httpx pytest pytest-asyncio ruff

echo ""
echo "Setup complete!"
echo "Next steps:"
echo "  1. python scripts/check_hardware.py"
echo "  2. python data_pipeline/prepare_sample_data.py"
echo "  3. python tokenizer/train_tokenizer.py"
echo "  4. python training/pretrain.py --config configs/debug_1m.yaml"
