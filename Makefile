# ThinkyLM Makefile
# ====================
# Cross-platform development shortcuts.
# Windows users: use the scripts/ PowerShell scripts instead.
# Linux/macOS users: use make <target>

.PHONY: help setup prepare-data train-tokenizer train test lint check-hardware api demo clean

PYTHON := python
CONFIG := configs/debug_1m.yaml

help:
	@echo "ThinkyLM — Available Targets"
	@echo "================================"
	@echo "  make setup           Install dependencies"
	@echo "  make check-hardware  Report hardware and safety status"
	@echo "  make prepare-data    Create sample training data"
	@echo "  make train-tokenizer Train BPE tokenizer on sample data"
	@echo "  make eval-tokenizer  Evaluate the trained tokenizer"
	@echo "  make train           Run 20-step debug training (safe locally)"
	@echo "  make test            Run full test suite"
	@echo "  make lint            Run Ruff linter"
	@echo "  make api             Start FastAPI server (requires checkpoint)"
	@echo "  make demo            Start Gradio demo (requires checkpoint)"
	@echo "  make clean           Remove generated files"

setup:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements-dev.txt

check-hardware:
	$(PYTHON) scripts/check_hardware.py

prepare-data:
	$(PYTHON) data_pipeline/prepare_sample_data.py

train-tokenizer: prepare-data
	$(PYTHON) tokenizer/train_tokenizer.py \
		--input data/sample \
		--vocab-size 4000 \
		--output tokenizer/generated

eval-tokenizer:
	$(PYTHON) tokenizer/evaluate_tokenizer.py \
		--tokenizer tokenizer/generated \
		--text tokenizer/sample_text.txt

train: train-tokenizer
	$(PYTHON) training/pretrain.py --config $(CONFIG)

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	ruff check thinkylm/ training/ evaluation/ inference/ api/ data_pipeline/ tokenizer/ tests/
	@echo "Lint OK"

api:
	uvicorn api.main:app --host 127.0.0.1 --port 8000

demo:
	$(PYTHON) demo/gradio_app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .ruff_cache .pytest_cache
	@echo "Cleaned."
