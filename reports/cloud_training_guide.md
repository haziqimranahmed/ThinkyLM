# ThinkyLM — Cloud Training Guide

## Overview

This guide covers setting up cloud GPU training for ThinkyLM's larger configurations.
**Never run `tiny_10m` or `portfolio_25m` locally on the development machine.**

---

## Recommended Providers (2026)

| Provider | GPU | VRAM | Spot Price | Notes |
|---|---|---|---|---|
| RunPod | RTX A10G | 24 GB | ~$0.22/hr | Good for tiny_10m |
| RunPod | A100 80 GB | 80 GB | ~$1.50/hr | Best for portfolio_25m |
| Lambda Labs | A10 | 24 GB | ~$0.60/hr | Stable, reserved |
| Google Colab Pro+ | A100 | 40 GB | ~$10/month | Interactive, limited hours |
| Vast.ai | Various | Various | ~$0.15/hr | Cheapest, less reliable |

---

## Setup on RunPod (Recommended)

### 1. Create a Pod

```bash
# Select: PyTorch template, CUDA 12.1+, Python 3.11
# Minimum: 24 GB VRAM for tiny_10m
# Minimum: 40 GB VRAM for portfolio_25m
# Storage: 50 GB for datasets and checkpoints
```

### 2. Clone the Repository

```bash
git clone https://github.com/haziqimran/thinkylm.git
cd thinkylm
```

### 3. Install Dependencies (CUDA)

```bash
# Install PyTorch with CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Prepare Data

```bash
# For tiny_10m: prepare a ~100MB corpus of philosophy/logic/argumentation text
# Use only public-domain or permissively licensed sources
# Place in data/processed/tiny_10m/

python data_pipeline/prepare_sample_data.py  # baseline sample
# Add more text manually to data/sample/ before tokenizing
```

### 5. Train Tokenizer (larger vocab for cloud runs)

```bash
python tokenizer/train_tokenizer.py \
    --input data/processed/tiny_10m \
    --vocab-size 8000 \
    --output tokenizer/generated_8k
```

### 6. Run Training

```bash
# tiny_10m — ~3-6 hours on A10G
python training/pretrain.py --config configs/tiny_10m.yaml

# portfolio_25m — ~12-24 hours on A100
python training/pretrain.py --config configs/portfolio_25m.yaml
```

### 7. TensorBoard Monitoring

```bash
tensorboard --logdir runs/tiny_10m --host 0.0.0.0 --port 6006
# Access at http://<runpod-ip>:6006
```

### 8. Save Checkpoint to Hugging Face Hub

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='checkpoints/tiny_10m',
    repo_id='haziqimran/thinkylm-tiny-10m',
    repo_type='model',
    token='YOUR_HF_TOKEN'
)
"
```

---

## Estimated Training Costs

| Config | Steps | GPU | Time | Cost |
|---|---|---|---|---|
| tiny_10m | 10,000 | A10G (24 GB) | 4–6 hr | $1–2 |
| portfolio_25m | 50,000 | A100 (80 GB) | 12–20 hr | $18–30 |

*Prices based on RunPod spot pricing. Check current rates before running.*

---

## Safety Checklist Before Cloud Training

- [ ] Verified the config file targets `device: cuda`
- [ ] Checked that `vocab_size` matches the trained tokenizer
- [ ] Confirmed `checkpoint_interval` is set (do not lose progress)
- [ ] Confirmed `max_runtime_minutes` allows enough time
- [ ] Set up TensorBoard access
- [ ] Dataset is in place and tokenized
- [ ] Git LFS or HF Hub configured for checkpoint storage

---

## Using Git LFS for Checkpoints

```bash
git lfs install
git lfs track "*.safetensors"
git lfs track "*.pt"
git add .gitattributes
git commit -m "Set up Git LFS for model checkpoints"
```

---

## Responsible Data Licensing

For cloud training, use only:
- Project Gutenberg (public domain)
- Wikipedia (CC-BY-SA) — clearly labelled
- OpenWebText (permissive)
- Original ThinkyLM sample texts (MIT)

Never use:
- Copyrighted books without licence
- Paywalled academic papers
- Proprietary datasets

Update `data_pipeline/manifest_example.csv` with every new data source.
