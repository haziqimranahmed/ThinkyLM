# Changelog

All notable changes to ThinkyLM are documented here.

## [0.1.0] — 2026-07-12

### Added
- Complete decoder-only Transformer architecture (Pre-LN, SiLU, causal masking)
- Custom BPE tokenizer with 7 special tokens
- Causal language model pretraining with AdamW + cosine schedule
- Instruction fine-tuning pipeline
- 50 hand-written instruction examples (thinky_instructions.jsonl)
- Thinky-Bench evaluation benchmark (30 prompts, 10 categories)
- FastAPI REST API with Swagger documentation
- Gradio web demo
- CLI inference interface
- Hardware safety checker
- Checkpoint save/load (safetensors)
- TensorBoard logging
- GitHub Actions CI workflow
- CPU-compatible Dockerfile
- Docker Compose configuration
- Full test suite (~40 tests)
- Three YAML configurations (debug_1m, tiny_10m, portfolio_25m)
- Cloud training guide
- Professional README, MODEL_CARD, DATASET_CARD
- CONTRIBUTING, SECURITY, CITATION files

### Architecture
- Verified causal masking (no future-token leakage)
- Weight tying option (embedding ↔ LM head)
- Fused QKV projection
- Configurable RoPE or learnable positional embeddings

### Hardware Safety
- Refuses cloud configs without CUDA
- RAM and disk pre-flight checks
- Runtime limit with graceful checkpoint on timeout
- Thread count limited to 2 for CPU safety
