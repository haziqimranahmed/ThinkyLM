# ThinkyLM — Roadmap

## Current Status: Portfolio Foundation (v0.1)

The current version demonstrates the full ML engineering lifecycle from architecture to API.
The model has random weights and has not undergone meaningful training.

---

## Phase 1 — Foundation (Complete)
- [x] From-scratch Transformer architecture
- [x] Custom BPE tokenizer
- [x] Data pipeline (clean, dedup, split, tokenize)
- [x] Causal LM pretraining loop
- [x] Instruction fine-tuning pipeline
- [x] Checkpoint save/load (safetensors)
- [x] FastAPI inference server
- [x] Gradio demo
- [x] CLI inference
- [x] Hardware safety checks
- [x] Thinky-Bench evaluation benchmark
- [x] GitHub Actions CI
- [x] Docker deployment
- [x] Professional documentation

## Phase 2 — Small-Scale Training (Planned: Cloud)
- [ ] Train `tiny_10m` on ~100MB of public-domain philosophy/logic text
- [ ] Achieve sub-100 perplexity on validation set
- [ ] Instruction fine-tune on expanded Thinky dataset (500+ examples)
- [ ] Publish checkpoint to Hugging Face Hub
- [ ] Add training loss chart to README

## Phase 3 — Portfolio Quality (Planned: Cloud)
- [ ] Train `portfolio_25m` on ~1GB curated corpus
- [ ] Add KV-caching for faster generation
- [ ] Expand Thinky-Bench to 100+ prompts with human scoring
- [ ] Publish human evaluation results
- [ ] Add weight visualisation notebook

## Phase 4 — Advanced Features (Future)
- [ ] Flash Attention integration
- [ ] Sliding window attention for longer contexts
- [ ] Mixture of Experts (MoE) layer option
- [ ] RLHF-style preference training
- [ ] Streaming generation in API
- [ ] Model quantisation (INT8 / GGUF)

---

## Cloud Training Plan

See `reports/cloud_training_guide.md` for detailed cost estimates and commands.

| Phase | Config | Provider | Estimated Cost | Duration |
|---|---|---|---|---|
| Phase 2 | tiny_10m | RunPod A10G | $3–8 | 3–6 hours |
| Phase 3 | portfolio_25m | RunPod A100 | $15–30 | 12–24 hours |
