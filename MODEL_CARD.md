# Model Card — ThinkyLM

## Model Description

ThinkyLM is an independently implemented decoder-only causal Transformer designed for
critical and philosophical argumentation. It is trained from random initialisation using
a custom BPE tokenizer and an original text corpus.

**Author**: Haziq Imran  
**Version**: 0.1.0  
**Type**: Causal Language Model  
**Architecture**: Decoder-only Transformer (Pre-LN)  
**Implementation**: From scratch in PyTorch (no pretrained weights)  
**Licence**: MIT  

> **Important**: ThinkyLM is an educational portfolio project. It is not a commercial
> reasoning system. The current checkpoint (~1M parameters, 20 training steps) demonstrates
> architecture correctness, not language understanding.

---

## Architecture

| Property | debug_1m | tiny_10m | portfolio_25m |
|---|---|---|---|
| Parameters | ~1.1M | ~10M | ~25M |
| Vocabulary | 4,000 | 8,000 | 16,000 |
| Context length | 128 | 256 | 512 |
| Hidden size | 128 | 256 | 512 |
| Layers | 3 | 6 | 8 |
| Attention heads | 4 | 8 | 8 |
| FFN size | 512 | 1,024 | 2,048 |
| Activation | SiLU | SiLU | SiLU |
| Norm | LayerNorm (Pre-LN) | LayerNorm | LayerNorm |
| Positional encoding | Learnable | RoPE | RoPE |
| Weight tying | Yes | Yes | No |

---

## Training Status

| Stage | Status | Notes |
|---|---|---|
| Architecture | ✅ Complete | All tests pass |
| Tokenizer | ✅ Trained | BPE, 4K vocab, sample corpus |
| Pretraining | ⚠️ Smoke only | 20 steps, loss decreasing |
| Instruction tuning | 🔲 Pending | Requires pretrained base |
| Evaluation | ⚠️ Framework ready | Human scoring not yet performed |
| Cloud training | 🔲 Planned | See cloud_training_guide.md |

---

## Dataset Summary

The local checkpoint (if any) was trained on the ThinkyLM original sample corpus:
approximately 50,000 tokens of original philosophy/logic text written under the MIT licence.

This is **not** a production-quality training corpus. It is sufficient to verify the
training pipeline and demonstrate loss decreasing.

---

## Intended Uses

- Portfolio demonstration of end-to-end ML engineering
- Architecture study and experimentation
- Custom tokenizer and data pipeline reference
- Starting point for researchers building small LMs from scratch

---

## Out-of-Scope Uses

- Production question answering
- Medical, legal, or safety-critical advice
- Replacement for commercial language models
- Any application requiring genuine language understanding

---

## Limitations

1. **Scale**: ~1M parameters is insufficient for any meaningful language generation.
2. **Data**: The training corpus is tiny and domain-narrow.
3. **Training**: Smoke-only training (20 steps) produces random-weight-like outputs.
4. **Evaluation**: No human evaluation has been performed on the current checkpoint.
5. **Architecture**: No flash attention, KV caching, or other production optimisations.

---

## Bias Risks

- The sample corpus is weighted toward Western analytic philosophy.
- The model may reflect writing patterns from the original sample corpus.
- At current scale and training, outputs are essentially random token sequences.
- No bias evaluation has been performed.

---

## Evaluation

Thinky-Bench provides 30 manually reviewed prompts across 10 categories.
Human scoring rubric: 0 (no attempt) to 4 (excellent).
See `evaluation/thinky_bench.json` for the full benchmark.

**Current benchmark status**: Not evaluated — checkpoint not yet meaningfully trained.

---

## Hardware

Developed on:
- Dell Inspiron 15 3511
- Intel Core i5-1135G7
- 16 GB RAM
- Intel Iris Xe (no CUDA)
- Windows 11

Cloud training planned on RunPod A10G/A100.

---

## Citation

```bibtex
@misc{thinkylm2026,
  author = {Haziq Imran},
  title  = {ThinkyLM: A From-Scratch Decoder-Only Transformer for Philosophical Argumentation},
  year   = {2026},
  url    = {https://github.com/haziqimran/thinkylm}
}
```
