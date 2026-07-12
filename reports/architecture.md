# ThinkyLM Architecture

## Overview

ThinkyLM is a **decoder-only causal Transformer** implemented from scratch in PyTorch. It follows the Pre-LayerNorm design used by modern LLMs, with configurable depth, width, and vocabulary.

This document describes the architecture, design decisions, and implementation choices.

---

## Architecture Diagram

```
Input token IDs
       │
       ▼
TokenEmbedding           [vocab_size → hidden_size]
       +
PositionalEmbedding      [context_length → hidden_size]
       │
       ▼
EmbeddingDropout
       │
       ├──────────────────────────────────────────┐
       ▼                                          │ × num_layers
TransformerBlock:                                 │
  ┌─────────────────────────────────────────┐    │
  │  LayerNorm (Pre-LN)                     │    │
  │  MultiHeadCausalSelfAttention           │    │
  │    ├── QKV Projection (3×hidden_size)   │    │
  │    ├── Causal Mask (upper-triangular)   │    │
  │    ├── Scaled Dot-Product Attention     │    │
  │    ├── AttnDropout                      │    │
  │    └── Output Projection               │    │
  │  Residual Add                           │    │
  │  LayerNorm (Pre-LN)                     │    │
  │  FeedForward (SiLU activation)          │    │
  │    ├── Linear (hidden → intermediate)   │    │
  │    ├── SiLU                             │    │
  │    ├── Dropout                          │    │
  │    └── Linear (intermediate → hidden)  │    │
  │  Residual Add                           │    │
  └─────────────────────────────────────────┘    │
       │                                          │
       └──────────────────────────────────────────┘
       │
       ▼
Final LayerNorm
       │
       ▼
LM Head: Linear [hidden_size → vocab_size]
(weights optionally tied to TokenEmbedding)
       │
       ▼
Logits [batch, seq_len, vocab_size]
       │
       ▼
Cross-Entropy Loss (next-token prediction)
```

---

## Key Design Decisions

### Pre-LayerNorm (Pre-LN)
Normalisation is applied **before** each sub-layer (attention and FFN) rather than after. Pre-LN is more stable during training and is the default in most modern LLMs (GPT-3, LLaMA, Mistral). It avoids vanishing/exploding gradients in deep networks.

### Causal Masking
An additive upper-triangular mask filled with `-inf` is registered as a non-persistent buffer. This ensures that:
- Position `i` can only attend to positions `0..i`.
- Future tokens receive zero attention weight after softmax.
- The mask is verified by automated tests (`tests/test_attention.py`).

### SiLU Activation
The feed-forward network uses SiLU (Swish) rather than GELU. SiLU is smooth, differentiable everywhere, and performs comparably to GELU while being simpler to implement.

### Weight Tying
By default, the LM head weight matrix is shared with the token embedding table. This reduces parameters by `vocab_size × hidden_size` and often improves performance at small scale.

### No Bias in Projections
Following GPT-style training practice, bias terms are omitted from `qkv_proj`, `out_proj`, `fc1`, and `fc2`. Biases would be exempt from weight decay anyway, and their contribution is marginal.

---

## Parameter Count

For the `debug_1m` configuration:

| Component | Formula | Count |
|---|---|---|
| Token embeddings | 4000 × 128 | 512,000 |
| Positional embeddings | 128 × 128 | 16,384 |
| Per-block (×3) | 4×128²+2×128×512+4×128 | ~198,656×3 |
| Final LayerNorm | 128 | 128 |
| LM Head | (tied) | 0 |
| **Total** | | **~1.1M** |

---

## Supported Configurations

| Config | Params | Context | Vocab | Hardware |
|---|---|---|---|---|
| debug_1m | ~1.1M | 128 | 4,000 | CPU (local) |
| tiny_10m | ~10M | 256 | 8,000 | CUDA 16 GB+ |
| portfolio_25m | ~25M | 512 | 16,000 | CUDA 24 GB+ |

---

## Generation

Autoregressive generation supports:
- **Temperature** — controls distribution sharpness
- **Top-k** — restricts sampling to top k logits
- **Top-p (nucleus)** — samples from minimum set covering p probability mass
- **Repetition penalty** — suppresses already-generated tokens
- **EOS stopping** — halts when `<eos>` is produced
- **Deterministic mode** — fixed seed for reproducible output

---

## Limitations

- At ~1M parameters, the model cannot exhibit genuine reasoning.
- The sample corpus (~50K tokens) is far smaller than real pretraining data.
- No sliding window attention, flash attention, or KV caching.
- These are deliberate simplifications for clarity and portability.
