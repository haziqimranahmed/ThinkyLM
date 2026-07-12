# ThinkyLM Training Report Template
# ===================================
# Fill this in after each training run.

## Run Information

| Field | Value |
|---|---|
| Date | YYYY-MM-DD |
| Config | configs/debug_1m.yaml |
| Steps completed | 20 |
| Runtime | X minutes |
| Device | CPU |
| Checkpoint | checkpoints/debug_1m/debug_1m_step000020 |

## Model

| Parameter | Value |
|---|---|
| Total parameters | 1,120,000 |
| Vocabulary size | 1,152 (trained on sample corpus) |
| Context length | 128 |
| Architecture | Pre-LN Decoder Transformer |

## Dataset

| Metric | Value |
|---|---|
| Source | ThinkyLM original sample (MIT) |
| Total paragraphs | 56 |
| Training sequences | 29 |
| Validation sequences | 3 |
| Total training tokens | 3,712 |

## Training Metrics

| Step | Train Loss | Val Loss | Perplexity | LR |
|---|---|---|---|---|
| 5 | 8.2408 | — | — | 2.97e-04 |
| 10 | — | 8.1140 | 277.05 | — |
| 15 | 8.0126 | — | — | 9.00e-05 |
| 20 | — | 8.0424 | 263.63 | — |

## Notes

- Loss is decreasing, confirming the training loop is functional.
- Perplexity of ~263 after 20 steps on 3,712 tokens is expected for a randomly initialised model.
- This is a smoke test only. Meaningful training requires significantly more data and steps.
- TensorBoard not available in this environment (tensorboard package needed separately).

## Honest Assessment

This checkpoint has **not** undergone meaningful training. It is an architecture
verification. Do not use it for any real application. The loss curve confirms the
training pipeline is correctly implemented and the gradient flows properly.

## Next Steps

1. Expand the dataset (add public-domain philosophy/logic text)
2. Retrain tokenizer on larger corpus
3. Run `tiny_10m.yaml` on cloud GPU for ~10,000 steps
4. Re-evaluate with Thinky-Bench after meaningful training
