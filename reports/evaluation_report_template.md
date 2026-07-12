# ThinkyLM Evaluation Report Template
# ======================================

## Evaluation Information

| Field | Value |
|---|---|
| Date | YYYY-MM-DD |
| Checkpoint | checkpoints/debug_1m/debug_1m_step000020 |
| Evaluator | Haziq Imran |
| Model step | 20 |

## Automatic Metrics

| Metric | Value | Notes |
|---|---|---|
| Val loss | 8.04 | Expected to decrease with more training |
| Perplexity | 263.63 | High — untrained model expected behaviour |
| Tokens/second | TBD | Run evaluate_checkpoint.py |
| Unique token ratio | TBD | Run evaluate_checkpoint.py |
| Bigram repetition rate | TBD | Run evaluate_checkpoint.py |
| Model size (MB) | ~4 MB | 1.1M × 4 bytes float32 |

## Thinky-Bench Human Evaluation

**Status**: Not evaluated — checkpoint not yet meaningfully trained.

Scoring rubric:
- 0 = No attempt / completely off-topic
- 1 = Attempted but fundamentally wrong
- 2 = Partial — some correct elements
- 3 = Good — mostly correct with minor gaps
- 4 = Excellent — clear, accurate, appropriately nuanced

| Category | Score (0–4) | Notes |
|---|---|---|
| Claim clarification | TBD | |
| Assumption detection | TBD | |
| Steelmanning | TBD | |
| Counterargument quality | TBD | |
| Fallacy recognition | TBD | |
| Belief updating | TBD | |
| Uncertainty calibration | TBD | |
| Philosophical comparison | TBD | |
| Humour appropriateness | TBD | |
| Respectful disagreement | TBD | |
| **Overall** | TBD / 40 | |

## Important Disclaimer

> ThinkyLM at ~1M parameters with 20 training steps does not exhibit reasoning.
> Thinky-Bench scores on this checkpoint reflect token pattern completion, not
> philosophical understanding. Human evaluation is meaningful only after
> significant cloud training (≥10K steps on ≥100M tokens).

## Comparison to Chance

A model that merely repeats the most common tokens would achieve very high
perplexity and score 0 on all Thinky-Bench categories. The architecture test
verifies that loss decreases with gradient descent — nothing more.
