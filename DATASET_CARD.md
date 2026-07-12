# Dataset Card — ThinkyLM

## Overview

ThinkyLM uses only original, legally clean text. No copyrighted books,
scraped web content, or unlicensed corpora are used.

---

## Data Sources

| Source | Licence | Type | Tokens (approx.) | Notes |
|---|---|---|---|---|
| ThinkyLM Original Sample | MIT | Philosophy/Logic (original) | ~50,000 | Written by Haziq Imran for this project |
| ThinkyLM Instruction Set | MIT | Synthetic instruction pairs | ~15,000 | 50 hand-written examples |

---

## Licensing Rules

All included data is original content released under the MIT licence.

For future training data:
- **Allowed**: Project Gutenberg (public domain), Wikipedia (CC-BY-SA with attribution), permissively licensed open datasets
- **Not allowed**: Copyrighted books, paywalled papers, proprietary datasets, scraped social media

---

## Cleaning Steps

1. UTF-8 loading with error replacement
2. Unicode NFC normalisation
3. Whitespace normalisation (collapse runs, limit newlines)
4. Control character removal
5. Minimum length filtering (20 characters)
6. Maximum length filtering (100,000 characters)
7. SHA-256 deduplication

---

## Deduplication

Exact paragraph-level deduplication using SHA-256 hashing.
Near-duplicate detection is planned for future versions.

---

## Splits

| Split | Fraction | Purpose |
|---|---|---|
| Train | 85% | Model training |
| Validation | 10% | Loss monitoring during training |
| Test | 5% | Final held-out evaluation |

Splits are deterministic with seed=42.

---

## Token Statistics (Sample Corpus)

| Metric | Value |
|---|---|
| Total documents | 6 |
| Total paragraphs (after cleaning) | ~80 |
| Estimated tokens (4K BPE vocab) | ~8,000–12,000 |
| Vocabulary coverage | >95% |
| Unknown-token rate | <1% |

---

## Synthetic Data Labelling

The instruction dataset (`data/instructions/thinky_instructions.jsonl`) is synthetic.
All 50 examples were written by Haziq Imran specifically for ThinkyLM.

**These examples are demonstration data. They are not sufficient to produce
a reasoning model. They demonstrate the instruction-tuning pipeline.**

---

## Known Limitations

1. The sample corpus is very small (~50K tokens vs. billions for production models).
2. The corpus is narrow in domain (philosophy and logic).
3. No multilingual content.
4. No code or mathematical content.
5. No diversity in writing style or register.

---

## Prohibited Data Sources

The following are explicitly prohibited:
- OpenAI API outputs
- ChatGPT or GPT-4 generated text (licensing restrictions)
- Copyrighted books (without explicit licence)
- Social media posts (GDPR/privacy concerns)
- Clinical or medical records
- Private communications
