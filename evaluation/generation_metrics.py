"""
ThinkyLM — Generation Metrics
================================
Repetition rate, unique-token ratio, generation length, and inference speed.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Optional

import torch

from thinkylm.model import ThinkyLM


def measure_generation(
    model: ThinkyLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    eos_token_id: Optional[int] = None,
) -> dict:
    """Generate text and compute generation quality metrics.

    Args:
        model: ThinkyLM model.
        prompt_ids: Prompt token IDs (1, seq_len).
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        top_p: Top-p (nucleus) sampling parameter.
        repetition_penalty: Repetition penalty factor.
        eos_token_id: EOS token ID.

    Returns:
        Dict with keys: generated_ids, new_token_count, tokens_per_second,
                        unique_ratio, repetition_rate.
    """
    from thinkylm.generation import generate

    t0 = time.perf_counter()
    generated = generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_token_id,
    )
    elapsed = time.perf_counter() - t0

    new_ids = generated[0, prompt_ids.shape[1]:].tolist()
    n_new = len(new_ids)

    counter = Counter(new_ids)
    n_unique = len(counter)
    unique_ratio = n_unique / max(n_new, 1)

    # Bigram repetition rate
    bigrams = list(zip(new_ids[:-1], new_ids[1:]))
    n_unique_bigrams = len(set(bigrams))
    bigram_rep_rate = 1.0 - (n_unique_bigrams / max(len(bigrams), 1))

    tokens_per_second = n_new / max(elapsed, 1e-6)

    return {
        "generated_ids": generated,
        "new_token_count": n_new,
        "tokens_per_second": tokens_per_second,
        "unique_ratio": unique_ratio,
        "bigram_repetition_rate": bigram_rep_rate,
        "elapsed_seconds": elapsed,
    }
