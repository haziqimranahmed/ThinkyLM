"""
ThinkyLM — Text Generation
============================
Autoregressive generation with temperature, top-k, top-p (nucleus),
and repetition penalty. Supports deterministic generation with fixed seed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from thinkylm.model import ThinkyLM


@torch.no_grad()
def generate(
    model: ThinkyLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate tokens autoregressively from a prompt.

    Args:
        model: The ThinkyLM model in eval mode.
        input_ids: Prompt token IDs of shape (1, seq_len) or (batch, seq_len).
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (1.0 = unchanged distribution,
                     <1.0 = sharper, >1.0 = more uniform).
        top_k: If > 0, restrict sampling to top-k logits.
        top_p: If < 1.0, restrict sampling to the smallest set of tokens
               whose cumulative probability exceeds top_p (nucleus sampling).
        repetition_penalty: Penalty factor for repeating tokens (1.0 = no penalty).
        eos_token_id: Stop generation when this token is produced.
        seed: If provided, set the random seed for reproducible generation.

    Returns:
        Token ID tensor of shape (batch, original_len + new_tokens).
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    device = input_ids.device
    context_length = model.config.context_length

    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # Truncate to context window
        ctx = generated[:, -context_length:]

        out = model(ctx)
        logits = out["logits"]  # (batch, seq_len, vocab_size)
        next_logits = logits[:, -1, :]  # (batch, vocab_size)

        # Repetition penalty
        if repetition_penalty != 1.0:
            next_logits = _apply_repetition_penalty(
                next_logits, generated, repetition_penalty
            )

        # Temperature scaling
        if temperature != 1.0:
            next_logits = next_logits / max(temperature, 1e-8)

        # Top-k filtering
        if top_k > 0:
            next_logits = _top_k_filter(next_logits, top_k)

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            next_logits = _top_p_filter(next_logits, top_p)

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

        generated = torch.cat([generated, next_token], dim=-1)

        # Stop at EOS
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return generated


def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Downscale logits for already-generated tokens.

    Args:
        logits: Float tensor of shape (batch, vocab_size).
        generated: Previously generated token IDs (batch, seq_len).
        penalty: Penalty factor > 1.0 suppresses repeats.

    Returns:
        Modified logits tensor.
    """
    for b in range(logits.shape[0]):
        unique_tokens = generated[b].unique()
        token_logits = logits[b, unique_tokens]
        # Divide positives, multiply negatives (standard repetition penalty)
        token_logits = torch.where(
            token_logits > 0,
            token_logits / penalty,
            token_logits * penalty,
        )
        logits[b, unique_tokens] = token_logits
    return logits


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits except the top-k.

    Args:
        logits: Float tensor of shape (batch, vocab_size).
        k: Number of top logits to keep.

    Returns:
        Filtered logits with -inf for non-top-k positions.
    """
    k = min(k, logits.size(-1))
    values, _ = torch.topk(logits, k)
    threshold = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering.

    Args:
        logits: Float tensor of shape (batch, vocab_size).
        p: Cumulative probability threshold.

    Returns:
        Filtered logits tensor.
    """
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[sorted_indices_to_remove] = float("-inf")

    # Scatter back to original ordering
    return logits.scatter(-1, sorted_indices, sorted_logits)


def greedy_generate(
    model: ThinkyLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Deterministic greedy decoding (argmax at each step).

    Args:
        model: ThinkyLM model.
        input_ids: Prompt token IDs.
        max_new_tokens: Maximum tokens to generate.
        eos_token_id: Stop token.

    Returns:
        Generated token ID tensor.
    """
    return generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.0,
        eos_token_id=eos_token_id,
        seed=0,
    )
