"""
ThinkyLM — Evaluation: Perplexity
====================================
Computes perplexity on a text dataset using a trained ThinkyLM model.
"""

from __future__ import annotations

import math

import torch

from thinkylm.model import ThinkyLM


@torch.no_grad()
def compute_perplexity(
    model: ThinkyLM,
    sequences: list[list[int]],
    batch_size: int = 2,
    device: str = "cpu",
) -> tuple[float, float]:
    """Compute perplexity over a set of token sequences.

    Args:
        model: Trained ThinkyLM model.
        sequences: List of fixed-length integer token sequences.
        batch_size: Number of sequences per forward pass.
        device: Device string.

    Returns:
        Tuple of (average cross-entropy loss, perplexity).
    """
    from data_pipeline.dataset import build_dataloader

    model.eval()
    dev = torch.device(device)
    loader = build_dataloader(sequences, batch_size=batch_size, shuffle=False, num_workers=0)

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(dev)
        labels = batch["labels"].to(dev)
        out = model(input_ids, labels=labels)
        total_loss += out["loss"].item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    perplexity = math.exp(min(avg_loss, 700))  # Prevent overflow
    return avg_loss, perplexity
