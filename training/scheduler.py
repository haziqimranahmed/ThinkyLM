"""
ThinkyLM — Learning Rate Scheduler
=====================================
Linear warmup followed by cosine decay to zero.
"""

from __future__ import annotations

import math

import torch


def get_lr(
    step: int,
    max_steps: int,
    warmup_steps: int,
    learning_rate: float,
    min_lr_ratio: float = 0.1,
) -> float:
    """Compute the learning rate for a given step.

    Schedule:
        - Linear warmup from 0 to learning_rate over warmup_steps.
        - Cosine decay from learning_rate to min_lr over remaining steps.

    Args:
        step: Current training step (0-indexed).
        max_steps: Total number of training steps.
        warmup_steps: Number of linear warmup steps.
        learning_rate: Peak learning rate.
        min_lr_ratio: Final LR as a fraction of peak LR.

    Returns:
        Learning rate float for this step.
    """
    min_lr = learning_rate * min_lr_ratio

    # Warmup phase
    if step < warmup_steps:
        return learning_rate * (step + 1) / max(warmup_steps, 1)

    # After max_steps: stay at min_lr
    if step >= max_steps:
        return min_lr

    # Cosine decay
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (learning_rate - min_lr) * cosine


def apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set learning rate on all optimiser parameter groups.

    Args:
        optimizer: PyTorch optimiser.
        lr: Learning rate to apply.
    """
    for group in optimizer.param_groups:
        group["lr"] = lr
