"""
ThinkyLM — AdamW Optimiser Wrapper
=====================================
Separates weight-decay parameters from non-weight-decay parameters
following the standard practice in language model training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Build AdamW with weight decay applied only to matrices and embeddings.

    Biases and LayerNorm parameters are excluded from weight decay,
    following the practice established in GPT-style training.

    Args:
        model: ThinkyLM model.
        learning_rate: Peak learning rate.
        weight_decay: L2 regularisation coefficient.
        betas: AdamW beta coefficients.
        eps: Numerical stability constant.

    Returns:
        Configured AdamW optimiser.
    """
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Exclude 1D parameters (bias, layernorm weight/bias, embedding scalars)
        if param.ndim == 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
        fused=False,  # fused requires CUDA; leave False for CPU safety
    )

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(
        f"[Optimiser] AdamW | decay params: {n_decay:,} | "
        f"no-decay params: {n_no_decay:,} | lr={learning_rate}"
    )
    return optimizer
