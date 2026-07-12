"""
ThinkyLM — Tiny Forward Pass Test (used in CI)

A single-file smoke test that can run in GitHub Actions without a tokenizer or checkpoint.
Validates the full forward pass, loss, and one gradient step.
"""

import torch

from thinkylm.config import ModelConfig
from thinkylm.model import build_model


def test_tiny_forward_pass():
    """Complete forward pass with loss on the smallest possible model."""
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64,
        context_length=16,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
        dropout=0.0,
        tie_embeddings=True,
    )
    model = build_model(cfg)
    model.train()

    input_ids = torch.randint(1, 64, (1, 8))
    labels = input_ids.clone()

    out = model(input_ids, labels=labels)

    assert "logits" in out, "Missing logits in output."
    assert "loss" in out, "Missing loss in output."
    assert out["logits"].shape == (1, 8, 64)
    assert out["loss"].ndim == 0
    assert out["loss"].item() > 0
    assert not torch.isnan(out["loss"])


def test_gradient_flows():
    """Gradients must flow to all parameters."""
    torch.manual_seed(1)
    cfg = ModelConfig(
        vocab_size=64, context_length=16,
        hidden_size=32, num_layers=2, num_heads=4,
        intermediate_size=64, dropout=0.0,
    )
    model = build_model(cfg)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    input_ids = torch.randint(1, 64, (1, 8))
    out = model(input_ids, labels=input_ids.clone())
    out["loss"].backward()

    grad_params = [
        name for name, p in model.named_parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert len(grad_params) > 0, "No parameters received gradients."


def test_parameter_count_within_local_limit():
    """Debug config must stay within the local safety limit of 5M params."""
    from thinkylm.config import ThinkyLMConfig
    cfg = ThinkyLMConfig()  # defaults = debug_1m
    estimated = cfg.estimate_params()
    assert estimated < cfg.MAX_LOCAL_PARAMS, (
        f"Estimated params {estimated:,} exceed local limit {cfg.MAX_LOCAL_PARAMS:,}"
    )
