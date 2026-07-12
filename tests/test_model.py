"""
ThinkyLM — Model Tests

Tests forward pass, loss computation, and parameter counting.
"""

import torch
import pytest
from thinkylm.config import ModelConfig
from thinkylm.model import ThinkyLM, build_model


@pytest.fixture
def tiny_config():
    return ModelConfig(
        vocab_size=100,
        context_length=16,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
        dropout=0.0,
        tie_embeddings=True,
    )


@pytest.fixture
def tiny_model(tiny_config):
    torch.manual_seed(42)
    return build_model(tiny_config)


def test_model_builds(tiny_model):
    assert tiny_model is not None


def test_forward_output_shape(tiny_model):
    tiny_model.eval()
    input_ids = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        out = tiny_model(input_ids)
    logits = out["logits"]
    assert logits.shape == (2, 8, 100), f"Expected (2, 8, 100), got {logits.shape}"


def test_forward_with_labels_returns_loss(tiny_model):
    tiny_model.eval()
    input_ids = torch.randint(0, 100, (2, 8))
    labels = input_ids.clone()
    with torch.no_grad():
        out = tiny_model(input_ids, labels=labels)
    assert "loss" in out
    loss = out["loss"]
    assert loss.ndim == 0, "Loss should be a scalar."
    assert not torch.isnan(loss), "Loss is NaN."
    assert loss.item() > 0, "Loss should be positive for a randomly initialised model."


def test_loss_decreases_with_training(tiny_config):
    """A single gradient step on a tiny batch should decrease the loss."""
    torch.manual_seed(42)
    model = build_model(tiny_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    input_ids = torch.randint(1, 100, (2, 8))
    labels = input_ids.clone()

    model.train()
    out1 = model(input_ids, labels=labels)
    loss1 = out1["loss"]

    loss1.backward()
    optimizer.step()
    optimizer.zero_grad()

    with torch.no_grad():
        out2 = model(input_ids, labels=labels)
        loss2 = out2["loss"]

    assert loss2.item() < loss1.item(), (
        f"Loss did not decrease: {loss1.item():.4f} → {loss2.item():.4f}"
    )


def test_parameter_count(tiny_model):
    counts = tiny_model.count_parameters()
    assert "total" in counts
    assert counts["total"] > 0


def test_context_length_exceeded_raises(tiny_model):
    tiny_model.eval()
    # Config context_length = 16; send 20 tokens
    input_ids = torch.randint(0, 100, (1, 20))
    with pytest.raises(ValueError, match="context_length"):
        tiny_model(input_ids)


def test_no_nan_in_outputs(tiny_model):
    tiny_model.eval()
    input_ids = torch.randint(1, 100, (2, 10))
    with torch.no_grad():
        out = tiny_model(input_ids)
    assert not torch.isnan(out["logits"]).any()


def test_tie_embeddings(tiny_config):
    tiny_config.tie_embeddings = True
    model = build_model(tiny_config)
    # When tied, lm_head weight and token embedding weight should be the same object
    assert model.lm_head.weight is model.token_emb.embedding.weight


def test_no_tie_embeddings():
    cfg = ModelConfig(
        vocab_size=100, context_length=16, hidden_size=32,
        num_layers=2, num_heads=4, intermediate_size=64,
        tie_embeddings=False,
    )
    model = build_model(cfg)
    assert model.lm_head.weight is not model.token_emb.embedding.weight
