"""
ThinkyLM — Checkpoint Tests
"""

import json
import torch
import pytest
from pathlib import Path
from thinkylm.config import ModelConfig
from thinkylm.model import build_model
from thinkylm.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
)


@pytest.fixture
def tiny_model():
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=50, context_length=16,
        hidden_size=32, num_layers=2, num_heads=4,
        intermediate_size=64, dropout=0.0,
    )
    return build_model(cfg)


def test_save_and_load_checkpoint(tmp_path, tiny_model):
    """Saved model should load with identical weights."""
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)

    ckpt_path = save_checkpoint(
        model=tiny_model,
        optimizer=optimizer,
        step=5,
        loss=2.5,
        config_dict={"test": True},
        checkpoint_dir=tmp_path,
        name="test",
    )

    # Load into a fresh model with same architecture
    cfg2 = ModelConfig(
        vocab_size=50, context_length=16,
        hidden_size=32, num_layers=2, num_heads=4,
        intermediate_size=64, dropout=0.0,
    )
    model2 = build_model(cfg2)
    meta = load_checkpoint(model2, ckpt_path, device="cpu")

    assert meta["step"] == 5
    assert abs(meta["loss"] - 2.5) < 1e-4

    # Weights must be identical
    for (n1, p1), (n2, p2) in zip(
        tiny_model.named_parameters(), model2.named_parameters()
    ):
        assert torch.allclose(p1, p2), f"Parameter mismatch: {n1}"


def test_checkpoint_meta_json(tmp_path, tiny_model):
    ckpt_path = save_checkpoint(
        model=tiny_model, optimizer=None,
        step=10, loss=3.0, config_dict={},
        checkpoint_dir=tmp_path, name="meta_test",
    )
    meta_file = ckpt_path / "meta.json"
    assert meta_file.exists()
    with meta_file.open() as f:
        meta = json.load(f)
    assert meta["step"] == 10
    assert "timestamp" in meta


def test_find_latest_checkpoint(tmp_path, tiny_model):
    save_checkpoint(
        model=tiny_model, optimizer=None,
        step=1, loss=1.0, config_dict={},
        checkpoint_dir=tmp_path, name="ck",
    )
    save_checkpoint(
        model=tiny_model, optimizer=None,
        step=2, loss=0.9, config_dict={},
        checkpoint_dir=tmp_path, name="ck",
    )
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.exists()


def test_load_nonexistent_checkpoint_raises(tiny_model):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tiny_model, "/nonexistent/path", device="cpu")


def test_resume_from_checkpoint_preserves_step(tmp_path, tiny_model):
    """After loading a checkpoint, the stored step must be accessible."""
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    ckpt_path = save_checkpoint(
        model=tiny_model, optimizer=optimizer,
        step=42, loss=1.23,
        config_dict={}, checkpoint_dir=tmp_path, name="resume_test",
    )
    cfg2 = ModelConfig(
        vocab_size=50, context_length=16,
        hidden_size=32, num_layers=2, num_heads=4,
        intermediate_size=64, dropout=0.0,
    )
    model2 = build_model(cfg2)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    meta = load_checkpoint(model2, ckpt_path, optimizer=optimizer2, device="cpu")
    assert meta["step"] == 42
