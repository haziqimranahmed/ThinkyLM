"""
ThinkyLM — Checkpoint Management
==================================
Save and load model checkpoints using safetensors (primary) with a
PyTorch fallback. Supports full training-state resumption.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

try:
    from safetensors.torch import load_file as st_load_file
    from safetensors.torch import save_file as st_save_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    loss: float,
    config_dict: dict,
    checkpoint_dir: str | Path,
    name: str = "checkpoint",
) -> Path:
    """Save a training checkpoint.

    Saves model weights (safetensors preferred), training state, and config.

    Args:
        model: The ThinkyLM model.
        optimizer: Optional optimizer state.
        step: Current training step.
        loss: Current training loss.
        config_dict: Serialised configuration dict.
        checkpoint_dir: Directory to write the checkpoint.
        name: Checkpoint subdirectory name.

    Returns:
        Path to the created checkpoint directory.
    """
    ckpt_path = Path(checkpoint_dir) / f"{name}_step{step:06d}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Save model weights
    state_dict = model.state_dict()
    if HAS_SAFETENSORS:
        # safetensors requires float tensors with no shared memory.
        # Weight tying creates shared memory between lm_head.weight and
        # token_emb.embedding.weight — clone to break the sharing.
        float_state = {k: v.float().clone() for k, v in state_dict.items()}
        st_save_file(float_state, ckpt_path / "model.safetensors")
        weights_file = "model.safetensors"
    else:
        torch.save(state_dict, ckpt_path / "model.pt")
        weights_file = "model.pt"

    # Save training metadata
    meta: dict[str, Any] = {
        "step": step,
        "loss": float(loss),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "weights_file": weights_file,
        "config": config_dict,
    }
    if optimizer is not None:
        torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")
        meta["has_optimizer"] = True

    with (ckpt_path / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    # Write a "latest" pointer
    latest_file = Path(checkpoint_dir) / "latest.txt"
    latest_file.write_text(str(ckpt_path))

    print(f"[Checkpoint] Saved → {ckpt_path}")
    return ckpt_path


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into a model (and optionally an optimizer).

    Args:
        model: ThinkyLM model to load weights into.
        checkpoint_path: Path to the checkpoint directory.
        optimizer: Optional optimizer to restore state into.
        device: Device to map tensors to.

    Returns:
        Metadata dictionary from the checkpoint (includes step, loss, etc.).

    Raises:
        FileNotFoundError: If checkpoint_path or weights file does not exist.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    meta_file = ckpt_path / "meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"No meta.json in checkpoint: {ckpt_path}")

    with meta_file.open() as f:
        meta = json.load(f)

    weights_file = meta.get("weights_file", "model.pt")
    weights_path = ckpt_path / weights_file

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    if weights_file.endswith(".safetensors") and HAS_SAFETENSORS:
        state_dict = st_load_file(str(weights_path), device=device)
    else:
        state_dict = torch.load(weights_path, map_location=device)

    model.load_state_dict(state_dict, strict=True)

    if optimizer is not None and (ckpt_path / "optimizer.pt").exists():
        opt_state = torch.load(ckpt_path / "optimizer.pt", map_location=device)
        optimizer.load_state_dict(opt_state)

    print(
        f"[Checkpoint] Loaded ← {ckpt_path} "
        f"(step={meta['step']}, loss={meta['loss']:.4f})"
    )
    return meta


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """Find the most recent checkpoint in a directory.

    Args:
        checkpoint_dir: Root checkpoint directory.

    Returns:
        Path to the latest checkpoint, or None if not found.
    """
    checkpoint_dir = Path(checkpoint_dir)
    latest_file = checkpoint_dir / "latest.txt"

    if latest_file.exists():
        path = Path(latest_file.read_text().strip())
        if path.exists():
            return path

    # Fallback: find highest step number
    candidates = sorted(checkpoint_dir.glob("*_step*"))
    return candidates[-1] if candidates else None
