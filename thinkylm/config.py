"""
ThinkyLM — Configuration System
================================
Author: Haziq Imran
Purpose: Dataclass-based configuration with YAML loading, validation,
         and hardware-aware safety checks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Transformer model hyper-parameters."""

    vocab_size: int = 4_000
    context_length: int = 128
    hidden_size: int = 128
    num_layers: int = 3
    num_heads: int = 4
    intermediate_size: int = 512
    dropout: float = 0.1
    tie_embeddings: bool = True
    use_rope: bool = False  # Learnable positional embeddings by default

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})."
            )
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if self.vocab_size < 10:
            raise ValueError("vocab_size must be at least 10.")
        if self.context_length < 8:
            raise ValueError("context_length must be at least 8.")

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_heads


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    device: str = "cpu"
    batch_size: int = 2
    gradient_accumulation_steps: int = 2
    max_steps: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 4
    eval_interval: int = 10
    checkpoint_interval: int = 20
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    max_runtime_minutes: int = 3
    mixed_precision: bool = False  # Only safe with CUDA; auto-disabled on CPU

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    train_path: str = "data/sample"
    val_split: float = 0.1
    test_split: float = 0.05
    min_length: int = 20
    max_length: int = 100_000
    shuffle: bool = True


@dataclass
class ThinkyLMConfig:
    """Top-level configuration container."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer_path: str = "tokenizer/generated"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    name: str = "debug_1m"

    # Safety flags
    allow_large_local: bool = False

    MAX_LOCAL_PARAMS: int = field(default=5_000_000, init=False, repr=False)
    MAX_LOCAL_CONTEXT: int = field(default=512, init=False, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ThinkyLMConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Parsed ThinkyLMConfig instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            raw: dict = yaml.safe_load(f) or {}

        model_cfg = ModelConfig(**raw.get("model", {}))
        training_cfg = TrainingConfig(**raw.get("training", {}))
        data_cfg = DataConfig(**raw.get("data", {}))

        return cls(
            model=model_cfg,
            training=training_cfg,
            data=data_cfg,
            tokenizer_path=raw.get("tokenizer_path", "tokenizer/generated"),
            checkpoint_dir=raw.get("checkpoint_dir", "checkpoints"),
            log_dir=raw.get("log_dir", "runs"),
            name=raw.get("name", path.stem),
        )

    def to_dict(self) -> dict:
        """Serialise configuration to a plain dictionary."""
        import dataclasses
        return dataclasses.asdict(self)

    def estimate_params(self) -> int:
        """Rough parameter count estimate (embedding + transformer layers).

        Returns:
            Integer estimate of total trainable parameters.
        """
        m = self.model
        # Token + positional embeddings
        embed = m.vocab_size * m.hidden_size
        if not m.use_rope:
            embed += m.context_length * m.hidden_size

        per_layer = (
            # Self-attention Q K V O
            4 * m.hidden_size * m.hidden_size
            # FFN
            + 2 * m.hidden_size * m.intermediate_size
            # LayerNorm (2 per layer)
            + 4 * m.hidden_size
        )
        head = m.hidden_size  # Final LN
        lm_head = 0 if m.tie_embeddings else m.vocab_size * m.hidden_size
        return embed + m.num_layers * per_layer + head + lm_head
