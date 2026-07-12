"""
ThinkyLM — Embeddings
======================
Token embeddings and learnable positional embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Learnable token embedding table.

    Args:
        vocab_size: Total vocabulary size.
        hidden_size: Embedding dimension.
        padding_idx: Token index treated as padding (gradient zeroed).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        # Initialise with small normal values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if padding_idx is not None:
            nn.init.zeros_(self.embedding.weight[padding_idx])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Long tensor of shape (batch, seq_len).

        Returns:
            Float tensor of shape (batch, seq_len, hidden_size).
        """
        return self.embedding(input_ids)


class LearnablePositionalEmbedding(nn.Module):
    """Learnable absolute positional embedding.

    Args:
        context_length: Maximum supported sequence length.
        hidden_size: Embedding dimension.
    """

    def __init__(self, context_length: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(context_length, hidden_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            seq_len: Current sequence length.
            device: Target device.

        Returns:
            Float tensor of shape (1, seq_len, hidden_size).
        """
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        return self.embedding(positions)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Applied per-head inside attention; frequencies are cached for efficiency.

    Args:
        head_dim: Dimension of each attention head.
        context_length: Maximum sequence length to pre-cache.
        base: Rotation base frequency (default 10_000).
    """

    def __init__(
        self,
        head_dim: int,
        context_length: int = 512,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE."
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(context_length)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors.

        Args:
            q: Query tensor of shape (batch, heads, seq_len, head_dim).
            k: Key tensor of shape (batch, heads, seq_len, head_dim).

        Returns:
            Rotated (q, k) tensors of the same shape.
        """
        seq_len = q.shape[2]
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        return _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate halves of the last dimension."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)
