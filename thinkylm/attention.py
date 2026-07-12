"""
ThinkyLM — Multi-Head Causal Self-Attention
=============================================
Implements scaled dot-product attention with an additive causal mask.
Supports both learnable positional embeddings and RoPE.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from thinkylm.embeddings import RotaryEmbedding


class MultiHeadCausalAttention(nn.Module):
    """Multi-head causal (decoder-only) self-attention.

    Uses an upper-triangular additive mask filled with -inf so that
    each position can only attend to itself and earlier positions.
    Future tokens receive exactly zero attention — verified in tests.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        dropout: Attention-weight dropout probability.
        use_rope: If True, apply RoPE instead of relying on external pos embeddings.
        context_length: Maximum sequence length (required when use_rope=True).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rope: bool = False,
        context_length: int = 512,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
        )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Fused QKV projection
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, context_length)

        # Causal mask buffer: registered so it moves with .to(device)
        self._register_causal_mask(context_length)

        # Weight initialisation
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02 / math.sqrt(2))

    def _register_causal_mask(self, max_len: int) -> None:
        mask = torch.full((max_len, max_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)  # upper triangle = -inf, lower = 0
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Output tensor of the same shape.
        """
        B, T, C = x.shape

        # Fused QKV
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.hidden_size, dim=-1)

        # Reshape to (B, heads, T, head_dim)
        def _reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = _reshape(q), _reshape(k), _reshape(v)

        if self.use_rope:
            q, k = self.rope(q, k)

        # Scaled dot-product attention with causal mask
        attn_bias = self.causal_mask[:T, :T]  # slice to current seq_len
        scores = (q @ k.transpose(-2, -1)) / self.scale + attn_bias
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        out = weights @ v  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))
