"""
ThinkyLM — Transformer Block
==============================
Pre-LN Transformer decoder block: LayerNorm → Attention → residual,
then LayerNorm → FFN → residual.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from thinkylm.attention import MultiHeadCausalAttention
from thinkylm.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """A single Pre-LN causal Transformer decoder block.

    Pre-LayerNorm placement (normalise before the sub-layer) is more
    stable during training than post-LN and is used by most modern LLMs.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        intermediate_size: FFN intermediate dimension.
        dropout: Dropout probability.
        use_rope: Whether to use RoPE.
        context_length: Maximum sequence length.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        use_rope: bool = False,
        context_length: int = 512,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadCausalAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            context_length=context_length,
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Tensor of the same shape.
        """
        # Pre-LN self-attention with residual
        x = x + self.attn(self.ln1(x))
        # Pre-LN FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x
