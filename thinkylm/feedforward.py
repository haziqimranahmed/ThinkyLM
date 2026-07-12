"""
ThinkyLM — Feed-Forward Network
=================================
Position-wise FFN with SiLU (Swish) activation and configurable dropout.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Two-layer position-wise feed-forward network.

    Architecture: x → Linear → SiLU → Dropout → Linear → Dropout

    Args:
        hidden_size: Input and output dimension.
        intermediate_size: Projection dimension (typically 4× hidden_size).
        dropout: Dropout probability applied after each linear layer.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        # Initialise weights
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02 / math.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Tensor of the same shape.
        """
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))
