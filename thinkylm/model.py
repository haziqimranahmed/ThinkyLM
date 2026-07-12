"""
ThinkyLM — Decoder-Only Causal Language Model
===============================================
Author: Haziq Imran

A from-scratch PyTorch implementation of a decoder-only Transformer.
This is NOT a wrapped pretrained model. Weights are randomly initialised.

Design notes:
- Pre-LayerNorm for training stability.
- Optional embedding weight tying (LM head shares token embedding weights).
- Cross-entropy next-token prediction loss computed internally.
- Parameter count is reported on instantiation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from thinkylm.block import TransformerBlock
from thinkylm.config import ModelConfig
from thinkylm.embeddings import LearnablePositionalEmbedding, TokenEmbedding


class ThinkyLM(nn.Module):
    """ThinkyLM: decoder-only causal language model.

    Args:
        config: ModelConfig with architecture hyper-parameters.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # ── Embeddings ─────────────────────────────────────────────────────
        self.token_emb = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            padding_idx=0,
        )
        self.pos_emb: LearnablePositionalEmbedding | None = None
        if not config.use_rope:
            self.pos_emb = LearnablePositionalEmbedding(
                context_length=config.context_length,
                hidden_size=config.hidden_size,
            )
        self.embed_dropout = nn.Dropout(config.dropout)

        # ── Transformer Blocks ─────────────────────────────────────────────
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    intermediate_size=config.intermediate_size,
                    dropout=config.dropout,
                    use_rope=config.use_rope,
                    context_length=config.context_length,
                )
                for _ in range(config.num_layers)
            ]
        )

        # ── Output ─────────────────────────────────────────────────────────
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying: share token embedding ↔ LM head
        if config.tie_embeddings:
            self.lm_head.weight = self.token_emb.embedding.weight

        # Report parameter count on construction
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[ThinkyLM] Initialised | "
            f"total params: {total:,} | trainable: {trainable:,}"
        )

    # ── Forward pass ───────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with optional loss computation.

        Args:
            input_ids: Long tensor of shape (batch, seq_len).
            labels: Optional long tensor of shape (batch, seq_len).
                    If provided, loss is computed (shifted by 1 for causal LM).

        Returns:
            Dictionary with keys:
              - ``logits``: shape (batch, seq_len, vocab_size)
              - ``loss`` (optional): scalar cross-entropy loss
        """
        B, T = input_ids.shape
        if T > self.config.context_length:
            raise ValueError(
                f"Input length {T} exceeds context_length {self.config.context_length}."
            )

        device = input_ids.device

        # Token + positional embeddings
        x = self.token_emb(input_ids)
        if self.pos_emb is not None:
            x = x + self.pos_emb(T, device)
        x = self.embed_dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm + projection
        x = self.ln_f(x)
        logits: torch.Tensor = self.lm_head(x)

        result: dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            # Shift: predict token[t+1] from token[t]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=0,  # ignore <pad>
            )
            result["loss"] = loss

        return result

    # ── Utility ────────────────────────────────────────────────────────────

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts by component.

        Returns:
            Dictionary mapping component name to parameter count.
        """
        counts: dict[str, int] = {}
        for name, module in self.named_children():
            counts[name] = sum(p.numel() for p in module.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Total parameter count.

        Args:
            non_embedding: If True, subtract embedding parameters.

        Returns:
            Integer parameter count.
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_emb.embedding.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.embedding.weight.numel()
        return n


def build_model(config: ModelConfig) -> ThinkyLM:
    """Convenience factory that builds and returns a ThinkyLM model.

    Args:
        config: ModelConfig instance.

    Returns:
        Randomly initialised ThinkyLM model.
    """
    return ThinkyLM(config)
