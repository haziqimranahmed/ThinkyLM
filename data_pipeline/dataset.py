"""
ThinkyLM — PyTorch Dataset
============================
Memory-efficient Dataset and DataLoader wrappers for token sequences.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class TokenSequenceDataset(Dataset):
    """Dataset of fixed-length token ID sequences.

    Each sample is a pair (input_ids, labels) where labels = input_ids
    (shifted internally by the model's loss function for causal LM training).

    Args:
        sequences: List of token ID lists, all the same length.
    """

    def __init__(self, sequences: list[list[int]]) -> None:
        if not sequences:
            raise ValueError("sequences must not be empty.")
        self.sequences = sequences
        self.seq_len = len(sequences[0])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = torch.tensor(self.sequences[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}

    def stats(self) -> dict[str, int]:
        """Return dataset statistics.

        Returns:
            Dict with num_sequences, seq_len, total_tokens.
        """
        return {
            "num_sequences": len(self.sequences),
            "seq_len": self.seq_len,
            "total_tokens": len(self.sequences) * self.seq_len,
        }


def build_dataloader(
    sequences: list[list[int]],
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    """Build a DataLoader from token sequences with safe CPU defaults.

    Args:
        sequences: Token ID sequences.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data.
        num_workers: DataLoader worker processes (0 = main process only).
        pin_memory: Pin memory for GPU transfers (False on CPU).
        persistent_workers: Keep workers alive between epochs (False on CPU).

    Returns:
        Configured DataLoader instance.
    """
    dataset = TokenSequenceDataset(sequences)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )


def build_dataset_from_texts(
    texts: list[str],
    tokenizer_path: str | Path,
    context_length: int = 128,
) -> TokenSequenceDataset:
    """Convenience function: tokenize texts and return a dataset.

    Args:
        texts: List of text strings.
        tokenizer_path: Path to the tokenizer directory.
        context_length: Fixed sequence length.

    Returns:
        TokenSequenceDataset ready for training.
    """
    from data_pipeline.tokenize import load_tokenizer, tokenize_texts

    tokenizer = load_tokenizer(tokenizer_path)
    sequences = tokenize_texts(texts, tokenizer, context_length)
    if not sequences:
        raise ValueError(
            "No sequences produced. The corpus may be too small "
            f"for context_length={context_length}. "
            "Add more text to data/sample/ and try again."
        )
    return TokenSequenceDataset(sequences)
