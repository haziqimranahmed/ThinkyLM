"""
ThinkyLM — Dataset Tokenization
==================================
Tokenizes text files into fixed-length integer sequences for training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator


def load_tokenizer(tokenizer_path: str | Path):
    """Load a trained tokenizer from disk.

    Args:
        tokenizer_path: Path to the tokenizer directory (contains tokenizer.json).

    Returns:
        Tokenizers.Tokenizer instance.

    Raises:
        FileNotFoundError: If tokenizer.json does not exist.
        ImportError: If the tokenizers package is not installed.
    """
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("ERROR: 'tokenizers' not installed. Run: pip install tokenizers")
        sys.exit(1)

    tok_file = Path(tokenizer_path) / "tokenizer.json"
    if not tok_file.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tok_file}\n"
            "Train it first: python tokenizer/train_tokenizer.py"
        )
    return Tokenizer.from_file(str(tok_file))


def tokenize_texts(
    texts: list[str],
    tokenizer,
    context_length: int = 128,
    bos_id: int = 2,
    eos_id: int = 3,
) -> list[list[int]]:
    """Tokenize texts and chunk into fixed-length sequences.

    Args:
        texts: List of text strings.
        tokenizer: Trained Tokenizers.Tokenizer.
        context_length: Target sequence length.
        bos_id: Beginning-of-sequence token ID.
        eos_id: End-of-sequence token ID.

    Returns:
        List of integer sequences, each of length context_length.
    """
    all_tokens: list[int] = []

    for text in texts:
        enc = tokenizer.encode(text)
        all_tokens.append(bos_id)
        all_tokens.extend(enc.ids)
        all_tokens.append(eos_id)

    # Chunk into fixed-length windows (drop the last incomplete chunk)
    sequences: list[list[int]] = []
    for i in range(0, len(all_tokens) - context_length, context_length):
        chunk = all_tokens[i: i + context_length]
        if len(chunk) == context_length:
            sequences.append(chunk)

    return sequences


def save_sequences(sequences: list[list[int]], output_path: Path) -> None:
    """Save tokenized sequences to a JSON lines file.

    Args:
        sequences: List of integer token ID sequences.
        output_path: Path to the output .jsonl file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for seq in sequences:
            f.write(json.dumps(seq) + "\n")
    print(f"[Tokenize] Saved {len(sequences):,} sequences → {output_path}")


def load_sequences(path: Path) -> list[list[int]]:
    """Load tokenized sequences from a JSON lines file.

    Args:
        path: Path to the .jsonl file created by save_sequences.

    Returns:
        List of integer sequences.
    """
    sequences: list[list[int]] = []
    with path.open("r") as f:
        for line in f:
            sequences.append(json.loads(line))
    return sequences
