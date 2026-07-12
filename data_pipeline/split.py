"""
ThinkyLM — Dataset Splitting
==============================
Deterministic train / validation / test splitting.
"""

from __future__ import annotations

import random


def split_texts(
    texts: list[str],
    val_fraction: float = 0.1,
    test_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Split a list of texts into train/val/test.

    Args:
        texts: All text documents.
        val_fraction: Fraction to use for validation.
        test_fraction: Fraction to use for testing.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train, val, test) text lists.
    """
    rng = random.Random(seed)
    shuffled = texts.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))

    test = shuffled[:n_test]
    val = shuffled[n_test: n_test + n_val]
    train = shuffled[n_test + n_val:]

    return train, val, test
