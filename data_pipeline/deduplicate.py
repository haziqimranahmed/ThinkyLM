"""
ThinkyLM — Deduplication
==========================
Hash-based deduplication for text documents.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator


def dedup_paragraphs(paragraphs: list[str]) -> list[str]:
    """Remove exact-duplicate paragraphs using SHA-256 hashing.

    Args:
        paragraphs: List of text strings.

    Returns:
        Deduplicated list preserving original order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for p in paragraphs:
        digest = hashlib.sha256(p.encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            result.append(p)
    return result


def dedup_stream(texts: Iterator[str]) -> Iterator[str]:
    """Streaming deduplication of texts.

    Args:
        texts: Iterator of text strings.

    Yields:
        Unique text strings in original order.
    """
    seen: set[str] = set()
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            yield text
