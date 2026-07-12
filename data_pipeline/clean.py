"""
ThinkyLM — Text Cleaning Pipeline
====================================
UTF-8 loading, Unicode normalisation, whitespace cleaning,
and minimum/maximum length filtering.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterator


def clean_text(text: str, min_length: int = 20, max_length: int = 100_000) -> str | None:
    """Clean a single text string.

    Steps applied:
        1. Unicode NFC normalisation.
        2. Whitespace normalisation (collapse runs of spaces/tabs).
        3. Strip leading/trailing whitespace.
        4. Remove control characters (except newlines).
        5. Minimum/maximum length filter.

    Args:
        text: Raw input string.
        min_length: Minimum character length (returns None if shorter).
        max_length: Maximum character length (truncated).

    Returns:
        Cleaned string, or None if the text should be discarded.
    """
    # Unicode NFC normalisation
    text = unicodedata.normalize("NFC", text)

    # Remove control characters except newlines/tabs
    text = re.sub(r"[^\S\n\t ]+", " ", text)  # collapse unusual whitespace
    text = re.sub(r"[ \t]+", " ", text)        # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)     # limit consecutive newlines
    text = text.strip()

    if len(text) < min_length:
        return None
    if len(text) > max_length:
        text = text[:max_length]

    return text


def clean_file(
    path: Path,
    min_length: int = 20,
    max_length: int = 100_000,
) -> Iterator[str]:
    """Yield cleaned non-empty paragraphs from a text file.

    Args:
        path: Path to the UTF-8 text file.
        min_length: Minimum paragraph character length.
        max_length: Maximum paragraph character length.

    Yields:
        Cleaned paragraph strings.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"WARNING: Could not read {path}: {e}")
        return

    for paragraph in raw.split("\n\n"):
        cleaned = clean_text(paragraph, min_length, max_length)
        if cleaned:
            yield cleaned


def clean_directory(
    input_dir: Path,
    output_dir: Path,
    min_length: int = 20,
    max_length: int = 100_000,
) -> tuple[int, int]:
    """Clean all .txt files in a directory and write results.

    Args:
        input_dir: Source directory containing .txt files.
        output_dir: Destination directory for cleaned files.
        min_length: Minimum paragraph length.
        max_length: Maximum paragraph length.

    Returns:
        Tuple of (documents_processed, paragraphs_kept).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    docs = 0
    paragraphs = 0

    for txt_file in sorted(input_dir.rglob("*.txt")):
        cleaned_paragraphs = list(clean_file(txt_file, min_length, max_length))
        if not cleaned_paragraphs:
            continue

        out_file = output_dir / txt_file.relative_to(input_dir)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text("\n\n".join(cleaned_paragraphs), encoding="utf-8")

        docs += 1
        paragraphs += len(cleaned_paragraphs)

    return docs, paragraphs
