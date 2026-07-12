"""
ThinkyLM — Tokenizer Evaluation
=================================
Reports vocabulary size, compression ratio, unknown-token rate,
average tokens per sentence, and a round-trip verification.

Usage:
    python tokenizer/evaluate_tokenizer.py --tokenizer tokenizer/generated \\
           --text tokenizer/sample_text.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def evaluate_tokenizer(tokenizer_dir: Path, text_path: Path) -> None:
    """Evaluate a trained tokenizer on a text file.

    Args:
        tokenizer_dir: Directory containing tokenizer.json.
        text_path: Path to the text file for evaluation.
    """
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("ERROR: 'tokenizers' not installed.")
        sys.exit(1)

    tok_file = tokenizer_dir / "tokenizer.json"
    if not tok_file.exists():
        print(f"ERROR: Tokenizer not found at {tok_file}")
        print("Train first: python tokenizer/train_tokenizer.py")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(str(tok_file))
    unk_id = tokenizer.token_to_id("<unk>")

    text = text_path.read_text(encoding="utf-8")
    sentences = [s.strip() for s in text.split("\n") if s.strip()]

    total_chars = sum(len(s) for s in sentences)
    total_tokens = 0
    unk_tokens = 0

    for sentence in sentences:
        enc = tokenizer.encode(sentence)
        total_tokens += len(enc.ids)
        unk_tokens += sum(1 for t in enc.ids if t == unk_id)

    compression_ratio = total_chars / max(total_tokens, 1)
    avg_tokens = total_tokens / max(len(sentences), 1)
    unk_rate = unk_tokens / max(total_tokens, 1)

    print("=" * 55)
    print("ThinkyLM Tokenizer Evaluation Report")
    print("=" * 55)
    print(f"Tokenizer path    : {tokenizer_dir}")
    print(f"Vocabulary size   : {tokenizer.get_vocab_size():,}")
    print(f"Text file         : {text_path}")
    print(f"Sentences         : {len(sentences):,}")
    print(f"Total characters  : {total_chars:,}")
    print(f"Total tokens      : {total_tokens:,}")
    print(f"Compression ratio : {compression_ratio:.2f} chars/token")
    print(f"Avg tokens/sent   : {avg_tokens:.1f}")
    print(f"Unknown-token rate: {unk_rate:.4%}")
    print("-" * 55)

    # Round-trip test
    sample = sentences[0] if sentences else "Hello world."
    enc = tokenizer.encode(sample)
    dec = tokenizer.decode(enc.ids)
    match = dec.strip() == sample.strip()
    print(f"Round-trip test   : {'PASS' if match else 'PARTIAL'}")
    print(f"  Input   : {sample[:80]}")
    print(f"  Decoded : {dec[:80]}")
    print()

    # Show first few tokens of a philosophical sentence
    demo = "Your conclusion is possible, although premise two appears to have entered the argument without identification."
    enc = tokenizer.encode(demo)
    print(f"Sample tokenisation:")
    print(f"  '{demo[:60]}...'")
    print(f"  Tokens: {enc.tokens[:20]}")
    print("=" * 55)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ThinkyLM tokenizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/generated"))
    parser.add_argument("--text", type=Path, default=Path("tokenizer/sample_text.txt"))
    args = parser.parse_args()
    evaluate_tokenizer(args.tokenizer, args.text)


if __name__ == "__main__":
    main()
