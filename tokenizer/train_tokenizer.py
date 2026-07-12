"""
ThinkyLM — BPE Tokenizer Training
====================================
Author: Haziq Imran

Trains an original BPE tokenizer using Hugging Face Tokenizers library.
This is NOT a pretrained or copied vocabulary. The tokenizer is trained
from the included sample corpus or any text files you supply.

Usage:
    python tokenizer/train_tokenizer.py --input tokenizer/sample_text.txt \\
           --vocab-size 4000 --output tokenizer/generated

    # With more text files:
    python tokenizer/train_tokenizer.py --input data/sample/ \\
           --vocab-size 4000 --output tokenizer/generated
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def train_tokenizer(
    input_path: Path,
    output_dir: Path,
    vocab_size: int = 4_000,
    min_frequency: int = 2,
) -> None:
    """Train a BPE tokenizer on the given text corpus.

    Args:
        input_path: Path to a text file or directory of .txt files.
        output_dir: Directory to save the trained tokenizer.
        vocab_size: Target vocabulary size.
        min_frequency: Minimum token frequency to include in vocabulary.
    """
    try:
        from tokenizers import Tokenizer, trainers
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.normalizers import NFKC
        from tokenizers.normalizers import Sequence as NormSequence
        from tokenizers.pre_tokenizers import ByteLevel
    except ImportError:
        print("ERROR: 'tokenizers' package not found. Run: pip install tokenizers")
        sys.exit(1)

    # Collect text files
    files: list[str] = []
    if input_path.is_dir():
        files = [str(p) for p in input_path.rglob("*.txt")]
    elif input_path.is_file():
        files = [str(input_path)]
    else:
        print(f"ERROR: Input path does not exist: {input_path}")
        sys.exit(1)

    if not files:
        print(f"ERROR: No .txt files found at {input_path}")
        sys.exit(1)

    print(f"[Tokenizer] Training on {len(files)} file(s)")
    print(f"[Tokenizer] Target vocabulary size: {vocab_size}")

    # Special tokens (in order of ID assignment)
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>",
                      "<system>", "<user>", "<assistant>"]

    # Build tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = NormSequence([NFKC()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    tokenizer.train(files=files, trainer=trainer)

    # Verify special tokens are at expected positions
    for idx, token in enumerate(special_tokens):
        actual = tokenizer.token_to_id(token)
        if actual != idx:
            print(
                f"WARNING: Special token '{token}' has id {actual}, expected {idx}. "
                "This may cause issues. Consider retraining."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))

    # Save a human-readable vocab file
    vocab = tokenizer.get_vocab()
    vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])
    with (output_dir / "vocab.txt").open("w", encoding="utf-8") as f:
        for token, idx in vocab_sorted:
            f.write(f"{idx}\t{token}\n")

    print(f"[Tokenizer] Saved to {output_dir}")
    print(f"[Tokenizer] Actual vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"[Tokenizer] Special tokens: {special_tokens}")
    print()
    print("Encode/decode round-trip test:")
    sample = "Does the absence of objective meaning imply that life is meaningless?"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded.ids)
    # Use ascii() to avoid Windows cp1252 encoding errors with byte-level BPE tokens
    safe_tokens = str([t for t in encoded.tokens[:15]]).encode("ascii", "replace").decode("ascii")
    safe_decoded = decoded.encode("ascii", "replace").decode("ascii")
    print(f"  Input:   {sample}")
    print(f"  Tokens:  {safe_tokens}{'...' if len(encoded.tokens) > 15 else ''}")
    print(f"  IDs:     {encoded.ids[:15]}{'...' if len(encoded.ids) > 15 else ''}")
    print(f"  Decoded: {safe_decoded}")
    print(f"  Round-trip OK: {decoded.strip() == sample.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer for ThinkyLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tokenizer/sample_text.txt"),
        help="Path to a .txt file or directory of .txt files.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=4_000,
        help="Target vocabulary size.",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tokenizer/generated"),
        help="Output directory for the tokenizer.",
    )
    args = parser.parse_args()
    train_tokenizer(args.input, args.output, args.vocab_size, args.min_frequency)


if __name__ == "__main__":
    main()
