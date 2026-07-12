"""
ThinkyLM — Tokenizer Tests
"""

import pytest
from pathlib import Path


TOKENIZER_PATH = Path("tokenizer/generated")


def _tokenizer_available():
    return (TOKENIZER_PATH / "tokenizer.json").exists()


@pytest.fixture(scope="module")
def tokenizer():
    if not _tokenizer_available():
        pytest.skip("Tokenizer not trained yet. Run: python tokenizer/train_tokenizer.py")
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZER_PATH / "tokenizer.json"))


def test_tokenizer_special_tokens(tokenizer):
    special = ["<pad>", "<unk>", "<bos>", "<eos>", "<system>", "<user>", "<assistant>"]
    for token in special:
        tok_id = tokenizer.token_to_id(token)
        assert tok_id is not None, f"Special token '{token}' not found in vocabulary."


def test_tokenizer_vocab_size(tokenizer):
    vocab_size = tokenizer.get_vocab_size()
    assert vocab_size > 100, "Vocabulary seems too small."
    assert vocab_size <= 8_000, "Vocabulary larger than expected for debug config."


def test_tokenizer_encode_decode_roundtrip(tokenizer):
    """Encoding then decoding should recover the original text."""
    sample = "The examined life is worth living."
    enc = tokenizer.encode(sample)
    dec = tokenizer.decode(enc.ids)
    assert dec.strip() == sample.strip(), (
        f"Round-trip failed:\n  Input:   '{sample}'\n  Decoded: '{dec}'"
    )


def test_tokenizer_encode_returns_ids(tokenizer):
    enc = tokenizer.encode("Philosophy begins in wonder.")
    assert isinstance(enc.ids, list)
    assert len(enc.ids) > 0
    assert all(isinstance(i, int) for i in enc.ids)


def test_tokenizer_pad_id_zero(tokenizer):
    pad_id = tokenizer.token_to_id("<pad>")
    assert pad_id == 0, f"Expected <pad> at ID 0, got {pad_id}"


def test_tokenizer_bos_id(tokenizer):
    bos_id = tokenizer.token_to_id("<bos>")
    assert bos_id == 2, f"Expected <bos> at ID 2, got {bos_id}"


def test_tokenizer_sample_output(tokenizer):
    text = "Your conclusion is possible, although premise two appears to have entered the argument."
    enc = tokenizer.encode(text)
    assert len(enc.tokens) > 5
    assert len(enc.ids) == len(enc.tokens)
