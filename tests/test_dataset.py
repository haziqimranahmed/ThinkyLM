"""
ThinkyLM — Dataset Tests
"""

import pytest
import torch

from data_pipeline.dataset import TokenSequenceDataset, build_dataloader


@pytest.fixture
def sample_sequences():
    # 10 sequences of length 16
    return [[i % 50 + 1 for i in range(16)] for _ in range(10)]


def test_dataset_length(sample_sequences):
    ds = TokenSequenceDataset(sample_sequences)
    assert len(ds) == 10


def test_dataset_item_shape(sample_sequences):
    ds = TokenSequenceDataset(sample_sequences)
    item = ds[0]
    assert "input_ids" in item
    assert "labels" in item
    assert item["input_ids"].shape == (16,)
    assert item["labels"].shape == (16,)


def test_dataset_item_types(sample_sequences):
    ds = TokenSequenceDataset(sample_sequences)
    item = ds[0]
    assert item["input_ids"].dtype == torch.long
    assert item["labels"].dtype == torch.long


def test_dataset_labels_equal_input(sample_sequences):
    """Labels should equal input_ids (loss shift is done inside the model)."""
    ds = TokenSequenceDataset(sample_sequences)
    item = ds[0]
    assert torch.equal(item["input_ids"], item["labels"])


def test_dataset_stats(sample_sequences):
    ds = TokenSequenceDataset(sample_sequences)
    stats = ds.stats()
    assert stats["num_sequences"] == 10
    assert stats["seq_len"] == 16
    assert stats["total_tokens"] == 160


def test_dataloader_batch_shape(sample_sequences):
    loader = build_dataloader(sample_sequences, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (2, 16)
    assert batch["labels"].shape == (2, 16)


def test_dataset_empty_raises():
    with pytest.raises(ValueError):
        TokenSequenceDataset([])


def test_tokenize_texts_produces_sequences():
    """Test tokenize_texts produces fixed-length chunks."""
    from data_pipeline.tokenize import tokenize_texts

    class MockEnc:
        def __init__(self, ids):
            self.ids = ids

    class MockTokenizer:
        def encode(self, text):
            return MockEnc(list(range(len(text.split()))))

    texts = ["word " * 200]  # 200 tokens
    seqs = tokenize_texts(texts, MockTokenizer(), context_length=32, bos_id=2, eos_id=3)
    assert len(seqs) > 0
    assert all(len(s) == 32 for s in seqs)
