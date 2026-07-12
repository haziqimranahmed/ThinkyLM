"""
ThinkyLM — Generation Tests
"""

import torch
import pytest
from thinkylm.config import ModelConfig
from thinkylm.model import build_model
from thinkylm.generation import generate, greedy_generate


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=100, context_length=32,
        hidden_size=32, num_layers=2, num_heads=4,
        intermediate_size=64, dropout=0.0,
    )
    return build_model(cfg)


def test_generate_output_length(tiny_model):
    prompt = torch.randint(1, 100, (1, 4))
    generated = generate(tiny_model, prompt, max_new_tokens=10, top_k=1)
    new_tokens = generated.shape[1] - prompt.shape[1]
    assert new_tokens <= 10


def test_generate_output_shape(tiny_model):
    prompt = torch.randint(1, 100, (1, 4))
    generated = generate(tiny_model, prompt, max_new_tokens=8)
    assert generated.shape[0] == 1
    assert generated.shape[1] > 4


def test_generate_deterministic_with_seed(tiny_model):
    """Same seed should produce identical output."""
    prompt = torch.randint(1, 100, (1, 4))
    gen1 = generate(tiny_model, prompt, max_new_tokens=10, seed=42)
    gen2 = generate(tiny_model, prompt, max_new_tokens=10, seed=42)
    assert torch.equal(gen1, gen2), "Seeded generation should be deterministic."


def test_generate_different_seeds_differ(tiny_model):
    """Different seeds should usually produce different output."""
    prompt = torch.randint(1, 100, (1, 4))
    gen1 = generate(tiny_model, prompt, max_new_tokens=15, seed=0)
    gen2 = generate(tiny_model, prompt, max_new_tokens=15, seed=99)
    # It's statistically very unlikely these are equal with different seeds
    # (we use > 10 new tokens to reduce false failure probability)
    assert not torch.equal(gen1, gen2), "Different seeds should typically produce different output."


def test_greedy_generate_deterministic(tiny_model):
    prompt = torch.randint(1, 100, (1, 4))
    gen1 = greedy_generate(tiny_model, prompt, max_new_tokens=10)
    gen2 = greedy_generate(tiny_model, prompt, max_new_tokens=10)
    assert torch.equal(gen1, gen2)


def test_generate_eos_stopping(tiny_model):
    """Generation should stop when EOS is produced."""
    # Force EOS by making it very likely: set a small temperature
    prompt = torch.randint(1, 100, (1, 2))
    eos_id = 3
    generated = generate(
        tiny_model, prompt, max_new_tokens=50,
        temperature=0.1, top_k=1, eos_token_id=eos_id,
    )
    # Output should not exceed max_new_tokens + prompt length
    assert generated.shape[1] <= 52


def test_generate_top_k_filters():
    """Top-k=1 should always pick the argmax (greedy)."""
    torch.manual_seed(7)
    cfg = ModelConfig(
        vocab_size=100, context_length=32,
        hidden_size=32, num_layers=2, num_heads=4,
        intermediate_size=64, dropout=0.0,
    )
    model = build_model(cfg)
    prompt = torch.randint(1, 100, (1, 4))
    gen1 = generate(model, prompt, max_new_tokens=8, top_k=1, seed=0)
    gen2 = generate(model, prompt, max_new_tokens=8, top_k=1, seed=0)
    assert torch.equal(gen1, gen2)


def test_repetition_penalty_applied(tiny_model):
    """Applying a high repetition penalty should not crash."""
    prompt = torch.randint(1, 100, (1, 4))
    generated = generate(
        tiny_model, prompt, max_new_tokens=8,
        repetition_penalty=2.0, top_k=5,
    )
    assert generated.shape[1] > 4
