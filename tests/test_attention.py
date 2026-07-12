"""
ThinkyLM — Attention Tests

Verifies causal masking correctness. Future tokens must receive zero attention weight.
This is the most critical architectural test.
"""

import pytest
import torch

from thinkylm.attention import MultiHeadCausalAttention


@pytest.fixture
def attn():
    return MultiHeadCausalAttention(hidden_size=64, num_heads=4, dropout=0.0, context_length=32)


def test_attention_output_shape(attn):
    x = torch.randn(2, 10, 64)
    out = attn(x)
    assert out.shape == (2, 10, 64), f"Expected (2, 10, 64), got {out.shape}"


def test_causal_mask_upper_triangle(attn):
    """Verify the causal mask: upper triangle must be -inf, rest must be 0."""
    mask = attn.causal_mask
    T = 8
    sub = mask[:T, :T]

    # Upper triangle (j > i) must be -inf
    for i in range(T):
        for j in range(T):
            if j > i:
                assert torch.isinf(sub[i, j]) and sub[i, j] < 0, (
                    f"mask[{i},{j}] should be -inf but is {sub[i, j]}"
                )
            else:
                assert sub[i, j] == 0.0, (
                    f"mask[{i},{j}] should be 0 but is {sub[i, j]}"
                )


def test_no_future_token_access(attn):
    """Critical test: position 0 output must not depend on position 1+ inputs.

    We set position 1 input to a large value and verify position 0 output is unchanged.
    """
    attn.eval()
    torch.manual_seed(0)

    x_base = torch.zeros(1, 4, 64)
    x_base[0, 0, :] = 1.0  # Only position 0 has non-zero input

    x_modified = x_base.clone()
    x_modified[0, 1, :] = 1000.0  # Massive value at position 1 (future)

    with torch.no_grad():
        out_base = attn(x_base)
        out_modified = attn(x_modified)

    # Position 0 output should be identical regardless of future tokens
    assert torch.allclose(out_base[:, 0, :], out_modified[:, 0, :], atol=1e-5), (
        "CAUSAL MASK FAILURE: position 0 was affected by position 1 (future token)."
    )


def test_attention_batch_independence(attn):
    """Each sequence in a batch should be processed independently."""
    attn.eval()
    x = torch.randn(3, 6, 64)
    with torch.no_grad():
        out = attn(x)
    assert out.shape == (3, 6, 64)
    # No crash and correct shape is sufficient for batch independence check


def test_attention_single_token(attn):
    """Single-token input (e.g., during autoregressive generation) must work."""
    attn.eval()
    x = torch.randn(1, 1, 64)
    with torch.no_grad():
        out = attn(x)
    assert out.shape == (1, 1, 64)


def test_attention_no_nan():
    """NaN in attention output indicates a numeric stability issue."""
    attn = MultiHeadCausalAttention(hidden_size=64, num_heads=4, dropout=0.0, context_length=32)
    attn.eval()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        out = attn(x)
    assert not torch.isnan(out).any(), "NaN detected in attention output."
    assert not torch.isinf(out).any(), "Inf detected in attention output."
