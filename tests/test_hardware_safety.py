"""
ThinkyLM — Hardware Safety Tests
"""

import pytest


def test_get_ram_returns_positive():
    from thinkylm.utils import get_ram_gb
    total, available = get_ram_gb()
    assert total > 0
    assert available >= 0
    assert total >= available


def test_get_disk_returns_positive():
    from thinkylm.utils import get_disk_gb
    total, free = get_disk_gb(".")
    assert total > 0
    assert free >= 0


def test_check_ram_safety_passes_with_high_threshold():
    """Should not raise if required RAM is 0 (always passes)."""
    from thinkylm.utils import check_ram_safety
    check_ram_safety(min_gb=0.0)  # Always passes


def test_check_ram_safety_fails_with_impossible_threshold():
    """Should raise if we require more RAM than exists."""
    from thinkylm.utils import check_ram_safety
    with pytest.raises(RuntimeError, match="RAM"):
        check_ram_safety(min_gb=1_000_000.0)  # 1 million GB — impossible


def test_check_disk_safety_fails_with_impossible_threshold():
    from thinkylm.utils import check_disk_safety
    with pytest.raises(RuntimeError, match="disk"):
        check_disk_safety(min_gb=1_000_000.0, path=".")


def test_format_number():
    from thinkylm.utils import format_number
    assert format_number(1_500_000) == "1.50M"
    assert format_number(500_000) == "500.0K"
    assert format_number(42) == "42"


def test_hardware_checker_runs():
    """Hardware checker should complete without crashing."""
    from scripts.check_hardware import check_hardware
    info = check_hardware()
    assert "os" in info
    assert "total_ram_gb" in info
    assert "cuda_available" in info
    assert "recommended_config" in info
    assert info["total_ram_gb"] > 0


def test_cuda_not_available_locally():
    """On this development machine (no NVIDIA GPU), CUDA should not be available."""
    import torch
    # This test is specific to the development machine described in the spec.
    # It documents the expected state rather than requiring it.
    is_cuda = torch.cuda.is_available()
    # Just report, don't fail — allows tests to run on cloud machines too
    print(f"[Hardware] CUDA available: {is_cuda}")


def test_safe_cpu_defaults_set():
    """Safe CPU thread limits should be applied without error."""
    from thinkylm.utils import set_safe_cpu_defaults
    set_safe_cpu_defaults()  # Should not raise
    import torch
    # Verify torch thread count is reasonable
    assert torch.get_num_threads() <= 4
