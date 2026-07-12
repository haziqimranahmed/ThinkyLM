"""
ThinkyLM — Shared Utilities
=============================
Hardware introspection, logging helpers, and safety guards.
"""

from __future__ import annotations

import os
import time

import psutil
import torch


def set_safe_cpu_defaults() -> None:
    """Apply safe thread-count defaults for CPU-only machines.

    Prevents excessive thread contention on the Intel Core i5-1135G7.
    Must be called before importing torch.distributed or starting DataLoaders.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)


def get_device(requested: str = "cpu") -> torch.device:
    """Return the best available device, respecting the requested preference.

    Args:
        requested: "cpu", "cuda", or "auto".

    Returns:
        A torch.device instance.
    """
    if requested == "cuda" or requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "cuda":
            raise RuntimeError(
                "CUDA requested but not available. "
                "Use device='cpu' for local development."
            )
    return torch.device("cpu")


def get_ram_gb() -> tuple[float, float]:
    """Return (total_gb, available_gb) for system RAM."""
    mem = psutil.virtual_memory()
    return mem.total / 1e9, mem.available / 1e9


def get_disk_gb(path: str = ".") -> tuple[float, float]:
    """Return (total_gb, free_gb) for the disk containing path."""
    usage = psutil.disk_usage(path)
    return usage.total / 1e9, usage.free / 1e9


def format_number(n: int) -> str:
    """Format an integer with M/K suffixes for readability.

    Args:
        n: Integer to format.

    Returns:
        Human-readable string like "1.3M" or "512K".
    """
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


class Timer:
    """Simple wall-clock timer.

    Usage::

        with Timer("Training") as t:
            ...
        print(t.elapsed_seconds)
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.start_time: float = 0.0
        self.elapsed_seconds: float = 0.0

    def __enter__(self) -> Timer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_seconds = time.perf_counter() - self.start_time
        if self.label:
            print(f"[Timer] {self.label}: {self.elapsed_seconds:.2f}s")


def check_ram_safety(min_gb: float = 3.0) -> None:
    """Raise RuntimeError if available RAM is below minimum.

    Args:
        min_gb: Minimum required available RAM in gigabytes.

    Raises:
        RuntimeError: If available RAM is insufficient.
    """
    _, available = get_ram_gb()
    if available < min_gb:
        raise RuntimeError(
            f"Insufficient RAM: {available:.1f} GB available, "
            f"{min_gb:.1f} GB required. Close other applications and try again."
        )


def check_disk_safety(min_gb: float = 10.0, path: str = ".") -> None:
    """Raise RuntimeError if free disk space is below minimum.

    Args:
        min_gb: Minimum required free disk in gigabytes.
        path: Path to check disk usage for.

    Raises:
        RuntimeError: If free disk space is insufficient.
    """
    _, free = get_disk_gb(path)
    if free < min_gb:
        raise RuntimeError(
            f"Insufficient disk space: {free:.1f} GB free, "
            f"{min_gb:.1f} GB required."
        )
