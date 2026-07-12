# -*- coding: utf-8 -*-
"""
ThinkyLM — Hardware Safety Checker
=====================================
Author: Haziq Imran

Reports system information and recommends the safest ThinkyLM configuration.

Usage:
    python scripts/check_hardware.py
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path


def check_hardware() -> dict:
    """Collect hardware and environment information.

    Returns:
        Dictionary of hardware metrics and recommendations.
    """
    import psutil
    import torch

    info: dict = {}

    # OS
    info["os"] = platform.system()
    info["os_version"] = platform.version()[:60]
    info["python_version"] = sys.version[:30]

    # CPU
    info["cpu"] = platform.processor() or "Unknown"
    info["physical_cores"] = psutil.cpu_count(logical=False)
    info["logical_cores"] = psutil.cpu_count(logical=True)

    # RAM
    mem = psutil.virtual_memory()
    info["total_ram_gb"] = round(mem.total / 1e9, 1)
    info["available_ram_gb"] = round(mem.available / 1e9, 1)
    info["ram_used_percent"] = mem.percent

    # GPU / CUDA
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        )
    else:
        info["gpu_name"] = "None (CPU only)"
        info["gpu_memory_gb"] = 0.0

    # Disk
    disk = psutil.disk_usage(".")
    info["total_disk_gb"] = round(disk.total / 1e9, 1)
    info["free_disk_gb"] = round(disk.free / 1e9, 1)

    # Torch version
    info["torch_version"] = torch.__version__

    # Recommendation
    if not torch.cuda.is_available():
        info["recommended_config"] = "configs/debug_1m.yaml (CPU-safe)"
        info["can_run_tiny_10m"] = False
        info["can_run_portfolio_25m"] = False
    elif info.get("gpu_memory_gb", 0) >= 24:
        info["recommended_config"] = "configs/portfolio_25m.yaml (CUDA 24+ GB)"
        info["can_run_tiny_10m"] = True
        info["can_run_portfolio_25m"] = True
    elif info.get("gpu_memory_gb", 0) >= 12:
        info["recommended_config"] = "configs/tiny_10m.yaml (CUDA 12+ GB)"
        info["can_run_tiny_10m"] = True
        info["can_run_portfolio_25m"] = False
    else:
        info["recommended_config"] = "configs/debug_1m.yaml (small GPU)"
        info["can_run_tiny_10m"] = False
        info["can_run_portfolio_25m"] = False

    # Safety warnings
    warnings = []
    if info["available_ram_gb"] < 3.0:
        warnings.append(f"LOW RAM: Only {info['available_ram_gb']} GB available. Close applications.")
    if info["free_disk_gb"] < 2.0:
        warnings.append(f"LOW DISK: Only {info['free_disk_gb']} GB free.")
    if not info["cuda_available"]:
        warnings.append("No CUDA: Use only debug_1m.yaml locally.")
    info["warnings"] = warnings

    return info


def print_report(info: dict) -> None:
    """Print a formatted hardware report.

    Args:
        info: Hardware info dict from check_hardware().
    """
    print("=" * 60)
    print("ThinkyLM — Hardware Safety Report")
    print("=" * 60)
    print(f"  OS               : {info['os']} {info['os_version'][:30]}")
    print(f"  Python           : {info['python_version']}")
    print(f"  PyTorch          : {info['torch_version']}")
    print()
    print(f"  CPU              : {info['cpu'][:50]}")
    print(f"  Physical cores   : {info['physical_cores']}")
    print(f"  Logical cores    : {info['logical_cores']}")
    print()
    print(f"  Total RAM        : {info['total_ram_gb']} GB")
    print(f"  Available RAM    : {info['available_ram_gb']} GB ({100-info['ram_used_percent']:.0f}% free)")
    print()
    print(f"  CUDA available   : {info['cuda_available']}")
    print(f"  GPU              : {info['gpu_name']}")
    if info["cuda_available"]:
        print(f"  GPU memory       : {info['gpu_memory_gb']} GB")
    print()
    print(f"  Total disk       : {info['total_disk_gb']} GB")
    print(f"  Free disk        : {info['free_disk_gb']} GB")
    print()
    print(f"  Recommended cfg  : {info['recommended_config']}")
    print(f"  Can run tiny_10m : {info['can_run_tiny_10m']}")
    print(f"  Can run 25m      : {info['can_run_portfolio_25m']}")
    print()

    if info["warnings"]:
        print("  WARNINGS:")
        for w in info["warnings"]:
            print(f"     - {w}")
    else:
        print("  All safety checks passed.")
    print("=" * 60)


def main() -> None:
    info = check_hardware()
    print_report(info)


if __name__ == "__main__":
    main()
