"""
ThinkyLM — Training Monitor
=============================
TensorBoard logging, RAM/disk monitoring, and graceful runtime enforcement.
"""

from __future__ import annotations

import time
from pathlib import Path

import psutil


class TrainingMonitor:
    """Monitors training metrics, resources, and enforces time limits.

    Args:
        log_dir: Directory for TensorBoard logs.
        max_runtime_minutes: Maximum allowed training time (wall clock).
        use_tensorboard: Whether to initialise TensorBoard SummaryWriter.
    """

    def __init__(
        self,
        log_dir: str | Path = "runs",
        max_runtime_minutes: float = 3.0,
        use_tensorboard: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.max_runtime_minutes = max_runtime_minutes
        self.start_time = time.time()
        self.writer: object | None = None  # SummaryWriter or None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.log_dir))
                print(f"[Monitor] TensorBoard → {self.log_dir}")
            except ImportError:
                print("[Monitor] TensorBoard not available. Skipping.")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value to TensorBoard.

        Args:
            tag: Metric name.
            value: Metric value.
            step: Training step.
        """
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_lr(self, lr: float, step: int) -> None:
        self.log_scalar("train/learning_rate", lr, step)

    def log_train_loss(self, loss: float, step: int) -> None:
        self.log_scalar("train/loss", loss, step)

    def log_val_loss(self, loss: float, step: int) -> None:
        self.log_scalar("val/loss", loss, step)
        perplexity = min(2**loss, 1e6)  # Cap to avoid overflow
        self.log_scalar("val/perplexity", perplexity, step)

    def log_ram(self, step: int) -> None:
        mem = psutil.virtual_memory()
        self.log_scalar("system/ram_used_gb", mem.used / 1e9, step)
        self.log_scalar("system/ram_available_gb", mem.available / 1e9, step)

    def elapsed_minutes(self) -> float:
        """Return wall-clock minutes elapsed since monitor creation."""
        return (time.time() - self.start_time) / 60.0

    def time_limit_reached(self) -> bool:
        """Return True if the configured runtime limit has been exceeded."""
        return self.elapsed_minutes() >= self.max_runtime_minutes

    def print_status(self, step: int, loss: float, lr: float) -> None:
        """Print a human-readable training status line.

        Args:
            step: Current step.
            loss: Current training loss.
            lr: Current learning rate.
        """
        mem = psutil.virtual_memory()
        elapsed = self.elapsed_minutes()
        print(
            f"  step {step:5d} | loss {loss:.4f} | lr {lr:.2e} | "
            f"ram {mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB | "
            f"elapsed {elapsed:.1f} min"
        )

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
