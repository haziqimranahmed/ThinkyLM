"""
ThinkyLM — Causal Language Model Pretraining
==============================================
Author: Haziq Imran

Trains ThinkyLM from random initialisation. No pretrained weights are loaded.

SAFE DEFAULTS:
- CPU-only, 2 threads, batch_size=2, max_steps=20 (debug_1m config)
- Refuses to run cloud configs without CUDA unless --override-safety-check

Usage:
    python training/pretrain.py --config configs/debug_1m.yaml
    python training/pretrain.py --config configs/debug_1m.yaml --resume checkpoints/debug_1m/checkpoint_step000020
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

# Safe CPU defaults — must be set before torch import
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch

torch.set_num_threads(2)

from pathlib import Path

from data_pipeline.clean import clean_file
from data_pipeline.dataset import build_dataloader
from data_pipeline.deduplicate import dedup_paragraphs
from data_pipeline.split import split_texts
from thinkylm.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from thinkylm.config import ThinkyLMConfig
from thinkylm.model import build_model
from thinkylm.utils import (
    check_disk_safety,
    check_ram_safety,
    format_number,
    get_disk_gb,
    get_ram_gb,
)
from training.monitor import TrainingMonitor
from training.optimizer import build_optimizer
from training.scheduler import apply_lr, get_lr

# ──────────────────────────────────────────────────────────────────────────────
# Signal handling — graceful Ctrl+C
# ──────────────────────────────────────────────────────────────────────────────
_INTERRUPTED = False


def _handle_sigint(signum, frame):
    global _INTERRUPTED
    print("\n[Pretrain] Ctrl+C received — will save checkpoint and stop gracefully.")
    _INTERRUPTED = True


signal.signal(signal.SIGINT, _handle_sigint)


# ──────────────────────────────────────────────────────────────────────────────
# Safety checks
# ──────────────────────────────────────────────────────────────────────────────

def safety_check(cfg: ThinkyLMConfig, override: bool = False) -> None:
    """Pre-flight safety checks. Stops with an informative message on failure.

    Args:
        cfg: ThinkyLMConfig instance.
        override: If True, skip checks (use only when explicitly requested).
    """
    if override:
        print("[Safety] WARNING: Safety checks overridden by user.")
        return

    # RAM check
    check_ram_safety(min_gb=3.0)

    # Disk check
    check_disk_safety(min_gb=2.0, path=".")

    # CUDA requirement for cloud configs
    training_device = cfg.training.device
    if training_device == "cuda" and not torch.cuda.is_available():
        print(
            "\n[Safety] STOP: Configuration requires CUDA but no GPU is available.\n"
            "  This configuration is intended for cloud training.\n"
            "  Use configs/debug_1m.yaml for local CPU training.\n"
            "  Use --override-safety-check ONLY if you know what you are doing.\n"
        )
        sys.exit(1)

    # Parameter limit for local training
    estimated = cfg.estimate_params()
    if training_device == "cpu" and estimated > cfg.MAX_LOCAL_PARAMS:
        print(
            f"\n[Safety] STOP: Estimated parameters ({format_number(estimated)}) exceed "
            f"the local CPU limit ({format_number(cfg.MAX_LOCAL_PARAMS)}).\n"
            "  Use a cloud config with CUDA for larger models.\n"
        )
        sys.exit(1)

    # Context length for local training
    if training_device == "cpu" and cfg.model.context_length > cfg.MAX_LOCAL_CONTEXT:
        print(
            f"\n[Safety] STOP: context_length ({cfg.model.context_length}) exceeds "
            f"the local limit ({cfg.MAX_LOCAL_CONTEXT}).\n"
        )
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helper
# ──────────────────────────────────────────────────────────────────────────────

def load_training_data(cfg: ThinkyLMConfig):
    """Load, clean, deduplicate, split, and tokenize the training corpus.

    Args:
        cfg: ThinkyLMConfig with data and model configuration.

    Returns:
        Tuple of (train_sequences, val_sequences).
    """
    data_path = Path(cfg.data.train_path)
    if not data_path.exists():
        print(
            f"[Data] Data path not found: {data_path}\n"
            "  Creating sample data first..."
        )
        from data_pipeline.prepare_sample_data import prepare_sample_data
        prepare_sample_data(data_path)

    print(f"[Data] Loading from {data_path}")
    raw_texts: list[str] = []
    for txt_file in sorted(data_path.rglob("*.txt")):
        paragraphs = list(
            clean_file(txt_file, cfg.data.min_length, cfg.data.max_length)
        )
        raw_texts.extend(paragraphs)

    raw_texts = dedup_paragraphs(raw_texts)
    print(f"[Data] {len(raw_texts)} paragraphs after deduplication")

    train_texts, val_texts, _ = split_texts(
        raw_texts,
        val_fraction=cfg.data.val_split,
        test_fraction=cfg.data.test_split,
        seed=cfg.training.seed,
    )
    print(f"[Data] Train: {len(train_texts)} | Val: {len(val_texts)}")

    from data_pipeline.tokenize import load_tokenizer, tokenize_texts

    tokenizer = load_tokenizer(cfg.tokenizer_path)
    train_seqs = tokenize_texts(
        train_texts, tokenizer, cfg.model.context_length
    )
    val_seqs = tokenize_texts(val_texts, tokenizer, cfg.model.context_length)

    if not train_seqs:
        print(
            "[Data] ERROR: No training sequences produced.\n"
            "  The corpus may be too small. Add more text to data/sample/."
        )
        sys.exit(1)

    print(f"[Data] Train sequences: {len(train_seqs)} | Val sequences: {len(val_seqs)}")
    return train_seqs, val_seqs


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_seqs, batch_size, device) -> tuple[float, float]:
    """Run validation and return (avg_loss, perplexity).

    Args:
        model: ThinkyLM model.
        val_seqs: Validation token sequences.
        batch_size: Evaluation batch size.
        device: Torch device.

    Returns:
        Tuple of (average cross-entropy loss, perplexity).
    """
    if not val_seqs:
        return 0.0, 1.0

    model.eval()
    total_loss = 0.0
    n_batches = 0

    loader = build_dataloader(
        val_seqs, batch_size=batch_size, shuffle=False, num_workers=0
    )
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids, labels=labels)
        total_loss += out["loss"].item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    perplexity = min(2**avg_loss, 1e6)
    model.train()
    return avg_loss, perplexity


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def pretrain(
    cfg: ThinkyLMConfig,
    resume_path: str | Path | None = None,
    override_safety: bool = False,
) -> None:
    """Main pretraining entry point.

    Args:
        cfg: Full ThinkyLMConfig.
        resume_path: Optional checkpoint directory to resume from.
        override_safety: Skip hardware safety checks (dangerous).
    """
    safety_check(cfg, override=override_safety)

    torch.manual_seed(cfg.training.seed)
    device = torch.device(cfg.training.device if torch.cuda.is_available()
                          else "cpu")

    # ── Print pre-training summary ──────────────────────────────────────────
    total_ram, avail_ram = get_ram_gb()
    _, free_disk = get_disk_gb(".")
    estimated_params = cfg.estimate_params()

    print("\n" + "=" * 60)
    print("ThinkyLM Pretraining — Haziq Imran")
    print("=" * 60)
    print(f"  Config           : {cfg.name}")
    print(f"  Device           : {device}")
    print(f"  Estimated params : {format_number(estimated_params)}")
    print(f"  Batch size       : {cfg.training.batch_size}")
    print(f"  Eff. batch size  : {cfg.training.batch_size * cfg.training.gradient_accumulation_steps}")
    print(f"  Max steps        : {cfg.training.max_steps}")
    print(f"  Context length   : {cfg.model.context_length}")
    print(f"  RAM available    : {avail_ram:.1f} / {total_ram:.1f} GB")
    print(f"  Free disk        : {free_disk:.1f} GB")
    print(f"  Max runtime      : {cfg.training.max_runtime_minutes} min")
    print("=" * 60)

    # ── Build model ─────────────────────────────────────────────────────────
    model = build_model(cfg.model).to(device)
    optimizer = build_optimizer(
        model,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    start_step = 0

    # ── Resume from checkpoint ──────────────────────────────────────────────
    if resume_path:
        meta = load_checkpoint(model, resume_path, optimizer, str(device))
        start_step = meta.get("step", 0)
        print(f"[Pretrain] Resuming from step {start_step}")
    elif (latest := find_latest_checkpoint(cfg.checkpoint_dir)) is not None:
        print(f"[Pretrain] Found checkpoint: {latest} — pass --resume to load it.")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_seqs, val_seqs = load_training_data(cfg)
    total_tokens = len(train_seqs) * cfg.model.context_length
    print(f"[Data] Total training tokens: {total_tokens:,}")

    train_loader = build_dataloader(
        train_seqs,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        persistent_workers=cfg.training.persistent_workers,
    )

    # ── Monitor ──────────────────────────────────────────────────────────────
    monitor = TrainingMonitor(
        log_dir=cfg.log_dir,
        max_runtime_minutes=cfg.training.max_runtime_minutes,
        use_tensorboard=True,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    global _INTERRUPTED

    step = start_step
    accum_loss = 0.0
    optimizer.zero_grad()

    print("\n[Pretrain] Starting training loop...")
    try:
        from tqdm import tqdm
        pbar = tqdm(total=cfg.training.max_steps - start_step, desc="Training", ncols=80)
    except ImportError:
        pbar = None

    while step < cfg.training.max_steps and not _INTERRUPTED:
        for batch in train_loader:
            if step >= cfg.training.max_steps or _INTERRUPTED:
                break

            # Time limit
            if monitor.time_limit_reached():
                print(
                    f"\n[Pretrain] Runtime limit ({cfg.training.max_runtime_minutes} min) reached."
                )
                _INTERRUPTED = True
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            out = model(input_ids, labels=labels)
            loss = out["loss"] / cfg.training.gradient_accumulation_steps
            loss.backward()
            accum_loss += loss.item()

            # Gradient accumulation
            if (step + 1) % cfg.training.gradient_accumulation_steps == 0 or \
               step == cfg.training.max_steps - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)

                lr = get_lr(
                    step=step,
                    max_steps=cfg.training.max_steps,
                    warmup_steps=cfg.training.warmup_steps,
                    learning_rate=cfg.training.learning_rate,
                )
                apply_lr(optimizer, lr)
                optimizer.step()
                optimizer.zero_grad()

                monitor.log_train_loss(accum_loss, step)
                monitor.log_lr(lr, step)
                monitor.log_ram(step)

                if step % 5 == 0:
                    monitor.print_status(step, accum_loss, lr)

                accum_loss = 0.0

            step += 1
            if pbar:
                pbar.update(1)

            # Validation
            if step % cfg.training.eval_interval == 0 and val_seqs:
                val_loss, ppl = evaluate(
                    model, val_seqs, cfg.training.batch_size, device
                )
                monitor.log_val_loss(val_loss, step)
                print(
                    f"\n[Val] step {step} | val_loss={val_loss:.4f} | perplexity={ppl:.2f}\n"
                )

            # Checkpoint
            if step % cfg.training.checkpoint_interval == 0 or \
               step == cfg.training.max_steps:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    loss=accum_loss,
                    config_dict=cfg.to_dict(),
                    checkpoint_dir=cfg.checkpoint_dir,
                    name=cfg.name,
                )

    if pbar:
        pbar.close()

    # Final checkpoint on interruption
    if _INTERRUPTED and step > start_step:
        print("[Pretrain] Saving emergency checkpoint...")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=step,
            loss=accum_loss,
            config_dict=cfg.to_dict(),
            checkpoint_dir=cfg.checkpoint_dir,
            name=f"{cfg.name}_interrupted",
        )

    monitor.close()
    elapsed = monitor.elapsed_minutes()
    print(f"\n[Pretrain] Done. Steps: {step} | Elapsed: {elapsed:.1f} min")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ThinkyLM Causal Language Model Pretraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/debug_1m.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from.",
    )
    parser.add_argument(
        "--override-safety-check",
        action="store_true",
        default=False,
        help="Bypass hardware safety checks. Use with caution.",
    )
    args = parser.parse_args()

    cfg = ThinkyLMConfig.from_yaml(args.config)
    pretrain(cfg, resume_path=args.resume, override_safety=args.override_safety_check)


if __name__ == "__main__":
    main()
