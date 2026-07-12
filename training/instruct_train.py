"""
ThinkyLM — Instruction Fine-Tuning
=====================================
Author: Haziq Imran

Supervised fine-tuning of a ThinkyLM base checkpoint on instruction pairs.
This uses ThinkyLM's OWN weights — NOT any external pretrained model.

Format:
    {
        "instruction": "Critically analyse the claim.",
        "input": "The claim.",
        "output": "The analysis."
    }

Usage:
    python training/instruct_train.py \\
        --checkpoint checkpoints/debug_1m/debug_1m_step000020 \\
        --data data/instructions/thinky_instructions.jsonl \\
        --config configs/debug_1m.yaml \\
        --steps 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch

torch.set_num_threads(2)

from pathlib import Path

from thinkylm.checkpoint import load_checkpoint, save_checkpoint
from thinkylm.config import ThinkyLMConfig
from thinkylm.model import build_model
from thinkylm.utils import format_number
from training.optimizer import build_optimizer
from training.scheduler import apply_lr, get_lr


def format_instruction(record: dict) -> str:
    """Format an instruction record into a single prompt string.

    Args:
        record: Dict with keys 'instruction', 'input' (optional), 'output'.

    Returns:
        Formatted string for tokenization.
    """
    parts = [f"<system>You are ThinkyLM, a careful and dry philosophical reasoner.</system>"]
    parts.append(f"<user>{record['instruction']}")
    if record.get("input"):
        parts.append(f"\n{record['input']}")
    parts.append(f"</user>")
    parts.append(f"<assistant>{record['output']}</assistant>")
    return "\n".join(parts)


def load_instruct_data(jsonl_path: Path) -> list[dict]:
    """Load instruction records from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL instruction dataset.

    Returns:
        List of instruction dicts.
    """
    records = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def instruct_train(
    cfg: ThinkyLMConfig,
    checkpoint_path: str | Path,
    data_path: Path,
    max_steps: int = 20,
    output_dir: str | Path = "checkpoints/instruct",
) -> None:
    """Run instruction fine-tuning from a base checkpoint.

    Args:
        cfg: ThinkyLMConfig.
        checkpoint_path: Path to base model checkpoint directory.
        data_path: Path to JSONL instruction dataset.
        max_steps: Number of fine-tuning steps.
        output_dir: Directory to save the fine-tuned checkpoint.
    """
    from data_pipeline.tokenize import load_tokenizer, tokenize_texts

    device = torch.device("cpu")
    torch.manual_seed(cfg.training.seed)

    print("=" * 55)
    print("ThinkyLM Instruction Fine-Tuning")
    print("=" * 55)
    print(f"  Base checkpoint : {checkpoint_path}")
    print(f"  Data            : {data_path}")
    print(f"  Max steps       : {max_steps}")
    print(f"  Output          : {output_dir}")

    # Load base model
    model = build_model(cfg.model).to(device)
    meta = load_checkpoint(model, checkpoint_path, device=str(device))
    print(f"  Base model step : {meta.get('step', 0)}")

    # Load and format instruction data
    records = load_instruct_data(data_path)
    print(f"  Instruction records: {len(records)}")

    tokenizer = load_tokenizer(cfg.tokenizer_path)
    texts = [format_instruction(r) for r in records]
    sequences = tokenize_texts(texts, tokenizer, cfg.model.context_length)

    if not sequences:
        print("ERROR: No sequences produced. Instruction texts may be too short.")
        sys.exit(1)

    print(f"  Token sequences : {len(sequences)}")

    from data_pipeline.dataset import build_dataloader
    loader = build_dataloader(
        sequences, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0
    )

    optimizer = build_optimizer(
        model,
        learning_rate=cfg.training.learning_rate * 0.1,  # Lower LR for fine-tuning
        weight_decay=0.01,
    )

    model.train()
    step = 0
    print("\n[Instruct] Training...")

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids, labels=labels)
            loss = out["loss"]
            loss.backward()

            lr = get_lr(step, max_steps, 2, cfg.training.learning_rate * 0.1)
            apply_lr(optimizer, lr)
            optimizer.step()
            optimizer.zero_grad()

            print(f"  [Instruct] step {step:3d} | loss {loss.item():.4f} | lr {lr:.2e}")
            step += 1

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=step,
        loss=loss.item(),
        config_dict=cfg.to_dict(),
        checkpoint_dir=output_dir,
        name="instruct",
    )
    print(f"\n[Instruct] Done. Checkpoint saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ThinkyLM Instruction Fine-Tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/debug_1m.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--data", type=Path, default=Path("data/instructions/thinky_instructions.jsonl")
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=str, default="checkpoints/instruct")
    args = parser.parse_args()

    cfg = ThinkyLMConfig.from_yaml(args.config)
    instruct_train(cfg, args.checkpoint, args.data, args.steps, args.output)


if __name__ == "__main__":
    main()
