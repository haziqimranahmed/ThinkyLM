"""
ThinkyLM — Checkpoint Evaluator
==================================
Evaluates a checkpoint on perplexity and generation quality metrics.

Usage:
    python evaluation/evaluate_checkpoint.py \\
        --checkpoint checkpoints/debug_1m/debug_1m_step000020 \\
        --config configs/debug_1m.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch

torch.set_num_threads(2)


def evaluate_checkpoint(checkpoint_path: Path, config_path: str) -> dict:
    """Evaluate a checkpoint and return metrics.

    Args:
        checkpoint_path: Path to checkpoint directory.
        config_path: Path to YAML configuration.

    Returns:
        Dictionary of evaluation metrics.
    """
    from data_pipeline.clean import clean_file
    from data_pipeline.tokenize import load_tokenizer, tokenize_texts
    from evaluation.perplexity import compute_perplexity
    from evaluation.generation_metrics import measure_generation
    from thinkylm.checkpoint import load_checkpoint
    from thinkylm.config import ThinkyLMConfig
    from thinkylm.model import build_model
    from thinkylm.utils import get_ram_gb

    cfg = ThinkyLMConfig.from_yaml(config_path)
    device = torch.device("cpu")

    model = build_model(cfg.model).to(device)
    meta = load_checkpoint(model, checkpoint_path, device=str(device))

    tokenizer = load_tokenizer(cfg.tokenizer_path)
    bos_id = tokenizer.token_to_id("<bos>") or 2
    eos_id = tokenizer.token_to_id("<eos>") or 3

    # Build eval sequences
    data_path = Path(cfg.data.train_path)
    texts = []
    if data_path.exists():
        for txt_file in sorted(data_path.rglob("*.txt")):
            texts.extend(list(clean_file(txt_file)))

    sequences = tokenize_texts(texts, tokenizer, cfg.model.context_length)
    loss, ppl = compute_perplexity(model, sequences, batch_size=2, device="cpu") if sequences else (0.0, 1.0)

    # Generation speed
    prompt_text = "The nature of consciousness is"
    enc = tokenizer.encode(f"<bos>{prompt_text}")
    prompt_ids = torch.tensor([enc.ids], dtype=torch.long)
    gen_metrics = measure_generation(model, prompt_ids, max_new_tokens=30)

    # Model size
    total_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    _, avail_ram = get_ram_gb()

    metrics = {
        "checkpoint": str(checkpoint_path),
        "step": meta.get("step", 0),
        "val_loss": round(loss, 4),
        "perplexity": round(ppl, 2),
        "total_params": total_params,
        "model_size_mb": round(param_bytes / 1e6, 2),
        "tokens_per_second": round(gen_metrics["tokens_per_second"], 1),
        "unique_token_ratio": round(gen_metrics["unique_ratio"], 3),
        "bigram_repetition_rate": round(gen_metrics["bigram_repetition_rate"], 3),
        "available_ram_gb": round(avail_ram, 1),
        "disclaimer": (
            "This is a ~1M parameter educational model trained from scratch. "
            "Perplexity and generation quality should be interpreted as "
            "architecture validation, not production benchmarks."
        ),
    }

    print("\n[Evaluate] Checkpoint Evaluation Report")
    print("=" * 50)
    for k, v in metrics.items():
        if k != "disclaimer":
            print(f"  {k:<28}: {v}")
    print(f"\n  NOTE: {metrics['disclaimer']}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="ThinkyLM Checkpoint Evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=str, default="configs/debug_1m.yaml")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    metrics = evaluate_checkpoint(args.checkpoint, args.config)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[Evaluate] Saved → {args.output}")


if __name__ == "__main__":
    main()
