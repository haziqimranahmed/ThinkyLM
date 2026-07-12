"""
ThinkyLM — Evaluation Benchmark Runner
=========================================
Runs Thinky-Bench prompts through the model and produces an evaluation report.

Usage:
    python evaluation/benchmark.py \\
        --checkpoint checkpoints/debug_1m/debug_1m_step000020 \\
        --config configs/debug_1m.yaml \\
        --output reports/benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch

torch.set_num_threads(2)


def run_benchmark(
    checkpoint_path: Path,
    config_path: str,
    bench_path: Path,
    output_path: Path,
    max_new_tokens: int = 80,
) -> None:
    """Run all Thinky-Bench prompts through the model and save results.

    Args:
        checkpoint_path: Path to the model checkpoint directory.
        config_path: Path to the YAML configuration file.
        bench_path: Path to thinky_bench.json.
        output_path: Path to save results JSON.
        max_new_tokens: Tokens to generate per prompt.
    """
    from data_pipeline.tokenize import load_tokenizer
    from thinkylm.checkpoint import load_checkpoint
    from thinkylm.config import ThinkyLMConfig
    from thinkylm.generation import generate
    from thinkylm.model import build_model

    cfg = ThinkyLMConfig.from_yaml(config_path)
    device = torch.device("cpu")

    model = build_model(cfg.model).to(device)
    load_checkpoint(model, checkpoint_path, device=str(device))
    model.eval()

    tokenizer = load_tokenizer(cfg.tokenizer_path)
    tokenizer.token_to_id("<bos>") or 2
    eos_id = tokenizer.token_to_id("<eos>") or 3

    with bench_path.open() as f:
        bench = json.load(f)

    print(
        f"\n[Benchmark] Running Thinky-Bench — {len(bench)} prompts\n"
        "NOTE: This is a randomly initialised model. Outputs demonstrate\n"
        "      forward-pass mechanics, not reasoning ability.\n"
    )

    results = []
    for item in bench:
        prompt = item["prompt"]
        enc = tokenizer.encode(f"<bos>{prompt}")
        input_ids = torch.tensor([enc.ids], dtype=torch.long, device=device)

        t0 = time.perf_counter()
        generated = generate(
            model, input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=eos_id,
        )
        elapsed = time.perf_counter() - t0

        new_ids = generated[0, input_ids.shape[1]:].tolist()
        output_text = tokenizer.decode(new_ids)

        result = {
            "id": item["id"],
            "category": item["category"],
            "prompt": prompt,
            "model_output": output_text,
            "tokens_generated": len(new_ids),
            "elapsed_seconds": round(elapsed, 2),
            "expected_qualities": item["expected_qualities"],
            "human_score": None,
            "reviewer_notes": "",
            "disclaimer": (
                "This output is from a randomly initialised ~1M parameter model. "
                "It does not reflect reasoning ability. This is an architecture demonstration."
            ),
        }
        results.append(result)
        print(f"  [{item['id']}] {item['category']}: {len(new_ids)} tokens in {elapsed:.1f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Benchmark] Results saved → {output_path}")
    print("[Benchmark] Human scoring rubric: 0=no attempt, 1=poor, 2=partial, 3=good, 4=excellent")


def main() -> None:
    parser = argparse.ArgumentParser(description="ThinkyLM Thinky-Bench Evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=str, default="configs/debug_1m.yaml")
    parser.add_argument(
        "--bench", type=Path, default=Path("evaluation/thinky_bench.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/thinky_bench_results.json")
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()
    run_benchmark(args.checkpoint, args.config, args.bench, args.output, args.max_new_tokens)


if __name__ == "__main__":
    main()
