"""
ThinkyLM — CLI Inference
==========================
Interactive command-line interface for text generation.

Usage:
    python inference/cli.py --checkpoint checkpoints/debug_1m/debug_1m_step000020
    python inference/cli.py --checkpoint ... --prompt "Your argument assumes"
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch

torch.set_num_threads(2)

from pathlib import Path


def run_cli(
    checkpoint_path: Path,
    config_path: str,
    prompt: str | None = None,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> None:
    """Run the ThinkyLM CLI inference loop.

    Args:
        checkpoint_path: Path to checkpoint directory.
        config_path: Path to YAML configuration.
        prompt: Optional single prompt (if None, enters interactive loop).
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling.
        top_p: Top-p sampling.
        repetition_penalty: Repetition penalty.
    """
    from data_pipeline.tokenize import load_tokenizer
    from thinkylm.checkpoint import load_checkpoint
    from thinkylm.config import ThinkyLMConfig
    from thinkylm.generation import generate
    from thinkylm.model import build_model

    cfg = ThinkyLMConfig.from_yaml(config_path)
    device = torch.device("cpu")

    print("[ThinkyLM CLI] Loading model...")
    model = build_model(cfg.model).to(device)
    meta = load_checkpoint(model, checkpoint_path, device=str(device))
    model.eval()

    tokenizer = load_tokenizer(cfg.tokenizer_path)
    bos_id = tokenizer.token_to_id("<bos>") or 2
    eos_id = tokenizer.token_to_id("<eos>") or 3

    print(
        f"\n[ThinkyLM CLI] Ready | step={meta.get('step', 0)} | "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    print(
        "DISCLAIMER: This is a small educational model (~1M params) trained from scratch.\n"
        "            It does not exhibit genuine reasoning. This demonstrates architecture.\n"
    )

    def generate_response(text: str) -> str:
        enc = tokenizer.encode(f"<bos>{text}")
        input_ids = torch.tensor([enc.ids], dtype=torch.long, device=device)
        with torch.no_grad():
            generated = generate(
                model, input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=eos_id,
            )
        new_ids = generated[0, input_ids.shape[1]:].tolist()
        return tokenizer.decode(new_ids)

    if prompt:
        print(f"Prompt: {prompt}")
        print(f"Output: {generate_response(prompt)}")
        return

    # Interactive loop
    print("Enter a prompt (or 'quit' to exit):\n")
    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[ThinkyLM CLI] Goodbye.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("[ThinkyLM CLI] Goodbye.")
            break

        if not user_input:
            continue

        response = generate_response(user_input)
        print(f"\n{response}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ThinkyLM CLI Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=str, default="configs/debug_1m.yaml")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    args = parser.parse_args()

    run_cli(
        args.checkpoint, args.config, args.prompt,
        args.max_new_tokens, args.temperature, args.top_k,
        args.top_p, args.repetition_penalty,
    )


if __name__ == "__main__":
    main()
