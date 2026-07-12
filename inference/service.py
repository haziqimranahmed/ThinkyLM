"""
ThinkyLM — Inference Service
==============================
Model loading and generation logic shared by the API and Gradio demo.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch

torch.set_num_threads(2)


class InferenceService:
    """Thread-safe inference service wrapping ThinkyLM model and tokenizer.

    Args:
        config_path: Path to YAML configuration.
        checkpoint_path: Optional path to model checkpoint.
    """

    def __init__(self, config_path: str, checkpoint_path: str | None = None) -> None:
        self._lock = threading.Lock()
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.tokenizer = None
        self.cfg = None
        self.loaded = False
        self.load_error: str | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load the model. Sets self.loaded and self.load_error."""
        try:
            from data_pipeline.tokenize import load_tokenizer
            from thinkylm.checkpoint import load_checkpoint
            from thinkylm.config import ThinkyLMConfig
            from thinkylm.model import build_model

            self.cfg = ThinkyLMConfig.from_yaml(self.config_path)
            self.model = build_model(self.cfg.model)
            self.model.eval()

            if self.checkpoint_path and Path(self.checkpoint_path).exists():
                load_checkpoint(self.model, self.checkpoint_path, device="cpu")
                self.loaded = True
                print(f"[Service] Model loaded from {self.checkpoint_path}")
            else:
                self.loaded = False
                self.load_error = (
                    "No checkpoint found. The model has random weights. "
                    "Run training first: python training/pretrain.py --config configs/debug_1m.yaml"
                )
                print(f"[Service] WARNING: {self.load_error}")

            self.tokenizer = load_tokenizer(self.cfg.tokenizer_path)

        except Exception as e:
            self.load_error = str(e)
            print(f"[Service] Model load error: {e}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> dict:
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p (nucleus) sampling parameter.
            repetition_penalty: Penalty for repeated tokens.

        Returns:
            Dict with keys: text, tokens_generated, warning (if any).
        """
        from thinkylm.generation import generate as _generate

        if self.model is None or self.tokenizer is None:
            return {"error": f"Model not loaded: {self.load_error}"}

        self.tokenizer.token_to_id("<bos>") or 2
        eos_id = self.tokenizer.token_to_id("<eos>") or 3

        with self._lock:
            enc = self.tokenizer.encode(f"<bos>{prompt}")
            input_ids = torch.tensor([enc.ids], dtype=torch.long)

            with torch.no_grad():
                generated = _generate(
                    self.model, input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=eos_id,
                )

            new_ids = generated[0, input_ids.shape[1]:].tolist()
            text = self.tokenizer.decode(new_ids)

        result = {"text": text, "tokens_generated": len(new_ids)}
        if not self.loaded:
            result["warning"] = self.load_error
        return result

    def model_info(self) -> dict:
        """Return model metadata.

        Returns:
            Dict with model configuration and status information.
        """
        if self.cfg is None:
            return {"status": "not loaded", "error": self.load_error}

        total_params = sum(p.numel() for p in self.model.parameters()) if self.model else 0
        return {
            "status": "loaded" if self.loaded else "random_weights",
            "checkpoint": self.checkpoint_path or "none",
            "total_params": total_params,
            "vocab_size": self.cfg.model.vocab_size,
            "context_length": self.cfg.model.context_length,
            "hidden_size": self.cfg.model.hidden_size,
            "num_layers": self.cfg.model.num_layers,
            "num_heads": self.cfg.model.num_heads,
            "device": "cpu",
            "disclaimer": (
                "ThinkyLM is a small educational model (~1M parameters) trained from random "
                "initialisation. It is not comparable to large commercial language models."
            ),
        }
