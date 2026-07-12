"""
ThinkyLM — Gradio Demo
========================
Author: Haziq Imran

Lightweight Gradio interface for ThinkyLM generation.

Usage:
    python demo/gradio_app.py --checkpoint checkpoints/debug_1m/debug_1m_step000020
    python demo/gradio_app.py  # Runs with random weights (no checkpoint)
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch

torch.set_num_threads(2)


def build_demo(service) -> "gr.Blocks":
    """Build the Gradio interface.

    Args:
        service: InferenceService instance.

    Returns:
        Gradio Blocks app (not launched).
    """
    import gradio as gr

    info = service.model_info()
    model_loaded = service.loaded

    status_message = (
        f"✅ Checkpoint loaded | {info.get('total_params', 0):,} parameters"
        if model_loaded
        else f"⚠️ No checkpoint — random weights | {info.get('total_params', 0):,} parameters"
    )

    DISCLAIMER = (
        "**Research Model Disclaimer**: ThinkyLM is a small educational model (~1M parameters) "
        "trained from random initialisation. Outputs are syntactic patterns, not genuine reasoning. "
        "This demonstrates end-to-end ML engineering, not commercial AI."
    )

    def generate(prompt, max_new_tokens, temperature, top_k, top_p, rep_penalty):
        if not prompt.strip():
            return "Please enter a prompt."
        result = service.generate(
            prompt=prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            repetition_penalty=float(rep_penalty),
        )
        if "error" in result:
            return f"Error: {result['error']}"
        text = result["text"]
        if result.get("warning"):
            text = f"[{result['warning']}]\n\n{text}"
        return text

    with gr.Blocks(
        title="ThinkyLM Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(f"# 🧠 ThinkyLM")
        gr.Markdown(
            "An independently implemented decoder-only Transformer for critical and philosophical argumentation.\n\n"
            f"**Author**: Haziq Imran  |  **Status**: {status_message}"
        )
        gr.Markdown(DISCLAIMER)

        with gr.Row():
            with gr.Column(scale=2):
                prompt_box = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter a philosophical claim or question...",
                    lines=4,
                )
                generate_btn = gr.Button("Generate", variant="primary")
                output_box = gr.Textbox(label="Output", lines=8, interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("### Generation Settings")
                max_tokens = gr.Slider(10, 200, value=80, step=10, label="Max new tokens")
                temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
                top_k = gr.Slider(0, 100, value=40, step=5, label="Top-k")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                rep_penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repetition penalty")

        gr.Markdown("### Model Architecture")
        with gr.Accordion("Show model info", open=False):
            model_info_json = gr.JSON(value=info)

        generate_btn.click(
            fn=generate,
            inputs=[prompt_box, max_tokens, temperature, top_k, top_p, rep_penalty],
            outputs=output_box,
        )
        prompt_box.submit(
            fn=generate,
            inputs=[prompt_box, max_tokens, temperature, top_k, top_p, rep_penalty],
            outputs=output_box,
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="ThinkyLM Gradio Demo")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/debug_1m.yaml")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=False)
    args = parser.parse_args()

    from inference.service import InferenceService

    service = InferenceService(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )

    demo = build_demo(service)
    print(f"[ThinkyLM Demo] Starting at http://{args.host}:{args.port}")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
