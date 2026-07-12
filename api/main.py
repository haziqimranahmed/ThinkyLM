"""
ThinkyLM — FastAPI Application
================================
Author: Haziq Imran

REST API for ThinkyLM text generation and argument analysis.

Endpoints:
    GET  /health           — Health check
    GET  /model-info       — Model metadata
    POST /generate         — Text generation
    POST /analyse-argument — Argument analysis with philosophical framing

Usage:
    uvicorn api.main:app --host 127.0.0.1 --port 8000
    # Or: python api/main.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch

torch.set_num_threads(2)

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_inference_service
from api.schemas import (
    AnalyseArgumentRequest,
    AnalyseArgumentResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
)
from inference.service import InferenceService

app = FastAPI(
    title="ThinkyLM API",
    description=(
        "ThinkyLM is an independently implemented decoder-only Transformer for critical "
        "and philosophical argumentation. This API provides text generation and argument "
        "analysis endpoints.\n\n"
        "**Author**: Haziq Imran  \n"
        "**Disclaimer**: ThinkyLM is a small educational model (~1M parameters) trained "
        "from random initialisation. It is not a commercial reasoning system."
    ),
    version="0.1.0",
    contact={"name": "Haziq Imran", "url": "https://github.com/haziqimran/thinkylm"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(
    service: InferenceService = Depends(get_inference_service),
) -> HealthResponse:
    """Check API and model health.

    Returns OK regardless of whether a checkpoint is loaded,
    with a descriptive message about model status.
    """
    return HealthResponse(
        status="ok",
        model_loaded=service.loaded,
        message=(
            "Model checkpoint loaded and ready."
            if service.loaded
            else (
                f"API is running but no checkpoint is loaded. {service.load_error or ''} "
                "Run: python training/pretrain.py --config configs/debug_1m.yaml"
            )
        ),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
def model_info(
    service: InferenceService = Depends(get_inference_service),
) -> ModelInfoResponse:
    """Return model architecture and status metadata."""
    info = service.model_info()
    if "error" in info and not info.get("status"):
        raise HTTPException(status_code=503, detail=info["error"])
    return ModelInfoResponse(**info)


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
def generate_text(
    request: GenerateRequest,
    service: InferenceService = Depends(get_inference_service),
) -> GenerateResponse:
    """Generate text from a prompt.

    The model will continue the prompt autoregressively. Temperature,
    top-k, top-p, and repetition penalty control the generation.
    """
    result = service.generate(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
    )

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    return GenerateResponse(
        prompt=request.prompt,
        generated_text=result["text"],
        tokens_generated=result["tokens_generated"],
        model_loaded=service.loaded,
        warning=result.get("warning"),
    )


@app.post("/analyse-argument", response_model=AnalyseArgumentResponse, tags=["Generation"])
def analyse_argument(
    request: AnalyseArgumentRequest,
    service: InferenceService = Depends(get_inference_service),
) -> AnalyseArgumentResponse:
    """Analyse an argument with ThinkyLM's critical reasoning framing.

    The model generates a philosophical analysis of the provided argument.
    Note that at small scale, outputs are syntactic continuations rather
    than genuine reasoning.
    """
    prompt = (
        f"<system>You are ThinkyLM, a dry and careful philosophical reasoner.</system>"
        f"<user>Critically analyse the following argument: {request.argument}</user>"
        f"<assistant>"
    )

    result = service.generate(
        prompt=prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.15,
    )

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    return AnalyseArgumentResponse(
        argument=request.argument,
        analysis=result["text"],
        tokens_generated=result["tokens_generated"],
        model_loaded=service.loaded,
        warning=result.get("warning"),
    )


def main() -> None:
    """Entry point for running the API server."""
    import uvicorn

    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))
    print(f"[ThinkyLM API] Starting on http://{host}:{port}")
    print("[ThinkyLM API] Swagger docs: http://127.0.0.1:8000/docs")
    uvicorn.run("api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
