"""
ThinkyLM — Pydantic API Schemas
==================================
Request/response models for the FastAPI application.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""

    prompt: str = Field(..., min_length=1, max_length=2000, description="Input text prompt")
    max_new_tokens: int = Field(default=100, ge=1, le=500, description="Maximum tokens to generate")
    temperature: float = Field(default=0.8, ge=0.01, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=40, ge=0, le=1000, description="Top-k sampling (0 = disabled)")
    top_p: float = Field(default=0.9, ge=0.01, le=1.0, description="Top-p nucleus sampling")
    repetition_penalty: float = Field(
        default=1.1, ge=1.0, le=3.0, description="Repetition penalty factor"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "prompt": "Does the absence of objective meaning imply that life is meaningless?",
            "max_new_tokens": 100,
            "temperature": 0.8,
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        }
    }}


class GenerateResponse(BaseModel):
    """Response body for the /generate endpoint."""

    prompt: str
    generated_text: str
    tokens_generated: int
    model_loaded: bool
    warning: Optional[str] = None
    disclaimer: str = (
        "ThinkyLM is a small educational model (~1M parameters) trained from scratch. "
        "Outputs are not representative of genuine reasoning."
    )


class AnalyseArgumentRequest(BaseModel):
    """Request body for the /analyse-argument endpoint."""

    argument: str = Field(..., min_length=1, max_length=1000, description="Argument text to analyse")
    max_new_tokens: int = Field(default=120, ge=1, le=400)
    temperature: float = Field(default=0.75, ge=0.01, le=2.0)


class AnalyseArgumentResponse(BaseModel):
    """Response body for the /analyse-argument endpoint."""

    argument: str
    analysis: str
    tokens_generated: int
    model_loaded: bool
    warning: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Response body for the /model-info endpoint."""

    status: str
    checkpoint: str
    total_params: int
    vocab_size: int
    context_length: int
    hidden_size: int
    num_layers: int
    num_heads: int
    device: str
    disclaimer: str


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    model_loaded: bool
    message: str
