"""
ThinkyLM — FastAPI Dependencies
==================================
Singleton inference service injected via FastAPI dependency injection.
"""

from __future__ import annotations

import os
from functools import lru_cache

from inference.service import InferenceService


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Return the singleton InferenceService.

    The service is created once on first call and cached.
    Configuration is read from environment variables with safe defaults.

    Returns:
        Shared InferenceService instance.
    """
    config_path = os.environ.get("THINKYLM_CONFIG", "configs/debug_1m.yaml")
    checkpoint_path = os.environ.get("THINKYLM_CHECKPOINT", None)
    return InferenceService(config_path=config_path, checkpoint_path=checkpoint_path)
