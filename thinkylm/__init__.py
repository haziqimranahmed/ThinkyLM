"""
ThinkyLM — Package Initialisation
====================================
Author: Haziq Imran
Repository: https://github.com/haziqimran/thinkylm

ThinkyLM is an independently implemented decoder-only Transformer
trained from random initialisation for critical and philosophical argumentation.

This is NOT a wrapper around any pretrained model.
"""

__version__ = "0.1.0"
__author__ = "Haziq Imran"
__license__ = "MIT"

from thinkylm.config import DataConfig, ModelConfig, ThinkyLMConfig, TrainingConfig
from thinkylm.model import ThinkyLM, build_model

__all__ = [
    "ThinkyLM",
    "build_model",
    "ThinkyLMConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
]
