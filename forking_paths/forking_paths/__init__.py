"""Efficient Forking Paths Analysis (Bigelow et al., arXiv 2412.07961).

A prefix-cached, token-ID-space reimplementation for open-weight models via
HuggingFace transformers (GPU).
"""
from .config import ForkingConfig

__all__ = ["ForkingConfig"]
__version__ = "0.1.0"
