"""Unified model registry and loader.

Resolves the four public model identifiers used across the paper and loads
tokenizers / models, preferring a local checkpoint directory when one exists
(see ``cuewin.config.model_path``) and falling back to the public HuggingFace
id otherwise.
"""

from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

from cuewin.config import model_path

def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def tokenizer(key: str):
    return AutoTokenizer.from_pretrained(model_path(key))


def load_encoder(key: str):
    """Load a masked encoder (arabert / camelbert / arbert) on the available
    device, eval mode."""
    path = model_path(key)
    tk = AutoTokenizer.from_pretrained(path)
    model = AutoModelForMaskedLM.from_pretrained(path).to(device()).eval()
    return tk, model


def load_decoder(key: str = "aragpt2", path: str | None = None):
    """Load the causal decoder (AraGPT2). ``path`` overrides ``model_path``."""
    p = path or model_path(key)
    tk = AutoTokenizer.from_pretrained(p)
    model = AutoModelForCausalLM.from_pretrained(p).to(device()).eval()
    return tk, model
