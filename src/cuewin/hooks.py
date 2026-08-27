"""Minimal PyTorch forward-hook utilities for value-vector caching, patching
and zeroing on HF BERT models.

Hook point: ``model.bert.encoder.layer[i].attention.self.value`` is the
``nn.Linear`` whose output is the value vectors ``(B, L, hidden)`` of layer
``i``, before the multi-head reshape. Capturing / patching here keeps the
attention patterns intact — the value-patching rationale of
Amirzadeh et al. (2024).
"""

from __future__ import annotations

from contextlib import contextmanager

def value_linears(model):
    """The 12 value-projection ``nn.Linear`` modules of a HF ``BertForMaskedLM``."""
    return [layer.attention.self.value for layer in model.bert.encoder.layer]


class ValueCache:
    """Captures value-vector outputs of every layer during a forward pass."""

    def __init__(self, model):
        self.linears = value_linears(model)
        self.values = {}  # layer_idx -> tensor (B, L, H)

    @contextmanager
    def capture(self):
        handles = []
        self.values.clear()

        def make_hook(idx):
            def hook(module, inputs, output):
                self.values[idx] = output.detach()
            return hook

        for i, lin in enumerate(self.linears):
            handles.append(lin.register_forward_hook(make_hook(i)))
        try:
            yield self
        finally:
            for h in handles:
                h.remove()


class ValuePatcher:
    """Overwrites value vectors at given (row, start, end) spans of ONE layer
    with vectors from a cached source during a forward pass."""

    def __init__(self, model):
        self.linears = value_linears(model)

    @contextmanager
    def patch(self, layer, source, rows_spans):
        """layer: int; source: dict layer->tensor from ValueCache (same batch
        shape as the clean run); rows_spans: list of (row, start, end)."""
        src = source[layer]

        def hook(module, inputs, output):
            out = output.clone()
            for row, s, e in rows_spans:
                out[row, s:e, :] = src[row, s:e, :]
            return out

        handle = self.linears[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()


class ValueZeroer:
    """Zeroes value vectors at given (row, start, end) spans of ONE layer."""

    def __init__(self, model):
        self.linears = value_linears(model)

    @contextmanager
    def zero(self, layer, rows_spans):
        def hook(module, inputs, output):
            out = output.clone()
            for row, s, e in rows_spans:
                out[row, s:e, :] = 0.0
            return out

        handle = self.linears[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()


class HiddenStateCache:
    """Captures each encoder layer's output hidden states (for Value Zeroing's
    cosine-distance computation at the [MASK] representation)."""

    def __init__(self, model):
        self.layers = list(model.bert.encoder.layer)
        self.hidden = {}  # layer_idx -> (B, L, H)

    @contextmanager
    def capture(self):
        handles = []
        self.hidden.clear()

        def make_hook(idx):
            def hook(module, inputs, output):
                self.hidden[idx] = output[0].detach()
            return hook

        for i, lyr in enumerate(self.layers):
            handles.append(lyr.register_forward_hook(make_hook(i)))
        try:
            yield self
        finally:
            for h in handles:
                h.remove()
