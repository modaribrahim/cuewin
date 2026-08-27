"""Central path / environment configuration for the pipeline.

Unifies the historical two-env-var convention (``MI_BASE`` for most scripts,
``MI_BASE_FINAL`` for the plotting scripts) into a single resolution:

* ``CUEWIN_DATA``     — directory that directly contains the result CSVs and
                        ``padt/``  (default: ``<package root>/data``).
* ``CUEWIN_MODELS``   — directory holding the local model checkpoints
                        (default: ``<package root>/../models``).
* ``CUEWIN_FIGURES``  — where figures are written (default: ``<data>/../figures``).

For backwards compatibility, ``MI_BASE`` and ``MI_BASE_FINAL`` are still
honoured (in that order) when ``CUEWIN_DATA`` is unset.
"""

from __future__ import annotations

import os

import cuewin


def package_root() -> str:
    """Root of the shipped package tree (the directory holding ``data/``)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(cuewin.__file__))))


def data_dir() -> str:
    """Directory containing the result CSVs and the ``padt/`` corpus.

    Resolution order: ``CUEWIN_DATA`` env var, then a ``data/`` directory
    next to the current working directory (the repo root, so plain
    ``pip install .`` works without editable mode), then the package default.
    """
    for key in ("CUEWIN_DATA", "MI_BASE", "MI_BASE_FINAL"):
        val = os.environ.get(key, "").strip()
        if val:
            return os.path.abspath(val)
    cwd_data = os.path.join(os.getcwd(), "data")
    if os.path.isdir(cwd_data):
        return cwd_data
    return os.path.join(package_root(), "data")


def figures_dir() -> str:
    """Directory that figure-producing commands write to."""
    val = os.environ.get("CUEWIN_FIGURES", "").strip()
    if val:
        return os.path.abspath(val)
    # MI_BASE_FINAL was historically the dir *containing* data/ and figures/.
    base = os.environ.get("MI_BASE_FINAL", "").strip()
    if base:
        return os.path.join(os.path.abspath(base), "figures")
    return os.path.join(os.path.dirname(data_dir()), "figures")


def models_dir() -> str:
    """Directory holding the local model checkpoints (``models/<model_id>``).

    Resolution order: ``CUEWIN_MODELS`` env var, then ``models/`` next to the
    current working directory, then the package default.
    """
    val = os.environ.get("CUEWIN_MODELS", "").strip() or os.environ.get("MI_MODELS", "").strip()
    if val:
        return os.path.abspath(val)
    cwd_models = os.path.join(os.getcwd(), "models")
    if os.path.isdir(cwd_models):
        return cwd_models
    return os.path.join(os.path.dirname(package_root()), "models")


def model_path(model_key: str) -> str:
    """Resolve a model key (``arabert``/``camelbert``/``arbert``/``aragpt2``)
    to a local checkpoint directory if one exists, else the raw key."""
    root = models_dir()
    names = {
        "arabert": "arabertv02",
        "camelbert": "camelbert-msa",
        "arbert": "arbert",
        "aragpt2": "aragpt2-base",
    }
    local = os.path.join(root, names.get(model_key, model_key))
    return local if os.path.isdir(local) else names.get(model_key, model_key)
