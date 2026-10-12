"""Logit-lens and item-filter figures (port of ``09b_logit_lens_figs``).

  fig5_logit_lens.{pdf,png}          logit-lens Δ = logit(gold)−logit(foil) vs layer,
                                     per encoder, with 95% bootstrap CI bands.
  fig6_patching_all_vs_correct.{pdf,png}  T1 subject patching score vs layer,
                                     all-items vs correct-only, per encoder.

Vector PDF (print) + 300-dpi PNG. Read-only on CSVs.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cuewin.config import data_dir, figures_dir
from cuewin.stats import RNG

MODEL_LABEL = {"arabert": "AraBERTv02", "camelbert": "CAMeLBERT-MSA"}
MCOLOR = {"arabert": "#1f77b4", "camelbert": "#d62728"}
N_LAYERS = 12


def boot_band(per_item_by_layer, n=2000):
    """per_item_by_layer: dict layer -> 1d array. Returns mean, lo, hi arrays."""
    mean = np.array([per_item_by_layer[l].mean() for l in range(N_LAYERS)])
    lo, hi = np.zeros(N_LAYERS), np.zeros(N_LAYERS)
    for l in range(N_LAYERS):
        v = per_item_by_layer[l]
        idx = RNG.integers(0, len(v), size=(n, len(v)))
        b = v[idx].mean(1)
        lo[l], hi[l] = np.percentile(b, 2.5), np.percentile(b, 97.5)
    return mean, lo, hi


def fig_logit_lens():
    df = pd.read_csv(os.path.join(data_dir(), "logit_lens.csv"))
    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    for m in ("arabert", "camelbert"):
        d = df[df.model == m]
        byl = {l: d[d.layer == l]["delta"].values for l in range(N_LAYERS)}
        mean, lo, hi = boot_band(byl)
        ax.plot(range(N_LAYERS), mean, "-o", ms=4, color=MCOLOR[m],
                label=MODEL_LABEL[m])
        ax.fill_between(range(N_LAYERS), lo, hi, color=MCOLOR[m], alpha=0.18)
    ax.axvline(9, ls="--", lw=1, color="grey", alpha=0.7)
    ax.text(9, ax.get_ylim()[0], " layer 9", color="grey", fontsize=8, va="bottom")
    ax.set_xlabel("encoder layer")
    ax.set_ylabel(r"logit-lens $\Delta$ = logit(gold) − logit(foil)")
    ax.set_title("Logit-lens trajectory of the agreement decision (T1)", fontsize=10)
    ax.set_xticks(range(N_LAYERS))
    ax.legend(fontsize=8, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figures_dir(), f"fig5_logit_lens.{ext}"), dpi=300)
    plt.close(fig)


def fig_patching_all_vs_correct():
    pat = pd.read_csv(os.path.join(data_dir(), "patching_results.csv"))
    sub = pat[(pat.condition == "T1") & (pat.cue == "subject")]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), constrained_layout=True,
                             sharey=True)
    for ax, m in zip(axes, ("arabert", "camelbert")):
        d = sub[sub.model == m]
        for tag, style, col in (("all", "-o", "#1f77b4"),
                                ("correct", "--s", "#ff7f0e")):
            dd = d if tag == "all" else d[d.correct_clean]
            byl = {l: dd[dd.layer == l]["score"].values for l in range(N_LAYERS)}
            mean, lo, hi = boot_band(byl)
            ax.plot(range(N_LAYERS), mean, style, ms=4, color=col,
                    label=("all items" if tag == "all" else "correct-only"))
            ax.fill_between(range(N_LAYERS), lo, hi, color=col, alpha=0.15)
        ax.axvline(9, ls="--", lw=1, color="grey", alpha=0.7)
        ax.set_title(MODEL_LABEL[m], fontsize=10)
        ax.set_xlabel("encoder layer")
        ax.set_xticks(range(N_LAYERS))
    axes[0].set_ylabel("subject patching score")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Subject causal effect peaks at layer 9 regardless of item filter (T1)",
                 fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figures_dir(), f"fig6_patching_all_vs_correct.{ext}"), dpi=300)
    plt.close(fig)


def main():
    fig_logit_lens()
    fig_patching_all_vs_correct()
    print("saved fig5_logit_lens + fig6_patching_all_vs_correct (pdf+png)")


if __name__ == "__main__":
    main()
