"""Revision main-text figures (port of ``13_revision_main_figs``).

  fig_behavioral_corrected.{pdf,png}  T1 masc/fem accuracy, raw vs frequency-corrected
  fig_morpheme.{pdf,png}              per-layer collapse (Δ_clean − Δ_ablated) for the four
                                      ablation targets in the ة-morpheme probe.

Vector PDF + 300-dpi PNG. Read-only on CSVs.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cuewin.config import data_dir, figures_dir
from cuewin.constants import MODEL_LABEL
from cuewin.stats import RNG

N_LAYERS = 12


def fig_behavioral_corrected():
    d = pd.read_csv(os.path.join(data_dir(), "freq_baseline.csv"))
    d = d[d.baseline == "submask"].set_index("model")
    order = [m for m in ("arabert", "camelbert", "arbert") if m in d.index]
    fig, axes = plt.subplots(1, len(order), figsize=(3.4 * len(order), 3.1),
                             constrained_layout=True, sharey=True)
    x = np.arange(2)          # masc, fem
    w = 0.36
    for ax, m in zip(axes, order):
        r = d.loc[m]
        raw = [r["acc_masc_raw"], r["acc_fem_raw"]]
        cor = [r["acc_masc_corr"], r["acc_fem_corr"]]
        b1 = ax.bar(x - w / 2, raw, w, label="raw", color="#9ecae1")
        b2 = ax.bar(x + w / 2, cor, w, label="freq-corrected", color="#08519c")
        for b in (b1, b2):
            ax.bar_label(b, fmt="%.0f", fontsize=8, padding=1)
        ax.set_xticks(x, ["masc.", "fem."])
        ax.set_title(f"{MODEL_LABEL[m]}\n gap {r['gap_raw']:+.1f} $\\rightarrow$ {r['gap_corr']:+.1f}",
                     fontsize=9)
        ax.set_ylim(0, 105)
        ax.axhline(50, ls=":", lw=0.8, color="grey")
    axes[0].set_ylabel("T1 accuracy (%)")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("The masculine default is largely a verb-form frequency prior", fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figures_dir(), f"fig_behavioral_corrected.{ext}"), dpi=300)
    plt.close(fig)


def fig_morpheme():
    df = pd.read_csv(os.path.join(data_dir(), "morpheme_results.csv"))
    order = [("zero_subject", "whole subject", "#525252", "-o"),
             ("zero_ta", "ة morpheme only", "#d62728", "-o"),
             ("zero_stem", "stem only", "#1f77b4", "--s"),
             ("zero_control", "control token", "#bdbdbd", ":^")]
    fig, ax = plt.subplots(figsize=(5.4, 3.5), constrained_layout=True)
    for target, label, col, style in order:
        d = df[df.target == target]
        byl = {l: d[d.layer == l]["collapse"].values for l in range(N_LAYERS)}
        mean = np.array([byl[l].mean() for l in range(N_LAYERS)])
        lo = np.zeros(N_LAYERS); hi = np.zeros(N_LAYERS)
        for l in range(N_LAYERS):
            v = byl[l]; idx = RNG.integers(0, len(v), (2000, len(v)))
            b = v[idx].mean(1); lo[l], hi[l] = np.percentile(b, [2.5, 97.5])
        ax.plot(range(N_LAYERS), mean, style, ms=4, color=col, label=label)
        ax.fill_between(range(N_LAYERS), lo, hi, color=col, alpha=0.15)
    ax.axvline(9, ls="--", lw=1, color="grey", alpha=0.7)
    ax.axhline(0, lw=0.8, color="black", alpha=0.5)
    ax.set_xlabel("layer at which values are zeroed")
    ax.set_ylabel(r"collapse of agreement $\Delta$  ($\Delta_{\mathrm{clean}}-\Delta_{\mathrm{abl}}$)")
    ax.set_title("The feminine morpheme ة carries the causal gender signal\n(CAMeLBERT-MSA, T1)",
                 fontsize=9.5)
    ax.set_xticks(range(N_LAYERS))
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figures_dir(), f"fig_morpheme.{ext}"), dpi=300)
    plt.close(fig)


def main():
    fig_behavioral_corrected()
    fig_morpheme()
    print("figures written to", figures_dir())


if __name__ == "__main__":
    main()
    print("saved fig_behavioral_corrected + fig_morpheme (pdf+png)")
