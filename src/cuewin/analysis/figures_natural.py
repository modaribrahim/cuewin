"""Figures for the naturalistic PADT section.

  fig_natural_subject.{pdf,png}      layer-9 subject-patching score, template vs
                                     natural (T1/T4), per model, 95% CI bands.
  fig_natural_doseresponse.{pdf,png} attractor interference on natural T6 vs
                                     sentence length/density (the confound).

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


def boot_ci(values, n_boot=2000):
    boots = np.array([np.mean(RNG.choice(values, size=len(values), replace=True))
                      for _ in range(n_boot)])
    return np.percentile(boots, [2.5, 97.5])


def fig_natural_subject():
    pat = pd.read_csv(os.path.join(data_dir(), "natural_patching.csv"))
    tpl = pd.read_csv(os.path.join(data_dir(), "patching_results.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4), constrained_layout=True, sharey=True)
    labels = ["T1 templ.", "T1 natural", "T4 templ.", "T4 natural"]
    cols = ["#9ecae1", "#08519c", "#fcae91", "#cb181d"]

    for ax_idx, (ax, m) in enumerate(zip(axes, ("arabert", "camelbert", "arbert"))):
        x = np.arange(4)
        w = 0.36
        means, lo, hi = [], [], []
        for cond, src in (("T1", "template"), ("T1", "natural"),
                          ("T4", "template"), ("T4", "natural")):
            d = (tpl if src == "template" else pat)
            d = d[(d.model == m) & (d.condition == cond) &
                  (d.cue == "subject") & (d.layer == 9)]
            means.append(d.score.mean())
            c = boot_ci(d.score.values)
            lo.append(c[0]); hi.append(c[1])

        for i in range(4):
            label = labels[i] if ax_idx == 0 else None
            ax.bar(x[i], means[i], w, color=cols[i], label=label)
            ax.errorbar(x[i], means[i], yerr=[[means[i]-lo[i]], [hi[i]-means[i]]],
                        fmt="none", ecolor="black", capsize=2, elinewidth=0.8)
        ax.axhline(0, lw=0.8, color="black", alpha=0.5)
        ax.set_xticks(x, ["T1\ntempl.", "T1\nnat.", "T4\ntempl.", "T4\nnat."], fontsize=8)
        ax.set_title(MODEL_LABEL[m], fontsize=9)

    axes[0].set_ylabel("layer-9 subject causal score $s(j,9)$")
    fig.legend(loc="upper center", ncol=4, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("The subject mechanism transfers to naturalistic text", fontsize=10, y=1.15)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figures_dir(), f"fig_natural_subject.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_natural_doseresponse():
    dr = pd.read_csv(os.path.join(data_dir(), "natural_doseresponse.csv"))
    tpl = pd.read_csv(os.path.join(data_dir(), "patching_results.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.1), constrained_layout=True, sharey=True)
    for ax, m in zip(axes, ("arabert", "camelbert", "arbert")):
        d = dr[dr.model == m]
        d = d.copy()
        d["len_bin"] = pd.qcut(d.n_words, 3, labels=["short", "mid", "long"])
        bins = ["short", "mid", "long"]
        bmeans = [d.loc[d.len_bin == b, "interference"].mean() for b in bins]
        blo = [boot_ci(d.loc[d.len_bin == b, "interference"].values)[0] for b in bins]
        bhi = [boot_ci(d.loc[d.len_bin == b, "interference"].values)[1] for b in bins]

        x = np.arange(3)
        ax.bar(x, bmeans, 0.5, color="#756bb1", label="natural T6 (binned by length)")
        ax.errorbar(x, bmeans, yerr=[[bmeans[i]-blo[i] for i in range(3)],
                                     [bhi[i]-bmeans[i] for i in range(3)]],
                    fmt="none", ecolor="black", capsize=3, elinewidth=0.9)

        t = tpl[(tpl.model == m) & (tpl.condition == "T6") & (tpl.cue == "attractor")
                & (tpl.layer >= 5) & (tpl.layer <= 11)].score.mean()
        ax.axhline(t, ls="--", color="#333333", lw=1)
        ax.text(2.35, t, f"template T6 {t:+.2f}", fontsize=7, ha="right", va="bottom")

        ax.axhline(0, lw=0.8, color="black", alpha=0.5)
        ax.set_xticks(x, [r"short" "\n" r"($\leq$~46 w.)",
                          r"mid" "\n" r"(47--71 w.)",
                          r"long" "\n" r"($\geq$~72 w.)"], fontsize=8)
        ax.set_title(MODEL_LABEL[m], fontsize=9)
        ax.set_xlabel("natural T6 sentence length (words)")
    axes[0].set_ylabel("attractor interference (L5-11 score)")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Natural attractor interference is length-sensitive (confound)", fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figures_dir(), f"fig_natural_doseresponse.{ext}"), dpi=300)
    plt.close(fig)


def main():
    fig_natural_subject()
    fig_natural_doseresponse()
    print("figures written to", figures_dir())


if __name__ == "__main__":
    main()
    print("saved fig_natural_subject + fig_natural_doseresponse (pdf+png)")
