"""Figures for the naturalistic PADT section.

  fig_natural_subject.{pdf,png}      subject-patching score, mean over layers 8-11, template vs
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

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), constrained_layout=True, sharey=True)
    labels = ["T1 template", "T1 natural", "T4 template", "T4 natural"]
    cols = ["#9ecae1", "#08519c", "#fcae91", "#cb181d"]

    for ax_idx, (ax, m) in enumerate(zip(axes, ("arabert", "camelbert", "arbert"))):
        x = np.arange(4)
        w = 0.36
        means, lo, hi = [], [], []
        for cond, src in (("T1", "template"), ("T1", "natural"),
                          ("T4", "template"), ("T4", "natural")):
            d = (tpl if src == "template" else pat)
            d = d[(d.model == m) & (d.condition == cond) &
                  (d.cue == "subject") & d.layer.between(8, 11)]
            v = d.groupby("id").score.mean()                 # late window, per item
            means.append(v.mean())
            c = boot_ci(v.values)
            lo.append(c[0]); hi.append(c[1])

        for i in range(4):
            label = labels[i] if ax_idx == 0 else None
            ax.bar(x[i], means[i], w, color=cols[i], label=label)
            ax.errorbar(x[i], means[i], yerr=[[means[i]-lo[i]], [hi[i]-means[i]]],
                        fmt="none", ecolor="black", capsize=2, elinewidth=0.8)
        ax.axhline(0, lw=0.8, color="black", alpha=0.5)
        ax.set_xticks(x, ["T1\ntemplate", "T1\nnatural", "T4\ntemplate", "T4\nnatural"], fontsize=8)
        ax.set_title(MODEL_LABEL[m], fontsize=9)

    axes[0].set_ylabel("subject score, layers 8–11")
    fig.legend(loc="upper center", ncol=4, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figures_dir(), f"fig_natural_subject.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_natural_doseresponse():
    dr = pd.read_csv(os.path.join(data_dir(), "natural_doseresponse.csv"))
    tpl = pd.read_csv(os.path.join(data_dir(), "patching_results.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), constrained_layout=True, sharey=True)
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
                & (tpl.layer >= 8) & (tpl.layer <= 11)].score.mean()
        ax.axhline(t, ls="--", color="#333333", lw=1)
        ax.text(2.35, t, f"template T6 {t:+.2f}".replace("-", "\u2212"), fontsize=7, ha="right", va="bottom")

        ax.axhline(0, lw=0.8, color="black", alpha=0.5)
        e = pd.qcut(d.n_words, 3).cat.categories      # integer word counts
        lo1, hi1, hi2 = int(e[0].right), int(e[1].right), int(e[1].right) + 1
        ax.set_xticks(x, [f"short\n(≤{lo1})", f"mid\n({lo1 + 1}–{hi1})",
                          f"long\n(≥{hi2})"], fontsize=8)
        ax.set_title(MODEL_LABEL[m], fontsize=9)
    axes[0].set_ylabel("attractor score, layers 8–11")
    fig.supxlabel("sentence-length tercile (words)", fontsize=9)
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
