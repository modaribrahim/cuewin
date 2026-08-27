"""Main paper figures from the result CSVs (port of ``07_plots``).

  fig1_behavioral.png       condition × gender accuracy heatmap, per model
  fig2_patching.png         layer-wise causal score per cue, all conditions × models
  fig3_zeroing.png          layer-wise normalized contribution per role (T2)
  fig4_consistency.png      patching vs zeroing subject−adjective gap (T2, L7-11)

All saved under the figures dir at 300 dpi. Read-only on CSVs.
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

MODELS = ["arabert", "camelbert", "arbert"]
CONDS = ["T1", "T2", "T3", "T4", "T5", "T6"]
COND_LABEL = {
    "T1": "T1 baseline", "T2": "T2 congruent adj", "T3": "T3 conflict adj",
    "T4": "T4 VSO", "T5": "T5 distance", "T6": "T6 attractor",
}
CUE_COLORS = {"subject": "#1f77b4", "adjective": "#ff7f0e", "attractor": "#d62728"}


def fig1_behavioral(beh):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.6 * len(MODELS), 3.2),
                             constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        acc = (beh[beh.model == model]
               .pivot_table(index="condition", columns="gold",
                            values="correct", aggfunc="mean")
               .reindex(CONDS)[["m", "f"]] * 100)
        im = ax.imshow(acc.values, vmin=30, vmax=100, cmap="RdYlGn", aspect="auto")
        ax.set_xticks([0, 1], ["masc", "fem"])
        ax.set_yticks(range(len(CONDS)), [COND_LABEL[c] for c in CONDS])
        for i in range(len(CONDS)):
            for j in range(2):
                ax.text(j, i, f"{acc.values[i, j]:.0f}", ha="center",
                        va="center", fontsize=9)
        ax.set_title(MODEL_LABEL[model], fontsize=10)
    fig.colorbar(im, ax=axes, label="accuracy (%)", shrink=0.85)
    fig.suptitle("Forced-choice accuracy (gold vs foil verb form)", fontsize=11)
    fig.savefig(os.path.join(figures_dir(), "fig1_behavioral.png"), dpi=300)
    plt.close(fig)


def fig2_patching(pat):
    # Paper protocol (Sec. 4): decision effects (subject) are scored on
    # correct-only items; interference cues (adjective, attractor) on all items.
    fig, axes = plt.subplots(len(MODELS), 6, figsize=(16, 3.2 * len(MODELS)),
                             sharex=True, sharey="row", constrained_layout=True)
    for r, model in enumerate(MODELS):
        for c, cond in enumerate(CONDS):
            ax = axes[r, c]
            d = pat[(pat.model == model) & (pat.condition == cond)]
            for cue in d.cue.unique():
                dd = d[d.cue == cue]
                if cue == "subject":
                    dd = dd[dd.correct_clean]
                g = dd.groupby("layer").score
                mean, sem = g.mean(), g.sem()
                ax.plot(mean.index, mean.values, color=CUE_COLORS[cue],
                        label=cue, lw=1.8)
                ax.fill_between(mean.index, mean - 1.96 * sem,
                                mean + 1.96 * sem,
                                color=CUE_COLORS[cue], alpha=0.25)
            ax.axhline(0, color="grey", lw=0.6, ls="--")
            if r == 0:
                ax.set_title(COND_LABEL[cond], fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{MODEL_LABEL[model]}\ncausal score "
                              r"($\Delta_{clean}-\Delta_{patched}$)", fontsize=8)
            if r == 1:
                ax.set_xlabel("layer", fontsize=8)
            ax.tick_params(labelsize=7)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    h2, l2 = axes[0, 5].get_legend_handles_labels()
    for h, l in zip(h2, l2):
        if l not in labels:
            handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="upper right", fontsize=9, ncol=3)
    fig.suptitle("Value patching: per-cue causal effect on the gender decision "
                 "(95% CI; subject: correct items, interference: all items)", fontsize=11)
    fig.savefig(os.path.join(figures_dir(), "fig2_patching.png"), dpi=300)
    plt.close(fig)


def fig3_zeroing(zer):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.6 * len(MODELS), 3.4),
                             sharey=True, constrained_layout=True)
    roles = ["subject", "adjective", "others"]
    colors = {"subject": "#1f77b4", "adjective": "#ff7f0e", "others": "#7f7f7f"}
    for ax, model in zip(axes, MODELS):
        d = zer[(zer.model == model) & (zer.condition == "T2")]
        for role in roles:
            g = d[d.role == role].groupby("layer").score_norm
            mean, sem = g.mean(), g.sem()
            ax.plot(mean.index, mean.values, color=colors[role], label=role, lw=1.8)
            ax.fill_between(mean.index, mean - 1.96 * sem, mean + 1.96 * sem,
                            color=colors[role], alpha=0.25)
        ax.set_title(MODEL_LABEL[model], fontsize=10)
        ax.set_xlabel("layer")
    axes[0].set_ylabel("normalized contribution\nto [MASK] representation")
    axes[0].legend(fontsize=9)
    fig.suptitle("Value Zeroing (T2): context mixing into the masked-verb "
                 "representation", fontsize=11)
    fig.savefig(os.path.join(figures_dir(), "fig3_zeroing.png"), dpi=300)
    plt.close(fig)


def fig4_consistency(pat, zer):
    """Subject−adjective gap, both methods, T2, layers 7–11, per model."""
    fig, ax = plt.subplots(figsize=(5, 3.4), constrained_layout=True)
    width = 0.35
    xs = np.arange(len(MODELS))
    for k, (label, color) in enumerate(
            [("patching (causal)", "#2ca02c"), ("zeroing (repr.)", "#9467bd")]):
        gaps, errs = [], []
        for model in MODELS:
            if k == 0:
                d = pat[(pat.model == model) & (pat.condition == "T2")
                        & pat.correct_clean & pat.layer.isin(range(7, 12))]
                s = d[d.cue == "subject"].groupby("id").score.mean()
                a = d[d.cue == "adjective"].groupby("id").score.mean()
            else:
                d = zer[(zer.model == model) & (zer.condition == "T2")
                        & zer.layer.isin(range(7, 12))]
                s = d[d.role == "subject"].groupby("id").score_norm.mean()
                a = d[d.role == "adjective"].groupby("id").score_norm.mean()
            common = s.index.intersection(a.index)
            diff = (s[common] - a[common])
            gaps.append(diff.mean())
            errs.append(1.96 * diff.sem())
        ax.bar(xs + (k - 0.5) * width, gaps, width, yerr=errs, capsize=4,
               label=label, color=color, alpha=0.85)
    ax.set_xticks(xs, [MODEL_LABEL[m] for m in MODELS])
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("subject − adjective (L7–11 mean)")
    ax.set_title("Method consistency: subject outweighs adjective\n"
                 "in both analyses (T2)", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(figures_dir(), "fig4_consistency.png"), dpi=300)
    plt.close(fig)


def main():
    d = data_dir()
    beh = pd.read_csv(os.path.join(d, "behavioral_results.csv"))
    pat = pd.read_csv(os.path.join(d, "patching_results.csv"))
    zer = pd.read_csv(os.path.join(d, "zeroing_results.csv"))
    fig1_behavioral(beh)
    fig2_patching(pat)
    fig3_zeroing(zer)
    fig4_consistency(pat, zer)
    print("figures written to", figures_dir())


if __name__ == "__main__":
    main()
