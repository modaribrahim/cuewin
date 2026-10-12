"""Revision figures (read-only on CSVs).

  fig2_patching.png      layer-wise causal score per cue, conditions x encoders
                         (MARBERT row added when rev_marbert_patching.csv exists)
  fig_residual.png       residual-stream patching: where the subject's gender
                         information sits, layer by layer (subject vs [MASK])
  fig_repair_windows.png repair rate for every 4-layer window: zero vs
                         gender-only patch of the interfering cue vs control
  fig_heads.png          head-level value patching of the subject (T1)
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

D = data_dir()
CONDS = ["T1", "T2", "T3", "T4", "T5", "T6"]
COND_LABEL = {
    "T1": "T1 baseline", "T2": "T2 congruent adjective", "T3": "T3 conflicting adjective",
    "T4": "T4 VSO", "T5": "T5 distance", "T6": "T6 attractor",
}
TITLE2 = {"T1": "T1\nbaseline", "T2": "T2 congruent\nadjective", "T3": "T3 conflicting\nadjective",
          "T4": "T4\nVSO", "T5": "T5\ndistance", "T6": "T6\nattractor"}
CUE_COLORS = {"subject": "#1f77b4", "adjective": "#ff7f0e", "attractor": "#d62728"}
CUE_STYLE = {"subject": "-", "adjective": "--", "attractor": ":"}


def _models(prefix):
    return [m for m in ("arabert", "camelbert", "arbert", "marbert")
            if os.path.exists(os.path.join(D, f"{prefix}_{m}.csv"))]


def fig2_patching():
    pat = pd.read_csv(os.path.join(D, "patching_results.csv"),
                      usecols=["model", "condition", "cue", "layer", "score", "correct_clean"])
    p = os.path.join(D, "rev_marbert_patching.csv")
    if os.path.exists(p):
        pat = pd.concat([pat, pd.read_csv(p, usecols=pat.columns.tolist())])
    models = [m for m in ("arabert", "camelbert", "arbert", "marbert") if m in set(pat.model)]
    fig, axes = plt.subplots(len(models), 6, figsize=(11, 2.1 * len(models) + 0.7),
                             sharex=True, sharey="row")
    for r, model in enumerate(models):
        for c, cond in enumerate(CONDS):
            ax = axes[r, c]
            d = pat[(pat.model == model) & (pat.condition == cond)]
            for cue in ("subject", "adjective", "attractor"):
                dd = d[d.cue == cue]
                if dd.empty:
                    continue
                if cue == "subject":
                    dd = dd[dd.correct_clean]
                g = dd.groupby("layer").score
                mean, sem = g.mean(), g.sem()
                ax.plot(mean.index, mean.values, color=CUE_COLORS[cue], ls=CUE_STYLE[cue],
                        label=cue, lw=1.8)
                ax.fill_between(mean.index, mean - 1.96 * sem, mean + 1.96 * sem,
                                color=CUE_COLORS[cue], alpha=0.25)
            ax.axhline(0, color="grey", lw=0.6, ls="--")
            ax.axvspan(7.5, 11.5, color="0.9", zorder=0)   # late window, layers 8-11
            if r == 0:
                ax.set_title(TITLE2[cond], fontsize=12)
            if c == 0:
                ax.set_ylabel(f"{MODEL_LABEL[model]}\nscore (logits)", fontsize=12)
            if r == len(models) - 1:
                ax.set_xlabel("layer", fontsize=12)
            ax.set_xticks(range(0, 12, 2))
            ax.set_xlim(-0.3, 11.5)
            ax.tick_params(labelsize=11)
    leg = {"subject": "subject", "adjective": "adjective (T2 congruent, T3 conflicting)",
           "attractor": "attractor"}
    handles = [plt.Line2D([], [], color=CUE_COLORS[k], ls=CUE_STYLE[k], lw=1.8, label=leg[k])
               for k in ("subject", "adjective", "attractor")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=12, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(os.path.join(figures_dir(), "fig2_patching.png"), dpi=300)
    plt.close(fig)


def fig_residual():
    models = _models("rev_residual")
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(3.0 * len(models), 2.7), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        r = pd.read_csv(os.path.join(D, f"rev_residual_{m}.csv"))
        r = r[(r.delta_clean - r.delta_corr).abs() > 0.5]
        r["eff"] = (r.delta_clean - r.delta_patched) / (r.delta_clean - r.delta_corr)
        for cond, ls in (("T1", "-"), ("T4", "--")):
            for site, col in (("subject", "#1f77b4"), ("mask", "#2ca02c")):
                g = r[(r.condition == cond) & (r.site == site)].groupby("layer").eff.median()
                ax.plot(g.index, g.values, color=col, ls=ls, lw=1.8,
                        label=f"{site if site == 'subject' else '[MASK]'} ({cond})")
        ax.set_title(MODEL_LABEL[m], fontsize=10)
        ax.set_xlabel("output of layer", fontsize=9)
        ax.set_xticks([-1, 1, 3, 5, 7, 9, 11], ["emb", "1", "3", "5", "7", "9", "11"])
        ax.axhline(0.5, color="grey", lw=0.6, ls=":")
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("share of subject effect\n(median, normalized)", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(figures_dir(), "fig_residual.png"), dpi=300)
    plt.close(fig)


def fig_repair_windows():
    models = _models("rev_windows")
    if not models:
        return
    fig, axes = plt.subplots(2, len(models), figsize=(3.0 * len(models), 4.6),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    quads = [f"L{k}-{k + 3}" for k in range(9)]
    styles = {"patch": ("#9467bd", "-", "gender-only patch"),
              "zero": ("#d62728", "--", "cue zeroed"),
              "ctrl": ("0.4", ":", "object zeroed (control)")}
    for j, m in enumerate(models):
        w = pd.read_csv(os.path.join(D, f"rev_windows_{m}.csv"))
        w = w[w.delta_clean <= 0]
        for i, cond in enumerate(("T6", "T3")):
            ax = axes[i, j]
            d = w[w.condition == cond]
            for kind, (col, ls, lab) in styles.items():
                k = d[d.kind == kind]
                rates = [100 * (k[k.window == q].delta_int > 0).mean() for q in quads]
                ax.plot(range(9), rates, color=col, ls=ls, lw=1.8, marker="o", ms=3, label=lab)
            if i == 0:
                ax.set_title(MODEL_LABEL[m], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{'T6 attractor' if cond == 'T6' else 'T3 conflicting adjective'}\n"
                              "errors repaired (%)", fontsize=9)
            if i == 1:
                ax.set_xticks(range(9), [f"{k}–{k + 3}" for k in range(9)], rotation=60)
                ax.set_xlabel("4-layer window", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.set_ylim(0, 90)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(figures_dir(), "fig_repair_windows.png"), dpi=300)
    plt.close(fig)


def fig_heads():
    models = _models("rev_heads")
    if not models:
        return
    mats = {}
    for m in models:
        h = pd.read_csv(os.path.join(D, f"rev_heads_{m}.csv"))
        h = h[(h.condition == "T1") & h.correct_clean]
        mats[m] = h.groupby(["layer", "head"]).score.mean().unstack().to_numpy()
    lim = max(np.abs(x).max() for x in mats.values())      # one color scale for all panels
    fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models) + 0.8, 2.9))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        mat = mats[m]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower", aspect="auto")
        ax.set_title(MODEL_LABEL[m], fontsize=10)
        ax.set_xlabel("head", fontsize=9)
        ax.set_xticks(range(0, 12, 2))
        ax.set_yticks(range(0, 12, 2))
        ax.tick_params(labelsize=8)
        ax.set_ylabel("layer", fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    cax = fig.add_axes((0.91, 0.2, 0.012, 0.65))
    fig.colorbar(im, cax=cax).set_label("score (logits)", fontsize=8)
    fig.savefig(os.path.join(figures_dir(), "fig_heads.png"), dpi=300)
    plt.close(fig)


def main():
    fig2_patching()
    fig_residual()
    fig_repair_windows()
    fig_heads()
    print("figures written to", figures_dir())


if __name__ == "__main__":
    main()
