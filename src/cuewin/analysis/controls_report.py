#!/usr/bin/env python3
"""
Summaries of the control experiments (CPU), as fixed in the analysis plan:

  E1  head output at [MASK] (ctrl_maskheads_<model>.csv.gz): per-layer share of
      the subject-flip and attractor-flip effect reproduced at the verb
      position; correlation of the [MASK]-only head map with the value-patching
      head map (rev_heads_<model>.csv); the strongest subject head's
      [MASK]-only score as a share of its value-patching score (item bootstrap,
      seed 61) and its rank among [MASK]-only attractor scores.
  E2  same-gender swap control (ctrl_swap_<model>.csv): repair and break
      rates per window next to the gender-only patch (rev_windows_<model>.csv),
      and the late-window difference (paired item bootstrap, seed 62).

-> data/controls_report.txt
"""

import os

import numpy as np
import pandas as pd

from cuewin.config import data_dir

D = data_dir()
ENC = ("arabert", "camelbert", "arbert", "marbert")
B = 10_000
WIN = ("L0-3", "L4-7", "L8-11")
lines = []


def out(s=""):
    print(s)
    lines.append(s)


def boot_ratio(num, den, rng):
    n = len(num)
    idx = rng.integers(0, n, size=(B, n))
    r = num[idx].mean(1) / den[idx].mean(1)
    return np.percentile(r, [2.5, 97.5])


def boot_mean(x, rng):
    idx = rng.integers(0, len(x), size=(B, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


def e1():
    out("== E1. Head output at [MASK] (subject: correct T1 items; attractor: all T6 items) ==")
    rng = np.random.default_rng(61)
    for m in ENC:
        p = os.path.join(D, f"ctrl_maskheads_{m}.csv.gz")
        if not os.path.exists(p):
            out(f"  {m}: missing")
            continue
        d = pd.read_csv(p)
        d["score"] = d.delta_clean - d.delta_patched
        v = pd.read_csv(os.path.join(D, f"rev_heads_{m}.csv"))
        maps, vmaps, items = {}, {}, {}
        for cue in ("subject", "attractor"):
            g = d[d.cue == cue]
            if cue == "subject":
                g = g[g.delta_clean > 0]
            # layer share at [MASK] (all heads), Eq. 4 with site = attention output at [MASK]
            lay = g[g["head"] == -1].copy()
            den = lay.delta_clean - lay.delta_corr
            lay = lay[den.abs() > 0.5]
            lay["share"] = lay.score / (lay.delta_clean - lay.delta_corr)
            med = lay.groupby("layer").share.median()
            out(f"  {m:9s} {cue:9s} layer attention output at [MASK], median share by layer: "
                + " ".join(f"{int(l)}:{s:.2f}" for l, s in med.items()))
            gh = g[g["head"] >= 0]
            maps[cue] = gh.groupby(["layer", "head"]).score.mean()
            items[cue] = gh
            vv = v[(v.cue == cue)]
            vv = vv[vv.correct_clean] if cue == "subject" else vv
            vmaps[cue] = vv.groupby(["layer", "head"]).score.mean()
        for cue in ("subject", "attractor"):
            a, b = maps[cue], vmaps[cue].reindex(maps[cue].index)
            out(f"  {m:9s} {cue:9s} r([MASK]-only map, value-patching map) = {np.corrcoef(a, b)[0, 1]:+.2f}")
        top = vmaps["subject"].idxmax()
        # [MASK]-only vs value-patching score of the top subject head, same items
        ms = items["subject"].set_index(["layer", "head"]).sort_index().loc[top].set_index("id").score
        vs = v[(v.cue == "subject") & v.correct_clean & (v.layer == top[0]) & (v["head"] == top[1])]
        vs = vs.set_index("id").score
        ids = ms.index.intersection(vs.index)
        num, den = ms.loc[ids].to_numpy(), vs.loc[ids].to_numpy()
        lo, hi = boot_ratio(num, den, rng)
        ma = maps["attractor"]
        rank_a = int((ma < ma[top]).sum()) + 1
        rank_s = int((maps["subject"] > maps["subject"][top]).sum()) + 1
        out(f"  {m:9s} top subject head L{top[0]}H{top[1]}: [MASK]-only subject score {num.mean():+.3f}, "
            f"value-patching {den.mean():+.3f}, share {num.mean() / den.mean():.2f} [{lo:.2f}, {hi:.2f}] "
            f"(n={len(ids)}); rank among [MASK]-only subject heads {rank_s}/144; "
            f"[MASK]-only attractor score {ma[top]:+.3f} (rank {rank_a}/144 most negative)")
    out()


def e2():
    out("== E2. Same-gender swap control (repair on error items, break on correct items) ==")
    rng = np.random.default_rng(62)
    for m in ENC:
        p = os.path.join(D, f"ctrl_swap_{m}.csv")
        if not os.path.exists(p):
            out(f"  {m}: missing")
            continue
        s = pd.read_csv(p)
        w = pd.read_csv(os.path.join(D, f"rev_windows_{m}.csv"))
        w = w[(w.kind == "patch") & w.window.isin(WIN) & w.id.isin(s.id)]
        for cond in ("T6", "T3"):
            ss, ww = s[s.condition == cond], w[w.condition == cond]
            err = ss[ss.delta_clean <= 0]
            cor = ss[ss.delta_clean > 0]
            rep_s = {x: 100 * (err[err.window == x].delta_int > 0).mean() for x in WIN}
            brk_s = {x: 100 * (cor[cor.window == x].delta_int < 0).mean() for x in WIN}
            werr = ww[ww.delta_clean <= 0]
            rep_p = {x: 100 * (werr[werr.window == x].delta_int > 0).mean() for x in WIN}
            a = werr[werr.window == "L8-11"].set_index("id").delta_int > 0
            b = err[err.window == "L8-11"].set_index("id").delta_int > 0
            ids = a.index.intersection(b.index)
            diff = 100 * (a.loc[ids].astype(float) - b.loc[ids].astype(float)).to_numpy()
            lo, hi = boot_mean(diff, rng)
            out(f"  {m:9s} {cond} n_err={err.id.nunique()} n_corr={cor.id.nunique()}: "
                f"swap repair " + "/".join(f"{rep_s[x]:.1f}" for x in WIN)
                + " | gender-only patch " + "/".join(f"{rep_p[x]:.1f}" for x in WIN)
                + f" | swap break " + "/".join(f"{brk_s[x]:.1f}" for x in WIN)
                + f" | late patch-minus-swap {diff.mean():+.1f} [{lo:+.1f}, {hi:+.1f}]")
    it = pd.read_csv(os.path.join(D, "ctrl_swap_items.csv"))
    out(f"  items with a same-gender replacement: {len(it)} of 2400 (T3+T6)")
    out()


def main():
    e1()
    e2()
    with open(os.path.join(D, "controls_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
