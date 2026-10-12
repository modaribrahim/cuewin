#!/usr/bin/env python3
"""
Robustness summaries from existing outputs (CPU):

  A  repair ceiling: % of error items made correct by flipping the cue's
     gender in the input (Delta_cue > 0), next to the gender-only patch in all
     layers (S0-11) and in layers 8-11 (main and held-out sets; treebank).
  B  construct-state (T7) vs. prepositional attractor (T6) accuracy on the
     held-out set, restricted to subjects and attractors that occur in both,
     averaged over (subject, attractor, gold) cells present in both.
  C  hidden-state crossing layers (rev_residual_*) under exclusion thresholds
     of 0.25, 0.5 and 1.0 logits in Eq. 4.
  D  treebank attractor accuracy per encoder (from the cue-residual runs).

-> data/robustness_report.txt
"""

import json
import os

import pandas as pd

from cuewin.analysis import fixed_windows as fw
from cuewin.config import data_dir

D = data_dir()
ENC = ("arabert", "camelbert", "arbert", "marbert")
lines = []


def out(s=""):
    print(s)
    lines.append(s)


def pct(x):
    return f"{100 * x:.1f}"


def sec_A():
    out("== A. Repair ceiling: input flip vs. gender-only patch (error items, %) ==")
    for s in ("main", "rep"):
        for m in ENC:
            r = pd.read_csv(os.path.join(D, f"rev2_cueresid_{s}_{m}.csv"))
            r = r[(r.layer == -1) & (r.delta_clean <= 0)]
            c = pd.read_csv(os.path.join(D, f"rev2_cumwin_{s}_{m}.csv"))
            c = c[(c.kind == "patch") & (c.delta_clean <= 0)]
            w = None
            if s == "main":
                w = pd.read_csv(os.path.join(D, f"rev_windows_{m}.csv"))
                w = w[(w.kind == "patch") & (w.window == "L8-11") & (w.delta_clean <= 0)]
            else:
                w = pd.read_csv(os.path.join(D, f"rev2_rep_windows_{m}.csv"))
                w = w[(w.kind == "patch") & (w.window == "L8-11") & (w.delta_clean <= 0)]
                w = w[~w.id.isin(fw_excluded())]
                r = r[~r.id.isin(fw_excluded())]
                c = c[~c.id.isin(fw_excluded())]
            for cond, g in r.groupby("condition"):
                ceil = (g.delta_corr > 0).mean()
                allw = (c[(c.condition == cond) & (c.window == "S0-11")].delta_int > 0).mean()
                late = (w[w.condition == cond].delta_int > 0).mean()
                out(f"  {s:4s} {m:9s} {cond} n={len(g)}: input flip {pct(ceil)} | patch all layers "
                    f"{pct(allw)} | patch L8-11 {pct(late)} ({late / ceil:.2f} of input flip)")
    for m in ENC:
        r = pd.read_csv(os.path.join(D, f"rev2_nat_cueresid_{m}.csv"))
        r = r[(r.layer == -1) & (r.delta_clean <= 0)]
        out(f"  treebank {m:9s} n={len(r)}: input flip {pct((r.delta_corr > 0).mean())}")
    out()


_EXCL = None


def fw_excluded():
    global _EXCL
    if _EXCL is None:
        _EXCL = {it["id"] for it in rep_items() if fw.homograph_item(it)}
    return _EXCL


def rep_items():
    with open(os.path.join(D, "families_rep.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def sec_B():
    out("== B. T7 vs. T6 accuracy, held-out set, matched subjects and attractors (%) ==")
    its = [it for it in rep_items() if it["condition"] in ("T6", "T7") and not fw.homograph_item(it)]
    lex = pd.DataFrame([{"id": it["id"], "cond": it["condition"], "gold": it["gold"],
                         "subj": it["lex"]["subj_masc"], "attr": it["lex"]["attr_masc"]} for it in its])
    for m in ENC:
        b = pd.read_csv(os.path.join(D, f"rev2_rep_behavioral_{m}.csv"))[["id", "correct"]]
        d = lex.merge(b, on="id")
        S = set(d[d.cond == "T6"].subj) & set(d[d.cond == "T7"].subj)
        A = set(d[d.cond == "T6"].attr) & set(d[d.cond == "T7"].attr)
        x = d[d.subj.isin(S) & d.attr.isin(A)]
        cells = x.groupby(["cond", "subj", "attr", "gold"]).correct.mean().unstack(0).dropna()
        out(f"  {m:9s} all items T6 {pct(d[d.cond == 'T6'].correct.mean())} T7 "
            f"{pct(d[d.cond == 'T7'].correct.mean())} | matched ({len(S)} subjects, {len(A)} attractors, "
            f"{len(cells)} cells): T6 {pct(cells.T6.mean())} T7 {pct(cells.T7.mean())}")
    out()


def sec_C():
    out("== C. Hidden-state crossing layers by exclusion threshold (Eq. 4) ==")
    for m in ENC:
        r0 = pd.read_csv(os.path.join(D, f"rev_residual_{m}.csv"))
        for th in (0.25, 0.5, 1.0):
            r = r0[(r0.delta_clean - r0.delta_corr).abs() > th].copy()
            r["eff"] = (r.delta_clean - r.delta_patched) / (r.delta_clean - r.delta_corr)
            res = []
            for c in ("T1", "T4", "T6"):
                g = r[r.condition == c]
                pv = g.pivot_table(index="site", columns="layer", values="eff", aggfunc="median")
                vs, vm = pv.loc["subject"], pv.loc["mask"]
                cs = next((l for l in vs.index if vs[l] < 0.5), None)
                cm = next((l for l in vm.index if vm[l] >= 0.5), None)
                res.append(f"{c} n={g.id.nunique()} subject<0.5 L{cs} [MASK]>=0.5 L{cm}")
            out(f"  {m:9s} threshold {th:.2f}: " + " | ".join(res))
    out()


def sec_D():
    out("== D. Treebank attractor accuracy (%) ==")
    for m in ENC:
        r = pd.read_csv(os.path.join(D, f"rev2_nat_cueresid_{m}.csv"))
        r = r[r.layer == -1]
        out(f"  {m:9s} n={len(r)} errors={(r.delta_clean <= 0).sum()} accuracy {pct((r.delta_clean > 0).mean())}")
    out()


def main():
    sec_A()
    sec_B()
    sec_C()
    sec_D()
    with open(os.path.join(D, "robustness_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
