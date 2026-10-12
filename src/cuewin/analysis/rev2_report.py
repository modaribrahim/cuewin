"""Revision-2 summaries (read-only on CSVs; CPU). Writes data/rev2_report.txt.

Repair counts error items (delta_clean <= 0) whose intervened margin is
positive; breaks count correct items made incorrect. ``gap closed`` is the
ratio of means mean(delta_int - delta_clean) / mean(delta_corr - delta_clean)
over error items, where delta_corr is the margin with the cue fully flipped
in the input; it is 1 when an intervention matches the full flip.

Sections:
  R1  replication set: accuracy by condition (held-out cues, construct state)
  R2  replication set: value-patching scores in the fixed windows
  R3  replication set: repair in fixed windows (patch / zero / object control)
  R4  cumulative windows (prefix 0..k, suffix k..11), main and replication
  R5  cue hidden-state patching by layer, main and replication
  R6  main set: gap closed for the fixed-window repair of Table 5
  R7  name vs common-noun attractors
  R8  treebank: gender-only patch repair and cue hidden-state patching
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from cuewin.analysis import fixed_windows as fw
from cuewin.analysis.fixed_windows import (B, ENC, EARLY, LATE, MID, crossed, crossed_two,
                                           fmt, item_ci, lv, win)
from cuewin.config import data_dir

D = data_dir()
RNG_R = np.random.default_rng(50)
CUE_OF = {"T3": "adjective", "T6": "attractor", "T7": "attractor"}
WIN = (("0-3", "L0-3"), ("4-7", "L4-7"), ("8-11", "L8-11"))
lines = []


def out(s=""):
    print(s)
    lines.append(s)


def add_items(path, prefix_ok=None):
    n = 0
    for line in open(path, encoding="utf-8"):
        it = json.loads(line)
        lx = it["lex"]
        if fw.homograph_item(it):
            fw.EXCL.add(it["id"])
        cue = lx.get("attr_masc") or lx.get("adj_masc")
        fw.IT[it["id"]] = {"subj": lx["subj_masc"], "frame": lx["verb_masc"], "cue": cue,
                           "kind": lx["subj_kind"], "cue_kind": lx.get("attr_kind") or lx.get("adj_kind")}
        n += 1
    return n


def csv(name):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    if "id" in d.columns:
        d = d[~d.id.isin(fw.EXCL)].copy()
    if "score" not in d.columns and {"delta_clean", "delta_patched"} <= set(d.columns):
        d["score"] = d.delta_clean - d.delta_patched
    return d


def gap(wrong):
    den = (wrong.delta_corr - wrong.delta_clean).mean()
    return (wrong.delta_int - wrong.delta_clean).mean() / den if den else np.nan


def rate(x):
    return 100 * x.mean() if len(x) else np.nan


# ---------------------------------------------------------------- R1
def sec_R1():
    out("== R1. Replication set: accuracy (%) by condition, all items ==")
    main = {}
    for m in ENC:
        b = csv("rev_marbert_behavioral.csv") if m == "marbert" else csv("behavioral_results.csv")
        b = b[b.model == m]
        main[m] = b.groupby("condition").correct.mean() * 100
    for m in ENC:
        r = csv(f"rev2_rep_behavioral_{m}.csv")
        if r is None:
            continue
        acc = r.groupby("condition").correct.agg(["mean", "size"])
        parts = [f"{c} {100 * a:.1f} (n={n})" for c, (a, n) in acc.iterrows()]
        mm = main[m]
        out(f"  {m:9s} rep: " + "; ".join(parts))
        out(f"            main: T1 {mm['T1']:.1f}; T2 {mm['T2']:.1f}; T3 {mm['T3']:.1f}; T6 {mm['T6']:.1f}")
        for c in ("T3", "T6", "T7"):
            v = r[r.condition == c].set_index("id").correct.astype(float) * 100
            x = crossed(v, ("subj", "frame", "cue"))
            out(f"            rep {c} accuracy crossed s x f x cue {fmt(x, 1)} ({lv(x)})")
        for g in ("m", "f"):
            gg = r[r.gold == g].groupby("condition").correct.mean() * 100
            out(f"            gold {g}: " + "; ".join(f"{c} {v:.1f}" for c, v in gg.items()))
    out()


# ---------------------------------------------------------------- R2
def sec_R2():
    out("== R2. Replication set: value patching, fixed windows ==")
    for m in ENC:
        p = csv(f"rev2_rep_patching_{m}.csv")
        if p is None:
            continue
        out(f"[{m}]")
        t2 = p[p.condition == "T2"]
        sj, ad = win(t2[t2.cue == "subject"], LATE), win(t2[t2.cue == "adjective"], LATE)
        cm = sj.index.intersection(ad.index)
        x = crossed(sj[cm] - ad[cm], ("subj", "frame", "cue"))
        out(f"  T2 subject-minus-adjective late (all, n={len(cm)}): {fmt(x)} ({lv(x)}); "
            f"subj {sj[cm].mean():+.3f} adj {ad[cm].mean():+.3f}")
        for c, cue in CUE_OF.items():
            a = p[(p.condition == c) & (p.cue == cue)]
            if a.empty:
                continue
            per = a.groupby("layer").score.mean()
            x = crossed(win(a, LATE), ("subj", "frame", "cue"))
            out(f"  {c} {cue} late (all, n={a.id.nunique()}): {fmt(x)} ({lv(x)}); early "
                f"{win(a, EARLY).mean():+.3f} middle {win(a, MID).mean():+.3f}; most negative "
                f"L{per.idxmin()} ({per.min():+.2f})")
        s = p[p.cue == "subject"]
        pk = "; ".join(f"{c} L{s[s.condition == c].groupby('layer').score.mean().idxmax()}"
                       for c in sorted(s.condition.unique()))
        out(f"  subject peak layer (all items): {pk}")
    out()


# ---------------------------------------------------------------- R3
def repair_block(w, m, cond, factors=("subj", "frame", "cue")):
    d = w[w.condition == cond]
    wrong, right = d[d.delta_clean <= 0], d[d.delta_clean > 0]
    pw = wrong[wrong.kind == "patch"]
    late = pw[pw.window == "L8-11"].set_index("id")
    early = pw[pw.window == "L0-3"].set_index("id")
    diff = 100 * ((late.delta_int > 0).astype(float)
                  - (early.loc[late.index].delta_int > 0).astype(float))
    x = crossed(diff, factors) if len(diff) > 1 else (np.nan,) * 3 + ({},)
    rr = {(k, wn): rate(wrong[(wrong.kind == k) & (wrong.window == wl)].delta_int > 0)
          for k in ("patch", "zero", "ctrl") for wn, wl in WIN}
    br = {k: rate(right[(right.kind == k) & (right.window == "L8-11")].delta_int < 0)
          for k in ("patch", "zero", "ctrl")}
    out(f"  {m:9s} {cond} n_err={len(late)} n_correct={right.id.nunique()} late-minus-early "
        f"(patch) {fmt(x, 1)} ({lv(x)})")
    for k in ("patch", "zero", "ctrl"):
        out(f"            repair {k:5s} " + " / ".join(f"{wn} {rr[(k, wn)]:.1f}" for wn, _ in WIN)
            + f"; break 8-11 {br[k]:.1f}")


def sec_R3():
    out("== R3. Replication set: repair (error items) and breaks (correct items) ==")
    for m in ENC:
        w = csv(f"rev2_rep_windows_{m}.csv")
        if w is None:
            continue
        for c in CUE_OF:
            repair_block(w, m, c)
    out()


# ---------------------------------------------------------------- R4
def sec_R4():
    out("== R4. Cumulative windows (patch): repair % / gap closed, error items ==")
    for s in ("main", "rep"):
        for m in ENC:
            c = csv(f"rev2_cumwin_{s}_{m}.csv")
            if c is None:
                continue
            for cond in sorted(c.condition.unique()):
                d = c[(c.condition == cond) & (c.kind == "patch") & (c.delta_clean <= 0)]
                if d.empty:
                    continue
                n = d.id.nunique()
                pre = " ".join(f"{rate(g.delta_int > 0):.0f}/{gap(g):.2f}"
                               for _, g in sorted(d[d.window.str.startswith("P")].groupby("window"),
                                                  key=lambda t: int(t[0].split("-")[1])))
                suf = " ".join(f"{rate(g.delta_int > 0):.0f}/{gap(g):.2f}"
                               for _, g in sorted(d[d.window.str.startswith("S")].groupby("window"),
                                                  key=lambda t: int(t[0][1:].split("-")[0])))
                out(f"  {s:4s} {m:9s} {cond} n_err={n}")
                out(f"     prefix 0..k (k=0..11): {pre}")
                out(f"     suffix k..11 (k=0..11): {suf}")
                z = c[(c.condition == cond) & (c.kind == "zero") & (c.delta_clean <= 0)]
                zs = {wn: rate(z[z.window == wn].delta_int > 0) for wn in ("S0-11", "S4-11", "S8-11", "P0-3", "P0-7")}
                out("     zero: " + " ".join(f"{k} {v:.0f}" for k, v in zs.items()))
                # suffix(8) vs suffix(0), paired, crossed
                a = d[d.window == "S8-11"].set_index("id").delta_int > 0
                b = d[d.window == "S0-11"].set_index("id").delta_int > 0
                diff = 100 * (a.astype(float) - b.loc[a.index].astype(float))
                if len(diff) > 1:
                    x = crossed(diff, ("subj", "frame", "cue"))
                    out(f"     suffix 8..11 minus suffix 0..11 (repair, pts): {fmt(x, 1)} ({lv(x)})")
                a = d[d.window == "P0-7"].set_index("id").delta_int > 0
                if len(a) > 1:
                    x = crossed(100 * a.astype(float), ("subj", "frame", "cue"))
                    out(f"     prefix 0..7 repair: {fmt(x, 1)}")
    out()


# ---------------------------------------------------------------- R5
def sec_R5():
    out("== R5. Cue hidden-state patching by layer (l=-1 embedding): repair % / gap closed ==")
    for s in ("main", "rep"):
        for m in ENC:
            r = csv(f"rev2_cueresid_{s}_{m}.csv")
            if r is None:
                continue
            for cond in sorted(r.condition.unique()):
                d = r[(r.condition == cond) & (r.delta_clean <= 0)].rename(columns={"delta_patched": "delta_int"})
                row = " ".join(f"{rate(g.delta_int > 0):.0f}/{gap(g):.2f}"
                               for _, g in sorted(d.groupby("layer"), key=lambda t: t[0]))
                out(f"  {s:4s} {m:9s} {cond} n_err={d.id.nunique()}: {row}")
    out()


# ---------------------------------------------------------------- R6
def sec_R6():
    out("== R6. Main set: gap closed by the fixed-window repair (error items) ==")
    for m in ENC:
        r = csv(f"rev2_cueresid_main_{m}.csv")
        w = csv(f"rev_windows_{m}.csv")
        if r is None or w is None:
            continue
        corr = r[r.layer == -1].set_index("id").delta_corr
        w = w[w.id.isin(corr.index)].copy()
        w["delta_corr"] = w.id.map(corr)
        for cond in ("T3", "T6"):
            d = w[(w.condition == cond) & (w.delta_clean <= 0)]
            parts = []
            for k in ("patch", "zero", "ctrl"):
                parts.append(f"{k} " + " / ".join(f"{wn} {gap(d[(d.kind == k) & (d.window == wl)]):.2f}"
                                                  for wn, wl in WIN))
            out(f"  {m:9s} {cond} n_err={d.id.nunique()}: " + "; ".join(parts))
    out()


# ---------------------------------------------------------------- R7
def sec_R7():
    out("== R7. Name vs common-noun attractors: repair (patch, 8-11) and accuracy ==")
    for m in ENC:
        w = csv(f"rev_windows_{m}.csv")
        if w is not None:
            d = w[(w.condition == "T6") & (w.kind == "patch") & (w.window == "L8-11")].copy()
            d["is_name"] = d.id.map(lambda i: fw.IT[i]["cue"] in fw.NAME_ATTR)
            e = d[d.delta_clean <= 0]
            out(f"  main {m:9s} " + "; ".join(
                f"{'name' if k else 'noun'}: acc {rate(g.delta_clean > 0):.1f} (n={len(g)}), repair "
                f"{rate(e[e.is_name == k].delta_int > 0):.1f} (n_err={(e.is_name == k).sum()})"
                for k, g in d.groupby("is_name")))
        w = csv(f"rev2_rep_windows_{m}.csv")
        if w is not None:
            for cond in ("T6", "T7"):
                d = w[(w.condition == cond) & (w.kind == "patch") & (w.window == "L8-11")].copy()
                d["ck"] = d.id.map(lambda i: fw.IT[i]["cue_kind"])
                e = d[d.delta_clean <= 0]
                out(f"  rep  {m:9s} {cond} " + "; ".join(
                    f"{k}: acc {rate(g.delta_clean > 0):.1f} (n={len(g)}), repair "
                    f"{rate(e[e.ck == k].delta_int > 0):.1f} (n_err={(e.ck == k).sum()})"
                    for k, g in d.groupby("ck")))
    out()


# ---------------------------------------------------------------- R8
def sec_R8():
    out("== R8. Treebank attractor (PADT): gender-only patch ==")
    for m in ENC:
        w = csv(f"rev2_nat_windows_{m}.csv")
        if w is None:
            continue
        wrong, right = w[w.delta_clean <= 0], w[w.delta_clean > 0]
        n = wrong.id.nunique()
        out(f"  {m:9s} n_err={n} n_correct={right.id.nunique()}")
        for k in ("patch", "zero"):
            g = wrong[wrong.kind == k]
            vals = []
            for wl in ("L0-3", "L4-7", "L8-11", "L5-11", "S0-11", "S4-11", "S8-11"):
                gg = g[g.window == wl]
                vals.append(f"{wl} {rate(gg.delta_int > 0):.1f}/{gap(gg):.2f}")
            br = rate(right[(right.kind == k) & (right.window == "L8-11")].delta_int < 0)
            out(f"     {k:5s} repair %/gap: " + "; ".join(vals) + f"; break 8-11 {br:.1f}")
        g = wrong[wrong.kind == "patch"]
        late = g[g.window == "L8-11"].set_index("id").delta_int > 0
        early = g[g.window == "L0-3"].set_index("id").delta_int > 0
        if len(late) > 1:
            x = item_ci(100 * (late.astype(float) - early.loc[late.index].astype(float)), RNG_R)
            out(f"     late-minus-early repair (patch, pts, item bootstrap seed 50): {fmt(x, 1)}")
        r = csv(f"rev2_nat_cueresid_{m}.csv")
        if r is not None:
            d = r[r.delta_clean <= 0].rename(columns={"delta_patched": "delta_int"})
            out("     cue hidden state by layer: " + " ".join(
                f"{rate(g.delta_int > 0):.0f}/{gap(g):.2f}" for _, g in sorted(d.groupby("layer"))))
        # gap closed on all items (scale-free comparison with templates)
        allp = w[(w.kind == "patch") & (w.window == "L8-11")]
        out(f"     gap closed, all items, patch 8-11: {gap(allp):.2f}")
    out()


def main():
    add_items(os.path.join(D, "families_rep.jsonl"))
    add_items(os.path.join(D, "natural_families_T6_arbert.jsonl"))
    out(f"Revision-2 report. Crossed bootstrap B={B} (seed 47); treebank item bootstrap seed 50.")
    out()
    for f in (sec_R1, sec_R2, sec_R3, sec_R4, sec_R5, sec_R6, sec_R7, sec_R8):
        f()
    with open(os.path.join(D, "rev2_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
