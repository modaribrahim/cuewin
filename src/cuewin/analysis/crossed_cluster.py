"""Crossed subject x verb-frame bootstrap for the main template effects, and the
repair-by-depth profile (read-only on CSVs).

The two-way ("pigeonhole") bootstrap resamples subjects and verb frames
independently with replacement; each item is weighted by the number of times
its subject and its frame were drawn. Writes data/crossed_cluster.txt.

Effects:
  subject excess   T1, correct items: score at layer 9 (layer 8 for MARBERT)
                   minus the layer 0-6 mean
  attractor        T6, all items, mean score over layers 5-11
  adjective        T3, all items, mean score over layers 5-11
  repair L8-11 - L0-3   gender-only patch, error items, T6 and T3
  subject - adjective   T2, correct items, layers 7-11 (seed 46)
  word order T4 - T1    subject, correct items, layers 5-11 (seed 46)
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from cuewin.config import data_dir

D = data_dir()
ENC = ("arabert", "camelbert", "arbert", "marbert")
PEAK = {"arabert": 9, "camelbert": 9, "arbert": 9, "marbert": 8}
B = 10_000
RNG = np.random.default_rng(44)
RNG2 = np.random.default_rng(46)   # added effects; keeps the seed-44 rows unchanged
lines = []


def out(s=""):
    print(s)
    lines.append(s)


def item_clusters():
    path = os.path.join(D, "families_rev.jsonl")
    m = {}
    for line in open(path, encoding="utf-8"):
        it = json.loads(line)
        m[it["id"]] = (it["lex"]["subj_masc"], it["lex"]["verb_masc"])
    return m


def crossed_ci(vals: pd.Series, subj: pd.Series, frame: pd.Series, rng=RNG):
    """Mean of vals with a crossed subject x frame bootstrap CI."""
    df = pd.DataFrame({"v": vals.to_numpy(), "s": subj.to_numpy(), "f": frame.to_numpy()})
    s_codes, s_keys = pd.factorize(df.s)
    f_codes, f_keys = pd.factorize(df.f)
    S, F = len(s_keys), len(f_keys)
    sums = np.zeros((S, F))
    cnts = np.zeros((S, F))
    np.add.at(sums, (s_codes, f_codes), df.v.to_numpy())
    np.add.at(cnts, (s_codes, f_codes), 1.0)
    ws = rng.multinomial(S, np.full(S, 1 / S), size=B).astype(float)
    wf = rng.multinomial(F, np.full(F, 1 / F), size=B).astype(float)
    num = np.einsum("bs,sf,bf->b", ws, sums, wf)
    den = np.einsum("bs,sf,bf->b", ws, cnts, wf)
    ok = den > 0
    boots = num[ok] / den[ok]
    return df.v.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), S, F


def crossed_ci_two(vB: pd.Series, vA: pd.Series, clus, rng=RNG2):
    """mean(vB) - mean(vA) with a crossed subject x frame bootstrap CI; both
    groups are reweighted by the same subject and frame draws."""
    ids = list(vB.index) + list(vA.index)
    s_codes, s_keys = pd.factorize(pd.Series([clus[i][0] for i in ids]))
    f_codes, f_keys = pd.factorize(pd.Series([clus[i][1] for i in ids]))
    S, F = len(s_keys), len(f_keys)
    nB = len(vB)
    mats = []
    for sl, v in ((slice(0, nB), vB), (slice(nB, None), vA)):
        sums, cnts = np.zeros((S, F)), np.zeros((S, F))
        np.add.at(sums, (s_codes[sl], f_codes[sl]), v.to_numpy())
        np.add.at(cnts, (s_codes[sl], f_codes[sl]), 1.0)
        mats.append((sums, cnts))
    ws = rng.multinomial(S, np.full(S, 1 / S), size=B).astype(float)
    wf = rng.multinomial(F, np.full(F, 1 / F), size=B).astype(float)
    means = []
    for sums, cnts in mats:
        num = np.einsum("bs,sf,bf->b", ws, sums, wf)
        den = np.einsum("bs,sf,bf->b", ws, cnts, wf)
        with np.errstate(invalid="ignore", divide="ignore"):
            means.append(num / den)
    boots = means[0] - means[1]
    boots = boots[np.isfinite(boots)]
    return vB.mean() - vA.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), S, F


def load_patching(model):
    if model == "marbert":
        p = pd.read_csv(os.path.join(D, "rev_marbert_patching.csv"))
    else:
        p = pd.read_csv(os.path.join(D, "patching_results.csv"))
        p = p[p.model == model]
    return p


def sec_effects(clus):
    out("== Crossed subject x frame bootstrap (B=10000, seed 44) ==")
    for m in ENC:
        p = load_patching(m)
        rows = []
        # subject excess at the peak layer over the layer 0-6 mean (T1, correct items)
        s = p[(p.condition == "T1") & (p.cue == "subject") & p.correct_clean]
        peak = s[s.layer == PEAK[m]].set_index("id").score
        early = s[s.layer.between(0, 6)].groupby("id").score.mean()
        rows.append((f"subject T1 L{PEAK[m]} minus L0-6", (peak - early.loc[peak.index])))
        for cond, cue, lab in (("T6", "attractor", "attractor T6 L5-11"),
                               ("T3", "adjective", "adjective T3 L5-11")):
            a = p[(p.condition == cond) & (p.cue == cue) & p.layer.between(5, 11)]
            rows.append((lab, a.groupby("id").score.mean()))
        for lab, v in rows:
            sj = v.index.map(lambda i: clus[i][0])
            fr = v.index.map(lambda i: clus[i][1])
            pt, lo, hi, S, F = crossed_ci(v, pd.Series(sj), pd.Series(fr))
            out(f"  {m:9s} {lab:28s} {pt:+.4f} [{lo:+.4f},{hi:+.4f}] ({S} subj x {F} frames)")
        w = pd.read_csv(os.path.join(D, f"rev_windows_{m}.csv"))
        w = w[(w.kind == "patch") & (w.delta_clean <= 0)]
        for cond in ("T6", "T3"):
            d = w[w.condition == cond]
            late = d[d.window == "L8-11"].set_index("id")
            early = d[d.window == "L0-3"].set_index("id")
            diff = (late.delta_int > 0).astype(float) - (early.loc[late.index].delta_int > 0).astype(float)
            pt, lo, hi, S, F = crossed_ci(100 * diff, late.subj, late.frame)
            out(f"  {m:9s} {'repair ' + cond + ' L8-11 minus L0-3':28s} {pt:+.1f} pp "
                f"[{lo:+.1f},{hi:+.1f}] ({S} subj x {F} frames)")


def sec_effects2(clus):
    out()
    out("== Crossed bootstrap, added effects (B=10000, seed 46) ==")
    for m in ENC:
        p = load_patching(m)
        p = p[p.correct_clean]
        t2 = p[(p.condition == "T2") & p.layer.between(7, 11)]
        sj = t2[t2.cue == "subject"].groupby("id").score.mean()
        ad = t2[t2.cue == "adjective"].groupby("id").score.mean()
        common = sj.index.intersection(ad.index)
        d = sj[common] - ad[common]
        pt, lo, hi, S, F = crossed_ci(d, pd.Series(d.index.map(lambda i: clus[i][0])),
                                      pd.Series(d.index.map(lambda i: clus[i][1])), rng=RNG2)
        out(f"  {m:9s} {'subject - adjective T2 L7-11':28s} {pt:+.4f} [{lo:+.4f},{hi:+.4f}] "
            f"({S} subj x {F} frames)")
        sub = p[(p.cue == "subject") & p.layer.between(5, 11)]
        t4 = sub[sub.condition == "T4"].groupby("id").score.mean()
        t1 = sub[sub.condition == "T1"].groupby("id").score.mean()
        pt, lo, hi, S, F = crossed_ci_two(t4, t1, clus)
        out(f"  {m:9s} {'word order T4 - T1 L5-11':28s} {pt:+.4f} [{lo:+.4f},{hi:+.4f}] "
            f"({S} subj x {F} frames)")


def sec_depth():
    out()
    out("== Repair by depth: gender-only patch, error items ==")
    quads = [f"L{k}-{k + 3}" for k in range(9)]
    for m in ENC:
        w = pd.read_csv(os.path.join(D, f"rev_windows_{m}.csv"))
        w = w[(w.kind == "patch") & (w.delta_clean <= 0)]
        for cond in ("T6", "T3"):
            d = w[w.condition == cond]
            r = {q: 100 * (d[d.window == q].delta_int > 0).mean() for q in quads}
            best = max(r, key=r.get)
            singles = {l: 100 * (d[d.window == f"L{l}"].delta_int > 0).mean() for l in range(12)}
            bl = max(singles, key=singles.get)
            out(f"  {m:9s} {cond}: best 4-layer window {best} ({r[best]:.1f}), "
                f"last window L8-11 {r['L8-11']:.1f}; best single layer L{bl} "
                f"({singles[bl]:.1f}), L11 {singles[11]:.1f}")
            out("            4-layer: " + " ".join(f"{q[1:]}:{r[q]:.1f}" for q in quads))
            out("            single:  " + " ".join(f"{l}:{singles[l]:.0f}" for l in range(12)))


def main():
    clus = item_clusters()
    sec_effects(clus)
    sec_depth()
    sec_effects2(clus)
    with open(os.path.join(D, "crossed_cluster.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
