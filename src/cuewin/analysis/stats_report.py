"""Statistical backing for the paper's claims.

Output: ``data/stats_summary.csv`` + console report.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from cuewin.config import data_dir
from cuewin.stats import (
    N_BOOT,
    RNG,
    _cluster_ci_single,
    _cluster_ci_two,
    _series_dict,
    boot_ci,
    item_layer_mean,
    paired_perm_p,
    unpaired_perm_p,
)

CLAIMS = """
Claims tested (per model):
  C1  T2: subject causal score > adjective score      (layers 7-11 mean)
  C2  layer-9 concentration: subject score layer 9 vs mean of layers 0-6 (T1)
  C3  T6: attractor score < 0                          (layers 5-11 mean)
  C4  T3: conflicting-adjective score < 0              (layers 4-8 mean)
  C5  T4 subject weight < T1 subject weight            (layers 5-11 mean)
  C6  zeroing consistency: subject > adjective in T2   (layers 7-11 mean)
  C7  behavioral masc > fem accuracy                   (T1, proportion gap)
"""


def cluster_pass(pat, zer, beh, id2frame, id2subj, record):
    """Recompute each claim's per-item quantities and emit cluster-frame and
    cluster-subject bootstrap rows (Task 3s robustness pass)."""
    maps = (("cluster-frame", id2frame), ("cluster-subject", id2subj))

    for model in ("arabert", "camelbert", "arbert"):
        pm = pat[pat.model == model]
        zm = zer[zer.model == model]
        bm = beh[beh.model == model]

        t2 = pm[(pm.condition == "T2") & pm.correct_clean]
        s = item_layer_mean(t2[t2.cue == "subject"], range(7, 12))
        a = item_layer_mean(t2[t2.cue == "adjective"], range(7, 12))
        common = s.index.intersection(a.index)
        c1 = _series_dict(s[common] - a[common])

        t1 = pm[(pm.condition == "T1") & pm.correct_clean & (pm.cue == "subject")]
        l9 = t1[t1.layer == 9].set_index("id").score
        early = item_layer_mean(t1, range(0, 7))
        common = l9.index.intersection(early.index)
        c2 = _series_dict(l9[common] - early[common])

        t6 = pm[(pm.condition == "T6") & (pm.cue == "attractor")]
        c3 = _series_dict(item_layer_mean(t6, range(5, 12)))

        t3 = pm[(pm.condition == "T3") & (pm.cue == "adjective")]
        c4 = _series_dict(item_layer_mean(t3, range(4, 9)))

        c5_t1 = _series_dict(item_layer_mean(
            pm[(pm.condition == "T1") & pm.correct_clean & (pm.cue == "subject")],
            range(5, 12)))
        c5_t4 = _series_dict(item_layer_mean(
            pm[(pm.condition == "T4") & pm.correct_clean & (pm.cue == "subject")],
            range(5, 12)))

        z2 = zm[(zm.condition == "T2") & zm.layer.isin(range(7, 12))]
        zs = z2[z2.role == "subject"].groupby("id").score_norm.mean()
        za = z2[z2.role == "adjective"].groupby("id").score_norm.mean()
        common = zs.index.intersection(za.index)
        c6 = _series_dict(zs[common] - za[common])

        b1 = bm[bm.condition == "T1"]
        c7_m = _series_dict(b1[b1.gold == "m"].set_index("id").correct.astype(float))
        c7_f = _series_dict(b1[b1.gold == "f"].set_index("id").correct.astype(float))

        singles = [
            ("C1", "T2 subj−adj causal (L7-11)", c1),
            ("C2", "T1 subj: L9 − mean(L0-6)", c2),
            ("C3", "T6 attractor score (L5-11, all)", c3),
            ("C4", "T3 conflict-adj score (L4-8, all)", c4),
            ("C6", "T2 zeroing subj−adj (L7-11)", c6),
        ]
        twos = [
            ("C5", "subj weight T4−T1 (L5-11)", c5_t4, c5_t1),
            ("C7", "T1 acc masc−fem", c7_m, c7_f),
        ]

        for note, id2c in maps:
            for claim, desc, vals in singles:
                est, lo, hi, G = _cluster_ci_single(vals, id2c)
                record(claim, model, desc, est, lo, hi, None, G, note=note)
            for claim, desc, vB, vA in twos:
                est, lo, hi, G = _cluster_ci_two(vB, vA, id2c)
                record(claim, model, desc, est, lo, hi, None, G, note=note)


def main():
    base = data_dir()
    pat = pd.read_csv(os.path.join(base, "patching_results.csv"))
    zer = pd.read_csv(os.path.join(base, "zeroing_results.csv"))
    beh = pd.read_csv(os.path.join(base, "behavioral_results.csv"))

    rows = []

    def record(claim, model, desc, est, lo, hi, p, n, note=""):
        rows.append(dict(claim=claim, model=model, desc=desc,
                         estimate=round(est, 4), ci_lo=round(lo, 4),
                         ci_hi=round(hi, 4), p=p, n=n, note=note))
        star = "*" if (p is not None and p < 0.05) else " "
        ptxt = f"p={p:.4f}{star}" if p is not None else ""
        print(f"  [{claim}] {model:<10} {desc}: {est:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}] {ptxt} (n={n}) {note}")

    for model in ("arabert", "camelbert", "arbert"):
        print(f"\n================ {model} ================")
        pm = pat[pat.model == model]
        zm = zer[zer.model == model]
        bm = beh[beh.model == model]

        t2 = pm[(pm.condition == "T2") & pm.correct_clean]
        s = item_layer_mean(t2[t2.cue == "subject"], range(7, 12))
        a = item_layer_mean(t2[t2.cue == "adjective"], range(7, 12))
        common = s.index.intersection(a.index)
        diffs = (s[common] - a[common]).values
        est, lo, hi = boot_ci(diffs)
        record("C1", model, "T2 subj−adj causal (L7-11)", est, lo, hi,
               paired_perm_p(diffs), len(diffs))

        t1 = pm[(pm.condition == "T1") & pm.correct_clean & (pm.cue == "subject")]
        l9 = t1[t1.layer == 9].set_index("id").score
        early = item_layer_mean(t1, range(0, 7))
        common = l9.index.intersection(early.index)
        diffs = (l9[common] - early[common]).values
        est, lo, hi = boot_ci(diffs)
        record("C2", model, "T1 subj: L9 − mean(L0-6)", est, lo, hi,
               paired_perm_p(diffs), len(diffs))

        for tag in ("all", "correct"):
            t6 = pm[(pm.condition == "T6") & (pm.cue == "attractor")]
            if tag == "correct":
                t6 = t6[t6.correct_clean]
            vals = item_layer_mean(t6, range(5, 12)).values
            est, lo, hi = boot_ci(vals)
            record("C3", model, f"T6 attractor score (L5-11, {tag})",
                   est, lo, hi, paired_perm_p(vals), len(vals),
                   note=("PRIMARY" if tag == "all" else "secondary"))

        for tag in ("all", "correct"):
            t3 = pm[(pm.condition == "T3") & (pm.cue == "adjective")]
            if tag == "correct":
                t3 = t3[t3.correct_clean]
            vals = item_layer_mean(t3, range(4, 9)).values
            est, lo, hi = boot_ci(vals)
            record("C4", model, f"T3 conflict-adj score (L4-8, {tag})",
                   est, lo, hi, paired_perm_p(vals), len(vals),
                   note=("PRIMARY" if tag == "all" else "secondary"))

        t1v = item_layer_mean(
            pm[(pm.condition == "T1") & pm.correct_clean & (pm.cue == "subject")],
            range(5, 12)).values
        t4v = item_layer_mean(
            pm[(pm.condition == "T4") & pm.correct_clean & (pm.cue == "subject")],
            range(5, 12)).values
        d = t4v.mean() - t1v.mean()
        bi = RNG.integers(0, len(t1v), size=(N_BOOT, len(t1v)))
        bj = RNG.integers(0, len(t4v), size=(N_BOOT, len(t4v)))
        boots = t4v[bj].mean(axis=1) - t1v[bi].mean(axis=1)
        record("C5", model, "subj weight T4−T1 (L5-11)", d,
               np.percentile(boots, 2.5), np.percentile(boots, 97.5),
               unpaired_perm_p(t4v, t1v), len(t1v) + len(t4v))

        z2 = zm[(zm.condition == "T2") & zm.layer.isin(range(7, 12))]
        zs = z2[z2.role == "subject"].groupby("id").score_norm.mean()
        za = z2[z2.role == "adjective"].groupby("id").score_norm.mean()
        common = zs.index.intersection(za.index)
        diffs = (zs[common] - za[common]).values
        est, lo, hi = boot_ci(diffs)
        record("C6", model, "T2 zeroing subj−adj (L7-11)", est, lo, hi,
               paired_perm_p(diffs), len(diffs))

        b1 = bm[bm.condition == "T1"]
        am = b1[b1.gold == "m"].correct.values.astype(float)
        af = b1[b1.gold == "f"].correct.values.astype(float)
        d = am.mean() - af.mean()
        bi = RNG.integers(0, len(am), size=(N_BOOT, len(am)))
        bj = RNG.integers(0, len(af), size=(N_BOOT, len(af)))
        boots = am[bi].mean(axis=1) - af[bj].mean(axis=1)
        record("C7", model, "T1 acc masc−fem", d,
               np.percentile(boots, 2.5), np.percentile(boots, 97.5),
               unpaired_perm_p(am, af), len(am) + len(af))

    print("\n================ cluster bootstrap (frame & subject) ================")
    with open(os.path.join(base, "families.jsonl"), encoding="utf-8") as f:
        fam = [json.loads(l) for l in f]
    id2frame = {it["id"]: it["lex"]["verb_masc"] for it in fam}
    id2subj = {it["id"]: it["lex"]["subj_masc"] for it in fam}
    cluster_pass(pat, zer, beh, id2frame, id2subj, record)

    out = os.path.join(base, "stats_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
