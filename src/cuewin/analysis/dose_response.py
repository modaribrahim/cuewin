"""P0 diagnostic: is the attractor effect a graded function of sentence length
/ distractor density within the NATURAL data?

Outputs: ``data/natural_doseresponse.csv``, ``data/natural_doseresponse_report.txt``

Note: statsmodels is avoided; the identical closed-form OLS is provided by
``cuewin.stats.ols_fit`` so the script runs without statsmodels.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from cuewin.config import data_dir
from cuewin.stats import RNG, boot_ci_mean, cluster_boot_ols, ols_fit, ols_fit_multi

LOCATED = range(5, 12)  # the located interference window (matches repair script)
N_BOOT = 500


def main():
    base = data_dir()
    fams = [os.path.join(base, f) for f in
            ("natural_families_T1_arbert.jsonl", "natural_families_T4_arbert.jsonl",
             "natural_families_T6_arbert.jsonl")]
    patch = os.path.join(base, "natural_patching.csv")
    behav = os.path.join(base, "natural_behavioral.csv")
    out_csv = os.path.join(base, "natural_doseresponse.csv")
    out_txt = os.path.join(base, "natural_doseresponse_report.txt")

    fam = {}
    for p in fams:
        for l in open(p, encoding="utf-8"):
            it = json.loads(l)
            fam[it["id"]] = it

    pat = pd.read_csv(patch)
    beh = pd.read_csv(behav)

    rows = []
    t6 = pat[(pat.condition == "T6") & (pat.cue == "attractor")]
    for iid, sub in t6.groupby("id"):
        it = fam[iid]
        n_words = len(it["clean"].split())
        fem_count = sum(1 for w in it["clean"].split()
                        if w.endswith("ة") and len(w) > 2)
        for model, d in sub.groupby("model"):
            inter = d.loc[d.layer.isin(LOCATED), "score"].mean()
            rows.append({
                "id": iid, "model": model, "verb_lemma": it["lex"]["verb_gloss"],
                "n_words": n_words, "fem_count": fem_count,
                "interference": inter,
            })
    dr = pd.DataFrame(rows)

    bcols = []
    for iid, sub in beh[beh.condition == "T6"].groupby("id"):
        bcols.append({"id": iid,
                      "b_logit_diff": sub.logit_diff.mean(),
                      "b_correct": sub.correct.mean()})
    bdf = pd.DataFrame(bcols)
    dr = dr.merge(bdf, on="id", how="left")

    dr.to_csv(out_csv, index=False)

    txt = []
    txt.append("P0 dose-response (natural T6 attractor), N items: "
               f"{dr.id.nunique()}\n")
    txt.append("Length/density range in the natural T6 set:")
    txt.append(f"  n_words   min..max: {dr.n_words.min()}..{dr.n_words.max()}  "
               f"median {dr.n_words.median():.0f}")
    txt.append(f"  fem_count min..max: {dr.fem_count.min()}..{dr.fem_count.max()}  "
               f"median {dr.fem_count.median():.0f}\n")

    for model in ("arabert", "camelbert", "arbert"):
        d = dr[dr.model == model]
        for xname, xcol in (("log(n_words)", np.log(d.n_words)),
                            ("fem_count", d.fem_count)):
            slope, p = ols_fit(xcol.values, d.interference.values)
            cs, c_lo, c_hi = cluster_boot_ols(d, "verb_lemma", xcol.name, "interference")
            txt.append(f"[{model}] interference ~ {xname}: "
                       f"OLS slope {slope:+.4f} (p={p:.3f}); "
                       f"cluster-boot slope {cs:+.4f} 95%CI [{c_lo:+.4f},{c_hi:+.4f}]")

    txt.append("")
    txt.append("Joint regression interference ~ log(n_words) + fem_count:")
    for model in ("arabert", "camelbert", "arbert"):
        d = dr[dr.model == model]
        beta, se, p = ols_fit_multi(
            [np.log(d.n_words).values, d.fem_count.values],
            d.interference.values)
        r = np.corrcoef(np.log(d.n_words), d.fem_count)[0, 1]
        txt.append(f"[{model}] len slope {beta[0]:+.3f} (p={p[0]:.3f}); "
                   f"fem slope {beta[1]:+.3f} (p={p[1]:.3f}); "
                   f"corr(log(n_words), fem_count) r={r:.2f}")

    txt.append("")
    for model in ("arabert", "camelbert", "arbert"):
        d = dr[dr.model == model]
        r1, p1 = stats.spearmanr(d.n_words, d.interference)
        r2, p2 = stats.spearmanr(d.fem_count, d.interference)
        txt.append(f"[{model}] Spearman interference~n_words r={r1:.3f} p={p1:.3f}; "
                   f"interference~fem_count r={r2:.3f} p={p2:.3f}")

    txt.append("\nBinned mean interference by n_words tercile:")
    dr["len_bin"] = pd.qcut(dr.n_words, 3, labels=["short", "mid", "long"])
    for model in ("arabert", "camelbert", "arbert"):
        d = dr[dr.model == model]
        txt.append(f"  {model}: " + "; ".join(
            f"{b}={d.loc[d.len_bin == b, 'interference'].mean():+.3f}"
            for b in ("short", "mid", "long")))

    txt.append("\n=== thin-n point estimates with cluster-boot CIs ===")
    for cond in ("T1", "T4"):
        for model in ("arabert", "camelbert", "arbert"):
            sc = pat[(pat.condition == cond) & (pat.cue == "subject") &
                     (pat.model == model) & (pat.layer == 9)]
            if not len(sc):
                continue
            n = len(sc)
            m, lo, hi = boot_ci_mean(sc.score.values, N_BOOT, RNG)
            txt.append(f"subject-patch L9 [{cond} {model}] n={n}: "
                       f"score {m:+.3f} 95%CI [{lo:+.3f},{hi:+.3f}]")

    rep = pd.read_csv(os.path.join(base, "natural_repair.csv"))
    for model in ("arabert", "camelbert", "arbert"):
        d = rep[rep.model == model]
        w = d[~d.clean_correct]
        if len(w):
            r = (w.delta_neut > 0).mean()
            ci = np.array([np.mean(RNG.choice((w.delta_neut > 0).values,
                                              size=len(w), replace=True))
                           for _ in range(2000)])
            txt.append(f"repair rate [{model}] n_wrong={len(w)}: "
                       f"{r:.3f} 95%CI [{np.percentile(ci, 2.5):.3f},"
                       f"{np.percentile(ci, 97.5):.3f}]")

    print("\n".join(txt))
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(txt) + "\n")
    print(f"\nsaved {out_csv}\nsaved {out_txt}")


if __name__ == "__main__":
    main()
