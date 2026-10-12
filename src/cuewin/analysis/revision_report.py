#!/usr/bin/env python3
"""
Revision analyses (CPU). Reads committed result files plus the ``rev_*``
outputs of ``cuewin.pipeline.revision`` / ``decoder_fullsentence`` and writes
``data/revision_report.txt`` with every number quoted in the revised paper.

Sections:
  A  prior-corrected accuracy for all six conditions (submask and subjless
     frame priors from freq_baseline_frames.csv)
  B  lexical-ambiguity robustness: accuracy and patching with the eleven
     homograph subjects (feminine form is also a common inanimate noun or an
     adjective) excluded
  C  object-complement gender (T1/T4)
  D  per-layer extrema of the patching curves (subject vs interfering cue)
  E  repair by window: single layers, equal-length 4-layer windows, held-out
     window selection (subject split), zero vs gender-only patch vs control
  F  residual-stream patching: normalized effect at subject and [MASK]
  G  head-level value patching: share of the layer effect in the top head
  H  decoder full-sentence scoring
  I  natural dose-response: exact p-values of the joint regression
  J  MARBERT replication (behavior, patching peaks)
"""

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from cuewin.config import data_dir

D = data_dir()
OUT = os.path.join(D, "revision_report.txt")
ENC = ("arabert", "camelbert", "arbert", "marbert")
RNG = np.random.default_rng(43)
B = 10_000

# feminine forms that are also frequent inanimate nouns / adjectives
HOMOGRAPHS = {"الحكم", "المؤسس", "الجراح", "الحلاق", "الصيدلي", "المساعد", "المعالج",
              "الملاكم", "المترجم", "المستخدم", "المتخصص"}

lines = []


def out(s=""):
    print(s)
    lines.append(s)


def boot_ci(x, stat=np.mean, b=B):
    x = np.asarray(x)
    idx = RNG.integers(0, len(x), (b, len(x)))
    v = stat(x[idx], axis=1)
    return np.percentile(v, [2.5, 97.5])


def cluster_ci(df, col, cluster, stat="mean", b=2000):
    groups = {k: g[col].to_numpy() for k, g in df.groupby(cluster)}
    keys = list(groups)
    vals = []
    for _ in range(b):
        pick = RNG.integers(0, len(keys), len(keys))
        x = np.concatenate([groups[keys[i]] for i in pick])
        vals.append(x.mean())
    return np.percentile(vals, [2.5, 97.5])


def fam():
    items = {}
    path = os.path.join(D, "families_rev.jsonl")
    if not os.path.exists(path):
        path = os.path.join(D, "families_arbert.jsonl")
    for l in open(path, encoding="utf-8"):
        it = json.loads(l)
        items[it["id"]] = it
    return items


def behavioral():
    beh = pd.read_csv(os.path.join(D, "behavioral_results.csv"))
    p = os.path.join(D, "rev_marbert_behavioral.csv")
    if os.path.exists(p):
        beh = pd.concat([beh, pd.read_csv(p)], ignore_index=True)
    return beh


def sec_prior(beh, items):
    out("== A. Prior-corrected accuracy (masc / fem, %), all conditions ==")
    fr = pd.read_csv(os.path.join(D, "freq_baseline_frames.csv"))
    p = os.path.join(D, "rev_marbert_prior_frames.csv")
    if os.path.exists(p):
        fr = pd.concat([fr, pd.read_csv(p)], ignore_index=True)
    beh = beh.copy()
    beh["frame"] = beh.id.map(lambda i: items[i]["lex"]["verb_masc"])
    for base in ("submask", "subjless"):
        prior = fr[fr.baseline == base].set_index(["model", "verb_frame"])[
            "delta_prior_masc_minus_fem"]
        out(f"-- baseline {base}")
        for m in sorted(beh.model.unique(), key=ENC.index):
            sub = beh[beh.model == m].copy()
            key = list(zip(sub.model, sub.frame))
            if not all(k in prior.index for k in key):
                out(f"  {m}: no prior")
                continue
            dp = prior.loc[key].to_numpy()
            sub["corr"] = sub.logit_diff - np.where(sub.gold == "m", dp, -dp) > 0
            cells = []
            for c in sorted(sub.condition.unique()):
                s = sub[sub.condition == c]
                am, af = (100 * s[s.gold == g]["corr"].mean() for g in ("m", "f"))
                rm, rf = (100 * s[s.gold == g]["correct"].mean() for g in ("m", "f"))
                cells.append(f"{c} {am:.1f}/{af:.1f} (raw {rm:.1f}/{rf:.1f})")
            out(f"  {m}: " + " | ".join(cells))
            # gap with subject-cluster CI for T1 under this baseline
            s = sub[sub.condition == "T1"].copy()
            s["subj"] = s.id.map(lambda i: items[i]["lex"]["subj_masc"])
            gap = 100 * (s[s.gold == "m"]["corr"].mean() - s[s.gold == "f"]["corr"].mean())
            out(f"     T1 corrected masc-fem gap {gap:+.1f} pp")
    out()


def sec_homograph(beh, items):
    out("== B. Homograph subjects excluded ==")
    beh = beh.copy()
    beh["subj"] = beh.id.map(lambda i: items[i]["lex"]["subj_masc"])
    beh["kind"] = beh.id.map(lambda i: items[i]["lex"]["subj_kind"])
    beh["amb"] = beh.subj.isin(HOMOGRAPHS)
    n_amb = len({s for s in beh.subj.unique() if s in HOMOGRAPHS})
    out(f"homograph subjects present: {n_amb}")
    for m in sorted(beh.model.unique(), key=ENC.index):
        s = beh[(beh.model == m) & (beh.kind == "noun")]
        row = []
        for c in ("T1", "T3", "T4", "T6"):
            x = s[(s.condition == c) & (s.gold == "f")]
            a, u = 100 * x[x.amb].correct.mean(), 100 * x[~x.amb].correct.mean()
            row.append(f"{c} fem amb {a:.1f} vs unamb {u:.1f}")
        out(f"  {m} (noun subjects): " + " | ".join(row))
    out("  accuracy table with homographs excluded (masc/fem):")
    for m in sorted(beh.model.unique(), key=ENC.index):
        s = beh[(beh.model == m) & ~beh.amb]
        cells = [f"{c} {100 * s[(s.condition == c) & (s.gold == 'm')].correct.mean():.1f}/"
                 f"{100 * s[(s.condition == c) & (s.gold == 'f')].correct.mean():.1f}"
                 for c in sorted(s.condition.unique())]
        out(f"  {m}: " + " | ".join(cells))
    pat = pd.read_csv(os.path.join(D, "patching_results.csv"),
                      usecols=["id", "model", "condition", "cue", "layer", "score",
                               "correct_clean"])
    pat["amb"] = pat.id.map(lambda i: items[i]["lex"]["subj_masc"] in HOMOGRAPHS)
    out("  patching headline with homographs excluded (vs all):")
    for m in ("arabert", "camelbert", "arbert"):
        p = pat[pat.model == m]
        t1 = p[(p.condition == "T1") & (p.cue == "subject") & p.correct_clean]
        pk = lambda d: d.groupby("layer").score.mean()
        a, u = pk(t1), pk(t1[~t1.amb])
        t6 = p[(p.condition == "T6") & (p.cue == "attractor") & p.layer.between(5, 11)]
        out(f"    {m}: T1 subj peak L{a.idxmax()} {a.max():+.2f} -> L{u.idxmax()} {u.max():+.2f}; "
            f"T6 attr L5-11 {t6.score.mean():+.3f} -> {t6[~t6.amb].score.mean():+.3f}")
    # VSO check: inanimate feminine readings make masculine agreement possible in T4
    out("  word order (subject L5-11, correct items) with homographs excluded (vs all):")
    for m in ("arabert", "camelbert", "arbert"):
        p = pat[(pat.model == m) & (pat.cue == "subject") & pat.correct_clean
                & pat.layer.between(5, 11) & pat.condition.isin(["T1", "T4"])]
        a, u = p.groupby("condition").score.mean(), p[~p.amb].groupby("condition").score.mean()
        out(f"    {m}: T1 {a['T1']:.2f} T4 {a['T4']:.2f} -> T1 {u['T1']:.2f} T4 {u['T4']:.2f}")
    out()


def sec_object(beh, items):
    out("== C. Object-complement gender (accuracy, fem gold) ==")
    beh = beh.copy()
    beh["objF"] = beh.id.map(lambda i: items[i]["clean"].split()[-1].endswith("ة"))
    for m in sorted(beh.model.unique(), key=ENC.index):
        row = []
        for c in ("T1", "T4"):
            for g in ("m", "f"):
                s = beh[(beh.model == m) & (beh.condition == c) & (beh.gold == g)]
                row.append(f"{c}{g}: objF {100 * s[s.objF].correct.mean():.1f} "
                           f"objM {100 * s[~s.objF].correct.mean():.1f}")
        out(f"  {m}: " + " | ".join(row))
    out()


def sec_curves():
    out("== D. Patching-curve extrema (subject: correct-only; cues: all items) ==")
    frames = [pd.read_csv(os.path.join(D, "patching_results.csv"),
                          usecols=["model", "condition", "cue", "layer", "score",
                                   "correct_clean"])]
    p = os.path.join(D, "rev_marbert_patching.csv")
    if os.path.exists(p):
        frames.append(pd.read_csv(p, usecols=frames[0].columns.tolist()))
    pat = pd.concat(frames, ignore_index=True)
    for (m, c, cue), g in pat.groupby(["model", "condition", "cue"]):
        if cue == "subject":
            g = g[g.correct_clean]
        curve = g.groupby("layer").score.mean()
        L = curve.idxmax() if cue == "subject" else curve.idxmin()
        out(f"  {m:9s} {c} {cue:9s} extremum L{L} {curve[L]:+.3f} | "
            + " ".join(f"{v:+.2f}" for v in curve.to_numpy()))
    out()


def _rate(d, col="delta_int"):
    return 100 * (d[col] > 0).mean()


def sec_windows():
    out("== E. Repair by window ==")
    for m in ENC:
        p = os.path.join(D, f"rev_windows_{m}.csv")
        if not os.path.exists(p):
            continue
        w = pd.read_csv(p)
        for c in ("T6", "T3"):
            d = w[w.condition == c]
            wrong = d[d.delta_clean <= 0]
            right = d[d.delta_clean > 0]
            n_items = wrong.id.nunique()
            out(f"-- {m} {c}: n_wrong={n_items} n_correct={right.id.nunique()}")
            for kind in ("zero", "patch", "ctrl"):
                k = wrong[wrong.kind == kind]
                r = k.groupby("window").apply(_rate, include_groups=False)
                br = right[right.kind == kind].groupby("window").apply(
                    lambda x: 100 * (x.delta_int < 0).mean(), include_groups=False)
                singles = " ".join(f"{r[f'L{l}']:.0f}" for l in range(12))
                quads = " ".join(f"{k_}-{k_ + 3}:{r[f'L{k_}-{k_ + 3}']:.1f}" for k_ in range(9))
                out(f"   {kind:5s} single L0..11: {singles}")
                out(f"   {kind:5s} 4-layer: {quads} | 5-11: {r['L5-11']:.1f}")
                out(f"   {kind:5s} break 0-3 {br['L0-3']:.1f} 4-7 {br['L4-7']:.1f} "
                    f"8-11 {br['L8-11']:.1f} 5-11 {br['L5-11']:.1f}")
            # CIs for equal-length windows (zero and patch)
            for kind in ("zero", "patch", "ctrl"):
                cis = []
                for win in ("L0-3", "L4-7", "L8-11"):
                    x = (wrong[(wrong.kind == kind) & (wrong.window == win)].delta_int > 0
                         ).astype(float).to_numpy()
                    lo, hi = boot_ci(x)
                    cis.append(f"{win} {100 * x.mean():.1f} [{100 * lo:.1f},{100 * hi:.1f}]")
                out(f"   {kind:5s} CI: " + "; ".join(cis))
            # late-minus-early repair difference with a subject-cluster bootstrap
            for kind in ("patch", "zero"):
                k = wrong[wrong.kind == kind]
                late = k[k.window == "L8-11"].set_index("id").delta_int > 0
                early = k[k.window == "L0-3"].set_index("id").delta_int > 0
                d = pd.DataFrame({"diff": late.astype(float) - early.loc[late.index].astype(float),
                                  "subj": k[k.window == "L8-11"].set_index("id").subj})
                lo, hi = cluster_ci(d, "diff", "subj")
                out(f"   {kind:5s} L8-11 minus L0-3: {100 * d['diff'].mean():+.1f} pp, "
                    f"subject-cluster CI [{100 * lo:+.1f},{100 * hi:+.1f}] "
                    f"({d.subj.nunique()} subjects)")
            # held-out window selection: choose the best 4-layer window on half of
            # the subjects, score it on the other half (20 random subject splits)
            subs = sorted(wrong.subj.unique())
            quads = [f"L{k}-{k + 3}" for k in range(9)]
            for kind in ("patch", "zero"):
                kw = wrong[wrong.kind == kind]
                picks, test_rates, test_early = [], [], []
                for s_ in range(20):
                    rng = np.random.default_rng(100 + s_)
                    half = set(rng.permutation(subs)[: len(subs) // 2])
                    sel = kw[kw.subj.isin(half)]
                    best = max(quads, key=lambda q: (sel[sel.window == q].delta_int > 0).mean())
                    tst = kw[~kw.subj.isin(half)]
                    picks.append(best)
                    test_rates.append(100 * (tst[tst.window == best].delta_int > 0).mean())
                    test_early.append(100 * (tst[tst.window == "L0-3"].delta_int > 0).mean())
                vc = pd.Series(picks).value_counts().to_dict()
                out(f"   held-out {kind} (20 subject splits): chosen {vc}; test repair "
                    f"{np.mean(test_rates):.1f} (range {min(test_rates):.1f}-"
                    f"{max(test_rates):.1f}) vs early 0-3 {np.mean(test_early):.1f}")
    out()


def sec_residual():
    out("== F. Residual-stream patching (median normalized effect; 1 = full flip) ==")
    for m in ENC:
        p = os.path.join(D, f"rev_residual_{m}.csv")
        if not os.path.exists(p):
            continue
        r = pd.read_csv(p)
        r = r[(r.delta_clean - r.delta_corr).abs() > 0.5]       # items with a real effect
        r["eff"] = (r.delta_clean - r.delta_patched) / (r.delta_clean - r.delta_corr)
        for c in ("T1", "T4", "T6"):
            g = r[r.condition == c]
            pv = g.pivot_table(index="site", columns="layer", values="eff", aggfunc="median")
            for site in ("subject", "mask"):
                v = pv.loc[site]
                cross = next((l for l in v.index if v[l] < 0.5), None) if site == "subject" \
                    else next((l for l in v.index if v[l] >= 0.5), None)
                out(f"  {m:9s} {c} {site:7s} n={g.id.nunique()} 50%-crossing L{cross} | "
                    + " ".join(f"{x:.2f}" for x in v.to_numpy()))
    out()


def sec_heads():
    out("== G. Head-level value patching (mean score; top heads) ==")
    for m in ENC:
        p = os.path.join(D, f"rev_heads_{m}.csv")
        if not os.path.exists(p):
            continue
        h = pd.read_csv(p)
        maps = {}
        for c, cue in (("T1", "subject"), ("T6", "attractor")):
            g = h[(h.condition == c) & (h.cue == cue)]
            if cue == "subject":
                g = g[g.correct_clean]
            hm = g.groupby(["layer", "head"]).score.mean()
            total = hm.abs().sum()
            top = hm.abs().sort_values(ascending=False).head(3)
            desc = ", ".join(f"L{l}H{hh} {hm[(l, hh)]:+.2f}" for (l, hh) in top.index)
            share = 100 * top.iloc[0] / total
            share3 = 100 * top.sum() / total
            out(f"  {m:9s} {c} {cue}: top heads {desc}; top-1 share {share:.0f}%, "
                f"top-3 {share3:.0f}% of total |effect| over 144 heads")
            maps[cue] = hm
        # do the heads that read the subject also carry the attractor's effect?
        a, b = maps["subject"], maps["attractor"].reindex(maps["subject"].index)
        r = np.corrcoef(a, b)[0, 1]
        top5 = set(a.sort_values(ascending=False).head(5).index)
        top5b = set(b.sort_values().head(5).index)
        perm = [np.corrcoef(a, RNG.permutation(b.to_numpy()))[0, 1] for _ in range(10000)]
        p = (np.sum(np.abs(perm) >= abs(r)) + 1) / (len(perm) + 1)
        out(f"  {m:9s} head maps: r(T1 subject, T6 attractor) = {r:+.2f} "
            f"(permutation p={p:.1e}); top-5 overlap {len(top5 & top5b)}/5; "
            f"top subject head {tuple(int(v) for v in a.idxmax())} has attractor score {b[a.idxmax()]:+.2f} "
            f"(rank {int((b < b[a.idxmax()]).sum()) + 1}/144 most negative)")
        # robustness: rank correlation, and r without the three largest subject heads
        keep = ~a.index.isin(a.abs().sort_values(ascending=False).head(3).index)
        out(f"  {m:9s} head maps: Spearman rho = {a.corr(b, method='spearman'):+.2f}; "
            f"r without top-3 subject heads = {np.corrcoef(a[keep], b[keep])[0, 1]:+.2f}")
    out()


def sec_decoder():
    p = os.path.join(D, "rev_decoder_fullsentence.csv")
    if not os.path.exists(p):
        return
    out("== H. AraGPT2 full-sentence scoring (masc/fem %) ==")
    d = pd.read_csv(p)
    for c in sorted(d.condition.unique()):
        s = d[d.condition == c]
        f = lambda col, g: 100 * s[s.gold == g][col].mean()
        out(f"  {c}: full {f('correct_full', 'm'):.1f}/{f('correct_full', 'f'):.1f} "
            f"(overall {100 * s.correct_full.mean():.1f}) | verb-only "
            f"{f('correct_verb', 'm'):.1f}/{f('correct_verb', 'f'):.1f} "
            f"(overall {100 * s.correct_verb.mean():.1f}) | mean verb term "
            f"{s.delta_verb.mean():+.2f}, continuation term {s.delta_cont.mean():+.2f}")
    out()


def sec_dose():
    import statsmodels.formula.api as smf
    out("== I. Natural dose-response, joint OLS (exact p) ==")
    d = pd.read_csv(os.path.join(D, "natural_doseresponse.csv"))
    d["logw"] = np.log(d.n_words)
    for m in ("arabert", "camelbert", "arbert"):
        s = d[d.model == m]
        f = smf.ols("interference ~ logw + fem_count", s).fit()
        fc = smf.ols("interference ~ logw + fem_count", s).fit(
            cov_type="cluster", cov_kwds={"groups": s.verb_lemma.astype("category").cat.codes})
        out(f"  {m}: logw {f.params.logw:+.3f} p={f.pvalues.logw:.2e} (verb-cluster p="
            f"{fc.pvalues.logw:.2e}); fem {f.params.fem_count:+.4f} p={f.pvalues.fem_count:.3f}")
    out()


def sec_marbert():
    p = os.path.join(D, "rev_marbert_behavioral.csv")
    if not os.path.exists(p):
        return
    out("== J. MARBERT ==")
    b = pd.read_csv(p)
    out("  accuracy masc/fem: " + " | ".join(
        f"{c} {100 * b[(b.condition == c) & (b.gold == 'm')].correct.mean():.1f}/"
        f"{100 * b[(b.condition == c) & (b.gold == 'f')].correct.mean():.1f}"
        for c in sorted(b.condition.unique())))
    out(f"  n items {len(b)}")
    pat = pd.read_csv(os.path.join(D, "rev_marbert_patching.csv"))
    c1 = pat[(pat.condition == "T1") & (pat.cue == "subject") & pat.correct_clean]
    curve = c1.groupby("layer").score.mean()
    early = c1[c1.layer <= 6].groupby("id").score.mean()
    l9 = c1[c1.layer == curve.idxmax()].set_index("id").score
    diff = (l9 - early.loc[l9.index]).to_numpy()
    lo, hi = boot_ci(diff)
    out(f"  T1 subject peak L{curve.idxmax()} {curve.max():+.2f}; peak - mean(L0-6) "
        f"{diff.mean():+.2f} [{lo:.2f},{hi:.2f}]")
    t2 = pat[(pat.condition == "T2") & pat.correct_clean & pat.layer.between(7, 11)]
    sa = t2.groupby(["id", "cue"]).score.mean().unstack()
    x = (sa.subject - sa.adjective).to_numpy()
    lo, hi = boot_ci(x)
    out(f"  T2 subj {sa.subject.mean():+.2f} adj {sa.adjective.mean():+.2f} "
        f"diff {x.mean():+.2f} [{lo:.2f},{hi:.2f}]")
    for c, cue in (("T6", "attractor"), ("T3", "adjective")):
        g = pat[(pat.condition == c) & (pat.cue == cue) & pat.layer.between(5, 11)]
        per = g.groupby("id").score.mean().to_numpy()
        lo, hi = boot_ci(per)
        out(f"  {c} {cue} L5-11 (all items) {per.mean():+.3f} [{lo:.3f},{hi:.3f}]")
    t1 = pat[(pat.condition == "T1") & (pat.cue == "subject") & pat.correct_clean
             & pat.layer.between(5, 11)].groupby("id").score.mean()
    t4 = pat[(pat.condition == "T4") & (pat.cue == "subject") & pat.correct_clean
             & pat.layer.between(5, 11)].groupby("id").score.mean()
    a, b_ = t1.to_numpy(), t4.to_numpy()
    diffs = [RNG.choice(b_, len(b_)).mean() - RNG.choice(a, len(a)).mean() for _ in range(B)]
    out(f"  word order T4-T1 (L5-11) {b_.mean() - a.mean():+.3f} "
        f"[{np.percentile(diffs, 2.5):.3f},{np.percentile(diffs, 97.5):.3f}]")
    out()


def main():
    items = fam()
    beh = behavioral()
    sec_prior(beh, items)
    sec_homograph(beh, items)
    sec_object(beh, items)
    sec_curves()
    sec_windows()
    sec_residual()
    sec_heads()
    sec_decoder()
    sec_dose()
    sec_marbert()
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
