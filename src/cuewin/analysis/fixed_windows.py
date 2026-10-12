"""Summary statistics under one fixed window rule (read-only on CSVs; CPU).

Every layer summary uses the fixed four-layer windows early = 0-3, middle = 4-7,
late = 8-11; peak layers are per-encoder argmax. The crossed bootstrap resamples
subjects, verb frames and, for the adjective and attractor conditions, the
interfering-cue lexical item (adjective pair or attractor pair). Writes
data/fixed_windows_report.txt.

Sections:
  A  value-patching summary (Table 4), fixed windows, item and crossed CIs
  B  alternative (longer) windows vs fixed windows (point estimates)
  H  morpheme probe: suffix-minus-stem difference at layer 9 (item bootstrap)
  C  causal repair: late-minus-early contrast with cue resampling; break rates
  D  hidden-state patching: items excluded by the 0.5-logit rule (Eq. 3)
  E  lexicon shares (name attractors) and item counts per behavioral cell
  F  robustness (homographs, attractor type, value zeroing) in fixed windows
  G  treebank: attractor (8-11), regression, subject scores, VSO adjacency, repair
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from cuewin.config import data_dir

D = data_dir()
ENC = ("arabert", "camelbert", "arbert", "marbert")
MSA = ("arabert", "camelbert", "arbert")
EARLY, MID, LATE = (0, 3), (4, 7), (8, 11)
B = 10_000
RNG_X = np.random.default_rng(47)    # crossed bootstrap
RNG_I = np.random.default_rng(48)    # item bootstrap
RNG_N = np.random.default_rng(49)    # treebank bootstrap
HOMOGRAPHS = {"الحكم", "المؤسس", "الجراح", "الحلاق", "الصيدلي", "المساعد", "المعالج",
              "الملاكم", "المترجم", "المستخدم", "المتخصص"}
NAME_ATTR = {"محمد", "أحمد", "خالد"}
INANIMATE_NAT = {"الحكم", "المؤسسة"}  # treebank subjects inanimate in context
# Held-out items whose attractor, or construct-state (T7) subject, is one of the
# homographs above are excluded from all revision-2 analyses (filled by add_items).
EXCL = set()


def homograph_item(it):
    lx = it["lex"]
    return it["id"].startswith("R") and (
        lx.get("attr_masc") in HOMOGRAPHS
        or (it["condition"] == "T7" and lx["subj_masc"] in HOMOGRAPHS))
lines = []


def out(s=""):
    print(s)
    lines.append(s)


def items():
    m = {}
    for line in open(os.path.join(D, "families_rev.jsonl"), encoding="utf-8"):
        it = json.loads(line)
        lx = it["lex"]
        cue = lx["attr_masc"] if it["condition"] == "T6" else lx["adj_masc"]
        m[it["id"]] = {"subj": lx["subj_masc"], "frame": lx["verb_masc"], "cue": cue,
                       "kind": lx["subj_kind"]}
    return m


IT = items()


def load_patching(model):
    if model == "marbert":
        return pd.read_csv(os.path.join(D, "rev_marbert_patching.csv"))
    p = pd.read_csv(os.path.join(D, "patching_results.csv"))
    return p[p.model == model]


def win(df, w):
    return df[df.layer.between(*w)].groupby("id").score.mean()


def factor_weights(ids, factors, rng):
    """Bootstrap weights (B x n_items): product over factors of the number of
    times each item's level was drawn."""
    W = np.ones((B, len(ids)))
    for f in factors:
        codes, keys = pd.factorize(pd.Series([IT[i][f] for i in ids]))
        k = len(keys)
        draws = rng.multinomial(k, np.full(k, 1 / k), size=B).astype(float)
        W *= draws[:, codes]
    return W


def crossed(v: pd.Series, factors=("subj", "frame"), rng=RNG_X):
    ids = list(v.index)
    W = factor_weights(ids, factors, rng)
    den = W.sum(1)
    ok = den > 0
    boots = (W @ v.to_numpy())[ok] / den[ok]
    lv = {f: len({IT[i][f] for i in ids}) for f in factors}
    return v.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), lv


def crossed_two(vB: pd.Series, vA: pd.Series, factors=("subj", "frame"), rng=RNG_X):
    ids = list(vB.index) + list(vA.index)
    W = factor_weights(ids, factors, rng)
    nB = len(vB)
    with np.errstate(invalid="ignore", divide="ignore"):
        mB = (W[:, :nB] @ vB.to_numpy()) / W[:, :nB].sum(1)
        mA = (W[:, nB:] @ vA.to_numpy()) / W[:, nB:].sum(1)
    boots = (mB - mA)[np.isfinite(mB - mA)]
    lv = {f: len({IT[i][f] for i in ids}) for f in factors}
    return vB.mean() - vA.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), lv


def item_ci(v, rng=RNG_I):
    x = np.asarray(v, float)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def item_ci_two(b, a, rng=RNG_I):
    b, a = np.asarray(b, float), np.asarray(a, float)
    mb = b[rng.integers(0, len(b), size=(B, len(b)))].mean(1)
    ma = a[rng.integers(0, len(a), size=(B, len(a)))].mean(1)
    d = mb - ma
    return b.mean() - a.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def fmt(t, d=3):
    pt, lo, hi = t[:3]
    return f"{pt:+.{d}f} [{lo:+.{d}f},{hi:+.{d}f}]"


def lv(t):
    return " x ".join(f"{v} {k}" for k, v in t[3].items())


# ---------------------------------------------------------------- A
def sec_A():
    out("== A. Value-patching summary, fixed windows (early 0-3, late 8-11) ==")
    out(f"   crossed bootstrap B={B}, seed 47; item bootstrap B={B}, seed 48")
    for m in ENC:
        p = load_patching(m)
        sub = p[p.cue == "subject"]
        out(f"[{m}]")
        # peak layers (argmax), correct items and all items, every condition
        pk = []
        for c in sorted(sub.condition.unique()):
            sc = sub[(sub.condition == c) & sub.correct_clean].groupby("layer").score.mean()
            sa = sub[sub.condition == c].groupby("layer").score.mean()
            pk.append(f"{c} L{sc.idxmax()} ({sc.max():+.2f}) / all L{sa.idxmax()}")
        out("  subject peak (correct / all items): " + "; ".join(pk))
        s1 = sub[(sub.condition == "T1") & sub.correct_clean]
        d = win(s1, LATE) - win(s1, EARLY)
        x = crossed(d)
        out(f"  subject T1 late-minus-early (correct, n={len(d)}): item {fmt(item_ci(d))}; "
            f"crossed {fmt(x)} ({lv(x)})")
        out(f"    T1 late mean {win(s1, LATE).mean():+.3f}, early mean {win(s1, EARLY).mean():+.3f}")
        # subject minus adjective, T2, late window
        for tag, pp in (("all", p), ("correct", p[p.correct_clean])):
            t2 = pp[pp.condition == "T2"]
            sj = win(t2[t2.cue == "subject"], LATE)
            ad = win(t2[t2.cue == "adjective"], LATE)
            cm = sj.index.intersection(ad.index)
            dd = sj[cm] - ad[cm]
            x = crossed(dd, ("subj", "frame", "cue"))
            out(f"  T2 subject-minus-adjective late ({tag}, n={len(dd)}): item {fmt(item_ci(dd))}; "
                f"crossed s x f x adj {fmt(x)} ({lv(x)}); subj {sj[cm].mean():+.3f} "
                f"adj {ad[cm].mean():+.3f} ratio {sj[cm].mean() / ad[cm].mean():.1f}")
        for cond, cue in (("T3", "adjective"), ("T6", "attractor")):
            a = p[(p.condition == cond) & (p.cue == cue)]
            v = win(a, LATE)
            x2 = crossed(v)
            x3 = crossed(v, ("subj", "frame", "cue"))
            per = a.groupby("layer").score.mean()
            out(f"  {cond} {cue} late (all, n={len(v)}): item {fmt(item_ci(v))}; crossed s x f "
                f"{fmt(x2)}; crossed s x f x cue {fmt(x3)} ({lv(x3)}); most negative "
                f"L{per.idxmin()} ({per.min():+.2f})")
        for tag, pp in (("all", p), ("correct", p[p.correct_clean])):
            s = pp[pp.cue == "subject"]
            t4, t1 = win(s[s.condition == "T4"], LATE), win(s[s.condition == "T1"], LATE)
            x = crossed_two(t4, t1)
            out(f"  subject VSO-minus-SVO late ({tag}, n={len(t4)}/{len(t1)}): item "
                f"{fmt(item_ci_two(t4, t1))}; crossed {fmt(x)}; T4 {t4.mean():+.3f} T1 {t1.mean():+.3f}")
    out()


# ---------------------------------------------------------------- B
def sec_B():
    out("== B. Alternative windows vs fixed windows (point estimates) ==")
    old_peak = {"arabert": 9, "camelbert": 9, "arbert": 9, "marbert": 8}
    for m in ENC:
        p = load_patching(m)
        c = p[p.correct_clean]
        s1 = c[(c.condition == "T1") & (c.cue == "subject")]
        pk = s1[s1.layer == old_peak[m]].set_index("id").score
        o1 = (pk - win(s1, (0, 6)).loc[pk.index]).mean()
        n1 = (win(s1, LATE) - win(s1, EARLY)).mean()
        t2 = c[c.condition == "T2"]
        o2 = (win(t2[t2.cue == "subject"], (7, 11)) - win(t2[t2.cue == "adjective"], (7, 11))).dropna().mean()
        t2a = p[p.condition == "T2"]
        n2 = (win(t2a[t2a.cue == "subject"], LATE) - win(t2a[t2a.cue == "adjective"], LATE)).dropna().mean()
        o3 = win(p[(p.condition == "T3") & (p.cue == "adjective")], (5, 11)).mean()
        n3 = win(p[(p.condition == "T3") & (p.cue == "adjective")], LATE).mean()
        o6 = win(p[(p.condition == "T6") & (p.cue == "attractor")], (5, 11)).mean()
        n6 = win(p[(p.condition == "T6") & (p.cue == "attractor")], LATE).mean()
        sc = c[c.cue == "subject"]
        o4 = win(sc[sc.condition == "T4"], (5, 11)).mean() - win(sc[sc.condition == "T1"], (5, 11)).mean()
        sa = p[p.cue == "subject"]
        n4 = win(sa[sa.condition == "T4"], LATE).mean() - win(sa[sa.condition == "T1"], LATE).mean()
        out(f"  {m:9s} subj excess old(L{old_peak[m]}-L0-6) {o1:+.3f} new(8-11 - 0-3) {n1:+.3f} | "
            f"subj-adj old(7-11,corr) {o2:+.3f} new(8-11,all) {n2:+.3f} | "
            f"T3 old(5-11) {o3:+.3f} new {n3:+.3f} | T6 old(5-11) {o6:+.3f} new {n6:+.3f} | "
            f"VSO-SVO old(5-11,corr) {o4:+.3f} new(8-11,all) {n4:+.3f}")
    out()


# ---------------------------------------------------------------- C
def sec_C():
    out("== C. Causal repair (error items; break = correct items made incorrect) ==")
    for m in ENC:
        w = pd.read_csv(os.path.join(D, f"rev_windows_{m}.csv"))
        for cond in ("T6", "T3"):
            d = w[w.condition == cond]
            wrong, right = d[d.delta_clean <= 0], d[d.delta_clean > 0]
            pw = wrong[wrong.kind == "patch"]
            late = pw[pw.window == "L8-11"].set_index("id")
            early = pw[pw.window == "L0-3"].set_index("id")
            diff = 100 * ((late.delta_int > 0).astype(float)
                          - (early.loc[late.index].delta_int > 0).astype(float))
            x2 = crossed(diff)
            x3 = crossed(diff, ("subj", "frame", "cue"))
            br = {k: 100 * (right[(right.kind == k) & (right.window == "L8-11")].delta_int < 0).mean()
                  for k in ("patch", "zero", "ctrl")}
            rr = {k: 100 * (wrong[(wrong.kind == k) & (wrong.window == "L8-11")].delta_int > 0).mean()
                  for k in ("patch", "zero", "ctrl")}
            out(f"  {m:9s} {cond} n_err={len(late)} late-minus-early {fmt(x2, 1)} s x f; "
                f"{fmt(x3, 1)} s x f x cue ({lv(x3)})")
            out(f"            L8-11 repair patch/zero/ctrl {rr['patch']:.1f}/{rr['zero']:.1f}/{rr['ctrl']:.1f}; "
                f"break patch/zero/ctrl {br['patch']:.1f}/{br['zero']:.1f}/{br['ctrl']:.1f} "
                f"(n_correct={right[(right.kind == 'patch') & (right.window == 'L8-11')].id.nunique()})")
            # threshold check: split error items at the median |delta_clean|
            med = pw.delta_clean.abs().median()
            half = {}
            for nm, s in (("near", pw[pw.delta_clean.abs() <= med]),
                          ("far", pw[pw.delta_clean.abs() > med])):
                half[nm] = tuple(100 * (s[s.window == x].delta_int > 0).mean()
                                 for x in ("L0-3", "L8-11"))
            out(f"            by |delta_clean| (median {med:.2f}): near early/late "
                f"{half['near'][0]:.1f}/{half['near'][1]:.1f}; far early/late "
                f"{half['far'][0]:.1f}/{half['far'][1]:.1f}")
    out()


# ---------------------------------------------------------------- D
def sec_D():
    out("== D. Hidden-state patching: items excluded by |delta_clean - delta_corr| <= 0.5 ==")
    for m in ENC:
        r = pd.read_csv(os.path.join(D, f"rev_residual_{m}.csv"))
        r = r.drop_duplicates("id")
        row = []
        for c in sorted(r.condition.unique()):
            s = r[r.condition == c]
            ex = ((s.delta_clean - s.delta_corr).abs() <= 0.5).sum()
            row.append(f"{c} {ex}/{len(s)} ({100 * ex / len(s):.1f}%)")
        out(f"  {m:9s} " + "; ".join(row))
    out()


# ---------------------------------------------------------------- E
def sec_E():
    out("== E. Lexicon shares and item counts ==")
    t6 = [i for i, v in IT.items() if i.startswith("T6-")]
    nm = sum(IT[i]["cue"] in NAME_ATTR for i in t6)
    out(f"  benchmark T6 items with a name attractor: {nm}/{len(t6)} ({100 * nm / len(t6):.1f}%)")
    for m in ENC:
        p = load_patching(m)
        ids = p[(p.condition == "T6") & (p.cue == "attractor")].id.unique()
        nm = sum(IT[i]["cue"] in NAME_ATTR for i in ids)
        out(f"    {m:9s} T6 items {len(ids)}, name attractor {nm} ({100 * nm / len(ids):.1f}%)")
    b = pd.read_csv(os.path.join(D, "behavioral_results.csv"))
    b = pd.concat([b[b.model != "marbert"], pd.read_csv(os.path.join(D, "rev_marbert_behavioral.csv"))])
    out("  behavioral n per cell (m/f):")
    for m in ENC:
        s = b[b.model == m]
        out(f"    {m:9s} " + " ".join(
            f"{c} {len(s[(s.condition == c) & (s.gold == 'm')])}/{len(s[(s.condition == c) & (s.gold == 'f')])}"
            for c in sorted(s.condition.unique())))
    out()


# ---------------------------------------------------------------- F
def sec_F():
    out("== F. Robustness in fixed windows ==")
    for m in MSA:
        p = load_patching(m)
        p = p.assign(amb=p.id.map(lambda i: IT[i]["subj"] in HOMOGRAPHS))
        s1 = p[(p.condition == "T1") & (p.cue == "subject") & p.correct_clean]
        f = lambda d: (win(d, LATE) - win(d, EARLY)).mean()
        pk = lambda d: d.groupby("layer").score.mean()
        a6 = p[(p.condition == "T6") & (p.cue == "attractor")]
        sa = p[p.cue == "subject"]
        wo = lambda d: win(d[d.condition == "T4"], LATE).mean() - win(d[d.condition == "T1"], LATE).mean()
        out(f"  homographs {m:9s}: T1 peak L{pk(s1).idxmax()} {pk(s1).max():+.2f} -> "
            f"L{pk(s1[~s1.amb]).idxmax()} {pk(s1[~s1.amb]).max():+.2f}; late-minus-early "
            f"{f(s1):+.2f} -> {f(s1[~s1.amb]):+.2f}; T6 attr late {win(a6, LATE).mean():+.3f} -> "
            f"{win(a6[~a6.amb], LATE).mean():+.3f}; VSO-SVO late (all) {wo(sa):+.3f} -> {wo(sa[~sa.amb]):+.3f}")
    for m in ENC:
        p = load_patching(m)
        a6 = p[(p.condition == "T6") & (p.cue == "attractor")]
        v = win(a6, LATE)
        isn = np.asarray(v.index.map(lambda i: IT[i]["cue"] in NAME_ATTR), bool)
        out(f"  attractor type {m:9s}: late name {v[isn].mean():+.3f} (n={isn.sum()}) vs common "
            f"{v[~isn].mean():+.3f} (n={(~isn).sum()})")
    z = pd.read_csv(os.path.join(D, "zeroing_results.csv"))
    for m in MSA:
        s = z[(z.model == m) & z.layer.between(*LATE)]
        t1 = s[s.condition == "T1"].groupby(["id", "role"]).score_norm.mean().unstack()
        t2 = s[s.condition == "T2"].groupby(["id", "role"]).score_norm.mean().unstack()
        d2 = (t2.subject - t2.adjective).dropna()
        out(f"  zeroing {m:9s}: T1 late others {fmt(item_ci(t1.others.dropna()), 2)}, subject "
            f"{fmt(item_ci(t1.subject.dropna()), 2)}; T2 late subject-minus-adjective {fmt(item_ci(d2), 2)}")
    out()


# ---------------------------------------------------------------- G
def sec_G():
    import statsmodels.formula.api as smf
    out("== G. Treebank (fixed windows) ==")
    fam = {}
    for c in ("T1", "T4", "T6"):
        for line in open(os.path.join(D, f"natural_families_{c}_arbert.jsonl"), encoding="utf-8"):
            it = json.loads(line)
            fam[it["id"]] = it
    pat = pd.read_csv(os.path.join(D, "natural_patching.csv"))
    beh = pd.read_csv(os.path.join(D, "natural_behavioral.csv"))
    dr = pd.read_csv(os.path.join(D, "natural_doseresponse.csv"))
    tmpl = {m: win(load_patching(m).query("condition == 'T6' and cue == 'attractor'"), LATE).mean()
            for m in MSA}
    adj = {i for i, it in fam.items() if i.startswith("NT4")
           and it.get("subject_word") == it.get("mask_verb_word", -9) + 1}
    n4 = sum(i.startswith("NT4") for i in fam)
    out(f"  VSO items with the subject directly after the verb: {len(adj)}/{n4} ({100 * len(adj) / n4:.1f}%)")
    for m in MSA:
        out(f"[{m}]")
        b = beh[beh.model == m]
        b4 = b[b.condition == "T4"]
        out(f"  accuracy SVO {100 * b[b.condition == 'T1'].correct.mean():.1f} "
            f"VSO {100 * b4.correct.mean():.1f} (n={len(b4)}) VSO adjacent "
            f"{100 * b4[b4.id.isin(adj)].correct.mean():.1f} (n={b4.id.isin(adj).sum()}) "
            f"attractor {100 * b[b.condition == 'T6'].correct.mean():.1f}")
        pm = pat[pat.model == m]
        for c in ("T1", "T4"):
            s = pm[(pm.condition == c) & (pm.cue == "subject")]
            v = win(s, LATE)
            line = f"  subject {c} late (all, n={len(v)}): {fmt(item_ci(v, RNG_N), 2)}"
            if c == "T4":
                va = v[v.index.isin(adj)]
                line += f"; adjacent n={len(va)} {fmt(item_ci(va, RNG_N), 2)}"
                line += f"; layer-9 all {s[s.layer == 9].score.mean():+.2f}"
            # without subjects that are inanimate in context (fresh seed-49 generator,
            # so the intervals above are unchanged)
            va = v[[fam[i]["lex"]["subj_masc"] not in INANIMATE_NAT for i in v.index]]
            line += (f"; animate only n={len(va)} "
                     f"{fmt(item_ci(va, np.random.default_rng(49)), 2)}")
            out(line)
        a = win(pm[(pm.condition == "T6") & (pm.cue == "attractor")], LATE)
        out(f"  attractor late (n={len(a)}): {fmt(item_ci(a, RNG_N))}; template {tmpl[m]:+.3f}; "
            f"ratio {a.mean() / tmpl[m]:.2f}")
        d = dr[dr.model == m].set_index("id")
        d = d.assign(late=a.reindex(d.index), logw=np.log(d.n_words))
        d["tert"] = pd.cut(d.n_words, [0, 37, 56, 10_000], labels=["short", "mid", "long"])
        out("  terciles (<=37, 38-56, >=57 words): " + "; ".join(
            f"{t} {d[d.tert == t].late.mean():+.3f} (n={(d.tert == t).sum()})" for t in ("short", "mid", "long")))
        f = smf.ols("late ~ logw + fem_count", d).fit()
        fc = smf.ols("late ~ logw + fem_count", d).fit(
            cov_type="cluster", cov_kwds={"groups": d.verb_lemma.astype("category").cat.codes})
        out(f"  OLS late ~ log(words) + fem: b1 {f.params.logw:+.4f} p={f.pvalues.logw:.2e} "
            f"(verb-cluster p={fc.pvalues.logw:.2e}); b2 {f.params.fem_count:+.4f} "
            f"p={f.pvalues.fem_count:.3f} (verb-cluster p={fc.pvalues.fem_count:.3f}); "
            f"r(log words, fem) {np.corrcoef(d.logw, d.fem_count)[0, 1]:.2f}")
    rp = os.path.join(D, "natural_repair.csv")
    rep = pd.read_csv(rp)
    if "delta_late" in rep:
        out("  treebank repair, zeroing the attractor (error items; item bootstrap B=2000, seed 49):")
        for m in MSA:
            w = rep[(rep.model == m) & ~rep.clean_correct]
            r = w[(w.model == m)]
            res = []
            for col, lab in (("delta_ctrl", "0-3"), ("delta_late", "8-11"), ("delta_neut", "5-11")):
                x = (r[col] > 0).to_numpy(float)
                bs = x[RNG_N.integers(0, len(x), size=(2000, len(x)))].mean(1)
                res.append(f"{lab} {100 * x.mean():.1f} [{np.percentile(bs, 2.5) * 100:.1f},"
                           f"{np.percentile(bs, 97.5) * 100:.1f}]")
            dd = ((r.delta_late > 0).astype(float) - (r.delta_ctrl > 0).astype(float)).to_numpy()
            bs = dd[RNG_N.integers(0, len(dd), size=(2000, len(dd)))].mean(1)
            out(f"    {m:9s} n_err={len(r)}: " + "; ".join(res)
                + f"; 8-11 minus 0-3 {100 * dd.mean():+.1f} [{100 * np.percentile(bs, 2.5):+.1f},"
                  f"{100 * np.percentile(bs, 97.5):+.1f}] pp")
    out()


def sec_H():
    out("== H. Morpheme probe (CAMeLBERT, layer 9; item bootstrap seed 48) ==")
    m = pd.read_csv(os.path.join(D, "morpheme_results.csv"))
    d = m[m.layer == 9].pivot_table(index="id", columns="target", values="collapse")
    out(f"  suffix-minus-stem (n={len(d)}): {fmt(item_ci(d.zero_ta - d.zero_stem))}")
    out()


def main():
    sec_A()
    sec_B()
    sec_C()
    sec_D()
    sec_E()
    sec_F()
    sec_G()
    sec_H()
    with open(os.path.join(D, "fixed_windows_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
