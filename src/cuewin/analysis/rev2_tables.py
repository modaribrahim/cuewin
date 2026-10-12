"""Revision-2 tables and figure (read-only on CSVs; CPU).

Writes LaTeX table bodies to data/rev2_tables.tex, a plain-text log of every
value to data/rev2_tables.txt, and figures/fig_cueread.png.

Definitions (error items: delta_clean <= 0):
  repair %     share of error items whose intervened margin is positive
  gap closed   mean(delta_int - delta_clean) / mean(delta_corr - delta_clean),
               delta_corr = margin with the cue flipped in the input
  windows      prefix P0-k = value patch in layers 0..k; suffix Sk-11 = layers
               k..11; hidden state Hl = cue span hidden state at the output of
               layer l (l = -1: embedding output)
Bootstrap: crossed over subjects, verb frames and cue items (seed 51);
treebank item bootstrap (seed 52); B = 10,000.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cuewin.analysis import fixed_windows as fw
from cuewin.analysis.fixed_windows import EARLY, ENC, LATE, crossed, item_ci, win
from cuewin.config import data_dir, figures_dir
from cuewin.constants import MODEL_LABEL

D = data_dir()
RNG_T = np.random.default_rng(51)
RNG_NT = np.random.default_rng(52)
RNG_S = np.random.default_rng(53)    # sensitivity checks
CUE_OF = {"T3": "adjective", "T6": "attractor", "T7": "attractor"}
FAC = ("subj", "frame", "cue")
tex, log = [], []


def say(s=""):
    print(s)
    log.append(s)


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


def add_items(path):
    for line in open(path, encoding="utf-8"):
        it = json.loads(line)
        lx = it["lex"]
        if fw.homograph_item(it):
            fw.EXCL.add(it["id"])
        fw.IT[it["id"]] = {"subj": lx["subj_masc"], "frame": lx["verb_masc"],
                           "cue": lx.get("attr_masc") or lx.get("adj_masc"),
                           "kind": lx["subj_kind"],
                           "cue_kind": lx.get("attr_kind") or lx.get("adj_kind")}


def gap(d, col="delta_int"):
    den = (d.delta_corr - d.delta_clean).mean()
    return (d[col] - d.delta_clean).mean() / den


def pct(x):
    return 100 * np.mean(x)


def ci_txt(t, d=1, sign=True):
    f = f"{{:+.{d}f}}" if sign else f"{{:.{d}f}}"
    return f"${f.format(t[0])}$ [${f.format(t[1])}$, ${f.format(t[2])}$]"


def paired_diff(d, a, b):
    """Per-item repair difference (pp) between windows a and b."""
    x = d[d.window == a].set_index("id").delta_int > 0
    y = d[d.window == b].set_index("id").delta_int > 0
    return 100 * (x.astype(float) - y.loc[x.index].astype(float))


# ---------------------------------------------------------------- cue read-out
def cueread_rows(set_name, conds):
    rows = []
    for m in ENC:
        c = csv(f"rev2_cumwin_{set_name}_{m}.csv")
        r = csv(f"rev2_cueresid_{set_name}_{m}.csv")
        for cond in conds:
            cw = c[(c.condition == cond) & (c.kind == "patch") & (c.delta_clean <= 0)]
            rr = r[(r.condition == cond) & (r.delta_clean <= 0)]
            n = cw.id.nunique()
            h = {l: gap(rr[rr.layer == l], "delta_patched") for l in (3, 7, 9)}
            rep = {k: pct(cw[cw.window == f"S{k}-11"].delta_int > 0) for k in (0, 4, 8)}
            g = {k: gap(cw[cw.window == f"S{k}-11"]) for k in (0, 4, 8)}
            d4 = crossed(paired_diff(cw, "S4-11", "S0-11"), FAC, RNG_T)
            d8 = crossed(paired_diff(cw, "S8-11", "S0-11"), FAC, RNG_T)
            say(f"cueread {set_name} {m} {cond} n={n} H3/H7/H9 gap {h[3]:.2f}/{h[7]:.2f}/{h[9]:.2f}; "
                f"S0/S4/S8 repair {rep[0]:.1f}/{rep[4]:.1f}/{rep[8]:.1f} gap {g[0]:.2f}/{g[4]:.2f}/{g[8]:.2f}; "
                f"S4-S0 {d4[0]:+.1f} [{d4[1]:+.1f},{d4[2]:+.1f}]; S8-S0 {d8[0]:+.1f} [{d8[1]:+.1f},{d8[2]:+.1f}]")
            rows.append(f"{MODEL_LABEL[m]} & {cond} & {n} & {h[3]:.2f} & {h[7]:.2f} & {h[9]:.2f} & "
                        f"{rep[0]:.1f} & {rep[4]:.1f} & {rep[8]:.1f} & {ci_txt(d4)} & {ci_txt(d8)}\\\\")
    return rows


# ---------------------------------------------------------------- replication
def rep_rows():
    rows = []
    for m in ENC:
        mb = csv("rev_marbert_behavioral.csv") if m == "marbert" else csv("behavioral_results.csv")
        mb = mb[(mb.model == m) & (mb.condition == "T1")]
        rows.append(f"{MODEL_LABEL[m]} & T1 (main) & {len(mb)} & {pct(mb.correct):.1f} & & & & & \\\\")
        say(f"rep {m} T1 main acc {pct(mb.correct):.1f} n={len(mb)}")
        b = csv(f"rev2_rep_behavioral_{m}.csv")
        p = csv(f"rev2_rep_patching_{m}.csv")
        w = csv(f"rev2_rep_windows_{m}.csv")
        t2 = b[b.condition == "T2"]
        rows.append(f" & T2 & {len(t2)} & {pct(t2.correct):.1f} & & & & & \\\\")
        say(f"rep {m} T2 n={len(t2)} acc {pct(t2.correct):.1f}; by gold "
            + "; ".join(f"{g} {pct(t2[t2.gold == g].correct):.1f}" for g in ("m", "f")))
        for cond, cue in CUE_OF.items():
            bb = b[b.condition == cond]
            a = p[(p.condition == cond) & (p.cue == cue)]
            late = crossed(win(a, LATE), FAC, RNG_T)
            d = w[(w.condition == cond) & (w.kind == "patch")]
            wrong = d[d.delta_clean <= 0]
            r = {wl: pct(wrong[wrong.window == wl].delta_int > 0) for wl in ("L0-3", "L4-7", "L8-11")}
            le = crossed(paired_diff(wrong, "L8-11", "L0-3"), FAC, RNG_T)
            right = d[(d.delta_clean > 0) & (d.window == "L8-11")]
            brk = pct(right.delta_int < 0)
            lv = len({fw.IT[i]["cue"] for i in bb.id})
            say(f"rep {m} {cond} n={len(bb)} cue items={lv} acc {pct(bb.correct):.1f}; late "
                f"{late[0]:+.3f} [{late[1]:+.3f},{late[2]:+.3f}]; repair {r['L0-3']:.1f}/{r['L4-7']:.1f}/"
                f"{r['L8-11']:.1f} (n_err={wrong.id.nunique()}); late-early {le[0]:+.1f} "
                f"[{le[1]:+.1f},{le[2]:+.1f}]; break {brk:.1f}")
            rows.append(f" & {cond} & {len(bb)} & {pct(bb.correct):.1f} & {ci_txt(late, 2)} & "
                        f"{r['L0-3']:.1f} & {r['L4-7']:.1f} & {r['L8-11']:.1f} & {ci_txt(le)} & {brk:.1f}\\\\")
        t3 = w[(w.condition == "T3") & (w.kind == "patch") & (w.delta_clean <= 0)].copy()
        t3 = t3[t3.id.map(lambda i: fw.IT[i]["cue_kind"] == "descriptive")]
        a = p[(p.condition == "T3") & (p.cue == "adjective")]
        a = a[a.id.map(lambda i: fw.IT[i]["cue_kind"] == "descriptive")]
        late = crossed(win(a, LATE), FAC, RNG_S)
        le = crossed(paired_diff(t3, "L8-11", "L0-3"), FAC, RNG_S)
        acc = pct(b[(b.condition == "T3") & b.id.map(lambda i: fw.IT[i]["cue_kind"] == "descriptive")].correct)
        say(f"rep {m} T3 descriptive adjectives only: n={a.id.nunique()} acc {acc:.1f}; late "
            f"{late[0]:+.3f} [{late[1]:+.3f},{late[2]:+.3f}]; repair L0-3/L8-11 "
            f"{pct(t3[t3.window == 'L0-3'].delta_int > 0):.1f}/{pct(t3[t3.window == 'L8-11'].delta_int > 0):.1f} "
            f"(n_err={t3.id.nunique()}); late-early {le[0]:+.1f} [{le[1]:+.1f},{le[2]:+.1f}]")
        rows.append("\\addlinespace")
    return rows[:-1]


# ---------------------------------------------------------------- treebank
def nat_rows():
    cols = {k: [] for k in ("n", "r03", "r47", "r811", "le", "z03", "z811", "gnat", "gtpl")}
    for m in ENC:
        w = csv(f"rev2_nat_windows_{m}.csv")
        wrong = w[w.delta_clean <= 0]
        pw = wrong[wrong.kind == "patch"]
        cols["n"].append(f"{pw.id.nunique()}")
        for key, wl in (("r03", "L0-3"), ("r47", "L4-7"), ("r811", "L8-11")):
            x = item_ci((pw[pw.window == wl].delta_int > 0).astype(float) * 100, RNG_NT)
            cols[key].append(f"{x[0]:.1f} [{x[1]:.1f}, {x[2]:.1f}]")
        le = item_ci(paired_diff(pw, "L8-11", "L0-3"), RNG_NT)
        cols["le"].append(ci_txt(le))
        zw = wrong[wrong.kind == "zero"]
        for key, wl in (("z03", "L0-3"), ("z811", "L8-11")):
            cols[key].append(f"{pct(zw[zw.window == wl].delta_int > 0):.1f}")
        allp = w[(w.kind == "patch") & (w.window == "L8-11")]
        gn = gap(allp)
        tw = csv(f"rev_windows_{m}.csv")
        tr = csv(f"rev2_cueresid_main_{m}.csv")
        corr = tr[(tr.layer == -1) & (tr.condition == "T6")].set_index("id").delta_corr
        tt = tw[(tw.condition == "T6") & (tw.kind == "patch") & (tw.window == "L8-11")].copy()
        tt["delta_corr"] = tt.id.map(corr)
        gt = gap(tt.dropna(subset=["delta_corr"]))
        cols["gnat"].append(f"{gn:.2f}")
        cols["gtpl"].append(f"{gt:.2f}")
        say(f"nat {m}: n_err {pw.id.nunique()}; patch {cols['r03'][-1]} / {cols['r47'][-1]} / "
            f"{cols['r811'][-1]}; late-early {le[0]:+.1f} [{le[1]:+.1f},{le[2]:+.1f}]; zero 0-3/8-11 "
            f"{cols['z03'][-1]}/{cols['z811'][-1]}; gap all items 8-11 treebank {gn:.2f} template {gt:.2f}")
    lab = {"n": "Attractor errors", "r03": "Patch, layers 0--3 (\\%)", "r47": "Patch, layers 4--7 (\\%)",
           "r811": "Patch, layers 8--11 (\\%)", "le": "\\quad 8--11 minus 0--3 (pp)",
           "z03": "Zeroing, layers 0--3 (\\%)", "z811": "Zeroing, layers 8--11 (\\%)",
           "gnat": "Gap closed, patch 8--11, treebank", "gtpl": "\\quad template (T6)"}
    return [f"{lab[k]} & " + " & ".join(v) + "\\\\" for k, v in cols.items()]


# ---------------------------------------------------------------- names
def name_rows():
    rows = []
    for m in ENC:
        cells = []
        w = csv(f"rev_windows_{m}.csv")
        d = w[(w.condition == "T6") & (w.kind == "patch") & (w.window == "L8-11")].copy()
        d["k"] = np.where(d.id.map(lambda i: fw.IT[i]["cue"] in fw.NAME_ATTR), "name", "noun")
        sets = [("main T6", d)]
        w2 = csv(f"rev2_rep_windows_{m}.csv")
        for cond in ("T6", "T7"):
            e = w2[(w2.condition == cond) & (w2.kind == "patch") & (w2.window == "L8-11")].copy()
            e["k"] = e.id.map(lambda i: fw.IT[i]["cue_kind"])
            sets.append((f"held-out {cond}", e))
        for tag, e in sets:
            for k in ("noun", "name"):
                g = e[e.k == k]
                err = g[g.delta_clean <= 0]
                cells.append(f"{pct(g.delta_clean > 0):.1f} / {pct(err.delta_int > 0):.1f}")
                say(f"names {m} {tag} {k}: acc {pct(g.delta_clean > 0):.1f} (n={len(g)}); repair "
                    f"{pct(err.delta_int > 0):.1f} (n_err={len(err)})")
        rows.append(f"{MODEL_LABEL[m]} & " + " & ".join(cells) + "\\\\")
    return rows


# ---------------------------------------------------------------- heads
def head_lines():
    rows = {"r6": [], "r7": [], "rk6": [], "rk7": [], "o6": [], "o7": []}
    for m in ENC:
        h = csv(f"rev_heads_{m}.csv")
        s = h[(h.condition == "T1") & (h.cue == "subject") & h.correct_clean]
        a = s.groupby(["layer", "head"]).score.mean()
        top = a.idxmax()
        top5 = set(a.sort_values(ascending=False).head(5).index)
        hr = csv(f"rev2_rep_heads_{m}.csv")
        for cond, k in (("T6", "6"), ("T7", "7")):
            b = hr[hr.condition == cond].groupby(["layer", "head"]).score.mean().reindex(a.index)
            r = np.corrcoef(a, b)[0, 1]
            rank = int((b < b[top]).sum()) + 1
            ov = len(top5 & set(b.sort_values().head(5).index))
            rows["r" + k].append(f"${r:+.2f}$")
            rows["rk" + k].append(f"{rank}")
            rows["o" + k].append(f"{ov}")
            say(f"heads {m} held-out {cond}: r(main T1 subject, attractor) {r:+.2f}; top subject head "
                f"L{top[0]}H{top[1]} rank {rank}/144 most negative; top-5 overlap {ov}/5")
    lab = {"r6": "Subject--attractor $r$, held-out T6", "r7": "\\quad held-out T7",
           "rk6": "Top subject head, rank among attractor heads, T6", "rk7": "\\quad T7",
           "o6": "Top-five overlap, held-out T6", "o7": "\\quad held-out T7"}
    return [f"{lab[k]} & " + " & ".join(v) + "\\\\" for k, v in rows.items()]


# ---------------------------------------------------------------- treebank by gender
def nat_gender():
    nb = csv("natural_behavioral.csv")
    for m in ENC:
        parts = []
        for cond in ("T1", "T4", "T6"):
            if m == "marbert" and cond == "T6":
                w = csv("rev2_nat_windows_marbert.csv")
                d = w[(w.window == "L0") & (w.kind == "patch")]
                d = d.assign(correct=d.delta_clean > 0)
            else:
                d = nb[(nb.model == m) & (nb.condition == cond)]
            if d.empty:
                continue
            parts.append(f"{cond} " + ", ".join(
                f"{g} {pct(d[d.gold == g].correct):.1f} (n={int((d.gold == g).sum())})" for g in ("m", "f")))
        say(f"treebank accuracy by gold {m}: " + "; ".join(parts))


# ---------------------------------------------------------------- noun-only subjects
def noun_subjects():
    d = pd.concat([csv("patching_results.csv"), csv("rev_marbert_patching.csv")])
    s = d[(d.cue == "subject") & d.correct_clean]
    for (m, c), g in s.groupby(["model", "condition"]):
        parts = []
        for k, gg in (("all", g), ("noun", g[g.subj_kind == "noun"])):
            per = gg.groupby("layer").score.mean()
            parts.append(f"{k} peak L{per.idxmax()} late-early "
                         f"{per.loc[8:11].mean() - per.loc[0:3].mean():+.2f} (n={gg.id.nunique()})")
        say(f"subject {m} {c}: " + "; ".join(parts))


# ---------------------------------------------------------------- logit lens
def fig_logitlens():
    ll = csv("logit_lens.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    for ax, m in zip(axes, [e for e in ENC if e in set(ll.model)]):
        for g, col, lab in ((None, "k", "all items"), ("m", "#1f77b4", "masculine gold"),
                            ("f", "#d62728", "feminine gold")):
            d = ll[(ll.model == m) & ((ll.gold == g) if g else True)]
            mu = d.groupby("layer").delta.mean()
            ax.plot(mu.index, mu.values, color=col, marker="o", ms=3, lw=2.0 if g is None else 1.1,
                    ls="-" if g is None else "--", label=lab)
            say(f"logit lens {m} gold {g} (n={d.id.nunique()}): "
                + " ".join(f"L{l} {mu[l]:+.2f}" for l in (3, 7, 9, 11)))
        ax.axhline(0, color="0.5", lw=0.6)
        ax.axvspan(7.5, 11.5, color="0.85", zorder=0)
        ax.set_title(MODEL_LABEL[m], fontsize=11)
        ax.set_xlabel("layer", fontsize=10)
        ax.set_xticks([0, 3, 7, 11])
    axes[0].set_ylabel("margin (logits)", fontsize=10)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(os.path.join(figures_dir(), "fig_logitlens.png"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- figure
def fig_cueread():
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.9), sharex=True, sharey=True)
    for j, m in enumerate(ENC):
        c = csv(f"rev2_cumwin_main_{m}.csv")
        r = csv(f"rev2_cueresid_main_{m}.csv")
        for i, cond in enumerate(("T6", "T3")):
            ax = axes[i, j]
            cw = c[(c.condition == cond) & (c.kind == "patch") & (c.delta_clean <= 0)]
            rr = r[(r.condition == cond) & (r.delta_clean <= 0)]
            ls = list(range(12))
            ax.plot([-1] + ls, [gap(rr[rr.layer == l], "delta_patched") for l in [-1] + ls],
                    color="k", lw=1.8, marker="o", ms=3, label="cue hidden state at output of layer $\\ell$")
            ax.plot(ls, [gap(cw[cw.window == f"S{k}-11"]) for k in ls], color="#9467bd", ls="--",
                    lw=1.8, marker="s", ms=3, label="cue values patched in layers $\\ell$–11")
            ax.plot(ls, [gap(cw[cw.window == f"P0-{k}"]) for k in ls], color="#ff7f0e", ls=":",
                    lw=1.8, marker="^", ms=3, label="cue values patched in layers 0–$\\ell$")
            ax.axvspan(7.5, 11.5, color="0.85", zorder=0)
            ax.axhline(1, color="0.5", lw=0.6)
            if i == 0:
                ax.set_title(MODEL_LABEL[m], fontsize=12)
            if j == 0:
                ax.set_ylabel(("T6 attractor" if cond == "T6" else "T3 conflicting adj.")
                              + "\ngap closed", fontsize=11)
            if i == 1:
                ax.set_xlabel("layer $\\ell$", fontsize=11)
                ax.set_xticks([-1, 3, 7, 11], ["emb", "3", "7", "11"])
            ax.tick_params(labelsize=10)
            ax.set_ylim(-0.05, 1.12)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(figures_dir(), "fig_cueread.png"), dpi=300)
    plt.close(fig)


def main():
    add_items(os.path.join(D, "families_rep.jsonl"))
    add_items(os.path.join(D, "natural_families_T6_arbert.jsonl"))
    blocks = [("cueread_main", cueread_rows("main", ("T6", "T3"))),
              ("cueread_rep", cueread_rows("rep", ("T6", "T7", "T3"))),
              ("rep", rep_rows()), ("natrepair", nat_rows()), ("names", name_rows()),
              ("heads_rep", head_lines())]
    for name, rows in blocks:
        tex.append(f"% ---- {name}")
        tex.extend(rows)
    with open(os.path.join(D, "rev2_tables.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    say(f"excluded held-out homograph items: {len(fw.EXCL)}")
    nat_gender()
    noun_subjects()
    with open(os.path.join(D, "rev2_tables.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    fig_cueread()
    fig_logitlens()
    with open(os.path.join(D, "rev2_tables.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
