"""Attention check on the strongest subject head (T6 attractor items).

For each encoder, the head with the largest mean T1 subject score
(rev_heads_<model>.csv, correct items) is taken as the top head. On every T6
item we read that head's attention weights from the [MASK] query to the
subject span and to the attractor span (summed over each span's tokens) on the
clean sentence, and compare items the model gets wrong with items it gets
right. Writes data/attention_check.csv and data/attention_check.txt.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from cuewin.config import data_dir, model_path
from cuewin.models import device
from cuewin.pipeline.revision import ENCODERS4, _batches, _enc, _margin, _meta, load_items

D = data_dir()
RNG = np.random.default_rng(45)
B = 10_000


def top_head(mname):
    h = pd.read_csv(os.path.join(D, f"rev_heads_{mname}.csv"))
    h = h[(h.condition == "T1") & h.correct_clean]
    s = h.groupby(["layer", "head"]).score.mean()
    return s.idxmax()


@torch.no_grad()
def run(mname, items):
    L, H = top_head(mname)
    path = model_path(mname)
    tk = AutoTokenizer.from_pretrained(path)
    model = AutoModelForMaskedLM.from_pretrained(path, attn_implementation="eager").to(device()).eval()
    rows = []
    for batch in _batches(items):
        enc = _enc(tk, [it["clean"] for it in batch])
        o = model(**enc, output_attentions=True)
        att = o.attentions[L][:, H]                     # batch x query x key
        margins = _margin(o.logits, _meta(batch, mname))
        for b, it in enumerate(batch):
            sp = it["models"][mname]["spans"]
            mp = it["models"][mname]["mask_pos"]
            a = att[b, mp]
            rows.append(dict(model=mname, id=it["id"], gold=it["gold"], layer=L, head=H,
                             subj=it["lex"]["subj_masc"],
                             att_subject=a[sp["subject"][0]:sp["subject"][1]].sum().item(),
                             att_attractor=a[sp["attractor"][0]:sp["attractor"][1]].sum().item(),
                             delta_clean=margins[b]))
    del model
    torch.cuda.empty_cache()
    return rows


def diff_ci(x, y):
    """Difference of means (x - y) with an item bootstrap CI."""
    x, y = np.asarray(x), np.asarray(y)
    bx = x[RNG.integers(0, len(x), (B, len(x)))].mean(1)
    by = y[RNG.integers(0, len(y), (B, len(y)))].mean(1)
    return x.mean() - y.mean(), *np.percentile(bx - by, [2.5, 97.5])


def main():
    items = [it for it in load_items() if it["condition"] == "T6"]
    rows = []
    for m in ENCODERS4:
        rows += run(m, [it for it in items if m in it["models"]])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(D, "attention_check.csv"), index=False)
    lines = ["== Top subject head: attention from [MASK], T6 clean items (error vs correct) =="]
    for m, d in df.groupby("model", sort=False):
        err, ok = d[d.delta_clean <= 0], d[d.delta_clean > 0]
        lines.append(f"{m} L{d.layer.iat[0]}H{d['head'].iat[0]}: n_error={len(err)} n_correct={len(ok)}")
        for col in ("att_attractor", "att_subject"):
            dlt, lo, hi = diff_ci(err[col], ok[col])
            lines.append(f"  {col:14s} error {err[col].mean():.3f}  correct {ok[col].mean():.3f}  "
                         f"diff {dlt:+.3f} [{lo:+.3f},{hi:+.3f}]")
        r = np.corrcoef(d.att_attractor - d.att_subject, d.delta_clean)[0, 1]
        lines.append(f"  corr(att_attractor - att_subject, delta_clean) r={r:+.3f}")
    print("\n".join(lines))
    with open(os.path.join(D, "attention_check.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
