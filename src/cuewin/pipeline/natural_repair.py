#!/usr/bin/env python3
"""
24_natural_repair.py — P0 Phase 2: interference repair on naturalistic T6 items.

Mirror of 12_intervention.py, but reads natural_families_T6.jsonl (attractor cue).
Neutralize (zero) the attractor span at the located window (L5-11) vs early control
(L0-3) and measure repair / control / breakage rates.

Output: data/natural_repair.csv , data/natural_repair_summary.csv
"""

import json
import os
from contextlib import ExitStack

import pandas as pd
import torch

from cuewin.config import data_dir, model_path
from cuewin.hooks import ValueZeroer
from cuewin.models import device, load_encoder

BASE = data_dir()
FAM_PATH = os.path.join(BASE, "natural_families_T6_arbert.jsonl")
OUT_RES = os.path.join(BASE, "natural_repair.csv")
OUT_SUM = os.path.join(BASE, "natural_repair_summary.csv")

MODELS = {m: model_path(m) for m in ("arabert", "camelbert", "arbert")}
DEVICE = device()
BATCH = 8
MASK = "[MASK]"
LOCATED = range(5, 12)
EARLY = range(0, 4)


@torch.no_grad()
def run_model(mname, mpath, items):
    tk, model = load_encoder(mname)
    zeroer = ValueZeroer(model)

    def deltas(enc, mask_pos, gold, foil, rows_spans=None, layers=None):
        with ExitStack() as stack:
            if rows_spans is not None:
                for L in layers:
                    stack.enter_context(zeroer.zero(L, rows_spans))
            logits = model(**enc).logits
        return [(logits[b, mask_pos[b], gold[b]] - logits[b, mask_pos[b], foil[b]]).item()
                for b in range(len(mask_pos))]

    rows = []
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        texts = [it["clean"].replace(MASK, tk.mask_token) for it in batch]
        enc = tk(texts, return_tensors="pt", padding=True).to(DEVICE)
        mask_pos = [it["models"][mname]["mask_pos"] for it in batch]
        gold = [it["models"][mname]["target_ids"]["gold"] for it in batch]
        foil = [it["models"][mname]["target_ids"]["foil"] for it in batch]
        rows_spans = []
        for b, it in enumerate(batch):
            s, e = it["models"][mname]["spans"]["attractor"]
            rows_spans.append((b, s, e))
        d_clean = deltas(enc, mask_pos, gold, foil)
        d_neut = deltas(enc, mask_pos, gold, foil, rows_spans, LOCATED)
        d_ctrl = deltas(enc, mask_pos, gold, foil, rows_spans, EARLY)
        for b, it in enumerate(batch):
            rows.append({"model": mname, "condition": it["condition"], "id": it["id"],
                         "gold": it["gold"], "delta_clean": d_clean[b],
                         "delta_neut": d_neut[b], "delta_ctrl": d_ctrl[b],
                         "clean_correct": d_clean[b] > 0})
        if (i // BATCH) % 10 == 0:
            print(f"  {mname}: {min(i + BATCH, len(items))}/{len(items)}")
    del model
    return rows


def main():
    items = [json.loads(l) for l in open(FAM_PATH, encoding="utf-8")]
    print(f"natural T6 items: {len(items)}   device: {DEVICE}")

    rows = []
    for mname, mpath in MODELS.items():
        print(f"\n{mname} ...")
        mitems = [it for it in items if mname in it.get("models", {})]
        rows.extend(run_model(mname, mpath, mitems))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_RES, index=False)
    print(f"\nsaved {OUT_RES} ({len(df)} rows)")

    summ = []
    for mname in MODELS:
        d = df[df.model == mname]
        wrong, right = d[~d.clean_correct], d[d.clean_correct]
        summ.append({
            "model": mname, "condition": "T6", "cue": "attractor",
            "n_wrong": len(wrong), "n_correct": len(right),
            "repair_rate_L5-11": round((wrong.delta_neut > 0).mean(), 3) if len(wrong) else None,
            "control_rate_L0-3": round((wrong.delta_ctrl > 0).mean(), 3) if len(wrong) else None,
            "mean_dd_neut": round((wrong.delta_neut - wrong.delta_clean).mean(), 3) if len(wrong) else None,
            "breakage_rate": round((right.delta_neut < 0).mean(), 3) if len(right) else None,
        })
    sdf = pd.DataFrame(summ)
    sdf.to_csv(OUT_SUM, index=False)
    pd.set_option("display.width", 160)
    print("\n=== natural repair summary ===")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()