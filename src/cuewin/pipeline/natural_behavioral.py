#!/usr/bin/env python3
"""
Behavioral gate on naturalistic items (P0 Phase 2).

Mirror of the template behavioral stage, but reads the natural families (T1/T4/T6)
built by the natural-families stage. Same metric: forced-choice gold-vs-foil logit
diff at [MASK].

Output: data/natural_behavioral.csv
"""

import json
import os

import pandas as pd
import torch

from cuewin.config import data_dir, model_path
from cuewin.models import device, load_encoder

BASE = data_dir()
FAM_DIR = BASE
FAM_NAMES = ("natural_families_T1_arbert.jsonl", "natural_families_T4_arbert.jsonl",
             "natural_families_T6_arbert.jsonl")
OUT_PATH = os.path.join(BASE, "natural_behavioral.csv")
FAMS = [os.path.join(FAM_DIR, f) for f in FAM_NAMES]

MODELS = {m: model_path(m) for m in ("arabert", "camelbert", "arbert")}
BATCH = 8
DEVICE = device()


@torch.no_grad()
def run_model(mname, mpath, items):
    tk, model = load_encoder(mname)
    rows = []
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        texts = [it["clean"].replace("[MASK]", tk.mask_token) for it in batch]
        enc = tk(texts, return_tensors="pt", padding=True).to(DEVICE)
        logits = model(**enc).logits
        for b, it in enumerate(batch):
            mp = it["models"][mname]["mask_pos"]
            gid = it["models"][mname]["target_ids"]["gold"]
            fid = it["models"][mname]["target_ids"]["foil"]
            lg, lf = logits[b, mp, gid].item(), logits[b, mp, fid].item()
            row = torch.softmax(logits[b, mp].float(), dim=-1)
            rows.append({
                "id": it["id"], "model": mname, "condition": it["condition"],
                "gold": it["gold"], "subj_kind": "natural",
                "logit_gold": lg, "logit_foil": lf, "logit_diff": lg - lf,
                "correct": lg > lf,
                "p_gold": row[gid].item(), "p_foil": row[fid].item(),
            })
        if (i // BATCH) % 20 == 0:
            print(f"  {mname}: {i + len(batch)}/{len(items)}")
    del model
    return rows


def main():
    global MODELS
    only = set(x.strip() for x in os.environ.get("MI_ONLY", "").split(",") if x.strip())
    if only:
        MODELS = {k: v for k, v in MODELS.items() if k in only}
    items = []
    for p in FAMS:
        items.extend(json.loads(l) for l in open(p, encoding="utf-8"))
    print(f"natural items: {len(items)}   device: {DEVICE}")

    rows = []
    for mname, mpath in MODELS.items():
        print(f"\n{mname} ...")
        mitems = [it for it in items if mname in it.get("models", {})]
        rows.extend(run_model(mname, mpath, mitems))
    df = pd.DataFrame(rows)
    only = set(x.strip() for x in os.environ.get("MI_ONLY", "").split(",") if x.strip())
    if only and os.path.exists(OUT_PATH):
        old = pd.read_csv(OUT_PATH)
        keep = old[~old.model.isin(list(only))]
        df = pd.concat([keep, df], ignore_index=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nsaved: {OUT_PATH} ({len(df)} rows)  [{','.join(sorted(df.model.unique().tolist()))}]")

    pd.set_option("display.width", 140)
    acc = df.pivot_table(index="condition", columns=["model", "gold"],
                         values="correct", aggfunc="mean")
    print("\n=== NATURAL ACCURACY (gold > foil) ===")
    print((acc * 100).round(1))
    ld = df.pivot_table(index="condition", columns=["model"],
                        values="logit_diff", aggfunc="mean")
    print("\n=== mean logit diff (gold - foil) ===")
    print(ld.round(2))


if __name__ == "__main__":
    main()