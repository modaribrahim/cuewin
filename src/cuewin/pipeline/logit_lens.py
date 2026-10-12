#!/usr/bin/env python3
"""
Logit-lens layer trajectory (corroborates layer-9).

For each T1 item and each encoder, capture the hidden state at the [MASK] position
at every layer, project it through the model's OWN MLM head
(cls.predictions: dense -> gelu -> LayerNorm -> decoder), and record
  Δ(ℓ) = logit(gold) − logit(foil)  at layer ℓ.

Caveat (stated in paper): the MLM head is trained only on final-layer states, so
intermediate readouts are APPROXIMATE — this corroborates the causal layer-9
patching result, it is not standalone evidence.

Expected shape: Δ near 0 through early layers, sharp rise around layer 9.

Additive only. Reads data/families.jsonl. Writes data/logit_lens.csv
(model, id, gold, layer, delta).
"""

import json
import os

import pandas as pd
import torch

from cuewin.config import data_dir, model_path
from cuewin.constants import ENCODERS
from cuewin.hooks import HiddenStateCache
from cuewin.models import device, load_encoder

FAM_PATH = os.path.join(data_dir(), "families_arbert.jsonl")
OUT_PATH = os.path.join(data_dir(), "logit_lens.csv")

MODELS = {k: model_path(k) for k in ENCODERS}
DEVICE = device()
N_LAYERS = 12
BATCH = 32
MASK = "[MASK]"


@torch.no_grad()
def run_model(mname, items):
    tk, model = load_encoder(mname)
    hcache = HiddenStateCache(model)
    mlm_head = model.cls                       # BertOnlyMLMHead: full MLM projection

    rows = []
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        texts = [it["clean"].replace(MASK, tk.mask_token) for it in batch]
        enc = tk(texts, return_tensors="pt", padding=True).to(DEVICE)
        with hcache.capture():
            model(**enc)
        for layer in range(N_LAYERS):
            hs = hcache.hidden[layer]          # (B, L, H)
            for b, it in enumerate(batch):
                mp = it["models"][mname]["mask_pos"]
                gold = it["models"][mname]["target_ids"]["gold"]
                foil = it["models"][mname]["target_ids"]["foil"]
                logits = mlm_head(hs[b, mp].unsqueeze(0)).squeeze(0)   # (V,)
                rows.append({"model": mname, "id": it["id"], "gold": it["gold"],
                             "layer": layer,
                             "delta": (logits[gold] - logits[foil]).item()})
        if (i // BATCH) % 10 == 0:
            print(f"  {mname}: {min(i + BATCH, len(items))}/{len(items)}")
    del model
    return rows


def main():
    global MODELS
    only = set(x.strip() for x in os.environ.get("MI_ONLY", "").split(",") if x.strip())
    if only:
        MODELS = {k: v for k, v in MODELS.items() if k in only}
    items = [json.loads(l) for l in open(FAM_PATH, encoding="utf-8")
             if json.loads(l)["condition"] == "T1"]
    print(f"T1 items: {len(items)}   device: {DEVICE}")

    all_rows = []
    for mname, mpath in MODELS.items():
        print(f"\n{mname} ...")
        mitems = [it for it in items if mname in it.get("models", {})]
        all_rows.extend(run_model(mname, mitems))

    df = pd.DataFrame(all_rows)
    only = set(x.strip() for x in os.environ.get("MI_ONLY", "").split(",") if x.strip())
    if only and os.path.exists(OUT_PATH):
        old = pd.read_csv(OUT_PATH)
        keep = old[~old.model.isin(list(only))]
        df = pd.concat([keep, df], ignore_index=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nsaved {OUT_PATH} ({len(df)} rows)  [{','.join(sorted(df.model.unique().tolist()))}]")

    pd.set_option("display.width", 120)
    pv = df.pivot_table(index="layer", columns="model", values="delta", aggfunc="mean")
    print("\nmean logit-lens Δ = logit(gold) − logit(foil) by layer:")
    print(pv.round(3))


if __name__ == "__main__":
    main()