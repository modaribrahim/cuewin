#!/usr/bin/env python3
"""
Per-cue value patching on naturalistic items (P0 Phase 2).

Mirror of the template value patching, but reads the natural families. Cues:
  T1n / T4n (condition T1/T4) -> subject cue (only the ~79 items that have one)
  T6n (condition T6)          -> attractor cue (all 404)

Output: data/natural_patching.csv
"""

import json
import os
from collections import defaultdict

import pandas as pd
import torch

from cuewin.config import data_dir, model_path
from cuewin.hooks import ValueCache, ValuePatcher
from cuewin.models import device, load_encoder

BASE = data_dir()
FAM_DIR = BASE
FAM_NAMES = ("natural_families_T1_arbert.jsonl", "natural_families_T4_arbert.jsonl",
             "natural_families_T6_arbert.jsonl")
OUT_PATH = os.path.join(BASE, "natural_patching.csv")
FAMS = [os.path.join(FAM_DIR, f) for f in FAM_NAMES]

MODELS = {m: model_path(m) for m in ("arabert", "camelbert", "arbert")}
BATCH = 8
N_LAYERS = 12
DEVICE = device()


def batches(items):
    groups = defaultdict(list)
    for it in items:
        ms = tuple(sorted(k for k in MODELS if k in it.get("models", {})))
        sig = (it["condition"], tuple(it["models"][m]["n_tokens"] for m in ms))
        groups[sig].append(it)
    for sig, group in groups.items():
        for i in range(0, len(group), BATCH):
            yield group[i:i + BATCH]


@torch.no_grad()
def run_model(mname, mpath, items):
    tk, model = load_encoder(mname)
    vcache, patcher = ValueCache(model), ValuePatcher(model)

    rows = []
    n_done = 0
    for batch in batches(items):
        enc_clean = tk([it["clean"].replace("[MASK]", tk.mask_token) for it in batch],
                       return_tensors="pt", padding=True).to(DEVICE)
        logits = model(**enc_clean).logits
        probs = torch.softmax(logits, dim=-1)
        meta = []
        for b, it in enumerate(batch):
            mm = it["models"][mname]
            mp = mm["mask_pos"]
            gid, fid = mm["target_ids"]["gold"], mm["target_ids"]["foil"]
            d_clean = (logits[b, mp, gid] - logits[b, mp, fid]).item()
            meta.append((mp, gid, fid, d_clean, probs[b, mp, gid].item()))

        for cue in list(batch[0]["cues"].keys()):
            enc_corr = tk([it["corrupted"][cue].replace("[MASK]", tk.mask_token)
                           for it in batch],
                          return_tensors="pt", padding=True).to(DEVICE)
            assert enc_corr["input_ids"].shape == enc_clean["input_ids"].shape
            with vcache.capture():
                model(**enc_corr)
            cached = dict(vcache.values)
            rows_spans = [(b, *it["models"][mname]["spans"][cue])
                          for b, it in enumerate(batch)]
            for layer in range(N_LAYERS):
                with patcher.patch(layer, cached, rows_spans):
                    plogits = model(**enc_clean).logits
                pprobs = torch.softmax(plogits, dim=-1)
                for b, it in enumerate(batch):
                    mp, gid, fid, d_clean, p_clean = meta[b]
                    d_patch = (plogits[b, mp, gid] - plogits[b, mp, fid]).item()
                    rows.append({
                        "id": it["id"], "model": mname, "condition": it["condition"],
                        "gold": it["gold"], "subj_kind": "natural",
                        "cue": cue, "layer": layer,
                        "delta_clean": d_clean, "delta_patched": d_patch,
                        "score": d_clean - d_patch,
                        "p_gold_clean": p_clean, "p_gold_patched": pprobs[b, mp, gid].item(),
                        "p_drop": p_clean - pprobs[b, mp, gid].item(),
                        "correct_clean": d_clean > 0,
                    })
        n_done += len(batch)
        if n_done % 640 < BATCH:
            print(f"  {mname}: {n_done}/{len(items)}")
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
    items = [it for it in items if it["cues"]]
    print(f"natural items with a cue (patching): {len(items)}   device: {DEVICE}")

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

    pd.set_option("display.width", 160)
    for cond in ("T1", "T4", "T6"):
        sub = df[df.condition == cond]
        if not len(sub):
            continue
        pv = sub.pivot_table(index="layer", columns=["model", "cue"],
                             values="score", aggfunc="mean")
        print(f"\n=== {cond} mean patching score by layer ===")
        print(pv.round(3))


if __name__ == "__main__":
    main()