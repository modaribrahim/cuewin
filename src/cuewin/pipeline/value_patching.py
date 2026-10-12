#!/usr/bin/env python3
"""
Per-cue value patching (Experiment 2, core).

For each model, item, cue, layer:
  1. clean run            -> Δ_clean   = logit(gold) − logit(foil) at [MASK]
  2. corrupted run (cue j flipped) -> cache value vectors of all layers
  3. patched run: clean input, but at layer L the value vectors of cue j's
     span are overwritten with the corrupted ones
                          -> Δ_patched(j, L)
  score(j, L) = Δ_clean − Δ_patched(j, L)
     > 0  : the cue's value at this layer causally supports the gold gender
     ≈ 0  : no causal contribution at this layer
     < 0  : patching helps gold (expected for T3/T6, where the flipped
            version of the incongruent cue is the CONGRUENT one)

Also records the probability-drop variant (p_gold drop) for comparability
with Amirzadeh et al. 2024.

Batching: items grouped by (condition, n_tokens) so each batch shares shape;
per-row spans are applied inside the hooks. Forward passes per batch:
  1 clean + per cue (1 corrupted + 12 patched) — cheap on GPU.

Output: data/patching_results.csv (long format):
  id, model, condition, gold, subj_kind, cue, layer,
  delta_clean, delta_patched, score, p_gold_clean, p_gold_patched, p_drop,
  correct_clean (behavioral filter flag)
"""

import json
import os

import pandas as pd
import torch

from cuewin.config import data_dir, model_path
from cuewin.families import batches
from cuewin.hooks import ValueCache, ValuePatcher
from cuewin.models import device, load_encoder

FAM_PATH = os.path.join(data_dir(), "families_arbert.jsonl")
OUT_PATH = os.path.join(data_dir(), "patching_results.csv")

MODELS = {
    "arabert": model_path("arabert"),
    "camelbert": model_path("camelbert"),
    "arbert": model_path("arbert"),
}

BATCH = 32
N_LAYERS = 12
DEVICE = device()
LIMIT = None


@torch.no_grad()
def run_model(mname, items):
    tk, model = load_encoder(mname)
    vcache = ValueCache(model)
    patcher = ValuePatcher(model)

    rows = []
    n_done = 0
    for batch in batches(items, None, BATCH):
        enc_clean = tk([it["clean"].replace("[MASK]", tk.mask_token)
                        for it in batch],
                       return_tensors="pt", padding=True).to(DEVICE)
        logits = model(**enc_clean).logits
        probs = torch.softmax(logits, dim=-1)

        meta = []
        for b, it in enumerate(batch):
            mm = it["models"][mname]
            mp = mm["mask_pos"]
            gid, fid = mm["target_ids"]["gold"], mm["target_ids"]["foil"]
            d_clean = (logits[b, mp, gid] - logits[b, mp, fid]).item()
            p_clean = probs[b, mp, gid].item()
            meta.append((mp, gid, fid, d_clean, p_clean))

        cues = list(batch[0]["cues"].keys())
        for cue in cues:
            # corrupted run -> cache values
            enc_corr = tk([it["corrupted"][cue].replace("[MASK]", tk.mask_token)
                           for it in batch],
                          return_tensors="pt", padding=True).to(DEVICE)
            assert enc_corr["input_ids"].shape == enc_clean["input_ids"].shape
            with vcache.capture():
                model(**enc_corr)
            cached = dict(vcache.values)

            rows_spans = []
            for b, it in enumerate(batch):
                s, e = it["models"][mname]["spans"][cue]
                rows_spans.append((b, s, e))

            for layer in range(N_LAYERS):
                with patcher.patch(layer, cached, rows_spans):
                    plogits = model(**enc_clean).logits
                pprobs = torch.softmax(plogits, dim=-1)
                for b, it in enumerate(batch):
                    mp, gid, fid, d_clean, p_clean = meta[b]
                    d_patch = (plogits[b, mp, gid] - plogits[b, mp, fid]).item()
                    p_patch = pprobs[b, mp, gid].item()
                    rows.append({
                        "id": it["id"], "model": mname,
                        "condition": it["condition"], "gold": it["gold"],
                        "subj_kind": it["lex"]["subj_kind"],
                        "cue": cue, "layer": layer,
                        "delta_clean": d_clean, "delta_patched": d_patch,
                        "score": d_clean - d_patch,
                        "p_gold_clean": p_clean, "p_gold_patched": p_patch,
                        "p_drop": p_clean - p_patch,
                        "correct_clean": d_clean > 0,
                    })
        n_done += len(batch)
        if n_done % 640 < BATCH:
            print(f"  {mname}: {n_done}/{len(items)} items")
    del model
    return rows


def main():
    global MODELS
    only = set(x.strip() for x in os.environ.get("MI_ONLY", "").split(",") if x.strip())
    if only:
        MODELS = {k: v for k, v in MODELS.items() if k in only}
    items = [json.loads(l) for l in open(FAM_PATH, encoding="utf-8")]
    if LIMIT:
        items = items[:LIMIT]
    print(f"items: {len(items)}   device: {DEVICE}")

    all_rows = []
    for mname in MODELS:
        print(f"\nrunning {mname} ...")
        mitems = [it for it in items if mname in it.get("models", {})]
        all_rows.extend(run_model(mname, mitems))

    df = pd.DataFrame(all_rows)
    only = set(x.strip() for x in os.environ.get("MI_ONLY", "").split(",") if x.strip())
    if only and os.path.exists(OUT_PATH):
        old = pd.read_csv(OUT_PATH)
        keep = old[~old.model.isin(list(only))]
        df = pd.concat([keep, df], ignore_index=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nsaved: {OUT_PATH}  ({len(df)} rows)  [{','.join(sorted(df.model.unique().tolist()))}]")

    # quick view: mean score per cue × layer, T2 congruent, correct items only
    pd.set_option("display.width", 140)
    sub = df[(df.condition == "T2") & df.correct_clean]
    pv = sub.pivot_table(index="layer", columns=["model", "cue"],
                         values="score", aggfunc="mean")
    print("\nT2 (congruent adj), correct items — mean score by layer:")
    print(pv.round(3))


if __name__ == "__main__":
    main()
