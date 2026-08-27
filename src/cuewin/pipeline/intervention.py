#!/usr/bin/env python3
"""
12_intervention.py — Task 4c: causal intervention / repair at the located site.

Closes the causal loop: if the mid-layer interference we localized is the CAUSE of
attraction errors, then NEUTRALIZING the interfering cue's value vectors at the
located layers should REPAIR those errors.

  *** PROOF OF MECHANISM, NOT A DEBIASING METHOD. ***
  The intervention uses ORACLE knowledge of the interfering cue's token position. It
  shows the located representations are causally responsible for the errors; it is NOT
  a deployable bias fix. No "we debias Arabic models" language anywhere (HANDOFF §8).

Conditions / interfering cue:
  T6 -> attractor (recency distractor)   ;  T3 -> conflicting adjective.

Intervention (core): ZERO the interfering cue's value vectors at layers 5–11 (the
located window), all layers simultaneously, one forward pass; recompute
Δ = logit(gold) − logit(foil) at the [MASK].

Controls (HANDOFF §8):
  - early-window control: same zeroing at layers 0–3 (should repair little);
  - correctness control: same intervention on items the model got RIGHT in the clean
    run (should NOT break them — we report the breakage rate).

Metrics (per model × condition):
  repair rate   = fraction of clean-WRONG items whose Δ flips to > 0 under neutralize;
  control rate  = same under the early-window (0–3) intervention;
  breakage rate = fraction of clean-CORRECT items whose Δ flips to < 0 under neutralize;
  mean ΔΔ       = mean(Δ_neutralize − Δ_clean) on clean-WRONG items.

Additive only. Reads data/families.jsonl. Writes:
  data/intervention_results.csv   per item: deltas + clean_correct flag
  data/intervention_summary.csv   the metrics table
"""

import json
import os
from contextlib import ExitStack

import pandas as pd
import torch

from cuewin.config import data_dir, model_path
from cuewin.constants import ENCODERS
from cuewin.families import word_spans
from cuewin.hooks import ValueZeroer
from cuewin.models import device, load_encoder

FAM_PATH = os.path.join(data_dir(), "families_arbert.jsonl")
OUT_RES = os.path.join(data_dir(), "intervention_results.csv")
OUT_SUM = os.path.join(data_dir(), "intervention_summary.csv")

MODELS = {k: model_path(k) for k in ENCODERS}
DEVICE = device()
BATCH = 32
MASK = "[MASK]"
LOCATED = range(5, 12)     # located interference window
EARLY = range(0, 4)        # control window
CUE_OF = {"T6": "attractor", "T3": "adjective"}


@torch.no_grad()
def run_model(mname, items):
    tk, model = load_encoder(mname)
    zeroer = ValueZeroer(model)

    def deltas(enc, mask_pos, gold, foil, rows_spans=None, layers=None):
        with ExitStack() as stack:
            if rows_spans is not None:
                for L in layers:
                    stack.enter_context(zeroer.zero(L, rows_spans))
            logits = model(**enc).logits
        out = []
        for b in range(len(mask_pos)):
            lm = logits[b, mask_pos[b]]
            out.append((lm[gold[b]] - lm[foil[b]]).item())
        return out

    rows = []
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        texts = [it["clean"].replace(MASK, tk.mask_token) for it in batch]
        enc = tk(texts, return_tensors="pt", padding=True).to(DEVICE)
        mask_pos = [it["models"][mname]["mask_pos"] for it in batch]
        gold = [it["models"][mname]["target_ids"]["gold"] for it in batch]
        foil = [it["models"][mname]["target_ids"]["foil"] for it in batch]

        # interfering-cue span per item
        rows_spans = []
        for b, it in enumerate(batch):
            words = it["clean"].split()
            cue_word = it["cues"][CUE_OF[it["condition"]]]["clean_word"]
            spans, _ = word_spans(tk, words)
            s, e = spans[words.index(cue_word)]
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
    global MODELS
    only = set(x.strip() for x in os.environ.get("MI_ONLY", "").split(",") if x.strip())
    if only:
        MODELS = {k: v for k, v in MODELS.items() if k in only}
    items = [json.loads(l) for l in open(FAM_PATH, encoding="utf-8")
             if json.loads(l)["condition"] in CUE_OF]
    print(f"items (T6+T3): {len(items)}   device: {DEVICE}")

    rows = []
    for mname, mpath in MODELS.items():
        print(f"\n{mname} ...")
        mitems = [it for it in items if mname in it.get("models", {})]
        rows.extend(run_model(mname, mitems))
    df = pd.DataFrame(rows)
    only = set(x.strip() for x in os.environ.get("MI_ONLY", "").split(",") if x.strip())
    if only and os.path.exists(OUT_RES):
        old = pd.read_csv(OUT_RES)
        keep = old[~old.model.isin(list(only))]
        df = pd.concat([keep, df], ignore_index=True)
    df.to_csv(OUT_RES, index=False)
    print(f"\nsaved {OUT_RES} ({len(df)} rows)  [{','.join(sorted(df.model.unique().tolist()))}]")

    summ = []
    for mname in sorted(df.model.unique()):
        for cond in CUE_OF:
            d = df[(df.model == mname) & (df.condition == cond)]
            wrong = d[~d.clean_correct]
            right = d[d.clean_correct]
            summ.append({
                "model": mname, "condition": cond, "cue": CUE_OF[cond],
                "n_wrong": len(wrong), "n_correct": len(right),
                "repair_rate_L5-11": round((wrong.delta_neut > 0).mean(), 3) if len(wrong) else None,
                "control_rate_L0-3": round((wrong.delta_ctrl > 0).mean(), 3) if len(wrong) else None,
                "mean_dd_neut": round((wrong.delta_neut - wrong.delta_clean).mean(), 3) if len(wrong) else None,
                "breakage_rate": round((right.delta_neut < 0).mean(), 3) if len(right) else None,
            })
    sdf = pd.DataFrame(summ)
    sdf.to_csv(OUT_SUM, index=False)
    pd.set_option("display.width", 160)
    print("\n=== intervention summary ===")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()