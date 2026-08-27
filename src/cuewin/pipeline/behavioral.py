#!/usr/bin/env python3
"""
Behavioral gate (Experiment 0).

For both models, on all items in families.jsonl:
  feed the clean masked sentence, read logits at mask_pos, and record
    logit_gold, logit_foil, logit_diff = gold - foil
    correct      = logit_gold > logit_foil   (forced choice between the pair)
    p_gold/p_foil (full-vocab softmax)
    top1 token   (unrestricted argmax — qualitative sanity)

Outputs:
  data/behavioral_results.csv   per item × model
  console summary: accuracy & mean logit diff per condition × gender × model,
  plus noun-vs-name subject breakdown.

Gate (design.md §2.3): if accuracy on the easy cells (T1/T2 congruent, SVO)
is >= 85% for a model, that model is analyzed pretrained (no fine-tuning).
"""

import json
import os

import pandas as pd
import torch

from cuewin.config import data_dir, model_path
from cuewin.models import device, load_encoder

FAM_PATH = os.path.join(data_dir(), "families_arbert.jsonl")
OUT_PATH = os.path.join(data_dir(), "behavioral_results.csv")

MODELS = {
    "arabert": model_path("arabert"),
    "camelbert": model_path("camelbert"),
    "arbert": model_path("arbert"),
}

BATCH = 32
DEVICE = device()


def run_model(mname, items):
    tk, model = load_encoder(mname)

    rows = []
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        texts = [it["clean"].replace("[MASK]", tk.mask_token) for it in batch]
        enc = tk(texts, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits          # (B, L, V)
        probs = torch.softmax(logits, dim=-1)
        for b, it in enumerate(batch):
            mp = it["models"][mname]["mask_pos"]
            gold_id = it["models"][mname]["target_ids"]["gold"]
            foil_id = it["models"][mname]["target_ids"]["foil"]
            lg = logits[b, mp, gold_id].item()
            lf = logits[b, mp, foil_id].item()
            top1 = logits[b, mp].argmax().item()
            rows.append({
                "id": it["id"], "model": mname,
                "condition": it["condition"], "gold": it["gold"],
                "subj_kind": it["lex"]["subj_kind"],
                "logit_gold": lg, "logit_foil": lf, "logit_diff": lg - lf,
                "correct": lg > lf,
                "p_gold": probs[b, mp, gold_id].item(),
                "p_foil": probs[b, mp, foil_id].item(),
                "top1_token": tk.convert_ids_to_tokens([top1])[0],
                "top1_is_gold": top1 == gold_id,
            })
        if (i // BATCH) % 20 == 0:
            print(f"  {mname}: {i + len(batch)}/{len(items)}")
    del model
    return rows


def main():
    items = [json.loads(l) for l in open(FAM_PATH, encoding="utf-8")]
    print(f"items: {len(items)}   device: {DEVICE}")

    all_rows = []
    for mname in MODELS:
        print(f"\nrunning {mname} ...")
        mitems = [it for it in items if mname in it.get("models", {})]
        all_rows.extend(run_model(mname, mitems))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nsaved: {OUT_PATH}")

    pd.set_option("display.width", 120)
    print("\n=== ACCURACY (gold logit > foil logit) ===")
    acc = df.pivot_table(index="condition", columns=["model", "gold"],
                         values="correct", aggfunc="mean")
    print((acc * 100).round(1))

    print("\n=== MEAN LOGIT DIFF (gold - foil) ===")
    ld = df.pivot_table(index="condition", columns=["model", "gold"],
                        values="logit_diff", aggfunc="mean")
    print(ld.round(2))

    print("\n=== ACCURACY by subject kind (noun vs name) ===")
    kind = df.pivot_table(index=["condition"], columns=["model", "subj_kind"],
                          values="correct", aggfunc="mean")
    print((kind * 100).round(1))

    print("\n=== unrestricted top-1 = gold form (hard cloze) ===")
    t1 = df.pivot_table(index="condition", columns="model",
                        values="top1_is_gold", aggfunc="mean")
    print((t1 * 100).round(1))

    print("\n=== GATE (T1+T2 accuracy, both genders pooled) ===")
    for mname in MODELS:
        easy = df[(df.model == mname) & (df.condition.isin(["T1", "T2"]))]
        a = easy.correct.mean() * 100
        verdict = "PASS — analyze pretrained" if a >= 85 else "FAIL — fine-tune"
        print(f"  {mname}: {a:.1f}%  → {verdict}")


if __name__ == "__main__":
    main()
