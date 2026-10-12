#!/usr/bin/env python3
"""
03b_freq_baseline.py — Task 1: frequency-baseline control for the masculine default.

The behavioral metric Δ = logit(gold) − logit(foil) carries a CONSTANT bias toward
the more frequent verb form (e.g. كتب ≫ كتبت as unigrams). Gender-balancing cancels
this in the cross-gender aggregate, but NOT in the per-cell masc/fem accuracy that the
"masculine default" narrative rests on. This script estimates that prior and subtracts it.

For each verb frame we measure the model's a-priori preference between the masculine and
feminine verb forms with the SUBJECT GENDER CUE REMOVED, under two null frames:
  subjless : "[MASK] <complement>"            (pro-drop VSO, no subject at all)
  submask  : "[MASK] [MASK] <complement>"     (subject position masked, length-matched-ish)
delta_prior = logit(masc_form) − logit(fem_form)  at the verb-mask position.

Then for every T1 item we form the frequency-corrected preference
  corrected = [logit(gold) − logit(foil)]  −  (prior preference for gold over foil)
            = logit_diff − (delta_prior if gold=='m' else −delta_prior)
and recompute T1 masc/fem accuracy and the masc−fem gap on corrected scores.

Additive only: reads families.jsonl + behavioral_results.csv, writes new files.
Outputs:
  data/freq_baseline_frames.csv   per (model, verb frame, baseline): delta_prior
  data/freq_baseline.csv          summary: raw vs corrected acc & gap, per model & baseline
"""

import json
import os

import pandas as pd
import torch

from cuewin.config import data_dir, model_path
from cuewin.models import device, load_encoder

FAM_PATH = os.path.join(data_dir(), "families_arbert.jsonl")
BEH_PATH = os.path.join(data_dir(), "behavioral_results.csv")
OUT_FRAMES = os.path.join(data_dir(), "freq_baseline_frames.csv")
OUT_SUMMARY = os.path.join(data_dir(), "freq_baseline.csv")

MODELS = {
    "arabert": model_path("arabert"),
    "camelbert": model_path("camelbert"),
    "arbert": model_path("arbert"),
}
DEVICE = device()


def build_frames(items):
    """One representative per verb frame: complement words + per-model masc/fem ids."""
    frames = {}
    for it in items:
        if it["condition"] != "T1":
            continue
        if not all(m in it.get("models", {}) for m in MODELS):
            continue
        vf = it["lex"]["verb_masc"]
        if vf in frames:
            continue
        words = it["clean"].split()
        mi = words.index("[MASK]")
        complement = words[mi + 1:]               # T1 = S [MASK] C  → C is after mask
        ids = {}
        for m in MODELS:
            if m not in it.get("models", {}):
                continue
            g = it["models"][m]["target_ids"]["gold"]
            f = it["models"][m]["target_ids"]["foil"]
            # gold/foil are gender-relative; recover absolute masc/fem ids
            masc_id, fem_id = (g, f) if it["gold"] == "m" else (f, g)
            ids[m] = {"masc": masc_id, "fem": fem_id}
        frames[vf] = {"complement": complement, "ids": ids,
                      "gloss": it["lex"]["verb_gloss"]}
    return frames


@torch.no_grad()
def prior_for_model(mname, frames):
    tk, model = load_encoder(mname)
    M = tk.mask_token

    rows = []
    for vf, info in frames.items():
        comp = " ".join(info["complement"])
        texts = {
            "subjless": f"{M} {comp}",
            "submask": f"{M} {M} {comp}",
        }
        masc_id = info["ids"][mname]["masc"]
        fem_id = info["ids"][mname]["fem"]
        for baseline, text in texts.items():
            enc = tk(text, return_tensors="pt").to(DEVICE)
            logits = model(**enc).logits[0]                       # (L, V)
            mask_pos = (enc["input_ids"][0] == tk.mask_token_id).nonzero(as_tuple=True)[0]
            vp = mask_pos[-1].item()                              # verb mask = last mask
            dp = (logits[vp, masc_id] - logits[vp, fem_id]).item()
            rows.append({"model": mname, "verb_frame": vf, "gloss": info["gloss"],
                         "baseline": baseline, "delta_prior_masc_minus_fem": dp})
    del model
    return rows


def main():
    global MODELS
    only = set(x.strip() for x in os.environ.get("MI_ONLY", "").split(",") if x.strip())
    if only:
        MODELS = {k: v for k, v in MODELS.items() if k in only}
    items = [json.loads(l) for l in open(FAM_PATH, encoding="utf-8")]
    frames = build_frames(items)
    print(f"unique T1 verb frames: {len(frames)}   device: {DEVICE}")

    frame_rows = []
    for mname in MODELS:
        print(f"  priors: {mname} ...")
        frame_rows.extend(prior_for_model(mname, frames))
    fdf = pd.DataFrame(frame_rows)
    fdf.to_csv(OUT_FRAMES, index=False)
    print(f"saved {OUT_FRAMES} ({len(fdf)} rows)")

    # map item id -> verb frame, for correcting T1 behavioral rows
    id2vf = {it["id"]: it["lex"]["verb_masc"] for it in items if it["condition"] == "T1"}

    beh = pd.read_csv(BEH_PATH)
    beh = beh[beh.condition == "T1"].copy()
    beh["verb_frame"] = beh["id"].map(id2vf)

    summary = []
    for mname in MODELS:
        for baseline in ("subjless", "submask"):
            prior = (fdf[(fdf.model == mname) & (fdf.baseline == baseline)]
                     .set_index("verb_frame")["delta_prior_masc_minus_fem"].to_dict())
            sub = beh[beh.model == mname].copy()
            sub["dp"] = sub["verb_frame"].map(prior)
            # prior preference for GOLD over FOIL: +dp if gold masc, −dp if gold fem
            sub["prior_pref_gold"] = sub.apply(
                lambda r: r["dp"] if r["gold"] == "m" else -r["dp"], axis=1)
            sub["corrected_diff"] = sub["logit_diff"] - sub["prior_pref_gold"]
            sub["correct_corr"] = sub["corrected_diff"] > 0

            def acc(df, g, col):
                d = df[df.gold == g]
                return 100.0 * d[col].mean()

            am_raw, af_raw = acc(sub, "m", "correct"), acc(sub, "f", "correct")
            am_cor, af_cor = acc(sub, "m", "correct_corr"), acc(sub, "f", "correct_corr")
            mean_prior_m = sub[sub.gold == "m"]["dp"].mean()
            summary.append({
                "model": mname, "baseline": baseline,
                "mean_delta_prior(masc-fem)": round(mean_prior_m, 3),
                "acc_masc_raw": round(am_raw, 1), "acc_fem_raw": round(af_raw, 1),
                "gap_raw": round(am_raw - af_raw, 1),
                "acc_masc_corr": round(am_cor, 1), "acc_fem_corr": round(af_cor, 1),
                "gap_corr": round(am_cor - af_cor, 1),
            })

    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUT_SUMMARY, index=False)
    print(f"saved {OUT_SUMMARY}\n")
    pd.set_option("display.width", 160)
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
