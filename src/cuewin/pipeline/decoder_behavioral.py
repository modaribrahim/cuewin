#!/usr/bin/env python3
"""
10_decoder_behavioral.py — Task 4: AraGPT2 decoder behavioral contrast.

Size-matched encoder-vs-decoder comparison: AraGPT2-base (~135M) vs the two
BERT-base encoders (~110M). Cleaner than the reference paper's 110M-vs-8B
comparison, which confounds architecture with scale.

BEHAVIORAL ONLY — no patching/zeroing on GPT-2 (HANDOFF §7). Metric mirrors the
encoders': a two-candidate, single-token forced choice between the masculine and
feminine verb forms, scored from the model's NEXT-TOKEN distribution.

Prediction (stated in advance): a decoder cannot see a post-verb subject, so in
VSO (T4) it must fall back on the verb-form frequency prior; in SVO (T1) and the
attractor condition (T6) it can use the left-context subject / suffers recency
attraction. So T4 ≈ prior baseline, while T1 tests left-context cue use.

Tokenization gates (HANDOFF §9):
- BPE leading-space trap: mid-sentence verbs are space-prefixed, so candidates are
  encoded as " "+form and required single-token on THAT encoding. 27/32 frames pass;
  the 5 that don't (شرب لبس غسل طبخ نظف) are DROPPED and logged (no silent caps).
- VSO (T4) verb is otherwise sentence-initial (bare, multi-token). We prepend a
  neutral temporal adverb "أمس" so the verb is space-prefixed and single-token; this
  also correctly represents VSO (subject follows the verb, unseen) and isolates the
  prior. T4 therefore coincides with the decoder prior by construction — that IS the
  architectural point, reported as such.
- Decoder frequency prior (Task-1 analogue): per frame, logit(masc)−logit(fem) after
  the neutral prefix "أمس". Accuracies reported raw AND prior-corrected.

Comparability: never compares raw logits across models; only within-model contrasts
and derived accuracies. Encoder and decoder metrics are matched in FORM but differ in
visible context by architectural necessity — that difference is the object of study.

Additive only. Reads data/families.jsonl + data/lexicon.json. Writes:
  data/decoder_prior_frames.csv   per frame: delta_prior (masc−fem)
  data/decoder_behavioral.csv     per item: condition, id, gold, frame, delta_raw,
                                   correct_raw, delta_corr, correct_corr
"""

import json
import os

import pandas as pd
import torch

from cuewin.config import data_dir
from cuewin.models import device, load_decoder

FAM_PATH = os.path.join(data_dir(), "families.jsonl")
LEX_PATH = os.path.join(data_dir(), "lexicon.json")
OUT_PRIOR = os.path.join(data_dir(), "decoder_prior_frames.csv")
OUT_BEH = os.path.join(data_dir(), "decoder_behavioral.csv")

DEVICE = device()
BATCH = 32
NEUTRAL = "أمس"   # gender-neutral temporal adverb: VSO prefix + prior probe


def usable_frames(tk, lex):
    """Verb frames whose BOTH forms are single-token WITH a leading space."""
    keep, dropped = {}, []
    for f in lex["vp_frames"]:
        m, fm = f["masc"], f["fem"]
        im = tk.encode(" " + m, add_special_tokens=False)
        ifm = tk.encode(" " + fm, add_special_tokens=False)
        if len(im) == 1 and len(ifm) == 1:
            keep[m] = {"masc_id": im[0], "fem_id": ifm[0], "fem": fm}
        else:
            dropped.append((m, fm, len(im), len(ifm)))
    return keep, dropped


@torch.no_grad()
def next_token_logits(model, tk, prefixes):
    """Left-padded batch; returns (B, V) next-token logits at the last position."""
    enc = tk(prefixes, return_tensors="pt", padding=True).to(DEVICE)
    logits = model(**enc).logits            # (B, L, V)
    return logits[:, -1, :]                  # left padding -> last pos is next-token


def main():
    tk, model = load_decoder("aragpt2")
    tk.pad_token = tk.eos_token
    tk.padding_side = "left"
    print(f"loaded aragpt2 | device {DEVICE}")

    lex = json.load(open(LEX_PATH, encoding="utf-8"))
    frames, dropped = usable_frames(tk, lex)
    print(f"usable verb frames (single-token w/ space): {len(frames)}/{len(lex['vp_frames'])}")
    print(f"DROPPED {len(dropped)}: " +
          ", ".join(f"{m}/{fm}(m{a},f{b})" for m, fm, a, b in dropped))

    # ---- decoder frequency prior per frame: logit(masc)-logit(fem) after NEUTRAL ----
    fr_list = list(frames.items())
    prior = {}
    prows = []
    for i in range(0, len(fr_list), BATCH):
        chunk = fr_list[i:i + BATCH]
        lg = next_token_logits(model, tk, [NEUTRAL] * len(chunk))
        for j, (m, info) in enumerate(chunk):
            dp = (lg[j, info["masc_id"]] - lg[j, info["fem_id"]]).item()
            prior[m] = dp
            prows.append({"verb_frame": m, "fem_form": info["fem"],
                          "delta_prior_masc_minus_fem": round(dp, 4)})
    pd.DataFrame(prows).to_csv(OUT_PRIOR, index=False)
    import numpy as np
    print(f"saved {OUT_PRIOR} | mean prior masc−fem = {np.mean(list(prior.values())):+.3f}")

    # ---- per-item next-token forced choice ----
    fam = [json.loads(l) for l in open(FAM_PATH, encoding="utf-8")]
    items = []
    for it in fam:
        frame = it["lex"]["verb_masc"]
        if frame not in frames:
            continue                       # dropped frame: logged above
        w = it["clean"].split()
        mi = w.index("[MASK]")
        pre = w[:mi]
        prefix = NEUTRAL if len(pre) == 0 else " ".join(pre)   # T4 -> neutral prefix
        gid = frames[frame]["masc_id"] if it["gold"] == "m" else frames[frame]["fem_id"]
        fid = frames[frame]["fem_id"] if it["gold"] == "m" else frames[frame]["masc_id"]
        items.append({"id": it["id"], "condition": it["condition"], "gold": it["gold"],
                      "frame": frame, "prefix": prefix, "gold_id": gid, "foil_id": fid})
    print(f"items scored: {len(items)} (of {len(fam)}; "
          f"{len(fam) - len(items)} dropped for frame)")

    rows = []
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        lg = next_token_logits(model, tk, [b["prefix"] for b in batch])
        for j, b in enumerate(batch):
            draw = (lg[j, b["gold_id"]] - lg[j, b["foil_id"]]).item()
            dp = prior[b["frame"]]
            prior_pref_gold = dp if b["gold"] == "m" else -dp
            dcorr = draw - prior_pref_gold
            rows.append({"model": "aragpt2", "condition": b["condition"], "id": b["id"],
                         "gold": b["gold"], "frame": b["frame"],
                         "delta_raw": round(draw, 4), "correct_raw": draw > 0,
                         "delta_corr": round(dcorr, 4), "correct_corr": dcorr > 0})
        if (i // BATCH) % 20 == 0:
            print(f"  {min(i + BATCH, len(items))}/{len(items)}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_BEH, index=False)
    print(f"\nsaved {OUT_BEH} ({len(df)} rows)")

    pd.set_option("display.width", 140)
    summ = df.groupby("condition").agg(
        n=("id", "size"),
        acc_raw=("correct_raw", "mean"),
        acc_corr=("correct_corr", "mean")).round(3)
    summ["acc_raw"] *= 100
    summ["acc_corr"] *= 100
    print("\nAraGPT2 accuracy by condition (%):")
    print(summ.round(1))
    print("\nby gold gender (raw acc %):")
    print((df.groupby(["condition", "gold"]).correct_raw.mean() * 100).round(1))


if __name__ == "__main__":
    main()