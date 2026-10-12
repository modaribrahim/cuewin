"""Encoder-vs-decoder comparison table (port of ``10b_encoder_decoder_compare``).

Builds a fair 3-model accuracy-by-condition comparison by restricting the two
encoders (behavioral_results.csv) to EXACTLY the verb frames the decoder could use
(the frames present in decoder_prior_frames.csv). Prints tables and writes
``data/enc_dec_compare.csv``. Read-only on CSVs.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from cuewin.config import data_dir


def main():
    base = data_dir()
    fam = [json.loads(l) for l in open(os.path.join(base, "families.jsonl"),
                                       encoding="utf-8")]
    id2frame = {it["id"]: it["lex"]["verb_masc"] for it in fam}

    dec = pd.read_csv(os.path.join(base, "decoder_behavioral.csv"))
    prior = pd.read_csv(os.path.join(base, "decoder_prior_frames.csv"))
    usable = set(prior["verb_frame"])
    print(f"decoder-usable frames: {len(usable)}")

    beh = pd.read_csv(os.path.join(base, "behavioral_results.csv"))
    beh["frame"] = beh["id"].map(id2frame)
    beh = beh[beh["frame"].isin(usable)].copy()   # restrict encoders to same frames

    enc = beh[["model", "condition", "gold", "correct"]].rename(columns={"correct": "acc"})
    dc = dec[["model", "condition", "gold", "correct_raw"]].rename(columns={"correct_raw": "acc"})
    allm = pd.concat([enc, dc], ignore_index=True)

    tab = (allm.groupby(["condition", "model"]).acc.mean().unstack() * 100).round(1)
    order = ["arabert", "camelbert", "aragpt2"]
    tab = tab[[c for c in order if c in tab.columns]]
    print("\n=== raw accuracy (%) by condition — SAME 27-frame subset, all 3 models ===")
    print(tab)

    dsum = (dec.groupby("condition").agg(n=("id", "size"),
            raw=("correct_raw", "mean"), corr=("correct_corr", "mean")))
    dsum[["raw", "corr"]] = (dsum[["raw", "corr"]] * 100).round(1)
    print("\n=== AraGPT2: raw vs frequency-prior-corrected accuracy (%) ===")
    print(dsum)

    g = (dec.groupby(["condition", "gold"]).correct_raw.mean() * 100).round(1).unstack()
    print("\n=== AraGPT2 raw accuracy (%) by gold gender ===")
    print(g)

    out = os.path.join(base, "enc_dec_compare.csv")
    tab.to_csv(out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
