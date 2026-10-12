#!/usr/bin/env python3
"""
The ة-morpheme ablation probe (CAMeLBERT only).

Question: does CAMeLBERT read feminine subject–verb agreement off the isolated
feminine morpheme ة, or off the whole lexical subject? Uniquely answerable in
CAMeLBERT because its 30k WordPiece vocab tokenizes 23 feminine subjects as
[stem, ##ة] where the stem token IS the masculine form. AraBERT fuses these,
so this experiment is CAMeLBERT-only BY TOKENIZER DESIGN (we say so).

Method: value ZEROING (not patching). The masculine sentence has no ة position,
so there is no aligned position to patch into; zeroing needs only one run and no
cross-run alignment (see HANDOFF §4 / §9.4).

Design (all spans are FIXED because every selected feminine subject tokenizes to
exactly [stem, ##ة] = 2 tokens):
  sentence  = <fem_subject> [MASK] <complement>
  positions = [CLS](0) stem(1) ##ة(2) [MASK](3) comp(4..) [SEP]
  Δ         = logit(fem_verb) − logit(masc_verb)  at the [MASK]  (both single-token)
  targets zeroed at each layer ℓ:
    zero_ta      : the ##ة token only      span (2,3)
    zero_stem    : the stem token only     span (1,2)
    zero_subject : whole subject (ceiling) span (1,3)
    zero_control : first complement token  span (4,5)   (length-matched floor)
  collapse(ℓ) = Δ_clean − Δ_ablated(ℓ)

Δ is within-item (same sentence, same mask), so the verb-form frequency prior is
a constant that the collapse score differences out — no Task-1 correction needed.

Additive only. Reads data/lexicon.json, writes:
  data/morpheme_subjects.csv   the 23-vs-30 subject classification (appendix table)
  data/morpheme_results.csv    long: id, subject, frame, gloss, target, layer,
                               delta_clean, delta_ablated, collapse
"""

import json
import os

import pandas as pd
import torch

from cuewin.config import data_dir
from cuewin.hooks import ValueZeroer
from cuewin.models import device, load_encoder

LEX_PATH = os.path.join(data_dir(), "lexicon.json")
OUT_SUBJ = os.path.join(data_dir(), "morpheme_subjects.csv")
OUT_RES = os.path.join(data_dir(), "morpheme_results.csv")

DEVICE = device()
N_LAYERS = 12
BATCH = 32
TA_PIECE = "##ة"

# fixed spans (start, end) given [CLS] stem ##ة [MASK] comp...
SPANS = {
    "zero_stem": (1, 2),
    "zero_ta": (2, 3),
    "zero_subject": (1, 3),
    "zero_control": (4, 5),
}
MASK_POS = 3


def select_subjects(lex):
    """Classify all subjects by the ة-rule; return (matches, full_table)."""
    matches, table = [], []
    for s in lex["subjects"]:
        cm = s["tok"]["camelbert"]
        mp, fp = cm["masc_pieces"], cm["fem_pieces"]
        rule = (fp == mp + [TA_PIECE]) and len(mp) >= 1
        table.append({
            "masc": s["masc"], "fem": s["fem"], "gloss": s["gloss"],
            "masc_pieces": " ".join(mp), "fem_pieces": " ".join(fp),
            "n_masc_tok": len(mp), "n_fem_tok": len(fp),
            "matches_ta_rule": rule,
            "reason": "clean [stem,##ة]" if rule else "ة fused into last piece",
        })
        if rule:
            matches.append(s)
    return matches, table


def select_frames(lex):
    """CAMeLBERT-usable verb frames that are single-token in BOTH genders."""
    frames = []
    for f in lex["vp_frames"]:
        if f["group"] not in ("BOTH", "CAMEL"):
            continue
        cm = f["tok"]["camelbert"]
        if cm["masc_ntok"] == 1 and cm["fem_ntok"] == 1:
            frames.append({
                "masc": f["masc"], "fem": f["fem"], "comp": f["comp"],
                "gloss": f["gloss"],
                "masc_id": cm["masc_ids"][0], "fem_id": cm["fem_ids"][0],
            })
    return frames


def build_items(subjects, frames):
    items = []
    for s in subjects:
        for f in frames:
            items.append({
                "id": f"{s['masc']}__{f['masc']}",
                "subject": s["masc"], "subject_fem": s["fem"],
                "frame": f["masc"], "gloss": f["gloss"],
                "text": f"{s['fem']} [MASK] {f['comp']}",
                "masc_id": f["masc_id"], "fem_id": f["fem_id"],
            })
    return items


@torch.no_grad()
def main():
    lex = json.load(open(LEX_PATH, encoding="utf-8"))
    subjects, table = select_subjects(lex)
    frames = select_frames(lex)
    pd.DataFrame(table).to_csv(OUT_SUBJ, index=False)
    n_match = sum(r["matches_ta_rule"] for r in table)
    print(f"subjects: {len(table)} total, {n_match} match ة-rule  -> {OUT_SUBJ}")
    print(f"CAMeLBERT single-token frames: {len(frames)}")

    items = build_items(subjects, frames)
    print(f"items: {len(subjects)} x {len(frames)} = {len(items)}   device: {DEVICE}")

    tk, model = load_encoder("camelbert")
    zeroer = ValueZeroer(model)

    # sanity: verify TA_PIECE is a real single vocab token and mask layout holds
    assert len(tk.tokenize(TA_PIECE)) >= 1

    rows = []
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        texts = [it["text"].replace("[MASK]", tk.mask_token) for it in batch]
        enc = tk(texts, return_tensors="pt", padding=True).to(DEVICE)

        # in-context assertion (HANDOFF §9.5): subject = [stem, ##ة], mask at pos 3
        ids = enc["input_ids"]
        for b, it in enumerate(batch):
            row = ids[b]
            assert row[MASK_POS].item() == tk.mask_token_id, \
                f"mask not at pos {MASK_POS} for {it['id']}"
            # ##ة must be the token immediately before the mask (pos 2)
            assert tk.convert_ids_to_tokens(row[2].item()) == TA_PIECE, \
                f"pos2 != ##ة for {it['id']}: got {tk.convert_ids_to_tokens(row[2].item())}"

        masc_ids = torch.tensor([it["masc_id"] for it in batch], device=DEVICE)
        fem_ids = torch.tensor([it["fem_id"] for it in batch], device=DEVICE)

        def delta(logits):
            # logits: (B, L, V) -> Δ at mask = logit(fem) − logit(masc) per item
            lm = logits[:, MASK_POS, :]
            return (lm.gather(1, fem_ids[:, None]) - lm.gather(1, masc_ids[:, None])).squeeze(1)

        d_clean = delta(model(**enc).logits)              # (B,)

        # per target, per layer: zero span, recompute Δ
        d_abl = {t: [None] * N_LAYERS for t in SPANS}
        for target, (s, e) in SPANS.items():
            rows_spans = [(b, s, e) for b in range(len(batch))]
            for layer in range(N_LAYERS):
                with zeroer.zero(layer, rows_spans):
                    d_abl[target][layer] = delta(model(**enc).logits)

        for b, it in enumerate(batch):
            dc = d_clean[b].item()
            for target in SPANS:
                for layer in range(N_LAYERS):
                    da = d_abl[target][layer][b].item()
                    rows.append({
                        "id": it["id"], "subject": it["subject"],
                        "frame": it["frame"], "gloss": it["gloss"],
                        "target": target, "layer": layer,
                        "delta_clean": dc, "delta_ablated": da,
                        "collapse": dc - da,
                    })
        print(f"  {min(i + BATCH, len(items))}/{len(items)} items")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_RES, index=False)
    print(f"\nsaved {OUT_RES}  ({len(df)} rows)")

    pd.set_option("display.width", 140)
    pv = df.pivot_table(index="layer", columns="target",
                        values="collapse", aggfunc="mean")
    print("\nmean collapse (Δ_clean − Δ_ablated) by layer & target:")
    print(pv.round(3))
    print("\nΔ_clean (mean over items):", round(df["delta_clean"].mean(), 3),
          " (positive = model prefers feminine verb, i.e. correct agreement)")


if __name__ == "__main__":
    main()