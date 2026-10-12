#!/usr/bin/env python3
"""
Decoder full-sentence scoring control (revision).

The next-token forced choice of ``decoder_behavioral`` asks the decoder to
pick the verb form before it has seen a post-verbal subject, so its VSO (T4)
score is close to definitional. Here the same AraGPT2-base scores whole
sentences instead, BLiMP-style: an item is correct when

    log P(sentence with gold verb) > log P(sentence with foil verb),

each sentence prefixed with the model's <|endoftext|> token. The log-ratio
splits exactly into a VERB term, log P(v_gold | left) - log P(v_foil | left),
which is all the next-token metric sees, and a CONTINUATION term,
log P(rest | left, v_gold) - log P(rest | left, v_foil), through which a
left-to-right model can register agreement with a subject that follows the
verb.

Sentences are the encoder items with [MASK] replaced by the verb form (T4 is
bare VSO, no neutral prefix). Verb forms need not be single-token here, so
all items are scored.

Additive only. Reads data/families.jsonl. Writes:
  data/rev_decoder_fullsentence.csv   per item: delta_full, delta_verb,
                                      delta_cont, correct_full, correct_verb
"""

import json
import os

import pandas as pd
import torch

from cuewin.config import data_dir
from cuewin.models import device, load_decoder

FAM_PATH = os.path.join(data_dir(), "families.jsonl")
OUT = os.path.join(data_dir(), "rev_decoder_fullsentence.csv")
DEVICE = device()
BATCH = 32


@torch.no_grad()
def token_logprobs(model, tk, texts):
    """Per-text list of (token_ids, logprobs) for every token after the BOS."""
    ids = [[tk.bos_token_id] + tk.encode(t, add_special_tokens=False) for t in texts]
    L = max(len(x) for x in ids)
    inp = torch.full((len(ids), L), tk.eos_token_id, dtype=torch.long)
    att = torch.zeros((len(ids), L), dtype=torch.long)
    for i, x in enumerate(ids):
        inp[i, :len(x)] = torch.tensor(x)
        att[i, :len(x)] = 1
    logits = model(input_ids=inp.to(DEVICE), attention_mask=att.to(DEVICE)).logits
    lp = torch.log_softmax(logits.float(), dim=-1)
    out = []
    for i, x in enumerate(ids):
        tgt = torch.tensor(x[1:], device=DEVICE)
        out.append((x[1:], lp[i, :len(x) - 1].gather(1, tgt[:, None]).squeeze(1).tolist()))
    return out


def main():
    tk, model = load_decoder("aragpt2")
    fam = [json.loads(l) for l in open(FAM_PATH, encoding="utf-8")]
    rows = []
    for i in range(0, len(fam), BATCH):
        batch = fam[i:i + BATCH]
        texts, meta = [], []
        for it in batch:
            left, right = it["clean"].split("[MASK]")
            left, right = left.strip(), right.strip()
            for form in (it["gold_form"], it["foil_form"]):
                texts.append(" ".join(w for w in (left, form, right) if w))
            # token boundaries: left part, left+verb, full sentence
            n_left = len(tk.encode(left, add_special_tokens=False)) if left else 0
            n_lv = [len(tk.encode(" ".join(w for w in (left, f) if w), add_special_tokens=False))
                    for f in (it["gold_form"], it["foil_form"])]
            meta.append((it, n_left, n_lv))
        scored = token_logprobs(model, tk, texts)
        for j, (it, n_left, n_lv) in enumerate(meta):
            parts = []
            for k in (0, 1):
                toks, lps = scored[2 * j + k]
                # BPE prefix consistency: the sentence's first tokens are the
                # left context's tokens, so the verb tokens start at n_left
                assert toks[:n_lv[k]] == tk.encode(texts[2 * j + k], add_special_tokens=False)[:n_lv[k]]
                full = sum(lps)
                verb = sum(lps[n_left:n_lv[k]])
                parts.append((full, verb, full - verb - sum(lps[:n_left])))
            (fg, vg, cg), (ff, vf, cf) = parts
            rows.append({"model": "aragpt2", "condition": it["condition"], "id": it["id"],
                         "gold": it["gold"], "frame": it["lex"]["verb_masc"],
                         "delta_full": fg - ff, "delta_verb": vg - vf, "delta_cont": cg - cf,
                         "correct_full": fg > ff, "correct_verb": vg > vf})
        if (i // BATCH) % 40 == 0:
            print(f"  {min(i + BATCH, len(fam))}/{len(fam)}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"saved {OUT} ({len(df)} rows)")
    pd.set_option("display.width", 140)
    print((df.pivot_table(index="condition", columns="gold",
                          values=["correct_full", "correct_verb"]) * 100).round(1))
    print(df.pivot_table(index="condition", columns="gold",
                         values=["delta_verb", "delta_cont"]).round(2))


if __name__ == "__main__":
    main()
