#!/usr/bin/env python3
"""
Value zeroing (Experiment 1): context mixing via Value Zeroing
(Mohebbi et al., EACL 2023; as used by Amirzadeh et al. 2024).

For each model, item, layer ℓ, and each WORD w in the sentence:
  zero the value vectors of w's token span at layer ℓ, re-run, and measure
  cosine distance between the original and modified representation of the
  [MASK] token at layer ℓ's output. Larger distance = w contributes more to
  the target representation at that layer.

Per item & layer, scores are normalized to sum to 1 over all context words
(excluding [CLS]/[SEP] and the mask itself), then mapped to roles:
  subject / adjective / attractor (from the item's cue spans)
  others = mean over remaining words (prep, complement, modifier) — the
  significance baseline, as in the source papers.

Output: data/zeroing_results.csv (long):
  id, model, condition, gold, subj_kind, layer, role, score_norm
"""

import json
import os
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F

from cuewin.config import data_dir, model_path
from cuewin.families import batches
from cuewin.hooks import ValueZeroer, HiddenStateCache
from cuewin.models import device, load_encoder

FAM_PATH = os.path.join(data_dir(), "families_arbert.jsonl")
OUT_PATH = os.path.join(data_dir(), "zeroing_results.csv")

MODELS = {
    "arabert": model_path("arabert"),
    "camelbert": model_path("camelbert"),
    "arbert": model_path("arbert"),
}

BATCH = 32
N_LAYERS = 12
DEVICE = device()
LIMIT = None
MASK = "[MASK]"


def word_spans(tk, words):
    """Token span [start, end) per word, [CLS] at 0 (mirrors 02_build_families)."""
    spans, pos = [], 1
    for w in words:
        n = 1 if w == MASK else len(tk.encode(w, add_special_tokens=False))
        spans.append((pos, pos + n))
        pos += n
    return spans


@torch.no_grad()
def run_model(mname, items):
    tk, model = load_encoder(mname)
    zeroer = ValueZeroer(model)
    hcache = HiddenStateCache(model)

    rows = []
    n_done = 0
    for batch in batches(items, None, BATCH):
        texts = [it["clean"].replace(MASK, tk.mask_token) for it in batch]
        enc = tk(texts, return_tensors="pt", padding=True).to(DEVICE)

        # original hidden states at every layer
        with hcache.capture():
            model(**enc)
        orig = {l: hcache.hidden[l].clone() for l in range(N_LAYERS)}

        # per-item word structure (shared template within batch)
        words_per_item = [it["clean"].split() for it in batch]
        n_words = len(words_per_item[0])
        mask_word_idx = words_per_item[0].index(MASK)
        spans_per_item = [word_spans(tk, ws) for ws in words_per_item]
        mask_pos = [it["models"][mname]["mask_pos"] for it in batch]

        # role of each word index, per item
        roles_per_item = []
        for it, ws in zip(batch, words_per_item):
            roles = ["others"] * n_words
            for cue, d in it["cues"].items():
                roles[ws.index(d["clean_word"])] = cue
            roles_per_item.append(roles)

        # distance[b][layer][word]
        dist = [[[0.0] * n_words for _ in range(N_LAYERS)] for _ in batch]
        for w_idx in range(n_words):
            if w_idx == mask_word_idx:
                continue
            rows_spans = [(b, *spans_per_item[b][w_idx])
                          for b in range(len(batch))]
            for layer in range(N_LAYERS):
                with zeroer.zero(layer, rows_spans), hcache.capture():
                    model(**enc)
                mod = hcache.hidden[layer]
                for b in range(len(batch)):
                    mp = mask_pos[b]
                    cos = F.cosine_similarity(orig[layer][b, mp],
                                              mod[b, mp], dim=0).item()
                    dist[b][layer][w_idx] = 1.0 - cos

        # normalize per item & layer; aggregate roles
        for b, it in enumerate(batch):
            for layer in range(N_LAYERS):
                d = dist[b][layer]
                total = sum(d) or 1.0
                norm = [x / total for x in d]
                role_scores = defaultdict(list)
                for w_idx in range(n_words):
                    if w_idx == mask_word_idx:
                        continue
                    role_scores[roles_per_item[b][w_idx]].append(norm[w_idx])
                for role, vals in role_scores.items():
                    rows.append({
                        "id": it["id"], "model": mname,
                        "condition": it["condition"], "gold": it["gold"],
                        "subj_kind": it["lex"]["subj_kind"],
                        "layer": layer, "role": role,
                        "score_norm": sum(vals) / len(vals),
                    })
        n_done += len(batch)
        if n_done % 640 < BATCH:
            print(f"  {mname}: {n_done}/{len(items)} items")
    del model
    return rows


def main():
    items = [json.loads(l) for l in open(FAM_PATH, encoding="utf-8")]
    if LIMIT:
        items = items[:LIMIT]
    print(f"items: {len(items)}   device: {DEVICE}")

    only = set(x.strip() for x in os.environ.get("MI_ONLY", "").split(",") if x.strip())
    if only:
        chosen = {k: v for k, v in MODELS.items() if k in only}
    else:
        chosen = MODELS

    all_rows = []
    for mname in chosen:
        print(f"\nrunning {mname} ...")
        mitems = [it for it in items if mname in it.get("models", {})]
        all_rows.extend(run_model(mname, mitems))

    df = pd.DataFrame(all_rows)
    if only and os.path.exists(OUT_PATH):
        old = pd.read_csv(OUT_PATH)
        keep = old[~old.model.isin(list(only))]
        df = pd.concat([keep, df], ignore_index=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nsaved: {OUT_PATH}  ({len(df)} rows)  [{','.join(sorted(df.model.unique().tolist()))}]")

    pd.set_option("display.width", 140)
    sub = df[df.condition == "T2"]
    pv = sub.pivot_table(index="layer", columns=["model", "role"],
                         values="score_norm", aggfunc="mean")
    print("\nT2 — mean normalized contribution to [MASK] rep by layer:")
    print(pv.round(3))


if __name__ == "__main__":
    main()
