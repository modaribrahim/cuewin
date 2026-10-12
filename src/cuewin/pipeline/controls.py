#!/usr/bin/env python3
"""
Control experiments (GPU). Additive: reads the frozen benchmark
(``families_rev.jsonl``) and writes new ``ctrl_*`` files.

Stages (``CUEWIN_STAGES``, comma-separated; default: all):

  maskheads  head output at the verb position: in the clean run, overwrite one
             head's 64-dim slice of the attention output (the input to
             ``attention.output.dense``) at the [MASK] position only, with the
             same slice from the cue-flipped run; also all 12 heads of a layer
             together (head = -1). T1 subject flip and T6 attractor flip.
             -> ctrl_maskheads_<model>.csv.gz
  swap       same-gender swap control for repair: as the gender-only patch, but
             the source run replaces the interfering cue with a different word
             of the same gender (same kind, equal token count, not the
             subject's lemma). T3 and T6, windows L0-3, L4-7, L8-11.
             -> ctrl_swap_<model>.csv, ctrl_swap_items.csv

``CUEWIN_LIMIT=n`` keeps the first n items per condition (smoke tests).
"""

import json
import os
import random
from contextlib import contextmanager

import pandas as pd
import torch

from cuewin.config import data_dir
from cuewin.families import word_spans
from cuewin.hooks import ValueCache
from cuewin.models import device, load_encoder, tokenizer
from cuewin.pipeline.revision import (CUE_OF, ENCODERS4, N_LAYERS, _enc, _margin,
                                      _meta, _value_edit, load_items)

DATA = data_dir()
DEVICE = device()
BATCH = int(os.environ.get("CUEWIN_BATCH", "32"))
LIMIT = int(os.environ.get("CUEWIN_LIMIT", "0"))
NAMES = {"محمد", "أحمد", "خالد", "فاطمة", "مريم", "ليلى"}
SWAP_WINDOWS = (("L0-3", range(0, 4)), ("L4-7", range(4, 8)), ("L8-11", range(8, 12)))


def _limit(items):
    if not LIMIT:
        return items
    out, seen = [], {}
    for it in items:
        k = it["condition"]
        if seen.get(k, 0) < LIMIT:
            out.append(it)
            seen[k] = seen.get(k, 0) + 1
    return out


def _batches(items):
    return [items[i:i + BATCH] for i in range(0, len(items), BATCH)]


def _attn_dense(model, layer):
    return model.base_model.encoder.layer[layer].attention.output.dense


# ---------------------------------------------------------------- maskheads

@contextmanager
def _capture_context(model):
    """Record the attention context (input of attention.output.dense) per layer."""
    store, handles = {}, []
    for L in range(N_LAYERS):
        def hook(module, args, L=L):
            store[L] = args[0].detach().clone()
        handles.append(_attn_dense(model, L).register_forward_pre_hook(hook))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def _context_edit(model, layer, mask_rows, source, head):
    """Overwrite the context at (row, mask_pos) of ``layer`` with ``source``;
    one head's slice, or all heads when head < 0."""
    hd = model.config.hidden_size // model.config.num_attention_heads
    sl = slice(None) if head < 0 else slice(head * hd, (head + 1) * hd)

    def hook(module, args):
        x = args[0].clone()
        for row, mp in mask_rows:
            x[row, mp, sl] = source[row, mp, sl]
        return (x,) + tuple(args[1:])

    h = _attn_dense(model, layer).register_forward_pre_hook(hook)
    try:
        yield
    finally:
        h.remove()


@torch.no_grad()
def run_maskheads(mname, items, cue):
    tk, model = load_encoder(mname)
    n_heads = model.config.num_attention_heads
    rows = []
    for bi, batch in enumerate(_batches(items)):
        meta = _meta(batch, mname)
        enc = _enc(tk, [it["clean"] for it in batch])
        enc_c = _enc(tk, [it["corrupted"][cue] for it in batch])
        assert enc_c["input_ids"].shape == enc["input_ids"].shape
        with _capture_context(model) as ctx:
            d_corr = _margin(model(**enc_c).logits, meta)
        src = dict(ctx)
        d_clean = _margin(model(**enc).logits, meta)
        mrows = [(b, mp) for b, (mp, _, _) in enumerate(meta)]
        for L in range(N_LAYERS):
            for h in range(-1, n_heads):
                with _context_edit(model, L, mrows, src[L], h):
                    d = _margin(model(**enc).logits, meta)
                for b, it in enumerate(batch):
                    rows.append({"model": mname, "condition": it["condition"], "id": it["id"],
                                 "gold": it["gold"], "subj": it["lex"]["subj_masc"],
                                 "frame": it["lex"]["verb_masc"], "cue": cue, "layer": L,
                                 "head": h, "delta_clean": round(d_clean[b], 5),
                                 "delta_corr": round(d_corr[b], 5),
                                 "delta_patched": round(d[b], 5)})
        if bi % 10 == 0:
            print(f"  maskheads {mname} {cue}: {bi * BATCH}/{len(items)}", flush=True)
    del model
    return rows


def stage_maskheads():
    items = load_items()
    for m in ENCODERS4:
        out = os.path.join(DATA, f"ctrl_maskheads_{m}.csv.gz")
        if os.path.exists(out) and not LIMIT:
            print(f"  skip {out} (exists)")
            continue
        rows = []
        for cond, cue in (("T1", "subject"), ("T6", "attractor")):
            mitems = _limit(sorted([it for it in items if it["condition"] == cond and m in it["models"]],
                                   key=lambda it: it["models"][m]["n_tokens"]))
            rows.extend(run_maskheads(m, mitems, cue))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  maskheads {m} done -> {out}")


# ---------------------------------------------------------------- swap

def _cue_index(it, cue):
    words = it["clean"].split()
    idx = [i for i, w in enumerate(words) if w == it["cues"][cue]["clean_word"]]
    assert len(idx) == 1, (it["id"], cue)
    return idx[0]


def _kind(word):
    return "name" if word in NAMES else "noun"


def build_swaps(items):
    """One same-gender replacement per T3/T6 item, valid in every encoder
    that has the item. Returns {id: (swap_word, swapped_sentence)}."""
    tks = {m: tokenizer(m) for m in ENCODERS4}
    # pools of cue forms by (condition, gender of the form, kind), with their lemma
    pool = {}
    for it in items:
        if it["condition"] not in CUE_OF:
            continue
        cue = CUE_OF[it["condition"]]
        c = it["cues"][cue]
        lemma = it["lex"]["attr_masc"] if cue == "attractor" else it["lex"]["adj_masc"]
        for w in (c["clean_word"], c["flipped_word"]):
            g = "m" if w == lemma else "f"
            pool.setdefault((it["condition"], g, _kind(w)), {})[w] = lemma
    rng = random.Random(60)
    swaps, dropped = {}, 0
    for it in items:
        if it["condition"] not in CUE_OF:
            continue
        cue = CUE_OF[it["condition"]]
        c = it["cues"][cue]
        lemma = it["lex"]["attr_masc"] if cue == "attractor" else it["lex"]["adj_masc"]
        w0 = c["clean_word"]
        g = "m" if w0 == lemma else "f"
        cands = sorted(w for w, lem in pool[(it["condition"], g, _kind(w0))].items()
                       if lem != lemma and lem != it["lex"]["subj_masc"])
        rng.shuffle(cands)
        words = it["clean"].split()
        k = _cue_index(it, cue)
        chosen = None
        for w in cands:
            new = words[:k] + [w] + words[k + 1:]
            ok = True
            for m, ann in it["models"].items():
                if m not in tks:
                    continue
                sp, n = word_spans(tks[m], new)
                full = len(tks[m].encode(" ".join(new).replace("[MASK]", tks[m].mask_token)))
                if (n != ann["n_tokens"] or full != n
                        or list(sp[k]) != list(ann["spans"][cue])):
                    ok = False
                    break
            if ok:
                chosen = (w, " ".join(new))
                break
        if chosen is None:
            dropped += 1
        else:
            swaps[it["id"]] = chosen
    print(f"  swap: {len(swaps)} items with a replacement, {dropped} dropped")
    return swaps


@torch.no_grad()
def run_swap(mname, items, swaps):
    tk, model = load_encoder(mname)
    cache = ValueCache(model)
    rows = []
    for bi, batch in enumerate(_batches(items)):
        cue = CUE_OF[batch[0]["condition"]]
        meta = _meta(batch, mname)
        enc = _enc(tk, [it["clean"] for it in batch])
        enc_s = _enc(tk, [swaps[it["id"]][1] for it in batch])
        assert enc_s["input_ids"].shape == enc["input_ids"].shape
        with cache.capture():
            d_swap = _margin(model(**enc_s).logits, meta)
        src = dict(cache.values)
        spans = [(b, *it["models"][mname]["spans"][cue]) for b, it in enumerate(batch)]
        d_clean = _margin(model(**enc).logits, meta)
        for wname, layers in SWAP_WINDOWS:
            with _value_edit(model, list(layers), spans, src):
                d = _margin(model(**enc).logits, meta)
            for b, it in enumerate(batch):
                rows.append({"model": mname, "condition": it["condition"], "id": it["id"],
                             "gold": it["gold"], "subj": it["lex"]["subj_masc"],
                             "frame": it["lex"]["verb_masc"], "window": wname, "kind": "swap",
                             "delta_clean": round(d_clean[b], 5),
                             "delta_swapsent": round(d_swap[b], 5),
                             "delta_int": round(d[b], 5)})
        if bi % 10 == 0:
            print(f"  swap {mname}: {bi * BATCH}/{len(items)}", flush=True)
    del model
    return rows


def stage_swap():
    items = [it for it in load_items() if it["condition"] in CUE_OF]
    items = _limit(items)
    swaps = build_swaps(items)
    pd.DataFrame([{"id": i, "swap_word": w, "swapped": s} for i, (w, s) in swaps.items()]).to_csv(
        os.path.join(DATA, "ctrl_swap_items.csv"), index=False)
    for m in ENCODERS4:
        out = os.path.join(DATA, f"ctrl_swap_{m}.csv")
        if os.path.exists(out) and not LIMIT:
            print(f"  skip {out} (exists)")
            continue
        rows = []
        for cond in CUE_OF:
            mitems = sorted([it for it in items if it["condition"] == cond and m in it["models"]
                             and it["id"] in swaps], key=lambda it: it["models"][m]["n_tokens"])
            rows.extend(run_swap(m, mitems, swaps))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  swap {m} done -> {out}")


STAGES = {"maskheads": stage_maskheads, "swap": stage_swap}


def main():
    sel = [s.strip() for s in os.environ.get("CUEWIN_STAGES", ",".join(STAGES)).split(",")
           if s.strip()]
    print(f"stages: {sel}   device: {DEVICE}   encoders: {ENCODERS4}   limit: {LIMIT or 'none'}")
    for s in sel:
        print(f"\n=== {s} ===")
        STAGES[s]()


if __name__ == "__main__":
    main()
