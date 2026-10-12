#!/usr/bin/env python3
"""
Revision-2 experiments (GPU). Additive only: writes new ``rev2_*`` files.

Item sets
  main  families_rev.jsonl, conditions T3 (adjective) and T6 (attractor)
  rep   families_rep.jsonl (held-out adjectives and attractors, plus the
        construct-state attractor condition T7; see build_replication)
  nat   natural_families_T6_arbert.jsonl (PADT sentences with an inserted
        attractor), MARBERT annotation added here

Stages (``CUEWIN_STAGES``, comma-separated; default: all)

  rep_core   behavioral, per-layer value patching of every cue, and the
             window sweep of revision.run_windows (patch / zero / object
             control) on the replication set
             -> rev2_rep_behavioral_<model>.csv, rev2_rep_patching_<model>.csv,
                rev2_rep_windows_<model>.csv
  rep_heads  head-level value patching of the attractor (T6, T7)
             -> rev2_rep_heads_<model>.csv
  cumwin     cumulative value windows on the interfering cue: prefix
             windows 0..k and suffix windows k..11 (k = 0..11), with the
             gender-only patch and with zeroing. A suffix window k..11
             replaces everything other tokens read from the cue's values
             from layer k on, so suffix(8) vs suffix(0) tests whether reads
             in layers 0-7 matter.
             -> rev2_cumwin_<set>_<model>.csv
  cueresid   hidden-state patching of the interfering cue: the cue span's
             hidden state at the output of layer l (l = -1 is the embedding
             output) is replaced by the one from the cue-flipped run.
             Records delta_corr (fully cue-flipped run) for gap-closed
             normalization.
             -> rev2_cueresid_<set>_<model>.csv
  natural    treebank T6 with the gender-only patch: single layers, fixed
             windows 0-3 / 4-7 / 8-11, 5-11, prefix and suffix windows, plus
             cue hidden-state patching, for all four encoders
             -> rev2_nat_windows_<model>.csv, rev2_nat_cueresid_<model>.csv

Every output is written as soon as it is complete, and existing outputs
are skipped, so an interrupted run can be restarted.

Set ``CUEWIN_LIMIT=n`` for a smoke test (first n items per condition).
"""

import json
import os

import pandas as pd
import torch

from cuewin.config import data_dir
from cuewin.families import diff_annotation
from cuewin.hooks import ValueCache
from cuewin.models import device, load_encoder, tokenizer
from cuewin.pipeline import revision as R

DATA = data_dir()
FAM = {"main": os.path.join(DATA, "families_rev.jsonl"),
       "rep": os.path.join(DATA, "families_rep.jsonl"),
       "nat": os.path.join(DATA, "natural_families_T6_arbert.jsonl")}
ENCODERS4 = R.ENCODERS4
DEVICE = device()
N_LAYERS = 12
LIMIT = int(os.environ.get("CUEWIN_LIMIT", "0") or 0)
CUE_OF = {"T3": "adjective", "T6": "attractor", "T7": "attractor"}
R.CUE_OF.update(CUE_OF)

CUM_WINDOWS = ([(f"P0-{k}", list(range(0, k + 1))) for k in range(N_LAYERS)]
               + [(f"S{k}-11", list(range(k, N_LAYERS))) for k in range(N_LAYERS)])
NAT_WINDOWS = ([(f"L{l}", [l]) for l in range(N_LAYERS)]
               + [("L0-3", [0, 1, 2, 3]), ("L4-7", [4, 5, 6, 7]),
                  ("L8-11", [8, 9, 10, 11]), ("L5-11", list(range(5, 12)))]
               + CUM_WINDOWS)


def load(set_name, conds=None):
    items = [json.loads(l) for l in open(FAM[set_name], encoding="utf-8")]
    if conds:
        items = [it for it in items if it["condition"] in conds]
    if LIMIT:
        out, seen = [], {}
        for it in items:
            k = it["condition"]
            if seen.get(k, 0) < LIMIT:
                out.append(it)
                seen[k] = seen.get(k, 0) + 1
        items = out
    return items


def by_model(items, m):
    """Items annotated for m, sorted so batches share a token length."""
    return sorted([it for it in items if m in it["models"]],
                  key=lambda it: (it["condition"], it["models"][m]["n_tokens"]))


def save(rows, name):
    path = os.path.join(DATA, name)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved {name} ({len(rows)} rows)")


def _row(m, it, **kw):
    return {"model": m, "condition": it["condition"], "id": it["id"], "gold": it["gold"],
            "subj": it["lex"]["subj_masc"], "frame": it["lex"]["verb_masc"],
            "cue_item": it["lex"].get("attr_masc") or it["lex"].get("adj_masc"),
            "cue_kind": it["lex"].get("attr_kind") or it["lex"].get("adj_kind"), **kw}


# ---------------------------------------------------------------- value windows

@torch.no_grad()
def run_value_windows(mname, items, windows):
    """Patch (gender-flipped values) and zero the interfering cue over each
    window; also record the fully cue-flipped run (delta_corr)."""
    tk, model = load_encoder(mname)
    cache = ValueCache(model)
    rows = []
    for bi, batch in enumerate(R._batches(items)):
        cue = CUE_OF[batch[0]["condition"]]
        meta = R._meta(batch, mname)
        enc = R._enc(tk, [it["clean"] for it in batch])
        enc_c = R._enc(tk, [it["corrupted"][cue] for it in batch])
        assert enc_c["input_ids"].shape == enc["input_ids"].shape
        with cache.capture():
            d_corr = R._margin(model(**enc_c).logits, meta)
        src = dict(cache.values)
        spans = [(b, *it["models"][mname]["spans"][cue]) for b, it in enumerate(batch)]
        d_clean = R._margin(model(**enc).logits, meta)
        res = {}
        for wname, layers in windows:
            for kind, source in (("patch", src), ("zero", None)):
                with R._value_edit(model, layers, spans, source):
                    res[(wname, kind)] = R._margin(model(**enc).logits, meta)
        for b, it in enumerate(batch):
            for (wname, kind), d in res.items():
                rows.append(_row(mname, it, window=wname, kind=kind, delta_clean=d_clean[b],
                                 delta_corr=d_corr[b], delta_int=d[b]))
        if bi % 20 == 0:
            print(f"    {mname}: {bi * R.BATCH}/{len(items)}")
    del model
    return rows


@torch.no_grad()
def run_cue_resid(mname, items):
    tk, model = load_encoder(mname)
    rows = []
    for bi, batch in enumerate(R._batches(items)):
        cue = CUE_OF[batch[0]["condition"]]
        meta = R._meta(batch, mname)
        enc = R._enc(tk, [it["clean"] for it in batch])
        enc_c = R._enc(tk, [it["corrupted"][cue] for it in batch])
        assert enc_c["input_ids"].shape == enc["input_ids"].shape
        out_c = model(**enc_c, output_hidden_states=True)
        hs_c = out_c.hidden_states
        d_corr = R._margin(out_c.logits, meta)
        d_clean = R._margin(model(**enc).logits, meta)
        spans = [(b, *it["models"][mname]["spans"][cue]) for b, it in enumerate(batch)]
        for layer in range(-1, N_LAYERS):
            with R._resid_edit(model, layer, spans, hs_c[layer + 1]):
                d = R._margin(model(**enc).logits, meta)
            for b, it in enumerate(batch):
                rows.append(_row(mname, it, layer=layer, delta_clean=d_clean[b],
                                 delta_corr=d_corr[b], delta_patched=d[b]))
        if bi % 20 == 0:
            print(f"    {mname}: {bi * R.BATCH}/{len(items)}")
    del model
    return rows


def per_cond(fn, items, m, *args):
    """Run fn separately per condition (batches must not mix cues)."""
    rows = []
    for cond in sorted({it["condition"] for it in items}):
        sub = [it for it in items if it["condition"] == cond]
        if sub:
            rows.extend(fn(m, sub, *args))
    return rows


# ---------------------------------------------------------------- stages

def done(name):
    """Resume support: skip outputs already written by an earlier run."""
    hit = os.path.exists(os.path.join(DATA, name))
    if hit:
        print(f"  skip {name} (exists)")
    return hit


def stage_rep_core():
    from cuewin.pipeline import behavioral, value_patching
    items = load("rep")
    for m in ENCODERS4:
        mi = by_model(items, m)
        print(f"  {m}: {len(mi)} items")
        if not done(f"rev2_rep_behavioral_{m}.csv"):
            save(behavioral.run_model(m, mi), f"rev2_rep_behavioral_{m}.csv")
        if not done(f"rev2_rep_patching_{m}.csv"):
            save(value_patching.run_model(m, mi), f"rev2_rep_patching_{m}.csv")
        wi = [it for it in mi if it["condition"] in CUE_OF]
        if not done(f"rev2_rep_windows_{m}.csv"):
            save(per_cond(R.run_windows, wi, m), f"rev2_rep_windows_{m}.csv")


def stage_rep_heads():
    items = load("rep", ("T6", "T7"))
    for m in ENCODERS4:
        if done(f"rev2_rep_heads_{m}.csv"):
            continue
        rows = []
        for cond in ("T6", "T7"):
            mi = by_model([it for it in items if it["condition"] == cond], m)
            rows.extend(R.run_heads(m, mi, "attractor", range(N_LAYERS)))
        save(rows, f"rev2_rep_heads_{m}.csv")


def stage_cumwin():
    for set_name in ("main", "rep"):
        items = load(set_name, tuple(CUE_OF))
        for m in ENCODERS4:
            if done(f"rev2_cumwin_{set_name}_{m}.csv"):
                continue
            save(per_cond(run_value_windows, by_model(items, m), m, CUM_WINDOWS),
                 f"rev2_cumwin_{set_name}_{m}.csv")


def stage_cueresid():
    for set_name in ("main", "rep"):
        items = load(set_name, tuple(CUE_OF))
        for m in ENCODERS4:
            if done(f"rev2_cueresid_{set_name}_{m}.csv"):
                continue
            save(per_cond(run_cue_resid, by_model(items, m), m),
                 f"rev2_cueresid_{set_name}_{m}.csv")


def nat_items():
    items = load("nat")
    tk = tokenizer("marbert")
    added = 0
    for it in items:
        if "marbert" not in it["models"]:
            ann = diff_annotation(it, tk)
            if ann is not None:
                it["models"]["marbert"] = ann
                added += 1
    print(f"  natural T6: {len(items)} items, MARBERT annotation added for {added}")
    return items


def stage_natural():
    items = nat_items()
    for m in ENCODERS4:
        mi = by_model(items, m)
        print(f"  {m}: {len(mi)} items")
        if not done(f"rev2_nat_windows_{m}.csv"):
            save(run_value_windows(m, mi, NAT_WINDOWS), f"rev2_nat_windows_{m}.csv")
        if not done(f"rev2_nat_cueresid_{m}.csv"):
            save(run_cue_resid(m, mi), f"rev2_nat_cueresid_{m}.csv")


STAGES = {"rep_core": stage_rep_core, "rep_heads": stage_rep_heads, "cumwin": stage_cumwin,
          "cueresid": stage_cueresid, "natural": stage_natural}


def main():
    sel = [s.strip() for s in os.environ.get("CUEWIN_STAGES", ",".join(STAGES)).split(",")
           if s.strip()]
    print(f"stages: {sel}   device: {DEVICE}   encoders: {ENCODERS4}   limit: {LIMIT or 'none'}")
    for s in sel:
        print(f"\n=== {s} ===")
        STAGES[s]()


if __name__ == "__main__":
    main()
