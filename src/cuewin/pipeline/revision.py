#!/usr/bin/env python3
"""
Revision experiments (GPU). Additive only: reads the frozen benchmark and
writes new ``rev_*`` files; committed result files are not modified.

Stages (select with ``CUEWIN_STAGES``, comma-separated; default: all):

  annotate  families_rev.jsonl = families_arbert.jsonl + a ``marbert``
            annotation on every item where MARBERT passes the same four
            alignment assertions (fourth encoder, dialect-heavy pretraining).
  marbert   behavioral + per-cue value patching for MARBERT, using the same
            run functions as the main pipeline
            -> rev_marbert_behavioral.csv, rev_marbert_patching.csv,
               rev_marbert_prior_frames.csv (verb-form prior, as freq_baseline)
  windows   repair sweep on T3/T6 for all encoders. For every window
            (12 single layers, nine 4-layer sliding windows, and the original
            5-11 / 0-3 windows) we score three interventions on the
            interfering cue and one control:
              zero    zero the cue's value vectors (original repair);
              patch   overwrite the cue's value vectors with those of its
                      gender-flipped (congruent) form: swaps only gender;
              ctrl    zero the object complement's value vectors (same
                      window, non-interfering word).
            -> rev_windows_<model>.csv (one row per item x window x intervention)
  residual  residual-stream patching: overwrite the hidden state of the
            subject span, or of the [MASK] position, at the output of
            layer l (l = -1 is the embedding output) with the hidden state
            from the subject-flipped run. Tracks where the subject's gender
            information resides layer by layer (T1, T4, T6).
            -> rev_residual_<model>.csv
  heads     head-level value patching of the subject (T1) and of the
            attractor (T6): patch one head's 64-dim value slice at one layer.
            -> rev_heads_<model>.csv
"""

import json
import os
from contextlib import ExitStack, contextmanager

import pandas as pd
import torch

from cuewin.config import data_dir
from cuewin.families import arbert_annotation, word_spans
from cuewin.hooks import ValueCache, value_linears
from cuewin.models import device, load_encoder, tokenizer

DATA = data_dir()
FAM_IN = os.path.join(DATA, "families_arbert.jsonl")
FAM_REV = os.path.join(DATA, "families_rev.jsonl")
ENCODERS4 = tuple(m.strip() for m in os.environ.get(
    "CUEWIN_ENCODERS", "arabert,camelbert,arbert,marbert").split(",") if m.strip())
DEVICE = device()
BATCH = 32
N_LAYERS = 12
MASK = "[MASK]"
CUE_OF = {"T6": "attractor", "T3": "adjective"}


def load_items(path=FAM_REV):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def stage_annotate():
    tk = tokenizer("marbert")
    assert tk.mask_token == MASK
    items = load_items(FAM_IN)
    added = 0
    for it in items:
        ann = arbert_annotation(it, tk)
        if ann is not None:
            it["models"]["marbert"] = ann
            added += 1
    with open(FAM_REV, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"marbert annotation: {added}/{len(items)} -> {FAM_REV}")


def stage_marbert():
    from cuewin.pipeline import behavioral, value_patching

    items = [it for it in load_items() if "marbert" in it["models"]]
    beh = pd.DataFrame(behavioral.run_model("marbert", items))
    beh.to_csv(os.path.join(DATA, "rev_marbert_behavioral.csv"), index=False)
    print((beh.pivot_table(index="condition", columns="gold", values="correct") * 100).round(1))
    pat = pd.DataFrame(value_patching.run_model("marbert", items))
    pat.to_csv(os.path.join(DATA, "rev_marbert_patching.csv"), index=False)

    from cuewin.pipeline import freq_baseline
    freq_baseline.MODELS = {"marbert": None}
    frames = freq_baseline.build_frames(items)
    pd.DataFrame(freq_baseline.prior_for_model("marbert", frames)).to_csv(
        os.path.join(DATA, "rev_marbert_prior_frames.csv"), index=False)


# ---------------------------------------------------------------- helpers

def _batches(items):
    """Batches whose clean/corrupted encodings share a length (padding-free
    alignment of spans): group by token count."""
    return [items[i:i + BATCH] for i in range(0, len(items), BATCH)]


def _margin(logits, meta):
    return [(logits[b, mp, g] - logits[b, mp, f]).item() for b, (mp, g, f) in enumerate(meta)]


def _meta(batch, m):
    return [(it["models"][m]["mask_pos"], it["models"][m]["target_ids"]["gold"],
             it["models"][m]["target_ids"]["foil"]) for it in batch]


@contextmanager
def _value_edit(model, layers, rows_spans, source=None, head=None):
    """Zero (source=None) or overwrite with ``source[layer]`` the value
    vectors at rows_spans for every layer in ``layers``; optionally only
    one head's 64-dim slice."""
    lins = value_linears(model)
    hd = model.config.hidden_size // model.config.num_attention_heads

    def make(layer):
        def hook(module, inputs, output):
            out = output.clone()
            sl = slice(None) if head is None else slice(head * hd, (head + 1) * hd)
            for row, s, e in rows_spans:
                if source is None:
                    out[row, s:e, sl] = 0.0
                else:
                    out[row, s:e, sl] = source[layer][row, s:e, sl]
            return out
        return hook

    handles = [lins[L].register_forward_hook(make(L)) for L in layers]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _enc(tk, texts):
    return tk([t.replace(MASK, tk.mask_token) for t in texts],
              return_tensors="pt", padding=True).to(DEVICE)


# ---------------------------------------------------------------- windows

WINDOWS = ([(f"L{l}", [l]) for l in range(N_LAYERS)]
           + [(f"L{k}-{k + 3}", list(range(k, k + 4))) for k in range(0, N_LAYERS - 3)]
           + [("L5-11", list(range(5, 12)))])


@torch.no_grad()
def run_windows(mname, items):
    tk, model = load_encoder(mname)
    cache = ValueCache(model)
    rows = []
    for bi, batch in enumerate(_batches(items)):
        cue = CUE_OF[batch[0]["condition"]]
        meta = _meta(batch, mname)
        enc = _enc(tk, [it["clean"] for it in batch])
        enc_c = _enc(tk, [it["corrupted"][cue] for it in batch])
        assert enc_c["input_ids"].shape == enc["input_ids"].shape
        with cache.capture():
            model(**enc_c)
        src = dict(cache.values)
        cue_spans = [(b, *it["models"][mname]["spans"][cue]) for b, it in enumerate(batch)]
        ctrl_spans = []
        for b, it in enumerate(batch):
            words = it["clean"].split()
            sp, _ = word_spans(tk, words)
            ctrl_spans.append((b, *sp[len(words) - 1]))      # object complement (last word)
        d_clean = _margin(model(**enc).logits, meta)
        res = {}
        for wname, layers in WINDOWS:
            for kind, spans, source in (("zero", cue_spans, None), ("patch", cue_spans, src),
                                        ("ctrl", ctrl_spans, None)):
                with _value_edit(model, layers, spans, source):
                    res[(wname, kind)] = _margin(model(**enc).logits, meta)
        for b, it in enumerate(batch):
            for (wname, kind), d in res.items():
                rows.append({"model": mname, "condition": it["condition"], "id": it["id"],
                             "gold": it["gold"], "subj": it["lex"]["subj_masc"],
                             "frame": it["lex"]["verb_masc"], "window": wname,
                             "kind": kind, "delta_clean": d_clean[b], "delta_int": d[b]})
        if bi % 20 == 0:
            print(f"  windows {mname}: {bi * BATCH}/{len(items)}")
    del model
    return rows


def stage_windows():
    items = [it for it in load_items() if it["condition"] in CUE_OF]
    for m in ENCODERS4:
        mitems = sorted([it for it in items if m in it["models"]],
                        key=lambda it: (it["condition"], it["models"][m]["n_tokens"]))
        rows = []
        # batches must not mix conditions (different interfering cue)
        for cond in CUE_OF:
            rows.extend(run_windows(m, [it for it in mitems if it["condition"] == cond]))
        pd.DataFrame(rows).to_csv(os.path.join(DATA, f"rev_windows_{m}.csv"), index=False)


# ---------------------------------------------------------------- residual

@contextmanager
def _resid_edit(model, layer, rows_pos, source):
    """Overwrite hidden states at (row, s, e) of the output of ``layer``
    (-1: embedding output) with ``source``."""
    mod = model.base_model.embeddings if layer < 0 else model.base_model.encoder.layer[layer]

    def hook(module, inputs, output):
        hs = output if torch.is_tensor(output) else output[0]
        hs = hs.clone()
        for row, s, e in rows_pos:
            hs[row, s:e] = source[row, s:e]
        if torch.is_tensor(output):
            return hs
        return (hs,) + tuple(output[1:])

    h = mod.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@torch.no_grad()
def run_residual(mname, items):
    tk, model = load_encoder(mname)
    rows = []
    for bi, batch in enumerate(_batches(items)):
        meta = _meta(batch, mname)
        enc = _enc(tk, [it["clean"] for it in batch])
        enc_c = _enc(tk, [it["corrupted"]["subject"] for it in batch])
        assert enc_c["input_ids"].shape == enc["input_ids"].shape
        out_c = model(**enc_c, output_hidden_states=True)
        hs_c = out_c.hidden_states          # (emb, L0..L11)
        d_corr = _margin(out_c.logits, meta)
        d_clean = _margin(model(**enc).logits, meta)
        subj = [(b, *it["models"][mname]["spans"]["subject"]) for b, it in enumerate(batch)]
        mpos = [(b, mp, mp + 1) for b, (mp, _, _) in enumerate(meta)]
        for layer in range(-1, N_LAYERS):
            for site, pos in (("subject", subj), ("mask", mpos)):
                with _resid_edit(model, layer, pos, hs_c[layer + 1]):
                    d = _margin(model(**enc).logits, meta)
                for b, it in enumerate(batch):
                    rows.append({"model": mname, "condition": it["condition"], "id": it["id"],
                                 "gold": it["gold"], "subj": it["lex"]["subj_masc"],
                                 "frame": it["lex"]["verb_masc"], "site": site, "layer": layer,
                                 "delta_clean": d_clean[b], "delta_corr": d_corr[b],
                                 "delta_patched": d[b]})
        if bi % 20 == 0:
            print(f"  residual {mname}: {bi * BATCH}/{len(items)}")
    del model
    return rows


def stage_residual():
    items = [it for it in load_items() if it["condition"] in ("T1", "T4", "T6")]
    for m in ENCODERS4:
        mitems = [it for it in items if m in it["models"]]
        pd.DataFrame(run_residual(m, mitems)).to_csv(
            os.path.join(DATA, f"rev_residual_{m}.csv"), index=False)


# ---------------------------------------------------------------- heads

@torch.no_grad()
def run_heads(mname, items, cue, layers):
    tk, model = load_encoder(mname)
    cache = ValueCache(model)
    n_heads = model.config.num_attention_heads
    rows = []
    for batch in _batches(items):
        meta = _meta(batch, mname)
        enc = _enc(tk, [it["clean"] for it in batch])
        enc_c = _enc(tk, [it["corrupted"][cue] for it in batch])
        with cache.capture():
            model(**enc_c)
        src = dict(cache.values)
        spans = [(b, *it["models"][mname]["spans"][cue]) for b, it in enumerate(batch)]
        d_clean = _margin(model(**enc).logits, meta)
        for L in layers:
            for h in range(n_heads):
                with _value_edit(model, [L], spans, src, head=h):
                    d = _margin(model(**enc).logits, meta)
                for b, it in enumerate(batch):
                    rows.append({"model": mname, "condition": it["condition"], "id": it["id"],
                                 "gold": it["gold"], "cue": cue, "layer": L, "head": h,
                                 "score": d_clean[b] - d[b], "correct_clean": d_clean[b] > 0})
    del model
    return rows


def stage_heads():
    items = load_items()
    for m in ENCODERS4:
        rows = []
        for cond, cue in (("T1", "subject"), ("T6", "attractor")):
            mitems = [it for it in items if it["condition"] == cond and m in it["models"]]
            rows.extend(run_heads(m, mitems, cue, range(N_LAYERS)))
        pd.DataFrame(rows).to_csv(os.path.join(DATA, f"rev_heads_{m}.csv"), index=False)
        print(f"  heads {m} done")


STAGES = {"annotate": stage_annotate, "marbert": stage_marbert, "windows": stage_windows,
          "residual": stage_residual, "heads": stage_heads}


def main():
    sel = [s.strip() for s in os.environ.get("CUEWIN_STAGES", ",".join(STAGES)).split(",")
           if s.strip()]
    print(f"stages: {sel}   device: {DEVICE}")
    for s in sel:
        print(f"\n=== {s} ===")
        STAGES[s]()


if __name__ == "__main__":
    main()
