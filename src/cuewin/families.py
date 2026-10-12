"""Shared item-family helpers: token-span location and the 1:1 alignment
assertions used across the lexicon, template and natural pipelines.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from cuewin.config import data_dir

MASK = "[MASK]"


def word_spans(tk, words):
    """Token span [start, end) of each word, with [CLS] at position 0."""
    spans, pos = [], 1  # 1 = after [CLS]
    for w in words:
        n = (1 if w == MASK else len(tk.encode(w, add_special_tokens=False)))
        spans.append((pos, pos + n))
        pos += n
    return spans, pos + 1  # +1 for [SEP]


def annotate_and_assert(item, tks):
    """Add per-model spans/ids; return None if any assertion fails."""
    item["models"] = {}
    clean_words = item["clean"].split()
    for mname, tk in tks.items():
        # word-level spans on the clean sentence
        spans, n_tokens = word_spans(tk, clean_words)
        mask_word_idx = clean_words.index(MASK)
        mask_pos = spans[mask_word_idx][0]

        # full-sentence encodings
        ids_clean = tk.encode(item["clean"].replace(MASK, tk.mask_token))
        if len(ids_clean) != n_tokens:
            return None  # assertion 4 / span bug
        if ids_clean[mask_pos] != tk.mask_token_id:
            return None

        # cue spans by word identity
        cue_spans = {}
        for cue, d in item["cues"].items():
            w_idx = clean_words.index(d["clean_word"])
            cue_spans[cue] = list(spans[w_idx])

        # corrupted variants: equal length, identical outside flipped span
        for cue, corr in item["corrupted"].items():
            ids_corr = tk.encode(corr.replace(MASK, tk.mask_token))
            if len(ids_corr) != len(ids_clean):  # assertion 1
                return None
            s, e = cue_spans[cue]
            outside_same = (ids_corr[:s] == ids_clean[:s]
                            and ids_corr[e:] == ids_clean[e:])
            if not outside_same:  # assertion 2
                return None

        # targets single-token  # assertion 3
        gold_ids = tk.encode(item["gold_form"], add_special_tokens=False)
        foil_ids = tk.encode(item["foil_form"], add_special_tokens=False)
        if len(gold_ids) != 1 or len(foil_ids) != 1:
            return None

        item["models"][mname] = {
            "mask_pos": mask_pos, "n_tokens": len(ids_clean),
            "spans": cue_spans,
            "target_ids": {"gold": gold_ids[0], "foil": foil_ids[0]},
        }
    return item


def arbert_annotation(item, tk):
    """Return the ARBERT ``models["arbert"]`` annotation dict, or None if an
    assertion fails. Ported verbatim from ``02c_add_arbert.py`` (same four
    assertions as the base pipeline, run for a single ARBERT tokenizer)."""
    clean_words = item["clean"].split()
    spans, n_tokens = word_spans(tk, clean_words)
    mask_word_idx = clean_words.index(MASK)
    mask_pos = spans[mask_word_idx][0]

    ids_clean = tk.encode(item["clean"].replace(MASK, tk.mask_token))
    if len(ids_clean) != n_tokens:  # assertion 4 / span bug
        return None
    if ids_clean[mask_pos] != tk.mask_token_id:
        return None

    cue_spans = {}
    for cue, d in item["cues"].items():
        w_idx = clean_words.index(d["clean_word"])
        cue_spans[cue] = list(spans[w_idx])

    for cue, corr in item["corrupted"].items():
        ids_corr = tk.encode(corr.replace(MASK, tk.mask_token))
        if len(ids_corr) != len(ids_clean):  # assertion 1
            return None
        s, e = cue_spans[cue]
        if not (ids_corr[:s] == ids_clean[:s]
                and ids_corr[e:] == ids_clean[e:]):  # assertion 2
            return None

    gold_ids = tk.encode(item["gold_form"], add_special_tokens=False)
    foil_ids = tk.encode(item["foil_form"], add_special_tokens=False)
    if len(gold_ids) != 1 or len(foil_ids) != 1:  # assertion 3
        return None

    return {
        "mask_pos": mask_pos, "n_tokens": len(ids_clean),
        "spans": cue_spans,
        "target_ids": {"gold": gold_ids[0], "foil": foil_ids[0]},
    }


def diff_annotation(item, tk):
    """Locate each cue span by diffing clean vs corrupted word lists (they
    differ only at the flipped slot)."""
    clean_words = item["clean"].split()
    spans, n_tokens = word_spans(tk, clean_words)
    mask_word_idx = clean_words.index(MASK)
    mask_pos = spans[mask_word_idx][0]

    ids_clean = tk.encode(item["clean"].replace(MASK, tk.mask_token))
    if len(ids_clean) != n_tokens:
        return None
    if ids_clean[mask_pos] != tk.mask_token_id:
        return None

    cue_spans = {}
    for cue, d in item["cues"].items():
        if d is None:
            continue
        corr_words = item["corrupted"][cue].split()
        diff = [i for i, (a, b) in enumerate(zip(clean_words, corr_words))
                if a != b]
        if not diff:
            return None
        cue_spans[cue] = list(spans[diff[0]])

    for cue, corr in item["corrupted"].items():
        ids_corr = tk.encode(corr.replace(MASK, tk.mask_token))
        if len(ids_corr) != len(ids_clean):
            return None
        s, e = cue_spans[cue]
        if not (ids_corr[:s] == ids_clean[:s]
                and ids_corr[e:] == ids_clean[e:]):
            return None

    gold_ids = tk.encode(item["gold_form"], add_special_tokens=False)
    foil_ids = tk.encode(item["foil_form"], add_special_tokens=False)
    if len(gold_ids) != 1 or len(foil_ids) != 1:
        return None

    return {
        "mask_pos": mask_pos, "n_tokens": len(ids_clean),
        "spans": cue_spans,
        "target_ids": {"gold": gold_ids[0], "foil": foil_ids[0]},
    }


def load_families(path=None):
    """Load a families.jsonl into a list of dicts."""
    path = path or os.path.join(data_dir(), "families.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_lexicon(path=None):
    """Load lexicon.json into a dict."""
    path = path or os.path.join(data_dir(), "lexicon.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def batch_signature(item, models=None):
    """Batch signature: same word structure AND per-model token lengths
    (complements differ in word count, so n_words is part of the signature).
    Iterates only the models present in the item, so items that skipped an
    additive annotation (e.g. arbert) still batch cleanly for the others."""
    model_list = models or list(item["models"])
    return (item["condition"], len(item["clean"].split()),
            tuple(item["models"][m]["n_tokens"] for m in model_list))


def batches(items, models, batch_size):
    """Group items into batches sharing a batch signature, chunked to
    ``batch_size``."""
    groups = defaultdict(list)
    for it in items:
        sig = batch_signature(it, models)
        groups[sig].append(it)
    for group in groups.values():
        for i in range(0, len(group), batch_size):
            yield group[i:i + batch_size]
