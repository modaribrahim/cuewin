"""Repair the T6-natural ATTRACTOR construction in the frozen natural family
files, in place, for all models.

The original bug: the corrupted T6 variant flipped the PREPOSITION slot instead of
the attractor noun, so the feminine attractor was present in BOTH clean and
corrupted. This repairs the corrupted sentence to flip the attractor noun in place
(the word immediately preceding ``[MASK]``), keeping the preposition, then
re-annotates all models with the same hard alignment assertions.

Output (overwrites with corrected data): ``data/natural_families_T6_arbert.jsonl``
"""

from __future__ import annotations

import json
import os

from transformers import AutoTokenizer

from cuewin.config import data_dir, model_path
from cuewin.constants import ENCODERS

MASK = "[MASK]"


def corrected_corrupted(item):
    """Flip the attractor noun in place (word before [MASK]), keep the prep."""
    cue = item["cues"].get("attractor")
    if cue is None:
        return None
    words = item["clean"].split()
    if MASK not in words:
        return None
    mi = words.index(MASK)
    if mi == 0:
        return None
    words[mi - 1] = cue["flipped_word"]
    return " ".join(words)


def annotation(item, tk):
    clean_words = item["clean"].split()
    if MASK not in clean_words:
        return None
    spans, _ = word_spans(tk, clean_words)
    mi = clean_words.index(MASK)
    mask_pos = spans[mi][0]

    ids_clean = tk.encode(item["clean"].replace(MASK, tk.mask_token))
    if len(ids_clean) != spans[-1][1] + 1:
        return None
    if ids_clean[mask_pos] != tk.mask_token_id:
        return None

    cue_spans = {}
    for cue, corr in item["corrupted"].items():
        corr_words = corr.split()
        diff = [j for j in range(min(len(clean_words), len(corr_words)))
                if clean_words[j] != corr_words[j]]
        if not diff:
            return None
        cue_spans[cue] = list(spans[diff[0]])
        ids_corr = tk.encode(corr.replace(MASK, tk.mask_token))
        if len(ids_corr) != len(ids_clean):
            return None
        s, e = cue_spans[cue]
        if not (ids_corr[:s] == ids_clean[:s] and ids_corr[e:] == ids_clean[e:]):
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


def word_spans(tk, words):
    spans, pos = [], 1
    for w in words:
        n = len(tk.encode(w, add_special_tokens=False))
        spans.append((pos, pos + n))
        pos += n
    return spans, pos + 1


def main():
    base = data_dir()
    tks = {}
    for m in ENCODERS:
        tks[m] = AutoTokenizer.from_pretrained(model_path(m))
    for tk in tks.values():
        assert tk.mask_token == MASK, "mask token mismatch"

    path = os.path.join(base, "natural_families_T6_arbert.jsonl")
    if not os.path.exists(path):
        print(f"SKIP (not found): {path}")
        return
    items = [json.loads(l) for l in open(path, encoding="utf-8")]
    kept = dropped = 0
    for it in items:
        new_corr = corrected_corrupted(it)
        if new_corr is None:
            dropped += 1
            continue
        it["corrupted"]["attractor"] = new_corr
        it["models"] = {}
        ok = True
        for mname, tk in tks.items():
            ann = annotation(it, tk)
            if ann is None:
                ok = False
                break
            it["models"][mname] = ann
        if not ok:
            dropped += 1
            continue
        kept += 1
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            if it.get("models"):
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{path}: kept {kept}, dropped {dropped}")


if __name__ == "__main__":
    main()
