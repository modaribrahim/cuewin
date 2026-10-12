#!/usr/bin/env python3
"""
ADDITIVE annotation of the frozen benchmark for ARBERT.

The frozen families.jsonl is a validated, shared benchmark built for
arabert + camelbert (group == BOTH). We must NOT regenerate it: doing so with
ARBERT in MODELS would make annotate_and_assert drop items that fail ARBERT
(e.g. the المتطوعة subject) for ALL models, shrinking the shared set and
changing already-validated numbers.

Instead this script reads families.jsonl and ADDS an "arbert" annotation to
each item where ARBERT's hard alignment assertions pass (same checks as the
families stage: assertion 1 equal clean/corrupted length, assertion 2
identical outside flipped span, assertion 3 single-token verb forms,
assertion 4 mask is one token). Items where ARBERT fails are left without an
arbert annotation so run scripts can skip them for ARBERT only.

Output: data/families_arbert.jsonl — identical to families.jsonl plus, for
supported items, models["arbert"]. The run scripts that iterate MODELS will
use this file and only score ARBERT on items that carry its annotation.
"""

import json
import os

from cuewin.families import arbert_annotation
from cuewin.config import data_dir
from cuewin.models import tokenizer

DATA = data_dir()
IN_PATH = os.path.join(DATA, "families.jsonl")
OUT_PATH = os.path.join(DATA, "families_arbert.jsonl")
NATURAL_FAMS = [("natural_families_T1.jsonl", "natural_families_T1_arbert.jsonl"),
                ("natural_families_T4.jsonl", "natural_families_T4_arbert.jsonl"),
                ("natural_families_T6.jsonl", "natural_families_T6_arbert.jsonl")]
MASK = "[MASK]"


def main():
    tk = tokenizer("arbert")
    assert tk.mask_token == MASK, "mask token mismatch"

    items = [json.loads(l) for l in open(IN_PATH, encoding="utf-8")]
    added, skipped = 0, 0
    for it in items:
        ann = arbert_annotation(it, tk)
        if ann is None:
            skipped += 1
            continue
        it.setdefault("models", {})["arbert"] = ann
        added += 1

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"items: {len(items)}")
    print(f"  arbert annotation added: {added}")
    print(f"  skipped (no arbert annotation): {skipped}")
    print(f"saved: {OUT_PATH}")

    for src, dst in NATURAL_FAMS:
        nitems = [json.loads(l) for l in open(os.path.join(DATA, src), encoding="utf-8")]
        na, ns = 0, 0
        for it in nitems:
            ann = arbert_annotation(it, tk)
            if ann is None:
                ns += 1
                continue
            it.setdefault("models", {})["arbert"] = ann
            na += 1
        with open(os.path.join(DATA, dst), "w", encoding="utf-8") as f:
            for it in nitems:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"  {src}: {na} added, {ns} skipped -> {dst}")


if __name__ == "__main__":
    main()
