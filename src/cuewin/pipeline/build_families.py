#!/usr/bin/env python3
"""
Generate the test items (sentence families) from the
frozen lexicon core (group == BOTH), with per-model span annotations and
hard alignment assertions.

Conditions (design.md §3.3):
  T1  S [MASK] C                      subject is the only gender cue
  T2  S Adj [MASK] C                  congruent adjective (two redundant cues)
  T3  S Adj* [MASK] C                 CONFLICTING adjective (Adj* = opposite
                                      gender; ungrammatical by design — the
                                      agreement-attraction diagnostic)
  T4  [MASK] S C                      VSO: cue AFTER the mask (encoder-only
                                      question; gender agreement obligatory)
  T5  S PP [MASK] C                   neutral PP increases subject-verb distance
  T6  S prep Attr* [MASK] C           opposite-gender attractor between S and V

Item semantics:
  gold gender ALWAYS follows the subject (syntactic head of agreement).
  Per-cue corruption: each item carries corrupted sentence(s), one per cue,
  in which ONLY that cue is gender-flipped. Patching cue j uses corrupted[j].

Hard assertions per item per model:
  1. clean and every corrupted variant tokenize to the SAME total length
  2. outside the flipped cue's span, token ids are IDENTICAL
  3. both verb forms are single tokens
  4. the mask slot is exactly one token position
Items failing any assertion are counted and dropped (expected: zero, because
the lexicon guarantees alignment — assertion failures mean a bug).

Sampling: seed=42, target N_PER_CELL items per (condition × gold gender),
round-robin over subjects, then frames, then adjectives/attractors/modifiers
so every lexical item appears as evenly as possible.

Output: data/families.jsonl — one JSON object per item:
{
  "id": "T2-f-0042",
  "condition": "T2", "gold": "f",
  "clean": "الممثلة الجديدة [MASK] الرسالة",
  "gold_form": "كتبت", "foil_form": "كتب",
  "cues": {
    "subject":   {"clean_word": "الممثلة", "flipped_word": "الممثل"},
    "adjective": {"clean_word": "الجديدة", "flipped_word": "الجديد"}
  },
  "corrupted": {"subject": "الممثل الجديدة [MASK] الرسالة",
                "adjective": "الممثلة الجديد [MASK] الرسالة"},
  "lex": {"subj_gloss": "actor", "verb_gloss": "wrote the letter",
          "subj_kind": "noun", "frame_idx": 0, ...},
  "models": {
    "arabert":   {"mask_pos": 3, "n_tokens": 7,
                  "spans": {"subject": [1,2], "adjective": [2,3]},
                  "target_ids": {"gold": 17402, "foil": 9011}},
    "camelbert": {...}
  }
}
Token positions include the leading [CLS] (position 0).
"""

import json
import os
import random
from collections import Counter, defaultdict

from cuewin.families import annotate_and_assert
from cuewin.config import data_dir
from cuewin.models import tokenizer

LEX_PATH = os.path.join(data_dir(), "lexicon.json")
OUT_PATH = os.path.join(data_dir(), "families.jsonl")

N_PER_CELL = 600        # items per (condition × gold gender)
SEED = 42
MASK = "[MASK]"         # same literal mask token in both tokenizers (asserted)

# ----------------------------------------------------------------------------
# lexicon loading
# ----------------------------------------------------------------------------

def load_core():
    lex = json.load(open(LEX_PATH, encoding="utf-8"))
    core = {
        # verbs: single-token everywhere + morph OK (tier A in legacy field)
        "frames": [r for r in lex["vp_frames"]
                   if r["group"] == "BOTH"
                   and r["tok"]["arabert"]["masc_ntok"] == 1
                   and r["tok"]["arabert"]["fem_ntok"] == 1
                   and r["tok"]["camelbert"]["masc_ntok"] == 1
                   and r["tok"]["camelbert"]["fem_ntok"] == 1
                   and r["morph_ok"]],
        "subjects": [r for r in lex["subjects"] if r["group"] == "BOTH"],
        "names": [r for r in lex["names"] if r["group"] == "BOTH"],
        "adjectives": [r for r in lex["adjectives"] if r["group"] == "BOTH"],
        "attractors": [r for r in lex["attractors"] if r["group"] == "BOTH"],
        "modifiers": lex["neutral_modifiers"],
        "preps": lex.get("attractor_preps", ["مع"]),
    }
    return core


# ----------------------------------------------------------------------------
# sentence assembly
# ----------------------------------------------------------------------------

def forms(pair, gender):
    """(form_for_gender, form_for_opposite) of a masc/fem pair record."""
    return (pair["fem"], pair["masc"]) if gender == "f" else (pair["masc"], pair["fem"])


def build_item(condition, gold, subj, frame, adj=None, attr=None, mod=None,
               prep=None):
    """Assemble clean sentence + per-cue corrupted variants (word level)."""
    s_form, s_flip = forms(subj, gold)
    v_gold, v_foil = forms(frame, gold)
    comp = frame["comp"]

    cues = {"subject": {"clean_word": s_form, "flipped_word": s_flip}}

    if condition == "T1":
        words = [s_form, MASK, comp]
    elif condition == "T2":
        a_form, a_flip = forms(adj, gold)
        cues["adjective"] = {"clean_word": a_form, "flipped_word": a_flip}
        words = [s_form, a_form, MASK, comp]
    elif condition == "T3":
        # conflicting adjective: surface form is the OPPOSITE of gold
        a_conflict, a_match = forms(adj, "m" if gold == "f" else "f")
        cues["adjective"] = {"clean_word": a_conflict, "flipped_word": a_match}
        words = [s_form, a_conflict, MASK, comp]
    elif condition == "T4":
        words = [MASK, s_form, comp]
    elif condition == "T5":
        words = [s_form, mod["text"], MASK, comp]
    elif condition == "T6":
        # attractor with gender OPPOSITE to the subject (incongruent);
        # prep drawn from the diversified pool (constant within item)
        t_form, t_flip = forms(attr, "m" if gold == "f" else "f")
        cues["attractor"] = {"clean_word": t_form, "flipped_word": t_flip}
        words = [s_form, prep or attr["prep"], t_form, MASK, comp]
    else:
        raise ValueError(condition)

    clean = " ".join(words)
    corrupted = {}
    for cue, d in cues.items():
        corrupted[cue] = clean.replace(d["clean_word"], d["flipped_word"], 1)

    return {
        "condition": condition, "gold": gold, "clean": clean,
        "gold_form": v_gold, "foil_form": v_foil,
        "cues": cues, "corrupted": corrupted,
        "lex": {
            "subj_masc": subj["masc"], "subj_gloss": subj.get("gloss", ""),
            "subj_kind": subj.get("kind", "noun"),
            "verb_masc": frame["masc"], "verb_gloss": frame.get("gloss", ""),
            "adj_masc": adj["masc"] if adj else None,
            "attr_masc": attr["masc"] if attr else None,
            "attr_prep": (prep or attr["prep"]) if attr else None,
            "modifier": mod["text"] if mod else None,
            "modifier_kind": mod.get("kind") if mod else None,
            "modifier_noun_gender": mod.get("noun_gender") if mod else None,
        },
    }


# ----------------------------------------------------------------------------
# balanced sampling
# ----------------------------------------------------------------------------

def round_robin_product(primary, *rest, rng):
    """Yield combinations cycling through `primary` so its items are used
    as evenly as possible; secondary lists are shuffled per cycle."""
    pools = [list(r) for r in rest]
    prim = list(primary)
    rng.shuffle(prim)
    idx = 0
    while True:
        subj = prim[idx % len(prim)]
        combo = [subj]
        for p in pools:
            combo.append(rng.choice(p))
        yield combo
        idx += 1


def main():
    rng = random.Random(SEED)
    core = load_core()
    tks = {n: tokenizer(n) for n in ("arabert", "camelbert")}
    for tk in tks.values():
        assert tk.mask_token == MASK, "mask token mismatch"

    # subjects: nouns + names (kind recorded); T2/T3 nouns only
    nouns = [dict(s, kind="noun") for s in core["subjects"]]
    names = [dict(s, kind="name") for s in core["names"]]
    all_subjects = nouns + names

    spec = {
        "T1": dict(subjects=all_subjects),
        "T2": dict(subjects=nouns, adjectives=core["adjectives"]),
        "T3": dict(subjects=nouns, adjectives=core["adjectives"]),
        "T4": dict(subjects=all_subjects),
        "T5": dict(subjects=all_subjects, modifiers=core["modifiers"]),
        "T6": dict(subjects=all_subjects, attractors=core["attractors"],
                   preps=core["preps"]),
    }

    items, n_dropped = [], 0
    counters = defaultdict(Counter)

    for cond, pools in spec.items():
        for gold in ("m", "f"):
            seen = set()
            made = 0
            gen = round_robin_product(
                pools["subjects"],
                core["frames"],
                *( [pools["adjectives"]] if "adjectives" in pools else [] ),
                *( [pools["attractors"]] if "attractors" in pools else [] ),
                *( [pools["modifiers"]] if "modifiers" in pools else [] ),
                rng=rng)
            attempts = 0
            while made < N_PER_CELL and attempts < N_PER_CELL * 60:
                attempts += 1
                combo = next(gen)
                subj, frame = combo[0], combo[1]
                adj = attr = mod = None
                k = 2
                if "adjectives" in pools:
                    adj = combo[k]; k += 1
                if "attractors" in pools:
                    attr = combo[k]; k += 1
                if "modifiers" in pools:
                    mod = combo[k]; k += 1
                # attractor must differ from subject
                if attr is not None and attr["masc"] == subj["masc"]:
                    continue
                # modifier must not share a content word with the complement
                # (avoids awkward repeats like "في المكتب ... ترك المكتب")
                if mod is not None:
                    mod_words = {w for w in mod["text"].split() if len(w) > 2}
                    comp_words = {w for w in frame["comp"].split() if len(w) > 2}
                    if mod_words & comp_words:
                        continue
                prep = rng.choice(pools["preps"]) if "attractors" in pools else None
                key = (subj["masc"], frame["masc"],
                       adj["masc"] if adj else None,
                       attr["masc"] if attr else None,
                       prep,
                       mod["text"] if mod else None)
                if key in seen:
                    continue
                seen.add(key)
                item = build_item(cond, gold, subj, frame,
                                  adj=adj, attr=attr, mod=mod, prep=prep)
                item = annotate_and_assert(item, tks)
                if item is None:
                    n_dropped += 1
                    continue
                item["id"] = f"{cond}-{gold}-{made:04d}"
                items.append(item)
                counters[cond][gold] += 1
                made += 1

    # ----------------------------------------------------------------- report
    print(f"items generated: {len(items)}   dropped by assertions: {n_dropped}")
    print(f"{'cond':<6} {'masc':>6} {'fem':>6}")
    for cond in spec:
        print(f"{cond:<6} {counters[cond]['m']:>6} {counters[cond]['f']:>6}")

    # lexical balance check (subjects in T1 as example)
    t1_subj = Counter(i["lex"]["subj_masc"] for i in items
                      if i["condition"] == "T1")
    mn, mx = min(t1_subj.values()), max(t1_subj.values())
    print(f"\nT1 subject usage min/max per pair: {mn}/{mx} "
          f"({len(t1_subj)} pairs used)")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"\nsaved: {OUT_PATH}")

    # show a few examples
    print("\nexamples:")
    for cond in spec:
        ex = next(i for i in items if i["condition"] == cond and i["gold"] == "f")
        print(f"\n[{ex['id']}] gold={ex['gold_form']} foil={ex['foil_form']}")
        print(f"  clean    : {ex['clean']}")
        for cue, corr in ex["corrupted"].items():
            print(f"  corr/{cue:<9}: {corr}")


if __name__ == "__main__":
    main()
