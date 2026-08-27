#!/usr/bin/env python3
"""
Build naturalistic test items (P0 Phase 1).

Takes the F4-eligible subject-verb pairs from UD Arabic-PADT (see the natural
extraction funnel) and emits items in the SAME schema as data/families.jsonl, so the
existing behavioral / patching / repair / stats scripts run on them unchanged.

Two conditions (reusing the synthetic labels on purpose):
  T1-natural  <subj ...> [MASK] <rest>   gold = the observed verb form; foil = the
              opposite-gender form of the same lemma+voice (corpus-attested).
              Subject BEFORE verb (SVO-ish). Subject cue included ONLY when the
              subject form has a validated lexicon gender counterpart (patching).
  T6-natural  <subj ...> prep Attr [MASK] <rest>  opposite-gender attractor (from the
              lexicon, inserted right before the masked verb). Always patching-able.
  T4-natural  [MASK] <subj ...>          subject AFTER the verb (VSO). Reported
              separately as a bonus; subject is a post-mask cue by construction.

Reconstruction: sentence rebuilt from conllu surface forms, honoring SpaceAfter=No,
so the token sequence the model sees is exactly the conllu token sequence and every
word-level span is derivable by tokenization. Gold verb gender IS the subject gender
(PADT leaves most noun genders unannotated; agreement is the ground truth).

Hard assertions per item per model (mirror the families stage):
  1. clean and every corrupted variant tokenize to the SAME total length
  2. outside the flipped cue's span, token ids are IDENTICAL
  3. both verb forms (gold/foil) are single tokens
  4. the mask slot is exactly one token position
Items failing any assertion are dropped and counted.

Outputs:
  data/natural_families_T1.jsonl
  data/natural_families_T4.jsonl
  data/natural_families_T6.jsonl
  data/padt/natural_families_report.txt
"""

import json
import os
from collections import Counter

import conllu
from transformers import AutoTokenizer

from cuewin.config import data_dir, model_path

PADT_DIR = os.path.join(data_dir(), "padt")
LEX_PATH = os.path.join(data_dir(), "lexicon.json")
OUT_REPORT = os.path.join(data_dir(), "padt", "natural_families_report.txt")

MODEL_PATHS = {
    "arabert": model_path("arabert"),
    "camelbert": model_path("camelbert"),
}

MASK = "[MASK]"
MAX_TOKENS = 256        # keep items focused; drop longer natural sentences


def feats(tok):
    return tok.get("feats") or {}


def rebuild_sentence(toks):
    """Join conllu surface forms honoring SpaceAfter=No. Returns the string and a
    parallel list of word forms (== conllu forms, one per word in order)."""
    out, words = [], []
    for i, t in enumerate(toks):
        form = t["form"]
        words.append(form)
        out.append(form)
        misc = t.get("misc") or {}
        if i < len(toks) - 1 and misc.get("SpaceAfter") == "No":
            out.append("")          # no space after this word
        else:
            out.append(" ")
    text = "".join(out).rstrip()
    return text, words


def load_lexicon():
    lex = json.load(open(LEX_PATH, encoding="utf-8"))
    by_form = {}
    for s in lex["subjects"]:
        by_form[s["masc"]] = s
        by_form[s["fem"]] = s
    return by_form, lex


def pick_foil(forms, gold_form, gold_gender, tks):
    """An attested opposite-gender form, single-token in both encoders."""
    opp = "F" if gold_gender == "M" else "M"
    for x in sorted(forms.get(opp, ())):
        if x == gold_form:
            continue
        if all(len(tk.encode(x, add_special_tokens=False)) == 1 for tk in tks.values()):
            return x
    return None


def annotate_and_assert(item, tks):
    """Mirror of 02_build_families.annotate_and_assert, but cue spans are given as
    word-level index ranges by the caller (word_indices), not string search."""
    item["models"] = {}
    words = item["_words"]
    for mname, tk in tks.items():
        spans, n_tokens = [], 1
        for w in words:
            if w == MASK:
                n_tok = 1
            else:
                n_tok = len(tk.encode(w, add_special_tokens=False))
                if n_tok == 0:
                    return None
            spans.append((n_tokens, n_tokens + n_tok))
            n_tokens += n_tok
        n_tokens += 1                       # [SEP]

        mask_word_idx = words.index(MASK)
        mask_pos = spans[mask_word_idx][0]

        ids_clean = tk.encode(item["clean"].replace(MASK, tk.mask_token))
        if len(ids_clean) != n_tokens:      # assertion 4 / span bug
            return None
        if ids_clean[mask_pos] != tk.mask_token_id:
            return None

        cue_spans = {}
        for cue, (w_start, w_end) in item["_cue_words"].items():
            s = spans[w_start][0]
            e = spans[w_end - 1][1]
            cue_spans[cue] = [s, e]

        for cue, corr in item["corrupted"].items():
            ids_corr = tk.encode(corr.replace(MASK, tk.mask_token))
            if len(ids_corr) != len(ids_clean):      # assertion 1
                return None
            s, e = cue_spans[cue]
            if not (ids_corr[:s] == ids_clean[:s] and ids_corr[e:] == ids_clean[e:]):
                return None                          # assertion 2

        g = tk.encode(item["gold_form"], add_special_tokens=False)
        f = tk.encode(item["foil_form"], add_special_tokens=False)
        if len(g) != 1 or len(f) != 1:               # assertion 3
            return None

        item["models"][mname] = {
            "mask_pos": mask_pos, "n_tokens": len(ids_clean),
            "spans": cue_spans,
            "target_ids": {"gold": g[0], "foil": f[0]},
        }
    return item


def main():
    tks = {m: AutoTokenizer.from_pretrained(p) for m, p in MODEL_PATHS.items()}
    for tk in tks.values():
        assert tk.mask_token == MASK

    lex_subjects, lex = load_lexicon()
    files = [os.path.join(PADT_DIR, f) for f in os.listdir(PADT_DIR)
             if f.endswith(".conllu")]

    # corpus-wide (lemma, voice) -> gender -> forms (same as 20)
    verb_forms = {}
    n_sent = 0
    for path in files:
        for sent in conllu.parse(open(path, encoding="utf-8").read()):
            n_sent += 1
            for t in sent:
                if t["upos"] != "VERB":
                    continue
                f = feats(t)
                if f.get("Aspect") != "Perf" or f.get("Person") != "3" \
                        or f.get("Number") != "Sing":
                    continue
                g = f.get("Gender")
                if g not in ("Masc", "Fem"):
                    continue
                verb_forms.setdefault((t["lemma"], f.get("Voice", "Act")),
                                      {"M": set(), "F": set()})[g[0]].add(t["form"])

    items = {"T1": [], "T4": [], "T6": []}
    counts = Counter()

    for path in files:
        for sent in conllu.parse(open(path, encoding="utf-8").read()):
            toks = list(sent)
            text, words = rebuild_sentence(toks)
            if not text or "[MASK]" in text:
                continue
            word_idx = {t["id"]: i for i, t in enumerate(toks)}
            for t in toks:
                if t["upos"] != "VERB":
                    continue
                fhead = feats(t)
                if fhead.get("Aspect") != "Perf" or fhead.get("Person") != "3" \
                        or fhead.get("Number") != "Sing":
                    continue
                vg = fhead.get("Gender")
                if vg not in ("Masc", "Fem"):
                    continue
                for ch in toks:
                    if ch.get("head") != t["id"]:
                        continue
                    if ch["deprel"] not in ("nsubj", "nsubj:pass"):
                        continue
                    if ch["upos"] != "NOUN":
                        continue
                    sf = feats(ch)
                    if sf.get("Number") not in ("Sing", None):
                        continue
                    counts["candidates"] += 1
                    vi = word_idx[t["id"]]
                    si = word_idx[ch["id"]]
                    lemma, voice = t["lemma"], fhead.get("Voice", "Act")
                    forms = verb_forms.get((lemma, voice), {})
                    gold_form = t["form"]
                    if gold_form not in forms.get(vg[0], ()):
                        continue
                    foil = pick_foil(forms, gold_form, vg[0], tks)
                    if foil is None:
                        continue
                    counts["verb_pair_ok"] += 1
                    # subject single-token in both encoders
                    subj_form = ch["form"]
                    if not all(len(tk.encode(subj_form, add_special_tokens=False)) == 1
                               for tk in tks.values()):
                        continue
                    counts["subject_tok_ok"] += 1

                    # ---- BARE-SUBJECT GATE (for patching validity): the subject
                    # must not carry an agreeing modifier, otherwise flipping its
                    # gender produces an ungrammatical NP (e.g. الرئيس الفلسطيني →
                    # *الرئيسة الفلسطيني). We check conllu: no dependent with an
                    # agreement-bearing relation directly under the subject.
                    # Items failing this stay behavioral-only (no subject cue). ----
                    subj_dependents = [d for d in toks if d.get("head") == ch["id"]]
                    bare_subject = not any(
                        d["deprel"] in ("amod", "nmod", "nummod", "appos", "acl",
                                        "fixed", "flat") for d in subj_dependents)

                    # word order: T1 (SVO-ish, subject before verb) vs T4 (VSO)
                    cond = "T1" if si < vi else "T4"
                    words_clean = list(words)
                    words_clean[vi] = MASK
                    if len(words_clean) > MAX_TOKENS:
                        counts["too_long"] += 1
                        continue
                    clean = " ".join(words_clean)

                    gold = "m" if vg == "Masc" else "f"

                    # subject cue only if a validated lexicon counterpart exists.
                    # NOTE: the corrupted run is a CAUSAL PROBE (its value vectors
                    # are patched into the clean run); as in synthetic T2/T3 it need
                    # NOT be grammatical — a flipped subject may clash with an
                    # agreeing adjective. Token alignment (asserted below) is the
                    # only validity requirement, consistent with value patching.
                    subj_pair = lex_subjects.get(subj_form)
                    cues, corrupted = {}, {}
                    cue_words = {}
                    if subj_pair is not None:
                        flipped = (subj_pair["fem"] if gold == "m"
                                   else subj_pair["masc"])
                        if flipped != subj_form:
                            cues["subject"] = {
                                "clean_word": subj_form, "flipped_word": flipped}
                            words_corr = list(words_clean)
                            words_corr[si] = flipped
                            corrupted["subject"] = " ".join(words_corr)
                            cue_words["subject"] = (si, si + 1)
                    counts["bare_subject"] += int(bare_subject)

                    item = {
                        "condition": cond, "gold": gold, "clean": clean,
                        "gold_form": gold_form, "foil_form": foil,
                        "cues": cues, "corrupted": corrupted,
                        "lex": {
                            "subj_masc": subj_form, "subj_gloss": "",
                            "subj_kind": "natural",
                            "verb_masc": gold_form,
                            "verb_gloss": f"{lemma} ({voice})",
                            "adj_masc": None, "attr_masc": None, "attr_prep": None,
                            "modifier": None, "modifier_kind": None,
                            "modifier_noun_gender": None,
                            "natural_sent_id": sent.metadata.get("sent_id", ""),
                        },
                        "_words": words_clean, "_cue_words": cue_words,
                        "mask_verb_word": vi, "subject_word": si,
                    }
                    item = annotate_and_assert(item, tks)
                    if item is None:
                        counts["assert_fail"] += 1
                        continue
                    del item["_words"], item["_cue_words"]
                    item["id"] = f"N{cond}-{gold}-{counts[f'N{cond}_made']:04d}"
                    counts[f"N{cond}_made"] += 1
                    items[cond].append(item)

                    # ---- T6 variant (SVO only): insert opposite-gender attractor
                    #          between subject and verb. For VSO there IS no slot
                    #          between them, so T6-natural is restricted to the
                    #          subject-before-verb items. ----
                    if cond == "T4":
                        continue
                    gold_flip = "m" if gold == "f" else "f"
                    attr = lex["attractors"][0]      # pick by gender below
                    matches = [a for a in lex["attractors"]
                               if (a["masc"] if gold_flip == "m" else a["fem"])
                               != subj_form]
                    attr = matches[0] if matches else lex["attractors"][0]
                    a_form = attr["fem"] if gold_flip == "f" else attr["masc"]
                    prep = attr.get("prep", "مع")
                    words_t6 = list(words_clean)
                    words_t6.insert(vi, a_form)
                    words_t6.insert(vi, prep)
                    if len(words_t6) > MAX_TOKENS:
                        counts["t6_too_long"] += 1
                        continue
                    clean6 = " ".join(words_t6)
                    a_flip = attr["masc"] if gold_flip == "f" else attr["fem"]
                    cues6 = {"attractor": {"clean_word": a_form, "flipped_word": a_flip}}
                    words_corr6 = list(words_t6)
                    words_corr6[vi + 1] = a_flip      # attractor at vi+1 (prep at vi)
                    corrupted6 = {"attractor": " ".join(words_corr6)}
                    item6 = {
                        "condition": "T6", "gold": gold, "clean": clean6,
                        "gold_form": gold_form, "foil_form": foil,
                        "cues": cues6, "corrupted": corrupted6,
                        "lex": {
                            "subj_masc": subj_form, "subj_gloss": "",
                            "subj_kind": "natural",
                            "verb_masc": gold_form,
                            "verb_gloss": f"{lemma} ({voice})",
                            "adj_masc": None, "attr_masc": attr["masc"],
                            "attr_prep": prep,
                            "modifier": None, "modifier_kind": None,
                            "modifier_noun_gender": None,
                            "natural_sent_id": sent.metadata.get("sent_id", ""),
                        },
                        "_words": words_t6,
                        "_cue_words": {"attractor": (vi + 1, vi + 2)},
                        "mask_verb_word": vi + 2, "subject_word": si,
                    }
                    item6 = annotate_and_assert(item6, tks)
                    if item6 is None:
                        counts["t6_assert_fail"] += 1
                        continue
                    del item6["_words"], item6["_cue_words"]
                    item6["id"] = f"NT6-{gold}-{counts['NT6_made']:04d}"
                    counts["NT6_made"] += 1
                    items["T6"].append(item6)

    # ---- write outputs ----
    rel = lambda p: os.path.relpath(p, os.path.dirname(OUT_REPORT))
    paths = {
        "T1": os.path.join(data_dir(), "natural_families_T1.jsonl"),
        "T4": os.path.join(data_dir(), "natural_families_T4.jsonl"),
        "T6": os.path.join(data_dir(), "natural_families_T6.jsonl"),
    }
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("natural families build report\n==============================\n")
        f.write(f"sentences scanned     : {n_sent}\n")
        for k in ("candidates", "verb_pair_ok", "subject_tok_ok", "too_long",
                  "assert_fail", "t6_too_long", "t6_assert_fail", "bare_subject"):
            f.write(f"{k:<20}: {counts[k]}\n")
        for cond, path in paths.items():
            f.write(f"{cond:<3} items -> {rel(path)} : {len(items[cond])}\n")
            with open(path, "w", encoding="utf-8") as out:
                for it in items[cond]:
                    out.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(open(OUT_REPORT, encoding="utf-8").read())
    for cond, path in paths.items():
        print(f"{cond:<3} items: {len(items[cond])}  ->  {path}")
    # gender balance
    for cond, lst in items.items():
        g = Counter(i["gold"] for i in lst)
        print(f"  {cond}: masc={g['m']} fem={g['f']}")


if __name__ == "__main__":
    main()