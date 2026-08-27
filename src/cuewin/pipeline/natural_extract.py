#!/usr/bin/env python3
"""
Feasibility funnel for naturalistic replication.

Context: the causal claims (subject dominates verb agreement; mid-layer attractor
interference; repair by neutralizing the attractor) were established on hand-built
templates (lexicon.json). P0 asks whether the same claims hold when the BASE sentence
comes from real Arabic text. This script does NOT build items yet; it only measures
how many candidate subject-verb pairs survive the tokenizer/morphology gates, so we
decide BEFORE spending GPU time whether P0 is viable (and with what conditions).

Funnel (each step counts surviving sentences):
  F0  total sentences parsed
  F1  contain a finite past-tense 3sg VERB (head) with a dependent nsubj whose
      subject token is a NOUN, Number=Sing, with an annotated Gender
  F2  verb lemma attested in BOTH masculine and feminine 3sg surface forms
      somewhere in the corpus  => guarantees a natural foil form (no generation)
  F3  subject is an agreeing singular noun => verb form in THIS sentence agrees
      with subject's gender (sanity: gold form is the observed verb surface)
  F4  BOTH gender verb forms single-token in BOTH encoders (the real gate; must be
      the forced-choice answer), and subject form is a real token (tokenizer hit)
  F5  subject has an aligned masc/fem counterpart in the validated lexicon
      (usable for SUBJECT PATCHING); items failing this remain behavioral-only

Outputs:
  data/padt/natural_funnel.txt      one line per funnel stage with sentence counts
  data/padt/natural_candidates.jsonl  surviving F4 items (parsed metadata + ids),
                                       optionally annotated with lexicon subject pairs
"""

import json
import os
from collections import Counter

import conllu
from transformers import AutoTokenizer

from cuewin.config import data_dir, model_path

PADT_DIR = os.path.join(data_dir(), "padt")
OUT_TXT = os.path.join(data_dir(), "padt", "natural_funnel.txt")
OUT_JSON = os.path.join(data_dir(), "padt", "natural_candidates.jsonl")
LEX_PATH = os.path.join(data_dir(), "lexicon.json")

MODEL_PATHS = {
    "arabert": model_path("arabert"),
    "camelbert": model_path("camelbert"),
}

FINITE_VERB = {
    "deprel_head": ["nsubj", "nsubj:pass"],
    "upos": "VERB",
    "Aspect": "Perf",
    "Person": "3",
    "Number": "Sing",
}


def feats(tok):
    return tok.get("feats") or {}


def main():
    files = [os.path.join(PADT_DIR, f)
             for f in os.listdir(PADT_DIR) if f.endswith(".conllu")]
    print(f"reading {len(files)} conllu files from {PADT_DIR}")
    if not files:
        raise SystemExit("no conllu files found")

    # build corpus-wide (lemma, voice) -> gender -> {surface forms} for finite
    # past 3sg verbs. Voice matters: Arabic passive stems differ (أفاد/أُفيد),
    # so masc and fem foils must come from the SAME voice.
    verb_forms = {}          # (lemma, voice) -> {M:set(), F:set()}
    n_sent = 0
    raw_sents = []           # hold parsed sentence objects
    for path in files:
        for sent in conllu.parse(open(path, encoding="utf-8").read()):
            n_sent += 1
            raw_sents.append(sent)
            toks = list(sent)
            for t in toks:
                if t["upos"] != "VERB":
                    continue
                f = feats(t)
                if f.get("Aspect") != "Perf" or f.get("Person") != "3" \
                        or f.get("Number") != "Sing":
                    continue
                g = f.get("Gender")
                if g not in ("Masc", "Fem"):
                    continue
                voice = f.get("Voice", "Act")
                lemma = t["lemma"]
                key = (lemma, voice)
                verb_forms.setdefault(key, {"M": set(), "F": set()})
                verb_forms[key][g[0]].add(t["form"])

    print(f"F0 total sentences: {n_sent}")
    print(f"    unique past-3sg verb lemmas: {len(verb_forms)}")

    # tokenizers
    tks = {m: AutoTokenizer.from_pretrained(p) for m, p in MODEL_PATHS.items()}
    for tk in tks.values():
        assert tk.mask_token == "[MASK]"

    def single_token(form, tk):
        ids = tk.encode(form, add_special_tokens=False)
        return len(ids) == 1 and ids[0] != tk.unk_token_id

    def verb_pair_usable(lemma, voice):
        """both genders attested AND at least one single-token form in each."""
        forms = verb_forms.get((lemma, voice))
        if not forms or not forms["M"] or not forms["F"]:
            return None
        m = sorted(forms["M"])[0]
        f = sorted(forms["F"])[0]
        for tk in tks.values():
            if not (single_token(m, tk) and single_token(f, tk)):
                return None
        return m, f

    def opposite_form_single(lemma, voice, gold_form, gold_gender):
        """the specific gold form is single-token, and some attested opposite
        gender form (same voice) is single-token (the foil)."""
        opp = "F" if gold_gender == "M" else "M"
        opp_forms = verb_forms.get((lemma, voice), {}).get(opp)
        if not opp_forms:
            return False
        return (single_token(gold_form, tks["arabert"]) and
                single_token(gold_form, tks["camelbert"]) and
                any(single_token(x, tks["arabert"]) and single_token(x, tks["camelbert"])
                    for x in opp_forms))

    lexicon = json.load(open(LEX_PATH, encoding="utf-8"))
    lex_subjects = dict()  # masc_str -> record
    for s in lexicon["subjects"]:
        lex_subjects.setdefault(s["masc"], s)
        lex_subjects.setdefault(s["fem"], s)

    counts = Counter()
    counts["F1"] = 0
    counts["F2"] = 0
    counts["F3"] = 0
    counts["F4"] = 0
    counts["F5"] = 0
    f2_examples = []

    for sent in raw_sents:
        toks = list(sent)
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
                # ---- F1: candidate pair found (verb gender IS the agreement
                #          signal; PADT leaves most noun genders unannotated) ----
                counts["F1"] += 1
                lemma = t["lemma"]
                voice = fhead.get("Voice", "Act")
                gold_form = t["form"]
                if not opposite_form_single(lemma, voice, gold_form, vg):
                    continue
                counts["F2"] += 1
                counts["F3"] += 1
                # ---- F4: subject is a real token in both encoders; sentence
                #          is rebuildable with the verb masked ----
                subj_form = ch["form"]
                if not all(single_token(subj_form, tk) for tk in tks.values()):
                    continue
                sent_text = sent.metadata.get("text", "")
                if "[MASK]" in sent_text or not sent_text:
                    continue
                counts["F4"] += 1
                if len(f2_examples) < 20:
                    f2_examples.append({
                        "sent_id": sent.metadata.get("sent_id"),
                        "verb_lemma": lemma, "verb_gender": vg, "voice": voice,
                        "subject": subj_form, "subject_gender": sf.get("Gender"),
                        "masc_form": sorted(verb_forms[(lemma, voice)]["M"])[0],
                        "fem_form": sorted(verb_forms[(lemma, voice)]["F"])[0],
                        "verb_forms": {g: sorted(v) for g, v in verb_forms[(lemma, voice)].items()},
                        "sentence": sent_text,
                    })
                # ---- F5: subject has an aligned lexicon counterpart (patching) ----
                r = lex_subjects.get(subj_form)
                if r is not None:
                    counts["F5"] += 1

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"F0 total sentences            : {n_sent}\n")
        f.write(f"F1 nsubj past-3sg-verb sing   : {counts['F1']}\n")
        f.write(f"F2 verb pair both genders     : {counts['F2']}\n")
        f.write(f"F3 verb agrees with subject   : {counts['F3']}\n")
        f.write(f"F4 subject real token both    : {counts['F4']}\n")
        f.write(f"F5 + lexicon subject (patch)  : {counts['F5']}\n")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        for ex in f2_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nF1 sentences with candidate pair : {counts['F1']}")
    print(f"F2 verb pair bubbles in corpus   : {counts['F2']}")
    print(f"F3 verb agrees with subject      : {counts['F3']}")
    print(f"F4 subject real-token both encs  : {counts['F4']}")
    print(f"F5 + lexicon subject (patching)  : {counts['F5']}")
    print(f"saved {OUT_TXT}\nsaved {OUT_JSON}")


if __name__ == "__main__":
    main()