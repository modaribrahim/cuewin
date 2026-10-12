#!/usr/bin/env python3
"""
Held-out replication set (revision 2). CPU only.

New cue items that are not used as adjectives or attractors in the main set,
plus a construct-state attractor condition:

  T2  S Adj [MASK] C              congruent adjective   (new adjectives)
  T3  S Adj* [MASK] C             conflicting adjective (new adjectives)
  T6  S prep Attr* [MASK] C       PP attractor          (new attractors)
  T7  S' Attr* [MASK] C           construct-state attractor: S' is the
                                  subject without the article (head of an
                                  idafa), Attr* the definite genitive
                                  (e.g. مترجمة الملك [MASK] الرسالة).
                                  The subject-plus-attractor string cannot
                                  form a complete nominal sentence.

Subjects, verb frames and preps are the main-set ones (group BOTH). Items
pass the same four alignment assertions as the main set for AraBERT and
CAMeLBERT (required); ARBERT, MARBERT and XLM-R annotations are added when
they pass. Ids carry the prefix ``R``.

Outputs: data/lexicon_rep.json, data/families_rep.jsonl
"""

import json
import os
import random
from collections import Counter, defaultdict

from cuewin.config import data_dir
from cuewin.families import annotate_and_assert, arbert_annotation
from cuewin.models import tokenizer
from cuewin.pipeline.build_families import load_core, round_robin_product

DATA = data_dir()
OUT_LEX = os.path.join(DATA, "lexicon_rep.json")
OUT_FAM = os.path.join(DATA, "families_rep.jsonl")
N_PER_CELL = 600
SEED = 4242
MASK = "[MASK]"
REQUIRED = ("arabert", "camelbert")
OPTIONAL = ("arbert", "marbert", "xlmr")

# New adjectives (none in the main lexicon). Both kinds apply to any human
# noun: 12 descriptive, 12 nationality (nisba) adjectives. Nisba forms whose
# feminine is also a common noun (العربية "Arabic", السعودية "Saudi Arabia")
# are left out.
ADJ_NEW = [
    ("الكريم", "generous"), ("الغني", "rich"), ("القوي", "strong"),
    ("الطيب", "kind"), ("الشريف", "honourable"), ("الجاد", "serious"),
    ("الغريب", "strange"), ("القصير", "short"), ("النحيف", "thin"),
    ("الناجح", "successful"), ("الأنيق", "elegant"), ("الكسول", "lazy"),
    ("المصري", "Egyptian"), ("السوري", "Syrian"), ("اللبناني", "Lebanese"),
    ("الأردني", "Jordanian"), ("المغربي", "Moroccan"), ("العراقي", "Iraqi"),
    ("التونسي", "Tunisian"), ("اليمني", "Yemeni"), ("الأمريكي", "American"),
    ("الفرنسي", "French"), ("الألماني", "German"), ("التركي", "Turkish"),
]

# New attractors: human nouns from the subject pool that are not main-set
# attractors, minus pairs whose feminine form is also a common inanimate
# noun (الحكمة "wisdom", المؤسسة "institution", الجراحة "surgery",
# الحلاقة "shaving", الصيدلية "pharmacy", المساعدة "help",
# المعالجة "processing"); plus the seven name pairs not used as attractors.
ATTR_NOUNS_NEW = ["المحرر", "المترجم", "المصور", "المضيف", "الشرطي", "المبرمج",
                  "الممثل", "المطرب", "الإعلامي", "المتطوع", "المتحدث", "المتخصص",
                  "المسؤول", "المستخدم", "المهاجر", "المفوض", "المؤيد", "الملاكم",
                  "المهرج", "الرئيس", "الأمير", "الملك"]
ATTR_NAMES_NEW = ["علي", "عمر", "حسن", "إبراهيم", "سعيد", "يوسف", "كريم"]
AMBIGUOUS_FEM = {"الحكم", "المؤسس", "الجراح", "الحلاق", "الصيدلي", "المساعد", "المعالج"}


def strip_article(w):
    return w[2:] if w.startswith("ال") else w


def ntok(tk, w):
    return len(tk.encode(w, add_special_tokens=False))


def aligned(tks, masc, fem):
    return all(ntok(tk, masc) == ntok(tk, fem) for tk in tks.values())


def forms(pair, gender):
    return (pair["fem"], pair["masc"]) if gender == "f" else (pair["masc"], pair["fem"])


def build_item(cond, gold, subj, frame, adj=None, attr=None, prep=None):
    other = "m" if gold == "f" else "f"
    v_gold, v_foil = forms(frame, gold)
    comp = frame["comp"]
    if cond == "T7":
        s_form, s_flip = forms(subj["construct"], gold)
    else:
        s_form, s_flip = forms(subj, gold)
    cues = {"subject": {"clean_word": s_form, "flipped_word": s_flip}}
    if cond == "T2":
        a, a_flip = forms(adj, gold)
        cues["adjective"] = {"clean_word": a, "flipped_word": a_flip}
        words = [s_form, a, MASK, comp]
    elif cond == "T3":
        a, a_flip = forms(adj, other)
        cues["adjective"] = {"clean_word": a, "flipped_word": a_flip}
        words = [s_form, a, MASK, comp]
    elif cond == "T6":
        t, t_flip = forms(attr, other)
        cues["attractor"] = {"clean_word": t, "flipped_word": t_flip}
        words = [s_form, prep, t, MASK, comp]
    elif cond == "T7":
        t, t_flip = forms(attr, other)
        cues["attractor"] = {"clean_word": t, "flipped_word": t_flip}
        words = [s_form, t, MASK, comp]
    else:
        raise ValueError(cond)
    clean_words = words
    corrupted = {}
    for cue, d in cues.items():
        i = clean_words.index(d["clean_word"])
        cw = list(clean_words)
        cw[i] = d["flipped_word"]
        corrupted[cue] = " ".join(cw)
    return {
        "set": "rep", "condition": cond, "gold": gold, "clean": " ".join(words),
        "gold_form": v_gold, "foil_form": v_foil, "cues": cues, "corrupted": corrupted,
        "lex": {
            "subj_masc": subj["masc"], "subj_gloss": subj.get("gloss", ""),
            "subj_kind": subj.get("kind", "noun"),
            "verb_masc": frame["masc"], "verb_gloss": frame.get("gloss", ""),
            "adj_masc": adj["masc"] if adj else None,
            "adj_kind": adj.get("kind") if adj else None,
            "attr_masc": attr["masc"] if attr else None,
            "attr_kind": attr.get("kind") if attr else None,
            "attr_new": attr.get("new") if attr else None,
            "attr_prep": prep, "modifier": None, "modifier_kind": None,
            "modifier_noun_gender": None,
        },
    }


def main():
    rng = random.Random(SEED)
    core = load_core()
    tks = {m: tokenizer(m) for m in REQUIRED}
    opt = {m: tokenizer(m) for m in OPTIONAL}
    lex = json.load(open(os.path.join(DATA, "lexicon.json"), encoding="utf-8"))

    # ---- adjectives
    adjs, dropped = [], []
    for k, (m, gloss) in enumerate(ADJ_NEW):
        rec = {"masc": m, "fem": m + "ة", "gloss": gloss,
               "kind": "nisba" if k >= 12 else "descriptive"}
        (adjs if aligned(tks, rec["masc"], rec["fem"]) else dropped).append(rec)
    old_adj = {a["masc"] for a in lex["adjectives"]}
    assert not old_adj & {a["masc"] for a in adjs}

    # ---- attractors
    subj_by_masc = {s["masc"]: s for s in lex["subjects"]}
    name_by_masc = {n["masc"]: n for n in lex["names"]}
    attrs = []
    for m in ATTR_NOUNS_NEW:
        s = subj_by_masc[m]
        attrs.append({"masc": s["masc"], "fem": s["fem"], "gloss": s.get("gloss", ""),
                      "kind": "noun", "new": True})
    for m in ATTR_NAMES_NEW:
        n = name_by_masc[m]
        attrs.append({"masc": n["masc"], "fem": n["fem"], "gloss": n.get("gloss", ""),
                      "kind": "name", "new": True})
    attrs = [a for a in attrs if aligned(tks, a["masc"], a["fem"])]
    old_attr = {a["masc"] for a in lex["attractors"]}
    assert not old_attr & {a["masc"] for a in attrs}
    # T7 uses old and new attractors (the condition itself is new)
    attrs_all = attrs + [{"masc": a["masc"], "fem": a["fem"], "gloss": a.get("gloss", ""),
                          "kind": "name" if a.get("is_name") else "noun", "new": False}
                         for a in core["attractors"]]

    # ---- subjects
    nouns = [dict(s, kind="noun") for s in core["subjects"]]
    names = [dict(s, kind="name") for s in core["names"]]
    construct = []
    for s in nouns:
        if s["masc"] in AMBIGUOUS_FEM:
            continue
        c = {"masc": strip_article(s["masc"]), "fem": strip_article(s["fem"])}
        if aligned(tks, c["masc"], c["fem"]):
            construct.append(dict(s, construct=c))

    spec = {
        "T2": dict(subjects=nouns, adjectives=adjs),
        "T3": dict(subjects=nouns, adjectives=adjs),
        "T6": dict(subjects=nouns + names, attractors=attrs),
        "T7": dict(subjects=construct, attractors=attrs_all),
    }

    items, n_drop = [], Counter()
    opt_ok = Counter()
    counts = defaultdict(Counter)
    for cond, pools in spec.items():
        for gold in ("m", "f"):
            seen, made, attempts = set(), 0, 0
            extra = [pools["adjectives"]] if "adjectives" in pools else [pools["attractors"]]
            gen = round_robin_product(pools["subjects"], core["frames"], *extra, rng=rng)
            while made < N_PER_CELL and attempts < N_PER_CELL * 80:
                attempts += 1
                subj, frame, cue_rec = next(gen)
                adj = cue_rec if "adjectives" in pools else None
                attr = cue_rec if "attractors" in pools else None
                if attr is not None and attr["masc"] in (subj["masc"], subj.get("fem")):
                    continue
                prep = rng.choice(core["preps"]) if cond == "T6" else None
                key = (subj["masc"], frame["masc"], cue_rec["masc"], prep)
                if key in seen:
                    continue
                seen.add(key)
                it = annotate_and_assert(build_item(cond, gold, subj, frame, adj, attr, prep), tks)
                if it is None:
                    n_drop[cond] += 1
                    continue
                for m, tk in opt.items():
                    ann = arbert_annotation(it, tk)
                    if ann is not None:
                        it["models"][m] = ann
                        opt_ok[(cond, m)] += 1
                it["id"] = f"R{cond}-{gold}-{made:04d}"
                items.append(it)
                counts[cond][gold] += 1
                made += 1

    json.dump({"adjectives": adjs, "adjectives_dropped": dropped, "attractors": attrs,
               "construct_subjects": [s["construct"] | {"gloss": s.get("gloss", "")}
                                      for s in construct]},
              open(OUT_LEX, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(OUT_FAM, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"adjectives kept {len(adjs)}/{len(ADJ_NEW)}  dropped {[d['masc'] for d in dropped]}")
    print(f"attractors kept {len(attrs)}  ({sum(a['kind'] == 'name' for a in attrs)} names)")
    print(f"construct-state subjects {len(construct)}")
    print(f"items {len(items)}  dropped by assertions {dict(n_drop)}")
    for cond in spec:
        tot = counts[cond]["m"] + counts[cond]["f"]
        cov = "  ".join(f"{m} {opt_ok[(cond, m)]}/{tot}" for m in OPTIONAL)
        print(f"  {cond}: m {counts[cond]['m']} f {counts[cond]['f']}   {cov}")
    for cond in spec:
        ex = next(i for i in items if i["condition"] == cond and i["gold"] == "f")
        print(f"[{ex['id']}] {ex['clean']}   gold={ex['gold_form']}")
    print(f"saved {OUT_LEX}, {OUT_FAM}")


if __name__ == "__main__":
    main()
