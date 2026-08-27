#!/usr/bin/env python3
"""
Exhaustively scan tokenizer vocabularies for same-lemma masc/fem pairs
(w / w+ة) with EQUAL token counts under each tokenizer.

Strategy (no strong assumptions, no cross-lemma substitution):
  Pass 1  Vocab scan: every whole-word vocab entry w starting with ال where
          w+ة is also a single vocab entry → guaranteed 1↔1 pair in that
          tokenizer. Run for BOTH tokenizers, intersect / union.
  Pass 2  Curated extended list (2↔2 candidates): forms absent from a vocab
          may still split into EQUAL counts (e.g. CAMeLBERT الممرض/الممرضة
          both 2). Checked per tokenizer.
  Pass 3  CALIMA validation: masc form has a (noun|adj, m, s) reading and
          fem form a (noun|adj, f, s) reading (advisory; DB has gaps).

Output: data/scan_pairs.json + console report grouped by alignment status:
  BOTH    equal counts in arabert AND camelbert  → cross-model core
  ARA     equal in arabert only                  → AraBERT extended set
  CAMEL   equal in camelbert only                → CAMeLBERT extended set

Human-noun vs adjective vs non-usable classification is done by native-speaker
review of this report (CALIMA does not mark humanness).
"""

import json
import os
import re

from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

from cuewin.config import data_dir
from cuewin.models import tokenizer

OUT_PATH = os.path.join(data_dir(), "scan_pairs.json")

ARABIC_WORD = re.compile(r"^ال[ء-ي]{2,}$")  # definite, Arabic letters only

# Pass-2 candidates: same-lemma pairs whose forms may be absent from a vocab
# but could split into equal counts. Includes everything from 01 plus more
# professions/roles. fem = masc + ة always (same lemma).
EXTENDED_MASC = [
    "الطالب", "المعلم", "الطبيب", "المهندس", "الكاتب", "المدير", "الممرض",
    "الموظف", "الجار", "المحامي", "الباحث", "الأستاذ", "العامل", "البائع",
    "المسافر", "الزائر", "الضيف", "السائق", "الوزير", "المريض", "الطفل",
    "الجد", "الصديق", "المخرج", "المغني", "اللاعب", "المدرب", "المحاضر",
    "المشرف", "المساعد", "المراسل", "المحرر", "المترجم", "المصور", "المذيع",
    "المضيف", "الجراح", "الصيدلي", "المحاسب", "المؤلف", "الشاعر", "الناقد",
    "الخبير", "العالم", "الرسام", "النجار", "الخياط", "الطباخ", "الحلاق",
    "الحارس", "الجندي", "الشرطي", "القاضي", "التاجر", "الفلاح", "المزارع",
    "السفير", "النائب", "القائد", "الزعيم", "البطل", "الشاهد", "المتهم",
    "المحقق", "المفتش", "المخترع", "المبرمج", "المصمم", "المعالج", "المنسق",
    "الوكيل", "الراقص", "الممثل", "المطرب", "الإعلامي", "الصحفي", "الفنان",
    "الرئيس", "الأمير", "الملك", "الشاب", "السباح", "العداء", "الملاكم",
    "الحكم", "المعلق", "المتسابق", "الفائز", "الخاسر", "المتطوع", "المؤسس",
    "المالك", "المستثمر", "المقاول", "المهرج", "الساحر", "الراوي", "المؤرخ",
]


def tok_ids(tk, w):
    return tk.encode(w, add_special_tokens=False)


def calima_ok(analyzer, masc, fem):
    """Advisory: masc has (noun/adj, m, s) and fem has (noun/adj, f, s)."""
    def has(form, gen):
        forms = [form, form[2:]] if form.startswith("ال") else [form]
        for f in forms:
            try:
                for a in analyzer.analyze(f):
                    if (a.get("pos") in ("noun", "adj")
                            and a.get("gen") == gen and a.get("num") == "s"):
                        return True
            except Exception:
                pass
        return False
    return has(masc, "m") and has(fem, "f")


def main():
    tks = {n: tokenizer(n) for n in ("arabert", "camelbert")}
    analyzer = Analyzer(MorphologyDB.builtin_db(flags="a"))

    # ---------------- Pass 1: exhaustive vocab scans (1↔1 within a vocab) ----
    vocab_pairs = {}          # masc -> set of tokenizers where pair is 1↔1
    for name, tk in tks.items():
        vocab = tk.get_vocab()
        n_found = 0
        for w in vocab:
            if not ARABIC_WORD.match(w):
                continue
            if w.endswith("ة"):
                continue
            if (w + "ة") in vocab:
                vocab_pairs.setdefault(w, set()).add(name)
                n_found += 1
        print(f"[pass 1] {name}: {n_found} definite w/w+ة single-token pairs in vocab")

    # ---------------- Pass 2: extended list, equal-count check per model -----
    candidates = {}           # masc -> {model: (n_masc, n_fem, aligned_bool)}
    all_masc = sorted(set(list(vocab_pairs.keys()) + EXTENDED_MASC))
    for masc in all_masc:
        fem = masc + "ة"
        per_model = {}
        for name, tk in tks.items():
            nm, nf = len(tok_ids(tk, masc)), len(tok_ids(tk, fem))
            per_model[name] = {"masc_ntok": nm, "fem_ntok": nf,
                               "aligned": nm == nf,
                               "single": nm == 1 and nf == 1}
        candidates[masc] = per_model

    # ---------------- Pass 3: CALIMA + grouping ------------------------------
    groups = {"BOTH": [], "ARA": [], "CAMEL": [], "NONE": []}
    records = []
    for masc, per_model in candidates.items():
        fem = masc + "ة"
        a_ok = per_model["arabert"]["aligned"]
        c_ok = per_model["camelbert"]["aligned"]
        group = ("BOTH" if a_ok and c_ok else
                 "ARA" if a_ok else
                 "CAMEL" if c_ok else "NONE")
        rec = {
            "masc": masc, "fem": fem, "group": group,
            "calima_ok": calima_ok(analyzer, masc, fem),
            "tok": per_model,
        }
        records.append(rec)
        groups[group].append(rec)

    # ---------------- report -------------------------------------------------
    for gname in ("BOTH", "CAMEL", "ARA"):
        items = sorted(groups[gname], key=lambda r: (
            r["tok"]["camelbert"]["masc_ntok"], r["masc"]))
        print(f"\n{'=' * 74}\n{gname}: equal-count same-lemma pairs "
              f"({len(items)})\n{'=' * 74}")
        print(f"{'masc':<14} {'fem':<14} {'calima':<7} "
              f"{'ara m/f':<9} {'camel m/f':<10}")
        print("-" * 60)
        for r in items:
            a, c = r["tok"]["arabert"], r["tok"]["camelbert"]
            cal = "OK" if r["calima_ok"] else "?"
            print(f"{r['masc']:<14} {r['fem']:<14} {cal:<7} "
                  f"{a['masc_ntok']}/{a['fem_ntok']:<7} "
                  f"{c['masc_ntok']}/{c['fem_ntok']:<8}")

    print(f"\nsummary: BOTH={len(groups['BOTH'])}  CAMEL-only={len(groups['CAMEL'])}  "
          f"ARA-only={len(groups['ARA'])}  unusable={len(groups['NONE'])}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
