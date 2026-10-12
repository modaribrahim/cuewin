#!/usr/bin/env python3
"""
Build and verify the lexicon for the MSA agreement dataset.

Every candidate item is checked against BOTH model tokenizers (AraBERTv02 64k,
CAMeLBERT-MSA 30k) and the CAMeL morphological analyzer:

  Check 1  single-token status of every gendered form, per tokenizer
  Check 2  masc/fem token-count equality per tokenizer
           (clean/corrupted alignment requirement — a flipped cue must not
            shift the positions of the tokens that follow it)
  Check 3  in-context tokenization == standalone tokenization
           (no cross-word merge surprises once words are placed in a sentence)
  Check 4  analyzer round-trip: the surface form must carry the intended
           morphological reading (pos / person / gender / number / aspect)

Tiers (for flipped-cue items: subjects, adjectives, names, attractors):
  A  single-token in both genders in both tokenizers  (ideal)
  B  equal masc/fem token count in both tokenizers    (usable, span patching)
  F  unequal count in at least one tokenizer          (drop or model-specific)

Verbs are the masked TARGET: hard requirement = single-token in both
tokenizers in both genders (logit comparison needs one vocab id per form).

Output: ../data/lexicon.json + console report.
"""

import json
import os
from collections import Counter

from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

from cuewin.config import data_dir
from cuewin.models import tokenizer

OUT_PATH = os.path.join(data_dir(), "lexicon.json")

# ----------------------------------------------------------------------------
# CANDIDATE LEXICON (hand-curated, generous — filters decide what survives)
# ----------------------------------------------------------------------------

# VP frames: verb pair (masc/fem past 3sg) + fixed object/complement.
# The object stays constant within a family, so it may be multi-token;
# plausibility must hold for ANY human subject.
VP_FRAMES = [
    {"masc": "كتب",   "fem": "كتبت",   "comp": "الرسالة",      "gloss": "wrote the letter"},
    {"masc": "قرأ",   "fem": "قرأت",   "comp": "الكتاب",       "gloss": "read the book"},
    {"masc": "دخل",   "fem": "دخلت",   "comp": "الغرفة",       "gloss": "entered the room"},
    {"masc": "خرج",   "fem": "خرجت",   "comp": "من البيت",     "gloss": "left the house"},
    {"masc": "ذهب",   "fem": "ذهبت",   "comp": "إلى السوق",    "gloss": "went to the market"},
    {"masc": "وصل",   "fem": "وصلت",   "comp": "إلى المدينة",  "gloss": "arrived at the city"},
    {"masc": "ركب",   "fem": "ركبت",   "comp": "الحافلة",      "gloss": "rode the bus"},
    {"masc": "حضر",   "fem": "حضرت",   "comp": "الاجتماع",     "gloss": "attended the meeting"},
    {"masc": "شاهد",  "fem": "شاهدت",  "comp": "الفيلم",       "gloss": "watched the film"},
    {"masc": "سمع",   "fem": "سمعت",   "comp": "الخبر",        "gloss": "heard the news"},
    {"masc": "فتح",   "fem": "فتحت",   "comp": "الباب",        "gloss": "opened the door"},
    {"masc": "أكل",   "fem": "أكلت",   "comp": "الطعام",       "gloss": "ate the food"},
    {"masc": "شرب",   "fem": "شربت",   "comp": "القهوة",       "gloss": "drank the coffee"},
    {"masc": "درس",   "fem": "درست",   "comp": "الدرس",        "gloss": "studied the lesson"},
    {"masc": "حمل",   "fem": "حملت",   "comp": "الحقيبة",      "gloss": "carried the bag"},
    {"masc": "طلب",   "fem": "طلبت",   "comp": "المساعدة",     "gloss": "asked for help"},
    {"masc": "وجد",   "fem": "وجدت",   "comp": "المفتاح",      "gloss": "found the key"},
    {"masc": "ترك",   "fem": "تركت",   "comp": "المكتب",       "gloss": "left the office"},
    {"masc": "زار",   "fem": "زارت",   "comp": "المتحف",       "gloss": "visited the museum"},
    {"masc": "باع",   "fem": "باعت",   "comp": "السيارة",      "gloss": "sold the car"},
    {"masc": "قال",   "fem": "قالت",   "comp": "الحقيقة",      "gloss": "told the truth"},
    {"masc": "عرف",   "fem": "عرفت",   "comp": "الجواب",       "gloss": "knew the answer"},
    {"masc": "لبس",   "fem": "لبست",   "comp": "المعطف",       "gloss": "wore the coat"},
    {"masc": "غسل",   "fem": "غسلت",   "comp": "الملابس",      "gloss": "washed the clothes"},
    {"masc": "طبخ",   "fem": "طبخت",   "comp": "الطعام",       "gloss": "cooked the food"},
    {"masc": "نظف",   "fem": "نظفت",   "comp": "البيت",        "gloss": "cleaned the house"},
    {"masc": "كسر",   "fem": "كسرت",   "comp": "الزجاج",       "gloss": "broke the glass"},
    {"masc": "أغلق",  "fem": "أغلقت",  "comp": "النافذة",      "gloss": "closed the window"},
    {"masc": "بدأ",   "fem": "بدأت",   "comp": "العمل",        "gloss": "started the work"},
    {"masc": "رسم",   "fem": "رسمت",   "comp": "اللوحة",       "gloss": "drew the painting"},
    {"masc": "أخذ",   "fem": "أخذت",   "comp": "الدواء",       "gloss": "took the medicine"},
    {"masc": "دفع",   "fem": "دفعت",   "comp": "الحساب",       "gloss": "paid the bill"},
]

# Subject noun pairs (definite, human, singular). SAME LEMMA ONLY (fem = masc+ة).
# Pairs marked scan=True were found by the exhaustive vocab scan
# (scan_pairs) and verified by native-speaker review.
#
# Homographic ة-forms. Several genuine feminine pairs surface an undiacritized
# ة-form that is ALSO a distinct abstract noun (e.g. الجراحة = female surgeon
# جَرّاحة OR surgery جِراحة). Both readings are grammatically feminine, so the
# agreement target is identical either way; these pairs are therefore INCLUDED.
# build_lexicon asserts this invariant: for every such pair, all noun readings
# CALIMA assigns to the ة-form are feminine. (The tokenization-aligned subset,
# group BOTH, is the core used for families.)
#
# Pairs excluded on TOKENIZATION only (not linguistics): those that are not
# 1↔1/2↔2 aligned in all three encoders (groups ARA/CAMEL/NONE), or whose
# in-context tokenization is inconsistent.
#
# Ambiguous reading recorded for the appendix/released scan_pairs.json (these
# are the homographic pairs whose abstract noun is feminine — kept in the core):
#   الجراح/الجراحة (surgery), الصيدلي/الصيدلية (pharmacy), الحلاق/الحلاقة
#   (haircutting), المساعد/المساعدة (help), المعالج/المعالجة (processing),
#   الملاكم/الملاكمة (boxing), المهرج/المهرجة (carnival).
SUBJECT_PAIRS = [
    # --- found by vocab scan, verified clean (aligned in BOTH models) ---
    {"masc": "المحرر",   "fem": "المحررة",   "gloss": "editor",        "scan": True},
    {"masc": "المترجم",  "fem": "المترجمة",  "gloss": "translator",    "scan": True},
    {"masc": "المصور",   "fem": "المصورة",   "gloss": "photographer",  "scan": True},
    {"masc": "المضيف",   "fem": "المضيفة",   "gloss": "host",          "scan": True},
    {"masc": "الشرطي",   "fem": "الشرطية",   "gloss": "police officer","scan": True},
    {"masc": "المبرمج",  "fem": "المبرمجة",  "gloss": "programmer",    "scan": True},
    {"masc": "الممثل",   "fem": "الممثلة",   "gloss": "actor",         "scan": True},
    {"masc": "المطرب",   "fem": "المطربة",   "gloss": "singer",        "scan": True},
    {"masc": "الإعلامي", "fem": "الإعلامية", "gloss": "media person",  "scan": True},
    {"masc": "المتطوع",  "fem": "المتطوعة",  "gloss": "volunteer",     "scan": True},
    {"masc": "المتحدث",  "fem": "المتحدثة",  "gloss": "spokesperson",  "scan": True},
    {"masc": "المتخصص",  "fem": "المتخصصة",  "gloss": "specialist",    "scan": True},
    {"masc": "المسؤول",  "fem": "المسؤولة",  "gloss": "official",      "scan": True},
    {"masc": "المستخدم", "fem": "المستخدمة", "gloss": "user",          "scan": True},
    {"masc": "المهاجر",  "fem": "المهاجرة",  "gloss": "migrant",       "scan": True},
    {"masc": "المفوض",   "fem": "المفوضة",   "gloss": "commissioner",  "scan": True},
    {"masc": "المؤيد",   "fem": "المؤيدة",   "gloss": "supporter",     "scan": True},
    # Native-speaker review reinstated these two (ة-form has a genuine
    # feminine-agentive reading: female referee / female founder, alongside
    # the abstract reading the scan flagged); both 1↔1 in all three encoders.
    {"masc": "الحكم",    "fem": "الحكمة",    "gloss": "referee",       "scan": True},
    {"masc": "المؤسس",   "fem": "المؤسسة",   "gloss": "founder",       "scan": True},
    # --- CAMeLBERT-only additions from scan (2↔2 there, unequal in AraBERT) ---
    {"masc": "الرسام",   "fem": "الرسامة",   "gloss": "painter",       "scan": True},
    {"masc": "التاجر",   "fem": "التاجرة",   "gloss": "merchant",      "scan": True},
    {"masc": "الناقد",   "fem": "الناقدة",   "gloss": "critic",        "scan": True},
    {"masc": "المفتش",   "fem": "المفتشة",   "gloss": "inspector",     "scan": True},
    # --- homographic ة-forms (genuine feminine + abstract noun, both f) ---
    {"masc": "الجراح",   "fem": "الجراحة",   "gloss": "surgeon",       "scan": True},
    {"masc": "الحلاق",   "fem": "الحلاقة",   "gloss": "barber",        "scan": True},
    {"masc": "الصيدلي",  "fem": "الصيدلية",  "gloss": "pharmacist",    "scan": True},
    {"masc": "المساعد",  "fem": "المساعدة",  "gloss": "assistant",     "scan": True},
    {"masc": "المعالج",  "fem": "المعالجة",  "gloss": "therapist",     "scan": True},
    {"masc": "الملاكم",  "fem": "الملاكمة",  "gloss": "boxer",         "scan": True},
    {"masc": "المهرج",   "fem": "المهرجة",   "gloss": "clown",         "scan": True},
    # --- original curated list ---
    {"masc": "الطالب",   "fem": "الطالبة",   "gloss": "student"},
    {"masc": "المعلم",   "fem": "المعلمة",   "gloss": "teacher"},
    {"masc": "الطبيب",   "fem": "الطبيبة",   "gloss": "doctor"},
    {"masc": "المهندس",  "fem": "المهندسة",  "gloss": "engineer"},
    {"masc": "الكاتب",   "fem": "الكاتبة",   "gloss": "writer"},
    {"masc": "المدير",   "fem": "المديرة",   "gloss": "manager"},
    {"masc": "الممرض",   "fem": "الممرضة",   "gloss": "nurse"},
    {"masc": "الموظف",   "fem": "الموظفة",   "gloss": "employee"},
    {"masc": "الجار",    "fem": "الجارة",    "gloss": "neighbor"},
    {"masc": "الصحفي",   "fem": "الصحفية",   "gloss": "journalist"},
    {"masc": "المحامي",  "fem": "المحامية",  "gloss": "lawyer"},
    {"masc": "الفنان",   "fem": "الفنانة",   "gloss": "artist"},
    {"masc": "الباحث",   "fem": "الباحثة",   "gloss": "researcher"},
    {"masc": "الأستاذ",  "fem": "الأستاذة",  "gloss": "professor"},
    {"masc": "العامل",   "fem": "العاملة",   "gloss": "worker"},
    {"masc": "البائع",   "fem": "البائعة",   "gloss": "seller"},
    {"masc": "المسافر",  "fem": "المسافرة",  "gloss": "traveler"},
    {"masc": "الزائر",   "fem": "الزائرة",   "gloss": "visitor"},
    {"masc": "الضيف",    "fem": "الضيفة",    "gloss": "guest"},
    {"masc": "السائق",   "fem": "السائقة",   "gloss": "driver"},
    {"masc": "الرئيس",   "fem": "الرئيسة",   "gloss": "president"},
    {"masc": "الوزير",   "fem": "الوزيرة",   "gloss": "minister"},
    {"masc": "المريض",   "fem": "المريضة",   "gloss": "patient"},
    {"masc": "الطفل",    "fem": "الطفلة",    "gloss": "child"},
    {"masc": "الجد",     "fem": "الجدة",     "gloss": "grandparent"},
    {"masc": "الأمير",   "fem": "الأميرة",   "gloss": "prince/princess"},
    {"masc": "الملك",    "fem": "الملكة",    "gloss": "king/queen"},
    {"masc": "الشاب",    "fem": "الشابة",    "gloss": "young person"},
    {"masc": "الصديق",   "fem": "الصديقة",   "gloss": "friend"},
    {"masc": "المخرج",   "fem": "المخرجة",   "gloss": "director"},
    {"masc": "المغني",   "fem": "المغنية",   "gloss": "singer"},
    {"masc": "اللاعب",   "fem": "اللاعبة",   "gloss": "player"},
]

# Adjective pairs (definite, attributive after definite subject).
ADJ_PAIRS = [
    {"masc": "المجتهد",  "fem": "المجتهدة",  "gloss": "hardworking"},
    {"masc": "الجديد",   "fem": "الجديدة",   "gloss": "new"},
    {"masc": "الصغير",   "fem": "الصغيرة",   "gloss": "young/small"},
    {"masc": "الكبير",   "fem": "الكبيرة",   "gloss": "old/big"},
    {"masc": "الطويل",   "fem": "الطويلة",   "gloss": "tall"},
    {"masc": "الجميل",   "fem": "الجميلة",   "gloss": "beautiful"},
    {"masc": "الذكي",    "fem": "الذكية",    "gloss": "smart"},
    {"masc": "النشيط",   "fem": "النشيطة",   "gloss": "active"},
    {"masc": "الشهير",   "fem": "الشهيرة",   "gloss": "famous"},
    {"masc": "الماهر",   "fem": "الماهرة",   "gloss": "skilled"},
    {"masc": "السعيد",   "fem": "السعيدة",   "gloss": "happy"},
    {"masc": "الحزين",   "fem": "الحزينة",   "gloss": "sad"},
    {"masc": "المشهور",  "fem": "المشهورة",  "gloss": "well-known"},
    {"masc": "المتعب",   "fem": "المتعبة",   "gloss": "tired"},
    {"masc": "الهادئ",   "fem": "الهادئة",   "gloss": "calm"},
    {"masc": "الشجاع",   "fem": "الشجاعة",   "gloss": "brave"},
]

# Proper-name pairs (unambiguous gender, high-frequency).
NAME_PAIRS = [
    {"masc": "محمد",     "fem": "فاطمة",  "gloss": "Mohammed/Fatima"},
    {"masc": "أحمد",     "fem": "مريم",   "gloss": "Ahmed/Mariam"},
    {"masc": "علي",      "fem": "زينب",   "gloss": "Ali/Zainab"},
    {"masc": "عمر",      "fem": "سارة",   "gloss": "Omar/Sara"},
    {"masc": "خالد",     "fem": "ليلى",   "gloss": "Khaled/Layla"},
    {"masc": "حسن",      "fem": "هدى",    "gloss": "Hassan/Huda"},
    {"masc": "إبراهيم",  "fem": "عائشة",  "gloss": "Ibrahim/Aisha"},
    {"masc": "سعيد",     "fem": "سلمى",   "gloss": "Saeed/Salma"},
    {"masc": "يوسف",     "fem": "هند",    "gloss": "Youssef/Hind"},
    {"masc": "كريم",     "fem": "رنا",    "gloss": "Karim/Rana"},
]

# Attractor PPs: human noun inside a PP between subject and verb (T6).
# The attractor noun itself is a gendered pair → must pass the same checks.
# First run showed CAMeLBERT splits the fem form of all common definite-noun
# attractor candidates, so the pool is: (a) noun pairs that survived tier A
# as subjects, (b) proper-name pairs (always tier A) — both fully valid
# attractors linguistically.
ATTRACTOR_PAIRS = [
    {"masc": "الفنان",   "fem": "الفنانة",   "prep": "مع", "gloss": "with the artist"},
    {"masc": "العامل",   "fem": "العاملة",   "prep": "مع", "gloss": "with the worker"},
    {"masc": "الصحفي",   "fem": "الصحفية",   "prep": "مع", "gloss": "with the journalist"},
    {"masc": "الطفل",    "fem": "الطفلة",    "prep": "مع", "gloss": "with the child"},
    {"masc": "الشاب",    "fem": "الشابة",    "prep": "مع", "gloss": "with the young person"},
    {"masc": "محمد",     "fem": "فاطمة",     "prep": "مع", "gloss": "with Mohammed/Fatima", "is_name": True},
    {"masc": "أحمد",     "fem": "مريم",      "prep": "مع", "gloss": "with Ahmed/Mariam",    "is_name": True},
    {"masc": "خالد",     "fem": "ليلى",      "prep": "مع", "gloss": "with Khaled/Layla",    "is_name": True},
]

# Modifiers for the distance condition (T5). Never flipped → no token-count
# pairing constraint; only in-context tokenization consistency required.
# DIVERSITY (reviewer fix, 2026-06-11): three structural kinds, not one PP
# template. CONFOUND CONTROL: "gender-neutral" words still carry grammatical
# gender on their nouns (الجامعة f, المكتب m), and a T5 audit showed a ~12-pt
# accuracy swing in AraBERT depending on modifier-noun gender — so noun gender
# is recorded per item and BALANCED m/f across the pool, with bare adverbs as
# the noun-free subset.
NEUTRAL_MODIFIERS = [
    # PP-locative (2m / 2f)
    {"text": "في المكتب",   "kind": "pp_loc",  "noun_gender": "m", "gloss": "at the office"},
    {"text": "في المطار",   "kind": "pp_loc",  "noun_gender": "m", "gloss": "at the airport"},
    {"text": "في الجامعة",  "kind": "pp_loc",  "noun_gender": "f", "gloss": "at the university"},
    {"text": "في المدينة",  "kind": "pp_loc",  "noun_gender": "f", "gloss": "in the city"},
    # PP-temporal (2m / 2f)
    {"text": "في الصباح",   "kind": "pp_temp", "noun_gender": "m", "gloss": "in the morning"},
    {"text": "في المساء",   "kind": "pp_temp", "noun_gender": "m", "gloss": "in the evening"},
    {"text": "بعد الرحلة",  "kind": "pp_temp", "noun_gender": "f", "gloss": "after the trip"},
    {"text": "قبل المغادرة","kind": "pp_temp", "noun_gender": "f", "gloss": "before departure"},
    # bare adverbials (no overt noun, or weakly nominal manner adverbs)
    {"text": "أمس",        "kind": "adv",     "noun_gender": "none", "gloss": "yesterday"},
    {"text": "بهدوء",      "kind": "adv",     "noun_gender": "m",    "gloss": "calmly"},
    {"text": "بسرعة",      "kind": "adv",     "noun_gender": "f",    "gloss": "quickly"},
    {"text": "دون تردد",   "kind": "adv",     "noun_gender": "m",    "gloss": "without hesitation"},
]

# Prepositions for the T6 attractor PP (constant within family, never flipped).
# Diversified beyond مع for the same reason as the modifiers.
ATTRACTOR_PREPS = ["مع", "أمام", "بجوار"]

# ----------------------------------------------------------------------------
# CHECKS
# ----------------------------------------------------------------------------

def tok_ids(tokenizer, word):
    """Token ids of a standalone word (no specials)."""
    return tokenizer.encode(word, add_special_tokens=False)


def in_context_consistent(tokenizer, word):
    """Check 3: tokenization of the word inside a carrier sentence equals its
    standalone tokenization (guards against any cross-word effects)."""
    standalone = tok_ids(tokenizer, word)
    carrier = f"قال إن {word} هنا"  # word in a mid-sentence slot
    sent_ids = tokenizer.encode(carrier, add_special_tokens=False)
    # find the standalone id sequence inside the carrier ids
    n, m = len(sent_ids), len(standalone)
    found = any(sent_ids[i:i + m] == standalone for i in range(n - m + 1))
    return found


def analyzer_has_reading(analyzer, form, pos, gen, per=None, num="s", asp=None):
    """Check 4: surface form carries the intended morphological reading.

    NB: CALIMA tags Arabic adjectives as pos='noun' in most entries, so for
    adjectives we accept either tag. This check is ADVISORY — items failing
    it are flagged for native-speaker confirmation, not auto-dropped
    (the DB has real gaps, e.g. الضيفة has no analysis at all).
    """
    accepted_pos = {pos}
    if pos == "adj":
        accepted_pos.add("noun")
    try:
        analyses = analyzer.analyze(form)
    except Exception:
        return False
    for a in analyses:
        if a.get("pos") not in accepted_pos:
            continue
        if a.get("gen") != gen:
            continue
        if num and a.get("num") != num:
            continue
        if per and a.get("per") != per:
            continue
        if asp and a.get("asp") != asp:
            continue
        return True
    return False


def all_readings_feminine(analyzer, form):
    """Check 5 (homograph pairs only): every noun/adj reading CALIMA assigns
    to the ة-form is feminine. If a distinct abstract noun coexists with the
    feminine agentive (e.g. جِراحة surgery vs جَرّاحة female surgeon), both
    readings must agree in feminine so the agreement target is unchanged."""
    try:
        analyses = analyzer.analyze(form)
    except Exception:
        return False
    readings = [a for a in analyses if a.get("pos") in ("noun", "adj")]
    if not readings:
        return False
    return all(a.get("gen") == "f" for a in readings)


HOMOGRAPHIC_SUBJECTS = {
    "الجراح", "الحلاق", "الصيدلي", "المساعد", "المعالج", "الملاكم", "المهرج",
}


def check_pair(pair, tokenizers, analyzer, pos, per=None, asp=None,
               skip_analyzer=False):
    """Run all checks on a masc/fem pair. Returns enriched record."""
    rec = dict(pair)
    rec["tok"] = {}
    single_everywhere = True
    context_ok = True

    for mname, tk in tokenizers.items():
        ids_m = tok_ids(tk, pair["masc"])
        ids_f = tok_ids(tk, pair["fem"])
        rec["tok"][mname] = {
            "masc_ids": ids_m,
            "fem_ids": ids_f,
            "masc_ntok": len(ids_m),
            "fem_ntok": len(ids_f),
            "masc_pieces": tk.convert_ids_to_tokens(ids_m),
            "fem_pieces": tk.convert_ids_to_tokens(ids_f),
        }
        if len(ids_m) != 1 or len(ids_f) != 1:
            single_everywhere = False
        if not (in_context_consistent(tk, pair["masc"])
                and in_context_consistent(tk, pair["fem"])):
            context_ok = False

    rec["context_ok"] = context_ok

    if skip_analyzer:
        rec["morph_ok"] = None  # names: not in the MSA lexicon DB, checked by hand
    else:
        # strip ال for analysis of definite forms (analyzer handles both, but
        # check the bare form too as fallback)
        def morph(form, gen):
            ok = analyzer_has_reading(analyzer, form, pos, gen, per=per, asp=asp)
            if not ok and form.startswith("ال"):
                ok = analyzer_has_reading(analyzer, form[2:], pos, gen,
                                          per=per, asp=asp)
            return ok
        rec["morph_ok"] = morph(pair["masc"], "m") and morph(pair["fem"], "f")

    # homograph pairs: the ة-form's abstract reading must agree in feminine
    rec["homograph_ok"] = None
    if pair.get("masc") in HOMOGRAPHIC_SUBJECTS and not skip_analyzer:
        rec["homograph_ok"] = all_readings_feminine(analyzer, pair["fem"])
    if rec["homograph_ok"] is False:
        raise ValueError(
            f"{pair['masc']}/{pair['fem']}: homograph ة-form has a "
            f"non-feminine noun/adj reading; agreement target not guaranteed")

    # per-model alignment (equal masc/fem token count under that tokenizer —
    # 1↔1 or 2↔2 both fine; patching is span-to-span within one model)
    rec["aligned"] = {m: rec["tok"][m]["masc_ntok"] == rec["tok"][m]["fem_ntok"]
                      for m in tokenizers}
    a, c = rec["aligned"]["arabert"], rec["aligned"]["camelbert"]
    if not context_ok:
        rec["group"] = "NONE"
    elif a and c:
        rec["group"] = "BOTH"
    elif a:
        rec["group"] = "ARA"
    elif c:
        rec["group"] = "CAMEL"
    else:
        rec["group"] = "NONE"
    # legacy field kept for compatibility
    rec["tier"] = {"BOTH": "A" if single_everywhere else "B",
                   "ARA": "ARA", "CAMEL": "CAMEL", "NONE": "F"}[rec["group"]]
    return rec


def check_single_form(item, tokenizers, analyzer):
    """Check a single (unpaired) replacement noun: per-tokenizer counts,
    in-context consistency, advisory gender reading."""
    rec = dict(item)
    rec["tok"] = {}
    context_ok = True
    for mname, tk in tokenizers.items():
        ids = tok_ids(tk, item["form"])
        rec["tok"][mname] = {
            "ids": ids,
            "ntok": len(ids),
            "pieces": tk.convert_ids_to_tokens(ids),
        }
        if not in_context_consistent(tk, item["form"]):
            context_ok = False
    rec["context_ok"] = context_ok
    form = item["form"]
    ok = analyzer_has_reading(analyzer, form, "noun", item["gender"])
    if not ok and form.startswith("ال"):
        ok = analyzer_has_reading(analyzer, form[2:], "noun", item["gender"])
    rec["morph_ok"] = ok
    return rec


def main():
    print("Loading tokenizers...")
    tokenizers = {name: tokenizer(name)
                  for name in ("arabert", "camelbert")}
    print("Loading morphology analyzer (calima-msa-r13)...")
    analyzer = Analyzer(MorphologyDB.builtin_db(flags="a"))

    results = {}

    # --- verbs: hard requirement = tier A + morph_ok -------------------------
    verbs = []
    for frame in VP_FRAMES:
        rec = check_pair(frame, tokenizers, analyzer,
                         pos="verb", per="3", asp="p")
        verbs.append(rec)
    results["vp_frames"] = verbs

    # --- flipped-cue categories ----------------------------------------------
    results["subjects"] = [check_pair(p, tokenizers, analyzer, pos="noun")
                           for p in SUBJECT_PAIRS]
    results["adjectives"] = [check_pair(p, tokenizers, analyzer, pos="adj")
                             for p in ADJ_PAIRS]
    results["names"] = [check_pair(p, tokenizers, analyzer, pos="noun_prop",
                                   skip_analyzer=True)
                        for p in NAME_PAIRS]
    results["attractors"] = [check_pair(p, tokenizers, analyzer, pos="noun",
                                        skip_analyzer=p.get("is_name", False))
                             for p in ATTRACTOR_PAIRS]
    results["neutral_modifiers"] = NEUTRAL_MODIFIERS
    results["attractor_preps"] = ATTRACTOR_PREPS

    # ----------------------------------------------------------------- report
    def report(name, items, verb=False):
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        hdr = f"{'masc':<12} {'fem':<12} {'tier':<5} {'morph':<6} " \
              f"{'arabert m/f':<12} {'camelbert m/f':<14} gloss"
        print(hdr)
        print("-" * len(hdr))
        for r in items:
            a = r["tok"]["arabert"]
            c = r["tok"]["camelbert"]
            morph = {True: "OK", False: "FAIL", None: "skip"}[r["morph_ok"]]
            flag = ""
            if verb and (r["tier"] != "A" or r["morph_ok"] is False):
                flag = "  << EXCLUDED (verb must be tier A + morph OK)"
            elif r["tier"] == "F":
                flag = "  << EXCLUDED"
            print(f"{r['masc']:<12} {r['fem']:<12} {r['tier']:<5} {morph:<6} "
                  f"{a['masc_ntok']}/{a['fem_ntok']:<10} "
                  f"{c['masc_ntok']}/{c['fem_ntok']:<12} {r['gloss']}{flag}")
        tiers = Counter(r["tier"] for r in items)
        print(f"\n  tiers: {dict(tiers)}")

    report("VP FRAMES (verbs are the masked target — tier A required)",
           results["vp_frames"], verb=True)
    report("SUBJECT NOUN PAIRS", results["subjects"])
    report("ADJECTIVE PAIRS", results["adjectives"])
    report("NAME PAIRS (morph check skipped — proper names)", results["names"])
    report("ATTRACTOR NOUN PAIRS", results["attractors"])


    # ---------------------------------------------------------------- summary
    usable_verbs = [r for r in results["vp_frames"]
                    if r["tier"] == "A" and r["morph_ok"]]
    print(f"\n{'#' * 78}")
    print(f"USABLE VP FRAMES (tier A + morph OK): {len(usable_verbs)} / {len(VP_FRAMES)}")
    for cat in ("subjects", "adjectives", "names", "attractors"):
        a = sum(1 for r in results[cat] if r["tier"] == "A")
        b = sum(1 for r in results[cat] if r["tier"] == "B")
        print(f"{cat:<12} tier A: {a:>3}   tier B: {b:>3}   "
              f"of {len(results[cat])}")
    print("#" * 78)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nLexicon with full check results saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
