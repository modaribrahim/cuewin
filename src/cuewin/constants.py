"""Static identifiers shared across scripts (no heavy imports)."""

from __future__ import annotations

#: internal key -> public HuggingFace identifier
MODEL_IDS = {
    "arabert": "aubmindlab/bert-base-arabertv02",
    "camelbert": "CAMeL-Lab/bert-base-arabic-camelbert-msa",
    "arbert": "UBC-NLP/ARBERT",
    "marbert": "UBC-NLP/MARBERT",
    "aragpt2": "aubmindlab/aragpt2-base",
    "xlmr": "FacebookAI/xlm-roberta-base",
}

#: the three masked encoders (ARBERT is additive, post-hoc)
ENCODERS = ("arabert", "camelbert", "arbert")

#: labels used by figures
MODEL_LABEL = {
    "arabert": "AraBERT",
    "camelbert": "CAMeLBERT",
    "arbert": "ARBERT",
    "marbert": "MARBERT",
    "aragpt2": "AraGPT2",
    "xlmr": "XLM-R",
}

#: labels used by error-examples table (11)
MODEL_LABEL_SHORT = {"arabert": "AraBERT", "camelbert": "CAMeLBERT", "both": "both"}
