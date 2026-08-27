# CueWIN

Causal analysis of grammatical gender agreement in Modern Standard Arabic (MSA)
language models. Companion code and data for the paper

> **Which Cue Wins? Causal Evidence for a Localized Gender-Agreement Mechanism
> in Arabic Encoders**

The paper analyses *why* MSA encoders default to the masculine verb form. We run
value patching and value zeroing interventions on three masked encoders —
**AraBERTv02**, **CAMeLBERT-MSA**, **ARBERT** — plus one decoder (**AraGPT2**),
across six template agreement constructions and a naturalistic corpus
(UD Arabic-PADT). All number reported in the paper are produced by this code.

---

## Quick start (CPU, ~5 minutes)

Verify that this release reproduces the paper's numbers byte-for-byte, no GPU
and no model weights needed:

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # or: pip install -e .
pip install pytest

# Regenerates data/stats_summary.csv, data/enc_dec_compare.csv, the dose
# response files, error-example table and all figures from the committed data.
cuewin-stats-report
cuewin-enc-dec-compare
cuewin-dose-response
cuewin-error-examples
cuewin-figures-paper
cuewin-figures-logit-lens
cuewin-figures-revision
cuewin-figures-natural

# Optional: smoke-test suite against the committed results
pytest tests/
```

Each `cuewin-*` command writes into `data/` (or `figures/`). The freshly
written files must match the committed ones byte-for-byte.

## Running the full pipeline (GPU, hours)

First download or point at the four models. They default to public HuggingFace
identifiers, or a local checkout under `models/<key>` (set `CUEWIN_MODELS`):

| key        | HuggingFace id                        |
|------------|----------------------------------------|
| `arabert`  | `aubmindlab/bert-base-arabertv02`      |
| `camelbert`| `CAMeL-Lab/bert-base-camelbert-msa`    |
| `arbert`   | `UBC-NLP/ARBERT`                       |
| `aragpt2`  | `aubmindlab/aragpt2-base`              |

Then run the pipeline stages in order (each writes its named CSV):

```bash
cuewin-build-lexicon      # data/lexicon.json  (camel_tools)
cuewin-scan-pairs         # data/families.jsonl
cuewin-build-families     # data/families_arbert.jsonl
cuewin-add-arbert
cuewin-behavioral         # data/behavioral_results.csv
cuewin-freq-baseline      # data/freq_baseline.csv
cuewin-value-zeroing      # data/zeroing_results.csv
cuewin-value-patching     # data/patching_results.csv
cuewin-morpheme-probe     # data/morpheme_results.csv (CAMeLBERT only)
cuewin-logit-lens         # data/logit_lens.csv
cuewin-decoder-behavioral # data/decoder_behavioral.csv
cuewin-intervention       # data/intervention_results.csv

cuewin-natural-extract    # data/padt/natural_candidates.jsonl (needs conllu)
cuewin-natural-families   # data/natural_families_T{1,4,6}_arbert.jsonl
cuewin-natural-behavioral # data/natural_behavioral.csv
cuewin-natural-patching   # data/natural_patching.csv
cuewin-natural-repair     # data/natural_repair.csv
cuewin-t6-repair          # in-place attractor fix
```

Set `PYTORCH_CUDA_ALLOC_CONF=expand_segments:True` for the large natural stages.

## Lexicon: scanned candidates vs. frozen core

`data/lexicon.json` is the full **scan output** (candidate pairs plus their
tokenization/check metadata). The benchmark uses only entries passing all of
the paper's filters (group `BOTH` = aligned in every encoder; verb frames
additionally single-token in both forms everywhere and morphologically
regular). Applying those criteria to the committed file reproduces the paper's
frozen lexicon exactly:

| set | frozen count | filter |
|-----|-------------|--------|
| subject professions | 34 | `group == BOTH` |
| names | 10 | — |
| verb frames | 21 | `BOTH` + single-token (both forms, all encoders) + `morph_ok` |
| adjectives | 11 | `group == BOTH` |
| attractors | 8 | `group == BOTH` |

Entries outside the core (e.g., frames نظف/أغلق, which fail the
single-token requirement) are retained for transparency but never enter
`families.jsonl`.

## Layout

```
code/
  pyproject.toml  package metadata + console entry points
  requirements.txt  runtime deps
  src/cuewin/       the package
    config.py        path/environment configuration (env-var aware)
    stats.py         bootstrap, permutation, cluster bootstrap, OLS
    hooks.py         value-vector cache/patch/zero forward hooks
    analysis/        CPU-only statistics, tables, figures
    pipeline/        GPU model-forwarding stages
  data/             committed result CSVs + PADT natural items
  figures/          regenerated figures
  tests/            pytest smoke tests (`pytest tests/`)
```

All random generators are seeded (item-level seed 42, cluster-bootstrap seed 43).
Figures are rendered by `matplotlib 3.10.9`; a different version changes the
rendered output but not the underlying numbers.

## License
MIT.