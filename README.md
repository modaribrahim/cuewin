# CueWIN

Causal analysis of grammatical gender agreement in Modern Standard Arabic (MSA)
language models. Companion code and data for the paper

> **Which Cue Wins? Causal Evidence on Gender Agreement in Arabic BERT Encoders**

The paper analyses where Arabic encoders decide verb gender and why competing
gender cues cause errors. We run
value patching, residual-stream, head-level and value-zeroing interventions on
four masked encoders, **AraBERTv02**, **CAMeLBERT-MSA**, **ARBERT**, **MARBERT**,
plus one decoder (**AraGPT2**),
across six template agreement constructions and a naturalistic corpus
(UD Arabic-PADT). All numbers reported in the paper are produced by this code.

---

## Quick start (CPU, ~5 minutes)

Verify that this release reproduces the paper's numbers byte-for-byte, no GPU
and no model weights needed:

```bash
git clone https://github.com/modaribrahim/cuewin.git && cd cuewin
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

First download or point at the five models. They default to public HuggingFace
identifiers, or a local checkout under `models/<key>` (set `CUEWIN_MODELS`):

| key        | HuggingFace id                        |
|------------|----------------------------------------|
| `arabert`  | `aubmindlab/bert-base-arabertv02`      |
| `camelbert`| `CAMeL-Lab/bert-base-arabic-camelbert-msa`    |
| `arbert`   | `UBC-NLP/ARBERT`                       |
| `marbert`  | `UBC-NLP/MARBERT`                      |
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

### Window, hidden-state and head experiments

Additive stages that write new `data/rev_*` files and never modify the
committed results above:

```bash
cuewin-revision               # stages: annotate, marbert, windows, residual, heads
                              #   (select with CUEWIN_STAGES=windows,residual ...;
                              #    encoders with CUEWIN_ENCODERS=arabert,...)
cuewin-decoder-fullsentence   # data/rev_decoder_fullsentence.csv
cuewin-revision-report        # data/revision_report.txt (summary of these analyses)
cuewin-crossed-cluster        # data/crossed_cluster.txt (subject x frame bootstrap,
                              #   repair by depth; CPU)
cuewin-attention-check        # data/attention_check.{csv,txt} (top subject head; GPU)
cuewin-controls               # data/ctrl_maskheads_<m>.csv.gz, ctrl_swap_<m>.csv: head output at [MASK], same-gender swap (GPU)
cuewin-controls-report        # data/controls_report.txt
cuewin-robustness-report      # data/robustness_report.txt (input-flip reference, T6 vs T7, thresholds)
cuewin-fixed-windows          # data/fixed_windows_report.txt (fixed windows 0-3/4-7/8-11,
                              #   crossed bootstrap incl. cue items, treebank; CPU)
cuewin-figures-rev            # fig2_patching, fig_residual, fig_repair_windows, fig_heads
```

Cue hidden-state patching, cumulative windows, held-out cue set, and
treebank repair with the gender-only patch:

```bash
cuewin-build-replication      # data/lexicon_rep.json, data/families_rep.jsonl (CPU)
cuewin-revision2              # GPU; stages rep_core, rep_heads, cumwin, cueresid, natural
                              #   (writes rev2_*.csv per model; skips existing outputs)
cuewin-rev2-tables            # data/rev2_tables.{tex,txt}, figures/fig_cueread.png (CPU)
cuewin-rev2-report            # data/rev2_report.txt (all summaries of these analyses; CPU)
```

Held-out items whose attractor, or construct-state (T7) subject, is one of the
eleven homograph professions (feminine form also an inanimate noun or adjective) are
excluded at analysis time (550 of 4,800 items; `fixed_windows.homograph_item`).

The released `rev2_*.csv` files keep the margins (`delta_*`) at seven significant
digits and omit columns that can be recovered from the item files (subject,
frame, cue item, probabilities); `score` is recomputed as `delta_clean - delta_patched`.

* `annotate` adds MARBERT (`UBC-NLP/MARBERT`) token alignment to the frozen
  benchmark (`data/families_rev.jsonl`; same four assertions as ARBERT).
* `windows` scores gender-only patching, zeroing, and a control word over
  single layers and every 4-layer window on T3/T6 (`rev_windows_<model>.csv`).
* `residual` patches the subject span or the `[MASK]` position in the residual
  stream at every layer (`rev_residual_<model>.csv`).
* `heads` patches one attention head's value slice at a time
  (`rev_heads_<model>.csv`).

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
cuewin/
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

All random generators are seeded. Item-level analyses and permutation tests use
seed 42; repair-rate intervals and head-map permutations 43; the attention check 45;
`cuewin-fixed-windows` 47 (crossed) and 48 (item); treebank score intervals 49;
`cuewin-rev2-tables` 51 (crossed), 52 (treebank repair) and 53 (check without
nationality adjectives); the held-out subject splits 100-119.
Figures are rendered by `matplotlib 3.8.2`; a different version changes the
rendered output but not the underlying numbers.

## License
MIT.