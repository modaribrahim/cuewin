"""Smoke tests for the CPU-only analysis stages.

These verify that the committed result CSVs in ``data/`` reproduce the exact
paper numbers, without a GPU or model weights. They exercise the same code
paths as the ``cuewin-stats-report`` and ``cuewin-enc-dec-compare`` commands.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from cuewin import stats
from cuewin.config import data_dir


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir(), name))


def test_data_directory_resolves():
    assert os.path.isdir(data_dir()), f"data dir not found: {data_dir()}"


def test_boot_ci_interval_contains_point_estimate():
    values = np.arange(1, 21, dtype=float)
    est, lo, hi = stats.boot_ci(list(values), rng=np.random.default_rng(42))
    assert lo <= est <= hi


def test_permutation_p_is_in_unit_interval():
    p = stats.paired_perm_p([1.0, 2.0, 3.0, 4.0], rng=np.random.default_rng(42))
    assert 0.0 <= p <= 1.0


def test_committed_stats_summary_has_all_claims():
    df = load("stats_summary.csv")
    claims = {f"C{i}" for i in range(1, 8)}
    assert claims.issubset(set(df["claim"]))


def test_stats_summary_arbert_word_order_null():
    # Paper: ARBERT's T4-T1 subject-weight effect is null (-0.010, p=0.53).
    df = load("stats_summary.csv")
    row = df[(df["claim"] == "C5") & (df["model"] == "arbert")].iloc[0]
    assert abs(float(row["estimate"]) - (-0.010)) < 0.001
    assert float(row["p"]) > 0.5


def test_stats_summary_arabert_subject_adj_gap():
    # Paper [C1]: subject-adjective causal gap at L7-11 = +0.26 for AraBERT.
    df = load("stats_summary.csv")
    row = df[(df["claim"] == "C1") & (df["model"] == "arabert")].iloc[0]
    assert 0.26 <= float(row["estimate"]) <= 0.27


def test_repair_rates_match_paper():
    df = load("intervention_summary.csv")
    row = df[(df["model"] == "arabert") & (df["condition"] == "T6")].iloc[0]
    assert abs(float(row["repair_rate_L5-11"]) - 0.441) < 0.001
    assert abs(float(row["control_rate_L0-3"]) - 0.117) < 0.001


def test_decoder_vso_collapse():
    # Paper: encoder overall accuracy in VSO is ~73-83%, decoder collapses to 51%.
    df = load("enc_dec_compare.csv")
    row = df[df["condition"] == "T1"].iloc[0]
    # decoder below both encoders on SVO (Table 6: 71.8 vs 88.9 and 78.0)
    assert float(row["aragpt2"]) < min(float(row["arabert"]), float(row["camelbert"]))
    t4 = df[df["condition"] == "T4"].iloc[0]
    assert float(t4["aragpt2"]) < 55.0       # next-token VSO near chance

def test_revision_windows_reproduce_committed_repair():
    """The revision repair sweep re-runs the original L5-11 / L0-3 zeroing;
    it must reproduce the committed intervention margins."""
    path = os.path.join(data_dir(), "rev_windows_arabert.csv")
    if not os.path.exists(path):
        return
    w = pd.read_csv(path)
    ir = load("intervention_results.csv")
    ir = ir[ir.model == "arabert"].set_index("id")
    for window, col in (("L5-11", "delta_neut"), ("L0-3", "delta_ctrl")):
        z = w[(w.window == window) & (w.kind == "zero")].set_index("id")
        diff = (z.delta_int - ir.loc[z.index, col]).abs().max()
        assert diff < 1e-3, (window, diff)


def test_revision_patch_single_layer_matches_value_patching():
    """Gender-only patching at one layer equals the per-cue patching score."""
    path = os.path.join(data_dir(), "rev_windows_arabert.csv")
    if not os.path.exists(path):
        return
    w = pd.read_csv(path)
    p = w[(w.window == "L9") & (w.kind == "patch") & (w.condition == "T6")].set_index("id")
    pr = load("patching_results.csv")
    pr = pr[(pr.model == "arabert") & (pr.cue == "attractor") & (pr.layer == 9)].set_index("id")
    diff = ((p.delta_clean - p.delta_int) - pr.loc[p.index, "score"]).abs().max()
    assert diff < 1e-3, diff
