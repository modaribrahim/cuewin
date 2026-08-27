"""Shared statistics: bootstrap CIs, permutation tests, cluster bootstrap, and
a statsmodels-free OLS (for the dose-response regression).

Ported so the paper's numbers reproduce exactly.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_BOOT = 10_000
N_PERM = 10_000

# Cluster bootstrap uses its own RNG so item-level rows stay byte-identical.
CRNG = np.random.default_rng(43)


# ---------------------------------------------------------------------------
# item-level inference
# ---------------------------------------------------------------------------

def boot_ci(values, stat=np.mean, rng=RNG, n_boot=N_BOOT):
    values = np.asarray(values)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = stat(values[idx], axis=1)
    return stat(values), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def paired_perm_p(diffs, rng=RNG, n_perm=N_PERM):
    """H0: mean difference == 0, via random sign flips. Two-sided."""
    diffs = np.asarray(diffs)
    obs = abs(diffs.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
    null = np.abs((signs * diffs).mean(axis=1))
    return float((null >= obs).mean())


def unpaired_perm_p(a, b, rng=RNG, n_perm=N_PERM):
    """H0: mean(a) == mean(b), label permutation. Two-sided."""
    a, b = np.asarray(a), np.asarray(b)
    obs = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n = len(a)
    null = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(pooled)
        null[i] = abs(pooled[:n].mean() - pooled[n:].mean())
    return float((null >= obs).mean())


def item_layer_mean(df, layers):
    """Per-item mean score over a layer window."""
    sub = df[df.layer.isin(layers)]
    return sub.groupby("id").score.mean()


# ---------------------------------------------------------------------------
# cluster bootstrap (pseudoreplication robustness)
# ---------------------------------------------------------------------------

def _cluster_ci_single(vals_by_id, id2cluster, rng=CRNG, n_boot=N_BOOT):
    """Cluster bootstrap CI for the MEAN of a per-item quantity: resample
    clusters with replacement, pool all their items. Point estimate equals the
    plain item-level mean."""
    d = defaultdict(list)
    for i, v in vals_by_id.items():
        c = id2cluster.get(i)
        if c is not None:
            d[c].append(v)
    sums = np.array([np.sum(v) for v in d.values()], dtype=float)
    cnts = np.array([len(v) for v in d.values()], dtype=float)
    G = len(sums)
    idx = rng.integers(0, G, size=(n_boot, G))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    point = sums.sum() / cnts.sum()
    return point, np.percentile(boots, 2.5), np.percentile(boots, 97.5), G


def _cluster_ci_two(valsB_by_id, valsA_by_id, id2cluster, rng=CRNG, n_boot=N_BOOT):
    """Cluster bootstrap CI for mean(B) − mean(A) (two independent groups that
    share cluster structure, e.g. T4 vs T1 within a verb frame)."""
    dB, dA = defaultdict(list), defaultdict(list)
    for i, v in valsB_by_id.items():
        c = id2cluster.get(i)
        if c is not None:
            dB[c].append(v)
    for i, v in valsA_by_id.items():
        c = id2cluster.get(i)
        if c is not None:
            dA[c].append(v)
    clusters = sorted(set(dB) | set(dA))
    sB = np.array([np.sum(dB.get(c, [])) for c in clusters], dtype=float)
    nB = np.array([len(dB.get(c, [])) for c in clusters], dtype=float)
    sA = np.array([np.sum(dA.get(c, [])) for c in clusters], dtype=float)
    nA = np.array([len(dA.get(c, [])) for c in clusters], dtype=float)
    G = len(clusters)
    idx = rng.integers(0, G, size=(n_boot, G))
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = sB[idx].sum(1) / nB[idx].sum(1) - sA[idx].sum(1) / nA[idx].sum(1)
    boots = boots[np.isfinite(boots)]
    point = sB.sum() / nB.sum() - sA.sum() / nA.sum()
    return point, np.percentile(boots, 2.5), np.percentile(boots, 97.5), G


def _series_dict(series):
    """pd.Series indexed by id -> {id: float}."""
    return {i: float(v) for i, v in series.items()}


# ---------------------------------------------------------------------------
# statsmodels-free OLS — identical closed-form math
# ---------------------------------------------------------------------------

def ols_fit(x, y):
    """OLS y ~ [1, x]. Returns (slope, p_value) with the same closed-form
    computation statsmodels uses (X'X inverse, residual variance, t-dist)."""
    import scipy.stats as sps

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = len(y) - X.shape[1]
    sigma2 = resid @ resid / df
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma2 * np.diag(XtX_inv))
    t = beta / se
    p = 2 * sps.t.sf(np.abs(t), df)
    return float(beta[1]), float(p[1])


def ols_fit_multi(xcols, y):
    """OLS y ~ [1, xcols...]. Returns (beta, se, p) for the coefficient rows,
    matching statsmodels' closed-form computation. ``xcols`` is a sequence of
    column arrays; the returned arrays exclude the intercept."""
    import scipy.stats as sps

    xcols = [np.asarray(c, dtype=float) for c in xcols]
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y))] + xcols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = len(y) - X.shape[1]
    sigma2 = resid @ resid / df
    se = np.sqrt(sigma2 * np.diag(np.linalg.inv(X.T @ X)))
    t = beta / se
    p = 2 * sps.t.sf(np.abs(t), df)
    return beta[1:], se[1:], p[1:]


def add_constant(x):
    """Equivalent of ``sm.add_constant`` (a leading column of ones)."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([np.ones(len(x)), x])


def cluster_boot_ols(d, g_key, x_key, y_key, n_boot=500, rng=RNG):
    """OLS slope of y on x, cluster-sampled by ``g_key``. Returns
    (mean_slope, ci_lo, ci_hi). Clusters are enumerated BEFORE dropping NaN
    rows, matching the fixed RNG-consumption order."""
    clusters = list(d[g_key].unique())
    d = d[[g_key, x_key, y_key]].dropna()
    slopes = []
    for _ in range(n_boot):
        resample = []
        for c in rng.choice(clusters, size=len(clusters), replace=True):
            resample.append(d[d[g_key] == c])
        b = pd.concat(resample)
        if len(b) < 20:
            continue
        X = add_constant(b[x_key].values)
        y = b[y_key].values
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0][1]
        except np.linalg.LinAlgError:
            continue
        slopes.append(beta)
    slopes = np.array(slopes)
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return float(slopes.mean()), float(lo), float(hi)


def boot_ci_mean(values, n_boot=500, rng=RNG):
    """Bootstrap CI for a mean."""
    boots = [np.mean(rng.choice(values, size=len(values), replace=True))
             for _ in range(n_boot)]
    return np.mean(boots), np.percentile(boots, 2.5), np.percentile(boots, 97.5)
