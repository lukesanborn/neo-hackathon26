"""Shared evaluation contract (numpy only; used by baselines, models and aggregation).

Per held-out intervention we compare a predicted mean profile against the observed mean
profile of that intervention's treated cells. Both are converted to change-from-control
using the SAME control reference. Metrics are computed per intervention and then averaged
with equal weight per intervention.
"""

from __future__ import annotations

import math

import numpy as np

DIRECTION_THRESHOLD = 0.1  # |observed delta| (log-expression units) required to score a direction


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r; NaN (undefined) when either vector is constant."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float((a * b).sum() / (na * nb))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt((d * d).mean()))


def gears_metrics(pred: np.ndarray, obs: np.ndarray, ctrl: np.ndarray, de_nd_idx: list[int],
                  de_rank20: list[int] | None) -> dict:
    """GEARS-compatible mean-based metrics for one perturbation (gears/inference.py conventions).

    pred / obs are the perturbation's mean predicted / observed profiles (absolute expression);
    ctrl is the mean of ALL control cells. Gene sets:
      * all genes;
      * ``de_rank20``: top-20 of ``uns['rank_genes_groups_cov_all'][condition_name]`` (GEARS "_de");
        None (cache predates the supplement) -> NaN;
      * ``de_nd_idx``: ``uns['top_non_dropout_de_20'][condition_name]`` (GEARS "top20_de_non_dropout").
    Delta variants subtract ctrl from both vectors before the Pearson. Undefined -> NaN.
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    ctrl = np.asarray(ctrl, dtype=np.float64)
    pd_, od = pred - ctrl, obs - ctrl
    nd = np.asarray(de_nd_idx, dtype=int)
    nan = float("nan")
    if de_rank20 is None:
        p_de = d_de = nan
    else:
        rk = np.asarray(de_rank20, dtype=int)
        p_de, d_de = pearson(pred[rk], obs[rk]), pearson(pd_[rk], od[rk])
    return {
        "gears_pearson": pearson(pred, obs),
        "gears_pearson_de": p_de,
        "gears_top20_de_nd": pearson(pred[nd], obs[nd]),
        "gears_delta_pearson": pearson(pd_, od),
        "gears_delta_pearson_de": d_de,
        "gears_delta_top20_de_nd": pearson(pd_[nd], od[nd]),
    }


def intervention_metrics(pred: np.ndarray, obs: np.ndarray, ctrl: np.ndarray,
                         de_idx: list[int], de_rank20: list[int] | None = None) -> dict:
    pd_, od = pred - ctrl, obs - ctrl
    de = np.asarray(de_idx, dtype=int)
    strong = de[np.abs(od[de]) > DIRECTION_THRESHOLD]
    if len(strong):
        direction = float((np.sign(pd_[strong]) == np.sign(od[strong])).mean())
    else:
        direction = float("nan")
    return {
        "delta_rmse": rmse(pd_, od),             # primary; == absolute RMSE under a common control
        "delta_pearson": pearson(pd_, od),
        "de20_delta_rmse": rmse(pd_[de], od[de]),
        "de20_delta_pearson": pearson(pd_[de], od[de]),
        "de20_direction_acc": direction,
        "de20_n_direction_scored": int(len(strong)),
        "abs_pearson": pearson(pred, obs),
        **gears_metrics(pred, obs, ctrl, de_idx, de_rank20),
    }


METRIC_KEYS = ["delta_rmse", "delta_pearson", "de20_delta_rmse", "de20_delta_pearson",
               "de20_direction_acc", "abs_pearson"]
# GEARS-compatible mean-based metrics (absolute + delta), reported alongside, never replacing,
# the keys above. gears_pearson == abs_pearson and gears_delta_top20_de_nd == de20_delta_pearson
# by construction (same control reference, same gene set) — kept separately for comparability.
GEARS_KEYS = ["gears_pearson", "gears_pearson_de", "gears_top20_de_nd",
              "gears_delta_pearson", "gears_delta_pearson_de", "gears_delta_top20_de_nd"]
GEARS_DE_MISSING_NOTE = ("gears_pearson_de / gears_delta_pearson_de undefined: cache has no de_rank20.json "
                         "(run modal_reference.py to add it)")


def evaluate_predictions(preds: dict, ds, conditions: list[str], ctrl_ref: np.ndarray) -> list[dict]:
    rows = []
    de_rank = getattr(ds, "de_rank20", None) or {}
    for c in conditions:
        m = intervention_metrics(np.asarray(preds[c]), ds.observed_mean(c), ctrl_ref, ds.de_idx[c],
                                 de_rank.get(c))
        rows.append({"condition": c, "n_cells": int(len(ds.cells_of(c))), **m})
    return rows


def summarize(rows: list[dict]) -> dict:
    """Equal-weight mean over interventions. Undefined (NaN) values are reported, not zeroed."""
    out = {"n_interventions": len(rows)}
    # GEARS keys are summarised when present (rows from older code paths may lack them).
    keys = METRIC_KEYS + [k for k in GEARS_KEYS if rows and all(k in r for r in rows)]
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        ok = ~np.isnan(vals)
        # Mean over defined interventions only; coverage is always reported alongside so a
        # partially-defined metric is never silently compared with a fully-defined one.
        out[k] = float(vals[ok].mean()) if ok.any() else float("nan")
        out[f"{k}_n_defined"] = int(ok.sum())
    if "gears_pearson_de" in out and out["gears_pearson_de_n_defined"] == 0 and rows:
        out["gears_note"] = GEARS_DE_MISSING_NOTE
    return out


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI of the mean, resampling *interventions* (not cells)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """CI of mean(a - b) over interventions that are defined for both."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    lo, hi = bootstrap_mean_ci(d, n_boot=n_boot, seed=seed)
    return {"mean_diff": float(d.mean()) if len(d) else math.nan, "ci_low": lo, "ci_high": hi,
            "n": int(ok.sum()), "frac_a_lower": float((d < 0).mean()) if len(d) else math.nan}
