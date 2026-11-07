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


def intervention_metrics(pred: np.ndarray, obs: np.ndarray, ctrl: np.ndarray,
                         de_idx: list[int]) -> dict:
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
    }


METRIC_KEYS = ["delta_rmse", "delta_pearson", "de20_delta_rmse", "de20_delta_pearson",
               "de20_direction_acc", "abs_pearson"]


def evaluate_predictions(preds: dict, ds, conditions: list[str], ctrl_ref: np.ndarray) -> list[dict]:
    rows = []
    for c in conditions:
        m = intervention_metrics(np.asarray(preds[c]), ds.observed_mean(c), ctrl_ref, ds.de_idx[c])
        rows.append({"condition": c, "n_cells": int(len(ds.cells_of(c))), **m})
    return rows


def summarize(rows: list[dict]) -> dict:
    """Equal-weight mean over interventions. Undefined (NaN) values are reported, not zeroed."""
    out = {"n_interventions": len(rows)}
    for k in METRIC_KEYS:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        ok = ~np.isnan(vals)
        # Mean over defined interventions only; coverage is always reported alongside so a
        # partially-defined metric is never silently compared with a fully-defined one.
        out[k] = float(vals[ok].mean()) if ok.any() else float("nan")
        out[f"{k}_n_defined"] = int(ok.sum())
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
