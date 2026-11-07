"""Predicted-effect retrieval scoring: per-gene scores, seed pooling, paired bootstrap CIs.

Pure numpy/scipy (no modal, no torch) so it is importable in tests. Used by diag/retrieval.py.

Per test gene i (P, T: predicted / true deltas of the n test genes; R: true deltas of the reference
(training) perturbations; all restricted to the same gene panel):
  ref_spearman  Spearman between cos(R, P_i) and cos(R, T_i)  (NaN if the prediction is constant w.r.t. R)
  ref_p@5       |top-5 of R by P_i ∩ top-5 of R by T_i| / 5
  id_hit        1 if T_i is the closest (cosine) true test delta to P_i, else 0  (rank ties averaged)
  id_rank       rank of T_i among all n true test deltas by cosine to P_i (1 = best)
cluster_agreement is one number per prediction set: Pearson between the upper triangles of the
cosine-similarity matrices of P and of T over the test genes.
"""
import re

import numpy as np

PER_GENE = ("ref_spearman", "ref_p@5", "id_hit", "id_rank")
# reported metric name -> per-gene key whose mean it is
MEAN_OF = {"ref_spearman": "ref_spearman", "ref_p@5": "ref_p@5", "id_top1": "id_hit", "id_mean_rank": "id_rank"}
METRICS = ("ref_spearman", "ref_p@5", "cluster_agreement", "id_top1", "id_mean_rank")


def unit(A):
    A = np.asarray(A, dtype=np.float64)
    return A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)


def per_gene_scores(P, T, R):
    """Returns ({per-gene key: (n,) array}, S_pred (n,n) cosine-sim matrix of predictions)."""
    from scipy.stats import rankdata, spearmanr

    Pu, Tu, Ru = unit(P), unit(T), unit(R)
    n = len(Pu)
    sp, p5 = np.full(n, np.nan), np.zeros(n)
    for i in range(n):
        a, b = Ru @ Pu[i], Ru @ Tu[i]
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            sp[i] = spearmanr(a, b).statistic
        p5[i] = len(set(np.argsort(-a)[:5]) & set(np.argsort(-b)[:5])) / 5
    S = Pu @ Tu.T                                         # prediction i vs true j
    ranks = np.array([rankdata(-S[i])[i] for i in range(n)], dtype=np.float64)
    return {"ref_spearman": sp, "ref_p@5": p5, "id_hit": (ranks == 1).astype(np.float64), "id_rank": ranks}, Pu @ Pu.T


def true_sim(T):
    Tu = unit(T)
    return Tu @ Tu.T


def cluster_agreement(S_pred, S_true, idx=None):
    """Pearson of upper-triangle similarities. idx: optional (possibly repeated) gene indices of a bootstrap
    resample; pairs made of two copies of the same gene are dropped (their self-similarity of 1 would inflate
    the correlation). NaN when either side is constant (e.g. a constant prediction) or < 3 pairs remain."""
    if idx is None:
        idx = np.arange(len(S_true))
    idx = np.asarray(idx)
    a, b = np.triu_indices(len(idx), 1)
    keep = idx[a] != idx[b]
    ia, ib = idx[a][keep], idx[b][keep]
    if len(ia) < 3:
        return float("nan")
    x, y = S_pred[ia, ib], S_true[ia, ib]
    if np.std(x) <= 1e-9 or np.std(y) <= 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _nanmean(x, axis=None):
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        cnt = np.sum(~np.isnan(x), axis=axis)
        s = np.nansum(x, axis=axis)
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def aggregate(pg, clus):
    """Old-style aggregate dict for one prediction set (or one seed-pooled arm)."""
    out = {m: float(_nanmean(pg[k])) for m, k in MEAN_OF.items()}
    out["cluster_agreement"] = float(clus)
    out["n_test"] = int(len(pg["id_rank"]))
    return out


def arm_of(name):
    """Seed-pooled arm name: drop a `_s<1-2 digits>` seed token (followed by `_` or end) from the last path
    component only (so experiment names and size tags like `_s512` are untouched)."""
    head, _, last = name.rpartition("/")
    last = re.sub(r"_s\d{1,2}(?=_|$)", "", last)
    return f"{head}/{last}" if head else last


def json_safe(x):
    """Recursively convert numpy types to Python and NaN/inf to None (strict JSON)."""
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def pool_seeds(sets):
    """sets: {set_name: {"per_gene": {...}, "S_pred": (n,n)}} all over the same ordered test genes.
    Returns {arm: {"per_gene": seed-averaged per-gene scores (NaN-aware), "S_pred": [per-seed matrices],
    "members": [set names]}}."""
    arms = {}
    for name in sorted(sets):
        a = arms.setdefault(arm_of(name), {"members": [], "_pg": [], "S_pred": []})
        a["members"].append(name)
        a["_pg"].append(sets[name]["per_gene"])
        a["S_pred"].append(np.asarray(sets[name]["S_pred"]))
    for a in arms.values():
        pgs = a.pop("_pg")
        a["per_gene"] = {k: _nanmean(np.stack([p[k] for p in pgs]), axis=0) for k in PER_GENE}
    return arms


def _arm_cluster(S_preds, S_true, idx=None):
    vals = [cluster_agreement(S, S_true, idx) for S in S_preds]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_draws(arms, S_true, n_boot=2000, seed=0):
    """Paired bootstrap over test genes: the same resampled gene sets are used for every arm.
    Returns (point {arm: {metric: value}}, draws {arm: {metric: (n_boot,) array}}). Per-seed-averaged per-gene
    scores are resampled; cluster_agreement is recomputed per seed on the resampled sub-matrix and averaged
    over seeds (NaN draws = undefined, skipped in CIs)."""
    n = len(S_true)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    point, draws = {}, {}
    for arm, a in arms.items():
        pg = a["per_gene"]
        point[arm] = {m: float(_nanmean(pg[k])) for m, k in MEAN_OF.items()}
        point[arm]["cluster_agreement"] = _arm_cluster(a["S_pred"], S_true)
        draws[arm] = {m: _nanmean(pg[k][idx], axis=1) for m, k in MEAN_OF.items()}
        draws[arm]["cluster_agreement"] = np.array([_arm_cluster(a["S_pred"], S_true, ix) for ix in idx])
    return point, draws


def ci(x, level=0.95):
    """Percentile CI over finite draws; returns (lo, hi, n_valid)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), 0
    q = (1 - level) / 2
    return float(np.quantile(x, q)), float(np.quantile(x, 1 - q)), int(len(x))


def summarize_ci(point, draws):
    return {arm: {m: dict(zip(("mean", "lo", "hi", "n_valid"), (point[arm][m], *ci(draws[arm][m]))))
                  for m in METRICS} for arm in point}


def diff_ci(point, draws, a, b):
    """Paired difference a − b per metric: point diff, 95% CI and fraction of valid resamples with diff > 0."""
    out = {}
    for m in METRICS:
        d = draws[a][m] - draws[b][m]
        lo, hi, nv = ci(d)
        valid = d[~np.isnan(d)]
        out[m] = {"diff": point[a][m] - point[b][m], "lo": lo, "hi": hi, "n_valid": nv,
                  "frac_gt0": float(np.mean(valid > 0)) if nv else float("nan")}
    return out
