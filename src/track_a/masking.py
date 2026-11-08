"""Mask units for JEPA pretraining ("views" knobs 2a / 2b). numpy (+ scipy for program building) only.

Knob 2a, what unit gets masked (``PretrainConfig.mask_unit``):
  * "random"        : scattered genes (Cell-JEPA / scGPT). Handled by ``pretrain_model.random_value_mask``;
                      this module is not used, so random runs stay bit-identical to earlier checkpoints.
  * "program"       : whole co-expression programs. Genes are clustered once from the pretraining
                      cache's TRAINING cells (``build_programs``); a cell's mask is the union of a few
                      programs' genes that are present in that cell.
  * "pathway"       : whole curated gene sets (e.g. MSigDB Hallmark). Sets overlap; genes in no set are
                      never masked as a unit.
  * "top_expressed" : the cell's highest-expressed genes (largest bins, random tie-break).

Knob 2b, how much: ``mask_ratio``. For unit masks it is a target, not an exact count: whole units are
added in random order until at least ``max(1, int(ratio * n))`` tokens are masked, so the last unit
may overshoot. At least one token always stays visible. If a cell contains no gene of any unit, the
cell falls back to a random mask (counted, so it can be reported). The realised fraction is logged
by the pretraining loop as ``mask_frac``.

Units file (JSON, written by ``modal_pretrain.build_mask_units``):
  {"kind": "program" | "pathway", "units": [[vocab_id, ...], ...], "unit_names": [...], "source": ..., "stats": ...}
"""

from __future__ import annotations

import hashlib
import json
from functools import partial

import numpy as np

MASK_UNITS = ("random", "program", "pathway", "top_expressed")
UNIT_FILE_KINDS = ("program", "pathway")
MASK_UNIT_DESCRIPTIONS = {
    "program": "views knob 2a: mask whole co-expression programs (genes clustered on training cells of the "
               "pretraining cache) instead of random genes; mask_ratio is a target (last unit may overshoot)",
    "pathway": "views knob 2a: mask whole curated gene sets (pathways) instead of random genes; genes in no set "
               "are never masked as a unit; mask_ratio is a target (last unit may overshoot)",
    "top_expressed": "views knob 2a: mask each cell's highest-expressed genes (largest bins) instead of random genes",
}


# ----------------------------------------------------------------------------- unit files


def load_units(path: str) -> dict:
    """Read a units file. Returns {"kind", "units": list[np.ndarray], "gene_units": {gene_id: tuple[unit idx]},
    "sha256", "n_units", "n_genes"}."""
    with open(path, "rb") as f:
        raw = f.read()
    d = json.loads(raw)
    if d.get("kind") not in UNIT_FILE_KINDS:
        raise ValueError(f"{path}: kind must be one of {UNIT_FILE_KINDS}, got {d.get('kind')!r}")
    units = [np.asarray(u, dtype=np.int64) for u in d["units"]]
    units = [u for u in units if len(u)]
    if not units:
        raise ValueError(f"{path}: no non-empty units")
    gene_units: dict[int, list[int]] = {}
    for i, u in enumerate(units):
        for g in np.unique(u):
            gene_units.setdefault(int(g), []).append(i)
    return {"kind": d["kind"], "units": units, "gene_units": {g: tuple(v) for g, v in gene_units.items()},
            "sha256": hashlib.sha256(raw).hexdigest(), "n_units": len(units), "n_genes": len(gene_units),
            "source": d.get("source"), "stats": d.get("stats")}


def parse_gene_sets(text: str) -> dict[str, list[str]]:
    """GMT or Enrichr text library: ``name <TAB> description <TAB> gene <TAB> gene ...`` per line.
    Enrichr may append weights (``GENE,1.0``); they are dropped. Symbols are upper-cased."""
    sets = {}
    for line in text.splitlines():
        f = line.rstrip("\n").split("\t")
        if len(f) < 3 or not f[0].strip():
            continue
        genes = [g.split(",")[0].strip().upper() for g in f[2:]]
        genes = sorted({g for g in genes if g})
        if genes:
            sets[f[0].strip()] = genes
    return sets


def gene_sets_to_units(sets: dict[str, list[str]], stoi: dict[str, int], min_size: int = 5) -> tuple[list, list, dict]:
    """Map symbol sets to vocab-id units; drop sets with fewer than ``min_size`` genes in the vocab."""
    upper = {k.upper(): v for k, v in stoi.items()}
    units, names, n_missing = [], [], 0
    for name, genes in sets.items():
        ids = sorted({upper[g] for g in genes if g in upper})
        n_missing += sum(g not in upper for g in genes)
        if len(ids) >= min_size:
            units.append(ids)
            names.append(name)
    stats = {"n_sets_in": len(sets), "n_units": len(units), "n_symbols_not_in_vocab": n_missing,
             "n_unique_genes": len({g for u in units for g in u})}
    return units, names, stats


# ----------------------------------------------------------------------------- program building


def _randomized_svd(matvec, rmatvec, n_cols: int, rank: int, rng: np.random.Generator, n_iter: int = 3):
    """Top-``rank`` SVD of an implicit (n x n_cols) matrix A given A @ M and A.T @ M."""
    k = rank + 10
    Q, _ = np.linalg.qr(matvec(rng.standard_normal((n_cols, k))))
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(rmatvec(Q))
        Q, _ = np.linalg.qr(matvec(Q))
    B = rmatvec(Q).T                                  # (k, n_cols) = Q.T A
    _, S, Vt = np.linalg.svd(B, full_matrices=False)
    return S[:rank], Vt[:rank]


def _spherical_kmeans(E: np.ndarray, k: int, rng: np.random.Generator, n_iter: int = 50) -> np.ndarray:
    """k-means on unit vectors with cosine similarity, k-means++ init. Returns labels (n,)."""
    n = len(E)
    centers = [E[rng.integers(n)]]
    d = 1.0 - E @ centers[0]
    for _ in range(1, k):
        p = np.clip(d, 0, None) ** 2
        p = p / p.sum() if p.sum() > 0 else np.full(n, 1.0 / n)
        c = E[rng.choice(n, p=p)]
        centers.append(c)
        d = np.minimum(d, 1.0 - E @ c)
    C = np.stack(centers)
    labels = np.full(n, -1)
    for _ in range(n_iter):
        new = np.argmax(E @ C.T, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                v = E[m].sum(0)
                C[j] = v / max(np.linalg.norm(v), 1e-12)
            else:  # empty cluster: reseed on the point worst served by its centre
                worst = int(np.argmin((E * C[labels]).sum(1)))
                C[j] = E[worst]
                labels[worst] = j
    return labels


def build_programs(cells: list[tuple[np.ndarray, np.ndarray]], n_programs: int = 64, rank: int = 50,
                   min_cells: int = 20, seed: int = 0) -> dict:
    """Co-expression programs from (gene_ids, bins) cells.

    Genes detected in at least ``min_cells`` cells are standardised (expression bins, zeros included),
    embedded with a randomized SVD of the cells x genes matrix (rows of V * S ~ the gene-gene
    correlation structure), L2-normalised and clustered with spherical k-means. Returns a units-file
    dict (kind "program")."""
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    rows = np.concatenate([np.full(len(g), i) for i, (g, _) in enumerate(cells)])
    gids = np.concatenate([g for g, _ in cells]).astype(np.int64)
    vals = np.concatenate([b for _, b in cells]).astype(np.float64)
    uniq, col = np.unique(gids, return_inverse=True)
    X = sp.csr_matrix((vals, (rows, col)), shape=(len(cells), len(uniq)))
    det = np.asarray((X > 0).sum(0)).ravel()
    keep = np.flatnonzero(det >= min_cells)
    if len(keep) < n_programs:
        raise ValueError(f"only {len(keep)} genes detected in >= {min_cells} cells; need >= n_programs={n_programs}")
    X = X[:, keep].tocsc().tocsr()
    n = X.shape[0]
    mu = np.asarray(X.mean(0)).ravel()
    sq = np.asarray(X.multiply(X).mean(0)).ravel()
    sd = np.sqrt(np.clip(sq - mu ** 2, 1e-12, None))
    ones = np.ones(n)

    def matvec(M):    # Z @ M with Z = (X - mu) / sd
        Ms = M / sd[:, None]
        return X @ Ms - np.outer(ones, mu @ Ms)

    def rmatvec(Q):   # Z.T @ Q
        return ((X.T @ Q) - np.outer(mu, ones @ Q)) / sd[:, None]

    rank = min(rank, len(keep) - 1, n - 1)
    S, Vt = _randomized_svd(matvec, rmatvec, len(keep), rank, rng)
    E = (Vt.T * S)
    E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
    labels = _spherical_kmeans(E, n_programs, rng)
    genes = uniq[keep]
    units = [sorted(genes[labels == j].tolist()) for j in range(n_programs)]
    units = [u for u in units if u]
    sizes = np.array([len(u) for u in units])
    tok_cov = float(np.isin(gids, genes).mean())
    return {"kind": "program", "units": units, "unit_names": [f"program_{j:03d}" for j in range(len(units))],
            "stats": {"n_cells_used": n, "n_genes_detected": int(len(uniq)), "n_genes_kept": int(len(keep)),
                      "min_cells": min_cells, "rank": int(rank), "n_programs": len(units),
                      "program_size_min_median_max": [int(sizes.min()), int(np.median(sizes)), int(sizes.max())],
                      "token_coverage": tok_cov, "seed": seed}}


def token_coverage(units: list, cells: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Fraction of gene-token occurrences in ``cells`` that belong to at least one unit."""
    covered = np.unique(np.concatenate([np.asarray(u, dtype=np.int64) for u in units]))
    g = np.concatenate([c[0] for c in cells]).astype(np.int64)
    return float(np.isin(g, covered).mean()) if len(g) else 0.0


# ----------------------------------------------------------------------------- maskers


def _n_target(n: int, ratio: float) -> int:
    return max(1, int(ratio * n))


def unit_value_mask(gene_ids: np.ndarray, values: np.ndarray, n_genes: np.ndarray, ratio: float,
                    rng: np.random.Generator, gene_units: dict) -> tuple[np.ndarray, int]:
    """(B, L) mask over gene tokens 1..n_genes[b] made of whole units. Returns (mask, n_fallback_rows)."""
    B, L = gene_ids.shape
    out = np.zeros((B, L), dtype=bool)
    fallback = 0
    for b in range(B):
        n = int(n_genes[b])
        if n == 0:
            continue
        k, cap = _n_target(n, ratio), max(1, n - 1)
        pos_of_unit: dict[int, list[int]] = {}
        for p in range(1, 1 + n):
            for u in gene_units.get(int(gene_ids[b, p]), ()):
                pos_of_unit.setdefault(u, []).append(p)
        count = 0
        if pos_of_unit:
            us = np.array(sorted(pos_of_unit))
            for u in us[rng.permutation(len(us))]:
                new = [p for p in pos_of_unit[int(u)] if not out[b, p]]
                if not new or (count and count + len(new) > cap):
                    continue
                if count + len(new) > cap:          # first unit alone is too big: take part of it
                    new = list(rng.choice(new, size=cap, replace=False))
                out[b, new] = True
                count += len(new)
                if count >= k:
                    break
        if count == 0:
            fallback += 1
            out[b, 1 + rng.choice(n, size=min(k, cap), replace=False)] = True
    return out, fallback


def top_expressed_mask(gene_ids: np.ndarray, values: np.ndarray, n_genes: np.ndarray, ratio: float,
                       rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Mask each cell's ``max(1, int(ratio * n))`` largest values (random tie-break)."""
    B, L = values.shape
    out = np.zeros((B, L), dtype=bool)
    for b in range(B):
        n = int(n_genes[b])
        if n == 0:
            continue
        v = values[b, 1:1 + n]
        order = np.lexsort((rng.random(n), -v))        # primary key: value descending
        out[b, 1 + order[:_n_target(n, ratio)]] = True
    return out, 0


def make_masker(mask_unit: str, units_path: str | None = None):
    """Returns (masker or None, info). None = the original random masking path."""
    if mask_unit not in MASK_UNITS:
        raise ValueError(f"mask_unit must be one of {MASK_UNITS}, got {mask_unit!r}")
    if mask_unit == "random":
        return None, {"mask_unit": "random"}
    if mask_unit == "top_expressed":
        return top_expressed_mask, {"mask_unit": "top_expressed"}
    if not units_path:
        raise ValueError(f"mask_unit={mask_unit!r} needs mask_units_path")
    u = load_units(units_path)
    if u["kind"] != mask_unit:
        raise ValueError(f"{units_path} has kind {u['kind']!r}, expected {mask_unit!r}")
    info = {"mask_unit": mask_unit, "mask_units_path": units_path, "units_sha256": u["sha256"],
            "n_units": u["n_units"], "n_unit_genes": u["n_genes"], "units_stats": u["stats"]}
    return partial(unit_value_mask, gene_units=u["gene_units"]), info
