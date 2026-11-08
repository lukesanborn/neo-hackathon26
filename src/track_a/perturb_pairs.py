"""Perturbation-pair JEPA views (attempts A28).

A27 showed JEPA continued pretraining trains knockdown information OUT of the embedding: with views
"masked cell vs the same full cell", JEPA spends its space on cell-to-cell differences. Here the two views are
DIFFERENT cells that received the same knockdown: the student sees cell i (masked, as usual), the teacher sees
k other cells of the same condition, and the JEPA target is their mean <cls>. Invariance is then to
cell-to-cell noise, and the shared signal the predictor must recover is the knockdown.

Corpus: Replogle K562 essential, TRAINING conditions only (incl. control cells, paired with other controls).
Validation and test knockdowns are never written to the cache. Cells keep the shard contract of the kidney
cache (``KidneyCache``) plus ``groups.npy`` (condition index per global cell) and ``groups.json``.
Values: per-cell quantile bins of the nonzero log1p values (binning is rank-based, as in fine-tuning's
``bins50`` input transform).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

PAIR_DEVIATION = ("JEPA views = perturbation pairs: student sees one masked cell, the EMA teacher sees k OTHER "
                  "cells with the same knockdown (training knockdowns only); target = mean teacher <cls> "
                  "(our A28 extension, not in Cell-JEPA)")


def build_perturb_cache(ds, out_dir: str, vocab: dict, aliases: dict, ctrl: str = "ctrl", val_frac: float = 0.02,
                        shard_size: int = 20000, seed: int = 0) -> dict:
    """Write a shard cache of all cells of ``ds.split['train']`` conditions. Returns the manifest."""
    from .census import coo_to_shard, write_json_atomic, write_npz_atomic

    conds = list(ds.split["train"])
    held = set(ds.split["val"]) | set(ds.split["test"])
    assert ctrl in conds and not (set(conds) & held), "train conditions must include ctrl and exclude val/test"
    cells, groups = [], []
    for j, c in enumerate(conds):
        idx = np.asarray(ds.cells_of(c))
        cells.append(idx)
        groups.append(np.full(len(idx), j, dtype=np.int32))
    cells, groups = np.concatenate(cells), np.concatenate(groups)
    order = np.random.default_rng(seed).permutation(len(cells))      # mix conditions across shards
    cells, groups = cells[order], groups[order]
    syms = [g if g in vocab else aliases.get(g, g) for g in ds.gene_names]
    col_vid = np.array([vocab.get(s, -1) for s in syms], dtype=np.int64)
    in_vocab = np.flatnonzero(col_vid >= 0)
    os.makedirs(os.path.join(out_dir, "shards"), exist_ok=True)
    n_shards = int(np.ceil(len(cells) / shard_size))
    for k in range(n_shards):
        rows_idx = cells[k * shard_size:(k + 1) * shard_size]
        X = np.asarray(ds.X[np.sort(rows_idx)], dtype=np.float32)
        X = X[np.argsort(np.argsort(rows_idx))]                      # back to rows_idx order
        X = X[:, in_vocab]
        r, c = np.nonzero(X > 0)
        sh = coo_to_shard(r, col_vid[in_vocab][c], X[r, c], len(rows_idx))
        write_npz_atomic(os.path.join(out_dir, "shards", f"shard_{k:03d}.npz"),
                         indptr=sh["indptr"], gene_ids=sh["gene_ids"], bins=sh["bins"], n_cells=sh["n_cells"])
    rng = np.random.default_rng([seed, 1])
    val = np.sort(rng.choice(len(cells), max(1, int(val_frac * len(cells))), replace=False))
    write_json_atomic(os.path.join(out_dir, "split.json"),
                      {"val_cells": [[int(g // shard_size), int(g % shard_size)] for g in val]})
    np.save(os.path.join(out_dir, "groups.npy"), groups)
    held_hash = hashlib.sha256(json.dumps(sorted(held)).encode()).hexdigest()[:12]
    meta = {"conditions": conds, "ctrl": ctrl, "n_conditions": len(conds),
            "excluded_heldout_conditions": len(held), "excluded_heldout_sha": held_hash}
    write_json_atomic(os.path.join(out_dir, "groups.json"), meta)
    man = {"corpus": "replogle_k562_essential_train_conditions", "source_cache": getattr(ds, "cache_dir", None),
           "split_hash": (ds.manifest or {}).get("split_hash"), "n_cells": int(len(cells)), "n_shards": n_shards,
           "n_val": int(len(val)), "genes_in_vocab": int(len(in_vocab)), "genes_total": len(ds.gene_names),
           "values": "per-cell quantile bins 1..50 of nonzero log1p values", **meta}
    man.pop("conditions")
    write_json_atomic(os.path.join(out_dir, "manifest.json"), man)
    return man


class PairSampler:
    """Draws, for each anchor cell, k partner cells of the same group from a fixed pool (train or val cells).
    Partners exclude the anchor itself when the group has another cell in the pool."""

    def __init__(self, groups: np.ndarray, pool: np.ndarray):
        pool = np.asarray(pool, dtype=np.int64)
        g = np.asarray(groups)[pool]
        order = np.argsort(g, kind="stable")
        pool, g = pool[order], g[order]
        bounds = np.flatnonzero(np.diff(g)) + 1
        self.members = {int(gg[0]): pp for gg, pp in zip(np.split(g, bounds), np.split(pool, bounds))}
        self.groups = np.asarray(groups)

    def stats(self) -> dict:
        sizes = np.array([len(v) for v in self.members.values()])
        return {"n_groups": int(len(sizes)), "min_group": int(sizes.min()), "median_group": float(np.median(sizes)),
                "n_singleton_groups": int((sizes == 1).sum())}

    def sample(self, cells: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        out = np.empty((len(cells), k), dtype=np.int64)
        for i, c in enumerate(np.asarray(cells)):
            m = self.members[int(self.groups[c])]
            if len(m) > 1:
                cand = m[rng.integers(0, len(m), size=k)]
                bad = cand == c                       # resample draws of the anchor itself
                while bad.any():
                    cand[bad] = m[rng.integers(0, len(m), size=int(bad.sum()))]
                    bad = cand == c
                out[i] = cand
            else:
                out[i] = m[0]
        return out.reshape(-1)
