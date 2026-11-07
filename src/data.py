"""Data contract: download, manifest, fixed intervention-level split, caches.

Everything here runs inside Modal (`prepare`). The cache written to the Volume is
consumed read-only by every training / evaluation job so all methods share one
data version, one split and one control-sampling rule.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass

import numpy as np

ADAMSON_URL = "https://dataverse.harvard.edu/api/access/datafile/6154417"  # from gears 0.0.2 pertdata.py
SCGPT_HUMAN_FOLDER = "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"

CTRL = "ctrl"


def sha256(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def target_gene(condition: str) -> str | None:
    """'AARS+ctrl' -> 'AARS'; 'ctrl' -> None. Adamson has single-gene perturbations only."""
    if condition == CTRL:
        return None
    genes = [g for g in condition.split("+") if g != CTRL]
    if len(genes) != 1:
        raise ValueError(f"expected single-gene condition, got {condition!r}")
    return genes[0]


# ----------------------------------------------------------------------------- download


def download_adamson(raw_dir: str) -> dict:
    os.makedirs(raw_dir, exist_ok=True)
    zpath = os.path.join(raw_dir, "adamson.zip")
    fresh = False
    if not os.path.exists(zpath):
        subprocess.run(["wget", "-q", "-O", zpath, ADAMSON_URL], check=True)
        fresh = True
    h5 = os.path.join(raw_dir, "adamson", "perturb_processed.h5ad")
    if fresh or not os.path.exists(h5):
        subprocess.run(["unzip", "-o", "-q", zpath, "-d", raw_dir], check=True)
    return {
        "source_url": ADAMSON_URL,
        "zip_path": zpath,
        "zip_sha256": sha256(zpath),
        "h5ad_path": h5,
        "h5ad_sha256": sha256(h5),
        "download_date_utc": _dt.datetime.utcfromtimestamp(os.path.getmtime(zpath)).isoformat(),
    }


def download_scgpt(ckpt_dir: str) -> dict:
    os.makedirs(ckpt_dir, exist_ok=True)
    needed = ["args.json", "best_model.pt", "vocab.json"]
    if not all(os.path.exists(os.path.join(ckpt_dir, f)) for f in needed):
        subprocess.run(["gdown", "--folder", SCGPT_HUMAN_FOLDER, "-O", ckpt_dir], check=True)
    return {
        "source": SCGPT_HUMAN_FOLDER,
        "name": "scGPT_human (whole-human pretrained, public)",
        "files": {f: sha256(os.path.join(ckpt_dir, f)) for f in needed},
    }


# ----------------------------------------------------------------------------- split


def make_split(adata, seed: int, train_gene_set_size: float = 0.75) -> dict:
    """GEARS 'simulation' split (cell-gears 0.0.2 DataSplitter): held-out *genes*.

    Returns {'train': [...], 'val': [...], 'test': [...]} condition lists; ctrl in train.
    """
    from gears.data_utils import DataSplitter

    ad = adata.copy()
    ds = DataSplitter(ad, split_type="simulation")
    ad, _subgroup = ds.split_data(train_gene_set_size=train_gene_set_size,
                                  combo_seen2_train_frac=0.75, seed=seed)
    split = {}
    for name in ["train", "val", "test"]:
        conds = sorted(set(ad.obs.loc[ad.obs["split"] == name, "condition"].astype(str)))
        split[name] = conds
    unassigned = ad.obs["split"].isna().sum()
    if unassigned:
        raise RuntimeError(f"{unassigned} cells have no split assignment")
    return split


def split_hash(split: dict) -> str:
    blob = json.dumps({k: sorted(v) for k, v in sorted(split.items())}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------- cache


@dataclass
class Dataset:
    """In-memory view of the prepared cache (loaded by training/eval jobs)."""

    X: np.ndarray                 # (n_cells, n_genes) float32, GEARS log-normalised expression
    conditions: np.ndarray        # (n_cells,) str
    gene_ids: list                # Ensembl ids, ordered
    gene_names: list              # symbols, ordered (scGPT vocab keys)
    split: dict                   # condition lists
    manifest: dict
    eval_controls: dict           # condition -> list[int] control cell indices used as eval inputs
    de_idx: dict                  # condition -> list[int] top-20 non-dropout DE gene positions (eval only)
    # condition -> list[int] top-20 positions of uns['rank_genes_groups_cov_all'] (GEARS "_de" metrics;
    # eval only). Optional: caches built before this field existed have None (-> metric NaN).
    de_rank20: dict | None = None

    @property
    def n_genes(self) -> int:
        return self.X.shape[1]

    def cells_of(self, condition: str) -> np.ndarray:
        return np.flatnonzero(self.conditions == condition)

    @property
    def ctrl_idx(self) -> np.ndarray:
        return self.cells_of(CTRL)

    def gene_pos(self, condition: str) -> int | None:
        g = target_gene(condition)
        return None if g is None else self.gene_names.index(g)

    def control_reference(self) -> np.ndarray:
        """Mean expression of the full control pool (all controls are in 'train')."""
        return self.X[self.ctrl_idx].mean(0)

    def observed_mean(self, condition: str) -> np.ndarray:
        return self.X[self.cells_of(condition)].mean(0)


def _positions(genes, pos: dict, what: str, cond: str) -> list[int]:
    out = []
    for g in genes:
        g = g.decode() if isinstance(g, bytes) else str(g)
        if g not in pos:
            raise KeyError(f"{cond}: {what} gene {g!r} not in cache gene ids")
        out.append(pos[g])
    return out


def de_gene_sets(ad, conditions: list[str], gene_ids: list[str], strict_rank: bool = True) -> tuple[dict, dict | None]:
    """GEARS evaluation gene sets as positions into ``gene_ids`` (evaluation only).

    Returns (de_idx, de_rank20):
      * de_idx: ``uns['top_non_dropout_de_20'][condition_name]`` (GEARS top20_de_non_dropout);
      * de_rank20: ``uns['rank_genes_groups_cov_all'][condition_name][:20]`` (GEARS pearson_de).
    With strict_rank=False a missing / short ``rank_genes_groups_cov_all`` gives de_rank20=None
    (the optional metric becomes NaN) instead of failing.
    """
    name_by_cond = dict(zip(ad.obs["condition"].astype(str), ad.obs["condition_name"].astype(str)))
    pos = {g: i for i, g in enumerate(gene_ids)}
    de_idx, de_rank20 = {}, {}
    for c in conditions:
        key = name_by_cond[c]
        de_idx[c] = _positions(ad.uns["top_non_dropout_de_20"][key], pos, "top_non_dropout_de_20", c)
    try:
        for c in conditions:
            key = name_by_cond[c]
            de_rank20[c] = _positions(list(ad.uns["rank_genes_groups_cov_all"][key])[:20], pos,
                                      "rank_genes_groups_cov_all", c)
            if len(de_rank20[c]) != 20:
                raise ValueError(f"{c}: rank_genes_groups_cov_all has {len(de_rank20[c])} < 20 genes")
    except (KeyError, ValueError):
        if strict_rank:
            raise
        de_rank20 = None
    return de_idx, de_rank20


def _valid_de_rank20(de_rank20: dict, split: dict, n_genes: int) -> bool:
    need = set(split["val"] + split["test"])
    return (need <= set(de_rank20)
            and all(len(v) == 20 and all(0 <= int(i) < n_genes for i in v) for v in de_rank20.values()))


def write_de_rank20(h5ad_path: str, cache_dir: str, force: bool = False) -> dict:
    """Supplement an existing cache with ``de_rank20.json`` WITHOUT touching any other cache file.

    Verifies the h5ad's ordered gene ids match the cache (so positions are valid) and that the
    regenerated ``top_non_dropout_de_20`` positions equal the cached ``de_idx.json``. An existing
    file is kept only if it covers every val/test condition with 20 in-range positions.
    """
    import scanpy as sc

    out = os.path.join(cache_dir, "de_rank20.json")
    with open(os.path.join(cache_dir, "genes.json")) as f:
        gene_ids = json.load(f)["gene_ids"]
    with open(os.path.join(cache_dir, "split.json")) as f:
        split = json.load(f)
    if os.path.exists(out) and not force:
        with open(out) as f:
            existing = json.load(f)
        if not _valid_de_rank20(existing, split, len(gene_ids)):
            raise RuntimeError(f"{out} exists but does not match this cache's split/genes; rerun with force")
        return {"path": out, "written": False, "n_conditions": len(existing)}
    with open(os.path.join(cache_dir, "de_idx.json")) as f:
        cached_de_idx = json.load(f)
    ad = sc.read_h5ad(h5ad_path, backed="r")
    try:
        if ad.var.index.astype(str).tolist() != gene_ids:
            raise RuntimeError("h5ad gene order differs from cache genes.json; refusing to write de_rank20")
        de_idx, de_rank20 = de_gene_sets(ad, split["val"] + split["test"], gene_ids, strict_rank=True)
    finally:
        ad.file.close()
    if de_idx != cached_de_idx:
        raise RuntimeError("regenerated top_non_dropout_de_20 differs from cached de_idx.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(de_rank20, f, indent=2)
    os.replace(tmp, out)
    return {"path": out, "written": True, "n_conditions": len(de_rank20), "h5ad_sha256": sha256(h5ad_path)}


def build_cache(h5ad_path: str, out_dir: str, split_seed: int, n_eval_controls: int,
                eval_control_seed: int, download_info: dict, ckpt_info: dict,
                vocab_genes: set | None) -> dict:
    import scanpy as sc
    import scipy.sparse as sp

    os.makedirs(out_dir, exist_ok=True)
    ad = sc.read_h5ad(h5ad_path)
    ad.obs["condition"] = ad.obs["condition"].astype(str)
    X = ad.X.toarray() if sp.issparse(ad.X) else np.asarray(ad.X)
    X = X.astype(np.float32)
    np.save(os.path.join(out_dir, "X.npy"), X)

    conditions = ad.obs["condition"].to_numpy().astype(str)
    gene_ids = ad.var.index.astype(str).tolist()
    gene_names = ad.var["gene_name"].astype(str).tolist()

    split = make_split(ad, seed=split_seed)
    all_conds = sorted(set(conditions))
    assigned = set(split["train"]) | set(split["val"]) | set(split["test"])
    assert assigned == set(all_conds), "split does not cover all conditions"
    assert not (set(split["train"]) & set(split["test"])), "train/test overlap"
    assert not (set(split["train"]) & set(split["val"])), "train/val overlap"
    assert not (set(split["val"]) & set(split["test"])), "val/test overlap"
    assert CTRL in split["train"]
    train_genes = {target_gene(c) for c in split["train"] if c != CTRL}
    for c in split["val"] + split["test"]:
        assert target_gene(c) not in train_genes, f"held-out gene {c} also in train"

    # Evaluation control inputs: one fixed draw per held-out condition, shared by all methods.
    rng = np.random.default_rng(eval_control_seed)
    ctrl_idx = np.flatnonzero(conditions == CTRL)
    eval_controls = {
        c: sorted(rng.choice(ctrl_idx, size=n_eval_controls, replace=False).tolist())
        for c in split["val"] + split["test"]
    }

    # Evaluation-only responsive genes (GEARS top-20 non-dropout DE, computed by GEARS on
    # the processed dataset). Never used as model input or for tuning.
    # de_rank20 is optional (GEARS "_de" metrics); a dataset without it still builds a cache.
    de_idx, de_rank20 = de_gene_sets(ad, split["val"] + split["test"], gene_ids, strict_rank=False)

    counts = {c: int((conditions == c).sum()) for c in all_conds}
    pert_targets = {c: target_gene(c) for c in all_conds if c != CTRL}
    missing_in_genes = [c for c, g in pert_targets.items() if g not in gene_names]
    coverage = None
    if vocab_genes is not None:
        in_vocab = [g in vocab_genes for g in gene_names]
        coverage = {
            "genes_in_scgpt_vocab": int(sum(in_vocab)),
            "genes_total": len(gene_names),
            "genes_not_in_vocab_first50": [g for g, ok in zip(gene_names, in_vocab) if not ok][:50],
            "targets_not_in_vocab": [c for c, g in pert_targets.items() if g not in vocab_genes],
        }

    manifest = {
        "dataset": "adamson (GEARS processed, cell-gears 0.0.2 dataverse file 6154417)",
        "track": "B",
        **download_info,
        "gears_revision": "cell-gears==0.0.2 (PyPI)",
        "n_cells": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
        "expression_layer": "adata.X as distributed by GEARS (log1p of normalised counts; "
                            "HVG selection performed upstream by GEARS on the whole dataset — inherited, "
                            "not refit on training data)",
        "cell_context": sorted(set(ad.obs["cell_type"].astype(str))),
        "batch_metadata": "none available in processed file",
        "n_conditions": len(all_conds),
        "n_interventions": len(all_conds) - 1,
        "cells_per_condition": counts,
        "control_label": CTRL,
        "n_control_cells": int(len(ctrl_idx)),
        "targets_not_in_gene_list": missing_in_genes,
        "vocab_coverage": coverage,
        "scgpt_checkpoint": ckpt_info,
        "split_generator": "gears.data_utils.DataSplitter(split_type='simulation'), "
                           "train_gene_set_size=0.75, val = simulation split of train at 0.9",
        "split_seed": split_seed,
        "split_hash": split_hash(split),
        "split_sizes": {k: len(v) for k, v in split.items()},
        "eval_control_rule": f"{n_eval_controls} control cells per held-out condition, sampled once "
                             f"without replacement (seed {eval_control_seed}); identical for every method",
        "control_reference": "mean of all control cells (control pool contains no treated outcomes)",
        "responsive_genes": "GEARS uns['top_non_dropout_de_20'] — evaluation only",
        "gears_de_genes": "GEARS uns['rank_genes_groups_cov_all'][:20] (de_rank20.json) — evaluation only",
        "ordered_gene_ids_sha256": hashlib.sha256("\n".join(gene_ids).encode()).hexdigest(),
    }

    def dump(name, obj):
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(obj, f, indent=2)

    np.save(os.path.join(out_dir, "conditions.npy"), conditions)
    dump("genes.json", {"gene_ids": gene_ids, "gene_names": gene_names})
    dump("split.json", split)
    dump("eval_controls.json", eval_controls)
    dump("de_idx.json", de_idx)
    if de_rank20 is not None:
        dump("de_rank20.json", de_rank20)
    dump("manifest.json", manifest)
    return manifest


def percell_pairs(ds: "Dataset", condition: str, n_cells: int, seed: int = 0):
    """Fixed (treated, control) cell pairs for paper-style per-cell scoring.

    Up to n_cells treated cells of `condition`, each paired with one control cell. The draw depends
    only on (seed, condition), so every method is scored on identical pairs. Controls contain no
    treated outcomes; treated cells are used only as evaluation targets."""
    import zlib

    rng = np.random.default_rng([seed, zlib.crc32(condition.encode())])
    cells = ds.cells_of(condition)
    treated = np.sort(rng.choice(cells, size=min(n_cells, len(cells)), replace=False))
    ctrl = rng.choice(ds.ctrl_idx, size=len(treated), replace=True)
    return treated, ctrl


def load_cache(cache_dir: str, mmap: bool = False) -> Dataset:
    def jload(name):
        with open(os.path.join(cache_dir, name)) as f:
            return json.load(f)

    genes = jload("genes.json")
    X = np.load(os.path.join(cache_dir, "X.npy"), mmap_mode="r" if mmap else None)
    return Dataset(
        X=X,
        conditions=np.load(os.path.join(cache_dir, "conditions.npy"), allow_pickle=False),
        gene_ids=genes["gene_ids"],
        gene_names=genes["gene_names"],
        split=jload("split.json"),
        manifest=jload("manifest.json"),
        eval_controls=jload("eval_controls.json"),
        de_idx=jload("de_idx.json"),
        de_rank20=jload("de_rank20.json") if os.path.exists(os.path.join(cache_dir, "de_rank20.json")) else None,
    )
