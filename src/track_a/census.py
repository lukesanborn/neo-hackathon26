"""Track A — Cell-JEPA reimplemented from the paper (unofficial): Census pretraining corpora.

Builds pretraining caches from CELLxGENE Census LTS release 2023-05-15. Corpora (CORPORA):

* ``kidney`` (default): scGPT's kidney filter (scGPT's 814k-cell kidney corpus ~ Cell-JEPA's
  "800K kidney cells"), all matching cells (or the first ``max_cells`` by soma_joinid).
* ``blood_marrow``: healthy blood + bone marrow, a seeded random sample of 2M matching cells
  (sorted by soma_joinid after sampling), with the PBMC-10K evaluation datasets screened out
  (see ``leakage_check``). Used for continued pretraining of public scGPT_human.

The contract below is identical for every corpus; only OUT differs (CORPORA[name]["out"]).

Cache contract (read by the pretraining unit), under OUT = /vol/trackA/kidney_census_2023-05-15/:

    manifest.json            written LAST; its presence means the cache is complete
    shards/shard_{k:03d}.npz indptr int64 (n_cells+1), gene_ids int32 (scGPT vocab ids),
                             bins uint8 (1..50), n_cells int64 scalar
    shards/shard_{k:03d}.json per-shard stats + npz sha256 (sidecar; written after the npz)
    split.json               {"val_cells": [[shard, row], ...]}  1% of cells, seed 0
    cells.npz                provenance: soma_joinid / dataset index / is_primary_data, in cache order
    genes.json               the in-vocab Census genes kept (feature_id, feature_name, vocab_id)
    var_vocab.npy, plan.json build internals: Census var soma_joinid -> vocab id (-1 = dropped);
                             cell count, shard bounds and input provenance

Cells are stored in ascending Census ``soma_joinid`` order, split into contiguous shards.

Only numpy is imported at module level so the binning code is unit-testable without the
Census stack; Census / tiledbsoma / scipy imports are local to the functions that need them.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from src.data import sha256

TRACK_LABEL = "Track A — Cell-JEPA reimplemented from the paper (unofficial)"
CENSUS_VERSION = "2023-05-15"
ORGANISM = "Homo sapiens"
CENSUS_ORGANISM_KEY = "homo_sapiens"
OBS_VALUE_FILTER = "tissue_general == 'kidney' and disease == 'normal' and suspension_type != 'na'"
X_LAYER = "raw"
N_BINS = 50
VAL_FRACTION = 0.01

# Named corpus specs. "kidney" must stay byte-identical to the original constants above (its
# cache already exists and its plan provenance is checked on reuse).
CORPORA: dict[str, dict] = {
    "kidney": {
        "filter": OBS_VALUE_FILTER,
        "out": "/vol/trackA/kidney_census_2023-05-15",
        "smoke_out": "/vol/smoke/census/kidney_census_2023-05-15",
        "max_cells": None,
        "sample": "first",
        "leakage_check": False,
        "rationale": "scGPT's kidney corpus filter (scGPT data/cellxgene build_soma_idx); "
                     "Cell-JEPA reports ~800K kidney cells, matching scGPT's 814k kidney set.",
    },
    "blood_marrow": {
        "filter": "tissue_general in ['blood', 'bone marrow'] and disease == 'normal' "
                  "and suspension_type != 'na'",
        "out": "/vol/trackA/blood_marrow_census_2023-05-15_2M",
        "smoke_out": "/vol/smoke/census/blood_marrow_census_2023-05-15",
        "max_cells": 2_000_000,
        "sample": "random",
        "sample_seed": 0,
        "leakage_check": True,
        "rationale": "In-domain continued pretraining of public scGPT_human for the PBMC (blood) "
                     "and K562 (leukemia, bone-marrow lineage) evaluations: healthy blood + bone "
                     "marrow, 2M-cell seeded random sample, PBMC-10K eval datasets screened out.",
    },
    # Corpora for the 2a/2b knob sweep: a broad human corpus (what scGPT_human itself was built
    # from, subsampled) and the complete blood+marrow set rather than the 2M sample.
    "human_3M": {
        "filter": "disease == 'normal' and suspension_type != 'na'",
        "out": "/vol/trackA/human_census_2023-05-15_3M",
        "smoke_out": "/vol/smoke/census/human_census_2023-05-15",
        "max_cells": 3_000_000,
        "sample": "random",
        "sample_seed": 0,
        "leakage_check": True,
        "rationale": "All healthy human tissues (scGPT_human's own corpus shape), 3M-cell seeded "
                     "random sample: tests whether the masking knobs behave differently when "
                     "pretraining is broad rather than single-tissue.",
    },
    "blood_marrow_full": {
        "filter": "tissue_general in ['blood', 'bone marrow'] and disease == 'normal' "
                  "and suspension_type != 'na'",
        "out": "/vol/trackA/blood_marrow_census_2023-05-15_full",
        "smoke_out": "/vol/smoke/census/blood_marrow_full_census_2023-05-15",
        "max_cells": None,            # every matching cell (~5.17M), not the 2M sample
        "sample": "first",
        "leakage_check": True,
        "rationale": "Every healthy blood + bone-marrow cell in the release (~5.17M). Closest "
                     "tissue match to the K562 evaluation, so pretraining gains have the best "
                     "chance of transferring to the perturbation benchmarks.",
    },
}

# PBMC-10K evaluation data = 10x Genomics public datasets pbmc4k + pbmc8k (2017). A Census
# dataset is excluded if its title or collection name matches any EXCLUDE pattern; anything
# matching a SEARCH pattern is recorded as a reviewed candidate (most blood data are PBMCs, so
# PBMC alone is NOT grounds for exclusion).
LEAKAGE_EXCLUDE_PATTERNS = [r"10x\s*genomics", r"pbmc[\s_-]*[48]\s*k\b", r"\b[48]\s*k[\s_-]*pbmc"]
LEAKAGE_SEARCH_PATTERNS = [r"pbmc", r"peripheral blood mononuclear", r"10x", r"\b[48]\s*k\b"]
LEAKAGE_FIELDS = ("dataset_title", "collection_name", "collection_doi", "dataset_id", "collection_id")
SPLIT_SEED = 0
DEFAULT_SHARD_SIZE = 80_000

BINNING_RULE = (
    "Per cell, over the cell's nonzero raw counts restricted to genes present in the scGPT "
    "vocab (zeros are dropped, not stored): edges = np.quantile(v, np.linspace(0, 1, 50)) "
    "(default 'linear' interpolation), bins = np.digitize(v, edges) (right=False), which "
    "lies in 1..50. This is scGPT 0.2.4 Preprocessor binning (n_bins=51, bin 0 reserved for "
    "zero) with deterministic np.digitize instead of scGPT's random tie-breaking `_digitize`. "
    "Equivalently: 49 non-trivial edges (quantiles k/49, k=1..49) and bin = 1 + #edges <= v. "
    "Ties map to the same bin; the cell maximum always gets bin 50, so a cell with one "
    "nonzero gene or all-equal nonzero values gets bin 50 everywhere. Duplicate Census "
    "features mapping to the same scGPT symbol are summed before binning. Within a cell, "
    "tokens are sorted by ascending vocab id."
)

DUPLICATE_PRIMARY_NOTE = (
    "Documented Census 2023-05-15 erratum: ~243,569 cells (Census-wide) are marked "
    "is_primary_data=True in more than one dataset. Following scGPT's corpus, this cache does "
    "NOT filter on is_primary_data; counts of True/False within this slice are recorded here, "
    "but cross-dataset duplicates cannot be identified from obs alone."
)


# ============================================================================ binning (numpy)


def bin_nonzero(values: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Quantile-bin one cell's nonzero values to integers 1..n_bins (uint8). See BINNING_RULE."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return np.zeros(0, dtype=np.uint8)
    if not (v > 0).all():
        raise ValueError("bin_nonzero expects strictly positive values (zeros must be dropped)")
    edges = np.quantile(v, np.linspace(0.0, 1.0, n_bins))
    out = np.digitize(v, edges)
    return out.astype(np.uint8)


def bin_csr(indptr: np.ndarray, values: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Apply bin_nonzero row-wise to a CSR value array; returns uint8 bins aligned with values."""
    bins = np.empty(values.shape[0], dtype=np.uint8)
    for i in range(indptr.shape[0] - 1):
        a, b = int(indptr[i]), int(indptr[i + 1])
        if b > a:
            bins[a:b] = bin_nonzero(values[a:b], n_bins)
    return bins


def coo_to_shard(rows: np.ndarray, gene_ids: np.ndarray, values: np.ndarray, n_cells: int,
                 n_bins: int = N_BINS) -> dict:
    """(row, vocab gene id, raw count) triplets of in-vocab genes -> shard arrays.

    Drops zeros, sums duplicate (row, gene) entries (several Census features with one scGPT
    symbol), sorts tokens by (row, gene id) and bins each row.
    """
    rows = np.asarray(rows, dtype=np.int32)
    gene_ids = np.asarray(gene_ids, dtype=np.int32)
    values = np.asarray(values, dtype=np.float32)
    keep = values > 0
    if not keep.all():
        rows, gene_ids, values = rows[keep], gene_ids[keep], values[keep]
    del keep
    if rows.size and (rows.min() < 0 or rows.max() >= n_cells or gene_ids.min() < 0):
        raise ValueError("row / gene index out of range")
    # one int64 sort key instead of lexsort over two arrays (keeps peak memory down)
    order = np.argsort(rows.astype(np.int64) * (int(gene_ids.max(initial=0)) + 1) + gene_ids, kind="stable")
    rows, gene_ids, values = rows[order], gene_ids[order], values[order]
    del order
    n_dup = 0
    if rows.size:
        new = np.empty(rows.size, dtype=bool)
        new[0] = True
        new[1:] = (rows[1:] != rows[:-1]) | (gene_ids[1:] != gene_ids[:-1])
        n_dup = int(rows.size - new.sum())
        if n_dup:
            starts = np.flatnonzero(new)
            values = np.add.reduceat(values.astype(np.float64), starts)
            rows, gene_ids = rows[starts], gene_ids[starts]
    counts = np.bincount(rows, minlength=n_cells)
    indptr = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    bins = bin_csr(indptr, values, n_bins)
    return {
        "indptr": indptr,
        "gene_ids": gene_ids.astype(np.int32),
        "bins": bins,
        "n_cells": np.int64(n_cells),
        "n_duplicate_symbol_entries_summed": n_dup,
    }


class ShardError(ValueError):
    pass


def _check(cond, msg: str) -> None:
    if not cond:
        raise ShardError(msg)


def validate_shard(sh: dict, vocab_size: int, n_bins: int = N_BINS) -> None:
    """Raise ShardError unless the shard matches the cache contract (not assert: survives -O)."""
    indptr, g, b, n = sh["indptr"], sh["gene_ids"], sh["bins"], int(sh["n_cells"])
    _check(indptr.dtype == np.int64 and g.dtype == np.int32 and b.dtype == np.uint8, "bad dtypes")
    _check(indptr.shape == (n + 1,) and indptr[0] == 0 and indptr[-1] == g.size == b.size, "bad indptr")
    _check(bool((np.diff(indptr) >= 0).all()), "indptr not monotonic")
    if g.size:
        _check(g.min() >= 0 and g.max() < vocab_size, "gene id outside vocab")
        _check(b.min() >= 1 and b.max() <= n_bins, "bin outside 1..n_bins")


def make_val_split(shard_sizes: list[int], frac: float = VAL_FRACTION, seed: int = SPLIT_SEED) -> list[list[int]]:
    """Held-out cells as sorted [shard, row] pairs; round(frac * n) cells (>= 1 if n > 0)."""
    n = int(sum(shard_sizes))
    if n == 0:
        return []
    n_val = max(1, int(round(frac * n)))
    idx = np.sort(np.random.default_rng(seed).choice(n, size=n_val, replace=False))
    offsets = np.concatenate([[0], np.cumsum(shard_sizes)])
    shard = np.searchsorted(offsets, idx, side="right") - 1
    row = idx - offsets[shard]
    return [[int(s), int(r)] for s, r in zip(shard, row)]


def get_corpus(name: str) -> dict:
    if name not in CORPORA:
        raise ValueError(f"unknown corpus {name!r}; choose from {sorted(CORPORA)}")
    return CORPORA[name]


def select_cells(n: int, max_cells: int | None, sample: str, seed: int = 0) -> np.ndarray:
    """Row positions (ascending) to keep out of n joinid-sorted matching cells.

    "first": the first max_cells; "random": a seeded uniform sample without replacement, sorted
    so shard reads stay contiguous. max_cells None or >= n keeps every cell.
    """
    if sample not in ("first", "random"):
        raise ValueError(f"unknown sample mode {sample!r}")
    if max_cells is not None and max_cells < 0:
        raise ValueError("max_cells must be >= 0")
    if max_cells is None or max_cells >= n:
        return np.arange(n, dtype=np.int64)
    if sample == "first":
        return np.arange(max_cells, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(n, size=max_cells, replace=False)).astype(np.int64)


def leakage_check(datasets: list[dict], slice_dataset_ids) -> dict:
    """Screen Census datasets in the slice for the PBMC-10K eval data (10x pbmc4k/pbmc8k).

    ``datasets`` are census_info "datasets" rows (dicts with LEAKAGE_FIELDS where available).
    Excluded = title/collection matches an EXCLUDE pattern; candidates = slice datasets that
    match only a broader SEARCH pattern (recorded for review, not excluded).
    """
    import re

    slice_ids = set(slice_dataset_ids)
    exc_re = [re.compile(p, re.I) for p in LEAKAGE_EXCLUDE_PATTERNS]
    srch_re = [re.compile(p, re.I) for p in LEAKAGE_SEARCH_PATTERNS]
    excluded, candidates = [], []
    for row in datasets:
        if row.get("dataset_id") not in slice_ids:
            continue
        text = " | ".join(str(row.get(k) or "") for k in ("dataset_title", "collection_name"))
        rec = {k: (None if row.get(k) is None else str(row.get(k))) for k in LEAKAGE_FIELDS}
        hit = [p.pattern for p in exc_re if p.search(text)]
        if hit:
            excluded.append({**rec, "matched": hit})
        elif any(p.search(text) for p in srch_re):
            candidates.append(rec)
    return {
        "target": "PBMC-10K eval = 10x Genomics public datasets pbmc4k + pbmc8k (2017)",
        "searched_fields": ["dataset_title", "collection_name"],
        "exclude_patterns": LEAKAGE_EXCLUDE_PATTERNS,
        "search_patterns": LEAKAGE_SEARCH_PATTERNS,
        "n_slice_datasets_screened": len(slice_ids),
        "excluded_dataset_ids": sorted(e["dataset_id"] for e in excluded),
        "excluded": excluded,
        "candidates_reviewed_not_excluded": candidates,
    }


def screen_leakage(census, slice_dataset_ids) -> dict:
    """leakage_check against the open Census's census_info datasets table.

    ``slice_dataset_ids``: the dataset_id of every cell in the filtered slice (a pandas Series);
    adds per-record and total slice cell counts.
    """
    ds = census["census_info"]["datasets"].read().concat().to_pandas()
    counts = slice_dataset_ids.astype(str).value_counts()
    out = leakage_check(ds.to_dict("records"), counts.index.tolist())
    for rec in out["excluded"] + out["candidates_reviewed_not_excluded"]:
        rec["n_cells_in_slice"] = int(counts.get(rec["dataset_id"], 0))
    out["n_cells_excluded"] = int(sum(counts.get(d, 0) for d in out["excluded_dataset_ids"]))
    return out


def shard_bounds(n_cells: int, shard_size: int) -> list[tuple[int, int]]:
    return [(a, min(a + shard_size, n_cells)) for a in range(0, n_cells, shard_size)]


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def shard_path(out_dir: str, k: int) -> str:
    return os.path.join(out_dir, "shards", f"shard_{k:03d}.npz")


def write_npz_atomic(path: str, **arrays) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def write_json_atomic(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# ============================================================================ Census access


def package_versions() -> dict:
    import importlib.metadata as md

    out = {}
    for p in ("cellxgene-census", "tiledbsoma", "tiledb", "numpy", "scipy", "pandas", "pyarrow", "anndata"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def open_census():
    import cellxgene_census

    return cellxgene_census.open_soma(census_version=CENSUS_VERSION)


def census_description() -> dict:
    import cellxgene_census

    try:
        return dict(cellxgene_census.get_census_version_description(CENSUS_VERSION))
    except Exception as e:  # informational only
        return {"error": repr(e)}


def plan(out_dir: str, vocab: dict[str, int], shard_size: int, max_cells: int | None,
         corpus: str = "kidney") -> dict:
    """Read obs/var, map genes to the vocab, fix cell order and shard boundaries.

    Writes cells.npz, genes.json, var_vocab.npy and plan.json (last) into out_dir. Raises if
    the filter matches no cells or no Census gene maps to the vocab.
    """
    spec = get_corpus(corpus)
    obs_filter = spec["filter"]
    os.makedirs(os.path.join(out_dir, "shards"), exist_ok=True)
    with open_census() as census:
        exp = census["census_data"][CENSUS_ORGANISM_KEY]
        obs = (exp.obs.read(value_filter=obs_filter,
                            column_names=["soma_joinid", "dataset_id", "is_primary_data", "tissue_general"])
               .concat().to_pandas())
        var = (exp.ms["RNA"].var.read(column_names=["soma_joinid", "feature_id", "feature_name"])
               .concat().to_pandas())
        n_obs_total = int(exp.obs.count)
        leak = screen_leakage(census, obs["dataset_id"]) if spec.get("leakage_check") else None
    obs = obs.sort_values("soma_joinid").reset_index(drop=True)
    n_filter = len(obs)
    if leak is not None and leak["excluded_dataset_ids"]:
        excl = obs["dataset_id"].astype(str).isin(set(leak["excluded_dataset_ids"]))
        obs = obs[~excl.to_numpy()].reset_index(drop=True)
    n_eligible = len(obs)
    seed = int(spec.get("sample_seed", 0))
    obs = obs.iloc[select_cells(n_eligible, max_cells, spec["sample"], seed)].reset_index(drop=True)
    if len(obs) == 0:
        raise RuntimeError(f"obs filter matched no cells: {obs_filter!r}")
    if not (np.diff(obs["soma_joinid"].to_numpy()) > 0).all():
        raise RuntimeError("selected soma_joinids are not strictly ascending")
    datasets = sorted(obs["dataset_id"].unique().tolist())
    ds_index = {d: i for i, d in enumerate(datasets)}
    np.savez(os.path.join(out_dir, "cells.npz"),
             soma_joinid=obs["soma_joinid"].to_numpy(np.int64),
             dataset_idx=obs["dataset_id"].map(ds_index).to_numpy(np.int32),
             is_primary_data=obs["is_primary_data"].to_numpy(bool))

    var = var.sort_values("soma_joinid").reset_index(drop=True)
    n_var = int(var["soma_joinid"].max()) + 1
    var_vocab = np.full(n_var, -1, dtype=np.int64)
    kept = []
    for jid, fid, name in zip(var["soma_joinid"], var["feature_id"], var["feature_name"]):
        if name in vocab:
            var_vocab[int(jid)] = int(vocab[name])
            kept.append({"soma_joinid": int(jid), "feature_id": fid, "feature_name": name,
                         "vocab_id": int(vocab[name])})
    if not kept:
        raise RuntimeError("no Census feature_name maps to the scGPT vocab")
    np.save(os.path.join(out_dir, "var_vocab.npy"), var_vocab)
    write_json_atomic(os.path.join(out_dir, "genes.json"), kept)
    symbols_kept = sorted({g["feature_name"] for g in kept})
    missing = sorted(set(var["feature_name"]) - set(vocab))

    per_ds = obs["dataset_id"].value_counts()
    per_tissue = obs["tissue_general"].astype(str).value_counts()
    info = {
        "corpus": corpus,
        "n_obs_census_homo_sapiens": n_obs_total,
        "n_cells_matching_filter": n_filter,
        "n_cells_eligible": n_eligible,
        "n_cells": len(obs),
        "max_cells": max_cells,
        "sampling": {
            "mode": spec["sample"], "seed": seed if spec["sample"] == "random" else None,
            "n_matching_filter": n_filter, "n_eligible_after_exclusions": n_eligible, "n_sampled": len(obs),
            "rule": ("np.sort(np.random.default_rng(seed).choice(n_eligible, max_cells, replace=False)) "
                     "over joinid-sorted eligible cells (all cells if max_cells >= n_eligible)"
                     if spec["sample"] == "random" else "first max_cells eligible cells by soma_joinid"),
        },
        "cells_per_tissue_general": {str(t): int(c) for t, c in per_tissue.items()},
        "leakage_check": leak,
        "n_datasets": len(datasets),
        "dataset_ids": datasets,
        "cells_per_dataset": {d: int(per_ds[d]) for d in datasets},
        "is_primary_data_true": int(obs["is_primary_data"].sum()),
        "is_primary_data_false": int((~obs["is_primary_data"]).sum()),
        "n_census_genes": int(len(var)),
        "n_var_index": n_var,
        "n_census_genes_in_vocab": len(kept),
        "n_unique_symbols_in_vocab": len(symbols_kept),
        "census_gene_vocab_coverage": len(kept) / max(1, len(var)),
        "n_census_symbols_missing_from_vocab": len(missing),
        "missing_symbol_examples": missing[:50],
        "gene_list_sha256": sha256_json([[g["feature_id"], g["feature_name"], g["vocab_id"]] for g in kept]),
        "shard_size": shard_size,
        "provenance": plan_provenance(vocab, corpus),
        "shards": shard_bounds(len(obs), shard_size),
    }
    remove_stale_shards(out_dir, len(info["shards"]))
    write_json_atomic(os.path.join(out_dir, "plan.json"), info)
    return info


def remove_stale_shards(out_dir: str, n_shards: int) -> list[str]:
    """Delete shard files numbered >= n_shards (left by an earlier plan); consumers glob shard_*.npz."""
    import glob
    import re

    removed = []
    for p in glob.glob(os.path.join(out_dir, "shards", "shard_*")):
        m = re.search(r"shard_(\d+)\.(npz|json)(\.tmp)?$", p)
        if m and int(m.group(1)) >= n_shards:
            os.remove(p)
            removed.append(p)
    return removed


def plan_provenance(vocab: dict[str, int], corpus: str = "kidney") -> dict:
    """Inputs that determine the cache contents; a reused plan must match these exactly.

    The kidney dict is unchanged from before the corpus registry, so its existing plan.json
    still validates on reuse.
    """
    spec = get_corpus(corpus)
    prov = {"census_version": CENSUS_VERSION, "organism": ORGANISM, "obs_value_filter": spec["filter"],
            "X_layer": X_LAYER, "n_bins": N_BINS, "vocab_sha256": sha256_json(vocab)}
    if corpus != "kidney":
        prov.update({"corpus": corpus, "sample": spec["sample"], "sample_seed": spec.get("sample_seed"),
                     "leakage_exclude_patterns": LEAKAGE_EXCLUDE_PATTERNS if spec.get("leakage_check") else None})
    return prov


def read_shard_triplets(soma_joinids: np.ndarray, var_vocab: np.ndarray):
    """Stream the raw X rows for the given (sorted) obs joinids; keep in-vocab nonzero entries.

    Returns (row positions, vocab ids, values, stats) — rows are positions within soma_joinids.
    """
    import tiledbsoma as soma

    joinids = np.asarray(soma_joinids, dtype=np.int64)
    rows, genes, vals = [], [], []
    nnz_total = 0
    with open_census() as census:
        exp = census["census_data"][CENSUS_ORGANISM_KEY]
        with exp.axis_query("RNA", obs_query=soma.AxisQuery(coords=(joinids,))) as q:
            for tbl in q.X(X_LAYER).tables():
                r = tbl["soma_dim_0"].to_numpy()
                c = tbl["soma_dim_1"].to_numpy()
                v = tbl["soma_data"].to_numpy()
                nnz_total += int(r.size)
                g = var_vocab[c]
                keep = (g >= 0) & (v > 0)
                rk = r[keep]
                pos = np.searchsorted(joinids, rk)
                if pos.size and not (joinids[np.minimum(pos, joinids.size - 1)] == rk).all():
                    raise RuntimeError("X returned a row outside the requested joinids")
                rows.append(pos.astype(np.int32))
                genes.append(g[keep].astype(np.int32))
                vals.append(v[keep].astype(np.float32))
    cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt)  # noqa: E731
    r, g, v = cat(rows, np.int32), cat(genes, np.int32), cat(vals, np.float32)
    return r, g, v, {"nnz_all_genes": nnz_total, "nnz_in_vocab": int(r.size),
                     "n_noninteger_values": int((v != np.round(v)).sum())}


def build_shard(out_dir: str, k: int, vocab_size: int, force: bool = False) -> dict:
    """Build shard k from plan.json / cells.npz; idempotent unless force."""
    p = shard_path(out_dir, k)
    side = p[:-4] + ".json"
    with open(os.path.join(out_dir, "plan.json")) as f:
        info = json.load(f)
    a, b = info["shards"][k]
    joinids = np.load(os.path.join(out_dir, "cells.npz"))["soma_joinid"][a:b]
    if not force and os.path.exists(p) and os.path.exists(side):
        with open(side) as f:
            st = json.load(f)
        if (st.get("n_cells") == b - a and st.get("start") == a
                and st.get("first_soma_joinid") == int(joinids[0])
                and st.get("last_soma_joinid") == int(joinids[-1])):
            return {**st, "skipped": True}
    # drop the sidecar first: an npz without a sidecar is never treated as complete
    if os.path.exists(side):
        os.remove(side)
    var_vocab = np.load(os.path.join(out_dir, "var_vocab.npy"))
    r, g, v, st = read_shard_triplets(joinids, var_vocab)
    sh = coo_to_shard(r, g, v, n_cells=b - a)
    del r, g, v
    validate_shard(sh, vocab_size)
    n_dup = sh.pop("n_duplicate_symbol_entries_summed")
    write_npz_atomic(p, **sh)
    lens = np.diff(sh["indptr"])
    stats = {
        "shard": k, "start": a, "stop": b, "n_cells": b - a,
        "first_soma_joinid": int(joinids[0]), "last_soma_joinid": int(joinids[-1]),
        "n_tokens": int(sh["gene_ids"].size), **st,
        "n_duplicate_symbol_entries_summed": n_dup,
        "n_empty_cells": int((lens == 0).sum()),
        "genes_per_cell_min": int(lens.min()), "genes_per_cell_median": float(np.median(lens)),
        "genes_per_cell_max": int(lens.max()),
        "bin_histogram": np.bincount(sh["bins"], minlength=N_BINS + 1)[1:].tolist(),
        "npz_sha256": sha256(p),
    }
    write_json_atomic(side, stats)
    return stats


def finalize(out_dir: str, vocab_size: int, vocab_path: str, census_desc: dict, extra: dict | None = None,
             corpus: str = "kidney") -> dict:
    """Check every shard against the plan, write split.json, then manifest.json (last)."""
    spec = get_corpus(corpus)
    with open(os.path.join(out_dir, "plan.json")) as f:
        info = json.load(f)
    if info.get("corpus", "kidney") != corpus:  # pre-registry kidney plans carry no "corpus" key
        raise RuntimeError(f"plan at {out_dir} is for corpus {info.get('corpus')!r}, not {corpus!r}")
    shard_stats = []
    for k, (a, b) in enumerate(info["shards"]):
        with np.load(shard_path(out_dir, k)) as z:
            sh = {key: z[key] for key in ("indptr", "gene_ids", "bins", "n_cells")}
        validate_shard(sh, vocab_size)
        with open(shard_path(out_dir, k)[:-4] + ".json") as f:
            st = json.load(f)
        if st["start"] != a or sha256(shard_path(out_dir, k)) != st["npz_sha256"]:
            raise RuntimeError(f"shard {k}: sidecar does not match the npz / plan; rebuild with --force")
        if int(sh["n_cells"]) != b - a:
            raise RuntimeError(f"shard {k}: n_cells {int(sh['n_cells'])} != planned {b - a}")
        shard_stats.append(st)
    sizes = [s["n_cells"] for s in shard_stats]
    val = make_val_split(sizes)
    write_json_atomic(os.path.join(out_dir, "split.json"), {"val_cells": val})
    nnz_all = sum(s["nnz_all_genes"] for s in shard_stats)
    nnz_voc = sum(s["nnz_in_vocab"] for s in shard_stats)
    n_tok = sum(s["n_tokens"] for s in shard_stats)
    manifest = {
        "track": TRACK_LABEL,
        "source": "CELLxGENE Census (LTS)",
        "census_version": CENSUS_VERSION,
        "census_version_description": census_desc,
        "organism": ORGANISM,
        "measurement": "RNA",
        "X_layer": X_LAYER,
        "corpus": corpus,
        "obs_value_filter": spec["filter"],
        "is_primary_data_filter": None,
        "filter_rationale": spec["rationale"],
        "package_versions": package_versions(),
        "census_stack": os.environ.get("CENSUS_STACK"),
        "vocab_path": vocab_path,
        "vocab_size": vocab_size,
        "n_cells": info["n_cells"],
        "n_cells_matching_filter": info["n_cells_matching_filter"],
        "max_cells": info["max_cells"],
        "n_cells_eligible": info.get("n_cells_eligible", info["n_cells_matching_filter"]),
        "sampling": info.get("sampling"),
        "cells_per_tissue_general": info.get("cells_per_tissue_general"),
        "leakage_check": info.get("leakage_check"),
        "n_datasets": info["n_datasets"],
        "dataset_ids": info["dataset_ids"],
        "cells_per_dataset": info["cells_per_dataset"],
        "is_primary_data_counts": {"true": info["is_primary_data_true"], "false": info["is_primary_data_false"]},
        "duplicate_primary_note": DUPLICATE_PRIMARY_NOTE,
        "duplicate_primary_census_wide_erratum": 243569,
        "n_census_genes": info["n_census_genes"],
        "n_genes_in_vocab": info["n_census_genes_in_vocab"],
        "n_unique_symbols_in_vocab": info["n_unique_symbols_in_vocab"],
        "census_gene_vocab_coverage": info["census_gene_vocab_coverage"],
        "n_census_symbols_missing_from_vocab": info["n_census_symbols_missing_from_vocab"],
        "missing_symbol_examples": info["missing_symbol_examples"],
        "gene_mapping": "Census var feature_name (symbol) looked up verbatim in scGPT vocab.json; "
                        "no HGNC alias map applied (genes absent from the vocab are dropped).",
        "gene_list_sha256": info["gene_list_sha256"],
        "nnz_all_genes": nnz_all,
        "nnz_in_vocab": nnz_voc,
        "nonzero_count_vocab_coverage": nnz_voc / max(1, nnz_all),
        "n_tokens": n_tok,
        "n_duplicate_symbol_entries_summed": sum(s["n_duplicate_symbol_entries_summed"] for s in shard_stats),
        "n_noninteger_values": sum(s["n_noninteger_values"] for s in shard_stats),
        "n_empty_cells": sum(s["n_empty_cells"] for s in shard_stats),
        "mean_genes_per_cell": n_tok / max(1, info["n_cells"]),
        "n_bins": N_BINS,
        "binning_rule": BINNING_RULE,
        "cell_order": "ascending Census soma_joinid; contiguous shards (see cells.npz)",
        "shard_format": {"indptr": "int64 (n_cells+1)", "gene_ids": "int32 scGPT vocab ids",
                         "bins": "uint8 1..50", "n_cells": "int64 scalar"},
        "n_shards": len(shard_stats),
        "shards": [{"file": os.path.basename(shard_path(out_dir, s["shard"])), "n_cells": s["n_cells"],
                    "n_tokens": s["n_tokens"], "sha256": s["npz_sha256"]} for s in shard_stats],
        "split": {"file": "split.json", "val_fraction": VAL_FRACTION, "seed": SPLIT_SEED,
                  "n_val": len(val), "n_train": info["n_cells"] - len(val),
                  "rule": "np.random.default_rng(0).choice(n_cells, round(0.01*n_cells), replace=False) "
                          "over global cache order, mapped to [shard, row]; all other cells are train."},
        **(extra or {}),
    }
    write_json_atomic(os.path.join(out_dir, "manifest.json"), manifest)
    return manifest
