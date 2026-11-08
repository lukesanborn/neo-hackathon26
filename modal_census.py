"""Track A — Cell-JEPA reimplemented from the paper (unofficial): Census pretraining caches (CPU only).

    modal run modal_census.py --stage smoke                 # first ~5k kidney cells -> /vol/smoke/census/
    modal run --detach modal_census.py --stage full         # full 2023-05-15 kidney slice -> /vol/trackA/
    modal run modal_census.py --stage verify                # re-check the full cache, print summary

    modal run modal_census.py --stage smoke --corpus blood_marrow          # 5k random blood+marrow cells
    modal run --detach modal_census.py --stage full --corpus blood_marrow  # 2M-cell sample
    modal run modal_census.py --stage verify --corpus blood_marrow         # + KidneyCache load check
    modal run modal_census.py --stage leakage --corpus blood_marrow        # PBMC-10K screen only

Corpora live in src/track_a/census.py CORPORA (default "kidney", unchanged paths/behaviour).
Flags: --force (rebuild plan + every shard), --shard-size N, --max-cells N (smoke default 5000;
full default = the corpus's max_cells). CENSUS_STACK=legacy selects the cellxgene-census 1.15.0
fallback image (environment/census_image.py).

The driver (`run`) executes remotely, so `--detach` keeps the whole pipeline alive after the client
exits: plan (obs/var + gene mapping) -> one container per missing shard -> split + manifest.
Completed shards are skipped on rerun; manifest.json is written last.
"""

from __future__ import annotations

import json
import os

import modal

from environment.census_image import PYTHON_VERSION, build_census_image

APP_NAME = "jepa-census"
VOL_PATH = "/vol"
VOCAB_PATH = f"{VOL_PATH}/checkpoints/scGPT_human/vocab.json"
# output dirs: src/track_a/census.py CORPORA[name]["out"] / ["smoke_out"]
MAX_SHARD_CONTAINERS = 40   # one container per shard for the 3M wholebody build (was 8)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("jepa-perturb")
image = build_census_image()
# KidneyCache (src.track_a.pretrain) needs torch; a small CPU image just for the load check.
load_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install("numpy==1.26.4", "torch==2.3.1", extra_index_url="https://download.pytorch.org/whl/cpu")
    .add_local_python_source("src", "environment")
)


def _load_vocab() -> dict[str, int]:
    with open(VOCAB_PATH) as f:
        return json.load(f)


def _vocab_size(vocab: dict[str, int]) -> int:
    return max(vocab.values()) + 1


def _out_dir(corpus: str, stage: str) -> str:
    from src.track_a import census as C

    spec = C.get_corpus(corpus)
    return spec["smoke_out"] if stage == "smoke" else spec["out"]


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=4, memory=65536, timeout=4 * 3600,
              retries=2, max_containers=MAX_SHARD_CONTAINERS)
def build_shard(out_dir: str, k: int, force: bool = False) -> dict:
    from src.track_a import census as C

    vol.reload()
    st = C.build_shard(out_dir, k, _vocab_size(_load_vocab()), force=force)
    vol.commit()
    print(f"shard {k}: {json.dumps({x: st.get(x) for x in ('n_cells', 'n_tokens', 'n_empty_cells', 'skipped')})}",
          flush=True)
    return st


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=2, memory=65536, timeout=20 * 3600)
def run(stage: str = "smoke", force: bool = False, shard_size: int = 0, max_cells: int = -1,
        corpus: str = "kidney") -> dict:
    import time

    from src.track_a import census as C

    if stage not in ("smoke", "full"):
        raise ValueError(stage)
    t0 = time.time()
    spec = C.get_corpus(corpus)
    out_dir = _out_dir(corpus, stage)
    mc = (5000 if stage == "smoke" else spec["max_cells"]) if max_cells < 0 else (max_cells or None)
    ss = shard_size or (2000 if stage == "smoke" else C.DEFAULT_SHARD_SIZE)
    vocab = _load_vocab()
    vsize = _vocab_size(vocab)

    plan_path = os.path.join(out_dir, "plan.json")
    info = None
    if not force and os.path.exists(plan_path):
        with open(plan_path) as f:
            info = json.load(f)
        if (info.get("max_cells") != mc or info.get("shard_size") != ss
                or info.get("provenance") != C.plan_provenance(vocab, corpus)):
            raise RuntimeError(f"existing plan at {plan_path} was built with different inputs "
                               f"(max_cells / shard_size / filter / sampling / vocab); rerun with --force to rebuild")
        print("reusing existing plan", flush=True)
    if info is None:
        if os.path.exists(os.path.join(out_dir, "manifest.json")):
            os.remove(os.path.join(out_dir, "manifest.json"))  # cache is incomplete until finalize
        info = C.plan(out_dir, vocab, ss, mc, corpus=corpus)
        vol.commit()
    t_plan = time.time() - t0
    print(json.dumps({k: info.get(k) for k in ("corpus", "n_cells_matching_filter", "n_cells_eligible", "n_cells",
                                               "n_datasets", "cells_per_tissue_general",
                                               "is_primary_data_true", "is_primary_data_false",
                                               "n_census_genes", "n_census_genes_in_vocab",
                                               "census_gene_vocab_coverage")}), flush=True)
    leak = info.get("leakage_check")
    if leak:
        print("LEAKAGE " + json.dumps({"excluded": leak["excluded"], "n_cells_excluded": leak["n_cells_excluded"],
                                       "candidates": leak["candidates_reviewed_not_excluded"]}), flush=True)
    n_shards = len(info["shards"])
    print(f"{n_shards} shards of <= {ss} cells (plan {t_plan:.0f}s)", flush=True)
    stats = list(build_shard.map([out_dir] * n_shards, range(n_shards), kwargs={"force": force}))
    t_shards = time.time() - t0 - t_plan
    vol.reload()
    manifest = C.finalize(out_dir, vsize, VOCAB_PATH, C.census_description(), corpus=corpus,
                          extra={"stage": stage, "build_seconds": {"plan": round(t_plan), "shards": round(t_shards)}})
    vol.commit()
    summary = {k: manifest[k] for k in (
        "corpus", "n_cells", "n_cells_matching_filter", "n_cells_eligible", "n_datasets", "is_primary_data_counts",
        "cells_per_tissue_general", "n_census_genes", "n_genes_in_vocab",
        "census_gene_vocab_coverage", "nonzero_count_vocab_coverage", "mean_genes_per_cell",
        "n_empty_cells", "n_noninteger_values", "n_shards", "package_versions")}
    summary["n_val"] = manifest["split"]["n_val"]
    summary["out_dir"] = out_dir
    summary["shards_skipped"] = sum(bool(s.get("skipped")) for s in stats)
    summary["seconds"] = {"plan": round(t_plan), "shards": round(t_shards), "total": round(time.time() - t0)}
    print("SUMMARY " + json.dumps(summary), flush=True)
    return summary


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=2, memory=65536, timeout=3600)
def leakage(corpus: str = "blood_marrow") -> dict:
    """PBMC-10K screen of the corpus's filtered slice, without building anything."""
    from src.track_a import census as C

    spec = C.get_corpus(corpus)
    with C.open_census() as census:
        exp = census["census_data"][C.CENSUS_ORGANISM_KEY]
        obs = exp.obs.read(value_filter=spec["filter"], column_names=["dataset_id"]).concat().to_pandas()
        out = C.screen_leakage(census, obs["dataset_id"])
    out["n_cells_matching_filter"] = int(len(obs))
    print("LEAKAGE " + json.dumps(out, indent=1), flush=True)
    return out


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=2, memory=16384, timeout=3600)
def verify(stage: str = "full", corpus: str = "kidney") -> dict:
    """Independent read-back of a finished cache (dtypes, ranges, split, manifest shard hashes)."""
    import numpy as np

    from src.track_a import census as C

    out_dir = _out_dir(corpus, stage)
    with open(os.path.join(out_dir, "manifest.json")) as f:
        m = json.load(f)
    if m.get("corpus", "kidney") != corpus:
        raise RuntimeError(f"manifest corpus {m.get('corpus')!r} != {corpus!r}")
    vsize = m["vocab_size"]
    n = 0
    for s in m["shards"]:
        p = os.path.join(out_dir, "shards", s["file"])
        if C.sha256(p) != s["sha256"]:
            raise RuntimeError(f"sha mismatch: {p}")
        with np.load(p) as z:
            sh = {k: z[k] for k in ("indptr", "gene_ids", "bins", "n_cells")}
        C.validate_shard(sh, vsize)
        n += int(sh["n_cells"])
    with open(os.path.join(out_dir, "split.json")) as f:
        val = json.load(f)["val_cells"]
    sizes = [s["n_cells"] for s in m["shards"]]
    if not (n == m["n_cells"] and len(val) == m["split"]["n_val"]
            and all(0 <= r < sizes[s] for s, r in val) and len({tuple(x) for x in val}) == len(val)):
        raise RuntimeError("cell count / split inconsistent with manifest")
    out = {"ok": True, "corpus": corpus, "n_cells": n, "n_shards": len(sizes), "n_val": len(val), "out_dir": out_dir}
    print("VERIFY " + json.dumps(out), flush=True)
    return out


@app.function(image=load_image, volumes={VOL_PATH: vol}, cpu=4, memory=131072, timeout=3600)
def load_check(stage: str = "full", corpus: str = "kidney") -> dict:
    """Load the cache with the pretraining consumer (src.track_a.pretrain.KidneyCache) unchanged."""
    import time

    from src.track_a.pretrain import KidneyCache

    out_dir = _out_dir(corpus, stage)
    t0 = time.time()
    cache = KidneyCache(out_dir)
    gi, bi = cache.cell(int(cache.train[0]))
    out = {"corpus": corpus, "out_dir": out_dir, "n_cells": cache.n_cells, "n_train": int(len(cache.train)),
           "n_val": int(len(cache.val)), "nnz_stats": cache.nnz_stats(), "n_shards": len(cache.shard_keys),
           "n_tokens": int(sum(len(g) for g in cache.gene_ids)),
           "first_train_cell_len": int(len(gi)), "bins_range_first_cell": [int(bi.min()), int(bi.max())],
           "load_seconds": round(time.time() - t0, 1),
           "manifest_n_cells_matches": cache.manifest.get("n_cells") == cache.n_cells,
           "vocab_size": cache.manifest.get("vocab_size")}
    if not out["manifest_n_cells_matches"]:
        raise RuntimeError("KidneyCache n_cells != manifest n_cells")
    print("LOAD " + json.dumps(out), flush=True)
    return out


@app.local_entrypoint()
def main(stage: str = "smoke", force: bool = False, shard_size: int = 0, max_cells: int = -1,
         corpus: str = "kidney"):
    if stage == "verify":
        print(verify.remote("full", corpus))
        if corpus != "kidney":
            print(load_check.remote("full", corpus))
    elif stage == "verify-smoke":
        print(verify.remote("smoke", corpus))
        print(load_check.remote("smoke", corpus))
    elif stage == "load-check":
        print(load_check.remote("full", corpus))
    elif stage == "leakage":
        print(leakage.remote(corpus))
    else:
        print(run.remote(stage, force, shard_size, max_cells, corpus))
