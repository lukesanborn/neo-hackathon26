"""Track A — Cell-JEPA reimplemented from the paper (unofficial): Census kidney cache (CPU only).

    modal run modal_census.py --stage smoke                 # first ~5k kidney cells -> /vol/smoke/census/
    modal run --detach modal_census.py --stage full         # full 2023-05-15 kidney slice -> /vol/trackA/
    modal run modal_census.py --stage verify                # re-check the full cache, print summary

Flags: --force (rebuild plan + every shard), --shard-size N, --max-cells N (smoke default 5000).
CENSUS_STACK=legacy selects the cellxgene-census 1.15.0 fallback image (environment/census_image.py).

The driver (`run`) executes remotely, so `--detach` keeps the whole pipeline alive after the client
exits: plan (obs/var + gene mapping) -> one container per missing shard -> split + manifest.
Completed shards are skipped on rerun; manifest.json is written last.
"""

from __future__ import annotations

import json
import os

import modal

from environment.census_image import build_census_image

APP_NAME = "jepa-census"
VOL_PATH = "/vol"
VOCAB_PATH = f"{VOL_PATH}/checkpoints/scGPT_human/vocab.json"
FULL_DIR = f"{VOL_PATH}/trackA/kidney_census_2023-05-15"
SMOKE_DIR = f"{VOL_PATH}/smoke/census/kidney_census_2023-05-15"
MAX_SHARD_CONTAINERS = 6

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("jepa-perturb")
image = build_census_image()


def _load_vocab() -> dict[str, int]:
    with open(VOCAB_PATH) as f:
        return json.load(f)


def _vocab_size(vocab: dict[str, int]) -> int:
    return max(vocab.values()) + 1


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


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=2, memory=32768, timeout=12 * 3600)
def run(stage: str = "smoke", force: bool = False, shard_size: int = 0, max_cells: int = -1) -> dict:
    from src.track_a import census as C

    if stage not in ("smoke", "full"):
        raise ValueError(stage)
    out_dir = SMOKE_DIR if stage == "smoke" else FULL_DIR
    mc = (5000 if stage == "smoke" else None) if max_cells < 0 else (max_cells or None)
    ss = shard_size or (2000 if stage == "smoke" else C.DEFAULT_SHARD_SIZE)
    vocab = _load_vocab()
    vsize = _vocab_size(vocab)

    plan_path = os.path.join(out_dir, "plan.json")
    info = None
    if not force and os.path.exists(plan_path):
        with open(plan_path) as f:
            info = json.load(f)
        if (info.get("max_cells") != mc or info.get("shard_size") != ss
                or info.get("provenance") != C.plan_provenance(vocab)):
            raise RuntimeError(f"existing plan at {plan_path} was built with different inputs "
                               f"(max_cells / shard_size / filter / vocab); rerun with --force to rebuild")
        print("reusing existing plan", flush=True)
    if info is None:
        if os.path.exists(os.path.join(out_dir, "manifest.json")):
            os.remove(os.path.join(out_dir, "manifest.json"))  # cache is incomplete until finalize
        info = C.plan(out_dir, vocab, ss, mc)
        vol.commit()
    print(json.dumps({k: info[k] for k in ("n_cells_matching_filter", "n_cells", "n_datasets",
                                           "is_primary_data_true", "is_primary_data_false",
                                           "n_census_genes", "n_census_genes_in_vocab",
                                           "census_gene_vocab_coverage")}), flush=True)
    n_shards = len(info["shards"])
    print(f"{n_shards} shards of <= {ss} cells", flush=True)
    stats = list(build_shard.map([out_dir] * n_shards, range(n_shards), kwargs={"force": force}))
    vol.reload()
    manifest = C.finalize(out_dir, vsize, VOCAB_PATH, C.census_description(), extra={"stage": stage})
    vol.commit()
    summary = {k: manifest[k] for k in (
        "n_cells", "n_datasets", "is_primary_data_counts", "n_census_genes", "n_genes_in_vocab",
        "census_gene_vocab_coverage", "nonzero_count_vocab_coverage", "mean_genes_per_cell",
        "n_empty_cells", "n_noninteger_values", "n_shards", "package_versions")}
    summary["n_val"] = manifest["split"]["n_val"]
    summary["out_dir"] = out_dir
    summary["shards_skipped"] = sum(bool(s.get("skipped")) for s in stats)
    print("SUMMARY " + json.dumps(summary), flush=True)
    return summary


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=2, memory=16384, timeout=3600)
def verify(stage: str = "full") -> dict:
    """Independent read-back of a finished cache (dtypes, ranges, split, manifest shard hashes)."""
    import numpy as np

    from src.track_a import census as C

    out_dir = SMOKE_DIR if stage == "smoke" else FULL_DIR
    with open(os.path.join(out_dir, "manifest.json")) as f:
        m = json.load(f)
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
    out = {"ok": True, "n_cells": n, "n_shards": len(sizes), "n_val": len(val), "out_dir": out_dir}
    print("VERIFY " + json.dumps(out), flush=True)
    return out


@app.local_entrypoint()
def main(stage: str = "smoke", force: bool = False, shard_size: int = 0, max_cells: int = -1):
    if stage == "verify":
        print(verify.remote("full"))
    elif stage == "verify-smoke":
        print(verify.remote("smoke"))
    else:
        print(run.remote(stage, force, shard_size, max_cells))
