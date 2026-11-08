"""Track A — Cell-JEPA reimplemented from the paper (unofficial): kidney pretraining on Modal.

Two objectives from an identical random initialisation on the CELLxGENE Census 2023-05-15 kidney
cache (built by the data unit): "jepa" (1000 x cosine JEPA + masked MSE) and "scgpt" (masked
MSE only, the matched baseline). See src/track_a/pretrain_model.py for the method and the
deviation log.

    modal run modal_pretrain.py --stage smoke                      # synthetic cache, /vol/smoke/pretrain only
    modal run modal_pretrain.py --stage pilot --gpu H100           # throughput pilot on the real cache
    modal run modal_pretrain.py --stage pilot --smoke --steps 60   # pilot on the synthetic cache
    modal run --detach modal_pretrain.py --stage train --epochs 4 --batch-size 32 --lr 1e-4
    modal run modal_pretrain.py --stage export                     # re-export from the run dirs
    modal run modal_pretrain.py --stage check                      # load exports via src.models.build_scgpt

Continued pretraining from a checkpoint (e.g. public scGPT_human) on another corpus:

    modal run modal_pretrain.py --stage pilot --gpu B300 --batch-sizes 32 --lr 3e-5 \
        --init-ckpt scGPT_human --tag scgpt_human_init               # pilot -> pilot/<gpu>_<tag>.json
    modal run --detach modal_pretrain.py --stage train --init-ckpt scGPT_human --lr 3e-5 \
        --cache /vol/trackA/blood_marrow_census_2023-05-15_2M --tag blood_cont ...

--cache defaults to the kidney cache; ``overrides["cache_dir"]`` does the same for direct calls.

`train` is resumable: re-running with the same --tag continues from last.pt (each call stops
itself after --max-minutes and exports what it has).
"""

from __future__ import annotations

import json
import os

import modal

from environment.image import build_image

# JEPA_APP_SUFFIX deploys this app under its own name (e.g. "jepa-pretrain-rhea") so an
# experiment branch can be deployed into a shared workspace without replacing the code that
# someone else's running jobs use. The Volume is shared either way.
APP_SUFFIX = os.environ.get("JEPA_APP_SUFFIX", "")
APP_NAME = f"jepa-pretrain{APP_SUFFIX}"
VOL_PATH = "/vol"
GPU = "H100"
VOCAB = f"{VOL_PATH}/checkpoints/scGPT_human/vocab.json"
REAL = {"cache": f"{VOL_PATH}/trackA/kidney_census_2023-05-15", "runs": f"{VOL_PATH}/trackA/pretrain/runs",
        "export": f"{VOL_PATH}/trackA/checkpoints", "pilot": f"{VOL_PATH}/trackA/pretrain_pilot"}
SMOKE = {"cache": f"{VOL_PATH}/smoke/pretrain/cache", "runs": f"{VOL_PATH}/smoke/pretrain/runs",
         "export": f"{VOL_PATH}/smoke/pretrain/checkpoints", "pilot": f"{VOL_PATH}/smoke/pretrain/pilot"}
# Smoke schedule: full architecture, tiny data; the jepa run is extended by one epoch afterwards
# to exercise resume.
SMOKE_OVERRIDES = {"epochs": 2, "max_cells_per_epoch": 3200, "max_val_cells": 256, "log_every": 20,
                   "max_minutes": 15.0, "reserve_minutes": 1.0}

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("jepa-perturb", create_if_missing=True)
image = build_image("blackwell")


def _paths(smoke: bool) -> dict:
    return SMOKE if smoke else REAL


def _export_dir(p: dict, objective: str, tag: str) -> str:
    return os.path.join(p["export"], f"{objective}_kidney" if tag == "v1" else f"{objective}_kidney_{tag}")


def _cache_dir(p: dict, cache: str | None, override: str | None = None) -> str:
    if cache and override and os.path.normpath(cache) != os.path.normpath(override):
        raise ValueError(f"conflicting cache dirs: cache={cache!r} overrides['cache_dir']={override!r}")
    d = os.path.normpath(cache or override or p["cache"])
    if not os.path.exists(os.path.join(d, "split.json")):
        raise FileNotFoundError(f"pretraining cache not ready (no split.json): {d}")
    return d


def _plain(x):
    return json.loads(json.dumps(x, default=str))  # local client has no torch / numpy


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=4, memory=16384, timeout=1800)
def make_smoke_cache(n_cells: int = 4000, force: bool = False) -> dict:
    from src.track_a.pretrain import make_synthetic_cache

    vol.reload()
    out = SMOKE["cache"]
    if not force and os.path.exists(os.path.join(out, "manifest.json")):
        with open(os.path.join(out, "manifest.json")) as f:
            return _plain(json.load(f))
    man = make_synthetic_cache(out, VOCAB, n_cells=n_cells)
    vol.commit()
    return _plain(man)


@app.function(image=image, gpu=GPU, volumes={VOL_PATH: vol}, cpu=8, memory=65536, timeout=24 * 3600)
def pretrain(objective: str, smoke: bool = False, tag: str = "v1", overrides: dict | None = None,
             cache: str | None = None) -> dict:
    """``cache`` (or ``overrides["cache_dir"]``) selects another corpus; default = kidney / smoke cache.
    ``overrides["init_ckpt"]`` = continued pretraining from a checkpoint (see PretrainConfig)."""
    from src.track_a.pretrain import PretrainConfig, run_pretraining

    vol.reload()
    p = _paths(smoke)
    ov = dict(overrides or {})
    cache_dir = _cache_dir(p, cache, ov.pop("cache_dir", None))
    if cache_dir != os.path.normpath(p["cache"]) and tag == "v1":
        # exports are named <obj>_kidney[_<tag>]: never let another corpus overwrite the canonical dirs
        raise ValueError(f"non-default cache {cache_dir} needs an explicit tag (not 'v1')")
    cfg = PretrainConfig(objective=objective, **ov)
    run_dir = os.path.join(p["runs"], f"{objective}_{tag}")
    export_dir = _export_dir(p, objective, tag)
    s = run_pretraining(cfg, cache_dir, VOCAB, run_dir, export_dir, commit=vol.commit)
    return _plain(s)


@app.function(image=image, gpu=GPU, volumes={VOL_PATH: vol}, cpu=8, memory=65536, timeout=3 * 3600)
def pilot(smoke: bool = False, gpu_label: str = GPU, steps: int = 200, warmup: int = 20,
          batch_sizes: str = "32,256", objectives: str = "jepa,scgpt", cache: str | None = None,
          init_ckpt: str | None = None, base_lr: float = 1e-4, tag: str = "v1") -> dict:
    from src.track_a.pretrain import run_pilot

    vol.reload()
    p = _paths(smoke)
    out = os.path.join(p["pilot"], f"{gpu_label}.json" if tag == "v1" else f"{gpu_label}_{tag}.json")
    rep = run_pilot(_cache_dir(p, cache), VOCAB, out, objectives=tuple(objectives.split(",")),
                    batch_sizes=tuple(int(b) for b in batch_sizes.split(",")), steps=steps, warmup=warmup,
                    base_lr=base_lr, init_ckpt=init_ckpt or None)
    vol.commit()
    rep["out_path"] = out
    return _plain(rep)


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=4, memory=32768, timeout=1800)
def export(objective: str, smoke: bool = False, tag: str = "v1") -> dict:
    from src.models import SimpleVocab
    from src.track_a.pretrain import export_checkpoint

    vol.reload()
    p = _paths(smoke)
    run_dir = os.path.join(p["runs"], f"{objective}_{tag}")
    meta = json.load(open(os.path.join(run_dir, "config.json")))
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    export_dir = _export_dir(p, objective, tag)
    out = export_checkpoint(run_dir, export_dir, SimpleVocab(VOCAB), meta, summary)
    vol.commit()
    return _plain(out)


PERTURB_PAIR_CACHE = f"{VOL_PATH}/trackA/replogle_train_pairs_d4237826c68c"
REPLOGLE_CACHE = f"{VOL_PATH}/data/replogle_k562_essential_gears_simulation_seed1"


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=8, memory=65536, timeout=3600)
def build_perturb_cache(force: bool = False) -> dict:
    """A28: pretraining cache of Replogle TRAINING-condition cells (+ controls) with per-cell knockdown groups,
    for view_mode="perturb_pair". Validation / test knockdowns are excluded (src/track_a/perturb_pairs.py)."""
    from src.data import load_cache
    from src.models import HGNC_ALIASES
    from src.track_a.perturb_pairs import build_perturb_cache as build

    vol.reload()
    out = PERTURB_PAIR_CACHE
    if not force and os.path.exists(os.path.join(out, "manifest.json")):
        return _plain(json.load(open(os.path.join(out, "manifest.json"))))
    ds = load_cache(REPLOGLE_CACHE, mmap=True)
    assert (ds.manifest or {}).get("split_hash", "d4237826c68c") == "d4237826c68c", ds.manifest.get("split_hash")
    vocab = json.load(open(VOCAB))
    al = {**HGNC_ALIASES, **((ds.manifest or {}).get("extra_hgnc_aliases") or {})}
    man = build(ds, out, vocab, al)
    vol.commit()
    return _plain(man)


MASK_UNITS_DIR = f"{VOL_PATH}/trackA/mask_units"
ENRICHR_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={}"


def mask_units_path(kind: str, n_programs: int = 64, library: str = "MSigDB_Hallmark_2020",
                    cache: str | None = None) -> str:
    """Canonical units-file path for views knob 2a (``PretrainConfig.mask_units_path``).

    Programs are co-expression clusters learned FROM a corpus, so the corpus is part of the
    identity: kidney programs are not blood programs and must not share a filename. Pathways are
    an external gene-set library and are corpus-independent."""
    if kind != "program":
        return os.path.join(MASK_UNITS_DIR, f"pathway_{library}.json")
    corpus = os.path.basename((cache or REAL["cache"]).rstrip("/"))
    return os.path.join(MASK_UNITS_DIR, f"program_k{n_programs}__{corpus}.json")


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=8, memory=65536, timeout=3600)
def build_mask_units(kind: str = "program", n_programs: int = 64, n_cells: int = 30000, min_cells: int = 20,
                     library: str = "MSigDB_Hallmark_2020", seed: int = 0, cache: str | None = None,
                     force: bool = False) -> dict:
    """Write the units file for mask_unit="program" or "pathway" (views knob 2a).

    program: co-expression programs clustered from ``n_cells`` TRAINING cells of the pretraining cache
             (validation cells are never read). pathway: an Enrichr text library (default MSigDB Hallmark),
             symbols mapped to the scGPT vocab. Both report token coverage on the sampled cells."""
    import urllib.request

    import numpy as np

    from src.track_a.masking import build_programs, gene_sets_to_units, parse_gene_sets, token_coverage
    from src.track_a.pretrain import KidneyCache

    vol.reload()
    out = mask_units_path(kind, n_programs, library, cache)
    if os.path.exists(out) and not force:
        d = json.load(open(out))
        return _plain({"path": out, "existing": True, "stats": d.get("stats")})
    c = KidneyCache(_cache_dir(REAL, cache))
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(c.train, size=min(n_cells, len(c.train)), replace=False))
    cells = [c.cell(int(g)) for g in pick]
    if kind == "program":
        d = build_programs(cells, n_programs=n_programs, min_cells=min_cells, seed=seed)
        d["source"] = f"co-expression programs from {len(cells)} training cells of {c.cache_dir}"
    elif kind == "pathway":
        url = ENRICHR_URL.format(library)
        with urllib.request.urlopen(url, timeout=120) as r:
            text = r.read().decode()
        stoi = json.load(open(VOCAB))
        units, names, stats = gene_sets_to_units(parse_gene_sets(text), stoi)
        stats["token_coverage"] = token_coverage(units, cells)
        d = {"kind": "pathway", "units": units, "unit_names": names, "stats": stats, "source": url}
    else:
        raise ValueError(f"kind must be program or pathway, got {kind!r}")
    os.makedirs(MASK_UNITS_DIR, exist_ok=True)
    with open(out + ".tmp", "w") as f:
        json.dump(d, f)
    os.replace(out + ".tmp", out)
    vol.commit()
    return _plain({"path": out, "existing": False, "stats": d["stats"]})


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=4, memory=32768, timeout=1800)
def integration_check(smoke: bool = True, objectives: str = "jepa,scgpt", tag: str = "v1") -> dict:
    """Load each export into the fine-tuning model exactly as units 3/5 will."""
    import torch

    from src.models import build_scgpt

    vol.reload()
    p = _paths(smoke)
    res = {}
    for obj in objectives.split(","):
        d = _export_dir(p, obj, tag)
        model, vocab, args, info = build_scgpt(d, seed=0)
        sd = torch.load(os.path.join(d, "best_model.pt"), map_location="cpu")
        own = model.state_dict()
        max_diff = max(float((own[k] - sd[k]).abs().max()) for k in own
                       if k.startswith(("encoder", "value_encoder", "transformer_encoder")))
        res[obj] = {"n_loaded": info["n_loaded"], "n_expected_with_prefix": info["n_expected_with_prefix"],
                    "not_loaded_with_prefix": info["not_loaded_with_prefix"],
                    "shape_or_name_mismatch": info["shape_or_name_mismatch"],
                    "newly_initialised": info["newly_initialised"], "loaded_weights_max_abs_diff": max_diff,
                    "ok": info["not_loaded_with_prefix"] == [] and info["shape_or_name_mismatch"] == []
                    and max_diff == 0.0,
                    "args": {k: args[k] for k in ("embsize", "nheads", "d_hid", "nlayers", "n_layers_cls", "dropout",
                                                  "input_style", "objective", "epochs_completed")},
                    "vocab_size": len(vocab)}
    return _plain(res)


@app.local_entrypoint()
def main(stage: str = "smoke", objectives: str = "jepa,scgpt", tag: str = "v1", smoke: bool = False,
         gpu: str = GPU, epochs: int = 4, batch_size: int = 32, lr: float = 1e-4, max_cells_per_epoch: int = 0,
         max_minutes: float = 23 * 60.0, steps: int = 200, warmup: int = 20, batch_sizes: str = "32,256",
         seed: int = 0, ema_momentum: float = 0.996, jepa_variant: str = "faithful",
         variants: str = "faithful,center,vicreg,token", cache: str = "", init_ckpt: str = ""):
    def show(x):
        print(json.dumps(x, indent=2, default=str), flush=True)

    objs = [o for o in objectives.split(",") if o]

    if stage == "smoke" and (cache or init_ckpt):
        raise SystemExit("--cache / --init-ckpt are not used by --stage smoke (synthetic cache, random init)")

    if stage == "smoke":
        show({"smoke_cache": make_smoke_cache.remote()})
        calls = {o: pretrain.with_options(gpu=gpu).spawn(o, True, tag, SMOKE_OVERRIDES) for o in objs}
        res = {o: c.get() for o, c in calls.items()}
        if "jepa" in res:  # resume check: extend the jepa run by one epoch from last.pt
            res["jepa_resumed"] = pretrain.with_options(gpu=gpu).remote(
                "jepa", True, tag, {**SMOKE_OVERRIDES, "epochs": SMOKE_OVERRIDES["epochs"] + 1})
        for k, r in res.items():
            show({k: {kk: r[kk] for kk in ("status", "epochs_completed", "global_step", "best_val_recon",
                                            "best_epoch", "init_state_hash", "peak_gpu_mem_gb", "history")}})
        hashes = {r["init_state_hash"] for r in res.values()}
        chk = integration_check.remote(True, objectives, tag)
        show({"integration_check": chk})
        ok = len(hashes) == 1 and all(c["ok"] for c in chk.values())
        show({"matched_init_across_objectives": len(hashes) == 1, "integration_ok": ok})
        if not ok:
            raise SystemExit("smoke failed")

    elif stage == "pilot":
        # pilot lr = --lr scaled linearly by batch size from bs 32 (default --lr 1e-4 as before)
        show(pilot.with_options(gpu=gpu).remote(smoke, gpu, steps, warmup, batch_sizes, ",".join(objs),
                                                cache or None, init_ckpt or None, lr, tag))

    elif stage == "train":
        ov = {"epochs": epochs, "batch_size": batch_size, "lr": lr, "max_minutes": max_minutes, "seed": seed,
              "ema_momentum": ema_momentum, "max_cells_per_epoch": max_cells_per_epoch or None,
              "init_ckpt": init_ckpt or None}
        calls = {o: pretrain.with_options(gpu=gpu).spawn(
            o, smoke, tag, {**ov, **({"jepa_variant": jepa_variant} if o == "jepa" else {})}, cache or None)
            for o in objs}
        print(f"submitted pretraining: {list(calls)} overrides={ov}", flush=True)
        for o, c in calls.items():
            try:
                r = c.get()
                show({o: {k: r[k] for k in ("status", "epochs_completed", "global_step", "best_val_recon",
                                            "best_epoch", "elapsed_seconds_this_call", "export")}})
            except Exception as e:
                show({o: {"status": "failed", "error": repr(e)[:2000]}})

    elif stage == "ablate":
        # Short anti-collapse ablations on kidney data only (no Adamson data involved). Each run gets
        # its own tag, so exports go to <obj>_kidney_abl_<variant> and never to the canonical dirs.
        ov = {"epochs": epochs, "batch_size": batch_size, "lr": lr, "max_minutes": max_minutes, "seed": seed,
              "ema_momentum": ema_momentum, "max_cells_per_epoch": max_cells_per_epoch or None,
              "init_ckpt": init_ckpt or None}
        jobs = {"scgpt": ("scgpt", {})}
        jobs.update({f"jepa_{v}": ("jepa", {"jepa_variant": v}) for v in variants.split(",") if v})
        calls = {name: pretrain.with_options(gpu=gpu).spawn(o, smoke, f"abl_{name}", {**ov, **extra}, cache or None)
                 for name, (o, extra) in jobs.items()}
        print(f"submitted ablations: {list(calls)} overrides={ov}", flush=True)
        for name, c in calls.items():
            try:
                r = c.get()
                show({name: {"status": r["status"], "best_val_recon": r["best_val_recon"],
                             "history": r.get("history")}})
            except Exception as e:
                show({name: {"status": "failed", "error": repr(e)[:2000]}})

    elif stage == "mask-units":
        # views knob 2a: build the program + pathway units files (CPU; needs the kidney cache)
        calls = {k: build_mask_units.spawn(k, cache=cache or None) for k in ("program", "pathway")}
        for k, c in calls.items():
            try:
                show({k: c.get()})
            except Exception as e:
                show({k: {"status": "failed", "error": repr(e)[:2000]}})

    elif stage == "export":
        for o in objs:
            show({o: export.remote(o, smoke, tag)})

    elif stage == "check":
        show(integration_check.remote(smoke, objectives, tag))

    else:
        raise SystemExit(f"unknown stage {stage}")
