"""Modal batch app: prepare -> pilot/benchmark -> train (parallel) -> aggregate.

Track B: public pretrained scGPT -> matched fine-tuning with / without our JEPA auxiliary
objective on GEARS-processed Adamson. No foundation-model pretraining is launched here.

All downloads, caches, checkpoints and results live in the Modal Volume `jepa-perturb`;
nothing is synchronised back to the local machine automatically.

    modal run modal_app.py --stage prepare
    modal run modal_app.py --stage benchmark         # H100 (old stack) vs H100/B300 (new stack)
    modal run modal_app.py --stage pilot
    modal run --detach modal_app.py --stage train --epochs 15
    modal run modal_app.py --stage aggregate
    modal volume get jepa-perturb experiments/adamson_trackB_v1/report ./reports   # optional, small
"""

from __future__ import annotations

import json
import os
import time

import modal

from environment.image import build_image

APP_NAME = "jepa-perturb"
VOL_PATH = "/vol"
DEFAULT_EXP = "adamson_trackB_v1"
TRAIN_GPU = "H100"
BLACKWELL_GPU = "B300"
MAX_TRAIN_CONTAINERS = 4
# JEPA_CPU_ONLY=1 omits GPU functions so prepare/aggregate can deploy on a workspace
# without GPU billing enabled (Modal rejects the whole app otherwise).
CPU_ONLY = os.environ.get("JEPA_CPU_ONLY") == "1"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("jepa-perturb", create_if_missing=True)
hopper_image = build_image("hopper")
blackwell_image = build_image("blackwell")

RAW = f"{VOL_PATH}/raw"
CKPT = f"{VOL_PATH}/checkpoints/scGPT_human"


def cache_dir(split_seed: int) -> str:
    return f"{VOL_PATH}/data/adamson_gears002_simulation_seed{split_seed}"


def exp_dir(exp_id: str) -> str:
    return f"{VOL_PATH}/experiments/{exp_id}"


def _exp_config(exp_id: str) -> dict:
    p = os.path.join(exp_dir(exp_id), "experiment.json")
    with open(p) as f:
        return json.load(f)


# =============================================================================== prepare


@app.function(image=hopper_image, volumes={VOL_PATH: vol}, cpu=4, memory=32768, timeout=3600)
def prepare(exp_id: str = DEFAULT_EXP, split_seed: int = 1, n_eval_controls: int = 256,
            eval_control_seed: int = 0, force: bool = False) -> dict:
    """Download once, validate, build the fixed split + caches, run the simple baselines."""
    import numpy as np

    from src import baselines as B
    from src.data import build_cache, download_adamson, download_scgpt, load_cache
    from src.evaluate import evaluate_predictions, summarize
    from src.train import _write, _write_rows

    dl = download_adamson(RAW)
    ck = download_scgpt(CKPT)
    with open(os.path.join(CKPT, "vocab.json")) as f:
        vocab_genes = set(json.load(f))
    cdir = cache_dir(split_seed)
    if force or not os.path.exists(os.path.join(cdir, "manifest.json")):
        build_cache(dl["h5ad_path"], cdir, split_seed, n_eval_controls, eval_control_seed, dl, ck, vocab_genes)
        vol.commit()
    ds = load_cache(cdir)

    edir = exp_dir(exp_id)
    os.makedirs(edir, exist_ok=True)
    exp_path = os.path.join(edir, "experiment.json")
    if os.path.exists(exp_path):
        prev = json.load(open(exp_path))
        if prev["split_hash"] != ds.manifest["split_hash"]:
            raise RuntimeError(f"experiment {exp_id} already bound to split {prev['split_hash']}")
    else:
        _write(edir, "experiment.json", {"exp_id": exp_id, "track": "B", "cache_dir": cdir,
                                          "split_seed": split_seed, "split_hash": ds.manifest["split_hash"],
                                          "created_unix": time.time()})

    ctrl = ds.control_reference()
    results = {}
    for name, fn in [("no_change", B.no_change), ("training_mean", B.training_mean), ("linear", None)]:
        bdir = os.path.join(edir, "baselines", name)
        os.makedirs(bdir, exist_ok=True)
        t0 = time.time()
        info = {}
        out = {}
        for split_name in ("val", "test"):
            conds = ds.split[split_name]
            if name == "linear":
                preds, info = B.linear_gene_embedding(ds, conds, ctrl)
            else:
                preds = fn(ds, conds, ctrl)
            rows = evaluate_predictions(preds, ds, conds, ctrl)
            out[split_name] = (preds, rows, summarize(rows))
        preds, rows, summ = out["test"]
        meta = {"method": name, "track": "B", "checkpoint": "-", "split_hash": ds.manifest["split_hash"],
                "fit_on": "training interventions only", "info": info, "val": out["val"][2], "test": summ,
                "elapsed_seconds": time.time() - t0, "status": "completed"}
        _write(bdir, "metrics.json", meta)
        _write_rows(os.path.join(bdir, "per_intervention.csv"), rows)
        np.savez_compressed(os.path.join(bdir, "test_predictions.npz"),
                            conditions=np.array(list(preds)), preds=np.stack(list(preds.values())))
        results[name] = {"val_delta_rmse": out["val"][2]["delta_rmse"], "test_delta_rmse": summ["delta_rmse"],
                         "test_delta_pearson": summ["delta_pearson"],
                         "test_delta_pearson_n_defined": summ["delta_pearson_n_defined"]}
    vol.commit()
    m = ds.manifest
    gate = {
        "n_cells": m["n_cells"], "n_genes": m["n_genes"], "n_interventions": m["n_interventions"],
        "split_sizes": m["split_sizes"], "split_hash": m["split_hash"],
        "targets_not_in_gene_list": m["targets_not_in_gene_list"],
        "vocab_coverage": {k: v for k, v in (m["vocab_coverage"] or {}).items() if k != "genes_not_in_vocab_first50"},
        "baselines": results,
    }
    print(json.dumps(gate, indent=2))
    return gate


def gpu_function(**kw):
    """app.function, unless CPU_ONLY — then the function stays a plain local callable."""
    return (lambda f: f) if CPU_ONLY else app.function(**kw)


# =============================================================================== pilot / benchmark


def _pilot(exp_id: str, config_name: str, steps: int, aux_weight: float) -> dict:
    from src.data import load_cache
    from src.train import _write, run_pilot

    vol.reload()
    ds = load_cache(_exp_config(exp_id)["cache_dir"])
    out = os.path.join(exp_dir(exp_id), "benchmark")
    os.makedirs(out, exist_ok=True)
    rep = run_pilot(ds, CKPT, steps=steps, aux_weight=aux_weight)
    rep["config_name"] = config_name
    _write(out, f"{config_name}.json", rep)
    vol.commit()
    return rep


@gpu_function(image=hopper_image, gpu=TRAIN_GPU, volumes={VOL_PATH: vol}, memory=65536, timeout=3600)
def pilot(exp_id: str = DEFAULT_EXP, config_name: str = "hopper_H100", steps: int = 150,
          aux_weight: float = 0.5) -> dict:
    return _pilot(exp_id, config_name, steps, aux_weight)


@gpu_function(image=blackwell_image, gpu=BLACKWELL_GPU, volumes={VOL_PATH: vol}, memory=65536, timeout=3600)
def pilot_blackwell(exp_id: str = DEFAULT_EXP, config_name: str = "blackwell_B300", steps: int = 150,
                    aux_weight: float = 0.5) -> dict:
    return _pilot(exp_id, config_name, steps, aux_weight)


# =============================================================================== train


def _train(exp_id, method, seed, epochs, aux_weight, max_minutes, tag) -> dict:
    from src.data import load_cache
    from src.train import TrainConfig, run_training

    vol.reload()
    ds = load_cache(_exp_config(exp_id)["cache_dir"])
    cfg = TrainConfig(method=method, seed=seed, epochs=epochs, aux_weight=aux_weight, max_minutes=max_minutes)
    run_dir = os.path.join(exp_dir(exp_id), "runs", f"{method}_s{seed}_{tag}")
    if os.path.exists(os.path.join(run_dir, "metrics.json")):
        raise FileExistsError(f"{run_dir} already completed; runs are immutable — use a new tag")
    return run_training(ds, CKPT, cfg, run_dir, commit=vol.commit)


@gpu_function(image=hopper_image, gpu=TRAIN_GPU, volumes={VOL_PATH: vol}, memory=65536,
              timeout=4 * 3600, max_containers=MAX_TRAIN_CONTAINERS)
def train(exp_id: str, method: str, seed: int, epochs: int = 15, aux_weight: float = 0.5,
          max_minutes: float = 150.0, tag: str = "v1") -> dict:
    return _train(exp_id, method, seed, epochs, aux_weight, max_minutes, tag)


@gpu_function(image=blackwell_image, gpu=BLACKWELL_GPU, volumes={VOL_PATH: vol}, memory=65536,
              timeout=4 * 3600, max_containers=MAX_TRAIN_CONTAINERS)
def train_blackwell(exp_id: str, method: str, seed: int, epochs: int = 15, aux_weight: float = 0.5,
                    max_minutes: float = 150.0, tag: str = "v1") -> dict:
    return _train(exp_id, method, seed, epochs, aux_weight, max_minutes, tag)


# =============================================================================== aggregate


@app.function(image=hopper_image, volumes={VOL_PATH: vol}, cpu=4, memory=32768, timeout=1800)
def aggregate(exp_id: str = DEFAULT_EXP) -> dict:
    from src.data import load_cache
    from src.report import aggregate as agg

    vol.reload()
    ds = load_cache(_exp_config(exp_id)["cache_dir"])
    out = os.path.join(exp_dir(exp_id), "report")
    rep = agg(ds, exp_dir(exp_id), out)
    vol.commit()
    with open(os.path.join(out, "results.md")) as f:
        print(f.read())
    return {"claim": rep["claim"], "report_dir": out}


# =============================================================================== launcher


@app.local_entrypoint()
def main(stage: str = "all", exp_id: str = DEFAULT_EXP, epochs: int = 15, aux_weight: float = 0.5,
         max_minutes: float = 150.0, seeds: str = "1,2", tag: str = "v1", stack: str = "hopper",
         gpu: str = "", steps: int = 150, blackwell_gpus: str = "B300"):
    def show(x):
        print(json.dumps(x, indent=2, default=str))

    if stage in ("prepare", "all"):
        show(prepare.remote(exp_id))

    if stage == "benchmark":
        calls = {"hopper_H100": pilot.spawn(exp_id, "hopper_H100", steps, aux_weight),
                 "blackwell_stack_H100": pilot_blackwell.with_options(gpu="H100").spawn(
                     exp_id, "blackwell_stack_H100", steps, aux_weight)}
        for g in [g for g in blackwell_gpus.split(",") if g]:
            calls[f"blackwell_{g}"] = pilot_blackwell.with_options(gpu=g).spawn(exp_id, f"blackwell_{g}", steps, aux_weight)
        for name, fc in calls.items():
            try:
                r = fc.get()
                show({name: {"gpu": r["environment"]["gpu"], "torch": r["environment"]["torch"],
                             "gates": r["gates"],
                             **{m: {k: v[k] for k in ("seconds_per_step", "validation_seconds", "peak_gpu_mem_gb",
                                                      "projected_15_epoch_minutes")}
                                for m, v in r["methods"].items()}}})
            except Exception as e:  # report, don't hide, failures
                show({name: {"status": "failed", "error": repr(e)[:2000]}})

    if stage in ("pilot", "all"):
        fn = pilot if stack == "hopper" else pilot_blackwell.with_options(gpu=gpu or BLACKWELL_GPU)
        rep = fn.remote(exp_id, f"pilot_{stack}", steps, aux_weight)
        show({"gates": rep["gates"], "all_gates_pass": rep["all_gates_pass"],
              **{m: {k: v[k] for k in ("seconds_per_step", "validation_seconds", "peak_gpu_mem_gb",
                                       "projected_15_epoch_minutes", "teacher_diagnostics")}
                 for m, v in rep["methods"].items()}})
        if stage == "all" and not rep["all_gates_pass"]:
            raise SystemExit("execution gate failed — fix on one GPU before launching training")

    if stage in ("train", "all"):
        fn = train if stack == "hopper" else train_blackwell
        if gpu:
            fn = fn.with_options(gpu=gpu)
        jobs = [(m, int(s)) for m in ("scgpt", "jepa") for s in seeds.split(",")]
        calls = [(m, s, fn.spawn(exp_id, m, s, epochs, aux_weight, max_minutes, tag)) for m, s in jobs]
        print(f"submitted {len(calls)} training jobs: " + ", ".join(f"{m}_s{s}" for m, s, _ in calls))
        for m, s, fc in calls:
            try:
                r = fc.get()
                show({f"{m}_s{s}": {k: r[k] for k in ("status", "epochs_completed", "selected_epoch",
                                                       "elapsed_seconds", "peak_gpu_mem_gb", "reload_ok")}
                      | {"test_delta_rmse": r["test"]["delta_rmse"]}})
            except Exception as e:
                show({f"{m}_s{s}": {"status": "failed", "error": repr(e)[:2000]}})

    if stage in ("aggregate", "all"):
        show(aggregate.remote(exp_id))
