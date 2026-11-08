"""A28: perturbation-pair JEPA views. Can JEPA keep knockdown information if its views are built from knockdowns?

A27: with Cell-JEPA's views (masked cell vs the same full cell), JEPA continued pretraining trains knockdown
information OUT of the embedding (real-cell ID percentile 0.931 public scGPT -> 0.677 blood-JEPA). Here the
teacher instead sees OTHER cells with the same knockdown, so the invariance is to cell-to-cell noise and the
knockdown is the shared signal.

Four arms, all continued from public scGPT_human on the SAME cache (Replogle TRAINING knockdowns + controls,
src/track_a/perturb_pairs.py), same seed / order / masks / schedule; only the objective or views differ:
  pv_no_jepa    plain scGPT objective (reconstruction only)                    <- the control JEPA must beat
  pv_same_cell  JEPA center, Cell-JEPA views (masked cell vs same full cell)   <- A27's failure mode, matched
  pv_pair1      JEPA center, teacher = 1 other cell of the same knockdown
  pv_pair4      JEPA center, teacher = mean of 4 other cells of the same knockdown

Readouts (decide the question on VALIDATION genes; test once):
  ceiling   real-cell split-half ID percentile in each embedding space (A27 protocol, no fine-tuning)
  finetune  Replogle perturbation fine-tune (scgpt_paper, bins50, retrieval selection, seeds 1,2) -> ID percentile

    JEPA_APP_SUFFIX=-rhea .venv/bin/modal deploy modal_pretrain.py     # ships view_mode / pair_k
    JEPA_APP_SUFFIX=-rhea python diag/launch_perturb_views.py plan
    JEPA_APP_SUFFIX=-rhea python diag/launch_perturb_views.py cache
    JEPA_APP_SUFFIX=-rhea python diag/launch_perturb_views.py smoke      # 300 steps of pv_pair4 (throwaway export jepa_kidney_pv_smoke)
    JEPA_APP_SUFFIX=-rhea python diag/launch_perturb_views.py pretrain
    JEPA_APP_SUFFIX=-rhea python diag/launch_perturb_views.py finetune --wait
    JEPA_APP_SUFFIX=-rhea python diag/launch_perturb_views.py ceiling    # after the 4 checkpoints export
    JEPA_APP_SUFFIX=-rhea python diag/launch_perturb_views.py status
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

SUFFIX = os.environ.get("JEPA_APP_SUFFIX", "")
PRETRAIN_APP, PERTURB_APP = f"jepa-pretrain{SUFFIX}", f"jepa-perturb{SUFFIX}"
VOL = "jepa-perturb"
CKPT_ROOT, RUNS_ROOT = "trackA/checkpoints", "trackA/pretrain/runs"
EXP = "replogle_k562_v1"
CACHE = "/vol/trackA/replogle_train_pairs_d4237826c68c"

PT = dict(init_ckpt="scGPT_human", lr=3e-5, batch_size=32, seed=0, ema_momentum=0.996, jepa_variant="center",
          epochs=8, max_cells_per_epoch=None, max_val_cells=1024, max_minutes=150.0, reserve_minutes=5.0,
          log_every=200)
ARMS = {
    "pv_no_jepa": ("scgpt", {}),
    "pv_same_cell": ("jepa", {"view_mode": "same_cell"}),
    "pv_pair1": ("jepa", {"view_mode": "perturb_pair", "pair_k": 1}),
    "pv_pair4": ("jepa", {"view_mode": "perturb_pair", "pair_k": 4}),
}
FT = dict(method="scgpt_paper", epochs=15, input_transform="bins50", max_minutes=90.0)
FT_OVERRIDES = {"selection": "retrieval"}
RETR_WEIGHT = 1.0
USD_PER_GPU_HOUR = 6.25        # assumed; verify on modal.com/pricing
MIN_PT, MIN_FT = 25, 35        # guesses: ~50k cells x 8 epochs at ~0.05-0.08 s/step; A20 Replogle fine-tune


def ckpt_of(arm):
    return f"{ARMS[arm][0]}_kidney_{arm}"


def run_dir_of(arm):
    return f"{ARMS[arm][0]}_{arm}"


def ft_tag(arm, retr=False):
    return arm + ("_r1" if retr else "")


def ids_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "call_ids_perturb_views.json")


def load_ids():
    return json.load(open(ids_path())) if os.path.exists(ids_path()) else {}


def save_ids(ids):
    merged = {**load_ids(), **ids}
    tmp = ids_path() + ".tmp"
    json.dump(merged, open(tmp, "w"), indent=1)
    os.replace(tmp, ids_path())


def read_json(vol, path):
    try:
        return json.loads(b"".join(vol.read_file(path)))
    except Exception:
        return None


def call_state(cid):
    import modal

    if not cid:
        return "-"
    try:
        modal.FunctionCall.from_id(cid).get(timeout=0)
        return "done"
    except TimeoutError:
        return "running"
    except Exception as e:
        return f"FAILED {e!r:.70}"


def plan(a):
    for arm, (obj, ov) in ARMS.items():
        print(f"{arm:14s} objective={obj:6s} {ov or ''}  -> {CKPT_ROOT}/{ckpt_of(arm)}")
    n_ft = len(ARMS) * len(a.seeds.split(","))
    gpu_h = (len(ARMS) * MIN_PT + n_ft * MIN_FT) / 60
    print(f"\n{len(ARMS)} pretrains + {n_ft} fine-tunes ~ {gpu_h:.1f} GPU-h ~ ${gpu_h * USD_PER_GPU_HOUR:.0f} "
          f"(guessed timings, assumed ${USD_PER_GPU_HOUR}/h); wall-clock ~{(MIN_PT + MIN_FT) / 60:.1f} h if parallel")
    print("Reference, Replogle test ID pct (A22/A27): fine-tuned public scGPT 0.680; linear (PCA) 0.542; "
          "real-cell ceiling public scGPT 0.931 / blood-JEPA 0.677")


def cache(a):
    import modal

    fn = modal.Function.from_name(PRETRAIN_APP, "build_perturb_cache")
    print(json.dumps(fn.remote(force=a.force), indent=1))


def smoke(a):
    import modal

    fn = modal.Function.from_name(PRETRAIN_APP, "pretrain").with_options(gpu=a.gpu)
    obj, ov = ARMS["pv_pair4"]
    s = fn.remote(obj, False, "pv_smoke", {**PT, **ov, "epochs": 1, "max_cells_per_epoch": 9600,
                                           "max_val_cells": 256, "log_every": 50, "max_minutes": 20.0,
                                           "reserve_minutes": 1.0}, CACHE)
    print(json.dumps({k: v for k, v in s.items() if k != "export"}, indent=1, default=str))


def pretrain(a):
    import modal

    fn = modal.Function.from_name(PRETRAIN_APP, "pretrain").with_options(gpu=a.gpu)
    ids = load_ids()
    for arm, (obj, ov) in ARMS.items():
        key = f"pretrain|{arm}"
        if key in ids and not a.force:
            print(f"  {arm:14s} already submitted")
            continue
        ids[key] = fn.spawn(obj, False, arm, {**PT, **ov}, CACHE).object_id
        print(f"  {arm:14s} submitted -> {CKPT_ROOT}/{ckpt_of(arm)}")
        save_ids(ids)


def finetune(a):
    import modal

    vol = modal.Volume.from_name(VOL)
    fn = modal.Function.from_name(PERTURB_APP, "train_blackwell").with_options(gpu=a.gpu)
    ids = load_ids()
    seeds = [int(s) for s in a.seeds.split(",")]
    pending = set(ARMS)
    t0 = time.time()
    while pending:
        for arm in sorted(pending):
            if read_json(vol, f"{CKPT_ROOT}/{ckpt_of(arm)}/args.json") is None:
                st = call_state(ids.get(f"pretrain|{arm}"))
                if st.startswith("FAILED") or not a.wait:
                    print(f"  {arm}: checkpoint not ready ({st}); skipped")
                    pending.discard(arm)
                continue
            for s in seeds:
                key = f"finetune|{arm}|s{s}" + ("|r1" if a.retr else "")
                if key in ids and not a.force:
                    continue
                ov = {**FT_OVERRIDES, **({"retr_weight": RETR_WEIGHT} if a.retr else {})}
                ids[key] = fn.spawn(exp_id=EXP, method=FT["method"], seed=s, epochs=FT["epochs"], aux_weight=0.5,
                                    max_minutes=FT["max_minutes"], tag=ft_tag(arm, a.retr),
                                    ckpt=f"/vol/{CKPT_ROOT}/{ckpt_of(arm)}", input_transform=FT["input_transform"],
                                    cfg_overrides=ov).object_id
                print(f"  finetune {arm:14s} seed {s}" + (" +retrieval-loss" if a.retr else ""))
                save_ids(ids)
            pending.discard(arm)
        if pending:
            print(f"  [{(time.time() - t0) / 60:5.1f} min] waiting on: {', '.join(sorted(pending))}")
            time.sleep(60)


def ceiling(a):
    spaces = ",".join(f"{arm}=/vol/{CKPT_ROOT}/{ckpt_of(arm)}" for arm in ARMS)
    subprocess.run([".venv/bin/modal", "run", "diag/whiten_check.py", "--spaces", spaces], check=True)
    print("\nthen: modal volume get jepa-perturb experiments/replogle_k562_v1/latent/whiten_<arm>.npz <dir>; "
          f"python diag/whiten_analyze.py <dir> {','.join(ARMS)}")


def status(a):
    import modal

    vol = modal.Volume.from_name(VOL)
    ids = load_ids()
    for arm in ARMS:
        s = read_json(vol, f"{RUNS_ROOT}/{run_dir_of(arm)}/summary.json")
        pt = (f"{s['status']} recon {s['best_val_recon']:.2f}" if s else call_state(ids.get(f"pretrain|{arm}")))
        cells = []
        for sd in a.seeds.split(","):
            m = read_json(vol, f"experiments/{EXP}/runs/{FT['method']}_s{sd}_{ft_tag(arm, a.retr)}/metrics.json")
            k = f"finetune|{arm}|s{sd}" + ("|r1" if a.retr else "")
            cells.append(f"s{sd}:" + (f"{m['test']['delta_rmse']:.4f}" if m else str(call_state(ids.get(k)))[:9]))
        print(f"{arm:14s} {pt[:28]:28s} {' '.join(cells)}")
    print(f"\nID percentile: modal run diag/retrieval.py --exps {EXP} --split val   (then --split test once)")


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("plan", "cache", "smoke", "pretrain", "finetune", "ceiling", "status"))
    p.add_argument("--seeds", default="1,2")
    p.add_argument("--gpu", default="B300")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--retr", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    {"plan": plan, "cache": cache, "smoke": smoke, "pretrain": pretrain, "finetune": finetune,
     "ceiling": ceiling, "status": status}[a.stage](a)


if __name__ == "__main__":
    main()
