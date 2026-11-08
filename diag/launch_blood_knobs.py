"""Masking knobs (2a / 2b) on the blood corpus, reusing A23's two trained arms.

A23 already trained both controls from public scGPT_human on the 2M blood + bone-marrow cache:
  scgpt_kidney_blood_lr3e5   plain scGPT objective   -> the no-JEPA control
  jepa_kidney_blood_lr3e5    JEPA center, random 15% -> the baseline for BOTH knobs
so only the six new masking variants need pretraining. Every new arm copies A23's config exactly
(init scGPT_human, 2 epochs over the full 2M cache = 123,750 steps, batch 32, lr 3e-5, seed 0,
jepa_variant "center", bins50 inputs); the ONLY difference is mask_unit / mask_ratio.

  2a what unit gets masked   program | pathway | top_expressed      (ratio 0.15; random = baseline)
  2b how much gets masked    0.30 | 0.50 | 0.80                      (random genes; 0.15 = baseline)

Gene programs are clustered from the BLOOD cache, not kidney: programs are co-expression structure
and do not transfer between tissues (blood keeps 19,464 genes vs kidney's 25,271).

Fine-tuning holds everything constant so the only JEPA in the comparison is the pretraining one:
recipe `scgpt_paper` (MSE + ECS, no JEPA term), Replogle, 2 seeds, checkpoint chosen by validation
identification percentile (A24 `selection="retrieval"`), test scored once.

    python diag/launch_blood_knobs.py plan
    python diag/launch_blood_knobs.py pretrain          # 6 jobs, ~1h45 each, all parallel
    python diag/launch_blood_knobs.py finetune --wait   # chains as each checkpoint exports
    python diag/launch_blood_knobs.py status
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

SUFFIX = os.environ.get("JEPA_APP_SUFFIX", "")
PRETRAIN_APP, PERTURB_APP = f"jepa-pretrain{SUFFIX}", f"jepa-perturb{SUFFIX}"
VOL = "jepa-perturb"
CKPT_ROOT, RUNS_ROOT = "trackA/checkpoints", "trackA/pretrain/runs"
EXP = "replogle_k562_v1"                      # A20/A22 split d4237826c68c, 103 test genes
BLOOD_CACHE = "/vol/trackA/blood_marrow_census_2023-05-15_2M"
PROGRAMS = "/vol/trackA/mask_units/program_k64__blood_marrow_census_2023-05-15_2M.json"
PATHWAYS = "/vol/trackA/mask_units/pathway_MSigDB_Hallmark_2020.json"

# A23's exact pretraining config. max_minutes is a safety valve well above the ~106 min A23 took;
# hitting it would export a truncated model and silently break the matched comparison.
PT = dict(epochs=2, batch_size=32, lr=3e-5, seed=0, ema_momentum=0.996, init_ckpt="scGPT_human",
          jepa_variant="center", max_cells_per_epoch=None, max_val_cells=1024,
          max_minutes=180.0, reserve_minutes=5.0, log_every=500)
FT = dict(method="scgpt_paper", epochs=15, input_transform="bins50", max_minutes=90.0)
FT_OVERRIDES = {"selection": "retrieval"}     # A24: pick the epoch with the best validation id_pct
# A25 measured the InfoNCE retrieval loss as the single biggest lever (+0.03 to +0.055 id_pct on
# every arm, far larger than any pretraining difference). --retr 1 reruns the fine-tune with it on,
# to test whether a masking knob adds anything ON TOP of the setting that actually works.
RETR_WEIGHT = 1.0                             # same weight A25 used (cfg_overrides retr_weight)

# label -> (checkpoint name, pretraining overrides or None when the checkpoint already exists)
REUSED = {"ctrl_no_jepa": "scgpt_kidney_blood_lr3e5",      # A23 plain scGPT
          "base_random15": "jepa_kidney_blood_lr3e5"}      # A23 JEPA center, random 15%
NEW = {
    "unit_program": dict(mask_unit="program", mask_ratio=0.15, mask_units_path=PROGRAMS),
    "unit_pathway": dict(mask_unit="pathway", mask_ratio=0.15, mask_units_path=PATHWAYS),
    "unit_topexpr": dict(mask_unit="top_expressed", mask_ratio=0.15),
    "ratio_30": dict(mask_unit="random", mask_ratio=0.30),
    "ratio_50": dict(mask_unit="random", mask_ratio=0.50),
    "ratio_80": dict(mask_unit="random", mask_ratio=0.80),
}
KNOB = {"ctrl_no_jepa": "control", "base_random15": "baseline",
        "unit_program": "2a", "unit_pathway": "2a", "unit_topexpr": "2a",
        "ratio_30": "2b", "ratio_50": "2b", "ratio_80": "2b"}
TAG = {k: f"bk_{k}" for k in NEW}             # run / checkpoint tag for the new arms
ARMS = list(REUSED) + list(NEW)

MIN_PT, MIN_FT = 106, 35                      # measured: A23 pretrain, A20 Replogle fine-tune
USD_PER_GPU_HOUR = 6.25                       # assumed; verify on modal.com/pricing
# A22 reference points on this exact split (ID percentile, 103 test genes)
REFERENCE = {"public scGPT -> scGPT FT": 0.680, "scGPT gene vectors (ridge)": 0.675,
             "linear (PCA gene embedding)": 0.542, "chance": 0.500}


def ckpt_of(label: str) -> str:
    return REUSED[label] if label in REUSED else f"jepa_kidney_{TAG[label]}"


def run_dir_of(label: str) -> str | None:
    return None if label in REUSED else f"jepa_{TAG[label]}"


def ft_tag(label: str, retr: float = 0.0) -> str:
    return f"bk_{label}" + ("_r1" if retr else "")


def ids_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "call_ids_blood_knobs.json")


def load_ids() -> dict:
    return json.load(open(ids_path())) if os.path.exists(ids_path()) else {}


def save_ids(ids: dict) -> None:
    """Merge rather than overwrite: several chain processes may write here concurrently."""
    merged = load_ids()
    merged.update(ids)
    ids.update(merged)
    tmp = ids_path() + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=1)
    os.replace(tmp, ids_path())


def read_json(vol, path: str):
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


# ----------------------------------------------------------------------------- stages


def plan(args):
    print(f"{'arm':16s} {'knob':9s} {'pretrain':9s} {'masking':34s} checkpoint")
    for a in ARMS:
        ov = NEW.get(a)
        mask = ("random 15% (A23)" if a == "base_random15" else
                "n/a - no JEPA term" if a == "ctrl_no_jepa" else
                f"{ov['mask_unit']} @ {ov['mask_ratio']:.0%}")
        print(f"{a:16s} {KNOB[a]:9s} {'REUSE' if a in REUSED else 'train':9s} {mask:34s} {ckpt_of(a)}")
    n_new, n_ft = len(NEW), len(ARMS) * len(args.seeds.split(","))
    gpu_h = (n_new * MIN_PT + n_ft * MIN_FT) / 60
    print(f"\n{n_new} pretrains (6 x ~{MIN_PT} min, parallel) + {n_ft} Replogle fine-tunes")
    print(f"~{gpu_h:.1f} GPU-hours ~= ${gpu_h * USD_PER_GPU_HOUR:.0f} (assumed ${USD_PER_GPU_HOUR}/h)")
    print(f"wall-clock if fully parallel: ~{(MIN_PT + MIN_FT) / 60:.1f} h")
    print("\nA22 reference, same split (ID percentile): " +
          ", ".join(f"{k} {v:.3f}" for k, v in REFERENCE.items()))


def pretrain(args):
    import modal

    vol = modal.Volume.from_name(VOL)
    for p in (PROGRAMS, PATHWAYS):
        if read_json(vol, p.removeprefix("/vol/")) is None:
            raise SystemExit(f"mask-unit file missing: {p}")
    for lbl, ck in REUSED.items():
        if read_json(vol, f"{CKPT_ROOT}/{ck}/args.json") is None:
            raise SystemExit(f"reused checkpoint missing: {ck}")
    fn = modal.Function.from_name(PRETRAIN_APP, "pretrain").with_options(gpu=args.gpu)
    ids = load_ids()
    for lbl, ov in NEW.items():
        key = f"pretrain|{lbl}"
        if key in ids and not args.force:
            print(f"  {lbl:16s} already submitted")
            continue
        ids[key] = fn.spawn("jepa", False, TAG[lbl], {**PT, **ov}, BLOOD_CACHE).object_id
        print(f"  {lbl:16s} submitted -> {CKPT_ROOT}/{ckpt_of(lbl)}")
        save_ids(ids)


def finetune(args):
    import modal

    vol = modal.Volume.from_name(VOL)
    fn = modal.Function.from_name(PERTURB_APP, "train_blackwell").with_options(
        gpu=args.gpu, max_containers=args.max_containers)
    ids = load_ids()
    seeds = [int(s) for s in args.seeds.split(",")]
    wanted = [a for a in (args.arms.split(",") if args.arms else ARMS) if a in ARMS]
    pending = {a: a for a in wanted}
    t0 = time.time()
    while pending:
        for lbl in list(pending):
            if read_json(vol, f"{CKPT_ROOT}/{ckpt_of(lbl)}/args.json") is None:
                st = call_state(ids.get(f"pretrain|{lbl}"))
                if st.startswith("FAILED"):
                    print(f"  pretrain {lbl}: {st}")
                    pending.pop(lbl)
                elif not args.wait:
                    print(f"  {lbl:16s} checkpoint not ready; skipped")
                    pending.pop(lbl)
                continue
            for s in seeds:
                key = f"finetune|{lbl}|s{s}" + ("|r1" if args.retr else "")
                if key in ids and not args.force:
                    continue
                ov = {**FT_OVERRIDES, **({"retr_weight": RETR_WEIGHT} if args.retr else {})}
                ids[key] = fn.spawn(exp_id=EXP, method=FT["method"], seed=s, epochs=FT["epochs"],
                                    aux_weight=0.5, max_minutes=FT["max_minutes"],
                                    tag=ft_tag(lbl, args.retr),
                                    ckpt=f"/vol/{CKPT_ROOT}/{ckpt_of(lbl)}",
                                    input_transform=FT["input_transform"],
                                    cfg_overrides=ov).object_id
                print(f"  finetune {lbl:16s} seed {s}" + (" +retrieval-loss" if args.retr else ""))
                save_ids(ids)
            pending.pop(lbl)
        if pending:
            print(f"  [{(time.time() - t0) / 60:5.1f} min] waiting on {len(pending)}: {', '.join(sorted(pending))}")
            time.sleep(30)
    print(f"\nall fine-tunes submitted after {(time.time() - t0) / 60:.1f} min")


def status(args):
    import modal

    vol = modal.Volume.from_name(VOL)
    ids = load_ids()
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"{'arm':16s} {'knob':9s} {'pretrain':22s} fine-tune (Replogle, id_pct selection)")
    for a in ARMS:
        if a in REUSED:
            pt = "reused (A23)"
        else:
            s = read_json(vol, f"{RUNS_ROOT}/{run_dir_of(a)}/summary.json")
            pt = (f"{s['status']} recon {s['best_val_recon']:.2f}" if s
                  else call_state(ids.get(f"pretrain|{a}")))
        cells = []
        for sd in seeds:
            m = read_json(vol, f"experiments/{EXP}/runs/{FT['method']}_s{sd}_{ft_tag(a, args.retr)}/metrics.json")
            k = f"finetune|{a}|s{sd}" + ("|r1" if args.retr else "")
            cells.append(f"s{sd}:" + (f"{m['test']['delta_rmse']:.4f}" if m else str(call_state(ids.get(k)))[:9]))
        print(f"{a:16s} {KNOB[a]:9s} {pt[:22]:22s} {' '.join(cells)}")
    print(f"\nID percentile (primary) comes from: modal run diag/retrieval.py --exps {EXP} --split test")


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("plan", "pretrain", "finetune", "status"))
    p.add_argument("--seeds", default="1,2")
    p.add_argument("--gpu", default="B300")
    p.add_argument("--max-containers", type=int, default=24)
    p.add_argument("--wait", action="store_true", help="finetune: poll until each checkpoint exports")
    p.add_argument("--retr", action="store_true",
                   help="fine-tune with the InfoNCE retrieval loss on (A25's biggest lever)")
    p.add_argument("--arms", default="", help="comma list of arm labels; default = all")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    {"plan": plan, "pretrain": pretrain, "finetune": finetune, "status": status}[a.stage](a)


if __name__ == "__main__":
    main()
