"""Pretraining-level JEPA knob sweep, scored on the 22 held-out Adamson genes.

Each arm changes ONE knob of the JEPA pretraining objective, continues pretraining from public
scGPT_human on the kidney cache, then fine-tunes on Adamson with the Cell-JEPA recipe and is
scored on the same 22 held-out genes as every earlier run. Everything else is identical across
arms (same init, seed, schedule, data order), so a difference is attributable to the knob.

Knobs (screenshot "Stage 2: Views"):
  2a  what unit gets masked   random | program | pathway | top_expressed
  2b  how much gets masked    mask_ratio 0.15 | 0.30 | 0.50 | 0.80
  2c  what the target is      full cell (center / faithful) | only the masked part (token)
      "a different cell (perturbation pair)" has no meaning during pretraining (kidney cells carry
      no perturbations) — it IS the fine-tuning JEPA term, which every arm then gets on Adamson.

Why `center` is the default variant, not the paper's `faithful`: attempts.md A19 measured that
continuing from public scGPT makes `faithful` collapse (teacher spread 0.009 -> 0.003; its JEPA
loss falls BELOW what a constant guess would score, so the term teaches nothing). `center` keeps
real per-cell signal there. One `faithful` arm is kept as a control that should visibly collapse.

Scoring (all on the 22 held-out genes, none of which any arm trained on):
  * delta RMSE            the repo's primary error metric. Read against linear 0.0575 (A14) —
                          no deep arm has beaten that yet.
  * identification        A16/A17: given a real response, is the right gene ranked first among the
                          22? This is the "where do I point CRISPR" direction. Chance top-1 = 0.045;
                          linear 0.091; best deep arm so far 0.227.
  * teacher spread        collapse diagnostic from pretraining (~0.00002 = collapsed, 0.03-0.04 healthy).

Stages (run from the repo root; `source .venv/bin/activate` or use .venv/bin/python):
    python diag/launch_model_knobs.py plan                  # arms, timing, cost. No Modal calls.
    python diag/launch_model_knobs.py setup                 # CPU: Adamson experiment + mask-unit files
    python diag/launch_model_knobs.py run                   # pretrain all arms, then auto-chain fine-tunes
    python diag/launch_model_knobs.py status                # poll; safe to run any time
    python diag/launch_model_knobs.py report                # final comparison table -> diag/results/

`run` chains fine-tuning as each pretrain finishes, so losing the terminal is recoverable: call ids
live in diag/call_ids_model_knobs.json, and `finetune` / `status` / `report` read results back from
the Volume. Re-running a stage skips work already submitted unless --force.

Deploy first (from this branch), using a suffix so a shared workspace keeps its existing apps:
    JEPA_APP_SUFFIX=-rhea .venv/bin/modal deploy modal_pretrain.py
    JEPA_APP_SUFFIX=-rhea .venv/bin/modal deploy modal_app.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

SUFFIX = os.environ.get("JEPA_APP_SUFFIX", "")
PRETRAIN_APP, PERTURB_APP = f"jepa-pretrain{SUFFIX}", f"jepa-perturb{SUFFIX}"
VOL = "jepa-perturb"

EXP = "adamson_knobs_v1"                  # own experiment id: never writes into another sweep's runs
# Second benchmark (A20): 103 held-out genes vs Adamson's 22, so ~5x the power to separate arms that
# differ by little. We reuse the data unit's prepared experiment; our tags keep the run dirs distinct.
DATASETS = {"adamson": EXP, "replogle": "replogle_k562_v1"}
DATASET = os.environ.get("KNOB_DATASET", "adamson")   # set by --dataset in main()
CKPT_ROOT = "trackA/checkpoints"          # volume-relative
RUNS_ROOT = "trackA/pretrain/runs"
INIT_CKPT = "scGPT_human"                 # A16: the strongest Adamson starting point
UNITS = {"program": "/vol/trackA/mask_units/program_k64.json",
         "pathway": "/vol/trackA/mask_units/pathway_MSigDB_Hallmark_2020.json"}

# Pretraining schedule shared by every arm. lr 3e-5 is the A8/A9 setting that trains without
# collapse; only the amount of data differs between tiers.
#   screen  3 epochs x 48k cells  = 4.5k steps (~4 min). A3-scale: catches bugs and collapse, but
#           A3-vs-A9 showed short-run rankings can FLIP versus the full schedule.
#   long    2 epochs x all ~814k  = ~51k steps (~42 min). Half the paper's 4 epochs; every cell is
#           seen twice instead of 18% of them being seen once.
PT_BASE = {"batch_size": 32, "lr": 3e-5, "seed": 0, "ema_momentum": 0.996, "init_ckpt": INIT_CKPT,
           "max_val_cells": 1024, "log_every": 200}
TIERS = {
    "screen": {**PT_BASE, "epochs": 3, "max_cells_per_epoch": 48000, "max_minutes": 40.0,
               "reserve_minutes": 3.0},
    # max_minutes is a SAFETY VALVE, not the schedule: a run that hits it stops early and exports a
    # truncated model, which silently invalidates the comparison. Keep it well above the real
    # projection (~21 min/epoch over the full corpus at the measured 0.044 s/step).
    "long": {**PT_BASE, "epochs": 4, "max_cells_per_epoch": None, "max_minutes": 150.0,
             "reserve_minutes": 5.0},
}
TIER_TAG = {"screen": "mk", "long": "mkL"}       # distinct tags: tiers never overwrite each other
# Wall-clock minutes per arm. screen: measured (6 min + margin). long: 0.044 s/step measured here,
# ~25k steps/epoch over the full corpus; cross-checks against A9's 73-88 min for 4 epochs.
MIN_PT = {"screen": 5.0, "long": 25_000 * 0.044 / 60}   # ~800k train cells / batch 32 = ~25k steps/epoch
FT = {"epochs": 15, "method": "celljepa", "input_transform": "bins50", "max_minutes": 60.0}
TIER = os.environ.get("KNOB_TIER", "screen")     # set by --tier in main()

# Measured on B300 (attempts.md A2/A9/A14/A18); used only for the plan estimate.
MIN_FINETUNE = 40
ASSUMED_USD_PER_GPU_HOUR = 6.25           # verify on modal.com/pricing
BASELINES = {"linear": {"delta_rmse": 0.0575, "id_top1": 0.091},
             "training mean": {"delta_rmse": 0.0588, "id_top1": 0.045},
             "no change": {"delta_rmse": 0.0915, "id_top1": 0.0}}
POLL_SECONDS = 30


@dataclass(frozen=True)
class Arm:
    knob: str
    label: str
    objective: str = "jepa"      # jepa | scgpt
    unit: str = "random"         # random | program | pathway | top_expressed
    ratio: float = 0.15
    variant: str = "center"      # center | faithful | token  (jepa only)
    note: str = ""

    @property
    def tag(self) -> str:
        return f"{TIER_TAG[TIER]}_{self.label}"

    @property
    def ckpt(self) -> str:
        return f"{self.objective}_kidney_{self.tag}"

    @property
    def run_dir(self) -> str:
        return f"{self.objective}_{self.tag}"

    def overrides(self) -> dict:
        ov = {**TIERS[TIER], "mask_unit": self.unit, "mask_ratio": self.ratio}
        if self.unit in UNITS:
            ov["mask_units_path"] = UNITS[self.unit]
        if self.objective == "jepa":
            ov["jepa_variant"] = self.variant
        return ov


ARMS = [
    # reference points
    Arm("ref", "base_center", note="JEPA baseline: random 15%, full-cell centred target (A19's working setting)"),
    Arm("ref", "no_jepa", objective="scgpt", variant="-", note="no JEPA term at all: matched reconstruction-only control"),
    # 2a — what unit gets masked
    Arm("2a", "unit_program", unit="program", note="whole co-expression programs"),
    Arm("2a", "unit_pathway", unit="pathway", note="whole curated pathways (MSigDB Hallmark)"),
    Arm("2a", "unit_topexpr", unit="top_expressed", note="the cell's highest-expressed genes"),
    # 2b — how much gets masked
    Arm("2b", "ratio_30", ratio=0.30),
    Arm("2b", "ratio_50", ratio=0.50),
    Arm("2b", "ratio_80", ratio=0.80, note="expected to be too hard; included to find the ceiling"),
    # 2c — what the target is
    Arm("2c", "target_token", variant="token", note="predict only the masked genes' embeddings (I-JEPA style)"),
    Arm("2c", "target_faithful", variant="faithful", note="paper's full-cell target; A19 predicts collapse here"),
    Arm("2c", "target_token_program", unit="program", variant="token",
        note="combines the two most promising: program masking + masked-part target"),
]
KNOB_TITLES = {"ref": "Reference arms", "2a": "2a — what unit gets masked",
               "2b": "2b — how much gets masked", "2c": "2c — what the target is"}


def arms_for(knobs: list[str]) -> list[Arm]:
    return [a for a in ARMS if a.knob in knobs or a.knob == "ref"]


# ----------------------------------------------------------------------------- bookkeeping


def ids_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "call_ids_model_knobs.json")


def load_ids() -> dict:
    return json.load(open(ids_path())) if os.path.exists(ids_path()) else {}


def save_ids(ids: dict) -> None:
    """Merge into the file rather than overwrite it.

    Several chain processes (one per dataset/tier) write here concurrently, each holding its own
    in-memory copy. A plain overwrite drops the keys another process added since this one loaded,
    which loses the ability to query those jobs later. Results are read from the Volume, so a lost
    key never loses a result, but merging keeps the record complete."""
    merged = load_ids()
    merged.update(ids)
    ids.update(merged)
    tmp = ids_path() + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=1)
    os.replace(tmp, ids_path())


def read_json(vol, path: str):
    """Read a JSON file from the Volume; None if absent (so `status` works from any machine).

    Note: the local client always reads the Volume's current committed state. Volume.reload()
    exists only for the in-container mount and raises here, so it must not be called."""
    try:
        return json.loads(b"".join(vol.read_file(path)))
    except Exception:
        return None


def call_state(cid: str | None) -> str:
    import modal

    if not cid:
        return "-"
    try:
        modal.FunctionCall.from_id(cid).get(timeout=0)
        return "done"
    except TimeoutError:
        return "running"
    except Exception as e:
        return f"FAILED {e!r:.60}"


def fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


# ----------------------------------------------------------------------------- stages


def plan(args, arms):
    n = len(arms)
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"{n} arms x {len(seeds)} fine-tune seed(s); knobs {','.join(args.knobs)}\n")
    for knob in ["ref"] + [k for k in args.knobs if k != "ref"]:
        group = [a for a in arms if a.knob == knob]
        if not group:
            continue
        print(KNOB_TITLES[knob])
        for a in group:
            spec = (f"{a.objective}/{a.unit}/r{a.ratio:.2f}" + (f"/{a.variant}" if a.objective == "jepa" else ""))
            print(f"   {a.label:22s} {spec:34s} {a.note}")
        print()
    min_pt = MIN_PT[TIER] * TIERS[TIER]["epochs"]
    gpu_h = (n * min_pt + n * len(seeds) * MIN_FINETUNE) / 60
    print(f"GPU jobs: {n} pretrain + {n * len(seeds)} Adamson fine-tune = {n + n * len(seeds)}")
    print(f"Billed ~{gpu_h:.1f} GPU-hours ~= ${gpu_h * ASSUMED_USD_PER_GPU_HOUR:.0f} "
          f"(ASSUMED ${ASSUMED_USD_PER_GPU_HOUR}/h — verify pricing)")
    print(f"Wall-clock if all run concurrently: ~{min_pt + MIN_FINETUNE:.0f} min "
          f"({min_pt:.0f} pretrain, then {MIN_FINETUNE} fine-tune).")
    print(f"With a concurrency limit of C containers, add a wave per {n}/C.")


def setup(args, arms):
    """CPU prep: bind the Adamson experiment (writes baselines) and build the mask-unit files."""
    import modal

    prepare = modal.Function.from_name(PERTURB_APP, "prepare")
    build_units = modal.Function.from_name(PRETRAIN_APP, "build_mask_units")
    calls = {"adamson_experiment": prepare.spawn(EXP)}
    for kind in sorted({a.unit for a in arms if a.unit in UNITS}):
        calls[f"mask_units_{kind}"] = build_units.spawn(kind, force=args.force)
    for name, c in calls.items():
        try:
            r = c.get()
            if name == "adamson_experiment":
                print(f"{name}: split {r['split_hash']} | {r['split_sizes']} | "
                      f"linear delta RMSE {r['baselines']['linear']['test_delta_rmse']:.4f}")
            else:
                print(f"{name}: {json.dumps(r)}")
        except Exception as e:
            print(f"{name}: FAILED {e!r:.300}")


def submit_pretrain(arms, ids, gpu, force):
    import modal

    fn = modal.Function.from_name(PRETRAIN_APP, "pretrain").with_options(gpu=gpu)
    for a in arms:
        key = f"pretrain|{a.tag}"
        if key in ids and not force:
            continue
        ids[key] = fn.spawn(a.objective, False, a.tag, a.overrides()).object_id
        print(f"  pretrain  {a.label:22s} -> {CKPT_ROOT}/{a.ckpt}")
        save_ids(ids)


def ft_key(arm, seed) -> str:
    return f"finetune|{DATASET}|{arm.tag}|s{seed}"


def ft_metrics_path(arm, seed) -> str:
    return f"experiments/{DATASETS[DATASET]}/runs/{FT['method']}_s{seed}_{arm.tag}/metrics.json"


def submit_finetune(arm, ids, seeds, gpu, force) -> bool:
    """Spawn fine-tunes for one arm on the active dataset. True if anything was submitted."""
    import modal

    fn = modal.Function.from_name(PERTURB_APP, "train_blackwell").with_options(
        gpu=gpu, max_containers=args_max_containers)
    out = False
    for s in seeds:
        key = ft_key(arm, s)
        if key in ids and not force:
            continue
        ids[key] = fn.spawn(exp_id=DATASETS[DATASET], method=FT["method"], seed=s, epochs=FT["epochs"],
                            aux_weight=0.5, max_minutes=FT["max_minutes"], tag=arm.tag,
                            ckpt=f"/vol/{CKPT_ROOT}/{arm.ckpt}",
                            input_transform=FT["input_transform"]).object_id
        print(f"  finetune  {DATASET}/{arm.label:22s} seed {s}")
        out = True
        save_ids(ids)
    return out


args_max_containers = 24   # set from --max-containers in main()


def run(args, arms):
    """Spawn every pretrain, then chain each arm's fine-tunes as soon as its checkpoint exports."""
    import modal

    vol = modal.Volume.from_name(VOL)
    ids = load_ids()
    missing = [p for p in sorted({UNITS[a.unit] for a in arms if a.unit in UNITS})
               if read_json(vol, p.removeprefix("/vol/")) is None]
    if missing:
        raise SystemExit(f"mask-unit files missing {missing}: run `setup` first")
    if read_json(vol, f"experiments/{EXP}/experiment.json") is None:
        raise SystemExit(f"experiment {EXP} not prepared: run `setup` first")

    print(f"submitting {len(arms)} pretraining jobs on {args.gpu}")
    submit_pretrain(arms, ids, args.gpu, args.force)
    if args.no_chain:
        print("\n--no-chain: stopping after pretrain. Run `finetune` once checkpoints exist.")
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    pending = {a.label: a for a in arms}
    t0 = time.time()
    print(f"\nchaining fine-tunes as checkpoints appear (poll every {POLL_SECONDS}s; Ctrl-C is safe —"
          f" `status` and `finetune` resume)\n")
    while pending:
        for label, a in list(pending.items()):
            if read_json(vol, f"{CKPT_ROOT}/{a.ckpt}/args.json") is not None:
                submit_finetune(a, ids, seeds, args.gpu, args.force)
                pending.pop(label)
                continue
            st = call_state(ids.get(f"pretrain|{a.tag}"))
            if st.startswith("FAILED"):
                print(f"  pretrain  {label:22s} {st}")
                pending.pop(label)
        if pending:
            print(f"  [{(time.time() - t0) / 60:5.1f} min] waiting on {len(pending)}: "
                  f"{', '.join(sorted(pending))}")
            time.sleep(POLL_SECONDS)
    print(f"\nall fine-tunes submitted after {(time.time() - t0) / 60:.1f} min. "
          f"Track with: python diag/launch_model_knobs.py status")


def finetune(args, arms):
    """Submit fine-tunes for every arm whose checkpoint exists.

    With --wait, keep polling and submit each arm as its checkpoint appears, so this can be
    started before pretraining finishes. Independent of the chain inside `run`: a different
    dataset uses different call-id keys and a different experiment, so both can run at once."""
    import modal

    vol = modal.Volume.from_name(VOL)
    ids = load_ids()
    seeds = [int(s) for s in args.seeds.split(",")]
    pending = {a.label: a for a in arms}
    t0 = time.time()
    while pending:
        for label, a in list(pending.items()):
            if read_json(vol, f"{CKPT_ROOT}/{a.ckpt}/args.json") is not None:
                if not submit_finetune(a, ids, seeds, args.gpu, args.force):
                    print(f"  {DATASET}/{label:22s} already submitted")
                pending.pop(label)
                continue
            st = call_state(ids.get(f"pretrain|{a.tag}"))
            if st.startswith("FAILED"):
                print(f"  pretrain  {label:22s} {st}")
                pending.pop(label)
            elif not args.wait:
                print(f"  {DATASET}/{label:22s} checkpoint not ready; skipped")
                pending.pop(label)
        if pending:
            print(f"  [{(time.time() - t0) / 60:5.1f} min] waiting on {len(pending)}: "
                  f"{', '.join(sorted(pending))}")
            time.sleep(POLL_SECONDS)
    print(f"\nall {DATASET} fine-tunes submitted after {(time.time() - t0) / 60:.1f} min.")


def collect(arms, seeds):
    """Read every arm's pretraining summary and Adamson metrics back from the Volume."""
    import modal

    vol = modal.Volume.from_name(VOL)
    ids = load_ids()
    rows = {}
    for a in arms:
        summ = read_json(vol, f"{RUNS_ROOT}/{a.run_dir}/summary.json")
        hist = (summ or {}).get("history") or [{}]
        r = {"arm": a, "pretrain": (summ or {}).get("status") or call_state(ids.get(f"pretrain|{a.tag}")),
             "val_recon": (summ or {}).get("best_val_recon"),
             "mask_frac": hist[-1].get("val_mask_frac"),
             "teacher_spread": hist[-1].get("val_teacher_cls_std"),
             "jepa_loss": hist[-1].get("val_jepa"), "seeds": {}}
        for s in seeds:
            m = read_json(vol, ft_metrics_path(a, s))
            r["seeds"][s] = ({"delta_rmse": m["test"]["delta_rmse"],
                              "delta_pearson": m["test"]["delta_pearson"],
                              "de20_delta_pearson": m["test"].get("de20_delta_pearson"),
                              # validation (7 genes) — the honest selection metric; test is scored once
                              "val_delta_rmse": m.get("val_delta_rmse_selected"),
                              "status": m.get("status")} if m
                             else {"status": call_state(ids.get(ft_key(a, s)))})
        for key, out in (("delta_rmse", "delta_rmse"), ("val_delta_rmse", "val_delta_rmse"),
                         ("delta_pearson", "delta_pearson")):
            vals = [v[key] for v in r["seeds"].values() if v.get(key) is not None]
            r[out] = sum(vals) / len(vals) if vals else None
        r["n_seeds_done"] = sum("delta_rmse" in v for v in r["seeds"].values())
        rows[a.label] = r
    return rows


def status(args, arms):
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = collect(arms, seeds)
    n_pt = sum(r["pretrain"] in ("completed", "stopped_by_time_budget") for r in rows.values())
    n_ft = sum(r["n_seeds_done"] for r in rows.values())
    print(f"pretrain {n_pt}/{len(arms)} done | fine-tune {n_ft}/{len(arms) * len(seeds)} done\n")
    print(f"{'arm':24s} {'pretrain':12s} {'mask':>6s} {'recon':>7s} {'spread':>8s}  fine-tune")
    for a in arms:
        r = rows[a.label]
        ft = " ".join(f"s{s}:" + (fmt(v.get('delta_rmse'), 4) if 'delta_rmse' in v else str(v['status'])[:10])
                      for s, v in r["seeds"].items())
        print(f"{a.label:24s} {str(r['pretrain'])[:12]:12s} {fmt(r['mask_frac'], 2):>6s} "
              f"{fmt(r['val_recon'], 2):>7s} {fmt(r['teacher_spread'], 4):>8s}  {ft}")


def report(args, arms):
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = collect(arms, seeds)
    base = rows.get("base_center", {}).get("delta_rmse")
    pt = TIERS[TIER]
    cells = f"{pt['max_cells_per_epoch']:,} cells" if pt["max_cells_per_epoch"] else "the full corpus"
    L = [f"# Pretraining JEPA knobs ({TIER} tier) — {DATASET} held-out genes", "",
         f"Every arm: continued pretraining from public `{INIT_CKPT}` on the kidney cache "
         f"({pt['epochs']} epochs x {cells}, lr {pt['lr']}, bs {pt['batch_size']}, "
         f"seed {pt['seed']}), then Cell-JEPA fine-tuning on Adamson ({FT['epochs']} epochs, "
         f"seeds {','.join(map(str, seeds))}). Only the knob differs.", "",
         "`spread` is the teacher's `<cls>` spread at the end of pretraining — the collapse diagnostic. "
         "A19: ~0.003 means the JEPA term has nothing cell-specific left to teach; 0.03-0.04 is healthy.",
         "`mask` is the realised fraction of gene tokens masked (unit masks overshoot the target ratio).", ""]
    hdr = ("| arm | what changed | mask | teacher spread | val ΔRMSE (7 genes) | **test ΔRMSE (22 genes)** "
           "| vs baseline | test Δ-Pearson |")
    for knob in ["ref"] + [k for k in args.knobs if k != "ref"]:
        group = [a for a in arms if a.knob == knob]
        if not group:
            continue
        L += [f"## {KNOB_TITLES[knob]}", "", hdr, "|" + "---|" * 8]
        for a in group:
            r = rows[a.label]
            d = r["delta_rmse"]
            delta = "-" if (d is None or base is None or a.label == "base_center") else f"{d - base:+.4f}"
            n = f" ({r['n_seeds_done']}/{len(seeds)} seeds)" if 0 < r["n_seeds_done"] < len(seeds) else ""
            L.append(f"| `{a.label}` | {a.note or a.unit + ' ' + str(a.ratio)} | {fmt(r['mask_frac'], 2)} "
                     f"| {fmt(r['teacher_spread'], 4)} | {fmt(r['val_delta_rmse'], 4)} "
                     f"| **{fmt(d, 4)}**{n} | {delta} | {fmt(r['delta_pearson'])} |")
        L.append("")
    L += ["## Reference points (not run here)", "", "| predictor | delta RMSE | identification top-1 |", "|---|---|---|"]
    for k, v in BASELINES.items():
        L.append(f"| {k} | {v['delta_rmse']:.4f} | {v['id_top1']:.3f} |")
    L += ["",
          "Lower delta RMSE is better. No deep arm has yet beaten the linear baseline (0.0575) on this metric; "
          "`docs/clustering_finding.md` argues delta RMSE is not what the application needs, so treat "
          "identification as the metric that matches the CRISPR-targeting goal.", "",
          "Identification scores for these arms: run", "",
          f"    .venv/bin/modal run diag/retrieval.py --exps {EXP} --tag model_knobs", "",
          "which reads each run's saved predictions and reports top-1 / mean rank with bootstrap CIs (A17)."]
    md = "\n".join(L)
    print(md)
    os.makedirs("diag/results", exist_ok=True)
    with open(f"diag/results/model_knobs_{DATASET}_{TIER}.md", "w") as f:
        f.write(md + "\n")
    with open(f"diag/results/model_knobs_{DATASET}_{TIER}.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "arm"} for k, v in rows.items()},
                  f, indent=1, default=str)
    print(f"\nwrote diag/results/model_knobs_{DATASET}_{TIER}.md and .json")


def main(argv=None):
    global args_max_containers, TIER, DATASET

    sys.stdout.reconfigure(line_buffering=True)   # progress must be visible when piped to a log
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("plan", "setup", "run", "finetune", "status", "report"))
    p.add_argument("--dataset", choices=tuple(DATASETS), default=DATASET,
                   help="adamson = 22 held-out genes; replogle = 103 (A20), ~5x the power")
    p.add_argument("--tier", choices=tuple(TIERS), default=TIER,
                   help="screen = 3 x 48k cells (~6 min); long = full 814k corpus (--epochs sets how many)")
    p.add_argument("--epochs", type=int, default=0, help="override the tier's pretraining epochs")
    p.add_argument("--knobs", default="2a,2b,2c", help="comma list of 2a,2b,2c (reference arms always included)")
    p.add_argument("--seeds", default="1", help="Adamson fine-tuning seeds (default 1 for speed; '1,2' is safer)")
    p.add_argument("--gpu", default="B300")
    p.add_argument("--max-containers", type=int, default=24)
    p.add_argument("--no-chain", action="store_true", help="run: submit pretraining only, do not wait")
    p.add_argument("--wait", action="store_true",
                   help="finetune: poll until each checkpoint exports instead of skipping it")
    p.add_argument("--force", action="store_true", help="resubmit jobs already recorded in the call-id file")
    a = p.parse_args(argv)
    a.knobs = [k for k in a.knobs.split(",") if k]
    bad = [k for k in a.knobs if k not in KNOB_TITLES]
    if bad:
        p.error(f"unknown knobs {bad}; choose from 2a, 2b, 2c")
    args_max_containers = a.max_containers
    TIER, DATASET = a.tier, a.dataset
    if a.epochs:
        TIERS[TIER] = {**TIERS[TIER], "epochs": a.epochs}
    arms = arms_for(a.knobs)
    {"plan": plan, "setup": setup, "run": run, "finetune": finetune, "status": status, "report": report}[a.stage](a, arms)


if __name__ == "__main__":
    main()
