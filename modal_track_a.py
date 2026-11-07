"""Track A orchestration — Cell-JEPA reimplemented from the paper (unofficial).

Chains the separately deployed apps via `modal.Function.from_name`:

    census (jepa-census) -> pretrain pilot (jepa-pretrain; prints hours/cost, STOPS)
    -> pretrain x2 (jepa-pretrain; needs --confirm-pretrain + explicit schedule)
    -> fine-tune each kidney checkpoint with {scgpt, celljepa} x seeds {1,2} (jepa-perturb)
    -> Track A report (this app, CPU) under experiment `adamson_trackA_v1`.

Track A and Track B never share a matched comparison; Track B appears only in the report's
"for context" section. Nothing escalates automatically: every GPU stage is a separate,
explicit invocation, and pretraining refuses to start without --confirm-pretrain.

    modal run modal_track_a.py                                   # = --stage dry-run (lookups only)
    modal run modal_track_a.py --stage census --census-stage smoke
    modal run modal_track_a.py --stage census --census-stage full
    modal run modal_track_a.py --stage pilot                     # B300; prints projections; stops
    modal run modal_track_a.py --stage pretrain --confirm-pretrain \
        --pt-batch-size 32 --pt-epochs 44 --pt-lr 1e-4 --pt-max-minutes 1380
    modal run modal_track_a.py --stage collect --call-ids fc-...,fc-...
    modal run modal_track_a.py --stage finetune
    modal run modal_track_a.py --stage finetune-public
    modal run modal_track_a.py --stage report
    modal run modal_track_a.py --stage report-smoke              # synthetic layout under /vol/smoke/track_a

See docs/track_a.md for the run book and deviation log.
"""

from __future__ import annotations

import ast
import json
import math
import os
from dataclasses import dataclass, field

import modal

from environment.image import build_image

APP_NAME = "jepa-track-a"
VOL_NAME = "jepa-perturb"
VOL_PATH = "/vol"
TRACK_LABEL = "Track A — Cell-JEPA reimplemented from the paper (unofficial)"

TRACK_A_EXP = "adamson_trackA_v1"
PUBLIC_EXP = "adamson_celljepa_public_v1"   # Cell-JEPA FT recipe on public scGPT_human (separate experiment)
TRACK_B_EXP = "adamson_trackB_v1"           # context only, never compared as matched
SPLIT_SEED = 1                              # same GEARS simulation split as Track B

GPU = "B300"
MAX_FT_CONCURRENCY = 4
PT_OBJECTIVES = ("scgpt", "jepa")
FT_METHODS = ("scgpt", "celljepa")
FT_INPUT_TRANSFORM = "bins50"               # kidney models were pretrained on 50-bin values
PUBLIC_INPUT_TRANSFORM = "log1p"
PUBLIC_CKPT = "scGPT_human"
TRACK_A_CKPTS = {  # checkpoint key -> (path inside the container volume, run-tag prefix)
    "jepa_kidney": (f"{VOL_PATH}/trackA/checkpoints/jepa_kidney", "kidneyjepa"),
    "scgpt_kidney": (f"{VOL_PATH}/trackA/checkpoints/scgpt_kidney", "kidneyscgpt"),
}
CKPT_FILES = ("args.json", "best_model.pt", "vocab.json")
CENSUS_MANIFEST = "trackA/kidney_census_2023-05-15/manifest.json"   # volume-relative
PILOT_DIR = "trackA/pretrain_pilot"                                  # volume-relative

# Paper schedule (Cell-JEPA §E): ~814k kidney cells, 44 epochs, batch size 32.
PAPER_N_CELLS = 814_000
PAPER_EPOCHS = 44
PAPER_BATCH = 32
ALT_SCHEDULES = ((10, 64), (5, 128), (3, 256))   # (epochs, batch) projections printed after the pilot
# ASSUMED list price for projections only — verify the B300 rate on modal.com/pricing.
ASSUMED_USD_PER_GPU_HOUR = 6.25

# ---- other units' deployed apps / functions. Names are constants because they are built in
# ---- parallel; dry-run reports anything missing instead of crashing.
CENSUS_APP, CENSUS_FN, CENSUS_SRC = "jepa-census", "build_cache", "modal_census.py"
PRETRAIN_APP, PILOT_FN, PRETRAIN_FN, PRETRAIN_SRC = "jepa-pretrain", "pilot", "pretrain", "modal_pretrain.py"
PERTURB_APP, PREPARE_FN, TRAIN_FN, PERTURB_SRC = "jepa-perturb", "prepare", "train_blackwell", "modal_app.py"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
image = build_image("hopper")


# =============================================================================== report (CPU, this app)


def _report(exp_root: str, exp_id: str, context_exps: str, out_subdir: str, ckpt_root: str) -> dict:
    from src.track_a.report import aggregate_track_a

    vol.reload()
    exp = os.path.join(exp_root, exp_id)
    out = os.path.join(exp, out_subdir)
    ctx = [os.path.join(exp_root, e) for e in context_exps.split(",") if e and e != exp_id]
    rep = aggregate_track_a(exp, out, context_exp_dirs=ctx, ckpt_root=ckpt_root)
    vol.commit()
    with open(os.path.join(out, "results.md")) as f:
        print(f.read())
    return {"track": "A", "claim": rep["claim"], "report_dir": out, "split_hash": rep["split_hash"],
            "arms": [{"arm": r["arm"], "n_seeds": r["n_seeds"], "delta_rmse": r["delta_rmse"]} for r in rep["summary"]],
            "run_status": rep["run_status"]}


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=2, memory=8192, timeout=1800)
def report(exp_id: str = TRACK_A_EXP, context_exps: str = f"{TRACK_B_EXP},{PUBLIC_EXP}",
           out_subdir: str = "report_track_a") -> dict:
    return _report(f"{VOL_PATH}/experiments", exp_id, context_exps, out_subdir, f"{VOL_PATH}/trackA/checkpoints")


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=2, memory=8192, timeout=1800)
def report_smoke() -> dict:
    """End-to-end check of the report path on a synthetic layout (never touches real experiments)."""
    from src.track_a.report import write_synthetic_layout

    root = f"{VOL_PATH}/smoke/track_a/experiments"
    ckroot = f"{VOL_PATH}/smoke/track_a/checkpoints"
    write_synthetic_layout(os.path.join(root, TRACK_A_EXP), ckpt_root=ckroot)
    write_synthetic_layout(os.path.join(root, TRACK_B_EXP), seeds=(1,), seed=3, ckpt_root=ckroot)
    vol.commit()
    return _report(root, TRACK_A_EXP, f"{TRACK_B_EXP},{PUBLIC_EXP}", "report_track_a", ckroot)


# =============================================================================== plan


@dataclass
class Call:
    label: str
    app: str
    fn: str
    kwargs: dict
    gpu: str | None = None
    max_containers: int | None = None
    src: str | None = None          # source file in this checkout, for signature checks
    run_id: str | None = None       # run directory the call will write, if known

    def options(self) -> dict:
        o = {}
        if self.gpu:
            o["gpu"] = self.gpu
        if self.max_containers:
            o["max_containers"] = self.max_containers
        return o

    def describe(self) -> str:
        o = self.options()
        opts = f".with_options({', '.join(f'{k}={v!r}' for k, v in o.items())})" if o else ""
        args = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"Function.from_name({self.app!r}, {self.fn!r}){opts}.spawn({args})"


@dataclass
class Opts:
    census_stage: str = "smoke"
    force: bool = False
    gpu: str = GPU
    pt_batch_size: int = 0
    pt_epochs: int = 0
    pt_lr: float = 0.0
    pt_max_minutes: float = 0.0
    pt_extra: dict = field(default_factory=dict)
    pilot_extra: dict = field(default_factory=dict)
    ft_epochs: int = 15
    aux_weight: float = 0.5
    ft_max_minutes: float = 150.0
    seeds: tuple = (1, 2)
    tag: str = "v1"


def plan_census(o: Opts) -> list[Call]:
    return [Call("census", CENSUS_APP, CENSUS_FN, {"stage": o.census_stage, "force": o.force}, src=CENSUS_SRC)]


def plan_pilot(o: Opts) -> list[Call]:
    return [Call(f"pilot_{obj}", PRETRAIN_APP, PILOT_FN, {**o.pilot_extra, "objective": obj}, gpu=o.gpu,
                 src=PRETRAIN_SRC) for obj in PT_OBJECTIVES]


def pretrain_refusal(o: Opts, confirm: bool) -> str | None:
    missing = [n for n, v in (("--pt-batch-size", o.pt_batch_size), ("--pt-epochs", o.pt_epochs),
                              ("--pt-lr", o.pt_lr), ("--pt-max-minutes", o.pt_max_minutes)) if not v or v <= 0]
    if missing:
        return "pretraining needs an explicit schedule: " + ", ".join(missing) + " (decide after --stage pilot)"
    if "objective" in o.pt_extra:
        return "--pt-extra-kwargs must not set `objective` (both objectives always run with one schedule)"
    if not confirm:
        return "pretraining is expensive and never auto-launched: re-run with --confirm-pretrain"
    return None


def plan_pretrain(o: Opts) -> list[Call]:
    sched = {"batch_size": o.pt_batch_size, "epochs": o.pt_epochs, "lr": o.pt_lr, "max_minutes": o.pt_max_minutes,
             **o.pt_extra}
    # IDENTICAL schedule for both objectives — only `objective` differs.
    return [Call(f"pretrain_{obj}", PRETRAIN_APP, PRETRAIN_FN, {**sched, "objective": obj}, gpu=o.gpu,
                 src=PRETRAIN_SRC, run_id=f"/vol/trackA/checkpoints/{obj}_kidney") for obj in PT_OBJECTIVES]


def _train_calls(exp_id, ckpts: dict, methods, input_transform, o: Opts) -> list[Call]:
    calls = []
    for ck, (path, prefix) in ckpts.items():
        tag = f"{prefix}_{o.tag}" if prefix else o.tag
        for m in methods:
            for s in o.seeds:
                calls.append(Call(f"{ck}/{m}/s{s}", PERTURB_APP, TRAIN_FN,
                                  {"exp_id": exp_id, "method": m, "seed": s, "epochs": o.ft_epochs,
                                   "aux_weight": o.aux_weight, "max_minutes": o.ft_max_minutes, "tag": tag,
                                   "ckpt": path, "input_transform": input_transform},
                                  gpu=o.gpu, max_containers=MAX_FT_CONCURRENCY, src=PERTURB_SRC,
                                  run_id=f"experiments/{exp_id}/runs/{m}_s{s}_{tag}"))
    return calls


def plan_finetune(o: Opts) -> list[Call]:
    return [Call("prepare", PERTURB_APP, PREPARE_FN, {"exp_id": TRACK_A_EXP, "split_seed": SPLIT_SEED}, src=PERTURB_SRC),
            *_train_calls(TRACK_A_EXP, TRACK_A_CKPTS, FT_METHODS, FT_INPUT_TRANSFORM, o)]


def plan_finetune_public(o: Opts) -> list[Call]:
    return [Call("prepare", PERTURB_APP, PREPARE_FN, {"exp_id": PUBLIC_EXP, "split_seed": SPLIT_SEED}, src=PERTURB_SRC),
            *_train_calls(PUBLIC_EXP, {PUBLIC_CKPT: (PUBLIC_CKPT, "")}, ("celljepa",), PUBLIC_INPUT_TRANSFORM, o)]


PLANS = {"census": plan_census, "pilot": plan_pilot, "pretrain": plan_pretrain, "finetune": plan_finetune,
         "finetune-public": plan_finetune_public}


# =============================================================================== lookups / checks


def lookup(app_name: str, fn_name: str):
    """(Function | None, status string). Never raises."""
    try:
        f = modal.Function.from_name(app_name, fn_name)
        f.hydrate()
        return f, "found"
    except Exception as e:  # NotFoundError for undeployed apps / unknown functions
        return None, f"MISSING ({type(e).__name__}: {str(e)[:120]})"


def signature_check(c: Call) -> str:
    """Check our kwargs against the function's source in this checkout (deployed signatures are opaque)."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, c.src) if c.src else None
    if not path or not os.path.exists(path):
        return f"source {c.src} not in checkout — args unchecked"
    try:
        tree = ast.parse(open(path).read())
    except SyntaxError as e:
        return f"cannot parse {c.src}: {e}"
    fn = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == c.fn), None)
    if fn is None:
        return f"no def {c.fn} in {c.src}"
    a = fn.args
    params = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    n_req_pos = len(a.posonlyargs + a.args) - len(a.defaults)
    required = [x.arg for x in (a.posonlyargs + a.args)[:n_req_pos]]
    required += [x.arg for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is None]
    unknown = [k for k in c.kwargs if k not in params] if a.kwarg is None else []
    missing = [r for r in required if r not in c.kwargs]
    if unknown or missing:
        return f"ARG MISMATCH vs {c.src}: unknown={unknown} missing_required={missing}"
    return "args ok"


def methods_check(methods) -> str:
    """Fine-tuning methods must be accepted by src/train.py (TrainConfig asserts `method in METHODS`)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "train.py")
    try:
        tree = ast.parse(open(path).read())
        node = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", None) == "METHODS" for t in n.targets))
        known = ast.literal_eval(node.value)
    except Exception as e:
        return f"cannot read METHODS from src/train.py ({type(e).__name__}) — methods unchecked"
    bad = [m for m in methods if m not in known]
    return f"METHOD MISMATCH: {bad} not in src/train.py METHODS {tuple(known)}" if bad else "methods ok"


def preflight(calls: list[Call]) -> list[str]:
    """Argument / method problems detectable from this checkout (deployed code must match it)."""
    problems = []
    for c in calls:
        chk = signature_check(c)
        if chk.startswith(("ARG", "no def")) and f"{c.app}.{c.fn}: {chk}" not in problems:
            problems.append(f"{c.app}.{c.fn}: {chk}")
    methods = sorted({c.kwargs["method"] for c in calls if c.fn == TRAIN_FN})
    if methods:
        chk = methods_check(methods)
        if chk.startswith("METHOD"):
            problems.append(f"{PERTURB_APP}.{TRAIN_FN}: {chk}")
    return problems


def _vol_files(vol_rel_dir: str) -> set[str] | None:
    try:
        return {os.path.basename(e.path.rstrip("/")) for e in vol.listdir(vol_rel_dir)}
    except Exception:
        return None


def _vol_json(vol_rel_path: str):
    try:
        return json.loads(b"".join(vol.read_file(vol_rel_path)))
    except Exception:
        return None


def checkpoint_status() -> dict:
    out = {}
    for ck, (path, _) in TRACK_A_CKPTS.items():
        files = _vol_files(os.path.relpath(path, VOL_PATH))
        miss = list(CKPT_FILES) if files is None else [f for f in CKPT_FILES if f not in files]
        out[ck] = "ready" if not miss else f"missing {miss}"
    return out


# =============================================================================== projections


def _find(d, keys):
    """First numeric value for any of `keys`, searching nested dicts breadth-first."""
    q = [d]
    while q:
        x = q.pop(0)
        if isinstance(x, dict):
            for k in keys:
                if isinstance(x.get(k), (int, float)) and not isinstance(x.get(k), bool):
                    return float(x[k])
            q += [v for v in x.values() if isinstance(v, (dict, list))]
        elif isinstance(x, list):
            q += [v for v in x if isinstance(v, (dict, list))]
    return None


def projection(rep: dict, n_cells: float, usd_per_hour: float) -> dict:
    sps = _find(rep, ("seconds_per_step", "sec_per_step", "s_per_step"))
    bs = _find(rep, ("batch_size",))
    cps = _find(rep, ("cells_per_second", "cells_per_sec"))
    if cps is None and sps and bs:
        cps = bs / sps
    rows = []
    for name, ep, b in [("paper", PAPER_EPOCHS, PAPER_BATCH)] + [("alt", e, b) for e, b in ALT_SCHEDULES]:
        steps = math.ceil(n_cells * ep / b)
        hours = n_cells * ep / cps / 3600 if cps else None
        rows.append({"schedule": f"{name}: {ep} ep x bs {b}", "steps": steps,
                     "hours_per_model": None if hours is None else round(hours, 1),
                     "usd_both_models": None if hours is None else round(2 * hours * usd_per_hour)})
    return {"seconds_per_step": sps, "pilot_batch_size": bs, "cells_per_second": cps, "n_cells": n_cells,
            "assumption": "cells/s measured at the pilot batch size is held constant across batch sizes; "
                          f"cost uses an ASSUMED ${usd_per_hour}/GPU-h — verify on modal.com/pricing",
            "schedules": rows,
            "pilot_reported_schedules": {k: rep[k] for k in ("schedules", "projections", "alternatives") if k in rep}}


# =============================================================================== launcher helpers


def _show(x):
    print(json.dumps(x, indent=2, default=str))


_HANDLES: dict = {}


def _resolve(c: Call):
    """One handle per (app, fn, options): every with_options() call creates its own container pool,
    so reusing the handle is what makes max_containers a shared cap across the spawned runs."""
    key = (c.app, c.fn, tuple(sorted(c.options().items())))
    if key not in _HANDLES:
        f, st = lookup(c.app, c.fn)
        if f is None:
            raise SystemExit(f"[{c.label}] cannot launch {c.app}.{c.fn}: {st} — deploy it first (docs/track_a.md)")
        _HANDLES[key] = f.with_options(**c.options()) if c.options() else f
    return _HANDLES[key]


def _spawn(calls: list[Call]):
    out = []
    for c in calls:
        fc = _resolve(c).spawn(**c.kwargs)
        print(f"  launched {c.label:<26} call_id={fc.object_id}  run={c.run_id or '-'}")
        out.append((c, fc))
    print("collect later with: modal run modal_track_a.py --stage collect --call-ids "
          + ",".join(fc.object_id for _, fc in out))
    return out


def _wait(launched, summarise=lambda r: r) -> int:
    failed = 0
    for c, fc in launched:
        try:
            _show({c.label: summarise(fc.get())})
        except Exception as e:  # report, never swallow
            failed += 1
            _show({c.label: {"status": "failed", "call_id": fc.object_id, "error": repr(e)[:2000]}})
    print(f"{len(launched) - failed}/{len(launched)} calls succeeded")
    return failed


def _train_summary(r):
    keys = ("status", "epochs_completed", "selected_epoch", "elapsed_seconds", "peak_gpu_mem_gb", "reload_ok")
    return {k: r.get(k) for k in keys} | {"test_delta_rmse": (r.get("test") or {}).get("delta_rmse")}


def _dry_run(o: Opts, confirm: bool):
    print(f"== {TRACK_LABEL} — dry run (lookups only; nothing executes) ==\n")
    plans = {s: f(o) for s, f in PLANS.items()}
    fns = sorted({(c.app, c.fn) for cs in plans.values() for c in cs})
    status = {k: lookup(*k)[1] for k in fns}
    print(f"{'app':<16} {'function':<18} status")
    for (a, f), st in status.items():
        print(f"{a:<16} {f:<18} {st}")
    print(f"{APP_NAME:<16} {'report':<18} local (this app)")
    print(f"{APP_NAME:<16} {'report_smoke':<18} local (this app)\n")

    print("volume state:")
    print(f"  census manifest {CENSUS_MANIFEST}: {'present' if _vol_json(CENSUS_MANIFEST) else 'absent'}")
    pil = _vol_files(PILOT_DIR)
    print(f"  pretrain pilot reports {PILOT_DIR}: {sorted(pil) if pil else 'none'}")
    for ck, st in checkpoint_status().items():
        print(f"  checkpoint {ck}: {st}")
    print()

    problems = [f"{a}.{f}: not deployed / not found" for (a, f), st in status.items() if st != "found"]
    for stage, calls in plans.items():
        print(f"--stage {stage}:")
        if stage == "pretrain":
            why = pretrain_refusal(o, confirm)
            if why:
                print(f"  (would REFUSE: {why})")
        problems += [p for p in preflight(calls) if p not in problems]
        for c in calls:
            chk = signature_check(c)
            print(f"  {c.label:<24} {c.describe()}\n  {'':<24} [{status[(c.app, c.fn)].split(' (')[0]}; {chk}]")
        print()
    print("--stage report:\n  " + f"report.remote(exp_id={TRACK_A_EXP!r}) -> /vol/experiments/{TRACK_A_EXP}/report_track_a/\n")
    if problems:
        print("NOT READY (" + str(len(problems)) + "):\n  " + "\n  ".join(problems))
    else:
        print("all lookups and argument checks passed")
    return {"missing_or_mismatched": problems}


# =============================================================================== entrypoint


@app.local_entrypoint()
def main(stage: str = "dry-run", census_stage: str = "smoke", force: bool = False, gpu: str = GPU,
         confirm_pretrain: bool = False, pt_batch_size: int = 0, pt_epochs: int = 0, pt_lr: float = 0.0,
         pt_max_minutes: float = 0.0, pt_extra_kwargs: str = "{}", pilot_kwargs: str = "{}",
         ft_epochs: int = 15, aux_weight: float = 0.5, ft_max_minutes: float = 150.0, seeds: str = "1,2",
         tag: str = "v1", wait: str = "auto", call_ids: str = "", collect_timeout: float = 0.0,
         n_cells: int = 0, usd_per_gpu_hour: float = ASSUMED_USD_PER_GPU_HOUR,
         skip_ckpt_check: bool = False, skip_arg_check: bool = False):
    o = Opts(census_stage=census_stage, force=force, gpu=gpu, pt_batch_size=pt_batch_size, pt_epochs=pt_epochs,
             pt_lr=pt_lr, pt_max_minutes=pt_max_minutes, pt_extra=json.loads(pt_extra_kwargs),
             pilot_extra=json.loads(pilot_kwargs), ft_epochs=ft_epochs, aux_weight=aux_weight,
             ft_max_minutes=ft_max_minutes, seeds=tuple(int(s) for s in seeds.split(",") if s), tag=tag)
    def gate(calls):
        bad = preflight(calls)
        if bad and not skip_arg_check:
            raise SystemExit("REFUSED (argument/method mismatch vs this checkout; deploy the apps from a checkout "
                             "that matches, or --skip-arg-check):\n  " + "\n  ".join(bad))
        return calls

    should_wait = {"yes": True, "no": False}.get(wait, stage != "pretrain")
    print(f"[{TRACK_LABEL}] stage={stage}")

    if stage == "dry-run":
        _dry_run(o, confirm_pretrain)
        return

    if stage == "census":
        (c, fc), = _spawn(gate(plan_census(o)))
        if should_wait and _wait([(c, fc)]):
            raise SystemExit(1)
        return

    if stage == "pilot":
        launched = _spawn(gate(plan_pilot(o)))
        failed = 0
        man = _vol_json(CENSUS_MANIFEST) or {}
        for c, fc in launched:
            try:
                rep = fc.get()
            except Exception as e:
                failed += 1
                _show({c.label: {"status": "failed", "error": repr(e)[:2000]}})
                continue
            n = n_cells or _find(rep, ("n_train_cells", "n_cells")) or _find(man, ("n_train_cells", "n_cells")) \
                or PAPER_N_CELLS
            _show({c.label: {"pilot_report": rep, "projection": projection(rep, n, usd_per_gpu_hour)}})
        print("\nSTOP: choose ONE schedule for both objectives, record it in docs/track_a.md, then run "
              "--stage pretrain --confirm-pretrain --pt-batch-size B --pt-epochs E --pt-lr LR --pt-max-minutes M")
        if failed:
            raise SystemExit(1)
        return

    if stage == "pretrain":
        why = pretrain_refusal(o, confirm_pretrain)
        if why:
            raise SystemExit(f"REFUSED: {why}")
        print("launching both objectives concurrently with an IDENTICAL schedule "
              f"(bs={pt_batch_size}, epochs={pt_epochs}, lr={pt_lr}, max_minutes={pt_max_minutes}, extra={o.pt_extra})")
        launched = _spawn(gate(plan_pretrain(o)))
        if should_wait and _wait(launched):
            raise SystemExit(1)
        return

    if stage == "collect":
        ids = [i for i in call_ids.split(",") if i]
        if not ids:
            raise SystemExit("--call-ids required")
        for i in ids:
            try:
                _show({i: modal.FunctionCall.from_id(i).get(timeout=collect_timeout)})
            except Exception as e:
                state = "still running" if "Timeout" in type(e).__name__ else "failed"
                _show({i: {"status": state, "error": repr(e)[:2000]}})
        return

    if stage in ("finetune", "finetune-public"):
        calls = gate(PLANS[stage](o))
        if stage == "finetune" and not skip_ckpt_check:
            st = checkpoint_status()
            if any(v != "ready" for v in st.values()):
                _show(st)
                raise SystemExit("kidney checkpoints not ready — finish pretraining first (or --skip-ckpt-check)")
        prep, runs = calls[0], calls[1:]
        print(f"{prep.label}: {prep.describe()}")
        gate = _resolve(prep).remote(**prep.kwargs)
        _show({"prepare": {k: gate.get(k) for k in ("split_hash", "split_sizes", "baselines")}})
        launched = _spawn(runs)
        if should_wait and _wait(launched, _train_summary):
            raise SystemExit(1)
        return

    if stage == "report":
        _show(report.remote())
        return

    if stage == "report-smoke":
        _show(report_smoke.remote())
        return

    raise SystemExit(f"unknown --stage {stage!r}; one of dry-run, census, pilot, pretrain, collect, finetune, "
                     "finetune-public, report, report-smoke")
