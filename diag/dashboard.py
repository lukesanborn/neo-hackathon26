"""Live terminal dashboard: epoch progress of every recent run on the jepa-perturb volume.

Run in its own terminal tab from the repo root:
    .venv/bin/python diag/dashboard.py            # refreshes every 60 s (Ctrl-C to quit)
    .venv/bin/python diag/dashboard.py --once     # print once and exit
    .venv/bin/python diag/dashboard.py --hours 6  # include runs updated in the last 6 h (default 3)

Reads log.jsonl / config.json / status.json written by the training code (pretraining commits at
every epoch, Adamson fine-tunes every checkpoint, PBMC fine-tunes every epoch). Also shows how
many containers each Modal app has running.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime

import modal
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table

VOL = modal.Volume.from_name("jepa-perturb")
RUN_ROOTS = ["trackA/pretrain/runs", "experiments/pbmc_finetune_v1"]  # + experiments/*/runs


def _read(path: str) -> str | None:
    try:
        return b"".join(VOL.read_file(path)).decode()
    except Exception:
        return None


def _json(path: str):
    t = _read(path)
    try:
        return json.loads(t) if t else None
    except Exception:
        return None


def _listdir_recursive(root: str, tries: int = 4):
    for i in range(tries):
        try:
            return VOL.listdir(root, recursive=True)
        except modal.exception.ResourceExhaustedError:
            time.sleep(5 * (i + 1))
        except Exception:
            return []
    return []


def _run_dirs(max_age_s: float):
    """One recursive listing per root (avoids the volume API rate limit); a run = dir with log.jsonl."""
    now = time.time()
    for top in ("trackA/pretrain/runs", "experiments"):
        by_dir: dict = {}
        for f in _listdir_recursive(top):
            parent, name = f.path.rsplit("/", 1)
            by_dir.setdefault(parent, {})[name] = f
        for path, files in by_dir.items():
            lg = files.get("log.jsonl")
            if not lg or now - lg.mtime > max_age_s:
                continue
            root = path.rsplit("/", 1)[0]
            yield root, path, files, lg.mtime


def _summarise(root: str, path: str, files: dict, mtime: float) -> dict:
    rows = []
    for line in (_read(f"{path}/log.jsonl") or "").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    cfg = _json(f"{path}/config.json") if "config.json" in files else None
    status = (_json(f"{path}/status.json") or {}).get("status") if "status.json" in files else None
    done = any(r.get("msg") in ("done",) for r in rows) or "result.json" in files or "metrics.json" in files
    epochs_total = (cfg or {}).get("epochs") or next((r.get("epochs") for r in rows if r.get("msg") == "start"), None)
    if epochs_total is None and "pbmc_finetune" in root:
        epochs_total = 30
    ep_rows = [r for r in rows if r.get("msg") == "epoch" or ("epoch" in r and "train_gep" in r)
               or r.get("msg") == "checkpoint"]
    epoch_done = (max(r["epoch"] for r in ep_rows) + 1) if ep_rows else 0
    last = ep_rows[-1] if ep_rows else {}
    names = {"val_recon": "recon", "val_delta_rmse": "dRMSE", "val_gep": "gep"}
    metric = next((f"{names[k]} {last[k]:.3f}" for k in names if k in last), "")
    if "val_teacher_cls_std" in last:
        metric += f" sp {last['val_teacher_cls_std']:.3f}"
    kind = ("pretrain" if "pretrain" in root else "PBMC FT" if "pbmc" in root else
            "Adamson FT" if root.startswith("experiments") else root)
    age = (time.time() - mtime) / 60
    state = "done" if done else (status if status in ("failed", "timeout") else "running")
    if state == "running" and age > (45 if kind == "pretrain" else 20):
        state = "stopped"   # no update for a long time: cancelled or crashed
    return {"kind": kind, "group": root.split("/")[1] if root.startswith("experiments/") else "",
            "run": path.rsplit("/", 1)[-1], "epoch": epoch_done, "total": epochs_total,
            "state": state, "metric": metric, "age_min": age}


def _containers() -> dict:
    try:
        out = subprocess.run([".venv/bin/modal", "container", "list", "--json"], capture_output=True, text=True,
                             timeout=30).stdout
        counts: dict = {}
        for c in json.loads(out or "[]"):
            name = c.get("app_name") or c.get("app_id")
            counts[name] = counts.get(name, 0) + 1
        return counts
    except Exception as e:
        return {"(error)": repr(e)[:60]}


def render(hours: float, show_all: bool = False):
    runs = [_summarise(*x) for x in _run_dirs(hours * 3600)]
    if not show_all:  # hide finished/stopped runs older than 30 min
        runs = [r for r in runs if r["state"] == "running" or r["age_min"] <= 30]
    runs.sort(key=lambda r: (r["state"] == "done", r["kind"], r["group"], r["run"]))
    t = Table(title=f"Runs updated in the last {hours:g} h · {datetime.now().strftime('%H:%M:%S')}", pad_edge=False)
    t.add_column("run", overflow="ellipsis", max_width=34, no_wrap=True)
    for col, w in (("epoch", 5), ("progress", 10), ("state", 7), ("latest val", 18), ("upd", 4)):
        t.add_column(col, min_width=w, no_wrap=True)
    short = {"pretrain": "PT", "PBMC FT": "PBMC", "Adamson FT": "ADM"}
    for r in runs:
        tot = r["total"] or 0
        frac = min(1.0, r["epoch"] / tot) if tot else 0.0
        bar = "█" * int(frac * 10) + "·" * (10 - int(frac * 10))
        style = {"done": "green", "running": "bold", "stopped": "dim", "failed": "red", "timeout": "red"}.get(r["state"], "")
        name = f"{short.get(r['kind'], r['kind'])} {r['group'].replace('adamson_', '')+'/' if r['group'] else ''}{r['run']}"
        t.add_row(name, f"{r['epoch']}/{tot or '?'}", bar, r["state"], r["metric"],
                  f"{r['age_min']:.0f}m", style=style)
    c = Table(title="Running containers by Modal app")
    c.add_column("app")
    c.add_column("containers")
    for k, v in sorted(_containers().items()):
        c.add_row(str(k), str(v))
    note = ("upd = minutes since the run last wrote its log (pretraining writes once per ~20-min epoch). "
            "Finished runs older than 30 min are hidden (--all to show).")
    return Group(t, c, note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--every", type=float, default=60.0)
    ap.add_argument("--all", action="store_true", help="also show finished runs older than 30 min")
    a = ap.parse_args()
    con = Console()
    if a.once:
        con.print(render(a.hours, a.all))
        return
    with Live(render(a.hours, a.all), console=con, refresh_per_second=0.2, screen=False) as live:
        while True:
            time.sleep(a.every)
            live.update(render(a.hours, a.all))


if __name__ == "__main__":
    main()
