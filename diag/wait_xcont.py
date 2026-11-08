"""X1 continued pretraining from public scGPT (blood 2M + whole-body 3M): orchestration + health check.
Every 10 min: poll every FunctionCall (fail -> exit 1), check pretraining logs advance (stall > 50 min -> exit 1),
finite losses, no collapse. Spawns the whole-body arm once the census cache manifest exists. When all pretraining
and chained Step-2 fine-tunes finish: scores replogle_k562_v1 (val + test) with diag/retrieval.py.
  ../.venv/bin/python -u diag/wait_xcont.py"""
import json
import math
import subprocess
import sys
import time

import modal

sys.path.insert(0, ".")
from modal_xjepa import CONT, WHOLEBODY  # noqa: E402

M = "../.venv/bin/modal"
IDS = "diag/call_ids_XC.json"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)  # noqa: E731
RUNS = {"cont blood X1": "jepa_blood_ijepa", "cont wholebody X1": "jepa_wholebody_ijepa",
        "cont bloodscgpt X1": "jepa_bloodscgpt_ijepa"}
seen = {}


def status(i):
    try:
        modal.FunctionCall.from_id(i).get(timeout=0)
        return "done"
    except TimeoutError:
        return "running"
    except Exception as e:
        return f"FAILED: {e!r}"[:400]


def vol_exists(path):
    return subprocess.run([M, "volume", "ls", "jepa-perturb", path], capture_output=True).returncode == 0


while True:
    ids = json.load(open(IDS))
    bad = {}
    if "cont wholebody X1" not in ids:
        if vol_exists("trackA/wholebody_census_2023-05-15_3M/manifest.json"):
            fn = modal.Function.from_name("jepa-xexp", "pretrain_cont")
            ids["cont wholebody X1"] = fn.spawn("ijepa", "wholebody_ijepa", {**CONT, "max_cells_per_epoch": 1_980_000},
                                                WHOLEBODY, "wijepa").object_id
            json.dump(ids, open(IDS, "w"), indent=1)
            log("census cache complete -> spawned whole-body X1 pretraining")
        else:
            r = subprocess.run([M, "volume", "ls", "jepa-perturb", "trackA/wholebody_census_2023-05-15_3M/shards"],
                               capture_output=True, text=True)
            log("census build: shards written", sum(1 for l in r.stdout.splitlines() if l.endswith(".npz")))
    for k, i in list(ids.items()):
        if i == "-":            # bookkeeping marker ("<arm> chained"), not a FunctionCall
            continue
        st = status(i)
        if st.startswith("FAILED"):
            bad[k] = st
        if k in RUNS and st == "done" and k + " chained" not in ids:
            r = modal.FunctionCall.from_id(i).get()
            log(k, "pretraining", r["status"], "best_val_recon", r.get("best_val_recon"))
            ids[k + " chained"] = "-"
            for kk, vv in (r.get("chained_finetunes") or {}).items():
                ids[kk] = vv
            json.dump(ids, open(IDS, "w"), indent=1)
    for k, run in RUNS.items():
        if k not in ids or status(ids[k]) != "running":
            continue
        p = f"/tmp/xjepa_health/{run}.jsonl"
        subprocess.run([M, "volume", "get", "--force", "jepa-perturb", f"trackA/pretrain/runs/{run}/log.jsonl", p],
                       capture_output=True)
        try:
            tr = [json.loads(x) for x in open(p) if '"msg": "train"' in x]
            if not tr:
                log(k, "no train lines yet")
                continue
            r = tr[-1]
            key = (r["epoch"], r["step"])
            if seen.get(k, (None,))[0] != key:
                seen[k] = (key, time.time())
            if not all(math.isfinite(r[x]) for x in ("loss", "recon", "jepa")):
                bad[k] = f"non-finite loss {r}"
            if time.time() - seen[k][1] > 50 * 60:
                bad[k] = f"log stalled 50 min at {key}"
            if r["global_step"] > 5000 and r["jepa"] > 0.8 * r["jepa_const_ref"]:
                bad[k] = f"possible collapse: jepa {r['jepa']:.4f} vs ref {r['jepa_const_ref']:.3f}"
            log(k, f"epoch {r['epoch']} step {r['step']}/61875 recon {r['recon']:.2f} jepa {r['jepa']:.4f} "
                   f"ref {r['jepa_const_ref']:.3f} raw_ref {r.get('jepa_const_ref_raw', float('nan')):.3f}")
        except Exception as e:
            log(k, "log read error", repr(e)[:200])
    fts = {k: v for k, v in ids.items() if k.startswith("step2")}
    n_run = sum(status(v) == "running" for v in fts.values())
    log("fine-tunes: total", len(fts), "running", n_run, "| failures", len(bad))
    if bad:
        for k, v in bad.items():
            log("PROBLEM", k, v)
        raise SystemExit(1)
    pre_done = all(k in ids and k + " chained" in ids for k in RUNS)
    if pre_done and n_run == 0:
        break
    time.sleep(600)

for split in ("val", "test"):
    r = subprocess.run([M, "run", "diag/retrieval.py", "--cache", "/vol/data/replogle_k562_essential_gears_simulation_seed1",
                        "--exps", "replogle_k562_v1", "--tag", f"XC_replogle_{split}", "--split", split],
                       capture_output=True, text=True)
    log("scored", split, "exit", r.returncode, r.stderr[-500:] if r.returncode else "")
log("ALL DONE")
