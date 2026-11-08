"""Health-check one continued-pretraining arm + its chained A25 fine-tunes, then score replogle_k562_v1 (val+test).
  ../.venv/bin/python -u diag/wait_one.py <ids.json> <call key> <run dir name> <score tag>
Exits 1 on a failed call, stalled log (50 min), non-finite loss or collapse; 0 when scored."""
import json
import math
import subprocess
import sys
import time

import modal

IDS, KEY, RUN, TAG = sys.argv[1:5]
M = "../.venv/bin/modal"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)  # noqa: E731
seen = None


def status(i):
    try:
        modal.FunctionCall.from_id(i).get(timeout=0)
        return "done"
    except TimeoutError:
        return "running"
    except Exception as e:
        return f"FAILED: {e!r}"[:400]


while True:
    ids = json.load(open(IDS))
    bad = {k: s for k, s in ((k, status(v)) for k, v in ids.items() if v != "-") if s.startswith("FAILED")}
    st = status(ids[KEY])
    if st == "running":
        p = f"/tmp/xjepa_health/{RUN}.jsonl"
        subprocess.run([M, "volume", "get", "--force", "jepa-perturb", f"trackA/pretrain/runs/{RUN}/log.jsonl", p],
                       capture_output=True)
        try:
            tr = [json.loads(x) for x in open(p) if '"msg": "train"' in x]
            r = tr[-1]
            key = (r["epoch"], r["step"])
            if seen is None or seen[0] != key:
                seen = (key, time.time())
            if not all(math.isfinite(r[x]) for x in ("loss", "recon", "jepa")):
                bad[KEY] = f"non-finite loss {r}"
            if time.time() - seen[1] > 50 * 60:
                bad[KEY] = f"log stalled 50 min at {key}"
            if r["global_step"] > 5000 and r["jepa"] > 0.8 * r["jepa_const_ref"]:
                bad[KEY] = f"possible collapse: jepa {r['jepa']:.4f} vs ref {r['jepa_const_ref']:.3f}"
            log(f"pretrain epoch {r['epoch']} step {r['step']}/61875 recon {r['recon']:.2f} jepa {r['jepa']:.4f} "
                f"ref {r['jepa_const_ref']:.3f}")
        except Exception as e:
            log("pretrain: no log yet", repr(e)[:120])
    elif st == "done" and KEY + " chained" not in ids:
        r = modal.FunctionCall.from_id(ids[KEY]).get()
        log("pretraining", r["status"], "best_val_recon", r.get("best_val_recon"))
        ids.update(r.get("chained_finetunes") or {})
        ids[KEY + " chained"] = "-"
        json.dump(ids, open(IDS, "w"), indent=1)
    fts = {k: v for k, v in ids.items() if k.startswith("step2")}
    n_run = sum(status(v) == "running" for v in fts.values())
    log("fine-tunes: total", len(fts), "running", n_run, "| failures", len(bad))
    if bad:
        for k, v in bad.items():
            log("PROBLEM", k, v)
        raise SystemExit(1)
    if KEY + " chained" in ids and n_run == 0:
        break
    time.sleep(600)

for split in ("val", "test"):
    r = subprocess.run([M, "run", "diag/retrieval.py", "--cache", "/vol/data/replogle_k562_essential_gears_simulation_seed1",
                        "--exps", "replogle_k562_v1", "--tag", f"{TAG}_{split}", "--split", split],
                       capture_output=True, text=True)
    log("scored", split, "exit", r.returncode, r.stderr[-400:] if r.returncode else "")
log("ALL DONE")
