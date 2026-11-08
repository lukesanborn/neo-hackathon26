"""Health check loop for X1/X2 jobs. Every 10 min: every FunctionCall in diag/call_ids_X.json (+ probes) is polled
(get(timeout=0): running -> TimeoutError, failed -> raises); pretraining logs on the volume must keep advancing
(stall > 50 min) with finite losses. Exits 1 on the first failure/stall (so the caller is notified), 0 when all
jobs are done.   ../.venv/bin/python diag/health_xjepa.py"""
import json
import math
import subprocess
import time

import modal

SP = "/tmp/xjepa_health"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)  # noqa: E731
last_step = {}
subprocess.run(["mkdir", "-p", SP])


def status(i):
    try:
        modal.FunctionCall.from_id(i).get(timeout=0)
        return "done"
    except (TimeoutError, modal.exception.OutputExpiredError) as e:
        return "running" if isinstance(e, TimeoutError) else "done"
    except Exception as e:
        return f"FAILED: {e!r}"[:400]


while True:
    ids = json.load(open("diag/call_ids_X.json"))
    try:
        ids.update({f"probe {k}": v for k, v in json.load(open("diag/call_ids_X_probe_baselines.json")).items() if v})
    except Exception:
        pass
    st = {k: status(v) for k, v in ids.items()}
    bad = {k: v for k, v in st.items() if v.startswith("FAILED")}
    counts = {s: sum(1 for v in st.values() if v == s) for s in ("running", "done")}
    log("jobs", counts, "failed", len(bad))
    for tag in ("lr3e5_ijepa", "lr3e5_progmask"):
        if st.get(f"pretrain {tag.split('_')[1]}") != "running":
            continue
        p = f"{SP}/{tag}.jsonl"
        subprocess.run(["../.venv/bin/modal", "volume", "get", "--force", "jepa-perturb",
                        f"trackA/pretrain/runs/jepa_{tag}/log.jsonl", p], capture_output=True)
        try:
            recs = [json.loads(x) for x in open(p)]
            tr = [r for r in recs if r["msg"] == "train"]
            r = tr[-1]
            key = (r["epoch"], r["step"])
            finite = all(math.isfinite(r[k]) for k in ("loss", "recon", "jepa"))
            prev = last_step.get(tag)
            if prev is None or prev[0] != key:
                last_step[tag] = (key, time.time())
            stalled = time.time() - last_step[tag][1] > 50 * 60
            log(tag, f"epoch {r['epoch']} step {r['step']} recon {r['recon']:.2f} jepa {r['jepa']:.4f} "
                     f"ref {r.get('jepa_const_ref', float('nan')):.3f}", "OK" if finite and not stalled else "PROBLEM")
            if not finite:
                bad[tag] = f"non-finite loss {r}"
            if stalled:
                bad[tag] = f"no new log for 50 min (last {key})"
            if r.get("jepa_const_ref") and r["jepa"] > 0.8 * r["jepa_const_ref"] and r["epoch"] >= 1:
                bad[tag] = f"possible collapse: jepa {r['jepa']:.4f} vs const ref {r['jepa_const_ref']:.3f}"
        except Exception as e:
            log(tag, "log read error", repr(e)[:200])
    if bad:
        for k, v in bad.items():
            log("PROBLEM", k, v)
        raise SystemExit(1)
    pending_chain = any(v == "running" for k, v in st.items() if k.startswith("pretrain"))
    if not pending_chain and counts["running"] == 0:
        log("ALL DONE")
        raise SystemExit(0)
    time.sleep(600)
