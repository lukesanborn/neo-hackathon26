"""Status + results of the A14 jobs (Task 2 zero-shot PBMC, Task 3 Adamson) from diag/call_ids_A14.json.

Prints one line per job and writes finished results to diag/results/A14_results.json (cumulative).
Run from the repo root:  .venv/bin/python diag/status_A14.py
"""
import json
import os

import modal

ids = json.load(open("diag/call_ids_A14.json"))
out_path = "diag/results/A14_results.json"
results = json.load(open(out_path)) if os.path.exists(out_path) else {}


def find(d, keys):
    if isinstance(d, dict):
        for k in keys:
            if k in d and isinstance(d[k], (int, float)):
                return d[k]
        for v in d.values():
            r = find(v, keys)
            if r is not None:
                return r
    return None


n_done = n_run = n_err = 0
for k, cid in ids.items():
    if k in results and "error" not in results[k]:
        n_done += 1
        continue
    try:
        r = modal.FunctionCall.from_id(cid).get(timeout=0)
    except TimeoutError:
        n_run += 1
        print(f"{k:45s} running")
        continue
    except Exception as e:
        n_err += 1
        results[k] = {"error": repr(e)[:300]}
        print(f"{k:45s} ERROR {e!r:.150}")
        continue
    n_done += 1
    if k.startswith("T2"):
        results[k] = {m: r[m] for m in ("AvgBIO", "NMI", "ARI", "ASW", "best_resolution")}
    else:
        t = r.get("test", r)
        results[k] = {"status": r.get("status"), "epochs": r.get("epochs_completed"),
                      "selected_epoch": r.get("selected_epoch"),
                      "delta_rmse": find(t, ("delta_rmse", "mean_delta_rmse")),
                      "delta_pearson": find(t, ("delta_pearson", "mean_delta_pearson")),
                      "de20_delta_pearson": find(t, ("de20_delta_pearson",)),
                      "minutes": round((r.get("elapsed_seconds") or 0) / 60, 1)}
    print(f"{k:45s} DONE  {json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in results[k].items()})}")
json.dump(results, open(out_path, "w"), indent=1)
print(f"\n{n_done} done, {n_run} running, {n_err} errors, of {len(ids)}")
