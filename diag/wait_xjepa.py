"""Block until X1/X2 pretraining + every chained/baseline fine-tune + probe is done, then run the evals and
write diag/results/X_*.json + X_report.md.   ../.venv/bin/python diag/wait_xjepa.py"""
import json
import subprocess
import sys
import time

import modal

sys.path.insert(0, ".")
from modal_xjepa import ADAMSON_EXP, BASELINE_CKPTS, REPLOGLE_EXP, VARIANTS, ckpt_name  # noqa: E402

ids = json.load(open("diag/call_ids_X.json"))
probe_ids = json.load(open("diag/call_ids_X_probe_baselines.json"))
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)  # noqa: E731
get = lambda i: modal.FunctionCall.from_id(i).get(timeout=6 * 3600)  # noqa: E731

chained = {}
extra = modal.Function.from_name("jepa-xexp", "finetune")
for ck in ("scgpt_kidney_lr3e5", "jepa_kidney_lr3e5", "jepa_kidney_lr3e5_token"):
    k = f"replogle {ck.replace('_kidney', '')} scgpt_paper s3"
    if k not in ids:
        ids[k] = extra.spawn("replogle_trackA_lr3e5", "scgpt_paper", 3, ck.replace("_kidney", ""),
                             f"/vol/trackA/checkpoints/{ck}").object_id
json.dump(ids, open("diag/call_ids_X.json", "w"), indent=1)
log("baseline s3 spawned")
for v in VARIANTS:
    r = get(ids[f"pretrain {v}"])
    log("pretrain", v, r["status"], "best_val_recon", r["best_val_recon"], "epochs", r["epochs_completed"])
    chained.update(r.get("chained_finetunes", {}))
    ck = ckpt_name(v)
    chained[f"replogle {ck.replace('_kidney', '')} scgpt_paper s3"] = extra.spawn(
        "replogle_trackA_lr3e5", "scgpt_paper", 3, ck.replace("_kidney", ""), f"/vol/trackA/checkpoints/{ck}").object_id
    json.dump({**ids, **chained}, open("diag/call_ids_X.json", "w"), indent=1)
    probe_ids[ckpt_name(v)] = r.get("chained_probe")
json.dump({**ids, **chained}, open("diag/call_ids_X.json", "w"), indent=1)
for name, i in {**{k: v for k, v in ids.items() if not k.startswith("pretrain")}, **chained}.items():
    try:
        r = get(i)
        log("finetune", name, "test delta RMSE", (r.get("test") or r).get("delta_rmse", "?") if isinstance(r, dict) else r)
    except Exception as e:
        log("finetune FAILED", name, repr(e)[:300])
ev = modal.Function.from_name("jepa-xexp", "evaluate")
cks = BASELINE_CKPTS + [ckpt_name(v) for v in VARIANTS]
calls = {e: ev.spawn(e, cks) for e in (ADAMSON_EXP, REPLOGLE_EXP)}
for e, c in calls.items():
    r = c.get()
    json.dump(r, open(f"diag/results/X_{e}_retrieval.json", "w"), indent=1)
    log("eval", e, "skipped:", r["skipped"])
pp = {}
for n, i in probe_ids.items():
    try:
        pp[n] = get(i)
        log("probe", n, round(pp[n]["mean_r2"], 4))
    except Exception as e:
        log("probe FAILED", n, repr(e)[:300])
json.dump(pp, open("diag/results/X_program_probe.json", "w"), indent=1)
rep = subprocess.run([sys.executable, "diag/xjepa_report.py", "X"], capture_output=True, text=True).stdout
open("diag/results/X_report.md", "w").write(rep)
log("DONE")
