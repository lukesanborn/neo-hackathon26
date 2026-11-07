"""A14: chain every downstream evaluation onto the lr 3e-5 pretraining runs (A9) as each one finishes.

Per finished checkpoint, spawns (all on deployed apps, all in parallel):
  * Task 2  zero-shot PBMC clustering            jepa-pbmc.zeroshot(name, 1200)
  * Task 1  PBMC fine-tune, seeds 0 and 1         jepa-pbmc.finetune(name, seed)
  * Task 3  Adamson fine-tune, {scgpt_paper, celljepa} x seeds {1, 2}, bins50 inputs,
            experiment adamson_trackA_lr3e5       jepa-perturb.train_blackwell(...)
Then waits for everything and writes one JSON line per job to diag/results/A14_downstream.txt.
Call ids go to diag/call_ids_A14.json as they are spawned.

Run from the repo root (keep it running):  .venv/bin/python diag/launch_after_pretrain.py
"""
import json
import time

import modal

PRETRAIN_IDS = json.load(open("diag/call_ids_A9.json"))
CKPT_OF = {"scgpt_lr3e5": "scgpt_kidney_lr3e5", "jepa_faithful_lr3e5": "jepa_kidney_lr3e5",
           "jepa_token_lr3e5": "jepa_kidney_lr3e5_token", "jepa_center_lr3e5": "jepa_kidney_lr3e5_center"}
EXP = "adamson_trackA_lr3e5"

zeroshot = modal.Function.from_name("jepa-pbmc", "zeroshot")
pbmc_ft = modal.Function.from_name("jepa-pbmc", "finetune")
prepare = modal.Function.from_name("jepa-perturb", "prepare")
adamson = modal.Function.from_name("jepa-perturb", "train_blackwell").with_options(gpu="B300", max_containers=16)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


log("preparing", EXP, prepare.remote(exp_id=EXP, split_seed=1).get("split_hash"))
spawned, ids, pending = {}, {}, dict(PRETRAIN_IDS)
while pending:
    for run, cid in list(pending.items()):
        try:
            modal.FunctionCall.from_id(cid).get(timeout=0)
        except TimeoutError:
            continue
        except Exception as e:
            log(f"pretrain {run} FAILED: {e!r:.200}; skipping its downstream jobs")
            pending.pop(run)
            continue
        pending.pop(run)
        name = CKPT_OF[run]
        log(f"pretrain {run} done -> launching downstream for {name}")
        jobs = {f"T2 zeroshot {name}": zeroshot.spawn(name, 1200)}
        for s in (0, 1):
            jobs[f"T1 pbmcFT {name} s{s}"] = pbmc_ft.spawn(name, s, None, "")
        short = name.replace("_kidney", "")
        for m in ("scgpt_paper", "celljepa"):
            for s in (1, 2):
                jobs[f"T3 adamson {short} {m} s{s}"] = adamson.spawn(
                    exp_id=EXP, method=m, seed=s, epochs=15, aux_weight=0.5, max_minutes=150.0, tag=short,
                    ckpt=f"/vol/trackA/checkpoints/{name}", input_transform="bins50")
        spawned.update(jobs)
        ids.update({k: c.object_id for k, c in jobs.items()})
        json.dump(ids, open("diag/call_ids_A14.json", "w"), indent=1)
    if pending:
        time.sleep(60)

log(f"all pretraining finished; waiting on {len(spawned)} downstream jobs")
with open("diag/results/A14_downstream.txt", "a") as f:
    for k, c in spawned.items():
        try:
            r = c.get()
        except Exception as e:
            r = {"error": repr(e)[:300]}
        keep = {kk: r[kk] for kk in ("AvgBIO", "NMI", "ARI", "ASW", "status", "epochs_completed", "test", "error", "run")
                if isinstance(r, dict) and kk in r}
        if "test" in keep and isinstance(keep["test"], dict):
            keep["test"] = {kk: keep["test"].get(kk) for kk in ("mean_delta_rmse", "delta_rmse", "mean_delta_pearson")}
        line = json.dumps({"job": k, **keep}, default=str)
        log(line)
        f.write(line + "\n")
log("done")
