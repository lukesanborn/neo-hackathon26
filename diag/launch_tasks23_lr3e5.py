"""A14: Task 2 (zero-shot PBMC) and Task 3 (Adamson) on the four lr 3e-5 kidney checkpoints.

Spawns everything on deployed apps and exits (results are reviewed before any next step).
  * Task 2: jepa-pbmc.zeroshot(name, 1200) for each checkpoint          -> /vol/experiments/pbmc_zeroshot_v1/
  * Task 3: jepa-perturb.train_blackwell, experiment adamson_trackA_lr3e5, methods {scgpt_paper, celljepa}
            x seeds {1, 2}, bins50 inputs, 15 epochs                    -> /vol/experiments/adamson_trackA_lr3e5/runs/
Call ids -> diag/call_ids_A14.json.   Run from the repo root:  .venv/bin/python diag/launch_tasks23_lr3e5.py
"""
import json

import modal

CKPTS = ["scgpt_kidney_lr3e5", "jepa_kidney_lr3e5", "jepa_kidney_lr3e5_token", "jepa_kidney_lr3e5_center"]
EXP = "adamson_trackA_lr3e5"   # prepared already (same split hash 5618e72828c1)

zeroshot = modal.Function.from_name("jepa-pbmc", "zeroshot")
adamson = modal.Function.from_name("jepa-perturb", "train_blackwell").with_options(gpu="B300", max_containers=16)

ids = {}
for name in CKPTS:
    ids[f"T2 zeroshot {name}"] = zeroshot.spawn(name, 1200).object_id
    short = name.replace("_kidney", "")
    for m in ("scgpt_paper", "celljepa"):
        for s in (1, 2):
            ids[f"T3 adamson {short} {m} s{s}"] = adamson.spawn(
                exp_id=EXP, method=m, seed=s, epochs=15, aux_weight=0.5, max_minutes=150.0, tag=short,
                ckpt=f"/vol/trackA/checkpoints/{name}", input_transform="bins50").object_id
json.dump(ids, open("diag/call_ids_A14.json", "w"), indent=1)
print(json.dumps(ids, indent=1))
