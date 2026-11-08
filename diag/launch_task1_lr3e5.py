"""A15: Task 1 (PBMC-10K fine-tune + clustering) on the four lr 3e-5 kidney checkpoints, seeds 0-2.

Spawns 12 jobs on the deployed jepa-pbmc app and exits. Each checkpoint is fine-tuned with its own
objective (JEPA checkpoint -> Cell-JEPA FT incl. JEPA term; scGPT checkpoint -> GEP + GEPC + ECS).
Results: /vol/experiments/pbmc_finetune_v1/<ckpt>_<obj>FT_s<seed>/result.json
Call ids -> diag/call_ids_A15.json.   Run from the repo root:  .venv/bin/python diag/launch_task1_lr3e5.py
"""
import json

import modal

CKPTS = ["scgpt_kidney_lr3e5", "jepa_kidney_lr3e5", "jepa_kidney_lr3e5_token", "jepa_kidney_lr3e5_center"]
ft = modal.Function.from_name("jepa-pbmc", "finetune")
ids = {f"T1 {name} s{s}": ft.spawn(name, s, None, "").object_id for name in CKPTS for s in (0, 1, 2)}
json.dump(ids, open("diag/call_ids_A15.json", "w"), indent=1)
print(json.dumps(ids, indent=1))
