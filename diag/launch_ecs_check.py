"""A13: why does PBMC fine-tuning lower public scGPT's clustering? Three variants, in parallel.

Run from the repo root:  .venv/bin/python diag/launch_ecs_check.py   (blocks until done, prints results)
"""
import json

import modal

ft = modal.Function.from_name("jepa-pbmc", "finetune")
variants = {
    "evalcurve": {"eval_every": 5},                       # current recipe, scored every 5 epochs
    "noecs": {"w_ecs": 0.0, "eval_every": 5},             # ECS off
    "ecs03": {"ecs_threshold": 0.3, "eval_every": 5},     # looser ECS threshold
}
calls = {tag: ft.spawn("scGPT_human", 0, ov, tag) for tag, ov in variants.items()}
with open("diag/results/A13_ecs_check.txt", "w") as f:
    for tag, c in calls.items():
        try:
            r = c.get()
        except Exception as e:
            r = {"error": repr(e)[:300]}
        line = json.dumps({"variant": tag, **r})
        print(line, flush=True)
        f.write(line + "\n")
