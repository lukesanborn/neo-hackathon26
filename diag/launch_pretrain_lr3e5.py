"""Launch full 4-epoch kidney pretraining at lr 3e-5 for all objectives (attempts.md A9).

Spawns on the DEPLOYED jepa-pretrain app so each job runs independently of this process.
Run from the repo root:  .venv/bin/python diag/launch_pretrain_lr3e5.py
Writes the FunctionCall ids to diag/call_ids_A9.json.
"""
import json
import os

import modal

fn = modal.Function.from_name("jepa-pretrain", "pretrain").with_options(gpu="B300")
base = {"epochs": 4, "batch_size": 32, "lr": 3e-5, "max_minutes": 600.0, "seed": 0, "ema_momentum": 0.996}
jobs = {  # name: (objective, tag, overrides) -> run dir trackA/pretrain/runs/<objective>_<tag>
    "scgpt_lr3e5": ("scgpt", "lr3e5", base),
    "jepa_faithful_lr3e5": ("jepa", "lr3e5", {**base, "jepa_variant": "faithful"}),
    "jepa_token_lr3e5": ("jepa", "lr3e5_token", {**base, "jepa_variant": "token"}),
    "jepa_center_lr3e5": ("jepa", "lr3e5_center", {**base, "jepa_variant": "center"}),
}
ids = {name: fn.spawn(obj, False, tag, ov).object_id for name, (obj, tag, ov) in jobs.items()}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "call_ids_A9.json")
json.dump(ids, open(out, "w"), indent=1)
print(json.dumps(ids, indent=1))
