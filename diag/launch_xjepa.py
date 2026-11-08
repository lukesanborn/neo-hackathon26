"""X1/X2 launcher. Spawns on DEPLOYED apps (jepa-xexp, jepa-perturb) and exits; ids -> diag/call_ids_X.json.
  1. X1 ijepa + X2 progmask: full 4-epoch kidney pretraining at lr 3e-5 (A9 schedule, mask_ratio 0.25); each
     chains, on completion, its fine-tunes (Adamson scgpt_paper+celljepa x s1,2; Replogle scgpt_paper x s1,2)
     and its program probe.
  2. Replogle Track A experiment (prepare = split + baselines), then baseline-checkpoint fine-tunes on it.
Run from the repo root:  ../.venv/bin/python diag/launch_xjepa.py
"""
import json
import sys

import modal

sys.path.insert(0, ".")
from modal_xjepa import BASE, REPLOGLE_DATASET, REPLOGLE_EXP, VARIANTS, spawn_finetunes  # noqa: E402

pretrain = modal.Function.from_name("jepa-xexp", "pretrain")
prepare = modal.Function.from_name("jepa-perturb", "prepare").with_options(memory=65536)
ids = {}
for v, tag in VARIANTS.items():
    ids[f"pretrain {v}"] = pretrain.spawn(v, tag, BASE, True).object_id
print(json.dumps(ids, indent=1), flush=True)
prep = prepare.remote(exp_id=REPLOGLE_EXP, dataset=REPLOGLE_DATASET)
print("replogle prepare:", json.dumps(prep, default=str)[:1500], flush=True)
for ck in ("scgpt_kidney_lr3e5", "jepa_kidney_lr3e5", "jepa_kidney_lr3e5_token"):
    ids.update(spawn_finetunes(ck, adamson_methods=(), replogle=True))
json.dump(ids, open("diag/call_ids_X.json", "w"), indent=1)
print(json.dumps(ids, indent=1))
