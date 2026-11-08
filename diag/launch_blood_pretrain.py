"""A23 (Step 1): continued pretraining of public scGPT_human on 2M blood + bone-marrow cells.

Two matched arms (identical init = scGPT_human, data order, schedule): plain scGPT objective vs JEPA `center`
(the paper's JEPA collapses from public scGPT, A19). 2 epochs, batch 32, lr 3e-5, B300.
Exports: /vol/trackA/checkpoints/{scgpt,jepa}_kidney_blood_lr3e5 (export names keep the "_kidney" prefix
by convention; the tag marks the blood corpus). Call ids -> diag/call_ids_A23.json.
Run from the repo root:  .venv/bin/python diag/launch_blood_pretrain.py
"""
import json

import modal

CACHE = "/vol/trackA/blood_marrow_census_2023-05-15_2M"
fn = modal.Function.from_name("jepa-pretrain", "pretrain").with_options(gpu="B300")
base = {"epochs": 2, "batch_size": 32, "lr": 3e-5, "max_minutes": 600.0, "seed": 0, "init_ckpt": "scGPT_human"}
ids = {
    "blood scGPT": fn.spawn("scgpt", False, "blood_lr3e5", base, CACHE).object_id,
    "blood JEPA center": fn.spawn("jepa", False, "blood_lr3e5", {**base, "jepa_variant": "center"}, CACHE).object_id,
}
json.dump(ids, open("diag/call_ids_A23.json", "w"), indent=1)
print(json.dumps(ids, indent=1))
