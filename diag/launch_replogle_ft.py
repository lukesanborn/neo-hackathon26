"""A22: Replogle K562 essential fine-tunes (experiment replogle_k562_v1, 103 held-out test genes).

Arms x seeds {1, 2}, 15 epochs, B300, all spawned at once on the deployed jepa-perturb app:
  * public scGPT_human (log1p inputs)  -> scgpt_paper FT, celljepa FT
  * kidney scGPT lr3e-5 (bins50)       -> scgpt_paper FT     (paper's baseline arm)
  * kidney JEPA  lr3e-5 (bins50)       -> celljepa FT        (paper's Cell-JEPA arm)
Call ids -> diag/call_ids_A22.json.   Run from the repo root:  .venv/bin/python diag/launch_replogle_ft.py
"""
import json

import modal

EXP = "replogle_k562_v1"
train = modal.Function.from_name("jepa-perturb", "train_blackwell").with_options(gpu="B300", max_containers=16)
ARMS = [  # (label, method, ckpt, input_transform, tag)
    ("public scgpt_paper", "scgpt_paper", "scGPT_human", "log1p", "public"),
    ("public celljepa", "celljepa", "scGPT_human", "log1p", "public"),
    ("kidney scGPT -> scgpt_paper", "scgpt_paper", "/vol/trackA/checkpoints/scgpt_kidney_lr3e5", "bins50", "scgpt_lr3e5"),
    ("kidney JEPA -> celljepa", "celljepa", "/vol/trackA/checkpoints/jepa_kidney_lr3e5", "bins50", "jepa_lr3e5"),
]
ids = {}
for label, method, ckpt, tf, tag in ARMS:
    for seed in (1, 2):
        ids[f"{label} s{seed}"] = train.spawn(exp_id=EXP, method=method, seed=seed, epochs=15, aux_weight=0.5,
                                              max_minutes=150.0, tag=tag, ckpt=ckpt, input_transform=tf).object_id
json.dump(ids, open("diag/call_ids_A22.json", "w"), indent=1)
print(json.dumps(ids, indent=1))
