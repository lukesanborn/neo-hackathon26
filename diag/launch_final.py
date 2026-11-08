"""A28: final data collection on Replogle (experiment replogle_k562_v1), 11 GPU runs, cap 11 concurrent.

  * Step 3: seeds 3, 4, 5 for the three finalists (same tags as Step 2, so they pool into 5-seed arms):
      blood-JEPA  -> Cell-JEPA FT + contrastive   (tag s2bjepa_r1)
      blood-scGPT -> Cell-JEPA FT + contrastive   (tag s2bscgpt_r1, matched control for JEPA pretraining)
      public scGPT -> scGPT FT + contrastive      (tag s2pub_r1, simplest strong option)
  * No-pretraining ablation: public scGPT architecture with RANDOM weights -> scGPT FT + contrastive, seeds 1, 2
All with checkpoint selection by validation ID percentile. Call ids -> diag/call_ids_A28.json.
Run from the repo root:  .venv/bin/python diag/launch_final.py
"""
import json

import modal

EXP = "replogle_k562_v1"
train = modal.Function.from_name("jepa-perturb", "train_blackwell").with_options(gpu="B300", max_containers=11)
BJ, BS = "/vol/trackA/checkpoints/jepa_kidney_blood_lr3e5", "/vol/trackA/checkpoints/scgpt_kidney_blood_lr3e5"
OV = {"selection": "retrieval", "retr_weight": 1.0}
jobs = []
for seed in (3, 4, 5):
    jobs += [("celljepa", seed, "s2bjepa_r1", BJ, "bins50", OV),
             ("celljepa", seed, "s2bscgpt_r1", BS, "bins50", OV),
             ("scgpt_paper", seed, "s2pub_r1", "scGPT_human", "log1p", OV)]
for seed in (1, 2):
    jobs.append(("scgpt_paper", seed, "noprt_r1", "scGPT_human", "log1p", {**OV, "random_init": True}))
ids = {}
for method, seed, tag, ckpt, tf, ov in jobs:
    ids[f"{method}_s{seed}_{tag}"] = train.spawn(exp_id=EXP, method=method, seed=seed, epochs=15, aux_weight=0.5,
                                                 max_minutes=150.0, tag=tag, ckpt=ckpt, input_transform=tf,
                                                 cfg_overrides=ov).object_id
json.dump(ids, open("diag/call_ids_A28.json", "w"), indent=1)
print(len(ids), "runs launched")
