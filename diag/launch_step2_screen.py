"""A25 (Step 2): Replogle K562 screen, selected on VALIDATION genes only.

3 starting points x 2 fine-tune recipes x contrastive retrieval loss {off, on} x seeds {1, 2} = 24 runs,
all with checkpoint selection by validation identification percentile (selection="retrieval").
  starting points: public scGPT_human (log1p inputs, as in all public runs), blood-scGPT and blood-JEPA
  `center` (continued pretraining from public scGPT on 2M blood cells, A23; bins50 inputs as pretrained).
Experiment replogle_k562_v1 (split d4237826c68c). Run dirs: runs/<method>_s<seed>_s2<start>_r<0|1>.
Call ids -> diag/call_ids_A25.json.   Run from the repo root:  .venv/bin/python diag/launch_step2_screen.py
"""
import json

import modal

EXP = "replogle_k562_v1"
train = modal.Function.from_name("jepa-perturb", "train_blackwell").with_options(gpu="B300", max_containers=24)
STARTS = {  # name: (ckpt, input_transform)
    "pub": ("scGPT_human", "log1p"),
    "bscgpt": ("/vol/trackA/checkpoints/scgpt_kidney_blood_lr3e5", "bins50"),
    "bjepa": ("/vol/trackA/checkpoints/jepa_kidney_blood_lr3e5", "bins50"),
}
ids = {}
for start, (ckpt, tf) in STARTS.items():
    for method in ("scgpt_paper", "celljepa"):
        for retr in (0, 1):
            for seed in (1, 2):
                ov = {"selection": "retrieval", "retr_weight": float(retr)}
                ids[f"{start} {method} retr{retr} s{seed}"] = train.spawn(
                    exp_id=EXP, method=method, seed=seed, epochs=15, aux_weight=0.5, max_minutes=150.0,
                    tag=f"s2{start}_r{retr}", ckpt=ckpt, input_transform=tf, cfg_overrides=ov).object_id
json.dump(ids, open("diag/call_ids_A25.json", "w"), indent=1)
print(len(ids), "runs launched")
