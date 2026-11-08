"""A19: does the JEPA term carry per-cell signal when continuing from public scGPT? (<cls> spread pilot)

Public scGPT_human was pretrained without a <cls> objective, so its teacher <cls> embedding is nearly
identical across cells (spread ~0.009 at step 0, A18). Three JEPA settings, all from scGPT_human,
kidney cache, bs 32, lr 3e-5, 2,000 steps, seed 0:
  * faithful_cls0   paper JEPA, <cls> input value 0 (our convention)
  * faithful_clsm2  paper JEPA, <cls> input value -2 (public scGPT's own embedding convention)
  * center_cls0     DINO-style centred target (only per-cell deviations are predicted)
Logged every 100 steps: train recon, JEPA loss, teacher <cls> spread (+ centred-target spread for center),
and a "mean-target" reference: the JEPA loss a constant predictor (batch-mean target) would get.
Validation (1,024 val cells) at the end. Run from the repo root:  .venv/bin/modal run diag/cls_pilot.py
"""
import json
import os
import sys

import modal

sys.path.insert(0, os.getcwd())
from environment.image import build_image  # noqa: E402

app = modal.App("jepa-cls-pilot")
vol = modal.Volume.from_name("jepa-perturb")
image = build_image("blackwell")
CACHE = "/vol/trackA/kidney_census_2023-05-15"
VOCAB = "/vol/checkpoints/scGPT_human/vocab.json"
CONFIGS = {
    "faithful_cls0": ("faithful", 0.0),
    "faithful_clsm2": ("faithful", -2.0),
    "center_cls0": ("center", 0.0),
}


@app.function(image=image, gpu="B300", volumes={"/vol": vol}, cpu=8, memory=65536, timeout=3600)
def pilot(name: str, steps: int = 2000) -> dict:
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F

    import src.track_a.pretrain as P
    from src.models import SimpleVocab

    variant, cls_value = CONFIGS[name]
    P.CLS_VALUE = cls_value            # collate() reads the module-level constant
    cache = P.KidneyCache(CACHE)
    cfg = P.PretrainConfig(objective="jepa", jepa_variant=variant, init_ckpt="scGPT_human", lr=3e-5,
                           seed=0, max_val_cells=1024)
    tr = P.Pretrainer(cfg, SimpleVocab(VOCAB), "cuda")
    cells = P.epoch_cells(cache, cfg, 0)[: steps * cfg.batch_size]
    curve, acc, t0 = [], [], time.time()
    for s, nb in P.iter_batches(cache, cfg, cells, 0, 0, tr.cls_id, tr.pad_id, 4, 8):
        out = tr.train_step(nb, 0, s)
        rec = {k: float(out[k]) for k in ("recon", "jepa", "teacher_cls_std", "target_spread") if k in out}
        # constant-predictor reference: JEPA loss if the predictor always output the batch-mean teacher target
        with torch.no_grad():
            b = tr.to_device(nb)
            xt = tr.teacher.encoder(b["gene_ids"]) + tr.teacher.value_encoder(b["values"])
            ht = tr.teacher.transformer_encoder(xt, src_key_padding_mask=b["pad"])[:, 0].float()
            if variant == "center" and "center" in tr.jepa_state:
                ht = ht - tr.jepa_state["center"]
            t = F.normalize(ht, dim=-1)
            rec["meanpred_ref"] = float((1 - (t * F.normalize(t.mean(0, keepdim=True), dim=-1)).sum(-1)).mean())
        acc.append(rec)
        if (s + 1) % 100 == 0 or s == 0:
            curve.append({"step": s + 1, **{k: round(float(np.mean([a[k] for a in acc])), 5) for k in acc[0]}})
            acc = []
    v = tr.validate(cache, P.val_cells(cache, cfg))
    return {"config": name, "variant": variant, "cls_value": cls_value, "curve": curve,
            "val": {k: round(float(x), 5) for k, x in v.items()}, "minutes": round((time.time() - t0) / 60, 1)}


@app.local_entrypoint()
def main():
    calls = {n: pilot.spawn(n) for n in CONFIGS}
    os.makedirs("diag/results", exist_ok=True)
    with open("diag/results/A19_cls_pilot.jsonl", "w") as f:
        for n, c in calls.items():
            try:
                r = c.get()
            except Exception as e:
                r = {"config": n, "error": repr(e)[:400]}
            f.write(json.dumps(r) + "\n")
            print(json.dumps(r)[:3000], flush=True)
