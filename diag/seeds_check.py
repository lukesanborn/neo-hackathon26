"""Short multi-seed pretraining runs on the real kidney cache (attempts.md A8 / A10).

Each run: N steps of one configuration from a seeded init, then validation recon on 1,024 fixed
val cells. Used to measure how often a configuration escapes the ~35 recon plateau.
Run from the repo root:  .venv/bin/modal run diag/seeds_check.py
"""
import json
import os
import sys

import modal

sys.path.insert(0, os.getcwd())
from environment.image import build_image  # noqa: E402

app = modal.App("jepa-diag-seeds2")
vol = modal.Volume.from_name("jepa-perturb")
image = build_image("blackwell")
CACHE = "/vol/trackA/kidney_census_2023-05-15"
VOCAB = "/vol/checkpoints/scGPT_human/vocab.json"
_CACHE = {}

# name: (PretrainConfig overrides, linear warmup steps, total steps)
CONFIGS = {
    "jepa_faithful_lr3e-5": ({"objective": "jepa", "jepa_variant": "faithful", "lr": 3e-5}, 0, 5000),
    "jepa_token_lr3e-5": ({"objective": "jepa", "jepa_variant": "token", "lr": 3e-5}, 0, 5000),
    # Xiong et al. 2020: post-norm needs warmup. Paper lr 1e-4, 4k-step linear warmup, longer run.
    "scgpt_warmup4k_lr1e-4": ({"objective": "scgpt", "lr": 1e-4}, 4000, 8000),
}


def _setup():
    import src.track_a.pretrain as P
    from src.models import SimpleVocab
    if "cache" not in _CACHE:
        _CACHE["cache"] = P.KidneyCache(CACHE)
    return P, _CACHE["cache"], SimpleVocab(VOCAB)


@app.function(image=image, gpu="B300", volumes={"/vol": vol}, cpu=8, memory=65536, timeout=3600)
def seed_run(name: str, seed: int) -> dict:
    import time

    import numpy as np

    over, warmup, steps = CONFIGS[name]
    P, cache, vocab = _setup()
    cfg = P.PretrainConfig(seed=seed, max_val_cells=1024, **over)
    tr = P.Pretrainer(cfg, vocab, "cuda")
    cells = P.epoch_cells(cache, cfg, 0)[: steps * cfg.batch_size]
    curve, acc, extra, t0 = [], [], {}, time.time()
    for s, nb in P.iter_batches(cache, cfg, cells, 0, 0, tr.cls_id, tr.pad_id, 4, 8):
        if warmup:
            for g in tr.opt.param_groups:
                g["lr"] = cfg.lr * min(1.0, (s + 1) / warmup)
        out = tr.train_step(nb, 0, s)
        acc.append(float(out["recon"]))
        if not np.isfinite(acc[-1]):
            return {"config": name, "seed": seed, "status": f"non-finite at step {s}"}
        if (s + 1) % 500 == 0:
            curve.append((s + 1, round(float(np.mean(acc)), 2)))
            acc = []
            extra = {k: round(float(out[k]), 6) for k in ("jepa", "teacher_cls_std", "teacher_tok_std") if k in out}
    v = tr.validate(cache, P.val_cells(cache, cfg))
    return {"config": name, "seed": seed, "val_recon": round(v["val_recon"], 3), "curve_every_500": curve,
            "last": extra, "minutes": round((time.time() - t0) / 60, 1)}


@app.local_entrypoint()
def main():
    jobs = [(n, s) for n in CONFIGS for s in (1, 2, 3, 4)]
    calls = [seed_run.spawn(n, s) for n, s in jobs]
    for (n, s), call in zip(jobs, calls):
        try:
            print("SEED", json.dumps(call.get()), flush=True)
        except Exception as e:
            print("SEED FAILED", n, s, repr(e)[:400], flush=True)
