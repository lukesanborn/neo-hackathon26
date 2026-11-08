"""Is the knockdown signal in JEPA embeddings squashed (recoverable by a linear readout) or gone?

A26 found that on REAL Replogle cells, blood-JEPA's embedding space matches split halves of the same knockdown
far worse than scGPT's (val ID pct 0.685 vs 0.91-0.94). This job embeds real cells once per space and saves
per-cell embeddings so readouts can be compared on CPU (`diag/whiten_analyze.py`):

  * controls        : 4096 control cells (half to fit whitening, half spare)
  * val / test genes: up to 256 cells each (split into halves for the ceiling)
  * train genes     : up to 64 cells each (only for the supervised within-condition readout)

    .venv/bin/modal run diag/whiten_check.py
    .venv/bin/modal volume get jepa-perturb experiments/replogle_k562_v1/latent/whiten_<space>.npz <dir>
"""
import json
import os
import sys

import modal

sys.path.insert(0, os.getcwd())
from environment.image import build_image  # noqa: E402

app = modal.App("jepa-whiten-check")
vol = modal.Volume.from_name("jepa-perturb")
image = build_image("blackwell")
CACHE = "/vol/data/replogle_k562_essential_gears_simulation_seed1"
OUT = "/vol/experiments/replogle_k562_v1/latent"
SPACES = {"pub": "/vol/checkpoints/scGPT_human",
          "bscgpt": "/vol/trackA/checkpoints/scgpt_kidney_blood_lr3e5",
          "bjepa": "/vol/trackA/checkpoints/jepa_kidney_blood_lr3e5"}
N_CTRL, N_HELD, N_TRAIN = 4096, 256, 64


@app.function(image=image, gpu="B300", volumes={"/vol": vol}, cpu=8, memory=65536, timeout=3600)
def embed(space: str, ckpt: str = "") -> dict:
    import numpy as np

    from src.data import CTRL, load_cache
    from src.models import HGNC_ALIASES
    from src.track_a.pbmc import embed_cells, load_encoder

    ds = load_cache(CACHE, mmap=True)
    model, vocab, conv = load_encoder(ckpt or SPACES[space], "cuda")
    al = {**HGNC_ALIASES, **((ds.manifest or {}).get("extra_hgnc_aliases") or {})}
    syms = [g if g in vocab else al.get(g, g) for g in ds.gene_names]
    rng = np.random.default_rng(0)

    def emb(idx):
        rows = np.asarray(ds.X[np.sort(np.asarray(idx))], dtype=np.float32)
        indptr, indices, data = [0], [], []
        for r in rows:                      # real cells: keep every expressed gene (same as A26 refs)
            nz = np.flatnonzero(r > 0)
            indices.append(nz)
            data.append(r[nz])
            indptr.append(indptr[-1] + len(nz))
        e, _ = embed_cells(model, vocab, conv, np.asarray(indptr), np.concatenate(indices),
                           np.concatenate(data).astype(np.float32), syms, max_genes=1200, seed=0, device="cuda")
        return e.astype(np.float32)

    out = {"ctrl": emb(rng.choice(ds.ctrl_idx, min(N_CTRL, len(ds.ctrl_idx)), replace=False))}
    for split, n in (("val", N_HELD), ("test", N_HELD), ("train", N_TRAIN)):
        conds = [c for c in ds.split[split] if c != CTRL]
        E, lab = [], []
        for j, c in enumerate(conds):
            cells = ds.cells_of(c)
            cells = rng.permutation(rng.choice(cells, min(n, len(cells)), replace=False))
            e = emb(cells)
            E.append(e)
            lab.append(np.full(len(e), j, dtype=np.int32))
        out[f"{split}_emb"] = np.concatenate(E)
        out[f"{split}_lab"] = np.concatenate(lab)
        out[f"{split}_conds"] = np.array(conds)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"whiten_{space}.npz"), **out)
    vol.commit()
    return {"space": space, **{k: list(v.shape) for k, v in out.items()}}


@app.local_entrypoint()
def main(spaces: str = ""):
    """--spaces name=/vol/ckpt_dir,... (default: the three A27 spaces)."""
    sp = dict(x.split("=", 1) for x in spaces.split(",")) if spaces else SPACES
    for r in embed.starmap(list(sp.items())):
        print("DONE", json.dumps(r), flush=True)
