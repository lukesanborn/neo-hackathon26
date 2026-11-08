"""A27: is the perturbation signal erased by JEPA's embedding, or only squashed? (real cells only, no models)

For each embedding space (public scGPT, blood-scGPT, blood-JEPA) x readout (<cls>, mean of gene tokens) x
transform (raw; ZCA whitening on control cells; remove top-10 control PCs; LDA fitted on TRAINING knockdowns),
split each knockdown's real cells into two halves and ask whether half A's latent shift identifies half B's
(ID percentile, top-1/3/10). Gene-expression space (500-gene panel from training deltas) is the reference.
Gene sets: 31 VALIDATION genes (all transforms) and the 278 TRAINING genes (transforms that do not learn
from training knockdowns: raw / whiten / remove-PCs), the latter for a larger, less noisy read.
Run from the repo root:  .venv/bin/modal run diag/latent_transforms.py
"""
import json
import os
import sys

import modal

sys.path.insert(0, os.getcwd())
from environment.image import build_image  # noqa: E402

app = modal.App("jepa-latent-transforms")
vol = modal.Volume.from_name("jepa-perturb")
image = build_image("blackwell")
CACHE = "/vol/data/replogle_k562_essential_gears_simulation_seed1"
SPACES = {"pub": "/vol/checkpoints/scGPT_human",
          "bscgpt": "/vol/trackA/checkpoints/scgpt_kidney_blood_lr3e5",
          "bjepa": "/vol/trackA/checkpoints/jepa_kidney_blood_lr3e5",
          # kidney models pretrained from scratch (A9), to test whether the JEPA trade-off is general
          "kscgpt": "/vol/trackA/checkpoints/scgpt_kidney_lr3e5",
          "kjepa": "/vol/trackA/checkpoints/jepa_kidney_lr3e5",
          "kjepa_token": "/vol/trackA/checkpoints/jepa_kidney_lr3e5_token",
          "kjepa_center": "/vol/trackA/checkpoints/jepa_kidney_lr3e5_center"}
NAMES = {"pub": "public scGPT", "bscgpt": "blood-scGPT", "bjepa": "blood-JEPA", "expr": "gene expression (no model)",
         "kscgpt": "kidney scGPT", "kjepa": "kidney JEPA (paper, faithful)", "kjepa_token": "kidney JEPA token",
         "kjepa_center": "kidney JEPA center"}
N_CTRL, N_VAL, N_TRAIN = 2048, 256, 64


def _halves(np, rng, cells, cap):
    cells = rng.choice(cells, min(cap, len(cells)), replace=False)
    h = len(cells) // 2
    return np.sort(cells[:h]), np.sort(cells[h:])


def _id(RT, np, A, B):
    """Symmetric: average of identifying B from A and A from B."""
    a, b = RT.id_scores(A, B), RT.id_scores(B, A)
    return {k: (a[k] + b[k]) / 2 for k in ("id_pct", "id_top1", "id_top3", "id_top10")}


@app.function(image=image, gpu="B300", volumes={"/vol": vol}, cpu=8, memory=65536, timeout=3600, max_containers=4)
def run(space: str) -> dict:
    import numpy as np

    from src import retrieval as RT
    from src.data import CTRL, load_cache
    from src.latent_retrieval import embed_rows_both, fit_lda, fit_remove_pcs, fit_whiten, n_expressed
    from src.models import HGNC_ALIASES
    from src.track_a.pbmc import load_encoder

    ds = load_cache(CACHE, mmap=True)
    rng = np.random.default_rng(0)
    val = list(ds.split["val"])
    train = [c for c in ds.split["train"] if c != CTRL]
    rows = lambda idx: np.asarray(ds.X[np.sort(np.asarray(idx))], dtype=np.float32)  # noqa: E731
    ctrl_cells = np.sort(rng.choice(ds.ctrl_idx, min(N_CTRL, len(ds.ctrl_idx)), replace=False))
    vh = {c: _halves(np, rng, ds.cells_of(c), N_VAL) for c in val}
    th = {c: _halves(np, rng, ds.cells_of(c), N_TRAIN) for c in train}

    if space == "expr":   # reference: the expression space itself, 500-gene panel from training deltas
        c0 = rows(ctrl_cells).mean(0)
        R = np.stack([rows(np.concatenate(th[c])).mean(0) - c0 for c in train])
        g = RT.panel_genes(R)
        sh = lambda cells: (rows(cells).mean(0) - c0)[g]  # noqa: E731
        res = {"raw": {"val": _id(RT, np, np.stack([sh(vh[c][0]) for c in val]), np.stack([sh(vh[c][1]) for c in val])),
                       "train": _id(RT, np, np.stack([sh(th[c][0]) for c in train]), np.stack([sh(th[c][1]) for c in train]))}}
        return {"space": space, "readouts": {"expr": res}}

    model, vocab, conv = load_encoder(SPACES[space], "cuda")
    al = {**HGNC_ALIASES, **((ds.manifest or {}).get("extra_hgnc_aliases") or {})}
    syms = [x if x in vocab else al.get(x, x) for x in ds.gene_names]

    def emb(cells):
        r = rows(cells)
        return embed_rows_both(model, vocab, conv, r, n_expressed(r), syms)

    Ec = emb(ctrl_cells)
    Ev = {c: (emb(vh[c][0]), emb(vh[c][1])) for c in val}
    Et = {c: (emb(th[c][0]), emb(th[c][1])) for c in train}
    out = {}
    for ri, rname in ((0, "cls"), (1, "tok")):
        ctrl = Ec[ri]
        Xtr = np.concatenate([np.concatenate([Et[c][0][ri], Et[c][1][ri]]) for c in train])
        ytr = np.concatenate([np.full(len(Et[c][0][ri]) + len(Et[c][1][ri]), i) for i, c in enumerate(train)])
        tf = {"raw": lambda E: E, "whiten": fit_whiten(ctrl), "remove_pcs10": fit_remove_pcs(ctrl, 10),
              "lda50": fit_lda(Xtr, ytr, 50)}
        res = {}
        for tname, f in tf.items():
            c0 = f(ctrl).mean(0)
            sh = lambda E: f(E).mean(0) - c0  # noqa: E731
            r = {"val": _id(RT, np, np.stack([sh(Ev[c][0][ri]) for c in val]), np.stack([sh(Ev[c][1][ri]) for c in val]))}
            if tname != "lda50":          # LDA is fitted on training knockdowns: no in-sample evaluation
                r["train"] = _id(RT, np, np.stack([sh(Et[c][0][ri]) for c in train]),
                                 np.stack([sh(Et[c][1][ri]) for c in train]))
            res[tname] = r
        out[rname] = res
    return {"space": space, "readouts": out, "n_val": len(val), "n_train": len(train)}


@app.local_entrypoint()
def main(spaces: str = "expr,pub,bscgpt,bjepa", tag: str = "A27_latent_transforms"):
    order = spaces.split(",")
    results = {r["space"]: r for r in run.map(order)}
    os.makedirs("diag/results", exist_ok=True)
    json.dump(results, open(f"diag/results/{tag}.json", "w"), indent=1)
    L = ["| Space | Readout | Transform | VAL ID pct (31) | VAL top-1 | VAL top-3 | VAL top-10 | TRAIN ID pct (278) | TRAIN top-1 | TRAIN top-10 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for sp in order:
        for rname, res in results[sp]["readouts"].items():
            for tname, r in res.items():
                v, t = r["val"], r.get("train")
                L.append(f"| {NAMES[sp]} | {rname} | {tname} | {v['id_pct']:.3f} | {v['id_top1']:.3f} | {v['id_top3']:.3f} | "
                         f"{v['id_top10']:.3f} | " + (f"{t['id_pct']:.3f} | {t['id_top1']:.3f} | {t['id_top10']:.3f}" if t else "– | – | –") + " |")
    open(f"diag/results/{tag}.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
