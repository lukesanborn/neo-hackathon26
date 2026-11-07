"""Track A — Cell-JEPA reimplemented from the paper (unofficial): PBMC-10K tasks on Modal.

Task 2 (paper Table 2): zero-shot cell-type clustering of frozen <cls> embeddings.

    modal run modal_pbmc.py --stage prepare                 # CPU: scvi-tools PBMC-10K -> /vol/data/pbmc10k
    modal run modal_pbmc.py --stage zeroshot                # every checkpoint + PCA baseline, in parallel
    modal run modal_pbmc.py --stage zeroshot --ckpts scGPT_human,jepa_kidney_v1_token --max-genes 1200

Checkpoint names: "scGPT_human" (public, /vol/checkpoints) or a directory under
/vol/trackA/checkpoints. Results: /vol/experiments/pbmc_zeroshot_v1/<name>_mg<max_genes>.json
(+ embeddings/), and summary.json / summary.md from the local entrypoint.
"""

from __future__ import annotations

import json
import os

import modal

from environment.image import build_image

APP_NAME = "jepa-pbmc"
VOL_PATH = "/vol"
DATA_DIR = f"{VOL_PATH}/data/pbmc10k"
OUT_DIR = f"{VOL_PATH}/experiments/pbmc_zeroshot_v1"
PAPER_TABLE2 = {"scGPT (paper)": {"AvgBIO": 0.5288, "NMI": 0.5585, "ASW": 0.5329, "ARI": 0.4951},
                "Cell-JEPA (paper)": {"AvgBIO": 0.7194, "NMI": 0.7690, "ASW": 0.5811, "ARI": 0.8081}}

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("jepa-perturb")
gpu_image = build_image("blackwell", extra_pip=("leidenalg==0.10.2", "igraph==0.11.6"))
scvi_image = (modal.Image.debian_slim(python_version="3.11")
              .pip_install("scvi-tools==1.1.6", "jax==0.4.30", "jaxlib==0.4.30", "scanpy==1.10.2", "anndata==0.10.9", "numpy==1.26.4", "scipy==1.13.1", "pandas==2.2.3")
              .add_local_python_source("src", "environment"))


def ckpt_dir(name: str) -> str:
    return f"{VOL_PATH}/checkpoints/{name}" if name == "scGPT_human" else f"{VOL_PATH}/trackA/checkpoints/{name}"


def _load_data():
    import numpy as np

    # our own file; early versions stored the string columns as object arrays
    z = np.load(os.path.join(DATA_DIR, "pbmc10k.npz"), allow_pickle=True)
    return {k: (z[k].astype(str) if z[k].dtype == object else z[k]) for k in z.files}


@app.function(image=scvi_image, volumes={VOL_PATH: vol}, cpu=4, memory=32768, timeout=3600)
def prepare(force: bool = False) -> dict:
    """PBMC-10K as in the paper: scvi-tools pbmc_dataset (10x pbmc4k + pbmc8k, 3,346 genes, labels)."""
    import numpy as np
    import scipy.sparse as sp
    import scvi

    man_path = os.path.join(DATA_DIR, "manifest.json")
    if os.path.exists(man_path) and not force:
        return json.load(open(man_path))
    os.makedirs(DATA_DIR, exist_ok=True)
    a = scvi.data.pbmc_dataset(save_path=f"{VOL_PATH}/raw/pbmc_scvi", remove_extracted_data=True)
    X = sp.csr_matrix(a.X)
    symbols = a.var["gene_symbols"].astype(str).tolist() if "gene_symbols" in a.var else a.var_names.astype(str).tolist()
    labels = a.obs["str_labels"].astype(str).to_numpy()
    batch = a.obs["batch"].astype(str).to_numpy()
    np.savez(os.path.join(DATA_DIR, "pbmc10k.npz"), indptr=X.indptr.astype(np.int64), indices=X.indices.astype(np.int64),
             data=X.data.astype(np.float32), gene_symbols=np.array(symbols, dtype=str),
             labels=labels.astype(str), batch=batch.astype(str))
    u, c = np.unique(labels, return_counts=True)
    ub, cb = np.unique(batch, return_counts=True)
    man = {"source": "scvi-tools 1.1.6 scvi.data.pbmc_dataset (10x pbmc4k + pbmc8k, scVI-data gene list/labels)",
           "n_cells": int(X.shape[0]), "n_genes": int(X.shape[1]), "labels": dict(zip(u.tolist(), c.tolist())),
           "batches": dict(zip(ub.tolist(), cb.tolist())), "paper_expectation": "3,346 HVGs; 7,982 + 4,008 cells",
           "nonzero_genes_per_cell_mean": float(np.diff(X.indptr).mean()),
           "is_integer_counts": bool(np.allclose(X.data, np.round(X.data)))}
    json.dump(man, open(man_path, "w"), indent=2)
    vol.commit()
    return man


def _save(name: str, res: dict, emb):
    import numpy as np

    os.makedirs(os.path.join(OUT_DIR, "embeddings"), exist_ok=True)
    np.save(os.path.join(OUT_DIR, "embeddings", f"{name}.npy"), emb.astype(np.float32))
    json.dump(res, open(os.path.join(OUT_DIR, f"{name}.json"), "w"), indent=2)
    vol.commit()


@app.function(image=gpu_image, gpu="B300", volumes={VOL_PATH: vol}, cpu=8, memory=32768, timeout=3600)
def zeroshot(name: str, max_genes: int = 1200) -> dict:
    from src.track_a.pbmc import TRACK_LABEL, cluster_metrics, embed_cells, load_encoder

    vol.reload()
    d = _load_data()
    model, vocab, conv = load_encoder(ckpt_dir(name), "cuda")
    emb, info = embed_cells(model, vocab, conv, d["indptr"], d["indices"], d["data"], d["gene_symbols"].tolist(),
                            max_genes=max_genes)
    res = {"track": TRACK_LABEL, "task": "Task 2: zero-shot PBMC-10K", "checkpoint": name, **info,
           "model": conv, **cluster_metrics(emb, d["labels"])}
    _save(f"{name}_mg{max_genes}", res, emb)
    return res


@app.function(image=gpu_image, volumes={VOL_PATH: vol}, cpu=8, memory=32768, timeout=3600)
def pca_baseline(n_comps: int = 50) -> dict:
    from src.track_a.pbmc import TRACK_LABEL, cluster_metrics, pca_embedding

    vol.reload()
    d = _load_data()
    emb = pca_embedding(d["indptr"], d["indices"], d["data"], len(d["gene_symbols"]), n_comps)
    res = {"track": TRACK_LABEL, "task": "Task 2: zero-shot PBMC-10K", "checkpoint": f"PCA-{n_comps} (log1p, 1e4)",
           **cluster_metrics(emb, d["labels"])}
    _save(f"pca{n_comps}", res, emb)
    return res


@app.local_entrypoint()
def main(stage: str = "zeroshot", ckpts: str = "", max_genes: str = "600,1200", force: bool = False):
    def show(x):
        print(json.dumps(x, indent=2, default=str), flush=True)

    if stage == "prepare":
        show(prepare.remote(force))
        return
    assert stage == "zeroshot", stage
    if ckpts:
        names = ckpts.split(",")
    else:
        names = ["scGPT_human"] + sorted(
            e.path.rstrip("/").split("/")[-1] for e in vol.listdir("trackA/checkpoints")
            if any(f.path.endswith("best_model.pt") for f in vol.listdir(e.path)))
    mgs = [int(m) for m in max_genes.split(",")]
    print(f"checkpoints: {names}\nmax_genes: {mgs}", flush=True)
    calls = {"pca50": pca_baseline.spawn()}
    calls.update({f"{n}_mg{m}": zeroshot.spawn(n, m) for n in names for m in mgs})
    rows = {}
    for k, c in calls.items():
        try:
            r = c.get()
            rows[k] = {m: round(r[m], 4) for m in ("AvgBIO", "NMI", "ARI", "ASW")} | {"res": r["best_resolution"]}
        except Exception as e:
            rows[k] = {"error": repr(e)[:300]}
        print(k, rows[k], flush=True)
    lines = ["| Embedding | AvgBIO | NMI | ARI | ASW |", "|---|---|---|---|---|"]
    for k, r in {**rows, **PAPER_TABLE2}.items():
        lines.append(f"| {k} | " + " | ".join(str(r.get(m, r.get('error', '-'))) for m in ("AvgBIO", "NMI", "ARI", "ASW")) + " |")
    print("\n".join(lines))
    os.makedirs("diag/results", exist_ok=True)
    with open("diag/results/pbmc_zeroshot_summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
