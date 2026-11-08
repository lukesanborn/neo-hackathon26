"""Track A — Cell-JEPA reimplemented from the paper (unofficial): PBMC-10K cell-embedding tasks.

Task 2 (zero-shot, paper Table 2) and the shared pieces of Task 1 (fine-tuning, Table 1):
  * tokenisation that follows scGPT's own zero-shot procedure (scgpt.tasks.cell_emb.embed_data):
    a cell's nonzero genes that are in the vocab (sampled down to ``max_genes`` if longer), values
    = per-cell quantile bins 1..50 of the raw counts (same rule as the kidney cache), a <cls> token
    first, pad with <pad>; cell embedding = encoder output at <cls>, L2-normalised.
  * model conventions come from each checkpoint's args.json: <cls> value = ``cls_value`` (kidney
    exports: 0) or, if absent, ``pad_value`` (public scGPT_human: -2, as scGPT's embed_data does).
  * scib-equivalent metrics (as used by scGPT / Cell-JEPA): NMI and ARI between cell-type labels and
    Leiden clusters of a kNN graph on the embedding, at the resolution in 0.1..2.0 that maximises
    NMI (scib's cluster_optimal_resolution); ASW = (silhouette(embedding, labels) + 1) / 2;
    AvgBIO = mean(NMI, ARI, ASW).
  * PCA baseline (absent from the paper): normalise to 1e4 counts, log1p, PCA-50.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .census import bin_nonzero

TRACK_LABEL = "Track A — Cell-JEPA reimplemented from the paper (unofficial)"
LOAD_PREFIXES = ("encoder", "value_encoder", "transformer_encoder")
RESOLUTIONS = tuple(round(0.1 * i, 1) for i in range(1, 21))


# ----------------------------------------------------------------------------- tokenisation


def column_vocab_ids(gene_symbols: list[str], vocab) -> np.ndarray:
    """Vocab id per expression-matrix column, or -1 if the gene is not in the vocab (dropped)."""
    return np.array([vocab[g] if g in vocab else -1 for g in gene_symbols], dtype=np.int64)


def cell_tokens(indices: np.ndarray, counts: np.ndarray, col_ids: np.ndarray, max_genes: int,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One cell (CSR row: column indices + raw counts) -> (gene ids, bins 1..50), no <cls>.

    Keeps nonzero genes that are in the vocab, samples ``max_genes`` of them uniformly if there are
    more, then bins the kept values per cell (scGPT bins after sampling, in its DataCollator)."""
    keep = (counts > 0) & (col_ids[indices] >= 0)
    gids, vals = col_ids[indices[keep]], counts[keep].astype(np.float64)
    if len(gids) > max_genes:
        sel = np.sort(rng.choice(len(gids), size=max_genes, replace=False))
        gids, vals = gids[sel], vals[sel]
    return gids, bin_nonzero(vals).astype(np.float32)


def pad_batch(cells: list[tuple[np.ndarray, np.ndarray]], cls_id: int, pad_id: int, cls_value: float,
              pad_value: float) -> dict:
    """<cls> + tokens, right-padded to the batch max length. Returns numpy gene_ids, values, pad."""
    L = 1 + max((len(g) for g, _ in cells), default=0)
    B = len(cells)
    gene_ids = np.full((B, L), pad_id, dtype=np.int64)
    values = np.full((B, L), pad_value, dtype=np.float32)
    pad = np.ones((B, L), dtype=bool)
    gene_ids[:, 0], values[:, 0], pad[:, 0] = cls_id, cls_value, False
    for b, (g, v) in enumerate(cells):
        gene_ids[b, 1:1 + len(g)] = g
        values[b, 1:1 + len(g)] = v
        pad[b, 1:1 + len(g)] = False
    return {"gene_ids": gene_ids, "values": values, "pad": pad}


# ----------------------------------------------------------------------------- model


def load_encoder(ckpt_dir: str, device: str = "cuda"):
    """Encoder-only view of any checkpoint in the contract (args.json / vocab.json / best_model.pt).

    Builds scGPT's TransformerModel (no dropout) and loads encoder / value_encoder /
    transformer_encoder weights; flash-attn 'Wqkv' names (public scGPT_human) are renamed."""
    import torch

    from src.models import SimpleVocab

    from .pretrain_model import build_student

    with open(os.path.join(ckpt_dir, "args.json")) as f:
        args = json.load(f)
    vocab = SimpleVocab(os.path.join(ckpt_dir, "vocab.json"))
    model = build_student(vocab, d_model=args["embsize"], nhead=args["nheads"], d_hid=args["d_hid"],
                          nlayers=args["nlayers"], nlayers_cls=args.get("n_layers_cls", 3), dropout=0.0)
    sd = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location="cpu")
    sd = {k.replace("Wqkv.", "in_proj_"): v for k, v in sd.items() if k.startswith(LOAD_PREFIXES)}
    own = model.state_dict()
    expected = [k for k in own if k.startswith(LOAD_PREFIXES)]
    missing = [k for k in expected if k not in sd or sd[k].shape != own[k].shape]
    if missing:
        raise RuntimeError(f"{ckpt_dir}: encoder weights missing or mis-shaped: {missing[:10]}")
    own.update({k: sd[k] for k in expected})
    model.load_state_dict(own)
    model.to(device).eval()
    conv = {"pad_value": float(args.get("pad_value", -2)),
            "cls_value": float(args.get("cls_value", args.get("pad_value", -2))),
            "n_loaded": len(expected), "objective": args.get("objective", "scgpt_human (public)"),
            "best_val_recon": args.get("best_val_recon"), "epochs_completed": args.get("epochs_completed")}
    return model, vocab, conv


def embed_cells(model, vocab, conv: dict, indptr: np.ndarray, indices: np.ndarray, data: np.ndarray,
                gene_symbols: list[str], max_genes: int = 1200, batch_size: int = 64, seed: int = 0,
                device: str = "cuda") -> tuple[np.ndarray, dict]:
    """L2-normalised <cls> embeddings (n_cells, d) for a CSR count matrix."""
    import torch

    col_ids = column_vocab_ids(gene_symbols, vocab)
    rng = np.random.default_rng(seed)
    cls_id, pad_id = vocab["<cls>"], vocab["<pad>"]
    n = len(indptr) - 1
    out, lens = [], []
    amp = dict(device_type="cuda" if str(device).startswith("cuda") else "cpu", dtype=torch.bfloat16,
               enabled=str(device).startswith("cuda"))
    for i in range(0, n, batch_size):
        cells = [cell_tokens(indices[indptr[r]:indptr[r + 1]], data[indptr[r]:indptr[r + 1]], col_ids, max_genes, rng)
                 for r in range(i, min(i + batch_size, n))]
        lens += [len(g) for g, _ in cells]
        b = pad_batch(cells, cls_id, pad_id, conv["cls_value"], conv["pad_value"])
        g = torch.from_numpy(b["gene_ids"]).to(device)
        v = torch.from_numpy(b["values"]).to(device)
        p = torch.from_numpy(b["pad"]).to(device)
        with torch.no_grad(), torch.autocast(**amp):
            h = model.transformer_encoder(model.encoder(g) + model.value_encoder(v), src_key_padding_mask=p)
        out.append(h[:, 0].float().cpu().numpy())
    emb = np.concatenate(out)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-12)
    info = {"n_genes_in_vocab": int((col_ids >= 0).sum()), "n_genes": len(gene_symbols),
            "tokens_per_cell_mean": float(np.mean(lens)), "tokens_per_cell_max": int(np.max(lens)),
            "max_genes": max_genes}
    return emb, info


# ----------------------------------------------------------------------------- baseline + metrics


def pca_embedding(indptr, indices, data, n_genes: int, n_comps: int = 50, seed: int = 0) -> np.ndarray:
    """Normalise each cell to 1e4 counts, log1p, PCA (sklearn, centred, no scaling)."""
    import scipy.sparse as sp
    from sklearn.decomposition import PCA

    X = sp.csr_matrix((data.astype(np.float32), indices, indptr), shape=(len(indptr) - 1, n_genes))
    tot = np.asarray(X.sum(1)).ravel().clip(1e-12)
    X = sp.diags(1e4 / tot) @ X
    X.data = np.log1p(X.data)
    return PCA(n_components=n_comps, random_state=seed).fit_transform(X.toarray()).astype(np.float32)


def asw_label(emb: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score

    return float((silhouette_score(emb, labels, metric="euclidean") + 1) / 2)


def cluster_metrics(emb: np.ndarray, labels: np.ndarray, n_neighbors: int = 15,
                    resolutions=RESOLUTIONS, seed: int = 0) -> dict:
    """scib-equivalent NMI / ARI (optimal-resolution Leiden) + label ASW + AvgBIO."""
    import anndata as ad
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    a = ad.AnnData(np.zeros((len(emb), 1), dtype=np.float32))
    a.obsm["X_emb"] = emb
    sc.pp.neighbors(a, n_neighbors=n_neighbors, use_rep="X_emb", random_state=seed)
    best = None
    for r in resolutions:
        sc.tl.leiden(a, resolution=r, key_added="c", random_state=seed)
        cl = a.obs["c"].to_numpy()
        nmi = normalized_mutual_info_score(labels, cl, average_method="arithmetic")
        if best is None or nmi > best[1]:
            best = (r, nmi, adjusted_rand_score(labels, cl), len(set(cl)))
    asw = asw_label(emb, labels)
    nmi, ari = float(best[1]), float(best[2])
    return {"NMI": nmi, "ARI": ari, "ASW": asw, "AvgBIO": float(np.mean([nmi, ari, asw])),
            "best_resolution": best[0], "n_clusters": best[3]}
