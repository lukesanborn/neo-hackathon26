"""Latent-space retrieval (branch dangelo/latent-retrieval, attempts A26).

Instead of comparing predicted vs observed *expression* changes, compare them in a pretrained model's
*embedding* space: embed cells with an encoder (the <cls> output, scGPT zero-shot procedure from
src/track_a/pbmc.py), and define a perturbation's latent shift as

    shift(g) = mean L2-normalised embedding of g's cells  -  mean embedding of control cells.

  * true shift      : g's real measured perturbed cells
  * predicted shift : the fine-tuned model's predicted cells for g (one per shared control input)

Predicted cells are dense (every gene gets a value, no exact zeros) while encoders were trained on a cell's
EXPRESSED genes. Tokenisation rule for any expression row: keep its top-k genes by value, where k = the
number of expressed genes of the paired real control cell (for real cells: all expressed genes), then bin
those per cell exactly like real data. Binning is rank-based, so log1p vs raw scale does not matter.
"""
from __future__ import annotations

import numpy as np


def topk_csr(rows: np.ndarray, ks: np.ndarray):
    """Dense (n, G) expression rows -> CSR (indptr, indices, data) keeping each row's top-k positive genes.
    ks[i] = number of genes to keep for row i (capped at the row's positive entries)."""
    rows = np.asarray(rows, dtype=np.float32)
    indptr, indices, data = [0], [], []
    for r, k in zip(rows, np.asarray(ks)):
        pos = np.flatnonzero(r > 0)
        k = int(min(k, len(pos)))
        if k <= 0:
            keep = np.zeros(0, dtype=np.int64)
        elif k == len(pos):
            keep = pos
        else:
            keep = pos[np.argpartition(-r[pos], k - 1)[:k]]
        keep = np.sort(keep)
        indices.append(keep)
        data.append(r[keep])
        indptr.append(indptr[-1] + len(keep))
    return (np.asarray(indptr, dtype=np.int64), np.concatenate(indices) if indices else np.zeros(0, np.int64),
            np.concatenate(data).astype(np.float32) if data else np.zeros(0, np.float32))


def n_expressed(rows: np.ndarray) -> np.ndarray:
    return (np.asarray(rows) > 0).sum(1)


def embed_rows(model, vocab, conv, rows, ks, gene_symbols, device="cuda", max_genes=1200, seed=0):
    """L2-normalised <cls> embeddings (n, d) of dense expression rows under the top-k rule."""
    from src.track_a.pbmc import embed_cells

    indptr, indices, data = topk_csr(rows, ks)
    emb, _ = embed_cells(model, vocab, conv, indptr, indices, data, list(gene_symbols), max_genes=max_genes,
                         seed=seed, device=device)
    return emb


def shift(emb: np.ndarray, ctrl_mean: np.ndarray) -> np.ndarray:
    return np.asarray(emb).mean(0) - ctrl_mean


# ----------------------------------------------------------------------------- transforms (A27 diagnostic)


def embed_rows_both(model, vocab, conv, rows, ks, gene_symbols, device="cuda", max_genes=1200, seed=0, batch_size=64):
    """Per-cell (cls, token_mean) embeddings, each L2-normalised. token_mean = mean of the gene-token outputs
    (excluding <cls> and padding). Same tokenisation as embed_rows / src.track_a.pbmc.embed_cells."""
    import torch

    from src.track_a.pbmc import cell_tokens, column_vocab_ids, pad_batch

    indptr, indices, data = topk_csr(rows, ks)
    col_ids = column_vocab_ids(list(gene_symbols), vocab)
    rng = np.random.default_rng(seed)
    cls_id, pad_id = vocab["<cls>"], vocab["<pad>"]
    amp = dict(device_type="cuda" if str(device).startswith("cuda") else "cpu", dtype=torch.bfloat16,
               enabled=str(device).startswith("cuda"))
    out_c, out_t = [], []
    n = len(indptr) - 1
    for i in range(0, n, batch_size):
        cells = [cell_tokens(indices[indptr[r]:indptr[r + 1]], data[indptr[r]:indptr[r + 1]], col_ids, max_genes, rng)
                 for r in range(i, min(i + batch_size, n))]
        b = pad_batch(cells, cls_id, pad_id, conv["cls_value"], conv["pad_value"])
        g = torch.from_numpy(b["gene_ids"]).to(device)
        v = torch.from_numpy(b["values"]).to(device)
        p = torch.from_numpy(b["pad"]).to(device)
        with torch.no_grad(), torch.autocast(**amp):
            h = model.transformer_encoder(model.encoder(g) + model.value_encoder(v), src_key_padding_mask=p).float()
        keep = (~p).float()
        keep[:, 0] = 0.0
        tok = (h * keep.unsqueeze(-1)).sum(1) / keep.sum(1, keepdim=True).clamp_min(1.0)
        out_c.append(h[:, 0].cpu().numpy())
        out_t.append(tok.cpu().numpy())
    norm = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)  # noqa: E731
    return norm(np.concatenate(out_c)), norm(np.concatenate(out_t))


def fit_whiten(ctrl, shrink=0.1):
    """ZCA whitening fitted on control cells; returns f(E) = (E - mu) W. Shrinkage toward a scaled identity."""
    mu = ctrl.mean(0)
    C = np.cov(ctrl - mu, rowvar=False)
    C = (1 - shrink) * C + shrink * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    w, V = np.linalg.eigh(C)
    W = V @ np.diag(1.0 / np.sqrt(np.clip(w, 1e-12, None))) @ V.T
    return lambda E: (E - mu) @ W


def fit_remove_pcs(ctrl, k=10):
    """Project out the top-k principal directions of control-cell variation (cell-state axes)."""
    mu = ctrl.mean(0)
    _, _, Vt = np.linalg.svd(ctrl - mu, full_matrices=False)
    P = Vt[:k].T
    return lambda E: (E - mu) - (E - mu) @ P @ P.T


def fit_lda(X, y, n_dims=50, shrink=0.1):
    """Shrinkage LDA on TRAINING-knockdown cells (labels = knockdown); returns projection onto the top
    discriminant directions (generalised eigenproblem Sb v = lambda Sw v)."""
    from scipy.linalg import eigh

    classes = np.unique(y)
    mu = X.mean(0)
    d = X.shape[1]
    Sw, Sb = np.zeros((d, d)), np.zeros((d, d))
    for c in classes:
        Xc = X[y == c]
        mc = Xc.mean(0)
        Sw += (Xc - mc).T @ (Xc - mc)
        Sb += len(Xc) * np.outer(mc - mu, mc - mu)
    Sw /= len(X)
    Sb /= len(X)
    Sw = (1 - shrink) * Sw + shrink * np.trace(Sw) / d * np.eye(d)
    w, V = eigh(Sb, Sw)
    Vk = V[:, np.argsort(-w)[:min(n_dims, len(classes) - 1)]]
    return lambda E: (E - mu) @ Vk
