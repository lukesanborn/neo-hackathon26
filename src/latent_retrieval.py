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
