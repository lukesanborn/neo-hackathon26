"""Simple baselines. All are fit on TRAINING interventions only (no val/test outcomes)."""

from __future__ import annotations

import numpy as np

from .data import CTRL


def train_pert_means(ds) -> tuple[list[str], np.ndarray]:
    conds = [c for c in ds.split["train"] if c != CTRL]
    return conds, np.stack([ds.observed_mean(c) for c in conds])  # (n_perts, n_genes)


def no_change(ds, conditions, ctrl_ref):
    return {c: ctrl_ref.copy() for c in conditions}


def training_mean(ds, conditions, ctrl_ref):
    _, M = train_pert_means(ds)
    mean = M.mean(0)  # equal weight per training intervention, not per cell
    return {c: mean.copy() for c in conditions}


def fit_linear(Y: np.ndarray, G: np.ndarray, P: np.ndarray, lam: float) -> np.ndarray:
    """W = argmin ||Y - G W P^T||^2 + lam ||W||^2 (closed form used by Ahlmann-Eltze et al.).

    Y: (n_genes, n_perts) centred training responses; G: (n_genes, K) gene embedding;
    P: (n_perts, K) perturbation embedding.
    """
    K = G.shape[1]
    left = np.linalg.solve(G.T @ G + lam * np.eye(K), G.T)          # (K, n_genes)
    right = P @ np.linalg.inv(P.T @ P + lam * np.eye(P.shape[1]))     # (n_perts, K)
    return left @ Y @ right                                           # (K, K)


def linear_gene_embedding(ds, conditions, ctrl_ref, K: int = 10, lam: float = 0.1):
    """Port of the 'linear model with training-data gene embedding' from
    Ahlmann-Eltze, Huber & Anders (Nat. Methods 2025; const-ae/linear_perturbation_prediction-Paper).

    * Y_train = pseudobulk of training interventions (genes x perts), centred by its row mean b.
    * G = top-K principal components (left singular vectors) of Y_train -> gene embedding.
    * P = rows of G for each intervention's target gene -> perturbation embedding. This works
      for unseen interventions because the target gene is a measured gene with an embedding
      derived from training data only (no external knowledge).
    * Prediction for unseen p: G W G[target(p)]^T + b.
    """
    tr_conds, M = train_pert_means(ds)
    Y = M.T.astype(np.float64)                     # (n_genes, n_perts)
    b = Y.mean(1, keepdims=True)
    Yc = Y - b
    U, _, _ = np.linalg.svd(Yc, full_matrices=False)
    K = min(K, U.shape[1])
    G = U[:, :K]
    P_train = np.stack([G[ds.gene_pos(c)] for c in tr_conds])
    W = fit_linear(Yc, G, P_train, lam)
    preds = {}
    for c in conditions:
        p = G[ds.gene_pos(c)]
        preds[c] = (G @ W @ p + b[:, 0]).astype(np.float32)
    info = {"K": K, "lambda": lam, "n_train_perts": len(tr_conds),
            "external_knowledge": "none; gene embedding = PCA of training pseudobulk"}
    return preds, info


BASELINES = {
    "no_change": no_change,
    "training_mean": training_mean,
}
