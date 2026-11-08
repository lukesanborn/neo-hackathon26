"""Evaluations for experiments X1 (ijepa) / X2 (progmask). See attempts.md X-section.

1. ``retrieval_sets``: predicted-effect retrieval incl. inverse identification (src/retrieval.py) over saved
   ``test_predictions.npz`` of fine-tuned runs + baselines, plus extra in-memory prediction sets.
2. ``gene_embedding_linear``: Ahlmann-Eltze linear model whose perturbation embedding P is the
   pretrained gene-ID embedding of the target gene (no fine-tuning) -> asks directly whether the
   pretraining objective put knockdown-relevant information into the gene embeddings.
3. ``program_probe``: can a gene program's activity be read from the cell embedding when the
   program's own genes are removed from the input? (screenshot test; ridge CV R^2).
"""
from __future__ import annotations

import json
import os

import numpy as np

from src import retrieval as RT

N_GENES = 500


# ----------------------------------------------------------------------------- retrieval


def deltas_setup(ds):
    from src.data import CTRL

    cond = ds.conditions
    ctrl = np.asarray(ds.X[np.flatnonzero(cond == CTRL)]).mean(0)
    mean_of = lambda c: np.asarray(ds.X[np.flatnonzero(cond == c)]).mean(0)  # noqa: E731
    train = [c for c in ds.split["train"] if c != CTRL]
    test = list(ds.split["test"])
    R_all = np.stack([mean_of(c) - ctrl for c in train])
    genes = np.argsort(-R_all.var(0))[:N_GENES]
    T = np.stack([mean_of(c) - ctrl for c in test])[:, genes]
    return {"ctrl": ctrl, "train": train, "test": test, "genes": genes, "R": R_all[:, genes], "T": T}


def collect_run_sets(exp_dirs: list, test: list, ctrl, genes, name_filter=None) -> tuple[dict, list]:
    sets, skipped = {}, []
    for edir in exp_dirs:
        exp = os.path.basename(edir.rstrip("/"))
        for kind in ("runs", "baselines"):
            d = os.path.join(edir, kind)
            if not os.path.isdir(d):
                continue
            for r in sorted(os.listdir(d)):
                if kind == "runs" and name_filter and not name_filter(r):
                    continue
                path = os.path.join(d, r, "test_predictions.npz")
                if not os.path.exists(path):
                    continue
                z = np.load(path, allow_pickle=False)
                pmap = dict(zip([str(c) for c in z["conditions"]], z["preds"]))
                if any(c not in pmap for c in test):
                    skipped.append(f"{exp}/{kind}/{r}")
                    continue
                key = f"baselines/{r}" if kind == "baselines" else f"{exp}/runs/{r}"
                sets[key] = np.stack([pmap[c] - ctrl for c in test])[:, genes]
    return sets, skipped


def score_sets(P_sets: dict, T, R, pairs=(), n_boot: int = 2000) -> dict:
    sets = {}
    for k, P in P_sets.items():
        pg, S = RT.per_gene_scores(P, T, R)
        sets[k] = {"per_gene": pg, "S_pred": S}
    pg, S = RT.per_gene_scores(T, T, R)
    sets["(upper bound) true deltas"] = {"per_gene": pg, "S_pred": S}
    arms = RT.pool_seeds(sets)
    point, draws = RT.bootstrap_draws(arms, RT.true_sim(T), n_boot=n_boot, seed=0)
    cis = RT.summarize_ci(point, draws)
    diffs = [{"a": a, "b": b, **RT.diff_ci(point, draws, a, b)} for a, b in pairs if a in arms and b in arms]
    return RT.json_safe({"n_test": len(T), "arms": {a: {"members": arms[a]["members"], "ci": cis[a],
                                                        "per_gene": arms[a]["per_gene"]} for a in arms},
                         "diffs": diffs})


# ----------------------------------------------------------------------------- gene-embedding linear probe


def gene_embedding_table(ckpt_dir: str, gene_names: list, extra_aliases: dict | None = None):
    """LayerNorm(Embedding[gene]) as the pretrained model computes it (scGPT GeneEncoder).
    Returns (E (n_genes, d) with NaN rows for genes not in the vocab, found mask)."""
    import torch

    sd = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location="cpu")
    W = sd["encoder.embedding.weight"].float()
    ln_w, ln_b = sd.get("encoder.enc_norm.weight"), sd.get("encoder.enc_norm.bias")
    from src.models import PAD_TOKEN, SimpleVocab, gene_vocab_ids

    vocab = SimpleVocab(os.path.join(ckpt_dir, "vocab.json"))
    vid, _ = gene_vocab_ids(gene_names, vocab, extra_aliases)   # same alias rule as fine-tuning
    ids = [int(i) if int(i) != vocab[PAD_TOKEN] else -1 for i in vid]
    found = np.array([i >= 0 for i in ids])
    E = torch.zeros(len(ids), W.shape[1])
    idx = torch.tensor([i for i in ids if i >= 0])
    e = W[idx]
    if ln_w is not None:
        e = torch.nn.functional.layer_norm(e, (W.shape[1],), ln_w.float(), ln_b.float())
    E[torch.from_numpy(found)] = e
    E = E.numpy().astype(np.float64)
    E[~found] = np.nan
    return E, found


def gene_embedding_linear(ds, E: np.ndarray, conditions: list, K: int = 10, lam: float = 0.1) -> dict:
    """Ahlmann-Eltze linear model, G = training-pseudobulk PCA (as in baselines.linear_gene_embedding),
    P = pretrained gene embedding of the target gene reduced to K dims by a PCA fitted on the embeddings
    of the TRAINING targets only."""
    from src.baselines import fit_linear, train_pert_means

    tr_conds, M = train_pert_means(ds)
    Y = M.T.astype(np.float64)
    b = Y.mean(1, keepdims=True)
    Yc = Y - b
    U, _, _ = np.linalg.svd(Yc, full_matrices=False)
    G = U[:, :min(K, U.shape[1])]
    Etr = np.stack([E[ds.gene_pos(c)] for c in tr_conds])
    mu = Etr.mean(0)
    _, _, Vt = np.linalg.svd(Etr - mu, full_matrices=False)
    proj = lambda e: (e - mu) @ Vt[:K].T  # noqa: E731
    sd = proj(Etr).std(0).clip(1e-8)
    W = fit_linear(Yc, G, proj(Etr) / sd, lam)
    return {c: (G @ W @ (proj(E[ds.gene_pos(c)]) / sd) + b[:, 0]).astype(np.float32) for c in conditions}


# ----------------------------------------------------------------------------- program probe


def residualize(y: np.ndarray, covars: np.ndarray) -> np.ndarray:
    """y minus its OLS fit on [1, covars] (removes depth / complexity effects from program scores)."""
    X = np.column_stack([np.ones(len(y)), covars])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def program_scores(gi: np.ndarray, bi: np.ndarray, members: list) -> np.ndarray:
    """Activity of each program in one cell: sum of its expressed genes' bins / program size."""
    pos = {int(g): float(b) for g, b in zip(gi, bi)}
    return np.array([sum(pos.get(int(g), 0.0) for g in m) / max(len(m), 1) for m in members])


def ridge_cv_r2(Z: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0,
                alphas=(1e-2, 1e-1, 1, 10, 100, 1000)) -> float:
    """Max over alphas of the 5-fold CV R^2 (identical protocol for every arm)."""
    n = len(y)
    perm = np.random.default_rng(seed).permutation(n)
    fold = np.array_split(perm, folds)
    best = -np.inf
    for a in alphas:
        pred = np.zeros(n)
        for f in fold:
            tr = np.setdiff1d(perm, f)
            mu, sd = Z[tr].mean(0), Z[tr].std(0).clip(1e-8)
            X = (Z[tr] - mu) / sd
            ym = y[tr].mean()
            w = np.linalg.solve(X.T @ X + a * np.eye(X.shape[1]), X.T @ (y[tr] - ym))
            pred[f] = ((Z[f] - mu) / sd) @ w + ym
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum().clip(1e-12)
        best = max(best, r2)
    return float(best)
