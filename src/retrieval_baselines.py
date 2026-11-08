"""Stronger, training-free-of-test baselines for the retrieval test (numpy only; A24).

Both predict an unseen gene's delta from a *description of the target gene* — here scGPT_human's
pretrained gene-token embedding — fitted on TRAINING perturbations only; their one hyperparameter is
chosen on VALIDATION perturbations by identification percentile (id_pct).
  * ridge: delta = mean_train_delta + (e_g - mean_e) @ W,  W = ridge(E_train -> R_train), alpha on val
  * knn:   delta = mean of the k training deltas whose target embeddings are most cosine-similar, k on val
"""
import numpy as np

from .retrieval import id_scores, unit

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
KS = (1, 3, 5, 10, 20)


def ridge_fit(E, R, alpha):
    mu_e, mu_r = E.mean(0), R.mean(0)
    Ec = E - mu_e
    W = np.linalg.solve(Ec.T @ Ec + alpha * np.eye(E.shape[1]), Ec.T @ (R - mu_r))
    return lambda Eq: mu_r + (Eq - mu_e) @ W


def knn_fit(E, R, k):
    Eu = unit(E)

    def predict(Eq):
        S = unit(Eq) @ Eu.T
        idx = np.argsort(-S, axis=1)[:, :k]
        return R[idx].mean(1)
    return predict


def select_and_predict(E_train, R_train, E_val, T_val, E_eval, kind):
    """Pick the hyperparameter on validation id_pct, refit on training perturbations, predict E_eval.
    Returns (predictions for E_eval, chosen hyperparameter, {hp: val id_pct})."""
    grid, fit = (ALPHAS, ridge_fit) if kind == "ridge" else (KS, knn_fit)
    grid = [g for g in grid if kind == "ridge" or g <= len(E_train)]
    scores = {g: id_scores(fit(E_train, R_train, g)(E_val), T_val)["id_pct"] for g in grid}
    best = max(grid, key=lambda g: scores[g])
    return fit(E_train, R_train, best)(E_eval), best, scores
