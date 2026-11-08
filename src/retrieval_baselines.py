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


# ----------------------------------------------------------------------------- A28: linear + contrastive loss


WDS = (1e-4, 1e-3, 1e-2, 1e-1)


def contrastive_linear_fit(E, R, wd, retr_weight=1.0, tau=0.1, epochs=400, lr=1e-3, seed=0):
    """Linear map gene-embedding -> full delta, trained like the fine-tuned models' extra loss:
    MSE(pred, R) + retr_weight * InfoNCE(cos(pred_i, R_j) / tau, target i), on TRAINING perturbations only."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    Et = torch.as_tensor(E, dtype=torch.float32)
    Rt = torch.as_tensor(R, dtype=torch.float32)
    mu_e = Et.mean(0)
    lin = torch.nn.Linear(Et.shape[1], Rt.shape[1])
    with torch.no_grad():
        lin.bias.copy_(Rt.mean(0))
        lin.weight.zero_()
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    tgt = torch.arange(len(Et))
    Ru = F.normalize(Rt, dim=-1)
    for _ in range(epochs):
        P = lin(Et - mu_e)
        loss = F.mse_loss(P, Rt) + retr_weight * F.cross_entropy(F.normalize(P, dim=-1) @ Ru.T / tau, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lambda Eq: lin(torch.as_tensor(Eq, dtype=torch.float32) - mu_e).detach().numpy()


def contrastive_linear_select(E_train, R_train, E_val, T_val_panel, panel, E_eval):
    """Weight decay chosen on validation id_pct (scored on the gene panel); returns eval predictions (full genes)."""
    scores, fits = {}, {}
    for wd in WDS:
        f = contrastive_linear_fit(E_train, R_train, wd)
        fits[wd] = f
        scores[wd] = id_scores(f(E_val)[:, panel], T_val_panel)["id_pct"]
    best = max(WDS, key=lambda w: scores[w])
    return fits[best](E_eval), best, scores


def pca_gene_embedding(R_train, K=10):
    """Ahlmann-Eltze gene embedding: top-K left singular vectors of centred training responses (genes x perts)."""
    Y = np.asarray(R_train, dtype=np.float64).T
    U, _, _ = np.linalg.svd(Y - Y.mean(1, keepdims=True), full_matrices=False)
    return U[:, :K]
