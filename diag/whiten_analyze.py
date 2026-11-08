"""CPU analysis for diag/whiten_check.py: does a linear readout recover knockdown identity from JEPA embeddings?

Split-half ceiling (as A26): each held-out gene's real cells are split in two halves; the latent shift of half A
must be closest (cosine) to half B of the SAME gene among all held-out genes. Readouts, all fit WITHOUT held-out
genes:
  raw          shift = mean(emb) - control mean                                   (A26)
  ctrl_pcK     project out the top-K principal directions of control cells           (unsupervised)
  ctrl_white   ZCA-whiten with the control covariance, shrinkage lam                 (unsupervised)
  within_white ZCA-whiten with the pooled WITHIN-knockdown covariance of TRAINING genes (uses train labels)
  lda          within_white, then keep the top-50 directions separating TRAINING-gene means (uses train labels)
Hyperparameters (K, lam) are chosen per space on VALIDATION genes (id_pct); test genes are scored once.

    python diag/whiten_analyze.py <dir with whiten_{pub,bscgpt,bjepa}.npz>
"""
import json
import os
import sys

import numpy as np

SPACES = {"pub": "public scGPT", "bscgpt": "blood-scGPT", "bjepa": "blood-JEPA (center)"}
LAMS = (1e-3, 1e-2, 1e-1, 1.0)
KS = (1, 3, 10, 30)


def zca(C, lam):
    d = len(C)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 0, None) + lam * w.mean()
    return V @ np.diag(w ** -0.5) @ V.T


def halves(E, lab, n_cond):
    A, B = [], []
    for j in range(n_cond):
        e = E[lab == j]
        h = len(e) // 2
        A.append(e[:h].mean(0))
        B.append(e[h:].mean(0))
    return np.stack(A), np.stack(B)


def ranks(P, T):
    Pu = P / np.linalg.norm(P, axis=1, keepdims=True).clip(1e-12)
    Tu = T / np.linalg.norm(T, axis=1, keepdims=True).clip(1e-12)
    S = Pu @ Tu.T
    return np.array([(S[i] > S[i, i]).sum() + 1 for i in range(len(S))], dtype=float)


def summ(r):
    n = len(r)
    return {"id_pct": float(np.mean(1 - (r - 1) / (n - 1))), "top1": float(np.mean(r == 1)),
            "top3": float(np.mean(r <= 3)), "top10": float(np.mean(r <= 10))}


def boot(ra, rb=None, n=2000, seed=0):
    """Bootstrap CI over genes for id_pct of ranks ra (or of ra - rb, paired)."""
    rng = np.random.default_rng(seed)
    m = len(ra)
    f = lambda r: 1 - (r - 1) / (m - 1)  # noqa: E731
    x = f(ra) if rb is None else f(ra) - f(rb)
    d = np.array([x[rng.integers(0, m, m)].mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def readouts(z):
    ctrl = z["ctrl"].astype(np.float64)
    fit, mu = ctrl[: len(ctrl) // 2], ctrl[: len(ctrl) // 2].mean(0)
    Cc = np.cov(fit, rowvar=False)
    # within-knockdown covariance over TRAINING genes only
    Et, lt = z["train_emb"].astype(np.float64), z["train_lab"]
    means = np.stack([Et[lt == j].mean(0) for j in range(lt.max() + 1)])
    Cw = np.cov(Et - means[lt], rowvar=False)
    _, Vc = np.linalg.eigh(Cc)
    R = {"raw": [lambda X: X - mu]}
    R["ctrl_pcK"] = [(lambda K: (lambda X: (X - mu) - (X - mu) @ Vc[:, -K:] @ Vc[:, -K:].T))(K) for K in KS]
    R["ctrl_white"] = [(lambda W: (lambda X: (X - mu) @ W))(zca(Cc, lam)) for lam in LAMS]
    R["within_white"] = [(lambda W: (lambda X: (X - mu) @ W))(zca(Cw, lam)) for lam in LAMS]
    lda = []
    for lam in LAMS:
        W = zca(Cw, lam)
        M = (means - mu) @ W
        M = M - M.mean(0)
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        P = W @ Vt[:50].T
        lda.append((lambda P: (lambda X: (X - mu) @ P))(P))
    R["lda"] = lda
    return R


def main(d, names=None):
    global SPACES
    out_name = "A27_whiten"
    if names:
        SPACES = {n: n for n in names}
        out_name = "A28_ceiling"
    res, test_ranks = {}, {}
    hp = {"raw": ["-"], "ctrl_pcK": [f"K={k}" for k in KS], "ctrl_white": [f"lam={l:g}" for l in LAMS],
          "within_white": [f"lam={l:g}" for l in LAMS], "lda": [f"lam={l:g}" for l in LAMS]}
    for sp in SPACES:
        z = np.load(os.path.join(d, f"whiten_{sp}.npz"))
        R = readouts(z)
        H = {s: halves(z[f"{s}_emb"].astype(np.float64), z[f"{s}_lab"], len(z[f"{s}_conds"])) for s in ("val", "test")}
        res[sp] = {}
        for name, fns in R.items():
            vals = [summ(ranks(f(H["val"][0]), f(H["val"][1])))["id_pct"] for f in fns]
            b = int(np.argmax(vals))
            f = fns[b]
            rt = ranks(f(H["test"][0]), f(H["test"][1]))
            test_ranks[(sp, name)] = rt
            res[sp][name] = {"chosen": hp[name][b], "val_id_pct": vals[b], "val_all": vals,
                             "test": summ(rt), "test_ci": boot(rt)}
        res[sp]["n_val"], res[sp]["n_test"] = len(z["val_conds"]), len(z["test_conds"])
    diffs = []
    keys = list(SPACES)
    pairs = ((("bjepa", "bscgpt"), ("bjepa", "pub")) if not names else tuple((k, keys[0]) for k in keys[1:]))
    for name in ("raw", "ctrl_pcK", "ctrl_white", "within_white", "lda"):
        for a, b in pairs:
            diffs.append({"readout": name, "a": a, "b": b, "ci": boot(test_ranks[(a, name)], test_ranks[(b, name)])})
    for sp in SPACES:
        for name in ("ctrl_white", "within_white", "lda"):
            diffs.append({"readout": f"{name} - raw", "a": sp, "b": sp,
                          "ci": boot(test_ranks[(sp, name)], test_ranks[(sp, "raw")])})
    json.dump({"res": res, "diffs": diffs}, open(f"diag/results/{out_name}.json", "w"), indent=1)

    L = ["# A27: is JEPA's knockdown signal squashed or gone? (real Replogle cells, split-half ceiling)", "",
         "Readouts fit on control cells or TRAINING genes only; hyperparameter chosen on validation genes; test scored once.",
         f"Test: {res[keys[0]]['n_test']} genes (chance id_pct 0.5, top-1 {1/res[keys[0]]['n_test']:.3f}). "
         "CIs: bootstrap over test genes.", "",
         "| Space | Readout | chosen | val id_pct | test id_pct [95% CI] | test top-1 | top-3 | top-10 |",
         "|---|---|---|---|---|---|---|---|"]
    for sp, nm in SPACES.items():
        for name in ("raw", "ctrl_pcK", "ctrl_white", "within_white", "lda"):
            r = res[sp][name]
            m, lo, hi = r["test_ci"]
            t = r["test"]
            L.append(f"| {nm} | {name} | {r['chosen']} | {r['val_id_pct']:.3f} | {m:.3f} [{lo:.3f}, {hi:.3f}] | "
                     f"{t['top1']:.3f} | {t['top3']:.3f} | {t['top10']:.3f} |")
    L += ["", "## Paired differences in test id_pct (A − B) [95% CI]", "", "| readout | A | B | diff |", "|---|---|---|---|"]
    for x in diffs:
        m, lo, hi = x["ci"]
        L.append(f"| {x['readout']} | {SPACES[x['a']]} | {SPACES[x['b']]} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    open(f"diag/results/{out_name}.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else None)
