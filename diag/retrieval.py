"""A16: predicted-effect retrieval — scoring Task 3 predictions the way a perturbation-selection tool would use them.

For every saved Adamson test prediction set (runs + baselines), with deltas = profile − mean of all control cells:
  * ref_spearman / ref_p@5: rank the 58 measured TRAINING perturbations by cosine similarity to the predicted
    delta of a held-out gene, and compare with the ranking by its TRUE delta (Spearman; top-5 overlap / 5).
  * cluster_agreement: Pearson correlation between the upper triangles of the cosine-similarity matrices of
    predicted vs true deltas among the 22 held-out genes (do predictions cluster the genes the right way?).
  * id_top1 / id_mean_rank: is each prediction closest (cosine) to its own gene's true delta among the 22
    held-out genes? Chance top-1 = 1/22 ≈ 0.045.
Genes: the 500 genes with the largest variance of TRAINING-perturbation mean deltas (no test information).
A constant prediction (training mean) scores near chance on cluster/identification by construction.

Run from the repo root:  .venv/bin/modal run diag/retrieval.py
"""
import json
import os
import sys

import modal

sys.path.insert(0, os.getcwd())
from environment.image import build_image  # noqa: E402

app = modal.App("jepa-retrieval")
vol = modal.Volume.from_name("jepa-perturb")
image = build_image("hopper")
CACHE = "/vol/data/adamson_gears002_simulation_seed1"
EXPS = ["adamson_trackA_lr3e5", "adamson_celljepa_public_v1", "adamson_trackB_v1"]
N_GENES = 500


def _scores(P, T, R):
    """P, T: (n_test, g) predicted / true deltas; R: (n_ref, g) true training deltas."""
    import numpy as np
    from scipy.stats import rankdata, spearmanr

    def unit(A):
        return A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)

    Pu, Tu, Ru = unit(P), unit(T), unit(R)
    sp, p5 = [], []
    for i in range(len(P)):
        a, b = Ru @ Pu[i], Ru @ Tu[i]
        sp.append(spearmanr(a, b).statistic if np.std(a) > 1e-12 else np.nan)
        p5.append(len(set(np.argsort(-a)[:5]) & set(np.argsort(-b)[:5])) / 5)
    iu = np.triu_indices(len(P), 1)
    sp_mat, st_mat = (Pu @ Pu.T)[iu], (Tu @ Tu.T)[iu]
    clus = float(np.corrcoef(sp_mat, st_mat)[0, 1]) if np.std(sp_mat) > 1e-9 else float("nan")
    S = Pu @ Tu.T                                        # prediction i vs true j
    ranks = np.array([rankdata(-S[i])[i] for i in range(len(P))])
    return {"ref_spearman": float(np.nanmean(sp)), "ref_p@5": float(np.mean(p5)), "cluster_agreement": clus,
            "id_top1": float(np.mean(ranks == 1)), "id_mean_rank": float(np.mean(ranks)), "n_test": len(P)}


@app.function(image=image, volumes={"/vol": vol}, cpu=4, memory=16384, timeout=1800)
def run() -> dict:
    import numpy as np

    from src.data import CTRL, load_cache

    ds = load_cache(CACHE, mmap=True)
    cond = ds.conditions
    ctrl = np.asarray(ds.X[np.flatnonzero(cond == CTRL)]).mean(0)
    mean_of = lambda c: np.asarray(ds.X[np.flatnonzero(cond == c)]).mean(0)  # noqa: E731
    train = [c for c in ds.split["train"] if c != CTRL]
    test = list(ds.split["test"])
    R_all = np.stack([mean_of(c) - ctrl for c in train])
    genes = np.argsort(-R_all.var(0))[:N_GENES]
    R = R_all[:, genes]
    T_full = {c: mean_of(c) - ctrl for c in test}
    out = {}
    for exp in EXPS:
        base = f"/vol/experiments/{exp}"
        sets = []
        for kind in ("runs", "baselines"):
            d = os.path.join(base, kind)
            if os.path.isdir(d):
                sets += [(kind, r, os.path.join(d, r, "test_predictions.npz")) for r in sorted(os.listdir(d))]
        for kind, r, path in sets:
            if not os.path.exists(path):
                continue
            z = np.load(path, allow_pickle=False)
            pmap = dict(zip([str(c) for c in z["conditions"]], z["preds"]))
            cs = [c for c in test if c in pmap]
            P = np.stack([pmap[c] - ctrl for c in cs])[:, genes]
            T = np.stack([T_full[c] for c in cs])[:, genes]
            out[f"{exp}/{kind}/{r}"] = _scores(P, T, R)
    # oracle-ish reference: the true held-out deltas themselves (upper bound = 1 on ref/cluster metrics)
    out["(upper bound) true deltas"] = _scores(np.stack([T_full[c] for c in test])[:, genes],
                                               np.stack([T_full[c] for c in test])[:, genes], R)
    return out


@app.local_entrypoint()
def main():
    import re

    res = run.remote()
    os.makedirs("diag/results", exist_ok=True)
    json.dump(res, open("diag/results/A16_retrieval_raw.json", "w"), indent=1)
    groups = {}
    for k, v in res.items():
        g = re.sub(r"_s\d+(_|$)", r"\1", k)          # pool seeds
        groups.setdefault(g, []).append(v)
    keys = ["ref_spearman", "ref_p@5", "cluster_agreement", "id_top1", "id_mean_rank"]
    lines = ["| Predictions | n | " + " | ".join(keys) + " |", "|---" * (len(keys) + 2) + "|"]
    for g, vs in sorted(groups.items()):
        import math
        cells = []
        for k in keys:
            xs = [x[k] for x in vs if not (isinstance(x[k], float) and math.isnan(x[k]))]
            cells.append(f"{sum(xs) / len(xs):.3f}" if xs else "undef")
        lines.append(f"| {g} | {len(vs)} | " + " | ".join(cells) + " |")
    print("\n".join(lines))
    with open("diag/results/A16_retrieval.md", "w") as f:
        f.write("\n".join(lines) + "\n")
