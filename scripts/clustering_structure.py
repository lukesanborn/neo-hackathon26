#!/usr/bin/env python3
"""Does a model's predicted perturbation responses carry *functional* information?

Motivation: low prediction error does not imply usefulness for the clustering / function-assignment
use case. The training-mean baseline has competitive delta RMSE while predicting an identical
profile for every gene — useless for clustering. This script measures the property clustering
actually needs: do predicted responses place perturbations in the same *relative* arrangement as
measured responses do?

Method
------
For each method, build the n x n cosine-similarity matrix of per-perturbation response vectors
(change from control), for predicted and measured responses. Compare the two matrices with a
**Mantel permutation test** (permute perturbation labels; the pairwise entries are not independent,
so an ordinary Pearson p-value would be far too optimistic).

Both matrices are computed after removing the *shared component* (the mean response across
perturbations). Without this, cosine similarity is dominated by a generic "something was perturbed"
signature that every prediction shares, which swamps the gene-specific part. The raw (uncentred)
figure is reported too, since that is what naive clustering would see.

Ceiling: measured-vs-measured on disjoint halves of the gene set — how much structure is
recoverable at all given the data's own noise.

Usage
-----
    python scripts/clustering_structure.py reports/demo_data.json
    python scripts/clustering_structure.py reports/demo.html      # parses the embedded payload

`demo_data.json` / `demo.html` are written by `modal_app.py --stage aggregate` into
<experiment>/report/ on the Volume. Read-only; nothing is modified.
"""

from __future__ import annotations

import json
import sys

import numpy as np

N_PERM = 20_000
SEED = 0


def load_payload(path: str) -> dict:
    """Accept demo_data.json directly, or demo.html with the payload substituted in."""
    text = open(path).read()
    if path.endswith(".json"):
        return json.loads(text)
    marker = "const D = "
    i = text.index(marker) + len(marker)
    j = text.index("\nconst $ =", i)
    return json.loads(text[i:j].rstrip().rstrip(";"))


def unit(M: np.ndarray) -> np.ndarray:
    return M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)


def sim(M: np.ndarray) -> np.ndarray:
    U = unit(M)
    return U @ U.T


def shared_fraction(M: np.ndarray) -> float:
    """Mean squared cosine between each response and the across-perturbation mean response."""
    mu = M.mean(0)
    nm = np.linalg.norm(mu)
    if nm < 1e-12:
        return float("nan")
    return float(np.mean([(np.dot(r, mu) / (np.linalg.norm(r) * nm)) ** 2 for r in M]))


def mantel(A: np.ndarray, B: np.ndarray, rng: np.random.Generator, n_perm: int = N_PERM):
    """One-sided Mantel test: P(permuted r >= observed r). Permutes perturbation labels."""
    n = len(A)
    iu = np.triu_indices(n, 1)
    b = B[iu]
    r0 = float(np.corrcoef(A[iu], b)[0, 1])
    hits = sum(1 for _ in range(n_perm)
               if np.corrcoef(A[np.ix_(p := rng.permutation(n), p)][iu], b)[0, 1] >= r0)
    return r0, (hits + 1) / (n_perm + 1)


def main(path: str) -> int:
    D = load_payload(path)
    methods = D["methods"]
    n = len(D["conditions"])
    iu = np.triu_indices(n, 1)
    rng = np.random.default_rng(SEED)

    OBS = np.array([c["obs_all"] for c in D["conditions"]], float)
    PRED = {m: np.array([c["pred_all"][m] for c in D["conditions"]], float) for m in methods}

    s_obs = sim(OBS)[iu]
    print(f"{n} held-out perturbations, {OBS.shape[1]} genes\n")
    print("MEASURED responses")
    print(f"  pairwise cosine: mean {s_obs.mean():+.3f}  sd {s_obs.std():.3f}  "
          f"{int((s_obs < 0).sum())}/{len(s_obs)} pairs anti-correlated")
    print(f"  shared 'generic response' component: {shared_fraction(OBS) * 100:.0f}% of a typical response\n")

    Sc_obs = sim(OBS - OBS.mean(0))
    print(f"{'method':16s} {'shared':>7s} {'raw r':>7s} {'centred r':>10s} {'p':>8s}   verdict")
    for m in methods:
        P = PRED[m]
        if np.linalg.norm(P - P.mean(0)) < 1e-9:
            print(f"{m:16s} {'100%':>7s} {'--':>7s} {'--':>10s} {'--':>8s}   identical profile for every "
                  f"perturbation — no clustering value")
            continue
        r_raw = float(np.corrcoef(sim(P)[iu], s_obs)[0, 1])
        r_cen, p = mantel(sim(P - P.mean(0)), Sc_obs, rng)
        verdict = "signal" if p < 0.05 else ("suggestive" if p < 0.15 else "chance")
        print(f"{m:16s} {shared_fraction(P) * 100:6.0f}% {r_raw:+7.3f} {r_cen:+10.3f} {p:8.4f}   {verdict}")

    half = OBS.shape[1] // 2
    perm = rng.permutation(OBS.shape[1])
    A, B = OBS[:, perm[:half]], OBS[:, perm[half:]]
    r_c, p_c = mantel(sim(A - A.mean(0)), sim(B - B.mean(0)), rng)
    print(f"\n{'CEILING':16s} {'':>7s} {'':>7s} {r_c:+10.3f} {p_c:8.4f}   measured vs measured, "
          f"disjoint gene halves")
    print("\nInterpretation: 'centred r' as a fraction of the ceiling is the share of recoverable\n"
          "functional structure the model captures. 'raw r' is what naive clustering would see.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
