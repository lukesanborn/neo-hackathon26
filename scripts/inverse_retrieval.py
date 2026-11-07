#!/usr/bin/env python3
"""The inverse question: given a transcriptional profile, which gene was knocked down?

Forward task (the rest of our metrics): gene -> predicted response.
Inverse task (this script):            response -> which perturbation produced it?

Why it needs no separate model: you run the forward model over every candidate perturbation and
rank candidates by similarity to the query profile. The inverse problem is the forward model used
as a search index. That also means it scores a *different* property — not how accurate each
prediction is, but how **discriminable** predictions are from one another. A model whose
predictions are nearly identical across genes can have excellent RMSE and still be unable to
retrieve anything.

Two things are measured:

A. **Identifiability.** For each perturbation, how similar is its measured response to the most
   similar *other* perturbation? If real responses are near-duplicates, no model can tell them
   apart and the inverse task is ill-posed regardless of method quality.

B. **Retrieval accuracy.** Query with a held-out perturbation's measured response, search the
   predicted responses, and ask whether the correct gene comes back top-1 / top-5. Significance is
   a permutation test over the query-to-database pairing.

Both are reported with and without the shared "generic perturbation" component removed. That
component dominates cosine similarity (see docs/clustering_finding.md) and must be subtracted or
every profile looks alike.

Caveat on scale: a 22-candidate database is far easier than the thousands a real application would
search. Treat top-1 here as an upper bound on what the same model would achieve at realistic scale.

Usage
-----
    python scripts/inverse_retrieval.py reports/demo.html
    python scripts/inverse_retrieval.py <experiment>/report/demo_data.json
"""

from __future__ import annotations

import json
import sys

import numpy as np

N_PERM = 20_000
SEED = 0


def load_payload(path: str) -> dict:
    text = open(path).read()
    if path.endswith(".json"):
        return json.loads(text)
    i = text.index("const D = ") + len("const D = ")
    return json.loads(text[i:text.index("\nconst $ =", i)].rstrip().rstrip(";"))


def unit(M: np.ndarray) -> np.ndarray:
    return M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)


def main(path: str) -> int:
    D = load_payload(path)
    conds = [c["condition"] for c in D["conditions"]]
    methods = D["methods"]
    n = len(conds)
    OBS = np.array([c["obs_all"] for c in D["conditions"]], float)
    PRED = {m: np.array([c["pred_all"][m] for c in D["conditions"]], float) for m in methods}
    rng = np.random.default_rng(SEED)
    short = lambda i: conds[i].split("+")[0]

    print(f"{n} held-out perturbations\n")
    print("=== A. Identifiability — is the inverse task well-posed? ===")
    for label, M in (("raw", OBS), ("shared component removed", OBS - OBS.mean(0))):
        S = unit(M) @ unit(M).T
        np.fill_diagonal(S, -9.0)
        nn, who = S.max(1), S.argmax(1)
        k = int(np.argmax(nn))
        print(f"  {label:26s} median nearest-other {np.median(nn):.3f} | "
              f"{int((nn > 0.9).sum())}/{n} above 0.9 | worst pair "
              f"{short(k)}~{short(who[k])} {nn[k]:.3f}")
    print("  Confusable pairs cap what ANY method can retrieve.\n")

    Oc = OBS - OBS.mean(0)
    print("=== B. Retrieval — query a measured profile, search predicted profiles ===")
    print(f"  random: top-1 {1 / n:.0%}, top-5 {5 / n:.0%}\n")
    print(f"  {'method':16s} {'top-1':>7s} {'p':>9s}   {'top-5':>7s} {'p':>9s}")
    for m in methods:
        P = PRED[m]
        if np.linalg.norm(P - P.mean(0)) < 1e-9:
            print(f"  {m:16s} {'--':>7s} {'--':>9s}   {'--':>7s} {'--':>9s}   identical predictions "
                  f"— cannot retrieve")
            continue
        C = unit(Oc) @ unit(P - P.mean(0)).T
        rank = (-C).argsort(1)
        t1 = float(np.mean([rank[i, 0] == i for i in range(n)]))
        t5 = float(np.mean([i in rank[i, :5] for i in range(n)]))
        n1 = np.empty(N_PERM)
        n5 = np.empty(N_PERM)
        for t in range(N_PERM):
            p = rng.permutation(n)
            n1[t] = np.mean([rank[i, 0] == p[i] for i in range(n)])
            n5[t] = np.mean([p[i] in rank[i, :5] for i in range(n)])
        p1 = (int((n1 >= t1).sum()) + 1) / (N_PERM + 1)
        p5 = (int((n5 >= t5).sum()) + 1) / (N_PERM + 1)
        print(f"  {m:16s} {t1:>6.0%} {p1:>9.4f}   {t5:>6.0%} {p5:>9.4f}")
    print("\n  Retrieval rewards DISCRIMINABILITY between predictions, which error metrics do not.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
