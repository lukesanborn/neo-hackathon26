#!/usr/bin/env python3
"""precision@k — does the model pick the right perturbations to validate?

The decision this scores
------------------------
Arrayed validation (Stage 3 in docs/use_case_map.md) is robotics-bound: a lab can follow up
~20–100 perturbations, never thousands. So the economically relevant question is not "how
accurate is each predicted value" but **"of the k perturbations the model ranks highest, how many
are real hits?"** Every wrong pick is a wasted validation well.

This differs from delta RMSE in both directions. A model with poor RMSE can rank perfectly (a
systematic bias shifts every value but preserves order), and a model with good RMSE can rank badly
(accurate on the many genes that barely move, wrong about the few that matter).

Ranking criterion
-----------------
Perturbations are ranked by predicted effect magnitude — the L2 norm of the predicted change from
control — against the same quantity measured. In words: *which knockdowns actually do something?*
A "hit" is defined as genuinely belonging to the top k by measured magnitude.

Other criteria matter for other questions (similarity to a target signature; membership of a
pathway cluster). Magnitude is the generic one and needs no extra inputs.

Baselines that cannot rank at all are reported as such rather than scored: a method predicting the
same profile for every perturbation produces an all-tied ordering, which is not a ranking. That is
the point of the metric — such a method may still post a competitive RMSE.

Usage
-----
    python scripts/precision_at_k.py reports/demo.html
    python scripts/precision_at_k.py <experiment>/report/demo_data.json [--k 5,10,20]

Read-only. Significance is a permutation test against random selection of k.
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


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    r = lambda v: np.argsort(np.argsort(v))
    return float(np.corrcoef(r(a), r(b))[0, 1])


def main(path: str, ks: list[int]) -> int:
    D = load_payload(path)
    methods = D["methods"]
    conds = [c["condition"] for c in D["conditions"]]
    n = len(conds)
    OBS = np.array([c["obs_all"] for c in D["conditions"]], float)
    PRED = {m: np.array([c["pred_all"][m] for c in D["conditions"]], float) for m in methods}
    rng = np.random.default_rng(SEED)

    true_mag = np.linalg.norm(OBS, axis=1)
    order = np.argsort(-true_mag)
    short = lambda i: conds[i].split("+")[0]
    print(f"{n} held-out perturbations\n")
    print("  largest measured effects :", ", ".join(short(i) for i in order[:5]))
    print("  smallest measured effects:", ", ".join(short(i) for i in order[-5:]), "\n")

    for K in ks:
        if K >= n:
            continue
        hits = set(order[:K].tolist())
        base = K / n
        print(f"--- precision@{K}: pick {K} of {n} for validation (random = {base:.0%}) ---")
        for m in methods:
            mag = np.linalg.norm(PRED[m], axis=1)
            if np.ptp(mag) < 1e-9:
                print(f"  {m:16s}  CANNOT RANK — identical predicted magnitude for every "
                      f"perturbation")
                continue
            got = len(set(np.argsort(-mag)[:K].tolist()) & hits)
            p = got / K
            null = np.array([len(set(rng.choice(n, K, replace=False).tolist()) & hits) / K
                             for _ in range(N_PERM)])
            pv = (int((null >= p).sum()) + 1) / (N_PERM + 1)
            flag = "  <-- better than chance" if pv < 0.05 else ""
            print(f"  {m:16s}  {got}/{K} = {p:>4.0%}   (p={pv:.3f}){flag}")
        print()

    print("--- Spearman rank correlation across all perturbations ---")
    for m in methods:
        mag = np.linalg.norm(PRED[m], axis=1)
        if np.ptp(mag) < 1e-9:
            print(f"  {m:16s}  undefined (all tied)")
        else:
            print(f"  {m:16s}  rho = {spearman(mag, true_mag):+.3f}")
    print("\nA method that cannot rank scores nothing here however good its RMSE — which is the\n"
          "whole reason this metric exists alongside the error metrics.")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    kv = {a.split("=")[0]: a.split("=")[1] for a in sys.argv[1:] if "=" in a and a.startswith("--")}
    if len(args) != 1:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(args[0], [int(x) for x in kv.get("--k", "5,10").split(",")]))
