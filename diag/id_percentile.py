"""Per-gene identification percentile: for each held-out observed response, the percentile of its true gene among
all held-out candidates ranked by cosine(predicted_i, observed_j) (100 = ranked first, 0 = last). Raw and centred
(mean training delta removed). Per seed, then averaged over seeds per gene.
  ../.venv/bin/modal run diag/id_percentile.py"""
import json
import os
import sys

import modal

sys.path.insert(0, os.getcwd())
from environment.image import build_image  # noqa: E402

app = modal.App("jepa-idpct")
vol = modal.Volume.from_name("jepa-perturb")
MODE = {(True, False): "inverse-centred", (False, False): "inverse-raw", (False, True): "forward-raw"}
ARMS = {"X1 ijepa": "jepa_lr3e5_ijepa", "X2 progmask": "jepa_lr3e5_progmask", "kidney scGPT": "scgpt_lr3e5",
        "kidney JEPA token": "jepa_lr3e5_token", "kidney JEPA faithful": "jepa_lr3e5"}


@app.function(image=build_image("hopper"), volumes={"/vol": vol}, cpu=4, memory=32768, timeout=1800)
def run(exp: str, methods: list) -> dict:
    import numpy as np

    from src.data import load_cache
    from src.retrieval import unit
    from src.xjepa_eval import deltas_setup

    cfg = json.load(open(f"/vol/experiments/{exp}/experiment.json"))
    ds = load_cache(cfg["cache_dir"], mmap=True)
    d = deltas_setup(ds)
    T, test, mu = d["T"], d["test"], d["R"].mean(0)
    n = len(test)

    def pct(P, center, forward=False):
        A, B = (P - mu, T - mu) if center else (P, T)
        S = unit(A) @ unit(B).T if forward else unit(B) @ unit(A).T   # forward: prediction i vs true j
        out = []
        for j in range(n):
            b = np.sum(S[j] > S[j, j] + 1e-9)
            m = np.sum(np.abs(S[j] - S[j, j]) <= 1e-9)
            ranks = np.arange(b + 1, b + m + 1)        # random tie-breaking, in expectation
            out.append((100 * np.mean((n - ranks) / (n - 1)), np.mean(ranks <= 1), np.mean(ranks <= 3)))
        return np.array(out)                           # (n, 3): percentile, top1, top3

    def load(path):
        z = np.load(path)
        pm = dict(zip([str(c) for c in z["conditions"]], z["preds"]))
        return np.stack([pm[c] - d["ctrl"] for c in test])[:, d["genes"]]

    res = {}
    rdir = f"/vol/experiments/{exp}/runs"
    for name, tag in ARMS.items():
        for m in methods:
            runs = sorted(r for r in os.listdir(rdir) if r.startswith(f"{m}_s") and r.endswith(f"_{tag}")
                          and os.path.exists(f"{rdir}/{r}/test_predictions.npz"))
            if not runs:
                continue
            Ps = [load(f"{rdir}/{r}/test_predictions.npz") for r in runs]
            for c, fw in ((True, False), (False, False), (False, True)):
                per = np.mean([pct(P, c, fw) for P in Ps], axis=0)
                res[f"{name} | {m} | {MODE[(c, fw)]}"] = {"seeds": len(runs), "per_gene": per.tolist()}
    lin = load(f"/vol/experiments/{exp}/baselines/linear/test_predictions.npz")
    for c, fw in ((True, False), (False, False), (False, True)):
        res[f"linear | - | {MODE[(c, fw)]}"] = {"seeds": 1, "per_gene": pct(lin, c, fw).tolist()}
    return {"n": n, "test": test, "res": res}


@app.local_entrypoint()
def main():
    import numpy as np

    out = {}
    for exp, methods in (("replogle_trackA_lr3e5", ["scgpt_paper"]), ("adamson_trackA_lr3e5", ["scgpt_paper", "celljepa"])):
        r = run.remote(exp, methods)
        out[exp] = r
        print(f"\n## {exp}: {r['n']} held-out genes (chance median = 50)")
        print("| arm | recipe | metric | seeds | ID percentile | top-1 | top-3 |")
        print("|---|---|---|---|---|---|---|")
        for k, v in sorted(r["res"].items(), key=lambda kv: (kv[0].split(" | ")[2], -np.mean(np.array(kv[1]["per_gene"])[:, 0]))):
            p = np.array(v["per_gene"])
            a, m, c = k.split(" | ")
            print(f"| {a} | {m} | {c} | {v['seeds']} | {p[:, 0].mean() / 100:.3f} | {p[:, 1].mean():.3f} | {p[:, 2].mean():.3f} |")
    json.dump(out, open("diag/results/X_id_percentile.json", "w"), indent=1)
