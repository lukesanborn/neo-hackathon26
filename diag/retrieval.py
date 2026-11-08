"""A16/A17: predicted-effect retrieval — scoring Task 3 predictions the way a perturbation-selection tool would use them,
with per-gene scores and paired bootstrap confidence intervals (A17).

For every saved test prediction set (runs + baselines), with deltas = profile − mean of all control cells:
  * ref_spearman / ref_p@5: rank the measured TRAINING perturbations by cosine similarity to the predicted
    delta of a held-out gene, and compare with the ranking by its TRUE delta (Spearman; top-5 overlap / 5).
  * cluster_agreement: Pearson correlation between the upper triangles of the cosine-similarity matrices of
    predicted vs true deltas among the held-out genes (do predictions cluster the genes the right way?).
  * id_top1 / id_mean_rank: is each prediction closest (cosine) to its own gene's true delta among the
    held-out genes? Chance top-1 = 1/n_test (1/22 ≈ 0.045 on Adamson).
Genes: the 500 genes with the largest variance of TRAINING-perturbation mean deltas (no test information).
A constant prediction (training mean) scores near chance on cluster/identification by construction.

CIs (src/retrieval.py): per-gene scores are averaged over seeds of the same arm (run names differing only by
`_s<seed>`), then a paired bootstrap over test genes (2000 resamples, rng seed 0, 95% percentile) gives CIs for
each arm's means and for paired differences (same resampled genes for both arms). cluster_agreement is
recomputed on each resampled sub-matrix (pairs of duplicated genes dropped; undefined draws skipped).

Run from the repo root:
  .venv/bin/modal run diag/retrieval.py [--cache /vol/data/...] [--exps a,b,c] [--tag A17_adamson]
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
N_BOOT = 2000
LINEAR = "baselines/linear"
UPPER = "(upper bound) true deltas"
# extra paired comparisons (skipped when an arm is absent)
PAIRS = [
    ("adamson_trackA_lr3e5/runs/celljepa_jepa_lr3e5", "adamson_trackA_lr3e5/runs/scgpt_paper_scgpt_lr3e5"),
    ("adamson_celljepa_public_v1/runs/celljepa_v1", "adamson_celljepa_public_v1/runs/scgpt_paper_v1"),
    ("adamson_trackB_v1/runs/jepa_v1", "adamson_trackB_v1/runs/scgpt_v1"),
]


@app.function(image=image, volumes={"/vol": vol}, cpu=4, memory=16384, timeout=1800)
def run(cache: str, exps: list, split: str = "test") -> dict:
    """split="test": score test_predictions.npz (final evaluation, once). split="val": score
    val_predictions.npz of the selected checkpoint (for choosing configs; never look at test for that)."""
    import numpy as np

    from src import retrieval as RT
    from src.data import CTRL, load_cache

    ds = load_cache(cache, mmap=True)
    cond = ds.conditions
    ctrl = np.asarray(ds.X[np.flatnonzero(cond == CTRL)]).mean(0)
    mean_of = lambda c: np.asarray(ds.X[np.flatnonzero(cond == c)]).mean(0)  # noqa: E731
    train = [c for c in ds.split["train"] if c != CTRL]
    test = list(ds.split[split])           # the evaluated conditions ("test" name kept below)
    pred_file = f"{split}_predictions.npz"
    R_all = np.stack([mean_of(c) - ctrl for c in train])
    genes = np.argsort(-R_all.var(0))[:N_GENES]
    R = R_all[:, genes]
    T = np.stack([mean_of(c) - ctrl for c in test])[:, genes]
    S_true = RT.true_sim(T)

    sets, skipped = {}, []
    for exp in exps:
        base = f"/vol/experiments/{exp}"
        for kind in ("runs", "baselines"):
            d = os.path.join(base, kind)
            if not os.path.isdir(d):
                continue
            for r in sorted(os.listdir(d)):
                path = os.path.join(d, r, pred_file)
                if not os.path.exists(path):
                    continue
                z = np.load(path, allow_pickle=False)
                pmap = dict(zip([str(c) for c in z["conditions"]], z["preds"]))
                missing = [c for c in test if c not in pmap]
                if missing:
                    skipped.append(f"{exp}/{kind}/{r} (missing {len(missing)} test conditions)")
                    continue
                P = np.stack([pmap[c] - ctrl for c in test])[:, genes]
                pg, S = RT.per_gene_scores(P, T, R)
                key = f"{exp}/{kind}/{r}"
                if kind == "baselines":            # baselines depend only on the dataset: keep one copy
                    short = f"baselines/{r}"
                    if short not in sets:
                        key = short
                    elif np.allclose(sets[short]["P"], P):
                        continue                   # identical copy in another experiment
                sets[key] = {"per_gene": pg, "S_pred": S, "P": P}
    # ---- seed ensembles: average the predicted deltas of all seeds of an arm, scored as one set ----
    by_arm = {}
    for k, v in sets.items():
        if "/runs/" in k:
            by_arm.setdefault(RT.arm_of(k), []).append(v["P"])
    for arm, Ps in by_arm.items():
        if len(Ps) >= 2:
            Pm = np.mean(Ps, axis=0)
            pg, S = RT.per_gene_scores(Pm, T, R)
            sets[f"{arm}__ens{len(Ps)}"] = {"per_gene": pg, "S_pred": S, "P": Pm}
    # ---- stronger baselines from scGPT_human gene embeddings (hyperparameter chosen on val) ----
    emb_info = {}
    try:
        import torch

        from src.data import target_gene
        from src.models import SimpleVocab, gene_vocab_ids
        from src.retrieval_baselines import select_and_predict

        vocab = SimpleVocab("/vol/checkpoints/scGPT_human/vocab.json")
        W_emb = torch.load("/vol/checkpoints/scGPT_human/best_model.pt", map_location="cpu")["encoder.embedding.weight"].numpy()
        aliases = (ds.manifest or {}).get("extra_hgnc_aliases") or {}
        emb_of = lambda cs: W_emb[gene_vocab_ids([target_gene(c) for c in cs], vocab, extra_aliases=aliases)[0]]  # noqa: E731
        val = list(ds.split["val"])
        T_val = np.stack([mean_of(c) - ctrl for c in val])[:, genes]
        for kind in ("ridge", "knn"):
            P, hp, sc = select_and_predict(emb_of(train), R, emb_of(val), T_val, emb_of(test), kind)
            pg, S = RT.per_gene_scores(P, T, R)
            sets[f"baselines/{kind}_scgpt_gene_emb"] = {"per_gene": pg, "S_pred": S, "P": P}
            emb_info[kind] = {"chosen": hp, "val_id_pct": sc}
    except Exception as e:                          # never let the extra baselines break the main scoring
        emb_info["error"] = repr(e)[:300]
    pg_ub, S_ub = RT.per_gene_scores(T, T, R)
    sets[UPPER] = {"per_gene": pg_ub, "S_pred": S_ub}

    agg = {k: RT.aggregate(v["per_gene"], RT.cluster_agreement(v["S_pred"], S_true)) for k, v in sets.items()}
    arms = RT.pool_seeds(sets)
    point, draws = RT.bootstrap_draws(arms, S_true, n_boot=N_BOOT, seed=0)
    cis = RT.summarize_ci(point, draws)
    # each deep arm vs linear: its own experiment's linear if that differs from the shared copy
    lin_of = lambda a: next((x for x in (f"{a.split('/')[0]}/{LINEAR}", LINEAR) if x in arms), None)  # noqa: E731
    pairs = [(a, lin_of(a)) for a in sorted(arms) if "/runs/" in a and lin_of(a)]
    pairs += [(a, b) for a in sorted(arms) if "/runs/" in a
              for b in ("baselines/ridge_scgpt_gene_emb", "baselines/knn_scgpt_gene_emb") if b in arms]
    pairs += [(a, b) for a, b in PAIRS if a in arms and b in arms]
    diffs = [{"a": a, "b": b, **RT.diff_ci(point, draws, a, b)} for a, b in pairs]
    return RT.json_safe({
        "cache": cache, "exps": exps, "split": split, "embedding_baselines": emb_info, "n_test": len(test), "n_ref": len(train), "n_genes": int(len(genes)),
        "n_boot": N_BOOT, "boot_seed": 0, "test_conditions": test, "skipped": skipped, "metrics": list(RT.METRICS),
        "aggregates": agg,
        "set_per_gene": {s: d["per_gene"] for s, d in sets.items()},
        "arms": {arm: {"members": a["members"], "n_seeds": len(a["members"]), "per_gene": a["per_gene"],
                       "point_aggregate_seed_mean": _seed_mean([agg[m] for m in a["members"]]),
                       "ci": cis[arm]} for arm, a in arms.items()},
        "diffs": diffs,
    })


def _seed_mean(aggs):
    """A16-style: average per-set aggregates over seeds, skipping undefined (NaN) ones."""
    out = {}
    for k in aggs[0]:
        xs = [x[k] for x in aggs if x[k] == x[k]]
        out[k] = sum(xs) / len(xs) if xs else float("nan")
    return out


def _fmt(x, nd=3):
    return "undef" if x is None or x != x else f"{x:.{nd}f}"


def _ci_cell(c):
    if c["mean"] is None or c["mean"] != c["mean"]:
        return "undef"
    return f"{_fmt(c['mean'])} [{_fmt(c['lo'])}, {_fmt(c['hi'])}]"


def _aggregate_table(res):
    keys = res["metrics"]
    lines = ["| Predictions | n | " + " | ".join(keys) + " |", "|---" * (len(keys) + 2) + "|"]
    for arm in sorted(res["arms"]):
        a = res["arms"][arm]
        cells = [_fmt(a["point_aggregate_seed_mean"][k]) for k in keys]
        lines.append(f"| {arm} | {a['n_seeds']} | " + " | ".join(cells) + " |")
    return lines


def _ci_markdown(res, tag):
    METRICS = res["metrics"]

    n = res["n_test"]
    seeds = sorted({a["n_seeds"] for k, a in res["arms"].items() if "/runs/" in k})
    L = [f"# {tag}: retrieval metrics with 95% bootstrap CIs", "",
         f"Cache `{res['cache']}`; experiments {', '.join(res['exps'])}. n_test = {n} held-out genes "
         f"(chance id_top1 = {1 / n:.3f}, chance id_mean_rank = {(n + 1) / 2:.1f}); {res['n_ref']} reference "
         f"(training) perturbations; {res['n_genes']} genes.",
         f"Per-gene scores averaged over seeds, then paired bootstrap over test genes ({res['n_boot']} resamples, "
         f"rng seed {res['boot_seed']}, percentile CIs). cluster_agreement recomputed on each resampled "
         "sub-matrix (duplicate-gene pairs dropped; undefined draws skipped).",
         f"**Small n caveat:** with only {n} test genes the CIs are wide; they cover resampling of these genes "
         "only (not other gene sets / datasets), and seed variation enters only through the seed average "
         f"({'/'.join(map(str, seeds)) or '0'} seed(s) per deep arm). NaN per-seed scores (constant "
         "predictions) are skipped in the seed average.", ""]
    if res["skipped"]:
        L += ["Skipped (incomplete prediction sets): " + "; ".join(res["skipped"]), ""]
    L += ["## Per-arm means [95% CI]", "", "| Arm | seeds | " + " | ".join(METRICS) + " |",
          "|---" * (len(METRICS) + 2) + "|"]
    for arm in sorted(res["arms"]):
        a = res["arms"][arm]
        L.append(f"| {arm} | {a['n_seeds']} | " + " | ".join(_ci_cell(a["ci"][m]) for m in METRICS) + " |")
    L += ["", "## Paired differences (A − B): diff [95% CI] (fraction of resamples with diff > 0)", "",
          "For id_mean_rank lower is better, so a negative difference favours A.", "",
          "| A | B | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 2) + "|"]
    for d in res["diffs"]:
        cells = []
        for m in METRICS:
            c = d[m]
            cells.append("undef" if c["n_valid"] == 0 else
                         f"{_fmt(c['diff'])} [{_fmt(c['lo'])}, {_fmt(c['hi'])}] ({_fmt(c['frac_gt0'], 2)})")
        L.append(f"| {d['a']} | {d['b']} | " + " | ".join(cells) + " |")
    L += ["", "## Point aggregates (per prediction set; seed aggregates averaged — A16 format)", ""]
    L += _aggregate_table(res)
    return L


@app.local_entrypoint()
def main(cache: str = CACHE, exps: str = ",".join(EXPS), tag: str = "A17_adamson", split: str = "test"):
    res = run.remote(cache, [e for e in exps.split(",") if e], split)
    os.makedirs("diag/results", exist_ok=True)
    with open(f"diag/results/{tag}_retrieval_ci.json", "w") as f:
        json.dump(res, f, indent=1, allow_nan=False)
    lines = _ci_markdown(res, tag)
    with open(f"diag/results/{tag}_retrieval_ci.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
