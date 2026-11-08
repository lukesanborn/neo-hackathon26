"""A26: latent-space retrieval on Replogle VALIDATION genes (see src/latent_retrieval.py for the method).

Three embedding spaces (pretrained encoders): pub = public scGPT_human, bscgpt = blood-scGPT, bjepa = blood-JEPA.
Stages (run from the repo root; GPU stages capped at 12 concurrent containers):
  refs   per space: control mean, TRUE latent shifts of val genes (+ split halves = ceiling), training-gene
         shifts (reference set), baselines (PCA linear, training mean) and an oracle (control + true delta)
         embedded through the same top-k rule as predictions
  preds  per Step 2 run (24): reload best.pt, predict one cell per shared control input for every val gene,
         embed in all three spaces
  score  per space: retrieval metrics + paired bootstrap CIs (src/retrieval.py), seeds pooled, 2-seed ensembles
  all    refs -> preds -> score
    .venv/bin/modal run diag/latent_retrieval.py --stage all
Outputs: /vol/experiments/replogle_k562_v1/latent/*.npz, diag/results/A26_latent_<space>.{md,json}
"""
import json
import os
import sys

import modal

sys.path.insert(0, os.getcwd())
from environment.image import build_image  # noqa: E402

app = modal.App("jepa-latent-retrieval")
vol = modal.Volume.from_name("jepa-perturb")
image = build_image("blackwell")
EXP = "replogle_k562_v1"
CACHE = "/vol/data/replogle_k562_essential_gears_simulation_seed1"
OUT = f"/vol/experiments/{EXP}/latent"
SPACES = {"pub": "/vol/checkpoints/scGPT_human",
          "bscgpt": "/vol/trackA/checkpoints/scgpt_kidney_blood_lr3e5",
          "bjepa": "/vol/trackA/checkpoints/jepa_kidney_blood_lr3e5"}
SPACE_NAMES = {"pub": "public scGPT", "bscgpt": "blood-scGPT", "bjepa": "blood-JEPA"}
STARTS = ("pub", "bscgpt", "bjepa")
RUNS = [f"{m}_s{s}_s2{st}_r{r}" for st in STARTS for m in ("scgpt_paper", "celljepa") for r in (0, 1) for s in (1, 2)]
N_CTRL_MEAN, N_TRUE, N_TRAIN = 2048, 256, 64
MAXC = 12


def _setup(space):
    import numpy as np

    from src.data import load_cache
    from src.models import HGNC_ALIASES
    from src.track_a.pbmc import load_encoder

    ds = load_cache(CACHE, mmap=True)
    model, vocab, conv = load_encoder(SPACES[space], "cuda")
    al = {**HGNC_ALIASES, **((ds.manifest or {}).get("extra_hgnc_aliases") or {})}
    syms = [g if g in vocab else al.get(g, g) for g in ds.gene_names]
    return np, ds, model, vocab, conv, syms


def _rows(np, ds, idx):
    return np.asarray(ds.X[np.sort(np.asarray(idx))], dtype=np.float32)


@app.function(image=image, gpu="B300", volumes={"/vol": vol}, cpu=8, memory=65536, timeout=3600, max_containers=MAXC)
def refs(space: str) -> dict:
    from src import retrieval as RT
    from src.baselines import linear_gene_embedding, training_mean
    from src.data import CTRL
    from src.latent_retrieval import embed_rows, n_expressed

    np, ds, model, vocab, conv, syms = _setup(space)
    rng = np.random.default_rng(0)
    emb = lambda rows, ks=None: embed_rows(model, vocab, conv, rows, n_expressed(rows) if ks is None else ks, syms)  # noqa: E731
    ctrl_idx = ds.ctrl_idx
    c0 = emb(_rows(np, ds, rng.choice(ctrl_idx, min(N_CTRL_MEAN, len(ctrl_idx)), replace=False))).mean(0)
    val = list(ds.split["val"])
    train = [c for c in ds.split["train"] if c != CTRL]
    true, half_a, half_b = [], [], []
    for c in val:
        cells = ds.cells_of(c)
        cells = rng.choice(cells, min(N_TRUE, len(cells)), replace=False)
        e = emb(_rows(np, ds, cells))
        true.append(e.mean(0) - c0)
        h = len(e) // 2
        half_a.append(e[:h].mean(0) - c0)
        half_b.append(e[h:].mean(0) - c0)
    R = []
    for c in train:
        cells = ds.cells_of(c)
        R.append(emb(_rows(np, ds, rng.choice(cells, min(N_TRAIN, len(cells)), replace=False))).mean(0) - c0)
    # baselines + oracle as pseudo-cells: each shared control input + the (predicted / true) mean delta
    ctrl_ref = np.asarray(ds.control_reference())
    lin = linear_gene_embedding(ds, val, ctrl_ref)[0]
    tm = training_mean(ds, val, ctrl_ref)
    out = {"lin": [], "tm": [], "oracle": []}
    for c in val:
        ctrl_rows = _rows(np, ds, ds.eval_controls[c])
        ks = n_expressed(ctrl_rows)
        for key, delta in (("lin", lin[c] - ctrl_ref), ("tm", tm[c] - ctrl_ref),
                           ("oracle", np.asarray(ds.observed_mean(c)) - ctrl_ref)):
            out[key].append(emb(np.clip(ctrl_rows + delta, 0, None), ks).mean(0) - c0)
    arrs = {"c0": c0, "true": np.stack(true), "half_a": np.stack(half_a), "half_b": np.stack(half_b),
            "train": np.stack(R), **{k: np.stack(v) for k, v in out.items()}}
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"refs_{space}.npz"), val=np.array(val), **arrs)
    vol.commit()
    return {"space": space, "ceiling_split_half": RT.id_scores(arrs["half_a"], arrs["half_b"]),
            "oracle_pseudo": RT.id_scores(arrs["oracle"], arrs["true"]), "n_val": len(val), "n_train": len(train),
            "encoder": conv.get("objective")}


@app.function(image=image, gpu="B300", volumes={"/vol": vol}, cpu=8, memory=65536, timeout=3600, max_containers=MAXC)
def preds(run: str) -> dict:
    import dataclasses

    import numpy as np
    import torch

    from src.data import load_cache
    from src.latent_retrieval import embed_rows, n_expressed
    from src.models import HGNC_ALIASES
    from src.track_a.pbmc import load_encoder
    from src.train import TrainConfig, Trainer

    vol.reload()
    rdir = f"/vol/experiments/{EXP}/runs/{run}"
    cfgd = json.load(open(os.path.join(rdir, "config.json")))
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in cfgd.items() if k in fields})
    ds = load_cache(CACHE)
    tr = Trainer(ds, cfgd["ckpt_dir"], cfg, device="cuda", log=lambda *_: None)
    model = tr.fresh_model_from(os.path.join(rdir, "best.pt"))
    val = list(ds.split["val"])
    al = {**HGNC_ALIASES, **((ds.manifest or {}).get("extra_hgnc_aliases") or {})}
    encs = {}
    for sp, d in SPACES.items():
        m, vocab, conv = load_encoder(d, "cuda")
        encs[sp] = (m, vocab, conv, [g if g in vocab else al.get(g, g) for g in ds.gene_names],
                    np.load(os.path.join(OUT, f"refs_{sp}.npz"))["c0"])
    res = {sp: [] for sp in SPACES}
    with torch.no_grad():
        for c in val:
            ctrl = np.asarray(ds.eval_controls[c])
            rows = np.concatenate([tr._forward_controls(model, ctrl[i:i + 64], ds.gene_pos(c)).float().cpu().numpy()
                                   for i in range(0, len(ctrl), 64)])
            rows = np.clip(rows, 0, None)                    # predictions are log1p; no negative expression
            ks = n_expressed(np.asarray(ds.X[ctrl]))          # keep as many genes as each paired control expresses
            for sp, (m, vocab, conv, syms, c0) in encs.items():
                res[sp].append(embed_rows(m, vocab, conv, rows, ks, syms).mean(0) - c0)
    np.savez(os.path.join(OUT, f"pred_{run}.npz"), val=np.array(val), **{sp: np.stack(v) for sp, v in res.items()})
    vol.commit()
    return {"run": run, "n_val": len(val)}


@app.function(image=image, volumes={"/vol": vol}, cpu=4, memory=16384, timeout=1800)
def score(space: str) -> dict:
    import numpy as np

    from src import retrieval as RT

    vol.reload()
    ref = np.load(os.path.join(OUT, f"refs_{space}.npz"))
    T, R = ref["true"], ref["train"]
    S_true = RT.true_sim(T)
    sets = {}
    for run in RUNS:
        p = os.path.join(OUT, f"pred_{run}.npz")
        if os.path.exists(p):
            P = np.load(p)[space]
            pg, S = RT.per_gene_scores(P, T, R)
            sets[f"runs/{run}"] = {"per_gene": pg, "S_pred": S, "P": P}
    by_arm = {}
    for k, v in list(sets.items()):
        by_arm.setdefault(RT.arm_of(k), []).append(v["P"])
    for arm, Ps in by_arm.items():
        if len(Ps) >= 2:
            pg, S = RT.per_gene_scores(np.mean(Ps, 0), T, R)
            sets[f"{arm}__ens{len(Ps)}"] = {"per_gene": pg, "S_pred": S}
    for key, name in (("lin", "baselines/linear"), ("tm", "baselines/training_mean"), ("oracle", "(oracle) control + true delta")):
        pg, S = RT.per_gene_scores(ref[key], T, R)
        sets[name] = {"per_gene": pg, "S_pred": S}
    arms = RT.pool_seeds(sets)
    point, draws = RT.bootstrap_draws(arms, S_true, n_boot=2000, seed=0)
    cis = RT.summarize_ci(point, draws)
    pairs = [(f"runs/{m}_s2bjepa_r{r}", f"runs/{m}_s2bscgpt_r{r}") for m in ("scgpt_paper", "celljepa") for r in (0, 1)]
    pairs += [(f"runs/{m}_s2bjepa_r{r}__ens2", f"runs/{m}_s2bscgpt_r{r}__ens2") for m in ("scgpt_paper", "celljepa") for r in (0, 1)]
    pairs += [(a, "baselines/linear") for a in sorted(arms) if a.startswith("runs/")]
    pairs = [(a, b) for a, b in pairs if a in arms and b in arms]
    diffs = [{"a": a, "b": b, **RT.diff_ci(point, draws, a, b)} for a, b in pairs]
    return RT.json_safe({"space": space, "n_val": len(T), "metrics": list(RT.METRICS),
                         "ceiling_split_half": RT.id_scores(ref["half_a"], ref["half_b"]),
                         "arms": {a: {"n_seeds": len(v["members"]), "ci": cis[a]} for a, v in arms.items()},
                         "diffs": diffs})


def _md(res):
    sp = res["space"]
    L = [f"# A26 latent retrieval, embedding space = {SPACE_NAMES[sp]} ({sp})", "",
         f"{res['n_val']} validation genes. Ceiling (split-half of real cells): "
         f"id_pct {res['ceiling_split_half']['id_pct']:.3f}, top-1 {res['ceiling_split_half']['id_top1']:.3f}.", "",
         "| Arm | seeds | " + " | ".join(res["metrics"]) + " |", "|---" * (len(res["metrics"]) + 2) + "|"]
    for a in sorted(res["arms"]):
        c = res["arms"][a]["ci"]
        cells = [("undef" if c[m]["mean"] is None else f"{c[m]['mean']:.3f} [{c[m]['lo']:.2f}, {c[m]['hi']:.2f}]")
                 for m in res["metrics"]]
        L.append(f"| {a} | {res['arms'][a]['n_seeds']} | " + " | ".join(cells) + " |")
    L += ["", "## Paired differences (A − B)", "", "| A | B | id_pct | id_top1 | id_top3 | id_top10 |", "|---|---|---|---|---|---|"]
    for d in res["diffs"]:
        cells = []
        for m in ("id_pct", "id_top1", "id_top3", "id_top10"):
            x = d[m]
            cells.append("undef" if not x.get("n_valid") else f"{x['diff']:+.3f} [{x['lo']:+.2f}, {x['hi']:+.2f}]")
        L.append(f"| {d['a']} | {d['b']} | " + " | ".join(cells) + " |")
    return L


@app.local_entrypoint()
def main(stage: str = "all"):
    os.makedirs("diag/results", exist_ok=True)
    if stage in ("refs", "all"):
        for r in refs.map(list(SPACES)):
            print("REFS", json.dumps(r), flush=True)
    if stage in ("preds", "all"):
        for r in preds.map(RUNS, return_exceptions=True):
            print("PREDS", r if isinstance(r, dict) else repr(r)[:300], flush=True)
    if stage in ("score", "all"):
        for res in score.map(list(SPACES)):
            json.dump(res, open(f"diag/results/A26_latent_{res['space']}.json", "w"), indent=1)
            open(f"diag/results/A26_latent_{res['space']}.md", "w").write("\n".join(_md(res)) + "\n")
            print("SCORED", res["space"], flush=True)
