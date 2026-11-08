"""X1 / X2: I-JEPA gene-token pretraining (ijepa) and gene-program masking (progmask) — Modal app.

Own app name (``jepa-xexp``) so teammates' deployed apps are untouched. Fine-tuning uses the DEPLOYED
``jepa-perturb.train_blackwell`` / ``prepare`` (the exact code that produced every baseline run).

    modal run modal_xjepa.py --stage smoke          # ~200 real-cache steps per variant: s/step + finite losses
    modal deploy modal_xjepa.py                     # needed once for the chained pipeline below
    .venv/bin/python diag/launch_xjepa.py           # pretrain X1+X2 (each chains its fine-tunes + probe),
                                                    # prepare Replogle + baseline-checkpoint fine-tunes now
    modal run modal_xjepa.py --stage eval           # retrieval / inverse identification + gene-emb probe
"""
from __future__ import annotations

import json
import os

import modal

from environment.image import build_image

APP_NAME = "jepa-xexp"
VOL = "/vol"
VOCAB = f"{VOL}/checkpoints/scGPT_human/vocab.json"
KIDNEY = f"{VOL}/trackA/kidney_census_2023-05-15"
RUNS = f"{VOL}/trackA/pretrain/runs"
CKPTS = f"{VOL}/trackA/checkpoints"
OUT = f"{VOL}/experiments/xjepa"
ADAMSON_EXP = "adamson_trackA_lr3e5"
REPLOGLE_EXP = "replogle_trackA_lr3e5"
REPLOGLE_DATASET = "replogle_k562_essential"
BASE = {"epochs": 4, "batch_size": 32, "lr": 3e-5, "max_minutes": 600.0, "seed": 0, "ema_momentum": 0.996,
        "mask_ratio": 0.25}
VARIANTS = {"ijepa": "lr3e5_ijepa", "progmask": "lr3e5_progmask"}          # variant -> pretrain tag
BASELINE_CKPTS = ["scgpt_kidney_lr3e5", "jepa_kidney_lr3e5", "jepa_kidney_lr3e5_token",
                  "jepa_kidney_lr3e5_center"]

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("jepa-perturb")
image = build_image("blackwell")


def _plain(x):
    return json.loads(json.dumps(x, default=str))


def ckpt_name(variant: str) -> str:
    return f"jepa_kidney_{VARIANTS[variant]}"


def short(ckpt: str) -> str:
    return ckpt.replace("_kidney", "")


def finetune_fn():
    return modal.Function.from_name("jepa-perturb", "train_blackwell").with_options(gpu="B300", max_containers=16)


def spawn_finetunes(ckpt: str, adamson_methods=("scgpt_paper", "celljepa"), replogle=True) -> dict:
    fn, ids = finetune_fn(), {}
    kw = dict(epochs=15, aux_weight=0.5, max_minutes=150.0, tag=short(ckpt), ckpt=f"{CKPTS}/{ckpt}",
              input_transform="bins50")
    for m in adamson_methods:
        for s in (1, 2):
            ids[f"adamson {short(ckpt)} {m} s{s}"] = fn.spawn(exp_id=ADAMSON_EXP, method=m, seed=s, **kw).object_id
    if replogle:
        for s in (1, 2):
            ids[f"replogle {short(ckpt)} scgpt_paper s{s}"] = fn.spawn(
                exp_id=REPLOGLE_EXP, method="scgpt_paper", seed=s, **kw).object_id
    return ids


@app.function(image=image, gpu="B300", volumes={VOL: vol}, cpu=8, memory=65536, timeout=24 * 3600)
def pretrain(variant: str, tag: str, overrides: dict, chain: bool = False) -> dict:
    from src.track_a.pretrain import PretrainConfig, run_pretraining

    vol.reload()
    cfg = PretrainConfig(objective="jepa", jepa_variant=variant, **overrides)
    runs, ckpts = (f"{VOL}/smoke/xjepa/runs", f"{VOL}/smoke/xjepa/checkpoints") if tag.startswith("smoke") \
        else (RUNS, CKPTS)
    s = run_pretraining(cfg, KIDNEY, VOCAB, os.path.join(runs, f"jepa_{tag}"), os.path.join(ckpts, f"jepa_kidney_{tag}"),
                        commit=vol.commit)
    s = _plain(s)
    if chain and s["status"] == "completed":
        ck = f"jepa_kidney_{tag}"
        s["chained_finetunes"] = spawn_finetunes(ck)
        s["chained_probe"] = program_probe.spawn(ck).object_id
        with open(os.path.join(RUNS, f"jepa_{tag}", "chained_calls.json"), "w") as f:
            json.dump({k: s[k] for k in ("chained_finetunes", "chained_probe")}, f, indent=1)
        vol.commit()
    return s


@app.function(image=image, gpu="B300", volumes={VOL: vol}, memory=65536, timeout=4 * 3600, max_containers=24)
def finetune(exp_id: str, method: str, seed: int, tag: str, ckpt: str, epochs: int = 15, aux_weight: float = 0.5,
             max_minutes: float = 150.0, input_transform: str = "bins50") -> dict:
    """Same body as jepa-perturb._train (src/train.py unchanged), in this app so its concurrency cap is ours."""
    from src.data import load_cache
    from src.train import TrainConfig, run_training

    vol.reload()
    cfg_e = json.load(open(f"{VOL}/experiments/{exp_id}/experiment.json"))
    ds = load_cache(cfg_e["cache_dir"])
    cfg = TrainConfig(method=method, seed=seed, epochs=epochs, aux_weight=aux_weight, max_minutes=max_minutes,
                      ckpt=ckpt, input_transform=input_transform, teacher_update="frozen", max_steps_per_epoch=None)
    run_dir = os.path.join(f"{VOL}/experiments/{exp_id}", "runs", f"{method}_s{seed}_{tag}")
    if os.path.exists(os.path.join(run_dir, "metrics.json")):
        raise FileExistsError(f"{run_dir} already completed")
    return _plain(run_training(ds, ckpt, cfg, run_dir, commit=vol.commit))


def spawn_extra_seeds(ckpt: str, seeds=(3,)) -> dict:
    """Extra Replogle seeds on this app's own fine-tune function (more parallel GPUs, tighter CIs)."""
    return {f"replogle {short(ckpt)} scgpt_paper s{s}": finetune.spawn(
        REPLOGLE_EXP, "scgpt_paper", s, short(ckpt), f"{CKPTS}/{ckpt}").object_id for s in seeds}


@app.function(image=image, gpu="H100", volumes={VOL: vol}, cpu=8, memory=65536, timeout=3 * 3600)
def program_probe(ckpt: str, n_cells: int = 2048, max_genes: int = 600, min_frac_cells: float = 0.2) -> dict:
    """Embed held-out kidney cells with each Hallmark program's genes REMOVED, ridge-probe that program's
    activity (5-fold CV R^2). Arm "pca_bins" = same probe on PCA-50 of the remaining genes' bins."""
    import numpy as np
    import torch

    from src.models import SimpleVocab
    from src.track_a.ijepa import GenePrograms
    from src.track_a.pretrain import KidneyCache
    from src.track_a.pretrain_model import CLS_VALUE, PAD_VALUE, build_student
    from src.xjepa_eval import program_scores, residualize, ridge_cv_r2

    vol.reload()
    vocab = SimpleVocab(VOCAB)
    progs = GenePrograms(vocab)
    cache = KidneyCache(KIDNEY)
    rng = np.random.default_rng(0)
    cells = rng.permutation(cache.val)[:n_cells]
    raw = [cache.cell(int(c)) for c in cells]
    S = np.stack([program_scores(gi, bi, progs.members) for gi, bi in raw])     # (n_cells, n_programs)
    keep = [p for p in range(len(progs.names)) if (S[:, p] > 0).mean() >= min_frac_cells and S[:, p].std() > 0]
    # depth / complexity covariates: log #expressed genes, mean bin. The primary score is residualised on them
    cov = np.column_stack([np.log([max(len(g), 1) for g, _ in raw]), [b.mean() if len(b) else 0 for _, b in raw]])

    use_pca = ckpt == "pca_bins"
    if not use_pca:
        model = build_student(vocab, seed=0)
        sd = torch.load(os.path.join(CKPTS, ckpt, "best_model.pt"), map_location="cpu")
        missing, _ = model.load_state_dict(sd, strict=False)
        assert not [k for k in missing if k.startswith(("encoder", "value_encoder", "transformer_encoder"))], missing
        model = model.cuda().eval()
    cls_id, pad_id = vocab["<cls>"], vocab["<pad>"]

    def embed(seqs):
        out = []
        for i in range(0, len(seqs), 128):
            chunk = seqs[i:i + 128]
            L = 1 + max(max(len(g) for g, _ in chunk), 1)
            gid = np.full((len(chunk), L), pad_id, dtype=np.int64)
            val = np.full((len(chunk), L), PAD_VALUE, dtype=np.float32)
            gid[:, 0], val[:, 0] = cls_id, CLS_VALUE
            for b, (g, v) in enumerate(chunk):
                gid[b, 1:1 + len(g)], val[b, 1:1 + len(g)] = g, v
            gid_t, val_t = torch.from_numpy(gid).cuda(), torch.from_numpy(val).cuda()
            pad = gid_t == pad_id
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.transformer_encoder(model.encoder(gid_t) + model.value_encoder(val_t),
                                              src_key_padding_mask=pad).float()
            w = (~pad).float().unsqueeze(-1)
            w[:, 0] = 0                                       # mean over gene tokens (exclude <cls>)
            out.append(((h * w).sum(1) / w.sum(1).clamp_min(1)).cpu().numpy())
        return np.concatenate(out)

    def pca_bins(seqs):
        universe = {g: j for j, g in enumerate(sorted({int(x) for g, _ in seqs for x in g}))}
        X = np.zeros((len(seqs), len(universe)), dtype=np.float32)
        for b, (g, v) in enumerate(seqs):
            X[b, [universe[int(x)] for x in g]] = v
        X -= X.mean(0)
        w, V = np.linalg.eigh((X @ X.T).astype(np.float64))     # Gram trick: n_cells x n_cells
        top = np.argsort(-w)[:50]
        return V[:, top] * np.sqrt(np.clip(w[top], 0, None))

    res, res_raw = {}, {}
    for p in keep:
        drop = set(progs.members[p].tolist())
        seqs = []
        for k, (gi, bi) in enumerate(raw):
            m = np.array([int(g) not in drop for g in gi], dtype=bool)
            g, v = gi[m], bi[m].astype(np.float32)
            if len(g) > max_genes:
                idx = np.random.default_rng([k, p]).choice(len(g), size=max_genes, replace=False)
                g, v = g[idx], v[idx]
            seqs.append((g, v))
        Z = pca_bins(seqs) if use_pca else embed(seqs)
        res[progs.names[p]] = ridge_cv_r2(Z, residualize(S[:, p], cov))
        res_raw[progs.names[p]] = ridge_cv_r2(Z, S[:, p])
    vals, vraw = np.array(list(res.values())), np.array(list(res_raw.values()))
    out = {"ckpt": ckpt, "n_cells": int(len(cells)), "n_programs_scored": len(res),
           "score": "program activity residualised on log(#expressed genes) and mean bin",
           "mean_r2": float(vals.mean()), "median_r2": float(np.median(vals)), "per_program": res,
           "raw_mean_r2": float(vraw.mean()), "raw_per_program": res_raw}
    os.makedirs(f"{OUT}/program_probe", exist_ok=True)
    json.dump(out, open(f"{OUT}/program_probe/{ckpt}.json", "w"), indent=1)
    vol.commit()
    return out


@app.function(image=image, volumes={VOL: vol}, cpu=8, memory=65536, timeout=3600)
def evaluate(exp_id: str, ckpts: list, tag: str = "X") -> dict:
    """Retrieval + inverse identification for all lr3e5 Track A runs of ``exp_id`` and for the
    gene-embedding linear probe of each checkpoint in ``ckpts`` (plus scGPT_human)."""
    import numpy as np

    from src.data import load_cache
    from src.xjepa_eval import collect_run_sets, deltas_setup, gene_embedding_linear, gene_embedding_table, score_sets

    vol.reload()
    cfg = json.load(open(f"{VOL}/experiments/{exp_id}/experiment.json"))
    ds = load_cache(cfg["cache_dir"], mmap=True)
    d = deltas_setup(ds)
    sets, skipped = collect_run_sets([f"{VOL}/experiments/{exp_id}"], d["test"], d["ctrl"], d["genes"],
                                     name_filter=lambda r: "lr3e5" in r)
    for ck in list(ckpts) + ["scGPT_human"]:
        cdir = f"{VOL}/checkpoints/scGPT_human" if ck == "scGPT_human" else f"{CKPTS}/{ck}"
        if not os.path.exists(os.path.join(cdir, "best_model.pt")):
            skipped.append(f"gene-emb {ck} (no checkpoint)")
            continue
        try:
            E, found = gene_embedding_table(cdir, ds.gene_names, ds.manifest.get("extra_hgnc_aliases"))
            need = [ds.gene_pos(c) for c in d["train"] + d["test"]]
            if not found[need].all():
                skipped.append(f"gene-emb {ck}: {int((~found[need]).sum())} targets not in vocab")
                continue
            preds = gene_embedding_linear(ds, E, d["test"])
            sets[f"geneemb_linear/{short(ck)}"] = np.stack(
                [preds[c] - d["ctrl"] for c in d["test"]])[:, d["genes"]]
        except Exception as e:  # report, don't hide
            skipped.append(f"gene-emb {ck}: {e!r}"[:300])
    rs = lambda m, ck: f"{exp_id}/runs/{m}_{short(ck)}"  # noqa: E731
    pairs = []
    for v in VARIANTS:
        ck = ckpt_name(v)
        for m in ("scgpt_paper", "celljepa"):
            for base in ("scgpt_kidney_lr3e5", "jepa_kidney_lr3e5", "jepa_kidney_lr3e5_token"):
                pairs.append((rs(m, ck), rs(m, base)))
            pairs.append((rs(m, ck), "baselines/linear"))
        pairs.append((f"geneemb_linear/{short(ck)}", "geneemb_linear/scgpt_lr3e5"))
        pairs.append((f"geneemb_linear/{short(ck)}", "geneemb_linear/jepa_lr3e5"))
        pairs.append((f"geneemb_linear/{short(ck)}", "baselines/linear"))
    for m in ("scgpt_paper", "celljepa"):
        pairs.append((rs(m, ckpt_name("progmask")), rs(m, ckpt_name("ijepa"))))
    pairs.append((f"geneemb_linear/{short(ckpt_name('progmask'))}", f"geneemb_linear/{short(ckpt_name('ijepa'))}"))
    res = score_sets(sets, d["T"], d["R"], pairs)
    res.update({"exp_id": exp_id, "n_ref": len(d["train"]), "skipped": skipped, "test_conditions": d["test"]})
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/{tag}_{exp_id}_retrieval.json", "w"), indent=1)
    vol.commit()
    return res


@app.local_entrypoint()
def main(stage: str = "smoke", steps: int = 200):
    if stage == "smoke":
        ov = {**BASE, "epochs": 1, "max_cells_per_epoch": steps * BASE["batch_size"], "max_val_cells": 512,
              "log_every": 250, "max_minutes": 30.0}
        calls = {v: pretrain.spawn(v, f"smoke{steps}c_{v}", ov) for v in VARIANTS}
        for v, c in calls.items():
            r = c.get()
            print(json.dumps({v: {k: r.get(k) for k in ("status", "global_step", "best_val_recon", "history",
                                                          "elapsed_seconds_this_call", "peak_gpu_mem_gb")}},
                             indent=1, default=str), flush=True)
    elif stage == "eval":
        cks = BASELINE_CKPTS + [ckpt_name(v) for v in VARIANTS]
        calls = {e: evaluate.spawn(e, cks) for e in (ADAMSON_EXP, REPLOGLE_EXP)}
        os.makedirs("diag/results", exist_ok=True)
        for e, c in calls.items():
            r = c.get()
            json.dump(r, open(f"diag/results/X_{e}_retrieval.json", "w"), indent=1)
            print(e, "skipped:", r["skipped"])
    elif stage == "probe":
        names = BASELINE_CKPTS + [ckpt_name(v) for v in VARIANTS] + ["pca_bins"]
        calls = {n: program_probe.spawn(n) for n in names}
        os.makedirs("diag/results", exist_ok=True)
        out = {}
        for n, c in calls.items():
            try:
                out[n] = c.get()
                print(n, round(out[n]["mean_r2"], 4), flush=True)
            except Exception as e:
                print(n, "failed", repr(e)[:300])
        json.dump(out, open("diag/results/X_program_probe.json", "w"), indent=1)
    else:
        raise SystemExit(stage)
