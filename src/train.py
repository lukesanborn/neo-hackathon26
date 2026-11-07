"""Matched fine-tuning for the two Track-B arms and the pilot / benchmark checks.

Arms (everything except the auxiliary objective is shared and seeded identically):
  * "scgpt": scGPT perturbation fine-tuning (tutorial recipe; MSE on post-perturbation
    expression of sampled genes, control cell + perturbation flag as input).
  * "jepa":  same, plus our JEPA-style auxiliary loss: a predictor maps the student's pooled,
    intervention-conditioned representation to the frozen teacher's representation shift
    (treated-population centroid minus control centroid) for that TRAINING intervention.
    Loss = MSE + aux_weight * (1 - cos). Teacher targets come only from training cells.

Deviations from the upstream tutorial (applied to BOTH arms, recorded in config.json):
  * bf16 autocast instead of fp16 + GradScaler.
  * Perturbed gene positions are always kept when sub-sampling max_seq_len genes, so the
    intervention label cannot be dropped by sequence sub-sampling.
  * Control pairing is drawn once per training seed (GEARS also fixes one pairing).
  * Validation/test use a fixed draw of control inputs per condition (shared by all methods)
    rather than GEARS's per-cell random pairing; predictions are averaged per condition.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .data import CTRL, Dataset
from .evaluate import evaluate_predictions, summarize
from .models import (LatentPredictor, build_scgpt, cosine_aux_loss, encode_and_decode,
                     gene_vocab_ids, make_teacher, pool, state_hash)

METHODS = ("scgpt", "jepa")


@dataclass
class TrainConfig:
    method: str
    seed: int
    track: str = "B"
    epochs: int = 15
    batch_size: int = 64
    eval_batch_size: int = 32
    lr: float = 1e-4
    lr_step_gamma: float = 0.9          # StepLR(step_size=1 epoch), as in the tutorial
    max_seq_len: int = 1536
    grad_clip: float = 1.0
    dropout: float = 0.0
    precision: str = "bf16"
    # JEPA adaptation (ignored by "scgpt")
    aux_weight: float = 0.5
    teacher_cells_per_condition: int = 128
    teacher_ctrl_cells: int = 512
    predictor_seed_offset: int = 10_000
    # budget
    max_minutes: float = 150.0
    reserve_minutes: float = 15.0       # time kept for reload check + test evaluation
    max_steps_per_epoch: int | None = None
    checkpoint_selection: str = "min validation mean per-intervention delta RMSE"
    deviations: list = field(default_factory=lambda: [
        "bf16 autocast (tutorial: fp16 + GradScaler)",
        "perturbed genes always retained under max_seq_len sub-sampling",
        "fixed per-seed control pairing for training",
        "fixed shared control draw per held-out condition for evaluation",
        "public whole-human scGPT checkpoint (not Cell-JEPA kidney pretraining)",
    ])

    def __post_init__(self):
        assert self.method in METHODS, self.method


def env_info() -> dict:
    info = {
        "stack": os.environ.get("JEPA_STACK", "?"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "attention": "nn.TransformerEncoder (scaled_dot_product_attention); flash-attn package not installed",
        "sdp_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "sdp_mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
    }
    try:
        import scgpt
        info["scgpt"] = getattr(scgpt, "__version__", "?")
    except Exception as e:  # pragma: no cover
        info["scgpt"] = f"import error: {e}"
    try:
        import torchtext
        info["torchtext"] = torchtext.__version__
    except Exception:
        info["torchtext"] = None
    return info


class Trainer:
    def __init__(self, ds: Dataset, ckpt_dir: str, cfg: TrainConfig, device: str = "cuda", log=print):
        self.ds, self.cfg, self.device, self.log = ds, cfg, device, log
        self.model, self.vocab, self.ckpt_args, self.load_info = build_scgpt(ckpt_dir, cfg.seed, cfg.dropout)
        self.init_hash = state_hash(self.model)  # must match across arms for the same seed
        self.ckpt_dir = ckpt_dir
        self.model.to(device)
        self.d = self.ckpt_args["embsize"]
        gids, self.vocab_info = gene_vocab_ids(ds.gene_names, self.vocab)
        pad = self.vocab["<pad>"]
        all_conds = ds.split["train"] + ds.split["val"] + ds.split["test"]
        self.vocab_info["targets_mapped_to_pad"] = [c for c in all_conds if c != CTRL and gids[ds.gene_pos(c)] == pad]
        if self.vocab_info["targets_mapped_to_pad"]:
            raise RuntimeError(f"intervention targets lost in vocab mapping: {self.vocab_info['targets_mapped_to_pad']}")
        self.gene_ids = torch.as_tensor(gids, device=device)
        self.n_genes = ds.n_genes
        self.X = torch.as_tensor(np.asarray(ds.X), device=device)  # ~1.4 GB float32 on GPU
        self.amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.precision == "bf16")

        # ---- training examples (train split only; includes ctrl -> ctrl like GEARS) ----
        train_conds = list(ds.split["train"])
        cond_of_cell = ds.conditions
        is_train = np.isin(cond_of_cell, train_conds)
        self.train_cells = np.flatnonzero(is_train)
        rng = np.random.default_rng(cfg.seed)
        ctrl_idx = ds.ctrl_idx
        self.pair = rng.choice(ctrl_idx, size=len(self.train_cells), replace=True)
        pos_by_cond = {c: (-1 if c == CTRL else ds.gene_pos(c)) for c in train_conds}
        self.train_gpos = np.array([pos_by_cond[c] for c in cond_of_cell[self.train_cells]])
        self.pert_conds = [c for c in train_conds if c != CTRL]
        cid = {c: i for i, c in enumerate(self.pert_conds)}
        self.train_cid = np.array([cid.get(c, -1) for c in cond_of_cell[self.train_cells]])
        self.steps_per_epoch = math.ceil(len(self.train_cells) / cfg.batch_size)
        if cfg.max_steps_per_epoch:
            self.steps_per_epoch = min(self.steps_per_epoch, cfg.max_steps_per_epoch)

        self.predictor = None
        self.teacher = None
        self.teacher_diag = None
        if cfg.method == "jepa":
            self.teacher = make_teacher(self.model)
            self.targets, self.teacher_diag = self._teacher_targets()
            g = torch.Generator().manual_seed(cfg.seed + cfg.predictor_seed_offset)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
                self.predictor = LatentPredictor(self.d).to(device)
            # the teacher is only needed to build targets; free it afterwards
            self.teacher.cpu()

        params = list(self.model.parameters()) + (list(self.predictor.parameters()) if self.predictor else [])
        self.opt = torch.optim.Adam(params, lr=cfg.lr)
        self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=1, gamma=cfg.lr_step_gamma)
        self.gen = torch.Generator(device=device).manual_seed(cfg.seed)
        self.np_rng = np.random.default_rng(cfg.seed + 1)

    # ------------------------------------------------------------------ teacher
    @torch.no_grad()
    def _encode_pooled(self, net, cells: np.ndarray, bs: int) -> torch.Tensor:
        out = []
        src_all = self.gene_ids.unsqueeze(0)
        for i in range(0, len(cells), bs):
            vals = self.X[torch.as_tensor(cells[i:i + bs], device=self.device)]
            B = vals.shape[0]
            src = src_all.expand(B, -1)
            flags = torch.zeros(B, self.n_genes, dtype=torch.long, device=self.device)
            mask = torch.zeros(B, self.n_genes, dtype=torch.bool, device=self.device)
            with torch.autocast(**self.amp):
                h = net._encode(src, vals, flags, mask)
            out.append(pool(h).float())
        return torch.cat(out)

    def _teacher_targets(self):
        """Centroid of teacher embeddings of treated TRAINING cells per training intervention,
        minus the control centroid. Teacher input: observed treated expression, all genes,
        perturbation flags = 0 (it encodes the observed state, not the intervention)."""
        ds, cfg = self.ds, self.cfg
        forbidden = set(ds.split["val"]) | set(ds.split["test"])
        assert not (set(self.pert_conds) & forbidden), "teacher target from held-out condition"
        rng = np.random.default_rng(cfg.seed + 2)
        t0 = time.time()
        ctrl_cells = rng.choice(ds.ctrl_idx, size=min(cfg.teacher_ctrl_cells, len(ds.ctrl_idx)), replace=False)
        ctrl_c = self._encode_pooled(self.teacher, ctrl_cells, cfg.eval_batch_size).mean(0)
        cents = []
        for c in self.pert_conds:
            cells = ds.cells_of(c)
            assert np.isin(ds.conditions[cells], ds.split["train"]).all()
            cells = rng.choice(cells, size=min(cfg.teacher_cells_per_condition, len(cells)), replace=False)
            cents.append(self._encode_pooled(self.teacher, cells, cfg.eval_batch_size).mean(0))
        C = torch.stack(cents)
        shifts = C - ctrl_c
        # determinism check: same inputs twice through the frozen teacher
        probe = ctrl_cells[: cfg.eval_batch_size]
        a = self._encode_pooled(self.teacher, probe, cfg.eval_batch_size)
        b = self._encode_pooled(self.teacher, probe, cfg.eval_batch_size)
        sn = F.normalize(shifts, dim=-1)
        cos = sn @ sn.T
        off = cos[~torch.eye(len(cos), dtype=torch.bool, device=cos.device)]
        diag = {
            "n_target_conditions": len(self.pert_conds),
            "teacher_cells_per_condition": cfg.teacher_cells_per_condition,
            "centroid_norm_mean": float(C.norm(dim=-1).mean()),
            "shift_norm_mean": float(shifts.norm(dim=-1).mean()),
            "shift_norm_min": float(shifts.norm(dim=-1).min()),
            "shift_to_centroid_norm_ratio": float((shifts.norm(dim=-1) / C.norm(dim=-1)).mean()),
            "raw_centroid_mean_pairwise_cos": float((F.normalize(C, dim=-1) @ F.normalize(C, dim=-1).T)[
                ~torch.eye(len(C), dtype=torch.bool, device=C.device)].mean()),
            "shift_mean_pairwise_cos": float(off.mean()),
            "shift_pairwise_cos_min": float(off.min()),
            "shift_pairwise_cos_max": float(off.max()),
            "teacher_determinism_max_abs_diff": float((a - b).abs().max()),
            "teacher_requires_grad": any(p.requires_grad for p in self.teacher.parameters()),
            "seconds": time.time() - t0,
        }
        self.log(f"[teacher] {json.dumps(diag)}")
        return shifts.detach(), diag

    # ------------------------------------------------------------------ training
    def _batch(self, b: np.ndarray):
        cfg, dev = self.cfg, self.device
        B = len(b)
        x = self.X[torch.as_tensor(self.pair[b], device=dev)]
        y = self.X[torch.as_tensor(self.train_cells[b], device=dev)]
        gpos = torch.as_tensor(self.train_gpos[b], device=dev)
        has = gpos >= 0
        forced = torch.unique(gpos[has])
        if self.n_genes > cfg.max_seq_len:
            keep = torch.ones(self.n_genes, dtype=torch.bool, device=dev)
            keep[forced] = False
            rest = torch.nonzero(keep).squeeze(1)
            rest = rest[torch.randperm(len(rest), generator=self.gen, device=dev)[: cfg.max_seq_len - len(forced)]]
            sel = torch.sort(torch.cat([forced, rest]))[0]
        else:
            sel = torch.arange(self.n_genes, device=dev)
        flags_full = torch.zeros(B, self.n_genes, dtype=torch.long, device=dev)
        rows = torch.nonzero(has).squeeze(1)
        flags_full[rows, gpos[rows]] = 1
        flags = flags_full[:, sel]
        assert int(flags.sum()) == int(has.sum()), "perturbation flag lost by gene sub-sampling"
        src = self.gene_ids[sel].unsqueeze(0).expand(B, -1)
        mask = torch.zeros(B, len(sel), dtype=torch.bool, device=dev)
        return src, x[:, sel], flags, mask, y[:, sel]

    def train_step(self, b: np.ndarray) -> dict:
        self.model.train()
        if self.predictor is not None:
            self.predictor.train()
        src, vals, flags, mask, target = self._batch(b)
        with torch.autocast(**self.amp):
            h, pred = encode_and_decode(self.model, src, vals, flags, mask)
        loss_mse = F.mse_loss(pred.float(), target)
        out = {"loss_mse": float(loss_mse)}
        loss = loss_mse
        if self.predictor is not None:
            cid = torch.as_tensor(self.train_cid[b], device=self.device)
            sel = cid >= 0   # ctrl->ctrl examples have a zero shift (cosine undefined): no aux term
            if sel.any():
                z = self.predictor(pool(h[sel]).float())
                loss_aux = cosine_aux_loss(z, self.targets[cid[sel]])
                loss = loss + self.cfg.aux_weight * loss_aux
                out["loss_aux"] = float(loss_aux)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss {out}")
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + (list(self.predictor.parameters()) if self.predictor else []),
            self.cfg.grad_clip)
        self.opt.step()
        out["loss"] = float(loss)
        return out

    # ------------------------------------------------------------------ prediction
    @torch.no_grad()
    def predict(self, conditions: list[str], model=None, gene_override: dict | None = None,
                n_controls: int | None = None) -> dict:
        """Mean predicted profile per condition from the fixed shared control inputs.
        Uses ONLY control expression + intervention identity (no treated observations)."""
        model = model or self.model
        model.eval()
        bs = self.cfg.eval_batch_size
        src_all = self.gene_ids.unsqueeze(0)
        preds = {}
        for c in conditions:
            ctrl = np.asarray(self.ds.eval_controls[c][: n_controls or None])
            gpos = (gene_override or {}).get(c, self.ds.gene_pos(c))
            acc = torch.zeros(self.n_genes, dtype=torch.float64, device=self.device)
            for i in range(0, len(ctrl), bs):
                vals = self.X[torch.as_tensor(ctrl[i:i + bs], device=self.device)]
                B = vals.shape[0]
                flags = torch.zeros(B, self.n_genes, dtype=torch.long, device=self.device)
                flags[:, gpos] = 1
                mask = torch.zeros(B, self.n_genes, dtype=torch.bool, device=self.device)
                with torch.autocast(**self.amp):
                    _, p = encode_and_decode(model, src_all.expand(B, -1), vals, flags, mask)
                acc += p.double().sum(0)
            preds[c] = (acc / len(ctrl)).float().cpu().numpy()
        return preds

    def evaluate(self, split_name: str, model=None, preds=None):
        conds = self.ds.split[split_name]
        preds = preds if preds is not None else self.predict(conds, model=model)
        rows = evaluate_predictions(preds, self.ds, conds, self.ds.control_reference())
        return preds, rows, summarize(rows)

    # ------------------------------------------------------------------ checkpoint
    def save(self, path: str, extra: dict):
        torch.save({"model": self.model.state_dict(),
                    "predictor": self.predictor.state_dict() if self.predictor else None,
                    "config": asdict(self.cfg), **extra}, path)

    def fresh_model_from(self, path: str):
        from .models import build_scgpt as _b
        m, *_ = _b(self.ckpt_dir, self.cfg.seed, self.cfg.dropout)
        m.load_state_dict(torch.load(path, map_location="cpu")["model"])
        return m.to(self.device).eval()


# ============================================================================ full run


def run_training(ds: Dataset, ckpt_dir: str, cfg: TrainConfig, run_dir: str, commit=lambda: None) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    log_f = open(os.path.join(run_dir, "log.jsonl"), "a")

    def log(msg, **kv):
        rec = {"t": round(time.time() - t_start, 1), "msg": msg, **kv}
        print(json.dumps(rec), flush=True)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    t_start = time.time()
    torch.cuda.reset_peak_memory_stats()
    status = "running"
    _write(run_dir, "status.json", {"status": status, "started_unix": t_start})
    tr = Trainer(ds, ckpt_dir, cfg, log=lambda m: log(m))
    _write(run_dir, "config.json", {**asdict(cfg), "init_state_hash": tr.init_hash,
                                    "pretrained_load": tr.load_info, "vocab_mapping": tr.vocab_info,
                                    "steps_per_epoch": tr.steps_per_epoch,
                                    "teacher_diagnostics": tr.teacher_diag})
    _write(run_dir, "manifest.json", ds.manifest)
    _write(run_dir, "environment.json", env_info())
    commit()

    best = (math.inf, -1)
    ckpt_path = os.path.join(run_dir, "best.pt")
    best_val_preds = None
    budget_s = cfg.max_minutes * 60
    reserve_s = cfg.reserve_minutes * 60
    epochs_done = 0
    for epoch in range(cfg.epochs):
        t_ep = time.time()
        perm = tr.np_rng.permutation(len(tr.train_cells))
        sums, n = {}, 0
        for s in range(tr.steps_per_epoch):
            b = perm[s * cfg.batch_size:(s + 1) * cfg.batch_size]
            if len(b) == 0:
                break
            out = tr.train_step(b)
            for k, v in out.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
            if s % 200 == 0:
                log("step", epoch=epoch, step=s, **out)
        tr.sched.step()
        val_preds, _, val_sum = tr.evaluate("val")
        epochs_done = epoch + 1
        ep_s = time.time() - t_ep
        log("epoch", epoch=epoch, seconds=round(ep_s, 1), train={k: v / max(n, 1) for k, v in sums.items()},
            val_delta_rmse=val_sum["delta_rmse"], val_delta_pearson=val_sum["delta_pearson"])
        if val_sum["delta_rmse"] < best[0]:
            best = (val_sum["delta_rmse"], epoch)
            best_val_preds = val_preds
            tr.save(ckpt_path, {"epoch": epoch, "val": val_sum})
            log("checkpoint", epoch=epoch, val_delta_rmse=best[0])
            commit()
        elapsed = time.time() - t_start
        if epoch + 1 < cfg.epochs and elapsed + ep_s + reserve_s > budget_s:
            status = "stopped_by_time_budget"
            log("stopping early: next epoch would exceed the time budget", elapsed=elapsed)
            break

    # ---- reload selected checkpoint; verify it reproduces its validation predictions ----
    model = tr.fresh_model_from(ckpt_path)
    re_preds = tr.predict(ds.split["val"], model=model)
    reload_diff = max(float(np.abs(re_preds[c] - best_val_preds[c]).max()) for c in re_preds)
    log("reload check", max_abs_diff=reload_diff)

    # ---- single test evaluation of the selected checkpoint ----
    test_preds, test_rows, test_sum = tr.evaluate("test", model=model)
    c0 = ds.split["test"][0]
    other = ds.gene_pos(ds.split["train"][0] if ds.split["train"][0] != CTRL else ds.split["train"][1])
    alt = tr.predict([c0], model=model, gene_override={c0: other}, n_controls=64)[c0]
    ref = tr.predict([c0], model=model, n_controls=64)[c0]
    sensitivity = float(np.abs(alt - ref).mean())

    if status == "running":
        status = "completed"
    metrics = {
        "method": cfg.method, "track": cfg.track, "seed": cfg.seed,
        "checkpoint": "public scGPT_human -> fine-tuned; selected epoch %d" % best[1],
        "split_hash": ds.manifest["split_hash"], "manifest_sha": ds.manifest["h5ad_sha256"],
        "epochs_completed": epochs_done, "selected_epoch": best[1], "val_delta_rmse_selected": best[0],
        "test": test_sum, "elapsed_seconds": time.time() - t_start,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "reload_max_abs_diff": reload_diff, "reload_tolerance": 1e-3, "reload_ok": reload_diff <= 1e-3,
        "intervention_sensitivity_mean_abs": sensitivity,
        "status": status, "deviations": cfg.deviations, "environment": env_info(),
        "init_state_hash": tr.init_hash,
    }
    _write(run_dir, "metrics.json", metrics)
    _write_rows(os.path.join(run_dir, "per_intervention.csv"), test_rows)
    np.savez_compressed(os.path.join(run_dir, "test_predictions.npz"),
                        conditions=np.array(list(test_preds)), preds=np.stack(list(test_preds.values())))
    _write(run_dir, "status.json", {"status": status, "finished_unix": time.time()})
    log("done", **{k: metrics[k] for k in ["status", "epochs_completed", "selected_epoch"]},
        test_delta_rmse=test_sum["delta_rmse"])
    log_f.close()
    commit()
    return metrics


# ============================================================================ pilot / benchmark


def run_pilot(ds: Dataset, ckpt_dir: str, methods=METHODS, steps: int = 150, warmup: int = 20,
              seed: int = 1, aux_weight: float = 0.5, out_dir: str | None = None) -> dict:
    """Execution + timing gate on one GPU. Also emits a numerical fingerprint used to compare
    software stacks / GPUs (prediction from the untrained, pretrained-loaded model)."""
    report = {"environment": env_info(), "methods": {}}
    hashes = {}
    for method in methods:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        cfg = TrainConfig(method=method, seed=seed, aux_weight=aux_weight)
        tr = Trainer(ds, ckpt_dir, cfg)
        setup_s = time.time() - t0
        hashes[method] = tr.init_hash
        r = {"setup_seconds": setup_s, "init_state_hash": tr.init_hash, "pretrained_load": tr.load_info,
             "vocab_mapping": tr.vocab_info,
             "teacher_diagnostics": tr.teacher_diag, "steps_per_epoch_full": tr.steps_per_epoch}

        val_c = ds.split["val"][0]
        fp = tr.predict([val_c], n_controls=32)[val_c]
        r["fingerprint"] = {"condition": val_c, "pred_first16": fp[:16].tolist(),
                            "pred_mean": float(fp.mean()), "pred_l2": float(np.linalg.norm(fp))}

        perm = tr.np_rng.permutation(len(tr.train_cells))
        losses = []
        for s in range(warmup + steps):
            if s == warmup:
                torch.cuda.synchronize()
                t_steps = time.time()
            b = perm[s * cfg.batch_size:(s + 1) * cfg.batch_size]
            losses.append(tr.train_step(b))
        torch.cuda.synchronize()
        step_s = (time.time() - t_steps) / steps
        r["first_losses"] = losses[:3]
        r["last_losses"] = losses[-3:]
        r["all_losses_finite"] = all(math.isfinite(l["loss"]) for l in losses)
        r["seconds_per_step"] = step_s
        r["cells_per_second"] = cfg.batch_size / step_s

        torch.cuda.synchronize()
        t_val = time.time()
        val_preds, _, val_sum = tr.evaluate("val")
        torch.cuda.synchronize()
        r["validation_seconds"] = time.time() - t_val
        r["val_after_pilot_steps"] = val_sum

        if tr.teacher is not None:
            r["teacher_has_grads"] = any(p.grad is not None for p in tr.teacher.parameters())
            r["teacher_requires_grad"] = any(p.requires_grad for p in tr.teacher.parameters())

        path = os.path.join(out_dir or "/tmp", f"pilot_{method}.pt")
        tr.save(path, {"epoch": -1})
        m2 = tr.fresh_model_from(path)
        p2 = tr.predict(ds.split["val"], model=m2)
        r["reload_max_abs_diff"] = max(float(np.abs(p2[c] - val_preds[c]).max()) for c in p2)
        if out_dir is None:
            os.remove(path)

        alt_gene = ds.gene_pos(ds.split["test"][0])
        alt = tr.predict([val_c], gene_override={val_c: alt_gene}, n_controls=32)[val_c]
        ref = tr.predict([val_c], n_controls=32)[val_c]
        r["intervention_sensitivity_mean_abs"] = float(np.abs(alt - ref).mean())
        r["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        r["projected_epoch_seconds"] = tr.steps_per_epoch * step_s + r["validation_seconds"]
        r["projected_15_epoch_minutes"] = (15 * r["projected_epoch_seconds"]) / 60
        report["methods"][method] = r
        print(json.dumps({method: {k: v for k, v in r.items() if k not in ("pretrained_load",)}}, default=str), flush=True)
        del tr, m2
        torch.cuda.empty_cache()

    report["init_hash_matched_across_arms"] = len(set(hashes.values())) == 1
    gates = {
        "finite_losses": all(m["all_losses_finite"] for m in report["methods"].values()),
        "reload_ok": all(m["reload_max_abs_diff"] <= 1e-3 for m in report["methods"].values()),
        "intervention_changes_prediction": all(m["intervention_sensitivity_mean_abs"] > 0
                                               for m in report["methods"].values()),
        "matched_init": report["init_hash_matched_across_arms"],
    }
    if "jepa" in report["methods"]:
        j = report["methods"]["jepa"]
        gates["teacher_frozen"] = not j.get("teacher_has_grads", True) and not j.get("teacher_requires_grad", True)
        gates["teacher_targets_vary"] = j["teacher_diagnostics"]["shift_mean_pairwise_cos"] < 0.95
    report["gates"] = gates
    report["all_gates_pass"] = all(gates.values())
    return report


def _write(d, name, obj):
    with open(os.path.join(d, name), "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _write_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
