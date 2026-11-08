"""Matched fine-tuning arms (Track-B scgpt / jepa, plus the Cell-JEPA FT recipe) and pilot checks.

Arms (everything except the auxiliary objective is shared and seeded identically):
  * "scgpt": scGPT perturbation fine-tuning (tutorial recipe; MSE on post-perturbation
    expression of sampled genes, control cell + perturbation flag as input).
  * "jepa":  same, plus our JEPA-style auxiliary loss: a predictor maps the student's pooled,
    intervention-conditioned representation to the frozen teacher's representation shift
    (treated-population centroid minus control centroid) for that TRAINING intervention.
    Loss = MSE + aux_weight * (1 - cos). Teacher targets come only from training cells.
  * "celljepa": Cell-JEPA's perturbation fine-tuning recipe, reimplemented from the paper
    (arXiv 2602.02093, §2.2 / App. E.3; UNOFFICIAL — no code was released). A <cls> token is
    prepended (cell embedding = output[:, 0]). Student input = gene token + perturbation flag +
    paired CONTROL expression; teacher (a frozen copy of the initial student, or optionally an
    EMA of it) input = gene token + the SAME perturbation flags + the paired ground-truth
    PERTURBED expression, on the same sampled gene subset. Per-cell targets (not memorisable
    centroids). Loss = MSE (genes only, <cls> excluded) + jepa_ft_weight * (1 - cos(
    predictor(projector(student cls)), sg(teacher cls))) + ecs_weight * scGPT-ECS(student cls).
    Weights 1.0 / 1.0 / 0.8, Adam 1e-4 x0.9/epoch, bs 64, 15 epochs, no masking (as the paper).
    The teacher, projector and predictor are training-only: inference uses the student alone.
    Deviations: see CELLJEPA_DEVIATIONS (recorded in config.json / metrics.json).
  * "scgpt_paper": the paper's scGPT fine-tuning baseline (§2.5 / App. E.3): the "celljepa" objective
    without the JEPA term, i.e. Loss = MSE (genes only) + ecs_weight * scGPT-ECS(student <cls>), weights
    1.0 / 0.8. Identical code path, <cls> handling, batches and RNG streams as "celljepa" (no teacher,
    projector or predictor), so Track A's comparison isolates the JEPA term. Deviations:
    SCGPT_PAPER_DEVIATIONS. (Track B's "scgpt" above stays MSE-only, without <cls>.)

input_transform:
  * "log1p" (default): GEARS log-normalised expression everywhere (Track B behaviour).
  * "bins50": for checkpoints pretrained on per-cell binned values (kidney checkpoints). The
    ENCODER input (student control values and, for celljepa, teacher perturbed values) is the
    per-cell 50-bin quantile transform of the full cell vector (models.bin_expression; computed
    once over all genes, before gene sub-sampling, like scGPT preprocessing). Recon targets, the
    decoder's affine `coeff * values + bias` term and therefore predictions stay in log1p space.
    Justification: per-cell bins are not invertible (the bin->value map differs per cell and
    depends on the unseen perturbed cell), so predicting bins would need an ad-hoc inverse; the
    expression decoder is newly initialised anyway, and keeping targets in log1p leaves the
    evaluation protocol and metrics directly comparable with every other arm.

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

from .data import CTRL, Dataset, percell_pairs
from .evaluate import PERCELL_N_CELLS, evaluate_predictions, percell_metrics, summarize, summarize_percell
from .models import (LatentPredictor, Projector, bin_expression, build_scgpt, cosine_aux_loss,
                     ema_update, encode_and_decode, gene_vocab_ids, make_teacher, pool,
                     prepend_cls, scgpt_ecs_loss, state_hash)

METHODS = ("scgpt", "jepa", "celljepa", "scgpt_paper")
CLS_METHODS = ("celljepa", "scgpt_paper")   # <cls>-prepended recipes sharing Trainer._train_step_cls
INPUT_TRANSFORMS = ("log1p", "bins50")
TEACHER_UPDATES = ("frozen", "ema")
DEFAULT_CKPT = "scGPT_human"
CELLJEPA_LABEL = "Cell-JEPA perturbation fine-tuning reimplemented from the paper (unofficial)"
ECS_DEVIATION = ("ECS: scGPT's unsupervised ECS (TransformerGenerator ECS=True, threshold 0.3) on student <cls> "
                 "embeddings instead of the paper's supervised-contrastive ECS (Adamson has no cell-type positives)")
CLS_DEVIATION = "<cls> token prepended (value 0, pert flag 0); reconstruction loss and predictions exclude it"
CELLJEPA_DEVIATIONS = [
    ECS_DEVIATION,
    "control<->perturbed pairing unspecified in the paper: fixed per-seed random pairing (Trainer.pair); "
    "for ctrl->ctrl examples the teacher target is the paired control cell",
    "projector (Linear-GELU-Linear) and predictor (LayerNorm-MLP) architectures unspecified in the paper; "
    "randomly initialised from a separate RNG stream",
    CLS_DEVIATION,
    "teacher = frozen copy of the initial student (App. E.3) by default; EMA (§2.2) optional",
]
SCGPT_PAPER_LABEL = ("scGPT perturbation fine-tuning as the paper's baseline (MSE + 0.8·ECS on <cls>), "
                     "reimplemented (unofficial)")
SCGPT_PAPER_DEVIATIONS = [ECS_DEVIATION, CLS_DEVIATION]
METHOD_LABELS = {"celljepa": CELLJEPA_LABEL, "scgpt_paper": SCGPT_PAPER_LABEL}
METHOD_DEVIATIONS = {"celljepa": CELLJEPA_DEVIATIONS, "scgpt_paper": SCGPT_PAPER_DEVIATIONS}


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
    # checkpoint / input (shared by all methods; defaults reproduce Track B)
    ckpt: str = DEFAULT_CKPT            # dir name under /vol/checkpoints/ or absolute path
    input_transform: str = "log1p"      # "log1p" | "bins50" (see module docstring)
    # Cell-JEPA FT recipe (method "celljepa"; ECS settings also used by "scgpt_paper")
    teacher_update: str = "frozen"      # "frozen" | "ema"
    ema_momentum: float = 0.996
    jepa_ft_weight: float = 1.0
    ecs_weight: float = 0.8
    ecs_threshold: float = 0.3
    n_bins: int = 50
    # JEPA adaptation (ignored by "scgpt")
    aux_weight: float = 0.5
    teacher_cells_per_condition: int = 128
    teacher_ctrl_cells: int = 512
    predictor_seed_offset: int = 10_000
    # budget
    max_minutes: float = 150.0
    reserve_minutes: float = 15.0       # time kept for reload check + test evaluation
    max_steps_per_epoch: int | None = None
    # retrieval (A24): checkpoint selection and an optional contrastive retrieval loss; defaults = unchanged
    selection: str = "delta_rmse"       # "delta_rmse" | "retrieval" (max validation identification percentile)
    retr_weight: float = 0.0            # >0: InfoNCE on predicted deltas vs TRAINING-perturbation mean deltas
    retr_tau: float = 0.1
    random_init: bool = False           # A28 ablation: do not load pretrained weights (architecture of `ckpt`)
    checkpoint_selection: str = "min validation mean per-intervention delta RMSE"
    deviations: list = field(default_factory=lambda: [
        "bf16 autocast (tutorial: fp16 + GradScaler)",
        "perturbed genes always retained under max_seq_len sub-sampling",
        "fixed per-seed control pairing for training",
        "fixed shared control draw per held-out condition for evaluation",
        "public whole-human scGPT checkpoint (not Cell-JEPA kidney pretraining)",
    ])

    @property
    def default_ckpt(self) -> bool:
        """True for the public scGPT_human checkpoint, by name or by its Volume path."""
        return self.ckpt.rstrip("/") in (DEFAULT_CKPT, f"/vol/checkpoints/{DEFAULT_CKPT}")

    def __post_init__(self):
        assert self.method in METHODS, self.method
        # Runs initialised from our kidney pretraining belong to Track A (Cell-JEPA reimplementation).
        if self.track == "B" and "/trackA/" in self.ckpt:
            self.track = "A"
        assert self.input_transform in INPUT_TRANSFORMS, self.input_transform
        assert self.teacher_update in TEACHER_UPDATES, self.teacher_update
        assert self.selection in ("delta_rmse", "retrieval"), self.selection
        if self.selection == "retrieval":
            self.checkpoint_selection = ("validation retrieval: max identification percentile (id_pct) of predicted vs observed "
                                         "deltas among validation conditions, 500-gene panel from training deltas")
        # extra deviations only for non-default settings, so default configs match Track B exactly
        dev = list(self.deviations)
        pub = "public whole-human scGPT checkpoint (not Cell-JEPA kidney pretraining)"
        if not self.default_ckpt and pub in dev:
            dev[dev.index(pub)] = f"initial checkpoint {self.ckpt!r} (our pretraining, not the authors' checkpoint)"
        bins = (f"encoder inputs binned per cell ({self.n_bins} quantile bins over all genes, before "
                "gene sub-sampling); recon targets, decoder affine term and predictions stay in log1p")
        if self.input_transform == "bins50" and bins not in dev:
            dev.append(bins)
        dev += [d for d in METHOD_DEVIATIONS.get(self.method, ()) if d not in dev]
        if self.retr_weight > 0:
            dev.append(f"contrastive retrieval loss (weight {self.retr_weight}, tau {self.retr_tau}): predicted "
                       "per-cell deltas vs training-perturbation mean deltas (training conditions only)")
        if self.selection == "retrieval":
            dev.append("checkpoint selected by validation retrieval (id_pct), not delta RMSE")
        if self.random_init:
            dev.append("ABLATION: no pretrained weights loaded (random init of the same architecture)")
        self.deviations = dev


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
        self.model, self.vocab, self.ckpt_args, self.load_info = build_scgpt(ckpt_dir, cfg.seed, cfg.dropout, load=not cfg.random_init)
        ck_style = self.ckpt_args.get("input_style")
        if ck_style is not None:
            self.load_info["ckpt_input_style"] = ck_style
            if not cfg.default_ckpt and ("bin" in str(ck_style)) != (cfg.input_transform == "bins50"):
                log(f"[warn] checkpoint input_style={ck_style!r} but input_transform={cfg.input_transform!r}")
        self.init_hash = state_hash(self.model)  # must match across arms for the same seed
        self.ckpt_dir = ckpt_dir
        self.model.to(device)
        self.d = self.ckpt_args["embsize"]
        gids, self.vocab_info = gene_vocab_ids(ds.gene_names, self.vocab,
                                               extra_aliases=ds.manifest.get("extra_hgnc_aliases"))
        pad = self.vocab["<pad>"]
        all_conds = ds.split["train"] + ds.split["val"] + ds.split["test"]
        self.vocab_info["targets_mapped_to_pad"] = [c for c in all_conds if c != CTRL and gids[ds.gene_pos(c)] == pad]
        if self.vocab_info["targets_mapped_to_pad"]:
            raise RuntimeError(f"intervention targets lost in vocab mapping: {self.vocab_info['targets_mapped_to_pad']}")
        self.gene_ids = torch.as_tensor(gids, device=device)
        self.n_genes = ds.n_genes
        self.X = torch.as_tensor(np.asarray(ds.X), device=device)  # ~1.4 GB float32 on GPU
        # encoder-input expression: log1p (same tensor) or per-cell bins (precomputed once, all genes)
        self.X_in = self.X if cfg.input_transform == "log1p" else bin_expression(self.X, cfg.n_bins)
        self.use_cls = cfg.method in CLS_METHODS
        self.cls_id = self.vocab["<cls>"]
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
        if cfg.retr_weight > 0:   # training-perturbation mean deltas (training cells only)
            ctrl_ref = self.X[torch.as_tensor(ctrl_idx, device=device)].mean(0)
            self.retr_ctrl = ctrl_ref
            self.retr_R = torch.stack([self.X[torch.as_tensor(np.flatnonzero(cond_of_cell == c), device=device)].mean(0)
                                       - ctrl_ref for c in self.pert_conds])
        self.steps_per_epoch = math.ceil(len(self.train_cells) / cfg.batch_size)
        if cfg.max_steps_per_epoch:
            self.steps_per_epoch = min(self.steps_per_epoch, cfg.max_steps_per_epoch)

        self.predictor = None
        self.projector = None
        self.teacher = None
        self.teacher_diag = None
        if cfg.method == "celljepa":
            self.teacher = make_teacher(self.model)   # copy of the initial student (stays on device)
            self.teacher_init_hash = state_hash(self.teacher)
            g = torch.Generator().manual_seed(cfg.seed + cfg.predictor_seed_offset)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
                self.projector = Projector(self.d).to(device)
                self.predictor = LatentPredictor(self.d).to(device)
            self.teacher_diag = {"teacher_update": cfg.teacher_update,
                                 "ema_momentum": cfg.ema_momentum if cfg.teacher_update == "ema" else None,
                                 "teacher_init_hash": self.teacher_init_hash,
                                 "teacher_requires_grad": any(p.requires_grad for p in self.teacher.parameters())}
        elif cfg.method == "jepa":
            self.teacher = make_teacher(self.model)
            self.targets, self.teacher_diag = self._teacher_targets()
            g = torch.Generator().manual_seed(cfg.seed + cfg.predictor_seed_offset)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
                self.predictor = LatentPredictor(self.d).to(device)
            # the teacher is only needed to build targets; free it afterwards
            self.teacher.cpu()

        self.opt = torch.optim.Adam(self.trainable_params(), lr=cfg.lr)
        self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=1, gamma=cfg.lr_step_gamma)
        self.gen = torch.Generator(device=device).manual_seed(cfg.seed)
        self.np_rng = np.random.default_rng(cfg.seed + 1)

    def trainable_params(self) -> list:
        heads = [m for m in (self.predictor, self.projector) if m is not None]
        return list(self.model.parameters()) + [p for m in heads for p in m.parameters()]

    # ------------------------------------------------------------------ teacher
    @torch.no_grad()
    def _encode_pooled(self, net, cells: np.ndarray, bs: int) -> torch.Tensor:
        out = []
        src_all = self.gene_ids.unsqueeze(0)
        for i in range(0, len(cells), bs):
            vals = self.X_in[torch.as_tensor(cells[i:i + bs], device=self.device)]
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
        pair_t = torch.as_tensor(self.pair[b], device=dev)
        cell_t = torch.as_tensor(self.train_cells[b], device=dev)
        x = self.X_in[pair_t]
        y = self.X[cell_t]
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
        y_sel = y[:, sel]
        binned = self.X_in is not self.X
        extra = {
            # decoder affine term in log1p space (None: same tensor as the encoder input)
            "dec_vals": self.X[pair_t[:, None], sel] if binned else None,
            # Cell-JEPA teacher input: the paired PERTURBED cell, same genes (encoder space)
            "teacher_vals": (self.X_in[cell_t[:, None], sel] if binned else y_sel) if self.cfg.method == "celljepa" else None,
            "sel": sel,
        }
        return src, x[:, sel], flags, mask, y_sel, extra

    def _retr_loss(self, pred_genes, b: np.ndarray, sel):
        """InfoNCE: each perturbed training cell's predicted delta should be closest (cosine) to the mean
        delta of its OWN training perturbation among all training perturbations. None if no such cells."""
        cid = torch.as_tensor(self.train_cid[b], device=self.device)
        m = cid >= 0
        if not m.any():
            return None
        d = F.normalize(pred_genes[m].float() - self.retr_ctrl[sel], dim=-1)
        R = F.normalize(self.retr_R[:, sel], dim=-1)
        return F.cross_entropy(d @ R.T / self.cfg.retr_tau, cid[m])

    def train_step(self, b: np.ndarray) -> dict:
        if self.use_cls:
            return self._train_step_cls(b, with_jepa=self.cfg.method == "celljepa")
        self.model.train()
        if self.predictor is not None:
            self.predictor.train()
        src, vals, flags, mask, target, ex = self._batch(b)
        with torch.autocast(**self.amp):
            h, pred = encode_and_decode(self.model, src, vals, flags, mask, ex["dec_vals"])
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
        if self.cfg.retr_weight > 0:
            lr_ = self._retr_loss(pred, b, ex["sel"])
            if lr_ is not None:
                loss = loss + self.cfg.retr_weight * lr_
                out["loss_retr"] = float(lr_)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss {out}")
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trainable_params(), self.cfg.grad_clip)
        self.opt.step()
        out["loss"] = float(loss)
        return out

    def _train_step_cls(self, b: np.ndarray, with_jepa: bool) -> dict:
        """<cls>-prepended step shared by "celljepa" (with_jepa=True: MSE + JEPA + ECS) and
        "scgpt_paper" (with_jepa=False: the paper's scGPT baseline, MSE + ECS; no teacher/heads)."""
        cfg = self.cfg
        self.model.train()
        if with_jepa:
            self.projector.train()
            self.predictor.train()
            self.teacher.eval()
        src, vals, flags, mask, target, ex = self._batch(b)
        s_src, s_vals, s_flags, s_mask, s_dec = prepend_cls(src, vals, flags, mask, self.cls_id, ex["dec_vals"])
        if with_jepa:
            t_vals = prepend_cls(src, ex["teacher_vals"], flags, mask, self.cls_id)[1]
        with torch.autocast(**self.amp):
            h, pred = encode_and_decode(self.model, s_src, s_vals, s_flags, s_mask, s_dec)
            if with_jepa:
                with torch.no_grad():
                    t_cls = self.teacher._encode(s_src, t_vals, s_flags, s_mask)[:, 0]
        loss_mse = F.mse_loss(pred[:, 1:].float(), target)
        s_cls = h[:, 0].float()
        out = {"loss_mse": float(loss_mse)}
        # op order (JEPA head before ECS) kept as in the original celljepa step: it fixes the autograd
        # accumulation order, so celljepa stays bit-identical
        if with_jepa:
            z = self.predictor(self.projector(s_cls))
            loss_jepa = cosine_aux_loss(z, t_cls.float())
            out["loss_jepa"] = float(loss_jepa)
        loss_ecs = scgpt_ecs_loss(s_cls, cfg.ecs_threshold)
        out["loss_ecs"] = float(loss_ecs)
        if not with_jepa:   # scgpt_paper: the paper's scGPT baseline
            loss = loss_mse + cfg.ecs_weight * loss_ecs
        else:
            loss = loss_mse + cfg.jepa_ft_weight * loss_jepa + cfg.ecs_weight * loss_ecs
            with torch.no_grad():   # diagnostics: how much per-cell signal the teacher targets carry
                t = F.normalize(t_cls.float(), dim=-1)
                # JEPA loss of a trivial predictor that always outputs the batch-mean target
                out["jepa_meanpred_baseline"] = float(cosine_aux_loss(t.mean(0, keepdim=True).expand_as(t), t))
                off = ~torch.eye(len(t), dtype=torch.bool, device=t.device)
                out["teacher_cls_pairwise_cos"] = float((t @ t.T)[off].mean()) if len(t) > 1 else float("nan")
        if cfg.retr_weight > 0:
            lr_ = self._retr_loss(pred[:, 1:], b, ex["sel"])
            if lr_ is not None:
                loss = loss + cfg.retr_weight * lr_
                out["loss_retr"] = float(lr_)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss {out}")
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trainable_params(), cfg.grad_clip)
        self.opt.step()
        if with_jepa and cfg.teacher_update == "ema":
            ema_update(self.teacher, self.model, cfg.ema_momentum)
        out["loss"] = float(loss)
        return out

    # ------------------------------------------------------------------ prediction
    @torch.no_grad()
    @torch.no_grad()
    def _forward_controls(self, model, ctrl_cells: np.ndarray, gpos: int) -> torch.Tensor:
        """Predicted post-perturbation profiles (B, n_genes) for control cells + target gene."""
        idx = torch.as_tensor(ctrl_cells, device=self.device)
        vals = self.X_in[idx]
        dec = None if self.X_in is self.X else self.X[idx]
        B = vals.shape[0]
        flags = torch.zeros(B, self.n_genes, dtype=torch.long, device=self.device)
        flags[:, gpos] = 1
        mask = torch.zeros(B, self.n_genes, dtype=torch.bool, device=self.device)
        src = self.gene_ids.unsqueeze(0).expand(B, -1)
        if self.use_cls:   # celljepa / scgpt_paper: <cls> at position 0, dropped from the output
            src, vals, flags, mask, dec = prepend_cls(src, vals, flags, mask, self.cls_id, dec)
        with torch.autocast(**self.amp):
            _, p = encode_and_decode(model, src, vals, flags, mask, dec)
        return p[:, 1:] if self.use_cls else p

    @torch.no_grad()
    def predict(self, conditions: list[str], model=None, gene_override: dict | None = None,
                n_controls: int | None = None) -> dict:
        """Mean predicted profile per condition from the fixed shared control inputs.
        Uses ONLY control expression + intervention identity (no treated observations)."""
        model = model or self.model
        model.eval()
        bs = self.cfg.eval_batch_size
        preds = {}
        for c in conditions:
            ctrl = np.asarray(self.ds.eval_controls[c][: n_controls or None])
            gpos = (gene_override or {}).get(c, self.ds.gene_pos(c))
            acc = torch.zeros(self.n_genes, dtype=torch.float64, device=self.device)
            for i in range(0, len(ctrl), bs):
                acc += self._forward_controls(model, ctrl[i:i + bs], gpos).double().sum(0)
            preds[c] = (acc / len(ctrl)).float().cpu().numpy()
        return preds

    @torch.no_grad()
    def evaluate_per_cell(self, split_name: str, model=None, n_cells: int = PERCELL_N_CELLS):
        """Paper-style per-cell scoring (see src/evaluate.py:percell_metrics).

        For each held-out condition, up to n_cells treated cells are each paired with one control
        cell (fixed draw shared by every method, src/data.py:percell_pairs); the model predicts from
        the control input only, and correlations are computed per cell then averaged."""
        model = model or self.model
        model.eval()
        bs = self.cfg.eval_batch_size
        rows = []
        de_rank = getattr(self.ds, "de_rank20", None) or {}
        for c in self.ds.split[split_name]:
            treated, ctrl = percell_pairs(self.ds, c, n_cells)
            gpos = self.ds.gene_pos(c)
            pred = torch.cat([self._forward_controls(model, ctrl[i:i + bs], gpos).float()
                              for i in range(0, len(ctrl), bs)]).cpu().numpy()
            X = np.asarray(self.ds.X)
            m = percell_metrics(pred, X[treated], X[ctrl], de_rank.get(c), self.ds.de_idx[c])
            rows.append({"condition": c, **m})
        return rows, summarize_percell(rows)

    def evaluate(self, split_name: str, model=None, preds=None):
        conds = self.ds.split[split_name]
        preds = preds if preds is not None else self.predict(conds, model=model)
        rows = evaluate_predictions(preds, self.ds, conds, self.ds.control_reference())
        return preds, rows, summarize(rows)

    # ------------------------------------------------------------------ checkpoint
    def save(self, path: str, extra: dict):
        torch.save({"model": self.model.state_dict(),
                    "predictor": self.predictor.state_dict() if self.predictor else None,
                    **({"projector": self.projector.state_dict()} if self.projector else {}),
                    "config": asdict(self.cfg), **extra}, path)

    def fresh_model_from(self, path: str):
        from .models import build_scgpt as _b
        m, *_ = _b(self.ckpt_dir, self.cfg.seed, self.cfg.dropout)
        m.load_state_dict(torch.load(path, map_location="cpu")["model"])
        return m.to(self.device).eval()


# ============================================================================ full run


def run_training(ds: Dataset, ckpt_dir: str, cfg: TrainConfig, run_dir: str, commit=lambda: None) -> dict:
    """`ckpt_dir` is the resolved directory of `cfg.ckpt` (args.json / best_model.pt / vocab.json)."""
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
    _write(run_dir, "config.json", {**asdict(cfg), "ckpt_dir": ckpt_dir, "init_state_hash": tr.init_hash,
                                    "pretrained_load": tr.load_info, "vocab_mapping": tr.vocab_info,
                                    "steps_per_epoch": tr.steps_per_epoch,
                                    "teacher_diagnostics": tr.teacher_diag})
    _write(run_dir, "manifest.json", ds.manifest)
    _write(run_dir, "environment.json", env_info())
    commit()

    # validation retrieval (always logged; used for selection when cfg.selection == "retrieval"):
    # deltas vs the control reference, on the 500-gene panel of highest TRAINING-delta variance
    from .retrieval import id_scores, panel_genes
    ctrl_ref = np.asarray(ds.control_reference())
    train_perts = [c for c in ds.split["train"] if c != CTRL]
    R_train = np.stack([np.asarray(ds.observed_mean(c)) - ctrl_ref for c in train_perts])
    panel = panel_genes(R_train)
    val_conds = list(ds.split["val"])
    T_val = np.stack([np.asarray(ds.observed_mean(c)) - ctrl_ref for c in val_conds])[:, panel]

    def val_retrieval(preds):
        if len(val_conds) < 2:
            return {}
        P = np.stack([np.asarray(preds[c]) - ctrl_ref for c in val_conds])[:, panel]
        return id_scores(P, T_val)

    best = (math.inf, -1)   # (criterion to minimise, epoch)
    best_val_retr = {}
    best_val_sum = {}
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
        vr = val_retrieval(val_preds)
        log("epoch", epoch=epoch, seconds=round(ep_s, 1), train={k: v / max(n, 1) for k, v in sums.items()},
            val_delta_rmse=val_sum["delta_rmse"], val_delta_pearson=val_sum["delta_pearson"],
            **{f"val_{k}": v for k, v in vr.items()})
        crit = -vr["id_pct"] if (cfg.selection == "retrieval" and vr) else val_sum["delta_rmse"]
        if crit < best[0]:
            best = (crit, epoch)
            best_val_preds = val_preds
            best_val_retr = vr
            best_val_sum = val_sum
            tr.save(ckpt_path, {"epoch": epoch, "val": val_sum, "val_retrieval": vr})
            log("checkpoint", epoch=epoch, val_delta_rmse=val_sum["delta_rmse"], **{f"val_{k}": v for k, v in vr.items()})
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
    pc_rows, pc_sum = tr.evaluate_per_cell("test", model=model)
    log("per-cell test metrics", **{k: v for k, v in pc_sum.items() if not k.endswith("_n_defined")})
    c0 = ds.split["test"][0]
    other = ds.gene_pos(ds.split["train"][0] if ds.split["train"][0] != CTRL else ds.split["train"][1])
    alt = tr.predict([c0], model=model, gene_override={c0: other}, n_controls=64)[c0]
    ref = tr.predict([c0], model=model, n_controls=64)[c0]
    sensitivity = float(np.abs(alt - ref).mean())

    if status == "running":
        status = "completed"
    metrics = {
        "method": cfg.method, "track": cfg.track, "seed": cfg.seed,
        "checkpoint": ("public scGPT_human" if cfg.default_ckpt else cfg.ckpt)
                      + " -> fine-tuned; selected epoch %d" % best[1],
        "split_hash": ds.manifest["split_hash"], "manifest_sha": ds.manifest["h5ad_sha256"],
        "epochs_completed": epochs_done, "selected_epoch": best[1],
        "val_delta_rmse_selected": best_val_sum.get("delta_rmse"), "selection": cfg.selection,
        "val_retrieval_selected": best_val_retr,
        "test": test_sum, "test_percell": pc_sum, "elapsed_seconds": time.time() - t_start,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "reload_max_abs_diff": reload_diff, "reload_tolerance": 1e-3, "reload_ok": reload_diff <= 1e-3,
        "intervention_sensitivity_mean_abs": sensitivity,
        "status": status, "deviations": cfg.deviations, "environment": env_info(),
        "init_state_hash": tr.init_hash,
    }
    if cfg.method in METHOD_LABELS:
        metrics["label"] = METHOD_LABELS[cfg.method]
        metrics["ecs_weight"], metrics["ecs_threshold"] = cfg.ecs_weight, cfg.ecs_threshold
    if cfg.method == "celljepa":
        metrics["teacher_state_hash_end"] = state_hash(tr.teacher)
        metrics["teacher_diagnostics"] = tr.teacher_diag
    if not cfg.default_ckpt or cfg.input_transform != "log1p":
        metrics["ckpt"], metrics["input_transform"] = cfg.ckpt, cfg.input_transform
    _write(run_dir, "metrics.json", metrics)
    _write_rows(os.path.join(run_dir, "per_intervention.csv"), test_rows)
    _write_rows(os.path.join(run_dir, "per_intervention_percell.csv"), pc_rows)
    np.savez_compressed(os.path.join(run_dir, "val_predictions.npz"),
                        conditions=np.array(list(best_val_preds)), preds=np.stack(list(best_val_preds.values())))
    np.savez_compressed(os.path.join(run_dir, "test_predictions.npz"),
                        conditions=np.array(list(test_preds)), preds=np.stack(list(test_preds.values())))
    _write(run_dir, "status.json", {"status": status, "finished_unix": time.time()})
    log("done", **{k: metrics[k] for k in ["status", "epochs_completed", "selected_epoch"]},
        test_delta_rmse=test_sum["delta_rmse"])
    log_f.close()
    commit()
    return metrics


# ============================================================================ pilot / benchmark


def run_pilot(ds: Dataset, ckpt_dir: str, methods=("scgpt", "jepa"), steps: int = 150, warmup: int = 20,
              seed: int = 1, aux_weight: float = 0.5, out_dir: str | None = None,
              ckpt: str = DEFAULT_CKPT, input_transform: str = "log1p", teacher_update: str = "frozen") -> dict:
    """Execution + timing gate on one GPU. Also emits a numerical fingerprint used to compare
    software stacks / GPUs (prediction from the untrained, pretrained-loaded model).
    `ckpt_dir` is the resolved directory of `ckpt` (the name is only recorded)."""
    report = {"environment": env_info(), "methods": {}, "ckpt": ckpt, "ckpt_dir": ckpt_dir,
              "input_transform": input_transform}
    hashes = {}
    for method in methods:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        cfg = TrainConfig(method=method, seed=seed, aux_weight=aux_weight, ckpt=ckpt,
                          input_transform=input_transform, teacher_update=teacher_update)
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
        if method == "celljepa":
            r["teacher_state_hash_end"] = state_hash(tr.teacher)
            r["teacher_unchanged"] = r["teacher_state_hash_end"] == tr.teacher_init_hash
            r["teacher_update"] = cfg.teacher_update
            # collapse check on the last 10 steps (per-cell targets should not be memorisable)
            r["loss_jepa_last10_mean"] = float(np.mean([l["loss_jepa"] for l in losses[-10:]]))
            r["loss_jepa_min"] = float(min(l["loss_jepa"] for l in losses))
            # reference: JEPA loss of always predicting the batch-mean teacher target. A loss near this
            # value means the student only learned the shared direction, not per-cell information.
            r["jepa_meanpred_baseline_last10_mean"] = float(np.mean([l["jepa_meanpred_baseline"] for l in losses[-10:]]))
            r["teacher_cls_pairwise_cos_last10_mean"] = float(np.mean([l["teacher_cls_pairwise_cos"] for l in losses[-10:]]))
            r["loss_components_finite"] = all(math.isfinite(l[k]) for l in losses
                                              for k in ("loss_mse", "loss_jepa", "loss_ecs"))
        elif method == "scgpt_paper":
            r["loss_components_finite"] = all(math.isfinite(l[k]) for l in losses for k in ("loss_mse", "loss_ecs"))

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
    if "celljepa" in report["methods"]:
        c = report["methods"]["celljepa"]
        no_grad = not c["teacher_has_grads"] and not c["teacher_requires_grad"]
        # never receives gradients; frozen: weights bit-identical after the pilot steps; ema: weights moved
        gates["celljepa_teacher_frozen"] = no_grad and (c["teacher_unchanged"] == (c["teacher_update"] == "frozen"))
        gates["celljepa_loss_components_finite"] = c["loss_components_finite"]
        gates["celljepa_jepa_not_collapsed"] = c["loss_jepa_last10_mean"] >= 1e-3
        # informative, not a gate: near-identical teacher <cls> targets make the JEPA term carry little
        # per-cell signal even when its loss is "not collapsed" (observed with public scGPT_human)
        if c["jepa_meanpred_baseline_last10_mean"] < 1e-3:
            report.setdefault("warnings", []).append(
                f"celljepa: teacher <cls> targets nearly identical across cells (pairwise cos "
                f"{c['teacher_cls_pairwise_cos_last10_mean']:.4f}); a constant batch-mean predictor scores "
                f"{c['jepa_meanpred_baseline_last10_mean']:.2e} vs student {c['loss_jepa_last10_mean']:.2e}")
    if "scgpt_paper" in report["methods"]:   # no teacher: only the loss-component gate applies
        gates["scgpt_paper_loss_components_finite"] = report["methods"]["scgpt_paper"]["loss_components_finite"]
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
