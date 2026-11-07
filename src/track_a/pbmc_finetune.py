"""Track A — Cell-JEPA reimplemented from the paper (unofficial): Task 1, PBMC-10K fine-tuning.

Paper §2.4 + App. E.2: fine-tune a pretrained checkpoint on PBMC-10K with masked inputs (40%) and
  L = w_GEP·GEP + w_GEPC·GEPC + w_ECS·ECS (+ w_JEPA·JEPA for Cell-JEPA; JEPA weight inherited from
  pretraining = 1000, all others 1). Adam lr 1e-4 ×0.9 per epoch, batch 64, 30 epochs, 9:1 cell split
  with seed 42. Then embed every cell (unmasked, eval mode, <cls>) and score cell-type clustering
  (same metrics as Task 2).

Pieces:
  * GEP  = masked-value reconstruction from the pretrained value head (pretrain_model.forward_losses)
  * GEPC = masked-value prediction from the cell embedding (scGPT MVCDecoder "inner product": pred_i =
           W·sigmoid(Q·gene_emb_i) · cell_emb), newly initialised (identical in both arms)
  * ECS  = scGPT's elastic cell similarity on the <cls> embedding (src.models.scgpt_ecs_loss)
  * JEPA = the pretraining JEPA term, same variant as the checkpoint, EMA teacher initialised as a copy
           of the loaded student, predictor newly initialised
Deviations are listed in FINETUNE_DEVIATIONS and written to every result.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .pbmc import TRACK_LABEL, cell_tokens, cluster_metrics, column_vocab_ids, embed_cells, pad_batch

FINETUNE_DEVIATIONS = [
    "ECS: scGPT's unsupervised ECS (cosine-similarity threshold) instead of the paper's supervised-contrastive "
    "form; cell-type labels are never used for training (they are the evaluation target)",
    "GEPC head: scGPT MVCDecoder 'inner product' style, newly initialised (paper gives no architecture)",
    "JEPA teacher for fine-tuning: EMA (m=0.996) initialised from the loaded student; predictor newly "
    "initialised (pretraining exports only the student)",
    "Attention: same masked-token rule as pretraining (masked tokens see unmasked tokens + themselves)",
    "Checkpoint used for evaluation: final epoch (paper does not state a selection rule)",
    "Inputs: nonzero genes only, up to 1200 per cell, per-cell bins 1..50 (scGPT integration tutorial style)",
]


@dataclass
class FinetuneConfig:
    objective: str                   # "jepa" (Cell-JEPA FT) or "scgpt" (scGPT FT)
    jepa_variant: str = "faithful"
    seed: int = 0
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-4
    lr_step_gamma: float = 0.9
    mask_ratio: float = 0.4
    max_genes: int = 1200
    w_gep: float = 1.0
    w_gepc: float = 1.0
    w_ecs: float = 1.0
    w_jepa: float = 1000.0
    ecs_threshold: float = 0.8       # scGPT integration tutorial value
    ema_momentum: float = 0.996
    grad_clip: float = 1.0
    dropout: float = 0.2
    val_frac: float = 0.1
    split_seed: int = 42
    precision: str = "bf16"
    track: str = TRACK_LABEL
    deviations: list = field(default_factory=lambda: list(FINETUNE_DEVIATIONS))

    def __post_init__(self):
        assert self.objective in ("jepa", "scgpt"), self.objective


def gepc_head(d: int):
    import torch.nn as nn

    class GEPCHead(nn.Module):
        """scGPT MVCDecoder, arch 'inner product': pred_i = (W·sigmoid(Q·g_i)) · c."""

        def __init__(self):
            super().__init__()
            self.gene2query = nn.Linear(d, d)
            self.W = nn.Linear(d, d, bias=False)

        def forward(self, cell_emb, gene_embs):
            import torch
            q = torch.sigmoid(self.gene2query(gene_embs))
            return torch.bmm(self.W(q), cell_emb.unsqueeze(2)).squeeze(2)

    return GEPCHead()


def split_cells(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(round(val_frac * n))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def load_student(ckpt_dir: str, dropout: float, builder=None):
    """Full student (all matching keys, incl. the pretrained value head) + vocab + conventions."""
    import torch

    from src.models import SimpleVocab

    from .pretrain_model import build_student

    with open(os.path.join(ckpt_dir, "args.json")) as f:
        args = json.load(f)
    vocab = SimpleVocab(os.path.join(ckpt_dir, "vocab.json"))
    builder = builder or build_student
    model = builder(vocab, d_model=args["embsize"], nhead=args["nheads"], d_hid=args["d_hid"],
                    nlayers=args["nlayers"], nlayers_cls=args.get("n_layers_cls", 3), dropout=dropout)
    sd = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location="cpu")
    sd = {k.replace("Wqkv.", "in_proj_"): v for k, v in sd.items()}
    own = model.state_dict()
    hit = [k for k in own if k in sd and sd[k].shape == own[k].shape]
    enc = [k for k in own if k.startswith(("encoder", "value_encoder", "transformer_encoder"))]
    if set(enc) - set(hit):
        raise RuntimeError(f"{ckpt_dir}: encoder weights not loaded: {sorted(set(enc) - set(hit))[:10]}")
    own.update({k: sd[k] for k in hit})
    model.load_state_dict(own)
    conv = {"pad_value": float(args.get("pad_value", -2)),
            "cls_value": float(args.get("cls_value", args.get("pad_value", -2))),
            "nhead": args["nheads"], "d_model": args["embsize"], "n_loaded": len(hit),
            "decoder_loaded": any(k.startswith("decoder") for k in hit),
            "pretrain_objective": args.get("objective", "scgpt_human (public)")}
    return model, vocab, conv


def run_finetune(cfg: FinetuneConfig, ckpt_dir: str, data: dict, out_dir: str, device: str = "cuda",
                 builder=None, log=print, commit=lambda: None) -> dict:
    import torch

    from src.models import scgpt_ecs_loss

    from .pretrain_model import (CLSPredictor, ema_update, forward_losses, make_teacher, masked_mse,
                                 random_value_mask)

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    student, vocab, conv = load_student(ckpt_dir, cfg.dropout, builder)
    student.to(device)
    d = conv["d_model"]
    g = torch.Generator().manual_seed(cfg.seed + 10_000)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
        gepc = gepc_head(d).to(device)
        predictor = CLSPredictor(d).to(device) if cfg.objective == "jepa" else None
    teacher = make_teacher(student) if cfg.objective == "jepa" else None
    params = list(student.parameters()) + list(gepc.parameters()) + (list(predictor.parameters()) if predictor else [])
    opt = torch.optim.Adam(params, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=cfg.lr_step_gamma)
    amp = dict(device_type="cuda" if str(device).startswith("cuda") else "cpu", dtype=torch.bfloat16,
               enabled=cfg.precision == "bf16" and str(device).startswith("cuda"))

    indptr, indices, vals_all = data["indptr"], data["indices"], data["data"]
    col_ids = column_vocab_ids(list(data["gene_symbols"]), vocab)
    n = len(indptr) - 1
    tok_rng = np.random.default_rng(cfg.seed)
    cells = [cell_tokens(indices[indptr[r]:indptr[r + 1]], vals_all[indptr[r]:indptr[r + 1]], col_ids, cfg.max_genes, tok_rng)
             for r in range(n)]
    tr_idx, va_idx = split_cells(n, cfg.val_frac, cfg.split_seed)
    cls_id, pad_id = vocab["<cls>"], vocab["<pad>"]

    def make_batch(idx, rng):
        b = pad_batch([cells[i] for i in idx], cls_id, pad_id, conv["cls_value"], conv["pad_value"])
        ns = np.array([len(cells[i][0]) for i in idx])
        b["masked"] = random_value_mask(ns, b["gene_ids"].shape[1], cfg.mask_ratio, rng)
        return {k: torch.from_numpy(v).to(device) for k, v in b.items()}

    jstate: dict = {}

    def losses(batch, train: bool):
        out = forward_losses(cfg.objective, student, teacher, predictor, batch, conv["nhead"], amp,
                             cfg.jepa_variant, jstate, update_state=train, return_hidden=True)
        h = out.pop("h")
        cell = h[:, 0].float()
        with torch.autocast(**amp):
            gp = gepc(cell, student.encoder(batch["gene_ids"]).float())
        l_gepc = masked_mse(gp, batch["values"], batch["masked"])
        l_ecs = scgpt_ecs_loss(cell, cfg.ecs_threshold)
        loss = cfg.w_gep * out["recon"] + cfg.w_gepc * l_gepc + cfg.w_ecs * l_ecs
        rec = {"gep": float(out["recon"]), "gepc": float(l_gepc), "ecs": float(l_ecs)}
        if cfg.objective == "jepa":
            loss = loss + cfg.w_jepa * out["jepa"]
            rec["jepa"] = float(out["jepa"])
        rec["loss"] = float(loss)
        return loss, rec

    history = []
    steps = math.ceil(len(tr_idx) / cfg.batch_size)
    log(json.dumps({"msg": "start", "objective": cfg.objective, "variant": cfg.jepa_variant, "n_train": len(tr_idx),
                    "n_val": len(va_idx), "steps_per_epoch": steps, **{k: conv[k] for k in ("n_loaded", "decoder_loaded")}}))
    for ep in range(cfg.epochs):
        rng = np.random.default_rng([cfg.seed, ep])
        perm = rng.permutation(tr_idx)
        student.train()
        gepc.train()
        if predictor is not None:
            predictor.train()
        acc = []
        for s in range(steps):
            loss, rec = losses(make_batch(perm[s * cfg.batch_size:(s + 1) * cfg.batch_size], rng), True)
            if not math.isfinite(rec["loss"]):
                raise FloatingPointError(f"non-finite loss at epoch {ep} step {s}: {rec}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            if teacher is not None:
                ema_update(teacher, student, cfg.ema_momentum)
            acc.append(rec)
        sched.step()
        student.eval()
        gepc.eval()
        vrng = np.random.default_rng([cfg.seed, 10_000])
        with torch.no_grad():
            vrec = [losses(make_batch(va_idx[i:i + cfg.batch_size], vrng), False)[1]
                    for i in range(0, len(va_idx), cfg.batch_size)]
        row = {"epoch": ep, **{f"train_{k}": float(np.mean([r[k] for r in acc])) for k in acc[0]},
               **{f"val_{k}": float(np.mean([r[k] for r in vrec])) for k in vrec[0]}, "t": round(time.time() - t0, 1)}
        history.append(row)
        log(json.dumps(row))

    student.eval()
    emb, info = embed_cells(student, vocab, conv, indptr, indices, vals_all, list(data["gene_symbols"]),
                            max_genes=cfg.max_genes, device=device)
    metrics = cluster_metrics(emb, data["labels"])
    res = {"track": TRACK_LABEL, "task": "Task 1: PBMC-10K fine-tuning", "checkpoint": ckpt_dir,
           "config": asdict(cfg), "model": conv, **info, **metrics, "history": history,
           "minutes": round((time.time() - t0) / 60, 1)}
    np.save(os.path.join(out_dir, "embedding.npy"), emb.astype(np.float32))
    json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=2)
    commit()
    return res
