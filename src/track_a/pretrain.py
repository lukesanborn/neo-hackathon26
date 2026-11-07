"""Track A — Cell-JEPA reimplemented from the paper (unofficial): kidney pretraining loop.

Reads the kidney cache contract (``shards/shard_{k:03d}.npz`` with indptr / gene_ids / bins /
n_cells, ``split.json`` with ``val_cells``), trains one objective ("jepa" or "scgpt") from a
seeded random initialisation, checkpoints every epoch (+ every ``save_every_minutes`` and at a
time-budget stop) and resumes from ``last.pt``. Exports to the checkpoint contract
(``args.json`` / ``best_model.pt`` / ``last.pt`` / ``vocab.json``).

Matching between objectives: same seed -> identical student init (state hash logged), identical
epoch cell order, gene sub-sampling and value masks (all drawn from numpy RNGs keyed by
(seed, epoch, step)), identical optimiser / schedule. Only the loss differs.

Schedule semantics: an "epoch" is one pass over the training cells, or a seeded random subset of
``max_cells_per_epoch`` of them if set. StepLR(gamma) steps once per epoch. Validation (masked
recon MSE on a fixed subset of validation cells with fixed masks) runs after each epoch and at a
budget stop; "best" = lowest validation recon MSE.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from src.models import SimpleVocab, state_hash

from .pretrain_model import (CLS_VALUE, DEVIATIONS, MASK_VALUE, OBJECTIVES, PAD_VALUE, TRACK_LABEL,
                             CLSPredictor, JEPA_VARIANTS, VARIANT_DESCRIPTIONS, build_student, ema_update,
                             forward_losses, make_teacher,
                             random_value_mask)

PAPER_N_CELLS = 814_000
PAPER_EPOCHS = 44
VAL_EPOCH_KEY = 1_000_000  # RNG key for the fixed validation batches


@dataclass
class PretrainConfig:
    objective: str
    seed: int = 0
    # architecture (= scGPT_human args.json)
    d_model: int = 512
    nhead: int = 8
    d_hid: int = 512
    nlayers: int = 12
    nlayers_cls: int = 3
    dropout: float = 0.2
    # data
    max_genes: int = 600            # nonzero genes sampled per cell (+ <cls> = 601 tokens)
    mask_ratio: float = 0.15
    n_bins: int = 50
    max_cells_per_epoch: int | None = None
    max_val_cells: int = 4096
    # optimisation
    epochs: int = PAPER_EPOCHS
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 2e-4
    lr_step_gamma: float = 0.9      # StepLR, step_size = 1 epoch
    grad_clip: float = 1.0
    precision: str = "bf16"
    ema_momentum: float = 0.996     # jepa only
    jepa_variant: str = "faithful"  # jepa only: faithful | center | vicreg | token (see pretrain_model)
    predictor_seed_offset: int = 10_000
    # budget / bookkeeping
    max_minutes: float = 23 * 60.0
    reserve_minutes: float = 10.0
    save_every_minutes: float = 30.0
    log_every: int = 100
    prefetch: int = 8
    loader_threads: int = 4
    track: str = TRACK_LABEL
    checkpoint_selection: str = "min validation masked-reconstruction MSE (fixed val cells and masks)"
    deviations: list = field(default_factory=lambda: list(DEVIATIONS))

    def __post_init__(self):
        assert self.objective in OBJECTIVES, self.objective
        assert self.jepa_variant in JEPA_VARIANTS, self.jepa_variant
        if self.objective == "jepa" and self.jepa_variant != "faithful":
            d = VARIANT_DESCRIPTIONS[self.jepa_variant]
            if d not in self.deviations:
                self.deviations = list(self.deviations) + [d]

    def matched_key(self) -> dict:
        """Fields that must agree between a run and its resume (schedule length may grow)."""
        d = asdict(self)
        for k in ("epochs", "max_minutes", "reserve_minutes", "save_every_minutes", "log_every",
                  "prefetch", "loader_threads", "deviations"):
            d.pop(k)
        return d


# ============================================================================ data


class KidneyCache:
    """In-memory view of the sharded CSR cache (unit 1's contract)."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        paths = sorted(glob.glob(os.path.join(cache_dir, "shards", "shard_*.npz")))
        if not paths:
            raise FileNotFoundError(f"no shards under {cache_dir}/shards")
        self.indptr, self.gene_ids, self.bins, sizes, self.shard_keys = [], [], [], [], []
        for p in paths:
            k = int(re.search(r"shard_(\d+)\.npz$", p).group(1))
            z = np.load(p)
            n = int(z["n_cells"])
            ip = z["indptr"].astype(np.int64)
            assert len(ip) == n + 1 and ip[0] == 0, p
            gi, bi = z["gene_ids"], z["bins"]
            assert len(gi) == len(bi) == ip[-1], p
            self.indptr.append(ip)
            self.gene_ids.append(gi)
            self.bins.append(bi)
            sizes.append(n)
            self.shard_keys.append(k)
        self.offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.n_cells = int(self.offsets[-1])
        pos_of_key = {k: i for i, k in enumerate(self.shard_keys)}
        with open(os.path.join(cache_dir, "split.json")) as f:
            val_pairs = json.load(f)["val_cells"]
        val = np.array([self.offsets[pos_of_key[int(s)]] + int(r) for s, r in val_pairs], dtype=np.int64)
        self.val = np.unique(val)
        assert len(self.val) == len(val_pairs), "duplicate val cells in split.json"
        self.train = np.setdiff1d(np.arange(self.n_cells, dtype=np.int64), self.val, assume_unique=True)
        mpath = os.path.join(cache_dir, "manifest.json")
        self.manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}

    def cell(self, g: int):
        s = int(np.searchsorted(self.offsets, g, side="right") - 1)
        r = g - self.offsets[s]
        a, b = self.indptr[s][r], self.indptr[s][r + 1]
        return self.gene_ids[s][a:b], self.bins[s][a:b]

    def nnz_stats(self) -> dict:
        n = np.concatenate([np.diff(ip) for ip in self.indptr])
        return {"mean": float(n.mean()), "median": float(np.median(n)), "min": int(n.min()), "max": int(n.max()),
                "frac_over_600": float((n > 600).mean())}


def collate(cache: KidneyCache, cells: np.ndarray, max_genes: int, mask_ratio: float, rng: np.random.Generator,
            cls_id: int, pad_id: int) -> dict:
    """<cls> + up to ``max_genes`` uniformly sampled nonzero genes per cell, right-padded to the
    batch max. Returns numpy arrays: gene_ids, values (true bins; cls 0, pad -2), masked, pad."""
    seqs = []
    for g in cells:
        gi, bi = cache.cell(int(g))
        if len(gi) > max_genes:
            idx = rng.choice(len(gi), size=max_genes, replace=False)
            gi, bi = gi[idx], bi[idx]
        seqs.append((gi, bi))
    ns = np.array([len(s[0]) for s in seqs], dtype=np.int64)
    B, L = len(seqs), 1 + int(ns.max(initial=0))
    gene_ids = np.full((B, L), pad_id, dtype=np.int64)
    values = np.full((B, L), PAD_VALUE, dtype=np.float32)
    pad = np.ones((B, L), dtype=bool)
    gene_ids[:, 0], values[:, 0], pad[:, 0] = cls_id, CLS_VALUE, False
    for b, (gi, bi) in enumerate(seqs):
        n = len(gi)
        gene_ids[b, 1:1 + n] = gi
        values[b, 1:1 + n] = bi
        pad[b, 1:1 + n] = False
    masked = random_value_mask(ns, L, mask_ratio, rng)
    return {"gene_ids": gene_ids, "values": values, "masked": masked, "pad": pad, "n_genes": ns}


def batch_rng(seed: int, epoch: int, step: int) -> np.random.Generator:
    return np.random.default_rng([seed, epoch, step, 7])


def epoch_cells(cache: KidneyCache, cfg: PretrainConfig, epoch: int) -> np.ndarray:
    perm = np.random.default_rng([cfg.seed, epoch, 1]).permutation(cache.train)
    return perm[: cfg.max_cells_per_epoch] if cfg.max_cells_per_epoch else perm


def val_cells(cache: KidneyCache, cfg: PretrainConfig) -> np.ndarray:
    return np.random.default_rng([cfg.seed, VAL_EPOCH_KEY]).permutation(cache.val)[: cfg.max_val_cells]


def iter_batches(cache, cfg, cells, epoch, start_step, cls_id, pad_id, threads: int, prefetch: int):
    """Yields (step, numpy batch) in order, built by a small thread pool with lookahead."""
    steps = math.ceil(len(cells) / cfg.batch_size)

    def make(s):
        b = cells[s * cfg.batch_size:(s + 1) * cfg.batch_size]
        return collate(cache, b, cfg.max_genes, cfg.mask_ratio, batch_rng(cfg.seed, epoch, s), cls_id, pad_id)

    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        futs = {}
        nxt = start_step
        for s in range(start_step, steps):
            while nxt < steps and nxt < s + prefetch:
                futs[nxt] = ex.submit(make, nxt)
                nxt += 1
            yield s, futs.pop(s).result()
        for f in futs.values():
            f.cancel()


# ============================================================================ trainer


class Pretrainer:
    def __init__(self, cfg: PretrainConfig, vocab: SimpleVocab, device: str = "cuda"):
        self.cfg, self.vocab, self.device = cfg, vocab, device
        self.model = build_student(vocab, cfg.d_model, cfg.nhead, cfg.d_hid, cfg.nlayers, cfg.nlayers_cls,
                                   cfg.dropout, seed=cfg.seed)
        self.init_hash = state_hash(self.model)  # identical for both objectives at the same seed
        self.model.to(device)
        self.teacher = self.predictor = None
        if cfg.objective == "jepa":
            self.teacher = make_teacher(self.model)
            g = torch.Generator().manual_seed(cfg.seed + cfg.predictor_seed_offset)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
                self.predictor = CLSPredictor(cfg.d_model).to(device)
        self.params = list(self.model.parameters()) + (list(self.predictor.parameters()) if self.predictor else [])
        self.opt = torch.optim.AdamW(self.params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=1, gamma=cfg.lr_step_gamma)
        self.amp = dict(device_type="cuda" if str(device).startswith("cuda") else "cpu",
                        dtype=torch.bfloat16, enabled=cfg.precision == "bf16")
        self.cls_id, self.pad_id = vocab["<cls>"], vocab["<pad>"]
        self.jepa_state: dict = {}   # e.g. the EMA centre for jepa_variant="center"

    def to_device(self, nb: dict) -> dict:
        return {k: torch.from_numpy(nb[k]).to(self.device, non_blocking=True)
                for k in ("gene_ids", "values", "masked", "pad")}

    def train_step(self, nb: dict, epoch: int = 0, step: int = 0) -> dict:
        """One optimiser step. Returns detached loss tensors (no host sync).

        The torch RNG (dropout) is reseeded from (seed, epoch, step) so both objectives draw the
        same dropout stream and a resumed run continues exactly as an uninterrupted one."""
        torch.manual_seed(int(np.random.default_rng([self.cfg.seed, epoch, step, 11]).integers(2**62)))
        self.model.train()
        if self.predictor is not None:
            self.predictor.train()
        out = forward_losses(self.cfg.objective, self.model, self.teacher, self.predictor, self.to_device(nb),
                             self.cfg.nhead, self.amp, self.cfg.jepa_variant, self.jepa_state, update_state=True)
        self.opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(self.params, self.cfg.grad_clip)
        self.opt.step()
        if self.teacher is not None:
            ema_update(self.teacher, self.model, self.cfg.ema_momentum)
        res = {k: v.detach() for k, v in out.items()}
        res["grad_norm"] = gn.detach()
        return res

    @torch.no_grad()
    def validate(self, cache: KidneyCache, cells: np.ndarray) -> dict:
        self.model.eval()
        if self.predictor is not None:
            self.predictor.eval()
        sums, n_tok, n_cell = {}, 0.0, 0
        bs = self.cfg.batch_size
        for i in range(0, len(cells), bs):
            nb = collate(cache, cells[i:i + bs], self.cfg.max_genes, self.cfg.mask_ratio,
                         batch_rng(self.cfg.seed, VAL_EPOCH_KEY, i // bs), self.cls_id, self.pad_id)
            out = forward_losses(self.cfg.objective, self.model, self.teacher, self.predictor, self.to_device(nb),
                                 self.cfg.nhead, self.amp, self.cfg.jepa_variant, self.jepa_state)
            # recon: token-weighted (exact mean over all masked tokens); jepa: cell-weighted
            w_tok, w_cell = float(nb["masked"].sum()), len(nb["gene_ids"])
            sums["recon"] = sums.get("recon", 0.0) + float(out["recon"]) * w_tok
            n_tok += w_tok
            for k in ("jepa", "teacher_cls_std", "target_spread", "teacher_tok_std", "var_loss"):
                if k in out:
                    sums[k] = sums.get(k, 0.0) + float(out[k]) * w_cell
            n_cell += w_cell
        res = {"val_recon": sums["recon"] / max(n_tok, 1.0)}
        if "jepa" in sums:
            res["val_jepa"] = sums["jepa"] / max(n_cell, 1)
            for k in ("teacher_cls_std", "target_spread", "teacher_tok_std", "var_loss"):
                if k in sums:
                    res[f"val_{k}"] = sums[k] / max(n_cell, 1)
            res["val_loss"] = 1000.0 * res["val_jepa"] + res["val_recon"]
        else:
            res["val_loss"] = res["val_recon"]
        return res

    def state(self) -> dict:
        return {"model": self.model.state_dict(), "opt": self.opt.state_dict(), "sched": self.sched.state_dict(),
                "teacher": self.teacher.state_dict() if self.teacher is not None else None,
                "predictor": self.predictor.state_dict() if self.predictor is not None else None,
                "jepa_state": {k: v.detach().cpu() for k, v in self.jepa_state.items()}}

    def load(self, st: dict):
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["opt"])
        self.sched.load_state_dict(st["sched"])
        if self.teacher is not None:
            self.teacher.load_state_dict(st["teacher"])
        if self.predictor is not None:
            self.predictor.load_state_dict(st["predictor"])
        self.jepa_state = {k: v.to(self.device) for k, v in (st.get("jepa_state") or {}).items()}


# ============================================================================ helpers


def env_info() -> dict:
    info = {"stack": os.environ.get("JEPA_STACK", "?"), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "attention": "nn.TransformerEncoder (SDPA) with an explicit (B*nhead, L, L) bool mask"}
    try:
        import scgpt
        info["scgpt"] = getattr(scgpt, "__version__", "?")
    except Exception as e:  # pragma: no cover
        info["scgpt"] = f"import error: {e}"
    return info


def _write(d: str, name: str, obj) -> None:
    tmp = os.path.join(d, name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, os.path.join(d, name))


def _atomic_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ============================================================================ full run


def run_pretraining(cfg: PretrainConfig, cache_dir: str, vocab_path: str, run_dir: str, export_dir: str,
                    commit=lambda: None, device: str = "cuda") -> dict:
    os.makedirs(run_dir, exist_ok=True)
    t_start = time.time()
    log_f = open(os.path.join(run_dir, "log.jsonl"), "a")

    def log(msg, **kv):
        rec = {"t": round(time.time() - t_start, 1), "msg": msg, **kv}
        print(json.dumps(rec, default=str), flush=True)
        log_f.write(json.dumps(rec, default=str) + "\n")
        log_f.flush()

    vocab = SimpleVocab(vocab_path)
    cache = KidneyCache(cache_dir)
    vcells = val_cells(cache, cfg)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tr = Pretrainer(cfg, vocab, device)

    last_path = os.path.join(run_dir, "last.pt")
    best_path = os.path.join(run_dir, "best_student.pt")
    epoch, step0, best, best_epoch, history, global_step = 0, 0, math.inf, None, [], 0
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        prev = PretrainConfig(**ck["config"]).matched_key()
        if prev != cfg.matched_key():
            diff = {k: (prev[k], cfg.matched_key()[k]) for k in prev if prev[k] != cfg.matched_key()[k]}
            raise RuntimeError(f"resume config mismatch in {run_dir}: {diff}")
        tr.load(ck["state"])
        epoch, step0, best, best_epoch = ck["epoch"], ck["step"], ck["best_val_recon"], ck["best_epoch"]
        history, global_step = ck["history"], ck["global_step"]
        assert ck["init_hash"] == tr.init_hash
        log("resumed", epoch=epoch, step=step0, global_step=global_step, best_val_recon=best)
        del ck

    meta = {**asdict(cfg), "init_state_hash": tr.init_hash, "n_cells": cache.n_cells, "n_train": len(cache.train),
            "n_val": len(cache.val), "n_val_used": len(vcells), "nnz": cache.nnz_stats(),
            "cache_dir": cache_dir, "cache_manifest": cache.manifest, "environment": env_info(),
            "steps_per_epoch": math.ceil(len(epoch_cells(cache, cfg, 0)) / cfg.batch_size)}
    _write(run_dir, "config.json", meta)
    _write(run_dir, "status.json", {"status": "running", "epoch": epoch, "started_unix": t_start})
    log("start", objective=cfg.objective, init_state_hash=tr.init_hash, n_train=len(cache.train),
        steps_per_epoch=meta["steps_per_epoch"], start_epoch=epoch, start_step=step0)
    commit()

    def save(ep, st):
        _atomic_save({"state": tr.state(), "config": asdict(cfg), "epoch": ep, "step": st, "best_val_recon": best,
                      "best_epoch": best_epoch, "history": history, "global_step": global_step,
                      "init_hash": tr.init_hash}, last_path)

    def do_val(tag):
        nonlocal best, best_epoch
        v = tr.validate(cache, vcells)
        improved = v["val_recon"] < best
        if improved:
            best, best_epoch = v["val_recon"], tag
            _atomic_save(tr.model.state_dict(), best_path)
        return v, improved

    def flush(s):
        rec = {k: float(v) / n_acc for k, v in acc.items()}
        if not all(math.isfinite(v) for v in rec.values()):
            raise FloatingPointError(f"non-finite loss at epoch {epoch} step {s}: {rec}")
        log("train", epoch=epoch, step=s, global_step=global_step, lr=tr.opt.param_groups[0]["lr"], **rec)

    budget_s, reserve_s = cfg.max_minutes * 60, cfg.reserve_minutes * 60
    status, last_save = "running", time.time()
    while epoch < cfg.epochs:
        cells = epoch_cells(cache, cfg, epoch)
        t_ep = time.time()
        acc, n_acc, stopped_at = None, 0, None
        for s, nb in iter_batches(cache, cfg, cells, epoch, step0, tr.cls_id, tr.pad_id,
                                  cfg.loader_threads, cfg.prefetch):
            if time.time() - t_start > budget_s - reserve_s:
                stopped_at = s
                break
            out = tr.train_step(nb, epoch, s)
            global_step += 1
            acc = out if acc is None else {k: acc[k] + out[k] for k in out}
            n_acc += 1
            if n_acc == cfg.log_every or s == 0:
                flush(s)
                acc, n_acc = None, 0
            if time.time() - last_save > cfg.save_every_minutes * 60:
                save(epoch, s + 1)
                commit()
                last_save = time.time()
        if n_acc:
            flush(stopped_at - 1 if stopped_at is not None else s)
            acc, n_acc = None, 0
        if stopped_at is not None:
            v, improved = do_val(f"{epoch}:{stopped_at}")
            history.append({"epoch": epoch, "partial_step": stopped_at, **v})
            save(epoch, stopped_at)
            log("stopping: time budget", epoch=epoch, step=stopped_at, improved=improved, **v)
            status = "stopped_by_time_budget"
            break
        tr.sched.step()
        v, improved = do_val(epoch)
        history.append({"epoch": epoch, "seconds": round(time.time() - t_ep, 1), **v})
        epoch, step0 = epoch + 1, 0
        save(epoch, 0)
        log("epoch", epoch=epoch - 1, seconds=round(time.time() - t_ep, 1), improved=improved, **v)
        commit()
        last_save = time.time()
    if status == "running":
        status = "completed"

    summary = {"objective": cfg.objective, "status": status, "epochs_completed": epoch,
               "global_step": global_step, "best_val_recon": best, "best_epoch": best_epoch,
               "history": history, "init_state_hash": tr.init_hash,
               "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
               "elapsed_seconds_this_call": time.time() - t_start}
    _write(run_dir, "status.json", {"status": status, "finished_unix": time.time(), "epochs_completed": epoch})
    _write(run_dir, "summary.json", summary)
    summary["export"] = export_checkpoint(run_dir, export_dir, vocab, meta, summary, tr.model.state_dict())
    log("done", **{k: v for k, v in summary.items() if k != "history"})
    log_f.close()
    commit()
    return summary


def export_checkpoint(run_dir: str, export_dir: str, vocab: SimpleVocab, meta: dict, summary: dict,
                      last_state: dict | None = None) -> dict:
    """Write the checkpoint contract consumed by fine-tuning (units 3/5)."""
    os.makedirs(export_dir, exist_ok=True)
    best_src = os.path.join(run_dir, "best_student.pt")
    if not os.path.exists(best_src):
        raise FileNotFoundError(f"{best_src}: no validated checkpoint to export")
    shutil.copyfile(best_src, os.path.join(export_dir, "best_model.pt"))
    if last_state is None:
        last_state = torch.load(os.path.join(run_dir, "last.pt"), map_location="cpu", weights_only=False)["state"]["model"]
    _atomic_save(last_state, os.path.join(export_dir, "last.pt"))
    with open(os.path.join(export_dir, "vocab.json"), "w") as f:
        json.dump(vocab.stoi, f)
    args = {
        "track": TRACK_LABEL,
        "embsize": meta["d_model"], "nheads": meta["nhead"], "d_hid": meta["d_hid"], "nlayers": meta["nlayers"],
        "n_layers_cls": meta["nlayers_cls"], "dropout": meta["dropout"],
        "input_style": "bins50", "input_emb_style": "continuous", "n_bins": meta["n_bins"],
        "mask_value": MASK_VALUE, "pad_value": PAD_VALUE, "cls_value": CLS_VALUE, "pad_token": "<pad>",
        "value_input_contract": ("token value = expression bin index 1..50 as float (per-cell binning of "
                                 "nonzero counts by the kidney cache); <cls> at position 0 with value 0; "
                                 "pad value -2; masked value -1. Fine-tuning must feed values on the same "
                                 "scale for the pretrained value_encoder to be in distribution."),
        "objective": meta["objective"], "epochs_completed": summary["epochs_completed"],
        "epochs_planned": meta["epochs"], "status": summary["status"], "global_step": summary["global_step"],
        "best_epoch": summary["best_epoch"], "best_val_recon": summary["best_val_recon"],
        "checkpoint_selection": meta["checkpoint_selection"], "best_model": "student state_dict at best_epoch",
        "last_model": "student state_dict at the end of the latest call",
        "max_genes": meta["max_genes"], "mask_ratio": meta["mask_ratio"], "batch_size": meta["batch_size"],
        "lr": meta["lr"], "weight_decay": meta["weight_decay"], "lr_step_gamma": meta["lr_step_gamma"],
        "max_cells_per_epoch": meta["max_cells_per_epoch"], "grad_clip": meta["grad_clip"],
        "ema_momentum": meta["ema_momentum"] if meta["objective"] == "jepa" else None,
        "seed": meta["seed"], "init_state_hash": meta["init_state_hash"], "n_train_cells": meta["n_train"],
        "cache_dir": meta["cache_dir"], "cache_manifest_version": meta["cache_manifest"].get("census_version"),
        "deviations": meta["deviations"], "run_dir": run_dir,
    }
    _write(export_dir, "args.json", args)
    return {"export_dir": export_dir, "files": sorted(os.listdir(export_dir))}


# ============================================================================ pilot


def run_pilot(cache_dir: str, vocab_path: str, out_path: str, objectives=OBJECTIVES, batch_sizes=(32, 256),
              steps: int = 200, warmup: int = 20, seed: int = 0, base_lr: float = 1e-4, base_bs: int = 32,
              device: str = "cuda") -> dict:
    """Throughput pilot: per (objective, batch size) run warmup + ``steps`` steps on real batches
    and project wall-clock for the paper schedule (44 epochs) and the bs=256 lr-scaled alternative."""
    vocab = SimpleVocab(vocab_path)
    cache = KidneyCache(cache_dir)
    rep = {"track": TRACK_LABEL, "environment": env_info(), "cache_dir": cache_dir, "n_cells": cache.n_cells,
           "n_train": len(cache.train), "nnz": cache.nnz_stats(), "steps": steps, "warmup": warmup, "runs": {}}
    hashes = {}
    for obj in objectives:
        for bs in batch_sizes:
            name = f"{obj}_bs{bs}"
            cfg = PretrainConfig(objective=obj, seed=seed, batch_size=bs, lr=base_lr * bs / base_bs)
            r = {"batch_size": bs, "lr_linear_scaled": cfg.lr, "lr_sqrt_scaled": base_lr * math.sqrt(bs / base_bs)}
            tr = None
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                tr = Pretrainer(cfg, vocab, device)
                hashes[name] = tr.init_hash
                cells = epoch_cells(cache, cfg, 0)
                need = warmup + steps
                if len(cells) < need * bs:  # small (smoke) caches: cycle through more epochs
                    reps = math.ceil(need * bs / len(cells))
                    cells = np.concatenate([epoch_cells(cache, cfg, e) for e in range(reps)])
                t_data = time.time()
                for i in range(5):
                    collate(cache, cells[i * bs:(i + 1) * bs], cfg.max_genes, cfg.mask_ratio,
                            batch_rng(seed, 0, i), tr.cls_id, tr.pad_id)
                r["collate_seconds_per_batch_1thread"] = (time.time() - t_data) / 5
                losses, Ls = [], []
                t0 = None
                for s, nb in iter_batches(cache, cfg, cells[: need * bs], 0, 0, tr.cls_id, tr.pad_id,
                                          cfg.loader_threads, cfg.prefetch):
                    if s == warmup:
                        torch.cuda.synchronize()
                        t0 = time.time()
                    out = tr.train_step(nb, 0, s)
                    Ls.append(nb["gene_ids"].shape[1])
                    if s < 3 or s >= need - 3:
                        losses.append({k: float(v) for k, v in out.items()})
                torch.cuda.synchronize()
                sps = (time.time() - t0) / steps
                spe = math.ceil(len(cache.train) / bs)
                spe_paper = math.ceil(PAPER_N_CELLS / bs)
                r.update({
                    "seconds_per_step": sps, "cells_per_second": bs / sps, "mean_seq_len": float(np.mean(Ls)),
                    "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "first_losses": losses[:3], "last_losses": losses[-3:],
                    "all_logged_losses_finite": all(math.isfinite(v) for l in losses for v in l.values()),
                    "steps_per_epoch_this_cache": spe,
                    "projected_hours_per_epoch_this_cache": spe * sps / 3600,
                    "projected_hours_44_epochs_this_cache": PAPER_EPOCHS * spe * sps / 3600,
                    "projected_hours_44_epochs_814k_cells": PAPER_EPOCHS * spe_paper * sps / 3600,
                    "status": "ok",
                })
            except torch.cuda.OutOfMemoryError as e:
                r.update({"status": "oom", "error": str(e)[:500]})
            except Exception as e:  # report, don't hide
                r.update({"status": "failed", "error": repr(e)[:1000]})
            finally:
                tr = None
            rep["runs"][name] = r
            print(json.dumps({name: r}, default=str), flush=True)
            torch.cuda.empty_cache()
    by_bs = {}
    for bs in batch_sizes:
        rs = [rep["runs"].get(f"{o}_bs{bs}", {}) for o in objectives]
        if all(x.get("status") == "ok" for x in rs):
            by_bs[f"bs{bs}"] = {
                "both_objectives_hours_44_epochs_this_cache": sum(x["projected_hours_44_epochs_this_cache"] for x in rs),
                "both_objectives_hours_44_epochs_814k_cells": sum(x["projected_hours_44_epochs_814k_cells"] for x in rs),
            }
    rep["projection_summary"] = by_bs
    rep["init_hash_matched_across_objectives"] = len({hashes[k] for k in hashes}) <= 1
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _write(os.path.dirname(out_path), os.path.basename(out_path), rep)
    return rep


# ============================================================================ synthetic cache (smoke)


def make_synthetic_cache(out_dir: str, vocab_path: str, n_cells: int = 4000, n_shards: int = 4,
                         val_frac: float = 0.1, n_universe: int = 5000, seed: int = 0) -> dict:
    """Synthetic kidney cache in exactly the unit-1 contract format. Bins depend on the gene
    (base bin + small noise) so masked reconstruction is learnable; nnz spans 1..1500 so both
    the <=600 sub-sampling and tiny cells are exercised."""
    rng = np.random.default_rng(seed)
    vocab = SimpleVocab(vocab_path)
    specials = {vocab["<pad>"], vocab["<cls>"], vocab["<eoc>"]}
    pool = np.array(sorted(set(range(len(vocab))) - specials), dtype=np.int64)
    universe = np.sort(rng.choice(pool, size=n_universe, replace=False))
    base = rng.integers(1, 51, size=n_universe)
    os.makedirs(os.path.join(out_dir, "shards"), exist_ok=True)
    per = np.array_split(np.arange(n_cells), n_shards)
    val = []
    for k, idx in enumerate(per):
        indptr, gids, bins = [0], [], []
        for _ in idx:
            n = int(rng.choice([1, 2, 5, 40]) if rng.random() < 0.02 else rng.integers(20, 1500))
            n = min(n, n_universe)
            g = np.sort(rng.choice(n_universe, size=n, replace=False))
            b = np.clip(base[g] + rng.integers(-2, 3, size=n), 1, 50)
            gids.append(universe[g].astype(np.int32))
            bins.append(b.astype(np.uint8))
            indptr.append(indptr[-1] + n)
        np.savez(os.path.join(out_dir, "shards", f"shard_{k:03d}.npz"), indptr=np.array(indptr, dtype=np.int64),
                 gene_ids=np.concatenate(gids), bins=np.concatenate(bins), n_cells=np.int64(len(idx)))
        for r in range(len(idx)):
            if rng.random() < val_frac:
                val.append([k, r])
    _write(out_dir, "split.json", {"val_cells": val})
    man = {"synthetic": True, "source": "unit-2 smoke: random genes/bins in the kidney cache format",
           "census_version": "synthetic", "n_cells": n_cells, "n_shards": n_shards, "n_bins": 50,
           "n_val": len(val), "vocab_path": vocab_path, "seed": seed}
    _write(out_dir, "manifest.json", man)
    return man
