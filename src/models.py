"""scGPT adapter, frozen teacher and latent predictor for our JEPA adaptation (Track B).

The student is scGPT's TransformerGenerator (perturbation model from the official tutorial),
initialised from the public whole-human checkpoint with the tutorial's prefix rule
(encoder / value_encoder / transformer_encoder). Newly initialised parts (pert_encoder,
expression decoder, cls head) are seeded identically in both arms; the JEPA predictor is
created afterwards from a separate RNG so the shared student initialisation is bit-identical.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
PERT_PAD_ID = 2  # scGPT default; flags are 0 (not perturbed) / 1 (perturbed)
LOAD_PREFIXES = ("encoder", "value_encoder", "transformer_encoder")


class SimpleVocab:
    """Token -> id mapping read from scGPT's vocab.json (no torchtext needed)."""

    def __init__(self, path: str):
        with open(path) as f:
            self.stoi: dict[str, int] = json.load(f)
        for tok in SPECIAL_TOKENS:
            if tok not in self.stoi:
                self.stoi[tok] = len(self.stoi)
        ids = sorted(self.stoi.values())
        assert ids == list(range(len(ids))), "vocab ids must be consecutive"

    def __getitem__(self, tok: str) -> int:
        return self.stoi[tok]

    def __contains__(self, tok: str) -> bool:
        return tok in self.stoi

    def __len__(self) -> int:
        return len(self.stoi)


def load_ckpt_args(ckpt_dir: str) -> dict:
    with open(os.path.join(ckpt_dir, "args.json")) as f:
        return json.load(f)


# Outdated symbols in the GEARS Adamson gene list -> current HGNC symbols used by scGPT's vocab.
# Without this, these 11 knockdown targets all map to <pad> and become indistinguishable
# (scGPT has no positional encoding). Applied identically in both arms.
HGNC_ALIASES = {
    "AARS": "AARS1", "CARS": "CARS1", "DARS": "DARS1", "HARS": "HARS1", "MARS": "MARS1",
    "QARS": "QARS1", "SARS": "SARS1", "TARS": "TARS1", "ATP5B": "ATP5F1B",
    "SLMO2": "PRELID3B", "SRPR": "SRPRA",
}


# Dataset-scoped renames (knockdown targets whose GEARS symbol is outdated; current HGNC symbol is
# in the scGPT vocab). Kept out of HGNC_ALIASES because several of these genes are also in the
# Adamson gene list, where they map to <pad>; adding them globally would change Adamson inputs.
# build_cache records them in the cache manifest ("extra_hgnc_aliases"); the Trainer applies them.
DATASET_HGNC_ALIASES = {
    "adamson": {},
    "replogle_k562_essential": {
        "CCDC84": "CENATAC", "CD3EAP": "POLR1G", "EPRS": "EPRS1", "GARS": "GARS1", "IARS": "IARS1",
        "NARS": "NARS1", "TWISTNB": "POLR1F",
    },
}


def gene_vocab_ids(gene_names: list[str], vocab: SimpleVocab,
                   extra_aliases: dict | None = None) -> tuple[np.ndarray, dict]:
    """Tutorial rule (genes absent from the vocab -> <pad>), after applying HGNC_ALIASES
    (+ dataset-scoped ``extra_aliases``)."""
    aliases = {**HGNC_ALIASES, **(extra_aliases or {})}
    ids, applied, missing = [], {}, []
    for g in gene_names:
        if g not in vocab and g in aliases and aliases[g] in vocab:
            applied[g] = aliases[g]
            g = aliases[g]
        if g in vocab:
            ids.append(vocab[g])
        else:
            ids.append(vocab[PAD_TOKEN])
            missing.append(g)
    info = {"aliases_applied": applied, "n_genes_mapped_to_pad": len(missing),
            "n_genes_in_vocab": len(gene_names) - len(missing)}
    return np.array(ids, dtype=np.int64), info


def build_scgpt(ckpt_dir: str, seed: int, dropout: float = 0.0, load: bool = True):
    from scgpt.model import TransformerGenerator

    args = load_ckpt_args(ckpt_dir)
    vocab = SimpleVocab(os.path.join(ckpt_dir, "vocab.json"))
    torch.manual_seed(seed)
    model = TransformerGenerator(
        ntoken=len(vocab),
        d_model=args["embsize"],
        nhead=args["nheads"],
        d_hid=args["d_hid"],
        nlayers=args["nlayers"],
        nlayers_cls=args.get("n_layers_cls", 3),
        n_cls=1,
        vocab=vocab,
        dropout=dropout,
        pad_token=PAD_TOKEN,
        pad_value=0,
        pert_pad_id=PERT_PAD_ID,
        use_fast_transformer=False,
    )
    if not load:   # A28 ablation: same architecture/vocab, seeded random weights (no pretraining)
        return model, vocab, args, {"n_loaded": 0, "random_init": True}
    load_info = load_pretrained(model, os.path.join(ckpt_dir, "best_model.pt"))
    return model, vocab, args, load_info


def load_pretrained(model: nn.Module, path: str) -> dict:
    sd = torch.load(path, map_location="cpu")
    # flash-attn checkpoints store fused qkv as 'Wqkv'; nn.MultiheadAttention uses 'in_proj_'.
    sd = {k.replace("Wqkv.", "in_proj_"): v for k, v in sd.items()}
    own = model.state_dict()
    loaded, shape_mismatch, skipped = [], [], []
    for k, v in sd.items():
        if not k.startswith(LOAD_PREFIXES):
            skipped.append(k)
            continue
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded.append(k)
        else:
            shape_mismatch.append(k)
    model.load_state_dict(own)
    expected = [k for k in model.state_dict() if k.startswith(LOAD_PREFIXES)]
    not_loaded = sorted(set(expected) - set(loaded))
    info = {
        "n_loaded": len(loaded),
        "n_expected_with_prefix": len(expected),
        "not_loaded_with_prefix": not_loaded,
        "shape_or_name_mismatch": shape_mismatch,
        "skipped_ckpt_keys_outside_prefixes": skipped,
        "newly_initialised": sorted({k.split(".")[0] for k in model.state_dict()
                                     if not k.startswith(LOAD_PREFIXES)}),
    }
    if not_loaded or shape_mismatch:
        raise RuntimeError(f"pretrained load incomplete: {json.dumps(info)[:2000]}")
    return info


def state_hash(module: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().float().numpy().tobytes())
    return h.hexdigest()[:16]


def encode_and_decode(model, src, values, flags, pad_mask, dec_values=None):
    """Same computation as TransformerGenerator.forward (no binning, no zero-prob head), but
    also returns the transformer output so the auxiliary objective can use it.

    `dec_values` (default: `values`) feeds the AffineExprDecoder's `coeff * values + bias` term.
    With input_transform="bins50" the encoder sees binned values while the decoder's affine
    term uses the log1p control values, so predictions stay in log1p (evaluation) space."""
    h = model._encode(src, values, flags, pad_mask)          # (B, L, d)
    pred = model.decoder(h, values if dec_values is None else dec_values)["pred"]   # (B, L)
    return h, pred


def pool(h: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Declared pooling convention: mean over non-padded gene tokens, no normalisation."""
    if pad_mask is None:
        return h.mean(1)
    keep = (~pad_mask).unsqueeze(-1).to(h.dtype)
    return (h * keep).sum(1) / keep.sum(1).clamp_min(1.0)


class LatentPredictor(nn.Module):
    """Maps the student's intervention-conditioned (pooled) representation to the predicted
    teacher-space representation shift of the post-intervention population."""

    def __init__(self, d: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or 2 * d
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, z):
        return self.net(z)


class Projector(nn.Module):
    """Cell-JEPA fine-tuning projector on the student <cls> embedding (architecture not given in
    the paper; declared choice: Linear-GELU-Linear, width d)."""

    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, z):
        return self.net(z)


def make_teacher(model: nn.Module) -> nn.Module:
    """Frozen copy of the initial (pretrained-loaded) student. Dropout disabled via eval()."""
    teacher = copy.deepcopy(model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def cosine_aux_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cos(pred, sg(target)), averaged over rows."""
    return (1.0 - F.cosine_similarity(pred.float(), target.detach().float(), dim=-1)).mean()


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    """teacher <- m * teacher + (1 - m) * student (parameters only; teacher stays requires_grad=False)."""
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.mul_(momentum).add_(ps.detach(), alpha=1.0 - momentum)


def scgpt_ecs_loss(cell_emb: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """scGPT's unsupervised elastic cell similarity, copied from TransformerGenerator.forward
    (ECS=True): normalise, cosine matrix, zero the diagonal, relu, mean(1 - (cos - thr)^2) over
    the full B x B matrix (diagonal entries included, as upstream)."""
    z = F.normalize(cell_emb.float(), p=2, dim=1)
    cos = z @ z.t()
    eye = torch.eye(cos.size(0), dtype=torch.bool, device=cos.device)
    cos = F.relu(cos.masked_fill(eye, 0.0))
    return torch.mean(1 - (cos - threshold) ** 2)


def prepend_cls(src, values, flags, pad_mask, cls_id: int, dec_values=None):
    """Prepend a <cls> token at position 0: gene id `cls_id`, value 0, perturbation flag 0,
    padding mask False. Returns (src, values, flags, pad_mask, dec_values-or-None)."""
    B = src.shape[0]

    def col(like, v):
        return torch.full((B, 1), v, dtype=like.dtype, device=like.device)

    out = (torch.cat([col(src, cls_id), src], 1), torch.cat([col(values, 0), values], 1),
           torch.cat([col(flags, 0), flags], 1), torch.cat([col(pad_mask, False), pad_mask], 1))
    dv = None if dec_values is None else torch.cat([col(dec_values, 0), dec_values], 1)
    return (*out, dv)


@torch.no_grad()
def bin_expression(x: torch.Tensor, n_bins: int = 50, chunk: int = 256) -> torch.Tensor:
    """Per-cell quantile binning ("bins50"). For each row: nonzero values -> integer bins 1..n_bins
    using n_bins-1 inner edges at the per-cell quantiles k/n_bins (k=1..n_bins-1, linear
    interpolation as np.quantile) and np.digitize semantics (bin = 1 + #edges <= value); zeros stay 0.
    Vectorised: rows are sorted, so a row's nonzeros are its last n_nz sorted entries.
    Requires x >= 0 (log1p expression). Returns float32 (fed to scGPT's continuous value encoder)."""
    assert x.dim() == 2
    if x.numel() and float(x.min()) < 0:
        raise ValueError("bin_expression expects non-negative expression")
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    G = x.shape[1]
    # same float64 grid as np.linspace(0, 1, n_bins + 1) (arange * step), so edges match numpy bit-exactly
    qs = (torch.arange(n_bins + 1, dtype=torch.float64, device=x.device) * (1.0 / n_bins))[1:-1]
    for i in range(0, x.shape[0], chunk):
        xc = x[i:i + chunk].double()
        nz = xc != 0
        n = nz.sum(1, keepdim=True)                                           # (b, 1)
        srt = torch.sort(xc, dim=1).values
        pos = qs.unsqueeze(0) * (n - 1).clamp_min(0)                          # (b, Q) position within nonzeros
        lo = pos.floor().long()
        hi = pos.ceil().long()
        frac = pos - lo
        off = (G - n)                                                          # first nonzero in sorted row
        vlo = torch.gather(srt, 1, (off + lo).clamp(max=G - 1))
        vhi = torch.gather(srt, 1, (off + hi).clamp(max=G - 1))
        diff = vhi - vlo                                                       # np.quantile's _lerp,
        edges = torch.where(frac >= 0.5, vhi - diff * (1 - frac), vlo + diff * frac)   # bit-exact (b, Q)
        b = 1 + (xc.unsqueeze(-1) >= edges.unsqueeze(1)).sum(-1)               # (b, G)
        out[i:i + chunk] = torch.where(nz, b, torch.zeros_like(b)).float()
    return out
