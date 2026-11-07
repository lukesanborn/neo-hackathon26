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


def gene_vocab_ids(gene_names: list[str], vocab: SimpleVocab) -> tuple[np.ndarray, dict]:
    """Tutorial rule (genes absent from the vocab -> <pad>), after applying HGNC_ALIASES."""
    ids, applied, missing = [], {}, []
    for g in gene_names:
        if g not in vocab and g in HGNC_ALIASES and HGNC_ALIASES[g] in vocab:
            applied[g] = HGNC_ALIASES[g]
            g = HGNC_ALIASES[g]
        if g in vocab:
            ids.append(vocab[g])
        else:
            ids.append(vocab[PAD_TOKEN])
            missing.append(g)
    info = {"aliases_applied": applied, "n_genes_mapped_to_pad": len(missing),
            "n_genes_in_vocab": len(gene_names) - len(missing)}
    return np.array(ids, dtype=np.int64), info


def build_scgpt(ckpt_dir: str, seed: int, dropout: float = 0.0):
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


def encode_and_decode(model, src, values, flags, pad_mask):
    """Same computation as TransformerGenerator.forward (no binning, no zero-prob head), but
    also returns the transformer output so the auxiliary objective can use it."""
    h = model._encode(src, values, flags, pad_mask)          # (B, L, d)
    pred = model.decoder(h, values)["pred"]                   # (B, L)
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
