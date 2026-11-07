"""Track A — Cell-JEPA reimplemented from the paper (unofficial): pretraining objectives.

Model, masking, attention mask, EMA teacher, CLS predictor and losses for kidney pretraining
(arXiv 2602.02093). No official code exists; every choice the paper leaves open is listed in
``DEVIATIONS`` and written into the exported ``args.json``.

Inputs (both objectives, identical):
  * tokens = gene embedding + value embedding (scGPT ``GeneEncoder`` + ``ContinuousValueEncoder``
    on the bin index 1..50 as a float); ``<cls>`` is prepended at position 0 with value 0.
  * value masking: a random 15% of each cell's gene tokens get value MASK_VALUE (-1); gene ids
    are never masked, ``<cls>`` is never masked. Padding uses gene ``<pad>`` and value -2.
  * attention: query i may attend key j iff j is an unmasked, non-pad token, or i == j.
    So each masked token sees all unmasked tokens and itself but no other masked token (paper);
    unmasked tokens and ``<cls>`` see only unmasked tokens (scGPT generative style — the paper
    does not say; this choice also keeps the student's ``<cls>`` from reading the -1 tokens).
    Padding is folded into the same mask (pads are never keys, except for their own row's
    diagonal), so no attention row is ever fully blocked -> no NaN softmax. The teacher (no
    masked tokens) uses an equivalent plain src_key_padding_mask (unit-tested equivalence).

Objectives:
  * "jepa":  loss = 1000 * (1 - cos(p(student_cls), sg(teacher_cls))) + 1 * masked-MSE.
  * "scgpt": loss = masked-MSE only; same masking, attention mask, data, init (matched baseline).
The teacher is an EMA of the student (no grad, eval mode) that sees the UNMASKED sequence.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TRACK_LABEL = "Track A — Cell-JEPA reimplemented from the paper (unofficial)"
OBJECTIVES = ("jepa", "scgpt")
MASK_VALUE = -1.0
PAD_VALUE = -2.0
CLS_VALUE = 0.0
JEPA_WEIGHT = 1000.0
RECON_WEIGHT = 1.0

RANDOM_INIT_DEVIATION = "Random initialisation (no scGPT_human weights), identical for both objectives (same seed)."

DEVIATIONS = [
    "No official Cell-JEPA code: reimplemented from the paper text (arXiv 2602.02093).",
    "EMA teacher momentum not given in the paper: constant m (default 0.996), no schedule.",
    "Predictor width not given: 2 x Linear(d_model, d_model) with GELU in between.",
    "Attention rule for unmasked/<cls> queries not given: they attend only unmasked tokens "
    "(scGPT generative style); masked tokens attend unmasked tokens + themselves (paper).",
    "Number of masked tokens per cell = max(1, int(0.15 * n_genes)) (scGPT random_mask_value "
    "convention; the paper only gives the 0.15 ratio).",
    "Gradient clipping at max-norm 1.0 (paper silent).",
    "bf16 autocast, no loss scaling (paper silent on precision).",
    "Value head = scGPT ExprDecoder (3-layer MLP per token); reconstruction target = bin index.",
    "Checkpoint selection: lowest validation masked-reconstruction MSE (paper silent); the "
    "same rule for both objectives.",
    RANDOM_INIT_DEVIATION,  # replaced when PretrainConfig.init_ckpt is set (continued pretraining)
]


# ----------------------------------------------------------------------------- model


def build_student(vocab, d_model: int = 512, nhead: int = 8, d_hid: int = 512, nlayers: int = 12,
                  nlayers_cls: int = 3, dropout: float = 0.2, seed: int = 0) -> nn.Module:
    """Random-init scGPT TransformerModel whose encoder / value_encoder / transformer_encoder
    parameter names and shapes match TransformerGenerator (so export -> fine-tuning loads)."""
    from scgpt.model import TransformerModel

    torch.manual_seed(seed)
    return TransformerModel(
        ntoken=len(vocab), d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=nlayers,
        nlayers_cls=nlayers_cls, n_cls=1, vocab=vocab, dropout=dropout, pad_token="<pad>",
        pad_value=int(PAD_VALUE), do_mvc=False, do_dab=False, use_batch_labels=False,
        input_emb_style="continuous", explicit_zero_prob=False, use_fast_transformer=False,
        pre_norm=False,
    )


class CLSPredictor(nn.Module):
    """ẽ = p(ê): two linear layers with GELU, applied to the student's <cls> output only."""

    def __init__(self, d: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or d
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, x):
        return self.net(x)


def make_teacher(student: nn.Module) -> nn.Module:
    t = copy.deepcopy(student)
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    """θ_T ← m θ_T + (1 − m) θ_S over all parameters (buffers are copied)."""
    tp = [p for p in teacher.parameters()]
    sp = [p.detach() for p in student.parameters()]
    assert len(tp) == len(sp)
    torch._foreach_mul_(tp, momentum)
    torch._foreach_add_(tp, sp, alpha=1.0 - momentum)
    for bt, bs in zip(teacher.buffers(), student.buffers()):
        bt.copy_(bs)


# ----------------------------------------------------------------------------- masks


def random_value_mask(n_genes: np.ndarray, L: int, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean (B, L) mask of value-masked positions. Row b has gene tokens at 1..n_genes[b]
    (position 0 is <cls>); exactly max(1, int(ratio * n)) of them are masked (0 if n == 0)."""
    B = len(n_genes)
    out = np.zeros((B, L), dtype=bool)
    for b, n in enumerate(n_genes):
        n = int(n)
        if n == 0:
            continue
        k = max(1, int(ratio * n))
        out[b, 1 + rng.choice(n, size=k, replace=False)] = True
    return out


def build_attn_mask(masked: torch.Tensor, pad: torch.Tensor, nhead: int) -> torch.Tensor:
    """Bool attention mask (B*nhead, L, L); True = NOT allowed (PyTorch convention).

    blocked[b, i, j] = (masked[b, j] or pad[b, j]) and i != j.
    """
    B, L = masked.shape
    bad_key = (masked | pad).unsqueeze(1)                                  # (B, 1, L)
    eye = torch.eye(L, dtype=torch.bool, device=masked.device).unsqueeze(0)  # (1, L, L)
    blocked = bad_key & ~eye                                               # (B, L, L)
    return blocked.repeat_interleave(nhead, dim=0)


def encode(model: nn.Module, gene_ids: torch.Tensor, values: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """scGPT _encode with an attention mask (0.2.4's _encode only takes a key-padding mask)."""
    x = model.encoder(gene_ids) + model.value_encoder(values)
    return model.transformer_encoder(x, mask=attn_mask)


# ----------------------------------------------------------------------------- losses


def jepa_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 − cos(ẽ, sg(e_teacher)), mean over the batch. No target layer-norm."""
    return (1.0 - F.cosine_similarity(pred.float(), target.detach().float(), dim=-1)).mean()


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(torch.float32)
    return (((pred.float() - target.float()) ** 2) * m).sum() / m.sum().clamp_min(1.0)


# Anti-collapse ablations (deviations from the paper; "faithful" = paper as written).
#   center : DINO-style centring — target = teacher cls − EMA mean of teacher cls (momentum 0.9),
#            so the predictor must explain per-cell deviations, not the shared component.
#   vicreg : faithful target + VICReg variance hinge on the student <cls> (per-dim batch std ≥ 1).
#   token  : I-JEPA-style — predict layer-normed teacher embeddings of the MASKED gene tokens
#            (per-token targets) instead of the <cls> embedding.
JEPA_VARIANTS = ("faithful", "center", "vicreg", "token")
CENTER_MOMENTUM = 0.9
VAR_WEIGHT = 1.0
VARIANT_DESCRIPTIONS = {
    "center": "anti-collapse: JEPA target centred by an EMA (m=0.9) of teacher <cls> embeddings (DINO-style)",
    "vicreg": "anti-collapse: VICReg variance hinge (weight 1.0, gamma 1.0) on student <cls> added to the JEPA term",
    "token": "anti-collapse: token-level JEPA — predictor maps student masked-token outputs to layer-normed teacher "
             "masked-token embeddings (I-JEPA style) instead of <cls>",
}


def variance_hinge(e: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """VICReg variance term: mean over dims of relu(gamma − std across the batch)."""
    if e.shape[0] < 2:
        return torch.zeros((), device=e.device)
    return F.relu(gamma - torch.sqrt(e.float().var(0) + eps)).mean()


@torch.no_grad()
def cls_spread(e: torch.Tensor) -> torch.Tensor:
    if e.shape[0] < 2:
        return torch.zeros((), device=e.device)
    return F.normalize(e.float(), dim=-1).std(0).mean()


def combine(objective: str, recon: torch.Tensor, jepa: torch.Tensor | None) -> torch.Tensor:
    if objective == "scgpt":
        return RECON_WEIGHT * recon
    assert objective == "jepa" and jepa is not None
    return JEPA_WEIGHT * jepa + RECON_WEIGHT * recon


def forward_losses(objective: str, student: nn.Module, teacher: nn.Module | None, predictor: nn.Module | None,
                   batch: dict, nhead: int, amp: dict, variant: str = "faithful", state: dict | None = None,
                   update_state: bool = False, return_hidden: bool = False) -> dict:
    """Single student forward on the masked sequence (+ teacher forward on the unmasked one).

    batch: gene_ids (B,L) long, values (B,L) float (true bins; cls 0, pad -2),
           masked (B,L) bool, pad (B,L) bool.
    Returns tensors: loss, recon, and (jepa objective) jepa; with return_hidden also "h", the
    student's output (B, L, d) on the masked sequence (used by the PBMC fine-tune for GEPC / ECS).
    """
    gids, vals, masked, pad = batch["gene_ids"], batch["values"], batch["masked"], batch["pad"]
    in_vals = vals.masked_fill(masked, MASK_VALUE)
    s_mask = build_attn_mask(masked, pad, nhead)
    with torch.autocast(**amp):
        h = encode(student, gids, in_vals, s_mask)
        pred = student.decoder(h)["pred"]
    recon = masked_mse(pred, vals, masked)
    out = {"recon": recon}
    if return_hidden:
        out["h"] = h
    jl = None
    if objective == "jepa":
        # Teacher sees the unmasked sequence: plain key padding is the same rule there (tested)
        # and avoids a second (B*nhead, L, L) mask. Pad query rows still see real keys -> no NaN.
        with torch.no_grad(), torch.autocast(**amp):
            xt = teacher.encoder(gids) + teacher.value_encoder(vals)
            Ht = teacher.transformer_encoder(xt, src_key_padding_mask=pad)
        ht = Ht[:, 0]
        if variant == "token":
            tgt = F.layer_norm(Ht[masked].float(), (Ht.shape[-1],))
            with torch.autocast(**amp):
                z = predictor(h[masked])
            jl = jepa_cosine_loss(z, tgt)
            out["teacher_tok_std"] = cls_spread(tgt)
        else:
            tgt = ht.float()
            if variant == "center":
                c = (state or {}).get("center")
                if c is None:
                    c = tgt.mean(0).detach()
                if update_state and state is not None:
                    state["center"] = (CENTER_MOMENTUM * c + (1 - CENTER_MOMENTUM) * tgt.mean(0)).detach()
                tgt = tgt - c
                out["target_spread"] = cls_spread(tgt)
            with torch.autocast(**amp):
                z = predictor(h[:, 0])
            jl = jepa_cosine_loss(z, tgt)
            if variant == "vicreg":
                vr = variance_hinge(h[:, 0])
                out["var_loss"] = vr
                jl = jl + VAR_WEIGHT * vr
        out["jepa"] = jl
        # collapse diagnostic (not a loss): per-dim std across the batch of L2-normalised teacher
        # <cls> embeddings; ~0 means every cell maps to the same target.
        out["teacher_cls_std"] = cls_spread(ht)
    out["loss"] = combine(objective, recon, jl)
    return out
