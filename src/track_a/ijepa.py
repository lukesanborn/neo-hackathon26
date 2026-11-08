"""I-JEPA-style gene-token objectives (experiments X1 / X2; deviations from the Cell-JEPA paper).

Both variants fix the *identity leak* of the existing ``token`` variant, where masked genes stay
in the student's sequence (gene id + value -1) and the student's own output at that position is
mapped to the teacher's token embedding:

  * the student (context encoder) sees ONLY the unmasked genes. Masked genes are removed from
    its input entirely (no key and no value), so nothing about them reaches the encoder;
  * a separate transformer predictor gets [student outputs of the context tokens ; one query per
    masked gene]. A query = the student's gene-ID embedding of that gene + a learned mask token.
    Queries attend to the context and to each other (they carry no values);
  * targets = layer-normed EMA-teacher outputs at the masked genes (teacher sees the full cell);
  * loss = 1000 * mean(1 - cos(pred, target)) over masked genes + 1 * masked-value MSE, where the
    value is decoded from the predictor's query output by the student's ExprDecoder.

The gradient of the predictive loss therefore flows into the gene-ID embedding table through
the queries: the gene embeddings are trained to say "what this gene's state is, given the rest
of the cell". This is the representation the perturbation model reads when it marks a knocked-
down gene.

Variants (``jepa_variant``):
  ijepa    : X1, random whole-gene masking (``mask_ratio`` of each cell's sampled genes).
  progmask : X2, whole gene-program masking. MSigDB Hallmark programs are drawn in random
             order; all of a program's expressed genes in the cell are masked together until the
             cell's masked fraction reaches ``mask_ratio``. If programs run out, the rest is filled
             with random genes. Same model and loss as ijepa, so X1 vs X2 isolates the masking.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IJEPA_VARIANTS = ("ijepa", "progmask")
MIN_PROGRAM_GENES = 2          # a program must have >= this many expressed genes in the cell to be masked
MAX_OVERSHOOT = 2.0            # skip a program that would push the masked count past MAX_OVERSHOOT * target

IJEPA_DESCRIPTIONS = {
    "ijepa": "X1 I-JEPA gene tokens: masked genes removed from the student (no key/value); transformer "
             "predictor with gene-ID-embedding queries predicts layer-normed teacher token embeddings, centred "
             "per gene by an EMA (m=0.9) so gene identity alone cannot predict the target; "
             "recon decoded from predictor queries; random whole-gene masking",
    "progmask": "X2 gene-program masking: X1 model/loss, but whole MSigDB Hallmark (v2023.2) programs are "
                "masked together (random fill if programs run out)",
}


# ----------------------------------------------------------------------------- gene programs


def hallmark_sets() -> dict:
    from .hallmark_sets import HALLMARK
    return HALLMARK


class GenePrograms:
    """Programs as vocab-id sets plus a vocab-id -> [program index] lookup."""

    def __init__(self, vocab, sets: dict | None = None):
        sets = sets if sets is not None else hallmark_sets()
        self.names = sorted(sets)
        self.members = []
        self.of_gene: dict[int, list[int]] = {}
        n_hit = n_all = 0
        for p, name in enumerate(self.names):
            ids = sorted({vocab[g] for g in sets[name] if g in vocab.stoi})
            n_hit += len(ids)
            n_all += len(sets[name])
            self.members.append(np.array(ids, dtype=np.int64))
            for i in ids:
                self.of_gene.setdefault(int(i), []).append(p)
        self.coverage = n_hit / max(n_all, 1)

    def info(self) -> dict:
        return {"n_programs": len(self.names), "n_genes_in_any": len(self.of_gene),
                "symbol_coverage_in_vocab": round(self.coverage, 4)}


def program_mask_row(gi: np.ndarray, k: int, programs: GenePrograms, rng: np.random.Generator):
    """Indices (into gi) of masked genes for one cell, and how many came from whole programs."""
    by_prog: dict[int, list[int]] = {}
    for j, g in enumerate(gi):
        for p in programs.of_gene.get(int(g), ()):
            by_prog.setdefault(p, []).append(j)
    chosen: set[int] = set()
    for p in rng.permutation(sorted(by_prog)):
        if len(chosen) >= k:
            break
        new = [j for j in by_prog[int(p)] if j not in chosen]
        if len(new) < MIN_PROGRAM_GENES or len(chosen) + len(new) > MAX_OVERSHOOT * k:
            continue
        chosen.update(new)
    from_prog = len(chosen)
    if len(chosen) < k:
        rest = np.setdiff1d(np.arange(len(gi)), np.fromiter(chosen, dtype=np.int64, count=len(chosen)))
        chosen.update(rng.choice(rest, size=k - len(chosen), replace=False).tolist())
    return np.fromiter(chosen, dtype=np.int64, count=len(chosen)), from_prog


def make_masker(variant: str, programs: GenePrograms | None):
    """Returns masker(seqs, L, ratio, rng) -> (masked (B,L) bool, stats dict). Position 0 is <cls>."""

    def masker(seqs, L, ratio, rng):
        B = len(seqs)
        out = np.zeros((B, L), dtype=bool)
        n_prog = n_tot = 0
        for b, (gi, _) in enumerate(seqs):
            n = len(gi)
            if n < 2:   # keep at least one context gene
                continue
            k = min(max(1, int(ratio * n)), n - 1)
            if variant == "progmask":
                idx, fp = program_mask_row(gi, k, programs, rng)
                n_prog += fp
            else:
                idx = rng.choice(n, size=k, replace=False)
            out[b, 1 + idx] = True
            n_tot += len(idx)
        return out, {"frac_from_programs": n_prog / max(n_tot, 1)}

    return masker


# ----------------------------------------------------------------------------- model


class GenePredictor(nn.Module):
    """Narrow I-JEPA predictor: context tokens + gene-ID queries -> per-query embeddings."""

    def __init__(self, d_model: int, d_pred: int = 384, nlayers: int = 4, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.ctx_in = nn.Linear(d_model, d_pred)
        self.q_in = nn.Linear(d_model, d_pred)
        self.mask_token = nn.Parameter(torch.zeros(d_pred))
        nn.init.normal_(self.mask_token, std=0.02)
        layer = nn.TransformerEncoderLayer(d_pred, nhead, 4 * d_pred, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, nlayers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_pred)
        self.out = nn.Linear(d_pred, d_model)   # -> teacher space (JEPA) and student decoder space (recon)

    def forward(self, h_ctx, ctx_pad, q_gene_emb, q_pad):
        Lc = h_ctx.shape[1]
        x = torch.cat([self.ctx_in(h_ctx), self.q_in(q_gene_emb) + self.mask_token], dim=1)
        pad = torch.cat([ctx_pad, q_pad], dim=1)
        y = self.blocks(x, src_key_padding_mask=pad)
        return self.out(self.norm(y[:, Lc:]))


def _gather_rows(order: torch.Tensor, n: torch.Tensor):
    """First max(n) columns of ``order`` and their padding mask."""
    Lm = max(int(n.max()), 1)
    idx = order[:, :Lm]
    pad = torch.arange(Lm, device=n.device).unsqueeze(0) >= n.unsqueeze(1)
    return idx, pad


def split_context_targets(masked: torch.Tensor, pad: torch.Tensor):
    """Positions of context tokens (unmasked, non-pad, incl. <cls>) and of masked tokens, left-packed."""
    ctx_key = (masked | pad).to(torch.int8)                       # 0 = context
    ctx_order = torch.argsort(ctx_key, dim=1, stable=True)
    ctx_idx, ctx_pad = _gather_rows(ctx_order, (ctx_key == 0).sum(1))
    tgt_key = (~masked).to(torch.int8)                            # 0 = masked
    tgt_order = torch.argsort(tgt_key, dim=1, stable=True)
    tgt_idx, tgt_pad = _gather_rows(tgt_order, masked.sum(1))
    return ctx_idx, ctx_pad, tgt_idx, tgt_pad


GENE_CENTER_MOMENTUM = 0.9


@torch.no_grad()
def update_gene_center(state: dict, gene_ids: torch.Tensor, tgt: torch.Tensor, n_vocab: int,
                       momentum: float = GENE_CENTER_MOMENTUM) -> None:
    """Per-gene EMA of layer-normed teacher token embeddings (one EMA step per batch a gene occurs in;
    initialised at its first batch mean)."""
    D = tgt.shape[-1]
    if "gene_center" not in state:
        state["gene_center"] = torch.zeros(n_vocab, D, device=tgt.device)
        state["gene_seen"] = torch.zeros(n_vocab, device=tgt.device)
    s = torch.zeros(n_vocab, D, device=tgt.device).index_add_(0, gene_ids, tgt)
    n = torch.zeros(n_vocab, device=tgt.device).index_add_(0, gene_ids, torch.ones_like(gene_ids, dtype=tgt.dtype))
    hit = n > 0
    mean = s[hit] / n[hit, None]
    seen = state["gene_seen"][hit] > 0
    c = state["gene_center"][hit]
    state["gene_center"][hit] = torch.where(seen[:, None], momentum * c + (1 - momentum) * mean, mean)
    state["gene_seen"][hit] = 1.0


def forward_losses_ijepa(student, teacher, predictor, batch: dict, amp: dict, jepa_weight: float,
                         recon_weight: float = 1.0, state: dict | None = None, update_state: bool = False,
                         n_vocab: int | None = None) -> dict:
    """Targets are per-gene CENTRED: LN(teacher token) minus a per-gene EMA of it, so the gene-identity
    component of the target (predictable from the gene-ID query alone) is removed and the predictor must
    predict this cell's deviation for that gene. Without a ``state`` (unit tests) no centring is done."""
    gids, vals, masked, pad = batch["gene_ids"], batch["values"], batch["masked"], batch["pad"]
    ctx_idx, ctx_pad, tgt_idx, tgt_pad = split_context_targets(masked, pad)
    take = lambda x, i: torch.gather(x, 1, i)  # noqa: E731
    c_g, c_v = take(gids, ctx_idx), take(vals, ctx_idx)
    t_g, t_v = take(gids, tgt_idx), take(vals, tgt_idx)
    valid = ~tgt_pad & take(masked, tgt_idx)
    with torch.autocast(**amp):
        h = student.transformer_encoder(student.encoder(c_g) + student.value_encoder(c_v),
                                        src_key_padding_mask=ctx_pad)
        q = student.encoder(t_g)                                   # gene-ID embedding only, no value
        z = predictor(h, ctx_pad, q, tgt_pad)
        rec = student.decoder(z)["pred"]
    m = valid.float()
    recon = (((rec.float() - t_v.float()) ** 2) * m).sum() / m.sum().clamp_min(1.0)
    with torch.no_grad(), torch.autocast(**amp):
        Ht = teacher.transformer_encoder(teacher.encoder(gids) + teacher.value_encoder(vals),
                                         src_key_padding_mask=pad)
    D = Ht.shape[-1]
    tgt = F.layer_norm(torch.gather(Ht.float(), 1, tgt_idx.unsqueeze(-1).expand(-1, -1, D)), (D,))[valid]
    g_valid = t_g[valid]
    out = {}
    if state is not None:
        if update_state or "gene_center" not in state:
            update_gene_center(state, g_valid, tgt, n_vocab or int(gids.max()) + 1)
        with torch.no_grad():
            raw = F.normalize(tgt, dim=-1)
            out["jepa_const_ref_raw"] = 1.0 - F.cosine_similarity(raw, raw.mean(0, keepdim=True).expand_as(raw),
                                                                  dim=-1).mean()
        tgt = tgt - state["gene_center"][g_valid]
    cos = F.cosine_similarity(z.float()[valid], tgt.detach(), dim=-1)
    jl = (1.0 - cos).mean()
    with torch.no_grad():
        tv = F.normalize(tgt, dim=-1)
        # constant-guess reference: cosine loss of always predicting the batch-mean target
        ref = 1.0 - F.cosine_similarity(tv, tv.mean(0, keepdim=True).expand_as(tv), dim=-1).mean()
        spread = tv.std(0).mean()
    out.update({"loss": jepa_weight * jl + recon_weight * recon, "recon": recon, "jepa": jl,
                "jepa_const_ref": ref, "teacher_tok_std": spread})
    return out
