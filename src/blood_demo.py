"""Blood GPT demo: bin a cell profile, embed with the blood JEPA student, rank knockdowns.

The pretrained student emits a <cls> embedding. It does not emit knockdown probabilities.
Ranking is cosine similarity of the query embedding against a library of Adamson knockdown
mean profiles (embedded the same way), then a softmax over those scores for relative match
weights. The UI should present these as "closest knockdown signatures in this screen", not as
clinical probabilities.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.track_a.census import N_BINS, bin_nonzero

# Keep these local so CPU tests do not need torch (src.models / pretrain_model import it).
CLS_VALUE = 0.0
PAD_VALUE = -2.0
HGNC_ALIASES = {
    "AARS": "AARS1", "CARS": "CARS1", "DARS": "DARS1", "HARS": "HARS1", "MARS": "MARS1",
    "QARS": "QARS1", "SARS": "SARS1", "TARS": "TARS1", "ATP5B": "ATP5F1B",
    "SLMO2": "PRELID3B", "SRPR": "SRPRA",
}

MAX_GENES = 600
DEFAULT_TOP_K = 8
DEFAULT_CARD_DELTAS = 6
SOFTMAX_TEMP = 0.07  # fallback; adaptive temp used when scores are tightly clustered
HEATMAP_PER_COLUMN = 14
HEATMAP_MAX_ROWS = 60
HEATMAP_MIN_ABS = 0.15  # drop near-zero genes so the matrix is the responsive program
HEATMAP_UNIT = "log1p change vs K562 control"
# Adamson GEARS log1p is typically well under 10; census bins run 1..50.
HEATMAP_QUERY_MAX_LOG1P = 12.0
HEATMAP_QUERY_MIN_COVERAGE = 0.5


@dataclass
class CellTokens:
    """One cell ready for the blood JEPA encoder (no padding yet)."""

    gene_ids: np.ndarray   # int64, length n (+ no <cls>; caller prepends)
    bins: np.ndarray       # float32 bin indices 1..50
    gene_names: list[str]  # symbols aligned with gene_ids


def resolve_symbol(name: str, vocab: Any) -> str | None:
    """Map a gene symbol into the vocab, applying HGNC aliases when needed."""
    if name in vocab:
        return name
    alias = HGNC_ALIASES.get(name)
    if alias is not None and alias in vocab:
        return alias
    return None


def expression_to_tokens(
    gene_names: list[str],
    values: np.ndarray,
    vocab: Any,
    max_genes: int = MAX_GENES,
) -> CellTokens:
    """Dense expression -> top ``max_genes`` nonzero tokens, quantile-binned 1..50.

    Zeros are dropped (pretraining never stored zero-valued gene tokens). Values may be raw
    counts or log1p; binning is within-cell rank, so the scale only needs to preserve order
    among nonzero genes.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(gene_names) != values.shape[0]:
        raise ValueError(f"genes ({len(gene_names)}) != values ({values.shape[0]})")
    kept_names: list[str] = []
    kept_vals: list[float] = []
    for name, v in zip(gene_names, values):
        if not np.isfinite(v) or v <= 0:
            continue
        resolved = resolve_symbol(str(name), vocab)
        if resolved is None:
            continue
        kept_names.append(resolved)
        kept_vals.append(float(v))
    if not kept_names:
        raise ValueError("no in-vocab nonzero genes in the uploaded cell")
    order = np.argsort(-np.asarray(kept_vals))[:max_genes]
    # Sort selected tokens by ascending vocab id (matches census cache convention).
    sel_names = [kept_names[i] for i in order]
    sel_vals = np.asarray([kept_vals[i] for i in order], dtype=np.float64)
    ids = np.asarray([vocab[n] for n in sel_names], dtype=np.int64)
    sort = np.argsort(ids, kind="mergesort")
    ids = ids[sort]
    sel_vals = sel_vals[sort]
    sel_names = [sel_names[i] for i in sort]
    bins = bin_nonzero(sel_vals, n_bins=N_BINS).astype(np.float32)
    return CellTokens(gene_ids=ids, bins=bins, gene_names=sel_names)


def prepend_cls(tokens: CellTokens, cls_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (gene_ids, values) with <cls> at position 0."""
    gids = np.concatenate([[cls_id], tokens.gene_ids]).astype(np.int64)
    vals = np.concatenate([[CLS_VALUE], tokens.bins.astype(np.float32)]).astype(np.float32)
    return gids, vals


def pad_cells(
    cells: list[tuple[np.ndarray, np.ndarray]],
    pad_id: int,
) -> dict[str, np.ndarray]:
    """Pad variable-length (gene_ids, values) cells into a batch with a boolean pad mask."""
    if not cells:
        raise ValueError("empty batch")
    L = max(len(g) for g, _ in cells)
    B = len(cells)
    gene_ids = np.full((B, L), pad_id, dtype=np.int64)
    values = np.full((B, L), PAD_VALUE, dtype=np.float32)
    pad = np.ones((B, L), dtype=bool)
    for b, (g, v) in enumerate(cells):
        n = len(g)
        gene_ids[b, :n] = g
        values[b, :n] = v
        pad[b, :n] = False
    return {"gene_ids": gene_ids, "values": values, "pad": pad}


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def cosine_scores(query: np.ndarray, library: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query (D,) against a library (N, D)."""
    q = l2_normalize(np.asarray(query, dtype=np.float64).reshape(1, -1))[0]
    lib = l2_normalize(np.asarray(library, dtype=np.float64), axis=1)
    return lib @ q


def softmax_weights(scores: np.ndarray, temperature: float | None = None) -> np.ndarray:
    """Relative match weights over the candidate set (sums to 1).

    When ``temperature`` is None, scale by the score spread so a clear top hit still gets
    mass even if raw cosines sit in a tiny band near 1.0 (common for mean-profile queries).
    """
    s = np.asarray(scores, dtype=np.float64)
    if temperature is None:
        spread = float(s.max() - s.min()) if s.size else 0.0
        temperature = max(spread / 6.0, 1e-4) if spread > 0 else SOFTMAX_TEMP
    z = (s - s.max()) / max(temperature, 1e-6)
    e = np.exp(z)
    return e / e.sum()


def rank_knockdowns(
    query_emb: np.ndarray,
    library_embs: np.ndarray,
    names: list[str],
    top_k: int = DEFAULT_TOP_K,
    temperature: float | None = None,
) -> list[dict]:
    """Rank library knockdowns by cosine similarity; attach relative match weights."""
    if len(names) != len(library_embs):
        raise ValueError("names and library_embs length mismatch")
    scores = cosine_scores(query_emb, library_embs)
    order = np.argsort(-scores)[:top_k]
    # Relative weights among the shown cards (not diluted across the full library).
    top_scores = scores[order]
    weights = softmax_weights(top_scores, temperature=temperature)
    out = []
    for rank, i in enumerate(order):
        out.append({
            "gene": names[int(i)],
            "cosine": float(scores[i]),
            "weight": float(weights[rank]),
        })
    return out


def abnormal_genes(
    query_values: np.ndarray,
    reference_values: np.ndarray,
    gene_names: list[str],
    top_k: int = 20,
) -> list[dict]:
    """Largest absolute deviations from a healthy reference (for the abnormality chart)."""
    q = np.asarray(query_values, dtype=np.float64)
    r = np.asarray(reference_values, dtype=np.float64)
    if q.shape != r.shape or len(gene_names) != q.shape[0]:
        raise ValueError("query, reference, and gene_names must align")
    delta = q - r
    order = np.argsort(-np.abs(delta))[:top_k]
    return [
        {"gene": gene_names[int(i)], "delta": float(delta[i]), "query": float(q[i]), "ref": float(r[i])}
        for i in order
    ]


def top_delta_genes(
    delta: np.ndarray,
    gene_names: list[str],
    k: int = DEFAULT_CARD_DELTAS,
) -> list[dict]:
    """Strongest up/down genes in a knockdown's mean change-from-control profile."""
    d = np.asarray(delta, dtype=np.float64)
    order = np.argsort(-np.abs(d))[:k]
    return [{"gene": gene_names[int(i)], "delta": float(d[i])} for i in order]


def deltas_for_genes(
    delta: np.ndarray,
    gene_names: list[str],
    genes: list[str],
) -> list[dict]:
    """Change-from-control for ``genes``, in that order.

    A gene missing from the panel is omitted. A real zero stays, so callers can
    tell "this gene did not move" from "we have no measurement."
    """
    d = np.asarray(delta, dtype=np.float64)
    if d.ndim != 1 or len(gene_names) != d.shape[0]:
        raise ValueError("gene_names length must match delta")
    index = {g: i for i, g in enumerate(gene_names)}
    out = []
    for gene in genes:
        i = index.get(gene)
        if i is None:
            continue
        out.append({"gene": gene, "delta": round(float(d[i]), 4)})
    return out


def query_compatible_with_log1p(
    aligned_values: np.ndarray,
    n_overlap: int,
    n_panel: int,
    min_coverage: float = HEATMAP_QUERY_MIN_COVERAGE,
    max_log1p: float = HEATMAP_QUERY_MAX_LOG1P,
) -> bool:
    """True when the upload looks like Adamson-scale log1p on most of the gene panel.

    Blood-census cells are within-cell bins 1..50 and usually cover a small fraction of
    the Adamson panel; mixing those with measured log1p deltas would corrupt the color scale.
    """
    if n_panel <= 0 or n_overlap <= 0:
        return False
    if n_overlap / n_panel < min_coverage:
        return False
    vals = np.asarray(aligned_values, dtype=np.float64)
    # Zeros mark genes absent from the upload (aligned sparse fill).
    present = vals[vals != 0.0]
    if present.size == 0:
        return False
    return float(np.max(np.abs(present))) <= max_log1p


def _order_heatmap_rows(block: np.ndarray) -> list[int]:
    """Group rows by the column of their peak |change|, then by that column's signed value.

    ``block`` is (n_rows, n_cols). Returns row indices. Genes that move together under the
    same knockdown sit in one band, which is the usual Perturb-seq reading order.
    """
    if block.size == 0:
        return []
    peak = np.argmax(np.abs(block), axis=1)
    groups: dict[int, list[int]] = {}
    for ri, col in enumerate(peak.tolist()):
        groups.setdefault(int(col), []).append(ri)
    ordered: list[int] = []
    for col in range(block.shape[1]):
        rows = groups.get(col, [])
        rows.sort(key=lambda ri: -float(block[ri, col]))
        ordered.extend(rows)
    return ordered


def heatmap_matrix(
    deltas: np.ndarray,
    knockdown_names: list[str],
    gene_names: list[str],
    ranked: list[dict],
    query_aligned: np.ndarray | None = None,
    control: np.ndarray | None = None,
    n_overlap: int | None = None,
    per_column: int = HEATMAP_PER_COLUMN,
    max_rows: int = HEATMAP_MAX_ROWS,
    min_abs: float = HEATMAP_MIN_ABS,
) -> dict:
    """Slice measured knockdown deltas into a genes × matches heatmap payload.

    ``deltas`` is (n_knockdowns, n_genes) change-from-control. Rows are the union of each
    matched column's strongest genes, dropping |delta| below ``min_abs``, capped at
    ``max_rows``. ``query`` is included only when the aligned upload passes the log1p scale gate.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.ndim != 2:
        raise ValueError("deltas must be 2-D (knockdowns, genes)")
    if len(knockdown_names) != deltas.shape[0]:
        raise ValueError("knockdown_names length must match deltas rows")
    if len(gene_names) != deltas.shape[1]:
        raise ValueError("gene_names length must match deltas columns")

    name_to_i = {n: i for i, n in enumerate(knockdown_names)}
    col_idxs: list[int] = []
    columns: list[dict] = []
    for row in ranked:
        g = str(row["gene"])
        if g not in name_to_i:
            continue
        col_idxs.append(name_to_i[g])
        columns.append({"gene": g, "cosine": float(row.get("cosine", 0.0))})

    empty = {
        "genes": [],
        "columns": columns,
        "values": [],
        "unit": HEATMAP_UNIT,
        "query": None,
    }
    if not col_idxs:
        return empty

    selected: list[int] = []
    seen: set[int] = set()
    for ci in col_idxs:
        order = np.argsort(-np.abs(deltas[ci]))
        taken = 0
        for gi in order:
            gi = int(gi)
            if abs(float(deltas[ci, gi])) < min_abs:
                break
            if gi in seen:
                continue
            seen.add(gi)
            selected.append(gi)
            taken += 1
            if taken >= per_column or len(selected) >= max_rows:
                break
        if len(selected) >= max_rows:
            break

    if not selected:
        return empty

    block = deltas[np.ix_(col_idxs, selected)].T  # genes × matched columns
    selected = [selected[i] for i in _order_heatmap_rows(block)]

    values = [
        [round(float(deltas[ci, gi]), 4) for ci in col_idxs]
        for gi in selected
    ]
    genes_out = [gene_names[gi] for gi in selected]

    query_out = None
    if query_aligned is not None and control is not None:
        q = np.asarray(query_aligned, dtype=np.float64)
        c = np.asarray(control, dtype=np.float64)
        if q.shape == c.shape == (len(gene_names),):
            overlap = int(n_overlap) if n_overlap is not None else int(np.count_nonzero(q))
            if query_compatible_with_log1p(q, overlap, len(gene_names)):
                qdelta = q - c
                query_out = [round(float(qdelta[gi]), 4) for gi in selected]

    return {
        "genes": genes_out,
        "columns": columns,
        "values": values,
        "unit": HEATMAP_UNIT,
        "query": query_out,
    }


def encode_cls_batch(model, batch: dict, device: str = "cuda"):
    """Encode padded cells; return <cls> embeddings (B, D) as float32 numpy."""
    import torch

    gids = torch.as_tensor(batch["gene_ids"], device=device)
    vals = torch.as_tensor(batch["values"], device=device)
    pad = torch.as_tensor(batch["pad"], device=device)
    model.eval()
    with torch.no_grad():
        x = model.encoder(gids) + model.value_encoder(vals)
        h = model.transformer_encoder(x, src_key_padding_mask=pad)
        cls = h[:, 0].float().cpu().numpy()
    return cls


def load_student(ckpt_path: str, vocab_path: str, args: dict | None = None, device: str = "cuda"):
    """Build the blood JEPA student and load ``best_student.pt`` / ``best_model.pt`` weights."""
    import torch

    from src.models import SimpleVocab
    from src.track_a.pretrain_model import build_student

    vocab = SimpleVocab(vocab_path)
    args = args or {}
    model = build_student(
        vocab,
        d_model=int(args.get("embsize", args.get("d_model", 512))),
        nhead=int(args.get("nheads", args.get("nhead", 8))),
        d_hid=int(args.get("d_hid", 512)),
        nlayers=int(args.get("nlayers", 12)),
        nlayers_cls=int(args.get("n_layers_cls", args.get("nlayers_cls", 3))),
        dropout=0.0,
        seed=0,
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
        sd = sd["model"]
    if isinstance(sd, dict) and "state" in sd and isinstance(sd["state"], dict):
        sd = sd["state"].get("model", sd["state"])
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    sd = {k.replace("Wqkv.", "in_proj_"): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Encoder stack must load; decoder / cls heads may differ across export formats.
    bad = [k for k in missing if k.startswith(("encoder.", "value_encoder.", "transformer_encoder."))]
    if bad:
        raise RuntimeError(f"blood JEPA load missing encoder keys: {bad[:12]}")
    model.to(device).eval()
    return model, vocab


def find_blood_checkpoint(vol_root: str = "/vol") -> dict:
    """Resolve Vincent's blood run weights + vocab on the Modal volume.

    Prefers the exported checkpoint contract (args.json / best_model.pt / vocab.json), then the
    training run directory (best_student.pt).
    """
    candidates = [
        # Strongest completed retrieval model: blood-JEPA -> Cell-JEPA fine-tune + contrastive loss.
        os.path.join(vol_root, "experiments/replogle_k562_v1/runs/celljepa_s5_s2bjepa_r1"),
        os.path.join(vol_root, "trackA/checkpoints/jepa_kidney_blood_lr3e5"),
        os.path.join(vol_root, "trackA/pretrain/runs/jepa_blood_lr3e5"),
    ]
    chosen = None
    ckpt = None
    for d in candidates:
        if not os.path.isdir(d):
            continue
        for name in ("best.pt", "best_model.pt", "best_student.pt"):
            p = os.path.join(d, name)
            if os.path.exists(p) and os.path.getsize(p) > 0:
                chosen, ckpt = d, p
                break
        if chosen is not None:
            break
    if chosen is None or ckpt is None:
        raise FileNotFoundError(
            "blood JEPA checkpoint not found; expected "
            f"{candidates[0]}/best_model.pt or {candidates[1]}/best_student.pt"
        )
    args_path = os.path.join(chosen, "args.json")
    config_path = os.path.join(chosen, "config.json")
    args = {}
    if os.path.exists(args_path) and os.path.getsize(args_path) > 0:
        with open(args_path) as f:
            args = json.load(f)
    elif os.path.exists(config_path):
        with open(config_path) as f:
            args = json.load(f)
    vocab_candidates = [
        os.path.join(chosen, "vocab.json"),
        os.path.join(vol_root, "checkpoints/scGPT_human/vocab.json"),
    ]
    vocab_path = next((p for p in vocab_candidates if os.path.exists(p) and os.path.getsize(p) > 0), None)
    if vocab_path is None:
        raise FileNotFoundError("vocab.json not found next to the blood checkpoint or under scGPT_human")
    return {"run_dir": chosen, "ckpt": ckpt, "vocab": vocab_path, "args": args}


def build_library_profiles(ds, conditions: list[str] | None = None) -> dict:
    """Adamson knockdown mean profiles + deltas vs control (for cards)."""
    from src.data import target_gene

    ctrl = ds.control_reference()
    conds = conditions or sorted(
        c for split in ("train", "val", "test") for c in ds.split[split] if c != "ctrl"
    )
    names, means, deltas = [], [], []
    for c in conds:
        g = target_gene(c)
        if g is None:
            continue
        # Prefer current HGNC symbol when the Adamson label is outdated.
        names.append(HGNC_ALIASES.get(g, g))
        m = ds.observed_mean(c)
        means.append(m)
        deltas.append(m - ctrl)
    return {
        "names": names,
        "means": np.stack(means).astype(np.float32),
        "deltas": np.stack(deltas).astype(np.float32),
        "gene_names": list(ds.gene_names),
        "control": ctrl.astype(np.float32),
        "conditions": conds,
    }
