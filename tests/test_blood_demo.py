"""CPU tests for Blood GPT ranking helpers (no GPU, no real model)."""

import numpy as np
import pytest

from src.blood_demo import (
    CellTokens,
    abnormal_genes,
    cosine_scores,
    expression_to_tokens,
    heatmap_matrix,
    pad_cells,
    prepend_cls,
    query_compatible_with_log1p,
    deltas_for_genes,
    rank_knockdowns,
    softmax_weights,
    top_delta_genes,
)
from src.blood_demo import CLS_VALUE, PAD_VALUE


class _FakeVocab:
    def __init__(self, symbols):
        self.stoi = {s: i for i, s in enumerate(symbols)}

    def __contains__(self, tok):
        return tok in self.stoi

    def __getitem__(self, tok):
        return self.stoi[tok]

    def __len__(self):
        return len(self.stoi)


def test_softmax_weights_sum_to_one():
    w = softmax_weights(np.array([0.1, 0.9, 0.2]))
    assert w.shape == (3,)
    assert abs(w.sum() - 1.0) < 1e-9
    assert w.argmax() == 1


def test_rank_knockdowns_prefers_identical_embedding():
    rng = np.random.default_rng(0)
    library = rng.normal(size=(5, 16)).astype(np.float64)
    names = [f"G{i}" for i in range(5)]
    query = library[2].copy()
    ranked = rank_knockdowns(query, library, names, top_k=3)
    assert ranked[0]["gene"] == "G2"
    assert ranked[0]["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert ranked[0]["weight"] > ranked[1]["weight"]
    assert sum(r["weight"] for r in ranked) <= 1.0 + 1e-9


def test_cosine_scores_self():
    a = np.array([1.0, 0.0, 0.0])
    lib = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    s = cosine_scores(a, lib)
    assert s[0] == pytest.approx(1.0)
    assert s[1] == pytest.approx(0.0)


def test_abnormal_genes_picks_largest_deltas():
    genes = ["A", "B", "C", "D"]
    ref = np.array([1.0, 1.0, 1.0, 1.0])
    q = np.array([1.0, 3.0, 0.0, 1.1])
    rows = abnormal_genes(q, ref, genes, top_k=2)
    assert [r["gene"] for r in rows] == ["B", "C"]
    assert rows[0]["delta"] == pytest.approx(2.0)
    assert rows[1]["delta"] == pytest.approx(-1.0)


def test_top_delta_genes():
    genes = ["A", "B", "C"]
    d = np.array([0.1, -2.0, 1.5])
    rows = top_delta_genes(d, genes, k=2)
    assert [r["gene"] for r in rows] == ["B", "C"]


def test_deltas_for_genes_keeps_real_zeros_and_skips_missing():
    delta = np.array([0.2, -1.5, 0.0, 0.4])
    genes = ["A", "B", "C", "D"]
    rows = deltas_for_genes(delta, genes, ["C", "A", "ZZZ"])
    assert [r["gene"] for r in rows] == ["C", "A"]
    assert rows[0]["delta"] == pytest.approx(0.0)
    assert rows[1]["delta"] == pytest.approx(0.2)


def test_expression_to_tokens_bins_and_caps():
    vocab = _FakeVocab(["<pad>", "<cls>", "AAA", "BBB", "CCC", "DDD"])
    names = ["AAA", "BBB", "CCC", "DDD", "ZZZ"]
    values = np.array([0.0, 10.0, 5.0, 1.0, 99.0])  # ZZZ out of vocab; AAA zero
    tok = expression_to_tokens(names, values, vocab, max_genes=2)
    assert len(tok.gene_ids) == 2
    assert set(tok.gene_names) == {"BBB", "CCC"}
    assert tok.bins.min() >= 1 and tok.bins.max() <= 50
    # Tokens sorted by ascending vocab id
    assert list(tok.gene_ids) == sorted(tok.gene_ids)


def test_prepend_cls_and_pad():
    vocab = _FakeVocab(["<pad>", "<cls>", "AAA"])
    tok = CellTokens(gene_ids=np.array([2], dtype=np.int64),
                     bins=np.array([7.0], dtype=np.float32),
                     gene_names=["AAA"])
    g, v = prepend_cls(tok, cls_id=1)
    assert list(g) == [1, 2]
    assert v[0] == CLS_VALUE
    batch = pad_cells([(g, v), (np.array([1, 2, 2]), np.array([0.0, 3.0, 4.0]))], pad_id=0)
    assert batch["gene_ids"].shape == (2, 3)
    assert bool(batch["pad"][0, 2])
    assert batch["values"][0, 2] == PAD_VALUE


def _toy_library():
    # genes: A B C D E F; knockdowns K0 K1 K2
    gene_names = ["A", "B", "C", "D", "E", "F"]
    knockdown_names = ["K0", "K1", "K2"]
    deltas = np.array([
        [0.1, -2.0, 0.05, 0.02, 1.5, 0.01],   # K0: B down, E up
        [0.0, 0.05, -1.8, 1.2, 0.03, 0.02],  # K1: C down, D up
        [0.9, 0.01, 0.02, 0.01, 0.04, -0.8],  # K2: A up, F down
    ], dtype=np.float64)
    return deltas, knockdown_names, gene_names


def test_heatmap_matrix_selected_columns_and_sign():
    deltas, names, genes = _toy_library()
    ranked = [
        {"gene": "K1", "cosine": 0.99},
        {"gene": "K0", "cosine": 0.95},
    ]
    heat = heatmap_matrix(deltas, names, genes, ranked, per_column=2, max_rows=32)
    assert [c["gene"] for c in heat["columns"]] == ["K1", "K0"]
    assert heat["query"] is None
    assert heat["unit"] == "log1p change vs K562 control"
    # Top-2 per column: K1→C,D and K0→B,E → rows among {B,C,D,E}
    assert set(heat["genes"]).issubset({"B", "C", "D", "E"})
    assert len(heat["genes"]) == len(heat["values"])
    assert all(len(row) == 2 for row in heat["values"])
    # Sign check: K1's C delta is -1.8 (column 0 of values for gene C)
    c_idx = heat["genes"].index("C")
    assert heat["values"][c_idx][0] == pytest.approx(-1.8)
    b_idx = heat["genes"].index("B")
    assert heat["values"][b_idx][1] == pytest.approx(-2.0)


def test_heatmap_matrix_row_cap():
    deltas, names, genes = _toy_library()
    ranked = [{"gene": "K0", "cosine": 1.0}, {"gene": "K1", "cosine": 0.9}, {"gene": "K2", "cosine": 0.8}]
    heat = heatmap_matrix(deltas, names, genes, ranked, per_column=3, max_rows=3)
    assert len(heat["genes"]) == 3
    assert len(heat["values"]) == 3


def test_heatmap_query_included_for_log1p_panel():
    deltas, names, genes = _toy_library()
    ranked = [{"gene": "K0", "cosine": 1.0}]
    control = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # Full-panel Adamson-like log1p upload.
    query = np.array([1.0, 0.2, 1.0, 1.0, 2.5, 1.0])
    heat = heatmap_matrix(
        deltas, names, genes, ranked,
        query_aligned=query, control=control, n_overlap=6, per_column=2,
    )
    assert heat["query"] is not None
    assert len(heat["query"]) == len(heat["genes"])
    b_idx = heat["genes"].index("B")
    assert heat["query"][b_idx] == pytest.approx(-0.8)


def test_heatmap_query_dropped_for_bin_scale():
    deltas, names, genes = _toy_library()
    ranked = [{"gene": "K0", "cosine": 1.0}]
    control = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # Sparse census-like bins (1..50) on a small overlap.
    query = np.array([0.0, 40.0, 0.0, 0.0, 35.0, 0.0])
    heat = heatmap_matrix(
        deltas, names, genes, ranked,
        query_aligned=query, control=control, n_overlap=2, per_column=2,
    )
    assert heat["query"] is None


def test_heatmap_drops_weak_genes_and_groups_by_peak():
    deltas, names, genes = _toy_library()
    ranked = [
        {"gene": "K1", "cosine": 0.99},
        {"gene": "K0", "cosine": 0.95},
    ]
    heat = heatmap_matrix(
        deltas, names, genes, ranked, per_column=4, max_rows=32, min_abs=0.15,
    )
    # K0's 0.1 / 0.05 / 0.02 and K1's 0.05 / 0.03 stay out.
    assert set(heat["genes"]) == {"B", "C", "D", "E"}
    # Peak column first (K1 owns C and D), high signed change above low.
    assert heat["genes"] == ["D", "C", "E", "B"]


def test_query_compatible_with_log1p_gate():
    panel = np.array([2.0, 3.0, 1.5, 4.0, 2.2, 0.8])
    assert query_compatible_with_log1p(panel, n_overlap=6, n_panel=6)
    bins = np.array([0.0, 45.0, 0.0, 0.0, 30.0, 0.0])
    assert not query_compatible_with_log1p(bins, n_overlap=2, n_panel=6)
    sparse_log = np.array([0.0, 2.0, 0.0, 0.0, 1.0, 0.0])
    assert not query_compatible_with_log1p(sparse_log, n_overlap=2, n_panel=6)
