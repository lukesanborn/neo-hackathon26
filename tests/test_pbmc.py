"""PBMC zero-shot helpers: tokenisation follows scGPT's embed_data; ASW scaling; PCA baseline shape."""
import numpy as np
import pytest

from src.track_a.pbmc import cell_tokens, column_vocab_ids, pad_batch


class V(dict):
    def __contains__(self, k):
        return dict.__contains__(self, k)


def test_column_ids_drop_unknown_genes():
    v = V({"A": 5, "B": 7})
    assert column_vocab_ids(["A", "X", "B"], v).tolist() == [5, -1, 7]


def test_cell_tokens_keeps_nonzero_in_vocab_and_bins():
    col = np.array([10, -1, 11, 12, 13])
    idx = np.array([0, 1, 2, 3, 4])
    cnt = np.array([5.0, 9.0, 0.0, 1.0, 3.0])
    g, b = cell_tokens(idx, cnt, col, max_genes=100, rng=np.random.default_rng(0))
    assert g.tolist() == [10, 12, 13]                     # zero and out-of-vocab genes dropped
    assert b.min() >= 1 and b.max() <= 50
    assert b[0] > b[2] > b[1]                              # bins preserve order of counts


def test_cell_tokens_subsamples_to_max_genes_deterministically():
    col = np.arange(100, 400)
    idx = np.arange(300)
    cnt = np.arange(1, 301, dtype=float)
    a = cell_tokens(idx, cnt, col, 50, np.random.default_rng(1))
    b = cell_tokens(idx, cnt, col, 50, np.random.default_rng(1))
    assert len(a[0]) == 50 and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_pad_batch_conventions():
    cells = [(np.array([3, 4]), np.array([1.0, 50.0], dtype=np.float32)), (np.array([5]), np.array([7.0], dtype=np.float32))]
    out = pad_batch(cells, cls_id=1, pad_id=0, cls_value=-2.0, pad_value=-2.0)
    assert out["gene_ids"].tolist() == [[1, 3, 4], [1, 5, 0]]
    assert out["values"][:, 0].tolist() == [-2.0, -2.0] and out["values"][1, 2] == -2.0
    assert out["pad"].tolist() == [[False, False, False], [False, False, True]]


def test_asw_and_pca_baseline():
    pytest.importorskip("sklearn")
    import scipy.sparse as sp

    from src.track_a.pbmc import asw_label, pca_embedding

    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal(0, 0.1, (50, 4)), rng.normal(5, 0.1, (50, 4))])
    lab = np.array([0] * 50 + [1] * 50)
    assert asw_label(emb, lab) > 0.9                       # well-separated clusters -> near 1
    X = sp.csr_matrix(rng.poisson(1.0, (60, 30)).astype(np.float32))
    p = pca_embedding(X.indptr, X.indices, X.data, 30, n_comps=5)
    assert p.shape == (60, 5) and np.isfinite(p).all()
