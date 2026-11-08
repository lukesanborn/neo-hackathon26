import numpy as np

from src.latent_retrieval import n_expressed, shift, topk_csr


def test_topk_csr_keeps_top_genes_and_respects_k():
    rows = np.array([[0.0, 3.0, 1.0, 2.0, 0.0],      # 3 positive
                     [0.5, 0.1, 0.2, 0.3, 0.4]])     # dense (like a prediction)
    indptr, idx, data = topk_csr(rows, ks=[5, 2])
    assert indptr.tolist() == [0, 3, 5]
    assert idx[:3].tolist() == [1, 2, 3]              # all positives kept for a real-like row
    assert sorted(idx[3:].tolist()) == [0, 4]         # top-2 of the dense row
    assert np.allclose(data[3:], rows[1, [0, 4]])


def test_real_cell_roundtrip_and_shift():
    rng = np.random.default_rng(0)
    real = rng.poisson(0.7, (6, 40)).astype(np.float32)
    k = n_expressed(real)
    indptr, idx, _ = topk_csr(real, k)               # top-k with k = own expressed count = its nonzero set
    for i in range(len(real)):
        assert set(idx[indptr[i]:indptr[i + 1]].tolist()) == set(np.flatnonzero(real[i]).tolist())
    e = rng.normal(size=(4, 3))
    assert np.allclose(shift(e, e.mean(0)), 0)
