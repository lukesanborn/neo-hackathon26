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


def test_transforms_recover_a_small_consistent_shift():
    from src import retrieval as RT
    from src.latent_retrieval import fit_lda, fit_remove_pcs, fit_whiten
    rng = np.random.default_rng(0)
    d, n_pert = 20, 12
    big = np.zeros(d); big[:3] = [5.0, 4.0, 3.0]                   # large cell-state variance on 3 axes
    noise = lambda n: rng.normal(size=(n, d)) * 0.05 + rng.normal(size=(n, 1)) * big   # noqa: E731
    dirs = np.zeros((n_pert, d)); dirs[:, 3:] = rng.normal(size=(n_pert, d - 3)) * 0.05   # small perturbation shifts
    ctrl = noise(2000)
    A = np.stack([(noise(100) + dirs[i]).mean(0) for i in range(n_pert)])
    B = np.stack([(noise(100) + dirs[i]).mean(0) for i in range(n_pert)])
    raw = RT.id_scores(A - ctrl.mean(0), B - ctrl.mean(0))["id_pct"]
    for f in (fit_whiten(ctrl), fit_remove_pcs(ctrl, k=3)):
        tA, tB, tc = f(A), f(B), f(ctrl).mean(0)
        assert RT.id_scores(tA - tc, tB - tc)["id_pct"] >= raw
    X = np.concatenate([noise(50) + dirs[i] for i in range(n_pert)]); y = np.repeat(np.arange(n_pert), 50)
    g = fit_lda(X, y, n_dims=5)
    assert g(A).shape == (n_pert, 5)
