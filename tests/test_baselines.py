import numpy as np
from src import baselines as B
from src.evaluate import evaluate_predictions, summarize
from tests.fakes import fake_dataset


def test_training_mean_is_equal_weight_and_train_only():
    ds = fake_dataset()
    ctrl = ds.control_reference()
    p = B.training_mean(ds, ds.split["test"], ctrl)
    expected = np.mean([ds.observed_mean(c) for c in ds.split["train"] if c != "ctrl"], axis=0)
    assert np.allclose(p[ds.split["test"][0]], expected)


def test_linear_uses_no_heldout_outcomes():
    ds = fake_dataset()
    ctrl = ds.control_reference()
    p1, _ = B.linear_gene_embedding(ds, ds.split["test"], ctrl)
    ds2 = fake_dataset()
    for c in ds2.split["val"] + ds2.split["test"]:
        ds2.X[ds2.cells_of(c)] += 100.0  # corrupt held-out outcomes; predictions must not move
    p2, _ = B.linear_gene_embedding(ds2, ds2.split["test"], ctrl)
    for c in p1:
        assert np.allclose(p1[c], p2[c])


def test_linear_beats_training_mean_on_low_rank_truth():
    ds = fake_dataset(low_rank=True)
    ctrl = ds.control_reference()
    lin, _ = B.linear_gene_embedding(ds, ds.split["test"], ctrl, K=3)
    tm = B.training_mean(ds, ds.split["test"], ctrl)
    r_lin = summarize(evaluate_predictions(lin, ds, ds.split["test"], ctrl))["delta_rmse"]
    r_tm = summarize(evaluate_predictions(tm, ds, ds.split["test"], ctrl))["delta_rmse"]
    assert r_lin < r_tm


def test_fit_linear_closed_form():
    rng = np.random.default_rng(0)
    G, P = rng.normal(size=(30, 4)), rng.normal(size=(10, 4))
    W = rng.normal(size=(4, 4))
    Y = G @ W @ P.T
    assert np.allclose(B.fit_linear(Y, G, P, 1e-9), W, atol=1e-5)
