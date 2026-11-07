import math

import numpy as np

from src.data import percell_pairs
from src.evaluate import PERCELL_KEYS, percell_metrics, summarize_percell
from tests.fakes import fake_dataset


def test_percell_pairs_deterministic_and_valid():
    ds = fake_dataset()
    c = ds.split["test"][0]
    t1, c1 = percell_pairs(ds, c, 10)
    t2, c2 = percell_pairs(ds, c, 10)
    assert np.array_equal(t1, t2) and np.array_equal(c1, c2)
    assert (ds.conditions[t1] == c).all() and (ds.conditions[c1] == "ctrl").all()
    assert len(t1) == len(c1) == 10


def test_percell_perfect_and_no_change():
    rng = np.random.default_rng(0)
    obs, ctrl = rng.normal(size=(2, 8, 30))
    m = percell_metrics(obs, obs, ctrl, list(range(20)), list(range(20)))
    for k in PERCELL_KEYS:
        assert np.isclose(m[k], 1.0)
    nc = percell_metrics(ctrl, obs, ctrl, None, list(range(20)))
    assert math.isnan(nc["percell_delta_pearson"])          # zero predicted delta: undefined
    assert math.isnan(nc["percell_pearson_de"])             # no rank-20 DE list
    assert not math.isnan(nc["percell_pearson"])


def test_percell_lower_than_mean_based_under_noise():
    """Single-cell noise lowers per-cell correlations relative to population means."""
    rng = np.random.default_rng(1)
    truth = rng.normal(size=50)
    obs = truth + rng.normal(scale=2.0, size=(200, 50))
    pred = np.tile(truth, (200, 1))
    m = percell_metrics(pred, obs, np.zeros_like(obs), None, list(range(20)))
    mean_r = np.corrcoef(pred.mean(0), obs.mean(0))[0, 1]
    assert m["percell_pearson"] < mean_r
    s = summarize_percell([{"condition": "a", **m}])
    assert s["percell_pearson_n_defined"] == 1
