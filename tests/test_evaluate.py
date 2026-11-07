import math
import numpy as np
from src.evaluate import intervention_metrics, pearson, summarize, paired_bootstrap_diff


def test_delta_mse_identity():
    """MSE(pred - ctrl, obs - ctrl) == MSE(pred, obs): delta MSE is not a new objective."""
    rng = np.random.default_rng(0)
    p, o, c = rng.normal(size=(3, 100))
    assert np.isclose(((p - c) - (o - c)).__pow__(2).mean(), ((p - o) ** 2).mean())


def test_no_change_pearson_undefined_not_zero():
    rng = np.random.default_rng(1)
    ctrl, obs = rng.normal(size=(2, 50))
    m = intervention_metrics(ctrl.copy(), obs, ctrl, list(range(10)))
    assert math.isnan(m["delta_pearson"]) and math.isnan(m["de20_delta_pearson"])
    assert m["de20_direction_acc"] in (0.0,) or math.isnan(m["de20_direction_acc"])


def test_summarize_reports_coverage():
    rows = [{"delta_rmse": 1.0, "delta_pearson": float("nan"), "de20_delta_rmse": 1, "de20_delta_pearson": 0.5,
             "de20_direction_acc": 1, "abs_pearson": 0.9},
            {"delta_rmse": 3.0, "delta_pearson": 0.4, "de20_delta_rmse": 1, "de20_delta_pearson": 0.5,
             "de20_direction_acc": 1, "abs_pearson": 0.9}]
    s = summarize(rows)
    assert s["delta_rmse"] == 2.0 and s["delta_pearson_n_defined"] == 1 and s["n_interventions"] == 2


def test_perfect_prediction():
    rng = np.random.default_rng(2)
    ctrl, obs = rng.normal(size=(2, 30))
    m = intervention_metrics(obs, obs, ctrl, list(range(30)))
    assert m["delta_rmse"] == 0 and np.isclose(m["delta_pearson"], 1)


def test_paired_bootstrap_sign():
    a = np.array([1.0, 1.1, 0.9, 1.0] * 5)
    r = paired_bootstrap_diff(a, a + 0.5)
    assert r["ci_high"] < 0 and r["frac_a_lower"] == 1.0
