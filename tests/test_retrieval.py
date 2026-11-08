import numpy as np

from src import retrieval as RT


def _data(n=12, g=40, nref=15, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, g)), rng.normal(size=(nref, g))


def test_perfect_prediction_scores_one():
    T, R = _data()
    pg, S = RT.per_gene_scores(T, T, R)
    assert np.allclose(pg["ref_spearman"], 1) and np.allclose(pg["ref_p@5"], 1)
    assert np.all(pg["id_hit"] == 1) and np.all(pg["id_rank"] == 1)
    assert np.isclose(RT.cluster_agreement(S, RT.true_sim(T)), 1)


def test_constant_prediction_is_undefined_and_chance_rank():
    T, R = _data()
    P = np.tile(T.mean(0), (len(T), 1))
    pg, S = RT.per_gene_scores(P, T, R)
    assert np.isnan(RT.cluster_agreement(S, RT.true_sim(T)))
    Z = np.zeros_like(T)
    pg0, _ = RT.per_gene_scores(Z, T, R)
    assert np.all(np.isnan(pg0["ref_spearman"]))
    assert np.allclose(pg0["id_rank"], (len(T) + 1) / 2)  # all ties -> average rank


def test_per_gene_matches_aggregate():
    T, R = _data()
    rng = np.random.default_rng(1)
    P = T + rng.normal(scale=2.0, size=T.shape)
    pg, S = RT.per_gene_scores(P, T, R)
    agg = RT.aggregate(pg, RT.cluster_agreement(S, RT.true_sim(T)))
    assert np.isclose(agg["id_top1"], pg["id_hit"].mean())
    assert np.isclose(agg["id_mean_rank"], pg["id_rank"].mean())


def test_cluster_agreement_drops_duplicate_pairs():
    T, _ = _data()
    St = RT.true_sim(T)
    Sp = RT.true_sim(T + np.random.default_rng(2).normal(size=T.shape))
    idx = np.array([0, 0, 1, 2, 3, 3])
    # same as distinct genes 0..3 with each cross pair weighted by multiplicity
    v = RT.cluster_agreement(Sp, St, idx)
    a, b = np.triu_indices(len(idx), 1)
    k = idx[a] != idx[b]
    exp = np.corrcoef(Sp[idx[a][k], idx[b][k]], St[idx[a][k], idx[b][k]])[0, 1]
    assert np.isclose(v, exp)
    assert np.isnan(RT.cluster_agreement(Sp, St, np.array([1, 1, 1, 2])))  # 3 kept pairs, all the same (1,2) pair -> constant


def test_arm_of_and_pool_seeds():
    assert RT.arm_of("celljepa_s1_jepa_lr3e5") == "celljepa_jepa_lr3e5"
    assert RT.arm_of("x/runs/celljepa_s2_v1") == "x/runs/celljepa_v1"
    assert RT.arm_of("baselines/linear") == "baselines/linear"
    assert RT.arm_of("e/runs/jepa_s512_v1") == "e/runs/jepa_s512_v1"      # size tag, not a seed
    assert RT.arm_of("exp_s1_x/runs/m_s3") == "exp_s1_x/runs/m"          # experiment name untouched
    n = 4
    S = np.eye(n)
    mk = lambda hit, sp: {"per_gene": {"ref_spearman": np.array(sp), "ref_p@5": np.zeros(n),  # noqa: E731
                                       "id_hit": np.array(hit, float), "id_rank": np.ones(n)}, "S_pred": S}
    arms = RT.pool_seeds({"e/runs/m_s1_v1": mk([1, 0, 1, 0], [0.5, np.nan, 1, 0]),
                          "e/runs/m_s2_v1": mk([1, 1, 0, 0], [0.1, np.nan, np.nan, 0])})
    a = arms["e/runs/m_v1"]
    assert a["members"] == ["e/runs/m_s1_v1", "e/runs/m_s2_v1"] and len(a["S_pred"]) == 2
    assert np.allclose(a["per_gene"]["id_hit"], [1, 0.5, 0.5, 0])
    sp = a["per_gene"]["ref_spearman"]
    assert np.isclose(sp[0], 0.3) and np.isnan(sp[1]) and np.isclose(sp[2], 1)


def test_bootstrap_paired_and_deterministic():
    T, R = _data(n=15)
    rng = np.random.default_rng(3)
    sets = {}
    for name, noise in [("e/runs/good_s1", 0.3), ("e/runs/good_s2", 0.3), ("baselines/linear", 3.0)]:
        pg, S = RT.per_gene_scores(T + rng.normal(scale=noise, size=T.shape), T, R)
        sets[name] = {"per_gene": pg, "S_pred": S}
    arms = RT.pool_seeds(sets)
    St = RT.true_sim(T)
    p1, d1 = RT.bootstrap_draws(arms, St, n_boot=300, seed=0)
    p2, d2 = RT.bootstrap_draws(arms, St, n_boot=300, seed=0)
    assert np.allclose(d1["e/runs/good"]["id_top1"], d2["e/runs/good"]["id_top1"])
    cis = RT.summarize_ci(p1, d1)
    for m in RT.METRICS:
        c = cis["e/runs/good"][m]
        assert c["lo"] <= c["hi"]
    diff = RT.diff_ci(p1, d1, "e/runs/good", "baselines/linear")
    assert diff["id_top1"]["diff"] > 0 and diff["id_top1"]["frac_gt0"] > 0.9
    assert diff["id_mean_rank"]["hi"] < 0
    # self-difference is exactly zero on every resample (pairing)
    self_d = RT.diff_ci(p1, d1, "baselines/linear", "baselines/linear")
    assert self_d["ref_p@5"]["lo"] == 0 == self_d["ref_p@5"]["hi"]


def test_json_safe():
    import json
    out = RT.json_safe({"a": np.float64("nan"), "b": [np.int64(3), np.array([1.0, np.inf])], "c": "x"})
    assert out == {"a": None, "b": [3, [1.0, None]], "c": "x"}
    json.dumps(out, allow_nan=False)


def test_inverse_id_metrics():
    from src import retrieval as RT

    rng = np.random.default_rng(0)
    T = rng.normal(size=(6, 30))
    R = rng.normal(size=(10, 30))
    pg, _ = RT.per_gene_scores(T + 0.01 * rng.normal(size=T.shape), T, R)
    assert pg["inv_hit1"].mean() == 1.0 and pg["cinv_rr"].mean() == 1.0
    const = np.tile(R.mean(0), (6, 1))       # training-mean prediction: centred -> all ties -> chance rank
    pg2, _ = RT.per_gene_scores(const, T, R)
    chance_rr = np.mean([1 / k for k in range(1, 7)])
    assert np.allclose(pg2["cinv_rr"], chance_rr) and np.allclose(pg2["cinv_hit1"], 1 / 6)
    assert np.allclose(pg2["cinv_hit5"], 5 / 6)
