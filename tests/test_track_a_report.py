import json
import os
import time

import pytest

from src.track_a.report import (MATCHED_ARM, PAPER_ARM, aggregate_track_a, ckpt_key, collect_track_a_runs,
                                write_synthetic_layout)


def test_ckpt_key_normalises_paths():
    assert ckpt_key("/vol/trackA/checkpoints/jepa_kidney/") == "jepa_kidney"
    assert ckpt_key("/vol/trackA/checkpoints/scgpt_kidney") == "scgpt_kidney"
    assert ckpt_key("scGPT_human") == "scGPT_human"
    assert ckpt_key(None) == "scGPT_human"


def test_runs_keyed_by_checkpoint_method_seed(tmp_path):
    exp = str(tmp_path / "adamson_trackA_v1")
    info = write_synthetic_layout(exp, seeds=(1, 2))
    status, chosen = collect_track_a_runs(exp, "synthetic")
    assert len(status) == info["n_runs"] == 8
    # (method, seed) collides across the two checkpoints; (checkpoint, method, seed) must not
    assert len(chosen) == 8
    assert {k[0] for k in chosen} == {"jepa_kidney", "scgpt_kidney"}
    assert {(k[1], k[2]) for k in chosen} == {(m, s) for m in ("scgpt", "celljepa") for s in (1, 2)}
    for (ck, m, s), d in chosen.items():
        cfg = json.load(open(os.path.join(d, "config.json")))
        assert (ckpt_key(cfg["ckpt"]), cfg["method"], cfg["seed"]) == (ck, m, s)


def test_identity_comes_from_config_not_dirname(tmp_path):
    exp = str(tmp_path / "e")
    write_synthetic_layout(exp, seeds=(1,))
    # a misleadingly named duplicate of jepa_kidney/celljepa/1, written later -> latest wins, no new key
    src = os.path.join(exp, "runs", "celljepa_s1_kidneyjepa_v1")
    dst = os.path.join(exp, "runs", "zzz_misnamed")
    os.makedirs(dst)
    for f in os.listdir(src):
        with open(os.path.join(src, f), "rb") as a, open(os.path.join(dst, f), "wb") as b:
            b.write(a.read())
    future = time.time() + 10
    os.utime(os.path.join(dst, "metrics.json"), (future, future))
    _, chosen = collect_track_a_runs(exp, "synthetic")
    assert len(chosen) == 4
    assert chosen[("jepa_kidney", "celljepa", 1)] == dst


def test_split_mismatch_excluded(tmp_path):
    exp = str(tmp_path / "e")
    write_synthetic_layout(exp, seeds=(1,))
    p = os.path.join(exp, "runs", "scgpt_s1_kidneyscgpt_v1", "metrics.json")
    m = json.load(open(p))
    m["split_hash"] = "other"
    json.dump(m, open(p, "w"))
    status, chosen = collect_track_a_runs(exp, "synthetic")
    assert ("scgpt_kidney", "scgpt", 1) not in chosen
    assert any(r["status"] == "excluded_split_mismatch" for r in status)


def test_aggregate_paper_comparison_and_published(tmp_path):
    exp = str(tmp_path / "adamson_trackA_v1")
    write_synthetic_layout(exp, seeds=(1, 2))
    ctx = str(tmp_path / "adamson_trackB_v1")
    write_synthetic_layout(ctx, seeds=(1,), seed=3)
    out = str(tmp_path / "out")
    rep = aggregate_track_a(exp, out, context_exp_dirs=[ctx, str(tmp_path / "missing")], make_plots=False)
    arms = {r["arm"]: r for r in rep["summary"]}
    assert arms[PAPER_ARM]["n_seeds"] == 2 and arms[MATCHED_ARM]["n_seeds"] == 2
    assert set(arms) >= {PAPER_ARM, MATCHED_ARM, "linear", "training_mean", "no_change"}
    key = f"{PAPER_ARM} - {MATCHED_ARM}"
    assert rep["comparisons"][key]["delta_rmse"]["mean_diff"] == pytest.approx(-0.02, abs=0.01)
    assert rep["claim"]["level"] == "paper_arm_better"
    assert "gears_pearson_de" in rep["gears_columns"]
    pub = {p["arm"]: p for p in rep["published"]}
    assert pub[PAPER_ARM]["metrics"]["gears_pearson"]["published"] == 0.937
    assert pub[PAPER_ARM]["metrics"]["gears_pearson"]["ours"] is not None
    # context is reported separately, never as a comparison
    assert [c["available"] for c in rep["context_not_comparable"]] == [True, False]
    assert all("adamson_trackB" not in k for k in rep["comparisons"])
    md = open(os.path.join(out, "results.md")).read()
    assert "Track A — Cell-JEPA reimplemented from the paper (unofficial)" in md
    assert "For context only" in md
    for f in ("report.json", "results_by_run.csv", "results_summary.csv", "per_intervention_all.csv", "run_status.csv"):
        assert os.path.exists(os.path.join(out, f))


def test_missing_ckpt_is_excluded_not_defaulted(tmp_path):
    exp = str(tmp_path / "e")
    write_synthetic_layout(exp, seeds=(1,))
    for d in ("scgpt_s1_kidneyjepa_v1", "scgpt_s1_kidneyscgpt_v1"):
        p = os.path.join(exp, "runs", d, "config.json")
        cfg = json.load(open(p))
        cfg.pop("ckpt")
        json.dump(cfg, open(p, "w"))
    status, chosen = collect_track_a_runs(exp, "synthetic")
    assert len(chosen) == 2  # the two ckpt-less runs must not collapse onto ('scGPT_human', 'scgpt', 1)
    assert sum(r["status"] == "excluded_unidentified" for r in status) == 2
    # single-checkpoint context experiments may opt into the public default
    _, chosen = collect_track_a_runs(exp, "synthetic", default_ckpt="scGPT_human")
    assert ("scGPT_human", "scgpt", 1) in chosen
