import numpy as np
import pytest

from src.track_a import census as C


def test_corpus_registry():
    assert set(C.CORPORA) >= {"kidney", "blood_marrow"}
    for spec in C.CORPORA.values():
        assert {"filter", "out", "smoke_out", "max_cells", "sample", "rationale"} <= set(spec)
        assert spec["out"] != spec["smoke_out"]
    assert len({s["out"] for s in C.CORPORA.values()}) == len(C.CORPORA)
    with pytest.raises(ValueError):
        C.get_corpus("nope")


def test_kidney_defaults_unchanged():
    k = C.CORPORA["kidney"]
    assert C.OBS_VALUE_FILTER == "tissue_general == 'kidney' and disease == 'normal' and suspension_type != 'na'"
    assert k["filter"] == C.OBS_VALUE_FILTER
    assert k["out"] == "/vol/trackA/kidney_census_2023-05-15"
    assert k["smoke_out"] == "/vol/smoke/census/kidney_census_2023-05-15"
    assert k["sample"] == "first" and k["max_cells"] is None and not k["leakage_check"]
    vocab = {"A": 0, "B": 1}
    # provenance dict must equal the pre-registry one so the existing kidney plan.json stays valid
    assert C.plan_provenance(vocab) == C.plan_provenance(vocab, "kidney") == {
        "census_version": "2023-05-15", "organism": "Homo sapiens", "obs_value_filter": C.OBS_VALUE_FILTER,
        "X_layer": "raw", "n_bins": 50, "vocab_sha256": C.sha256_json(vocab)}
    assert C.CENSUS_VERSION == "2023-05-15" and C.N_BINS == 50 and C.VAL_FRACTION == 0.01


def test_blood_marrow_spec():
    b = C.CORPORA["blood_marrow"]
    assert b["filter"] == ("tissue_general in ['blood', 'bone marrow'] and disease == 'normal' "
                           "and suspension_type != 'na'")
    assert b["out"] == "/vol/trackA/blood_marrow_census_2023-05-15_2M"
    assert b["max_cells"] == 2_000_000 and b["sample"] == "random" and b["sample_seed"] == 0
    p = C.plan_provenance({"A": 0}, "blood_marrow")
    assert p["corpus"] == "blood_marrow" and p["sample"] == "random" and p["sample_seed"] == 0
    assert p != C.plan_provenance({"A": 0}, "kidney")


def test_select_first_matches_old_behaviour():
    assert C.select_cells(10, None, "first").tolist() == list(range(10))
    assert C.select_cells(10, 4, "first").tolist() == [0, 1, 2, 3]
    assert C.select_cells(10, 50, "first").tolist() == list(range(10))


def test_select_random_deterministic_sorted_sized():
    a = C.select_cells(100_000, 2_000, "random", seed=0)
    b = C.select_cells(100_000, 2_000, "random", seed=0)
    c = C.select_cells(100_000, 2_000, "random", seed=1)
    assert a.size == 2_000 and np.array_equal(a, b) and not np.array_equal(a, c)
    assert (np.diff(a) > 0).all() and a.min() >= 0 and a.max() < 100_000
    assert a.max() > 50_000  # spread over the range, not a prefix
    assert C.select_cells(10, 10, "random").tolist() == list(range(10))
    assert C.select_cells(10, None, "random").tolist() == list(range(10))
    with pytest.raises(ValueError):
        C.select_cells(10, 3, "bogus")
    with pytest.raises(ValueError):
        C.select_cells(10, -1, "random")


def test_leakage_check():
    rows = [
        {"dataset_id": "d1", "dataset_title": "4k PBMCs from a Healthy Donor", "collection_name": "10x Genomics datasets"},
        {"dataset_id": "d2", "dataset_title": "pbmc8k", "collection_name": "Benchmarks"},
        {"dataset_id": "d3", "dataset_title": "PBMCs from COVID-19 convalescents", "collection_name": "Immune atlas"},
        {"dataset_id": "d4", "dataset_title": "Bone marrow HSPCs", "collection_name": "Marrow atlas"},
        {"dataset_id": "d5", "dataset_title": "pbmc4k", "collection_name": "not in slice"},
    ]
    out = C.leakage_check(rows, ["d1", "d2", "d3", "d4"])
    assert out["excluded_dataset_ids"] == ["d1", "d2"]
    assert [r["dataset_id"] for r in out["candidates_reviewed_not_excluded"]] == ["d3"]  # PBMC kept
    assert out["n_slice_datasets_screened"] == 4
    assert C.leakage_check(rows, ["d3", "d4"])["excluded_dataset_ids"] == []


def test_remove_stale_shards(tmp_path):
    d = tmp_path / "shards"
    d.mkdir()
    for k in range(4):
        (d / f"shard_{k:03d}.npz").write_bytes(b"x")
        (d / f"shard_{k:03d}.json").write_text("{}")
    removed = C.remove_stale_shards(str(tmp_path), 2)
    assert len(removed) == 4
    assert sorted(p.name for p in d.iterdir()) == ["shard_000.json", "shard_000.npz", "shard_001.json", "shard_001.npz"]
