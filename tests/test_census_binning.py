import numpy as np
import pytest

from src.track_a.census import (N_BINS, bin_csr, bin_nonzero, coo_to_shard, make_val_split,
                                shard_bounds, validate_shard)


def test_single_nonzero_gene_gets_top_bin():
    assert bin_nonzero(np.array([7.0])).tolist() == [N_BINS]


def test_all_equal_values_share_one_bin():
    b = bin_nonzero(np.full(37, 3.0))
    assert (b == N_BINS).all() and b.dtype == np.uint8


def test_empty_cell():
    assert bin_nonzero(np.zeros(0)).size == 0


def test_rejects_zeros():
    with pytest.raises(ValueError):
        bin_nonzero(np.array([0.0, 1.0]))


@pytest.mark.parametrize("n", [2, 5, 49, 50, 51, 500, 5000])
def test_range_and_monotonic(n):
    rng = np.random.default_rng(n)
    v = rng.integers(1, 1000, size=n).astype(np.float32)
    b = bin_nonzero(v)
    assert b.min() >= 1 and b.max() == N_BINS
    o = np.argsort(v, kind="stable")
    assert (np.diff(b[o].astype(int)) >= 0).all()  # monotone in value
    # ties -> same bin
    for x in np.unique(v):
        assert len(set(b[v == x].tolist())) == 1


def test_distinct_values_use_full_range():
    b = bin_nonzero(np.arange(1, 1001, dtype=float))
    assert b.min() == 1 and b.max() == N_BINS and len(np.unique(b)) == N_BINS


def test_deterministic_and_matches_scgpt_rule():
    v = np.array([1, 1, 2, 3, 3, 3, 10, 50], dtype=float)
    expected = np.digitize(v, np.quantile(v, np.linspace(0, 1, N_BINS)))
    assert (bin_nonzero(v) == expected).all() and (bin_nonzero(v) == bin_nonzero(v)).all()


def test_coo_to_shard_sums_duplicates_and_sorts():
    rows = np.array([1, 0, 1, 1, 2])
    genes = np.array([9, 4, 3, 9, 5])
    vals = np.array([1.0, 2.0, 5.0, 2.0, 0.0])  # last is zero -> dropped
    sh = coo_to_shard(rows, genes, vals, n_cells=4)
    assert sh["indptr"].tolist() == [0, 1, 3, 3, 3]
    assert sh["gene_ids"].tolist() == [4, 3, 9]
    assert sh["n_duplicate_symbol_entries_summed"] == 1
    # row 1: values 5 (gene 3) and 1+2=3 (gene 9) -> gene 3 is the max
    assert sh["bins"][1] == N_BINS and sh["bins"][2] < N_BINS
    validate_shard(sh, vocab_size=10)
    with pytest.raises(ValueError):
        validate_shard(sh, vocab_size=9)


def test_bin_csr_rowwise():
    indptr = np.array([0, 2, 2, 5])
    vals = np.array([1.0, 2.0, 4.0, 4.0, 4.0])
    b = bin_csr(indptr, vals)
    assert b[1] == N_BINS and (b[2:] == N_BINS).all() and b[0] < N_BINS


def test_val_split():
    sizes = [100, 50, 1]
    val = make_val_split(sizes)
    assert len(val) == 2 and val == make_val_split(sizes)
    assert all(0 <= r < sizes[s] for s, r in val)
    assert make_val_split([10]) == [[0, make_val_split([10])[0][1]]]


def test_shard_bounds():
    assert shard_bounds(5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert shard_bounds(0, 2) == []
