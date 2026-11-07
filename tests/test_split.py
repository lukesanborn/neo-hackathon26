from tests.fakes import fake_dataset
from src.data import target_gene, split_hash


def test_split_disjoint_by_gene():
    ds = fake_dataset()
    tr = {target_gene(c) for c in ds.split["train"] if c != "ctrl"}
    for c in ds.split["val"] + ds.split["test"]:
        assert target_gene(c) not in tr


def test_split_hash_order_invariant():
    a = {"train": ["b", "a"], "val": ["c"], "test": ["d"]}
    b = {"test": ["d"], "val": ["c"], "train": ["a", "b"]}
    assert split_hash(a) == split_hash(b)


def test_eval_controls_are_controls():
    ds = fake_dataset()
    for c, idx in ds.eval_controls.items():
        assert (ds.conditions[idx] == "ctrl").all()
