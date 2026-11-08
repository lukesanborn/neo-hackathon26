"""Perturbation-pair JEPA views (A28): sampler, cache builder, teacher target, end-to-end CPU loop."""
import json

import numpy as np
import pytest

from src.track_a.perturb_pairs import PairSampler

torch = pytest.importorskip("torch")

from src.track_a import pretrain_model as pm  # noqa: E402
from src.track_a.pretrain import (KidneyCache, PretrainConfig, add_pair_views, batch_rng, collate,  # noqa: E402
                                  make_synthetic_cache)

from tests.test_pretrain_objectives import AMP, D, H, TinyModel, _tiny_student  # noqa: E402


def test_sampler_same_group_never_self_and_pool_only():
    groups = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    pool = np.array([0, 1, 2, 3, 4, 5, 6, 9])          # 7, 8 held out of the pool
    s = PairSampler(groups, pool)
    rng = np.random.default_rng(0)
    cells = np.array([0, 3, 5, 6, 9])
    p = s.sample(cells, 4, rng).reshape(len(cells), 4)
    for c, row in zip(cells, p):
        assert (groups[row] == groups[c]).all()
        assert set(row) <= set(pool)
        if c != 9:                                       # group 3 has one member: falls back to itself
            assert c not in row
    assert (p[-1] == 9).all()
    assert s.stats()["n_singleton_groups"] == 1


def test_config_guards():
    with pytest.raises(AssertionError):
        PretrainConfig(objective="scgpt", view_mode="perturb_pair")
    with pytest.raises(AssertionError):
        PretrainConfig(objective="jepa", view_mode="perturb_pair", jepa_variant="token")
    c = PretrainConfig(objective="jepa", view_mode="perturb_pair", jepa_variant="center", pair_k=4)
    assert any("perturbation pairs" in d for d in c.deviations)
    assert PretrainConfig(objective="jepa").view_mode == "same_cell"


@pytest.fixture
def grouped_cache(tmp_path):
    vocab = {f"G{i}": i for i in range(200)}
    vp = tmp_path / "vocab.json"
    vp.write_text(json.dumps(vocab))
    out = tmp_path / "cache"
    make_synthetic_cache(str(out), str(vp), n_cells=60, n_shards=3, n_universe=150, seed=0)
    np.save(out / "groups.npy", (np.arange(60) % 6).astype(np.int32))
    return str(out), str(vp)


def test_pair_views_student_batch_unchanged_and_teacher_target_is_partner_mean(grouped_cache):
    c = KidneyCache(grouped_cache[0])
    assert c.groups is not None and len(c.groups) == c.n_cells
    cfg = PretrainConfig(objective="jepa", view_mode="perturb_pair", jepa_variant="center", pair_k=3)
    cells = c.train[:8]
    plain = collate(c, cells, 600, 0.15, batch_rng(0, 0, 0), cls_id=201, pad_id=200)
    nb = add_pair_views(collate(c, cells, 600, 0.15, batch_rng(0, 0, 0), cls_id=201, pad_id=200),
                        c, cfg, cells, PairSampler(c.groups, c.train), (0, 0, 0), 201, 200)
    for k in plain:
        assert np.array_equal(plain[k], nb[k]), "student batch must match a same_cell run"
    assert nb["t_gene_ids"].shape[0] == 8 * 3

    torch.manual_seed(0)
    student, teacher = TinyModel(ntoken=202), TinyModel(ntoken=202)
    pred = pm.CLSPredictor(D)
    b = {k: torch.from_numpy(v) for k, v in nb.items() if k in
         ("gene_ids", "values", "masked", "pad", "t_gene_ids", "t_values", "t_pad")}
    out = pm.forward_losses("jepa", student, teacher, pred, b, H, AMP, "faithful")
    # recompute the target by hand: teacher <cls> of each partner, averaged over the k partners
    with torch.no_grad():
        Ht = teacher.transformer_encoder(teacher.encoder(b["t_gene_ids"]) + teacher.value_encoder(b["t_values"]),
                                         src_key_padding_mask=b["t_pad"])[:, 0].view(8, 3, D).mean(1)
        z = pred(pm.encode(student, b["gene_ids"], b["values"].masked_fill(b["masked"], pm.MASK_VALUE),
                           pm.build_attn_mask(b["masked"], b["pad"], H))[:, 0])
    assert torch.allclose(out["jepa"], pm.jepa_cosine_loss(z, Ht), atol=1e-5)


def test_pair_run_cpu_end_to_end(grouped_cache, tmp_path, monkeypatch):
    import src.track_a.pretrain as P

    monkeypatch.setattr(P, "build_student", _tiny_student)
    out, vp = grouped_cache
    base = dict(objective="jepa", d_model=D, nhead=H, batch_size=8, log_every=2, max_val_cells=16,
                jepa_variant="center", view_mode="perturb_pair", pair_k=2, epochs=1, precision="fp32")
    s = P.run_pretraining(PretrainConfig(**base), out, vp, str(tmp_path / "r"), str(tmp_path / "e"), device="cpu")
    assert s["status"] == "completed" and np.isfinite(s["best_val_recon"])
    cfg = json.load(open(tmp_path / "r" / "config.json"))
    assert cfg["pair_info"]["train"]["n_groups"] == 6


def test_pair_run_needs_groups(tmp_path, monkeypatch):
    import src.track_a.pretrain as P

    monkeypatch.setattr(P, "build_student", _tiny_student)
    vocab = {f"G{i}": i for i in range(200)}
    vp = tmp_path / "vocab.json"
    vp.write_text(json.dumps(vocab))
    make_synthetic_cache(str(tmp_path / "c"), str(vp), n_cells=30, n_shards=1, n_universe=150, seed=0)
    cfg = PretrainConfig(objective="jepa", d_model=D, nhead=H, batch_size=8, view_mode="perturb_pair", epochs=1)
    with pytest.raises(ValueError, match="groups.npy"):
        P.run_pretraining(cfg, str(tmp_path / "c"), str(vp), str(tmp_path / "r"), str(tmp_path / "e"), device="cpu")


class _FakeDS:
    """Minimal Dataset stand-in for build_perturb_cache."""

    def __init__(self):
        rng = np.random.default_rng(0)
        self.gene_names = [f"G{i}" for i in range(20)] + ["OLD1"]
        self.X = np.log1p(rng.poisson(1.0, size=(50, 21)).astype(np.float32))
        self.cond = np.array(["ctrl"] * 10 + ["A+ctrl"] * 10 + ["B+ctrl"] * 10 + ["V+ctrl"] * 10 + ["T+ctrl"] * 10)
        self.split = {"train": ["ctrl", "A+ctrl", "B+ctrl"], "val": ["V+ctrl"], "test": ["T+ctrl"]}
        self.manifest = {"split_hash": "abc"}

    def cells_of(self, c):
        return np.flatnonzero(self.cond == c)


def test_build_perturb_cache_excludes_heldout_and_loads(tmp_path):
    from src.track_a.perturb_pairs import build_perturb_cache

    ds = _FakeDS()
    vocab = {f"G{i}": i for i in range(20)}
    vocab["NEW1"] = 20
    man = build_perturb_cache(ds, str(tmp_path / "pc"), vocab, {"OLD1": "NEW1"}, val_frac=0.1, shard_size=12)
    assert man["n_cells"] == 30 and man["n_conditions"] == 3 and man["genes_in_vocab"] == 21
    c = KidneyCache(str(tmp_path / "pc"))
    assert c.n_cells == 30 and len(c.val) == 3
    meta = json.load(open(tmp_path / "pc" / "groups.json"))
    names = np.array(meta["conditions"])[c.groups]
    assert set(names) == {"ctrl", "A+ctrl", "B+ctrl"}
    # every cached cell is exactly one training cell: compare its nonzero gene set
    src_sets = {frozenset(np.flatnonzero(ds.X[i] > 0).tolist()) for i in range(30)}
    for g in range(c.n_cells):
        gi, bi = c.cell(g)
        assert frozenset(gi.tolist()) in src_sets and bi.min() >= 1 and bi.max() <= 50
