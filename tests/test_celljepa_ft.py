"""CPU-only checks for the Cell-JEPA fine-tuning recipe (method "celljepa") and the bins50 input
transform. scGPT itself is replaced by a tiny stand-in with the same `_encode` / `decoder` API."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from src import train as T  # noqa: E402
from src.models import bin_expression, ema_update, prepend_cls, scgpt_ecs_loss, state_hash  # noqa: E402
from tests.fakes import fake_dataset  # noqa: E402


# ----------------------------------------------------------------------------- bins50


def _bins_ref(row, n_bins=50):
    out = np.zeros_like(row)
    nz = row != 0
    if nz.any():
        edges = np.quantile(row[nz], np.linspace(0, 1, n_bins + 1)[1:-1])
        out[nz] = np.digitize(row[nz], edges) + 1
    return out


def test_bins50_matches_numpy_reference_and_properties():
    rng = np.random.default_rng(0)
    x = np.log1p(rng.gamma(0.5, 2.0, size=(37, 300)) * (rng.random((37, 300)) < 0.4)).astype(np.float32)
    x[3] = 0.0                      # all-zero cell
    x[4, :5] = [0, 1.0, 1.0, 1.0, 0]  # heavy ties
    x[4, 5:] = 0
    b = bin_expression(torch.as_tensor(x), 50, chunk=8).numpy()
    ref = np.stack([_bins_ref(r.astype(np.float64)) for r in x])
    np.testing.assert_array_equal(b, ref)
    assert ((b == 0) == (x == 0)).all()               # zeros stay 0, nonzeros never 0
    assert b.min() >= 0 and b.max() <= 50
    assert set(np.unique(b[0][x[0] > 0])) <= set(range(1, 51))
    for r, br in zip(x, b):                           # monotone within a cell
        o = np.argsort(r[r > 0], kind="stable")
        assert (np.diff(br[r > 0][o]) >= 0).all()
    full = b[0][x[0] > 0]
    assert full.max() == 50 and full.min() <= 2


def test_bins50_rejects_negative():
    with pytest.raises(ValueError):
        bin_expression(torch.tensor([[1.0, -0.5]]))


# ----------------------------------------------------------------------------- cls / ECS / EMA


def test_prepend_cls_shapes_and_values():
    B, L = 3, 5
    src = torch.arange(B * L).reshape(B, L) + 10
    vals = torch.rand(B, L)
    flags = torch.zeros(B, L, dtype=torch.long)
    flags[:, 2] = 1
    mask = torch.zeros(B, L, dtype=torch.bool)
    dec = torch.rand(B, L)
    s, v, f, m, d = prepend_cls(src, vals, flags, mask, cls_id=7, dec_values=dec)
    assert s.shape == (B, L + 1) and v.shape == (B, L + 1) and f.shape == (B, L + 1) and m.shape == (B, L + 1)
    assert (s[:, 0] == 7).all() and (v[:, 0] == 0).all() and (f[:, 0] == 0).all() and (~m[:, 0]).all()
    assert (d[:, 0] == 0).all() and torch.equal(d[:, 1:], dec)
    assert torch.equal(s[:, 1:], src) and torch.equal(v[:, 1:], vals) and torch.equal(f[:, 1:], flags)
    assert s.dtype == src.dtype and f.dtype == flags.dtype and m.dtype == torch.bool
    assert prepend_cls(src, vals, flags, mask, 7)[4] is None


def test_ecs_matches_scgpt_formula():
    e = torch.tensor([[1.0, 0.0], [1.0, 1.0], [-1.0, 0.2]])
    z = e / e.norm(dim=1, keepdim=True)
    cos = (z @ z.T).numpy()
    np.fill_diagonal(cos, 0.0)
    cos = np.maximum(cos, 0.0)
    expected = np.mean(1 - (cos - 0.3) ** 2)
    assert abs(float(scgpt_ecs_loss(e, 0.3)) - expected) < 1e-6


def test_ema_update():
    a, b = nn.Linear(2, 2), nn.Linear(2, 2)
    a0 = [p.detach().clone() for p in a.parameters()]
    ema_update(a, b, 0.9)
    for p, p0, q in zip(a.parameters(), a0, b.parameters()):
        assert torch.allclose(p, 0.9 * p0 + 0.1 * q)


# ----------------------------------------------------------------------------- config


def test_default_config_reproduces_track_b():
    base = T.TrainConfig(method="scgpt", seed=1)
    assert base.deviations == T.TrainConfig.__dataclass_fields__["deviations"].default_factory()
    assert (base.ckpt, base.input_transform) == ("scGPT_human", "log1p")
    cj = T.TrainConfig(method="celljepa", seed=1)
    assert (cj.jepa_ft_weight, cj.ecs_weight, cj.teacher_update, cj.lr, cj.batch_size, cj.epochs) == \
        (1.0, 0.8, "frozen", 1e-4, 64, 15)
    assert all(d in cj.deviations for d in T.CELLJEPA_DEVIATIONS)
    k = T.TrainConfig(method="scgpt", seed=1, ckpt="/vol/trackA/checkpoints/jepa_kidney", input_transform="bins50")
    assert not any("public whole-human" in d for d in k.deviations)
    assert any("binned per cell" in d for d in k.deviations)
    assert T.TrainConfig(method="scgpt", seed=1, ckpt="/vol/checkpoints/scGPT_human/").deviations == base.deviations
    with pytest.raises(AssertionError):
        T.TrainConfig(method="scgpt", seed=1, input_transform="bins51")


# ----------------------------------------------------------------------------- Trainer (fake scGPT)


class _Vocab(dict):
    pass


class FakeScGPT(nn.Module):
    def __init__(self, ntoken, d=16):
        super().__init__()
        self.encoder = nn.Embedding(ntoken, d)
        self.value_encoder = nn.Linear(1, d)
        self.pert_encoder = nn.Embedding(3, d)
        layer = nn.TransformerEncoderLayer(d, 2, 32, 0.0, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, 1)
        self.decoder_lin = nn.Linear(d, 2)

    def _encode(self, src, values, flags, mask):
        x = self.encoder(src) + self.value_encoder(values.unsqueeze(-1)) + self.pert_encoder(flags)
        return self.transformer_encoder(x, src_key_padding_mask=mask)

    def decoder(self, h, values):
        cb = self.decoder_lin(h)
        return {"pred": cb[..., 0] * values + cb[..., 1]}


@pytest.fixture
def fake_env(monkeypatch):
    ds = fake_dataset(n_genes=40, n_perts=12, cells=30, n_ctrl=200)
    vocab = _Vocab({"<pad>": 0, "<cls>": 1, "<eoc>": 2, **{g: i + 3 for i, g in enumerate(ds.gene_names)}})

    def fake_build(ckpt_dir, seed, dropout=0.0):
        torch.manual_seed(seed)
        return FakeScGPT(len(vocab)), vocab, {"embsize": 16}, {"n_loaded": 0}

    monkeypatch.setattr(T, "build_scgpt", fake_build)
    monkeypatch.setattr("src.models.build_scgpt", fake_build)
    return ds


def _trainer(ds, method, **kw):
    cfg = T.TrainConfig(method=method, seed=1, precision="fp32", max_seq_len=24, batch_size=16,
                        eval_batch_size=8, teacher_cells_per_condition=8, teacher_ctrl_cells=16, **kw)
    return T.Trainer(ds, "/nonexistent", cfg, device="cpu", log=lambda *_: None)


def test_celljepa_trainer_step_predict_and_matched_init(fake_env):
    ds = fake_env
    trs = {m: _trainer(ds, m) for m in T.METHODS}
    assert len({t.init_hash for t in trs.values()}) == 1          # matched student init
    tr = trs["celljepa"]
    t0 = state_hash(tr.teacher)
    for s in range(3):
        out = tr.train_step(np.arange(s * 16, (s + 1) * 16))
        assert all(np.isfinite(out[k]) for k in ("loss", "loss_mse", "loss_jepa", "loss_ecs"))
        assert out["loss_jepa"] > 1e-3
    assert state_hash(tr.teacher) == t0                            # frozen teacher untouched
    assert all(p.grad is None for p in tr.teacher.parameters())
    c = ds.split["val"][0]
    p = tr.predict([c], n_controls=10)[c]
    assert p.shape == (ds.n_genes,) and np.isfinite(p).all()
    for m in ("scgpt", "jepa"):
        assert np.isfinite(trs[m].train_step(np.arange(16))["loss"])


def test_celljepa_batch_teacher_sees_perturbed_cell_same_genes(fake_env):
    tr = _trainer(fake_env, "celljepa")
    b = np.arange(16)
    src, vals, flags, mask, target, ex = tr._batch(b)
    assert vals.shape == target.shape == ex["teacher_vals"].shape == (16, 24)
    assert ex["dec_vals"] is None
    torch.testing.assert_close(ex["teacher_vals"], target)         # log1p: teacher input == perturbed cell
    torch.testing.assert_close(vals, tr.X[torch.as_tensor(tr.pair[b])][:, (src[0] - 3)])


def test_predict_without_cls_unchanged_for_scgpt(fake_env):
    ds = fake_env
    tr = _trainer(ds, "scgpt")
    c = ds.split["val"][0]
    ctrl = np.asarray(ds.eval_controls[c][:8])
    vals = tr.X[torch.as_tensor(ctrl)]
    flags = torch.zeros(8, ds.n_genes, dtype=torch.long)
    flags[:, ds.gene_pos(c)] = 1
    tr.model.eval()
    with torch.no_grad():
        h = tr.model._encode(tr.gene_ids.unsqueeze(0).expand(8, -1), vals, flags,
                             torch.zeros(8, ds.n_genes, dtype=torch.bool))
        ref = tr.model.decoder(h, vals)["pred"].double().mean(0).float().numpy()
    np.testing.assert_allclose(tr.predict([c], n_controls=8)[c], ref, rtol=1e-5, atol=1e-6)


def test_ema_teacher_moves_and_bins50_runs(fake_env):
    tr = _trainer(fake_env, "celljepa", teacher_update="ema", input_transform="bins50")
    t0 = state_hash(tr.teacher)
    _, vals, _, _, target, ex = tr._batch(np.arange(16))
    assert float(vals.max()) <= 50 and float(vals.max()) > 10      # encoder input in bin space
    assert ex["dec_vals"] is not None and float(target.max()) < 10  # targets / decoder term in log1p
    out = tr.train_step(np.arange(16))
    assert np.isfinite(out["loss"])
    assert state_hash(tr.teacher) != t0
    c = fake_env.split["val"][0]
    assert np.isfinite(tr.predict([c], n_controls=8)[c]).all()
