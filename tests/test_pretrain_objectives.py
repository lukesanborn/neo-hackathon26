"""CPU tests for Track A pretraining objectives (masking, attention mask, EMA, losses, data).

Needs torch (CPU is fine); skipped when torch is unavailable. scGPT is not needed: a tiny model
with scGPT's attribute names (encoder / value_encoder / transformer_encoder / decoder) is used.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from src.track_a import pretrain_model as pm  # noqa: E402
from src.track_a.pretrain import (KidneyCache, PretrainConfig, batch_rng, collate,  # noqa: E402
                                  epoch_cells, make_synthetic_cache)

D, H = 16, 2
AMP = dict(device_type="cpu", dtype=torch.bfloat16, enabled=False)


class _ValEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, D)

    def forward(self, v):
        return self.lin(v.unsqueeze(-1))


class _Dec(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(D, 1)

    def forward(self, h):
        return {"pred": self.fc(h).squeeze(-1)}


class TinyModel(nn.Module):
    def __init__(self, dropout=0.0, nlayers=2, ntoken=50):
        super().__init__()
        self.encoder = nn.Embedding(ntoken, D)
        self.value_encoder = _ValEnc()
        layer = nn.TransformerEncoderLayer(D, H, 32, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, nlayers, enable_nested_tensor=False)
        self.decoder = _Dec()


def _batch(B=3, L=8, seed=0):
    rng = np.random.default_rng(seed)
    ns = np.array([L - 1, 4, 0])[:B]
    gids = np.zeros((B, L), dtype=np.int64)
    vals = np.full((B, L), pm.PAD_VALUE, dtype=np.float32)
    pad = np.ones((B, L), dtype=bool)
    gids[:, 0], vals[:, 0], pad[:, 0] = 1, pm.CLS_VALUE, False
    for b, n in enumerate(ns):
        gids[b, 1:1 + n] = rng.integers(3, 50, size=n)
        vals[b, 1:1 + n] = rng.integers(1, 51, size=n)
        pad[b, 1:1 + n] = False
    masked = pm.random_value_mask(ns, L, 0.3, rng)
    t = lambda x: torch.from_numpy(x)  # noqa: E731
    return {"gene_ids": t(gids), "values": t(vals), "masked": t(masked), "pad": t(pad)}, ns


# ----------------------------------------------------------------------------- masking


def test_random_value_mask_counts_and_positions():
    rng = np.random.default_rng(0)
    ns = np.array([600, 100, 6, 1, 0])
    m = pm.random_value_mask(ns, 601, 0.15, rng)
    assert not m[:, 0].any(), "<cls> must never be masked"
    for b, n in enumerate(ns):
        assert m[b].sum() == (max(1, int(0.15 * n)) if n else 0)
        assert not m[b, 1 + n:].any(), "padding must never be masked"
    assert m[0].sum() == 90


def test_masked_values_replaced_gene_ids_kept():
    batch, _ = _batch()
    model = TinyModel()
    seen = {}

    def hook(mod, inp, out):
        seen["v"] = inp[0].clone()
    model.value_encoder.register_forward_hook(hook)
    pm.forward_losses("scgpt", model, None, None, batch, H, AMP)
    v = seen["v"]
    assert torch.all(v[batch["masked"]] == pm.MASK_VALUE)
    assert torch.equal(v[~batch["masked"]], batch["values"][~batch["masked"]])


# ----------------------------------------------------------------------------- attention mask


def test_attn_mask_rule():
    masked = torch.tensor([[False, True, False, True, False, False]])
    pad = torch.tensor([[False, False, False, False, True, True]])
    m = pm.build_attn_mask(masked, pad, nhead=3)
    assert m.shape == (3, 6, 6) and m.dtype == torch.bool
    a = ~m[0]  # allowed
    for i in range(6):
        for j in range(6):
            expect = (i == j) or (not masked[0, j] and not pad[0, j])
            assert bool(a[i, j]) == expect, (i, j)
    assert a[1, 0] and a[1, 2] and a[1, 1] and not a[1, 3]   # masked -> unmasked + self only
    assert not a[0, 1] and not a[2, 3]                       # cls / unmasked never see masked
    assert all(a[i].any() for i in range(6)), "no fully blocked row"


def test_attn_mask_batch_major_head_order():
    masked = torch.tensor([[False, True, False], [False, False, True]])
    pad = torch.zeros_like(masked)
    m = pm.build_attn_mask(masked, pad, nhead=2)
    assert torch.equal(m[0], m[1]) and torch.equal(m[2], m[3]) and not torch.equal(m[1], m[2])


def test_attention_semantics_numerically():
    torch.manual_seed(0)
    model = TinyModel().eval()
    batch, ns = _batch(B=2, L=8)
    masked, pad = batch["masked"], batch["pad"]
    vals = batch["values"].masked_fill(masked, pm.MASK_VALUE)
    am = pm.build_attn_mask(masked, pad, H)
    h1 = pm.encode(model, batch["gene_ids"], vals, am)
    # perturb the GENE ids of masked tokens and of pads: unmasked/cls outputs must not change,
    # and each masked token's output must not depend on the other masked tokens
    g2 = batch["gene_ids"].clone()
    g2[pad] = 7
    h2 = pm.encode(model, g2, vals, am)
    keep = ~pad
    assert torch.allclose(h1[keep], h2[keep], atol=1e-6)
    b, idx = 0, torch.nonzero(masked[0]).squeeze(1)
    assert len(idx) >= 2
    g3 = batch["gene_ids"].clone()
    g3[b, idx[1]] = (g3[b, idx[1]] + 5) % 50
    h3 = pm.encode(model, g3, vals, am)
    assert torch.allclose(h1[b, idx[0]], h3[b, idx[0]], atol=1e-6)   # other masked token invisible
    unm = (~masked[b]) & (~pad[b])
    assert torch.allclose(h1[b, unm], h3[b, unm], atol=1e-6)         # unmasked/cls don't see masked
    assert not torch.allclose(h1[b, idx[1]], h3[b, idx[1]])           # it does see itself


def test_no_nan_when_all_genes_masked_and_with_pads():
    model = TinyModel().eval()
    B, L = 2, 5
    gids = torch.randint(3, 50, (B, L))
    vals = torch.rand(B, L) * 50
    masked = torch.tensor([[False, True, True, True, True], [False, True, False, False, False]])
    pad = torch.tensor([[False] * 5, [False, False, True, True, True]])
    for grad in (True, False):
        with torch.set_grad_enabled(grad):
            h = pm.encode(model, gids, vals.masked_fill(masked, -1), pm.build_attn_mask(masked, pad, H))
        assert torch.isfinite(h).all()


def test_teacher_mask_matches_key_padding_mask():
    """Unmasked sequence: our folded mask == PyTorch's src_key_padding_mask on real tokens."""
    torch.manual_seed(1)
    model = TinyModel().eval()
    batch, _ = _batch(B=3, L=8)
    pad = batch["pad"]
    x = model.encoder(batch["gene_ids"]) + model.value_encoder(batch["values"])
    with torch.no_grad():
        ref = model.transformer_encoder(x, src_key_padding_mask=pad)
        ours = pm.encode(model, batch["gene_ids"], batch["values"],
                         pm.build_attn_mask(torch.zeros_like(pad), pad, H))
    assert torch.allclose(ref[~pad], ours[~pad], atol=1e-5)


def test_eval_fast_path_matches_grad_path():
    torch.manual_seed(2)
    model = TinyModel().eval()
    batch, _ = _batch()
    am = pm.build_attn_mask(batch["masked"], batch["pad"], H)
    vals = batch["values"].masked_fill(batch["masked"], -1)
    with torch.no_grad():
        a = pm.encode(model, batch["gene_ids"], vals, am)
    b = pm.encode(model, batch["gene_ids"], vals, am)
    keep = ~batch["pad"]
    assert torch.allclose(a[keep], b.detach()[keep], atol=1e-5)


# ----------------------------------------------------------------------------- EMA


def test_ema_update_math():
    torch.manual_seed(0)
    s = TinyModel()
    t = pm.make_teacher(s)
    assert all(not p.requires_grad for p in t.parameters()) and not t.training
    t0 = {k: v.clone() for k, v in t.state_dict().items()}
    with torch.no_grad():
        for p in s.parameters():
            p.add_(torch.randn_like(p))
    s_sd = s.state_dict()
    pm.ema_update(t, s, 0.9)
    for k, v in t.state_dict().items():
        assert torch.allclose(v, 0.9 * t0[k] + 0.1 * s_sd[k], atol=1e-6), k
    pm.ema_update(t, s, 0.0)
    for k, v in t.state_dict().items():
        assert torch.allclose(v, s_sd[k])


# ----------------------------------------------------------------------------- losses


def test_loss_weighting_and_gradients():
    torch.manual_seed(0)
    student = TinyModel()
    teacher = pm.make_teacher(student)
    pred = pm.CLSPredictor(D)
    batch, _ = _batch()
    out = pm.forward_losses("jepa", student, teacher, pred, batch, H, AMP)
    assert torch.allclose(out["loss"], 1000 * out["jepa"] + out["recon"])
    assert 0.0 <= float(out["jepa"].detach()) <= 2.0
    out["loss"].backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is not None for p in pred.parameters())
    out2 = pm.forward_losses("scgpt", student, None, None, batch, H, AMP)
    assert "jepa" not in out2 and torch.allclose(out2["loss"], out2["recon"])
    # same masking / attention -> identical reconstruction term for both objectives
    assert torch.allclose(out2["recon"], out["recon"])


def test_masked_mse_only_masked_positions():
    p = torch.tensor([[0.0, 1.0, 5.0]])
    y = torch.tensor([[9.0, 3.0, 5.0]])
    m = torch.tensor([[False, True, True]])
    assert float(pm.masked_mse(p, y, m)) == pytest.approx(2.0)


def test_jepa_loss_values():
    a = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    assert float(pm.jepa_cosine_loss(a, a)) == pytest.approx(0.0, abs=1e-6)
    assert float(pm.jepa_cosine_loss(a, -a)) == pytest.approx(2.0)
    t = a.clone().requires_grad_(True)
    pm.jepa_cosine_loss(a.clone().requires_grad_(True), t).backward()
    assert t.grad is None, "target must be stop-gradient"


# ----------------------------------------------------------------------------- data


@pytest.fixture
def cache_dir(tmp_path):
    vocab = {f"G{i}": i for i in range(200)}
    vp = tmp_path / "vocab.json"
    vp.write_text(json.dumps(vocab))
    out = tmp_path / "cache"
    make_synthetic_cache(str(out), str(vp), n_cells=60, n_shards=3, n_universe=150, seed=0)
    return str(out), str(vp)


def test_cache_contract_and_collate(cache_dir):
    out, vp = cache_dir
    z = np.load(f"{out}/shards/shard_000.npz")
    assert z["indptr"].dtype == np.int64 and z["gene_ids"].dtype == np.int32 and z["bins"].dtype == np.uint8
    assert z["bins"].min() >= 1 and z["bins"].max() <= 50
    c = KidneyCache(out)
    assert c.n_cells == 60 and len(c.train) + len(c.val) == 60
    assert not set(c.train) & set(c.val)
    cfg = PretrainConfig(objective="jepa", batch_size=8)
    cells = epoch_cells(c, cfg, 0)[:8]
    a = collate(c, cells, 600, 0.15, batch_rng(0, 0, 0), cls_id=201, pad_id=200)
    b = collate(c, cells, 600, 0.15, batch_rng(0, 0, 0), cls_id=201, pad_id=200)
    for k in a:
        assert np.array_equal(a[k], b[k]), "batches must be identical across objectives"
    L = a["gene_ids"].shape[1]
    assert L <= 601 and (a["gene_ids"][:, 0] == 201).all() and (a["values"][:, 0] == 0).all()
    assert (a["n_genes"] <= 600).all()
    assert ((a["values"] == pm.PAD_VALUE) == a["pad"]).all()
    assert not (a["masked"] & a["pad"]).any()
    gene_slots = ~a["pad"][:, 1:]
    assert a["values"][:, 1:][gene_slots].min() >= 1 and a["values"][:, 1:][gene_slots].max() <= 50
    # sampled genes belong to the cell
    g0, _ = c.cell(int(cells[0]))
    assert set(a["gene_ids"][0, 1:1 + a["n_genes"][0]]) <= set(g0.tolist())


def test_epoch_order_depends_only_on_seed(cache_dir):
    c = KidneyCache(cache_dir[0])
    a = epoch_cells(c, PretrainConfig(objective="jepa"), 3)
    b = epoch_cells(c, PretrainConfig(objective="scgpt"), 3)
    assert np.array_equal(a, b) and not np.array_equal(a, epoch_cells(c, PretrainConfig(objective="jepa"), 4))


def _tiny_student(vocab, *a, seed=0, **k):
    torch.manual_seed(seed)
    return TinyModel(ntoken=len(vocab))


def test_run_resume_and_export_cpu(cache_dir, tmp_path, monkeypatch):
    """Tiny end-to-end loop on CPU: train 1 epoch, extend to 2 (resume), export the contract."""
    import os

    import src.track_a.pretrain as P

    monkeypatch.setattr(P, "build_student", _tiny_student)
    out, vp = cache_dir
    run, exp = str(tmp_path / "run"), str(tmp_path / "exp")
    base = dict(objective="jepa", d_model=D, nhead=H, batch_size=8, log_every=2, max_val_cells=16)
    s1 = P.run_pretraining(PretrainConfig(epochs=1, **base), out, vp, run, exp, device="cpu")
    assert s1["status"] == "completed" and s1["epochs_completed"] == 1
    s2 = P.run_pretraining(PretrainConfig(epochs=2, **base), out, vp, run, exp, device="cpu")
    assert s2["epochs_completed"] == 2 and len(s2["history"]) == 2
    assert s1["init_state_hash"] == s2["init_state_hash"]
    with pytest.raises(RuntimeError, match="resume config mismatch"):
        P.run_pretraining(PretrainConfig(epochs=3, **{**base, "lr": 3e-4}), out, vp, run, exp, device="cpu")
    assert sorted(os.listdir(exp)) == ["args.json", "best_model.pt", "last.pt", "vocab.json"]
    with open(os.path.join(exp, "args.json")) as f:
        args = json.load(f)
    for k in ("embsize", "nheads", "d_hid", "nlayers", "n_layers_cls", "dropout", "input_style", "objective",
              "epochs_completed", "deviations"):
        assert k in args
    assert args["input_style"] == "bins50" and args["epochs_completed"] == 2
    sd = torch.load(os.path.join(exp, "best_model.pt"))
    assert any(k.startswith("transformer_encoder.") for k in sd)
    assert all(torch.isfinite(v).all() for v in sd.values())


def test_resumed_run_equals_uninterrupted(cache_dir, tmp_path, monkeypatch):
    import src.track_a.pretrain as P

    monkeypatch.setattr(P, "build_student", _tiny_student)
    out, vp = cache_dir
    base = dict(objective="jepa", d_model=D, nhead=H, batch_size=8, log_every=2, max_val_cells=16, dropout=0.2)
    a = P.run_pretraining(PretrainConfig(epochs=2, **base), out, vp, str(tmp_path / "a"), str(tmp_path / "ea"),
                          device="cpu")
    P.run_pretraining(PretrainConfig(epochs=1, **base), out, vp, str(tmp_path / "b"), str(tmp_path / "eb"),
                      device="cpu")
    b = P.run_pretraining(PretrainConfig(epochs=2, **base), out, vp, str(tmp_path / "b"), str(tmp_path / "eb"),
                          device="cpu")
    assert a["history"][-1]["val_recon"] == pytest.approx(b["history"][-1]["val_recon"], rel=1e-5)


# ----------------------------------------------------------------------------- init from checkpoint


def _tiny_ckpt(tmp_path, vp, seed=123, args=None):
    """A checkpoint dir in the public scGPT_human layout: flash-attn ``Wqkv`` names, an extra
    head the student does not have (like scGPT_human's mvc_decoder)."""
    import os

    d = tmp_path / f"ckpt_{seed}"
    os.makedirs(d, exist_ok=True)
    vocab = json.load(open(vp))
    torch.manual_seed(seed)
    src = TinyModel(ntoken=len(vocab) + 3)  # + <pad>/<cls>/<eoc> added by SimpleVocab
    sd = {k.replace("in_proj_", "Wqkv."): v for k, v in src.state_dict().items()}
    sd["mvc_decoder.w.weight"] = torch.zeros(3, 3)
    torch.save(sd, d / "best_model.pt")
    (d / "vocab.json").write_text(json.dumps(vocab))
    a = {"embsize": D, "nheads": H, "d_hid": 512, "nlayers": 12, "n_layers_cls": 3, "pad_value": -2, "n_bins": 51}
    (d / "args.json").write_text(json.dumps({**a, **(args or {})}))
    return str(d), src


def test_init_ckpt_loads_weights_and_matches_objectives(cache_dir, tmp_path, monkeypatch):
    import src.track_a.pretrain as P
    from src.models import SimpleVocab, state_hash

    monkeypatch.setattr(P, "build_student", _tiny_student)
    _, vp = cache_dir
    ck, src = _tiny_ckpt(tmp_path, vp)
    vocab = SimpleVocab(vp)
    trs = {o: P.Pretrainer(PretrainConfig(objective=o, d_model=D, nhead=H, init_ckpt=ck), vocab, "cpu")
           for o in ("jepa", "scgpt")}
    for o, tr in trs.items():
        for k, v in src.state_dict().items():
            assert torch.equal(tr.model.state_dict()[k], v), (o, k)
        assert tr.init_info["n_loaded"] == len(src.state_dict()) and tr.init_info["not_loaded"] == []
        assert tr.init_info["loaded_prefixes"] == ["decoder", "encoder", "transformer_encoder", "value_encoder"]
        assert tr.init_info["n_ckpt_keys_unused"] == 1 and tr.init_info["ckpt_unused_prefixes"] == ["mvc_decoder"]
    assert trs["jepa"].init_hash == trs["scgpt"].init_hash == state_hash(src)
    # teacher = copy of the loaded student; predictor freshly initialised (same as random-init path)
    assert state_hash(trs["jepa"].teacher) == state_hash(src)
    rnd = P.Pretrainer(PretrainConfig(objective="jepa", d_model=D, nhead=H), vocab, "cpu")
    assert rnd.init_hash != trs["jepa"].init_hash
    assert state_hash(rnd.predictor) == state_hash(trs["jepa"].predictor)
    cfg = trs["jepa"].cfg
    assert pm.RANDOM_INIT_DEVIATION not in cfg.deviations
    assert any("<cls> input value = 0" in d for d in cfg.deviations)


def test_random_init_path_unchanged(cache_dir, monkeypatch):
    import src.track_a.pretrain as P
    from src.models import SimpleVocab, state_hash

    monkeypatch.setattr(P, "build_student", _tiny_student)
    cfg = PretrainConfig(objective="scgpt", d_model=D, nhead=H)
    assert cfg.init_ckpt is None and cfg.deviations == list(pm.DEVIATIONS)
    assert pm.RANDOM_INIT_DEVIATION in cfg.deviations
    tr = P.Pretrainer(cfg, SimpleVocab(cache_dir[1]), "cpu")
    assert tr.init_info == {"init_ckpt": None}
    assert tr.init_hash == state_hash(_tiny_student(SimpleVocab(cache_dir[1]), seed=0))


def test_init_ckpt_in_matched_key_and_resolution():
    a = PretrainConfig(objective="jepa").matched_key()
    b = PretrainConfig(objective="jepa", init_ckpt="scGPT_human")
    assert "init_ckpt" in a and a["init_ckpt"] is None and a != b.matched_key()
    assert P_resolve("scGPT_human") == "/vol/checkpoints/scGPT_human"
    assert b.matched_key() == PretrainConfig(objective="jepa", init_ckpt="/vol/checkpoints/scGPT_human/").matched_key()
    assert P_resolve("/x/y/") == "/x/y" and P_resolve(None) is None
    with pytest.raises(ValueError):
        PretrainConfig(objective="jepa", init_ckpt="relative/dir")


def P_resolve(x):
    from src.track_a.pretrain import resolve_init_ckpt
    return resolve_init_ckpt(x)


@pytest.mark.parametrize("bad", [{"embsize": 32}, {"nlayers": 6}, {"n_layers_cls": 2}, {"pad_value": 0},
                                 {"input_emb_style": "category"}])
def test_init_ckpt_architecture_mismatch_fails(cache_dir, tmp_path, monkeypatch, bad):
    import src.track_a.pretrain as P
    from src.models import SimpleVocab

    monkeypatch.setattr(P, "build_student", _tiny_student)
    ck, _ = _tiny_ckpt(tmp_path, cache_dir[1], args=bad)
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        P.Pretrainer(PretrainConfig(objective="scgpt", d_model=D, nhead=H, init_ckpt=ck),
                     SimpleVocab(cache_dir[1]), "cpu")


def test_init_ckpt_missing_encoder_weights_fails(cache_dir, tmp_path, monkeypatch):
    import src.track_a.pretrain as P
    from src.models import SimpleVocab

    monkeypatch.setattr(P, "build_student", _tiny_student)
    ck, _ = _tiny_ckpt(tmp_path, cache_dir[1])
    sd = torch.load(f"{ck}/best_model.pt")
    sd.pop("value_encoder.lin.weight")
    sd["transformer_encoder.layers.0.linear1.weight"] = torch.zeros(3, 3)  # wrong shape
    torch.save(sd, f"{ck}/best_model.pt")
    with pytest.raises(RuntimeError, match="encoder weights not loaded") as e:
        P.Pretrainer(PretrainConfig(objective="jepa", d_model=D, nhead=H, init_ckpt=ck),
                     SimpleVocab(cache_dir[1]), "cpu")
    assert "value_encoder.lin.weight" in str(e.value) and "linear1.weight" in str(e.value)


def test_init_ckpt_vocab_mismatch_fails(cache_dir, tmp_path, monkeypatch):
    import src.track_a.pretrain as P
    from src.models import SimpleVocab

    monkeypatch.setattr(P, "build_student", _tiny_student)
    ck, _ = _tiny_ckpt(tmp_path, cache_dir[1])
    v = json.load(open(f"{ck}/vocab.json"))
    v["G0"], v["G1"] = v["G1"], v["G0"]
    open(f"{ck}/vocab.json", "w").write(json.dumps(v))
    with pytest.raises(RuntimeError, match="vocab"):
        P.Pretrainer(PretrainConfig(objective="jepa", d_model=D, nhead=H, init_ckpt=ck),
                     SimpleVocab(cache_dir[1]), "cpu")


def test_init_ckpt_run_export_and_resume_guard(cache_dir, tmp_path, monkeypatch):
    import src.track_a.pretrain as P

    monkeypatch.setattr(P, "build_student", _tiny_student)
    out, vp = cache_dir
    ck, src = _tiny_ckpt(tmp_path, vp)
    run, exp = str(tmp_path / "run"), str(tmp_path / "exp")
    base = dict(objective="jepa", d_model=D, nhead=H, batch_size=8, log_every=2, max_val_cells=16, epochs=1)
    s = P.run_pretraining(PretrainConfig(init_ckpt=ck, **base), out, vp, run, exp, device="cpu")
    with open(f"{exp}/args.json") as f:
        args = json.load(f)
    from src.models import state_hash
    assert args["init_ckpt"] == ck and args["init_ckpt_dir"] == ck and args["init_n_loaded"] == len(src.state_dict())
    assert args["init_state_hash"] == s["init_state_hash"] == state_hash(src)
    assert "transformer_encoder" in args["init_loaded_prefixes"] and args["cls_value"] == 0
    assert any("<cls> input value = 0" in d for d in args["deviations"])
    with open(f"{run}/config.json") as f:
        assert json.load(f)["init_ckpt_info"]["n_loaded"] == len(src.state_dict())
    # resume must refuse a different init (random init, or another checkpoint) and another cache
    with pytest.raises(RuntimeError, match="resume config mismatch"):
        P.run_pretraining(PretrainConfig(**{**base, "epochs": 2}), out, vp, run, exp, device="cpu")
    ck2, _ = _tiny_ckpt(tmp_path, vp, seed=7)
    with pytest.raises(RuntimeError, match="resume config mismatch"):
        P.run_pretraining(PretrainConfig(init_ckpt=ck2, **{**base, "epochs": 2}), out, vp, run, exp, device="cpu")
    import shutil
    other = str(tmp_path / "cache_copy")
    shutil.copytree(out, other)
    with pytest.raises(RuntimeError, match="cache_dir"):
        P.run_pretraining(PretrainConfig(init_ckpt=ck, **{**base, "epochs": 2}), other, vp, run, exp, device="cpu")
    # a legacy last.pt without cache_dir falls back to the run's config.json
    lp = f"{run}/last.pt"
    st = torch.load(lp, weights_only=False)
    st.pop("cache_dir")
    torch.save(st, lp)
    with pytest.raises(RuntimeError, match="cache_dir"):
        P.run_pretraining(PretrainConfig(init_ckpt=ck, **{**base, "epochs": 2}), other, vp, run, exp, device="cpu")
    s2 = P.run_pretraining(PretrainConfig(init_ckpt=ck, **{**base, "epochs": 2}), out + "/", vp, run, exp,
                           device="cpu")
    assert s2["epochs_completed"] == 2


# ----------------------------------------------------------------------------- anti-collapse variants
def test_jepa_variants_run_and_center_state(tmp_path=None):
    torch = pytest.importorskip("torch")
    from src.track_a.pretrain_model import JEPA_VARIANTS, variance_hinge

    assert set(JEPA_VARIANTS) == {"faithful", "center", "vicreg", "token"}
    # variance hinge: constant embeddings -> ~gamma; well-spread -> ~0
    assert float(variance_hinge(torch.zeros(16, 8))) > 0.9
    assert float(variance_hinge(torch.randn(256, 8) * 5)) < 0.05


@pytest.mark.parametrize("variant", ["faithful", "center", "vicreg", "token"])
def test_jepa_variant_forward_backward(variant):
    torch.manual_seed(0)
    student = TinyModel()
    teacher = pm.make_teacher(student)
    pred = pm.CLSPredictor(D)
    batch, _ = _batch()
    state = {}
    out = pm.forward_losses("jepa", student, teacher, pred, batch, H, AMP, variant, state, update_state=True)
    assert torch.isfinite(out["loss"]) and torch.allclose(out["loss"], 1000 * out["jepa"] + out["recon"])
    out["loss"].backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert any(p.grad is not None for p in pred.parameters())
    if variant == "center":
        assert "center" in state and state["center"].shape == (D,) and "target_spread" in out
    else:
        assert state == {}
    if variant == "vicreg":
        assert "var_loss" in out
    if variant == "token":
        assert "teacher_tok_std" in out
    # faithful default reproduces the original call exactly
    if variant == "faithful":
        torch.manual_seed(0)
        s2 = TinyModel(); t2 = pm.make_teacher(s2); p2 = pm.CLSPredictor(D)
        o2 = pm.forward_losses("jepa", s2, t2, p2, batch, H, AMP)
        assert torch.allclose(o2["loss"], out["loss"].detach())
