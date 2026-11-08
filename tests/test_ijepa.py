"""CPU tests for the X1 (ijepa) / X2 (progmask) objectives: no identity/value leak, masking, losses."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.track_a import ijepa as IJ  # noqa: E402
from tests.test_pretrain_objectives import AMP, D, TinyModel  # noqa: E402


class _Vocab:
    def __init__(self, names):
        self.stoi = {n: i for i, n in enumerate(names)}

    def __getitem__(self, k):
        return self.stoi[k]


def _batch():
    # row 0: cls + 6 genes (2 masked); row 1: cls + 3 genes (1 masked) + 3 pad
    gids = torch.tensor([[1, 10, 11, 12, 13, 14, 15], [1, 20, 21, 22, 0, 0, 0]])
    vals = torch.tensor([[0, 5, 6, 7, 8, 9, 10], [0, 3, 4, 5, -2, -2, -2]], dtype=torch.float32)
    masked = torch.zeros_like(gids, dtype=torch.bool)
    masked[0, 2] = masked[0, 5] = masked[1, 3] = True
    pad = torch.zeros_like(masked)
    pad[1, 4:] = True
    return {"gene_ids": gids, "values": vals, "masked": masked, "pad": pad}


def test_split_context_targets():
    b = _batch()
    ci, cp, ti, tp = IJ.split_context_targets(b["masked"], b["pad"])
    assert ci.shape[1] == 5 and ti.shape[1] == 2
    assert ci[0].tolist() == [0, 1, 3, 4, 6] and not cp[0].any()
    assert ci[1, :3].tolist() == [0, 1, 2] and cp[1].tolist() == [False, False, False, True, True]
    assert sorted(ti[0].tolist()) == [2, 5] and ti[1, 0].item() == 3 and tp[1].tolist() == [False, True]


def _models():
    torch.manual_seed(0)
    s, t = TinyModel(), TinyModel()
    t.load_state_dict(s.state_dict())
    p = IJ.GenePredictor(D, d_pred=16, nlayers=2, nhead=2, dropout=0.0)
    return s, t, p


def test_no_leak_of_masked_values_or_ids_into_student():
    """Changing a masked gene's value must not change the prediction for any target (student never
    sees it); the teacher target does change, so the loss does."""
    s, t, p = _models()
    s.eval(); p.eval()
    b = _batch()
    ci, cp, ti, tp = IJ.split_context_targets(b["masked"], b["pad"])
    g = lambda x, i: torch.gather(x, 1, i)  # noqa: E731

    def pred(batch):
        h = s.transformer_encoder(s.encoder(g(batch["gene_ids"], ci)) + s.value_encoder(g(batch["values"], ci)),
                                  src_key_padding_mask=cp)
        return p(h, cp, s.encoder(g(batch["gene_ids"], ti)), tp)

    z0 = pred(b)
    b2 = {k: v.clone() for k, v in b.items()}
    b2["values"][0, 2] = 49.0
    assert torch.allclose(z0, pred(b2))
    out0 = IJ.forward_losses_ijepa(s, t, p, b, AMP, 1000.0)
    out2 = IJ.forward_losses_ijepa(s, t, p, b2, AMP, 1000.0)
    assert not torch.isclose(out0["recon"], out2["recon"])


def test_losses_finite_and_grad_reaches_gene_embedding():
    s, t, p = _models()
    out = IJ.forward_losses_ijepa(s, t, p, _batch(), AMP, 1000.0)
    for k in ("loss", "recon", "jepa", "jepa_const_ref", "teacher_tok_std"):
        assert torch.isfinite(out[k]), k
    out["loss"].backward()
    # masked gene 11's embedding row only enters via the predictor query -> must get gradient
    assert s.encoder.weight.grad[11].abs().sum() > 0
    assert all(q.grad is None or torch.isfinite(q.grad).all() for q in p.parameters())


def test_program_masker():
    names = ["<pad>", "<cls>"] + [f"G{i}" for i in range(40)]
    vocab = _Vocab(names)
    progs = IJ.GenePrograms(vocab, {"P_A": [f"G{i}" for i in range(0, 5)], "P_B": [f"G{i}" for i in range(5, 10)],
                                    "P_C": ["G10", "NOT_IN_VOCAB"]})
    assert progs.info()["n_programs"] == 3
    gi = np.array([vocab[f"G{i}"] for i in range(20)])
    rng = np.random.default_rng(0)
    idx, from_prog = IJ.program_mask_row(gi, 5, progs, rng)
    assert len(idx) >= 5 and from_prog >= 5 and len(set(idx.tolist())) == len(idx)
    chosen = {int(gi[j]) for j in idx}
    a, bb = set(progs.members[0].tolist()), set(progs.members[1].tolist())
    assert a <= chosen or bb <= chosen          # one whole program masked
    masker = IJ.make_masker("progmask", progs)
    seqs = [(gi, np.ones(len(gi))), (gi[:1], np.ones(1))]
    m, st = masker(seqs, 21, 0.25, np.random.default_rng(1))
    assert m[0, 0] == False and m[1].sum() == 0 and m[0].sum() >= 5  # noqa: E712
    assert 0 < st["frac_from_programs"] <= 1
    mr, _ = IJ.make_masker("ijepa", None)(seqs, 21, 0.25, np.random.default_rng(1))
    assert mr[0].sum() == 5


def test_hallmark_loads():
    sets = IJ.hallmark_sets()
    assert len(sets) == 50 and "HALLMARK_INTERFERON_GAMMA_RESPONSE" in sets and "STAT1" in sets[
        "HALLMARK_INTERFERON_GAMMA_RESPONSE"]


@pytest.mark.parametrize("variant", ["ijepa", "progmask"])
def test_run_pretraining_end_to_end_cpu(tmp_path, monkeypatch, variant):
    """Full loop (collate -> masker -> train_step -> validate -> export) with a tiny model."""
    import json

    from src.track_a import pretrain as PT

    names = ["<pad>", "<cls>", "<eoc>"] + [f"G{i}" for i in range(197)]
    vpath = tmp_path / "vocab.json"
    vpath.write_text(json.dumps({n: i for i, n in enumerate(names)}))
    PT.make_synthetic_cache(str(tmp_path / "cache"), str(vpath), n_cells=96, n_shards=2, n_universe=150)
    monkeypatch.setattr(PT, "build_student", lambda vocab, *a, seed=0, **k: TinyModel(ntoken=len(vocab)))
    monkeypatch.setattr(PT.GenePredictor, "__init__", _small_pred_init)
    progs = {"P_A": [f"G{i}" for i in range(0, 60)], "P_B": [f"G{i}" for i in range(60, 120)]}
    monkeypatch.setattr(IJ, "hallmark_sets", lambda: progs)
    cfg = PT.PretrainConfig(objective="jepa", jepa_variant=variant, epochs=1, batch_size=16, max_genes=40,
                            mask_ratio=0.25, max_val_cells=16, precision="fp32", d_model=D, log_every=2,
                            loader_threads=1)
    s = PT.run_pretraining(cfg, str(tmp_path / "cache"), str(vpath), str(tmp_path / "run"), str(tmp_path / "exp"),
                           device="cpu")
    assert s["status"] == "completed" and np.isfinite(s["best_val_recon"])
    h = s["history"][0]
    assert np.isfinite(h["val_jepa"]) and "val_jepa_const_ref" in h
    logs = [json.loads(x) for x in open(tmp_path / "run" / "log.jsonl")]
    train = [r for r in logs if r["msg"] == "train"]
    if variant == "progmask":
        assert train and all(r["frac_from_programs"] > 0 for r in train)


_orig_pred_init = IJ.GenePredictor.__init__


def _small_pred_init(self, d_model, d_pred=384, nlayers=4, nhead=8, dropout=0.1):
    _orig_pred_init(self, d_model, d_pred=16, nlayers=1, nhead=2, dropout=0.0)


def test_gene_centring_state():
    s, t, p = _models()
    st = {}
    out = IJ.forward_losses_ijepa(s, t, p, _batch(), AMP, 1000.0, state=st, update_state=True, n_vocab=50)
    assert "gene_center" in st and st["gene_seen"][[11, 14, 22]].tolist() == [1.0, 1.0, 1.0]
    assert st["gene_seen"].sum() == 3 and torch.isfinite(out["jepa"]) and "jepa_const_ref_raw" in out
    c0 = st["gene_center"][11].clone()
    IJ.forward_losses_ijepa(s, t, p, _batch(), AMP, 1000.0, state=st, update_state=False, n_vocab=50)
    assert torch.equal(c0, st["gene_center"][11])
