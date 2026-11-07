"""Task 1 fine-tune loop on CPU with a tiny scGPT-shaped model and synthetic PBMC-like data."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.track_a import pbmc_finetune as F  # noqa: E402
from tests.test_pretrain_objectives import D, H, TinyModel  # noqa: E402


def _fake_ckpt(tmp_path, n_genes=40):
    genes = [f"G{i}" for i in range(n_genes)]
    vocab = {g: i + 3 for i, g in enumerate(genes)}
    vocab.update({"<pad>": 0, "<cls>": 1, "<eoc>": 2})
    d = tmp_path / "ckpt"
    d.mkdir()
    json.dump(vocab, open(d / "vocab.json", "w"))
    json.dump({"embsize": D, "nheads": H, "d_hid": 32, "nlayers": 2, "pad_value": -2.0, "cls_value": 0.0,
               "objective": "jepa"}, open(d / "args.json", "w"))
    torch.manual_seed(0)
    torch.save(TinyModel(ntoken=len(vocab)).state_dict(), d / "best_model.pt")
    return str(d), genes


def _fake_data(genes, n_cells=120, seed=0):
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    labels = np.array(["A", "B", "C"])[rng.integers(0, 3, n_cells)]
    lam = np.where(labels[:, None] == "A", 3.0, 0.5) * np.ones((1, len(genes)))
    X = sp.csr_matrix(rng.poisson(lam).astype(np.float32))
    return {"indptr": X.indptr, "indices": X.indices, "data": X.data, "gene_symbols": np.array(genes), "labels": labels}


def _builder(vocab, d_model, nhead, d_hid, nlayers, nlayers_cls=3, dropout=0.0, seed=0):
    return TinyModel(dropout=dropout, nlayers=nlayers, ntoken=len(vocab))


def test_split_is_deterministic_and_disjoint():
    a, b = F.split_cells(100, 0.1, 42)
    a2, b2 = F.split_cells(100, 0.1, 42)
    assert len(b) == 10 and not set(a) & set(b) and np.array_equal(a, a2) and np.array_equal(b, b2)


@pytest.mark.parametrize("objective", ["jepa", "scgpt"])
def test_finetune_runs_and_records_losses(tmp_path, monkeypatch, objective):
    ckpt, genes = _fake_ckpt(tmp_path)
    data = _fake_data(genes)
    monkeypatch.setattr(F, "cluster_metrics", lambda emb, lab: {"AvgBIO": 0.5, "n": len(emb)})
    cfg = F.FinetuneConfig(objective=objective, epochs=2, batch_size=16, max_genes=30)
    res = F.run_finetune(cfg, ckpt, data, str(tmp_path / "out"), device="cpu", builder=_builder, log=lambda m: None)
    assert res["n"] == 120 and res["model"]["n_loaded"] > 0
    h = res["history"]
    assert len(h) == 2 and all(np.isfinite(v) for row in h for k, v in row.items() if k != "epoch")
    assert ("train_jepa" in h[0]) == (objective == "jepa")
    for k in ("train_gep", "train_gepc", "train_ecs", "val_gep"):
        assert k in h[0]
    assert (tmp_path / "out" / "result.json").exists() and np.load(tmp_path / "out" / "embedding.npy").shape == (120, D)
