import numpy as np
from src.data import Dataset, CTRL


def fake_dataset(n_genes=40, n_perts=12, cells=30, n_ctrl=200, seed=0, low_rank=True):
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(n_genes)]
    perts = genes[:n_perts]
    base = rng.uniform(0.5, 2.0, n_genes)
    U = rng.normal(size=(n_genes, 3))
    blocks, conds = [base + 0.05 * rng.normal(size=(n_ctrl, n_genes))], [CTRL] * n_ctrl
    for i, g in enumerate(perts):
        eff = (U @ U[i]) * 0.1 if low_rank else rng.normal(size=n_genes) * 0.1
        blocks.append(base + eff + 0.05 * rng.normal(size=(cells, n_genes)))
        conds += [f"{g}+ctrl"] * cells
    X = np.vstack(blocks).astype(np.float32)
    c = [f"{g}+ctrl" for g in perts]
    split = {"train": [CTRL] + c[:8], "val": c[8:10], "test": c[10:]}
    ctrl_idx = np.arange(n_ctrl)
    ev = {k: rng.choice(ctrl_idx, 50, replace=False).tolist() for k in split["val"] + split["test"]}
    de = {k: list(range(20)) for k in split["val"] + split["test"]}
    return Dataset(X=X, conditions=np.array(conds), gene_ids=genes, gene_names=genes, split=split,
                   manifest={"split_hash": "fake"}, eval_controls=ev, de_idx=de)
