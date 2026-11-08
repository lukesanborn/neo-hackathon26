"""Views knobs 2a / 2b: mask units (numpy only; the collate integration test needs torch)."""

import json

import numpy as np
import pytest

from src.track_a.masking import (build_programs, gene_sets_to_units, load_units, make_masker, parse_gene_sets,
                                 top_expressed_mask, unit_value_mask)


def _batch(rows, L=None, pad_id=0, cls_id=1):
    """rows: list of (gene_ids, values) -> padded (B, L) arrays with <cls> at 0, plus n_genes."""
    ns = np.array([len(g) for g, _ in rows])
    L = L or 1 + int(ns.max())
    gids = np.full((len(rows), L), pad_id)
    vals = np.full((len(rows), L), -2.0, dtype=np.float32)
    gids[:, 0], vals[:, 0] = cls_id, 0.0
    for b, (g, v) in enumerate(rows):
        gids[b, 1:1 + len(g)] = g
        vals[b, 1:1 + len(g)] = v
    return gids, vals, ns


def _units(units):
    gu = {}
    for i, u in enumerate(units):
        for g in u:
            gu.setdefault(g, []).append(i)
    return {g: tuple(v) for g, v in gu.items()}


def test_unit_mask_masks_whole_units_and_reaches_target():
    units = [list(range(100 + 10 * j, 110 + 10 * j)) for j in range(10)]   # 10 units x 10 genes
    genes = np.arange(100, 200)
    gids, vals, ns = _batch([(genes, np.ones(100))] * 4)
    m, fb = unit_value_mask(gids, vals, ns, 0.25, np.random.default_rng(0), _units(units))
    assert fb == 0
    assert not m[:, 0].any()                                   # <cls> never masked
    for b in range(4):
        masked_genes = set(gids[b][m[b]].tolist())
        assert len(masked_genes) == 30                         # target 25 -> 3 whole units of 10
        for u in units:
            hit = masked_genes & set(u)
            assert not hit or hit == set(u)                    # all or nothing per unit
    assert not np.array_equal(m[0], np.zeros_like(m[0]))


def test_unit_mask_respects_padding_keeps_one_visible_and_falls_back():
    units = [[10, 11, 12]]
    gids, vals, ns = _batch([(np.array([10, 11, 12]), np.ones(3)), (np.array([50, 51, 52, 53]), np.ones(4))], L=8)
    m, fb = unit_value_mask(gids, vals, ns, 0.9, np.random.default_rng(1), _units(units))
    assert not m[:, 1 + ns.max():].any()                       # pads never masked
    assert m[0].sum() == 2                                     # whole unit would hide all 3; cap n-1
    assert fb == 1 and m[1].sum() >= 1                         # no unit genes -> random fallback


def test_unit_mask_deterministic_given_rng():
    units = [[g] for g in range(100, 160)]
    gids, vals, ns = _batch([(np.arange(100, 160), np.ones(60))] * 3)
    a, _ = unit_value_mask(gids, vals, ns, 0.3, np.random.default_rng(5), _units(units))
    b, _ = unit_value_mask(gids, vals, ns, 0.3, np.random.default_rng(5), _units(units))
    assert np.array_equal(a, b)


def test_top_expressed_masks_largest_values():
    v = np.array([1, 50, 3, 40, 2, 30, 4, 20, 5, 10], dtype=np.float32)
    gids, vals, ns = _batch([(np.arange(10), v)])
    m, _ = top_expressed_mask(gids, vals, ns, 0.3, np.random.default_rng(0))
    assert sorted(vals[0][m[0]].tolist()) == [30.0, 40.0, 50.0]


def test_parse_gene_sets_gmt_and_enrichr():
    text = "HALLMARK_A\tdesc\tTP53\tmdm2\tCDKN1A\nENRICHR_B\t\tGAPDH,1.0\tACTB,0.5\n\nbad_line\n"
    s = parse_gene_sets(text)
    assert s == {"HALLMARK_A": ["CDKN1A", "MDM2", "TP53"], "ENRICHR_B": ["ACTB", "GAPDH"]}
    units, names, stats = gene_sets_to_units(s, {"TP53": 7, "MDM2": 8, "CDKN1A": 9, "GAPDH": 3}, min_size=2)
    assert names == ["HALLMARK_A"] and units == [[7, 8, 9]] and stats["n_symbols_not_in_vocab"] == 1


def test_build_programs_recovers_planted_modules():
    rng = np.random.default_rng(0)
    n_cells, mods = 3000, [np.arange(100, 130), np.arange(200, 230), np.arange(300, 330)]
    cells = []
    for _ in range(n_cells):
        act = rng.random(3) < 0.5                              # each module on/off together
        g = np.concatenate([m for m, a in zip(mods, act) if a] + [rng.choice(np.arange(400, 600), 20, replace=False)])
        b = np.where(g < 400, 40 + rng.integers(0, 5, len(g)), rng.integers(1, 10, len(g)))
        cells.append((g, b.astype(np.float32)))
    d = build_programs(cells, n_programs=8, rank=10, min_cells=20, seed=0)
    unit_of = {g: i for i, u in enumerate(d["units"]) for g in u}
    for m in mods:                                             # each planted module lands in one program
        labs = [unit_of[g] for g in m]
        assert max(np.bincount(labs)) >= 0.9 * len(m)
    assert d["kind"] == "program" and d["stats"]["token_coverage"] > 0.9


def test_make_masker_and_units_file(tmp_path):
    p = tmp_path / "u.json"
    p.write_text(json.dumps({"kind": "program", "units": [[1, 2], [3], []], "stats": {}}))
    u = load_units(str(p))
    assert u["n_units"] == 2 and u["gene_units"][1] == (0,)
    assert make_masker("random")[0] is None
    with pytest.raises(ValueError):
        make_masker("pathway", str(p))                         # kind mismatch
    with pytest.raises(ValueError):
        make_masker("program")                                 # path required
    fn, info = make_masker("program", str(p))
    assert info["n_units"] == 2 and callable(fn)


def test_collate_random_path_unchanged_and_program_path(tmp_path):
    pytest.importorskip("torch")
    from src.track_a.pretrain import KidneyCache, batch_rng, collate, make_synthetic_cache

    # minimal vocab for the synthetic cache
    vocab = {f"G{i}": i for i in range(6000)}
    vocab.update({"<pad>": 6000, "<cls>": 6001, "<eoc>": 6002})
    vp = tmp_path / "vocab.json"
    vp.write_text(json.dumps(vocab))
    make_synthetic_cache(str(tmp_path / "c"), str(vp), n_cells=200, n_shards=2)
    c = KidneyCache(str(tmp_path / "c"))
    cells = c.train[:16]
    a = collate(c, cells, 600, 0.15, batch_rng(0, 0, 0), cls_id=6001, pad_id=6000)
    b = collate(c, cells, 600, 0.15, batch_rng(0, 0, 0), cls_id=6001, pad_id=6000, masker=None)
    assert np.array_equal(a["masked"], b["masked"]) and a["mask_fallback"] == 0

    progs = build_programs([c.cell(int(g)) for g in c.train[:150]], n_programs=16, rank=8, min_cells=5)
    up = tmp_path / "prog.json"
    up.write_text(json.dumps(progs))
    fn, _ = make_masker("program", str(up))
    p = collate(c, cells, 600, 0.3, batch_rng(0, 0, 0), cls_id=6001, pad_id=6000, masker=fn)
    frac = p["masked"].sum() / p["n_genes"].sum()
    assert 0.25 <= frac <= 0.6 and not p["masked"][p["pad"]].any() and not p["masked"][:, 0].any()
