import json
import math
import os

import numpy as np

from src import baselines as B
from src.data import load_cache
from src.evaluate import GEARS_KEYS, METRIC_KEYS, evaluate_predictions, gears_metrics, pearson, summarize
from src.report import gears_markdown_section, gears_tables, load_entries, reference_report
from tests.fakes import fake_dataset


def _with_rank(ds, offset=5):
    ds.de_rank20 = {c: list(range(offset, offset + 20)) for c in ds.de_idx}
    return ds


def test_gears_metric_math_matches_definition():
    ds = _with_rank(fake_dataset())
    ctrl = ds.control_reference()
    preds, _ = B.linear_gene_embedding(ds, ds.split["test"], ctrl, K=3)
    rows = evaluate_predictions(preds, ds, ds.split["test"], ctrl)
    for r in rows:
        c = r["condition"]
        p, o = np.asarray(preds[c], float), ds.observed_mean(c).astype(float)
        rk, nd = ds.de_rank20[c], ds.de_idx[c]
        assert np.isclose(r["gears_pearson"], np.corrcoef(p, o)[0, 1])
        assert np.isclose(r["gears_pearson_de"], np.corrcoef(p[rk], o[rk])[0, 1])
        assert np.isclose(r["gears_top20_de_nd"], np.corrcoef(p[nd], o[nd])[0, 1])
        assert np.isclose(r["gears_delta_pearson"], np.corrcoef(p - ctrl, o - ctrl)[0, 1])
        assert np.isclose(r["gears_delta_pearson_de"], np.corrcoef((p - ctrl)[rk], (o - ctrl)[rk])[0, 1])
        assert np.isclose(r["gears_delta_top20_de_nd"], np.corrcoef((p - ctrl)[nd], (o - ctrl)[nd])[0, 1])


def test_consistency_with_legacy_metrics():
    ds = _with_rank(fake_dataset())
    ctrl = ds.control_reference()
    preds = B.training_mean(ds, ds.split["test"], ctrl)
    for r in evaluate_predictions(preds, ds, ds.split["test"], ctrl):
        assert abs(r["gears_pearson"] - r["abs_pearson"]) < 1e-12
        assert abs(r["gears_delta_top20_de_nd"] - r["de20_delta_pearson"]) < 1e-12


def test_no_change_delta_undefined_stays_nan_with_coverage():
    ds = _with_rank(fake_dataset())
    ctrl = ds.control_reference()
    rows = evaluate_predictions(B.no_change(ds, ds.split["test"], ctrl), ds, ds.split["test"], ctrl)
    s = summarize(rows)
    for k in ("gears_delta_pearson", "gears_delta_pearson_de", "gears_delta_top20_de_nd"):
        assert math.isnan(s[k]) and s[f"{k}_n_defined"] == 0
    assert not math.isnan(s["gears_pearson"]) and s["gears_pearson_n_defined"] == len(rows)
    assert "gears_note" not in s
    md = "\n".join(gears_markdown_section([{"method": "no_change", "label": "No change", "n_seeds": 1,
                                            **{k: s[k] for k in GEARS_KEYS},
                                            **{f"{k}_n": s[f"{k}_n_defined"] for k in GEARS_KEYS}}], len(rows)))
    assert "undefined (0/" in md


def test_de_rank20_optional():
    ds = fake_dataset()
    assert ds.de_rank20 is None
    ctrl = ds.control_reference()
    rows = evaluate_predictions(B.training_mean(ds, ds.split["test"], ctrl), ds, ds.split["test"], ctrl)
    assert all(math.isnan(r["gears_pearson_de"]) and math.isnan(r["gears_delta_pearson_de"]) for r in rows)
    assert all(not math.isnan(r["gears_top20_de_nd"]) for r in rows)
    s = summarize(rows)
    assert s["gears_pearson_de_n_defined"] == 0 and "gears_note" in s
    m = gears_metrics(np.arange(5.0), np.arange(5.0) ** 2, np.zeros(5), [0, 1, 2], None)
    assert math.isnan(m["gears_pearson_de"]) and np.isclose(m["gears_pearson"], pearson(np.arange(5.0), np.arange(5.0) ** 2))


def test_summarize_without_gears_keys_unchanged():
    rows = [{k: 1.0 for k in METRIC_KEYS}]
    s = summarize(rows)
    assert not any(k in s for k in GEARS_KEYS)


def _write_fake_cache(ds, d, with_rank):
    np.save(os.path.join(d, "X.npy"), ds.X)
    np.save(os.path.join(d, "conditions.npy"), ds.conditions)
    for name, obj in [("genes.json", {"gene_ids": ds.gene_ids, "gene_names": ds.gene_names}),
                      ("split.json", ds.split), ("manifest.json", ds.manifest),
                      ("eval_controls.json", ds.eval_controls), ("de_idx.json", ds.de_idx)]:
        with open(os.path.join(d, name), "w") as f:
            json.dump(obj, f)
    if with_rank:
        with open(os.path.join(d, "de_rank20.json"), "w") as f:
            json.dump({c: list(range(3, 23)) for c in ds.de_idx}, f)


def test_load_cache_de_rank20_optional(tmp_path):
    ds = fake_dataset()
    a, b = tmp_path / "old", tmp_path / "new"
    a.mkdir(), b.mkdir()
    _write_fake_cache(ds, str(a), False)
    _write_fake_cache(ds, str(b), True)
    assert load_cache(str(a)).de_rank20 is None
    assert load_cache(str(b)).de_rank20[ds.split["test"][0]] == list(range(3, 23))


def _write_entry(d, method, preds, rows, extra=None):
    os.makedirs(d)
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump({"method": method, "split_hash": "fake", "status": "completed", **(extra or {})}, f)
    np.savez_compressed(os.path.join(d, "test_predictions.npz"), conditions=np.array(list(preds)),
                        preds=np.stack(list(preds.values())))
    keys = list(rows[0])
    with open(os.path.join(d, "per_intervention.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(repr(r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")


def test_reference_report_end_to_end(tmp_path):
    ds = _with_rank(fake_dataset())
    ctrl = ds.control_reference()
    test = ds.split["test"]
    exp = tmp_path / "exp"
    for name, fn in [("no_change", B.no_change), ("training_mean", B.training_mean)]:
        preds = fn(ds, test, ctrl)
        legacy = [{k: v for k, v in r.items() if k not in GEARS_KEYS} for r in evaluate_predictions(preds, ds, test, ctrl)]
        _write_entry(str(exp / "baselines" / name), name, preds, legacy)
    rep = reference_report(ds, str(exp), str(tmp_path / "out"))
    assert rep["consistency_all_pass"]
    md = (tmp_path / "out" / "reference_vs_published.md").read_text()
    assert "Cell-JEPA (paper)" in md and "0.937" in md and "undefined" in md
    for f in ("gears_metrics_by_run.csv", "gears_summary.csv", "report.json"):
        assert (tmp_path / "out" / f).exists()
    # a tampered saved metric must fail the consistency check
    entries = [(m, s, [dict(r, abs_pearson=r["abs_pearson"] + 0.01) for r in rows], p, meta)
               for m, s, rows, p, meta in load_entries(ds, str(exp))[0]]
    assert not any(c["pass"] for c in gears_tables(ds, entries)[2])


class _Col(list):
    def astype(self, t):
        return [t(x) for x in self]


class _FakeAd:
    def __init__(self, conds, uns):
        self.obs = {"condition": _Col(conds), "condition_name": _Col(f"K562_{c}_1+1" for c in conds)}
        self.uns = uns


def test_de_gene_sets_rank_optional_when_not_strict():
    from src.data import de_gene_sets
    genes = [f"E{i}" for i in range(30)]
    uns = {"top_non_dropout_de_20": {"K562_A+ctrl_1+1": genes[:20]}}
    ad = _FakeAd(["A+ctrl"], uns)
    de_idx, rank = de_gene_sets(ad, ["A+ctrl"], genes, strict_rank=False)
    assert de_idx["A+ctrl"] == list(range(20)) and rank is None
    uns["rank_genes_groups_cov_all"] = {"K562_A+ctrl_1+1": [g.encode() for g in genes[5:]]}
    _, rank = de_gene_sets(ad, ["A+ctrl"], genes)
    assert rank["A+ctrl"] == list(range(5, 25))
