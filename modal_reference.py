"""GEARS-compatible metrics for saved runs vs published Cell-JEPA numbers (CPU only).

1. Supplements the experiment's cache with ``de_rank20.json`` (GEARS
   ``uns['rank_genes_groups_cov_all'][:20]`` positions) read from the raw h5ad — no other cache
   file is rebuilt or overwritten.
2. Recomputes GEARS mean-based metrics (absolute + delta) for every baseline / completed run from
   its saved ``test_predictions.npz`` (run files are never modified), checks consistency with the
   saved legacy metrics, and writes ``experiments/<exp_id>/reference/``:
   gears_metrics_by_run.csv, gears_summary.csv, reference_vs_published.md, report.json.

    modal run modal_reference.py --exp-id adamson_trackB_v1
    modal volume get jepa-perturb experiments/adamson_trackB_v1/reference ./reference
"""

from __future__ import annotations

import json
import os

import modal

from environment.image import build_image

VOL_PATH = "/vol"
DEFAULT_EXP = "adamson_trackB_v1"
RAW_H5AD = f"{VOL_PATH}/raw/adamson/perturb_processed.h5ad"

app = modal.App("jepa-reference")
vol = modal.Volume.from_name("jepa-perturb")
image = build_image("hopper")


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=4, memory=32768, timeout=1800)
def reference(exp_id: str = DEFAULT_EXP, h5ad_path: str = RAW_H5AD, dataset: str = "",
              force_de: bool = False) -> dict:
    from src.data import load_cache, write_de_rank20
    from src.report import reference_report

    vol.reload()
    edir = f"{VOL_PATH}/experiments/{exp_id}"
    with open(os.path.join(edir, "experiment.json")) as f:
        cdir = json.load(f)["cache_dir"]
    de_info = write_de_rank20(h5ad_path, cdir, force=force_de)
    if de_info["written"]:
        vol.commit()
    ds = load_cache(cdir)
    out = os.path.join(edir, "reference")
    rep = reference_report(ds, edir, out, dataset=dataset or None)
    vol.commit()
    with open(os.path.join(out, "reference_vs_published.md")) as f:
        md = f.read()
    return json.loads(json.dumps({"de_rank20": de_info, "out_dir": out,
                                  "consistency_all_pass": rep["consistency_all_pass"],
                                  "consistency_checks": rep["consistency_checks"],
                                  "summary": rep["summary"], "markdown": md}, default=str))


@app.local_entrypoint()
def main(exp_id: str = DEFAULT_EXP, h5ad_path: str = RAW_H5AD, dataset: str = "", force_de: bool = False):
    r = reference.remote(exp_id, h5ad_path, dataset, force_de)
    print(r["markdown"])
    print(json.dumps({k: r[k] for k in ("de_rank20", "out_dir", "consistency_all_pass")}, indent=2))
    for c in r["consistency_checks"]:
        print(c["path"], "pass" if c["pass"] else "FAIL",
              {k: v["max_abs_diff"] for k, v in c.items() if isinstance(v, dict)})
    if not r["consistency_all_pass"]:
        raise SystemExit("consistency checks failed — see report.json")
