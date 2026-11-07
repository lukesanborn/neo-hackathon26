"""Aggregation: verify compatible manifests, combine metrics, export tables, plots and demo.

Every number shown in the report / demo is read from saved evaluation outputs.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os

import numpy as np

from .evaluate import METRIC_KEYS, bootstrap_mean_ci, paired_bootstrap_diff

METHOD_ORDER = ["jepa", "scgpt", "linear", "training_mean", "no_change"]
METHOD_LABEL = {"jepa": "scGPT + JEPA aux (ours)", "scgpt": "scGPT (matched)", "linear": "Linear (gene emb.)",
                "training_mean": "Training mean", "no_change": "No change"}
# fixed entity -> categorical slot (dataviz reference palette, light / dark); never by rank
METHOD_COLOR = {"jepa": ("#2a78d6", "#3987e5"), "scgpt": ("#eb6834", "#d95926"),
                "linear": ("#1baf7a", "#199e70"), "training_mean": ("#eda100", "#c98500"),
                "no_change": ("#8a8984", "#8a8984")}


def _read_json(p):
    with open(p) as f:
        return json.load(f)


def _read_rows(p):
    with open(p) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k != "condition":
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def collect_runs(exp_dir: str, split_hash: str):
    """Latest completed run per (method, seed); every run's status is reported."""
    status, chosen = [], {}
    for d in sorted(glob.glob(os.path.join(exp_dir, "runs", "*"))):
        st = _read_json(os.path.join(d, "status.json")) if os.path.exists(os.path.join(d, "status.json")) else {"status": "submitted"}
        cfg = _read_json(os.path.join(d, "config.json")) if os.path.exists(os.path.join(d, "config.json")) else {}
        mpath = os.path.join(d, "metrics.json")
        m = _read_json(mpath) if os.path.exists(mpath) else None
        state = st.get("status", "?")
        if state == "running" and m is None:
            state = "running_or_died"  # a container that hit its timeout leaves this behind
        row = {"run": os.path.basename(d), "method": cfg.get("method"), "seed": cfg.get("seed"), "status": state}
        if m:
            row.update(epochs=m["epochs_completed"], minutes=round(m["elapsed_seconds"] / 60, 1),
                       gpu=m["environment"].get("gpu"), reload_ok=m["reload_ok"])
            if m["split_hash"] != split_hash:
                row["status"] = "excluded_split_mismatch"
            elif m["status"] in ("completed", "stopped_by_time_budget"):
                key = (m["method"], m["seed"])
                if key not in chosen or os.path.getmtime(mpath) > os.path.getmtime(os.path.join(chosen[key], "metrics.json")):
                    chosen[key] = d
        status.append(row)
    return status, chosen


def aggregate(ds, exp_dir: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    ctrl = ds.control_reference()
    test = ds.split["test"]
    split_hash = ds.manifest["split_hash"]

    # ---------------------------------------------------------------- load everything
    entries = []  # (method, seed, per-intervention rows, preds dict, meta)
    for bdir in sorted(glob.glob(os.path.join(exp_dir, "baselines", "*"))):
        meta = _read_json(os.path.join(bdir, "metrics.json"))
        assert meta["split_hash"] == split_hash, f"baseline {bdir} split mismatch"
        z = np.load(os.path.join(bdir, "test_predictions.npz"))
        entries.append((meta["method"], 0, _read_rows(os.path.join(bdir, "per_intervention.csv")),
                        dict(zip(z["conditions"].tolist(), z["preds"])), meta))
    run_status, chosen = collect_runs(exp_dir, split_hash)
    for (method, seed), d in sorted(chosen.items()):
        meta = _read_json(os.path.join(d, "metrics.json"))
        z = np.load(os.path.join(d, "test_predictions.npz"))
        entries.append((method, seed, _read_rows(os.path.join(d, "per_intervention.csv")),
                        dict(zip(z["conditions"].tolist(), z["preds"])), meta))
    for _, _, rows, _, _ in entries:
        assert sorted(r["condition"] for r in rows) == sorted(test), "evaluation set differs between methods"

    # ---------------------------------------------------------------- per-run table
    table = []
    for method, seed, rows, _, meta in entries:
        rec = {"method": method, "seed": seed if method in ("jepa", "scgpt") else "-",
               "checkpoint": meta.get("checkpoint", "-"), "split_hash": split_hash,
               "epochs_completed": meta.get("epochs_completed", "-"), "status": meta.get("status", "completed"),
               "elapsed_min": round(meta.get("elapsed_seconds", 0) / 60, 1),
               "peak_gpu_mem_gb": round(meta.get("peak_gpu_mem_gb", 0), 1) if meta.get("peak_gpu_mem_gb") else "-",
               "deviations": "; ".join(meta.get("deviations", []))}
        for k in METRIC_KEYS:
            v = np.array([r[k] for r in rows], float)
            ok = ~np.isnan(v)
            rec[k] = float(v[ok].mean()) if ok.any() else float("nan")
            rec[f"{k}_n"] = int(ok.sum())
        lo, hi = bootstrap_mean_ci(np.array([r["delta_rmse"] for r in rows]))
        rec["delta_rmse_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
        table.append(rec)

    # ---------------------------------------------------------------- seed-averaged per intervention
    per_int = {}  # method -> condition -> metric -> mean over seeds
    for method, _, rows, _, _ in entries:
        for r in rows:
            per_int.setdefault(method, {}).setdefault(r["condition"], {k: [] for k in METRIC_KEYS})
            for k in METRIC_KEYS:
                per_int[method][r["condition"]][k].append(r[k])
    seed_avg = {m: {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in cs.items()} for m, cs in per_int.items()}
    methods = [m for m in METHOD_ORDER if m in seed_avg]

    summary = []
    for m in methods:
        rec = {"method": m, "label": METHOD_LABEL[m],
               "n_seeds": len([e for e in entries if e[0] == m])}
        for k in METRIC_KEYS:
            v = np.array([seed_avg[m][c][k] for c in test])
            ok = ~np.isnan(v)
            rec[k] = float(v[ok].mean()) if ok.any() else float("nan")
            rec[f"{k}_n"] = int(ok.sum())
        rec["delta_rmse_ci95"] = bootstrap_mean_ci(np.array([seed_avg[m][c]["delta_rmse"] for c in test]))
        summary.append(rec)

    def vec(m, k="delta_rmse"):
        return np.array([seed_avg[m][c][k] for c in test])

    comparisons = {}
    for a, b in [("jepa", "scgpt"), ("jepa", "linear"), ("scgpt", "linear"), ("jepa", "training_mean"),
                 ("scgpt", "training_mean"), ("linear", "training_mean")]:
        if a in seed_avg and b in seed_avg:
            comparisons[f"{a}_minus_{b}_delta_rmse"] = paired_bootstrap_diff(vec(a), vec(b))
            comparisons[f"{a}_minus_{b}_de20_delta_pearson"] = paired_bootstrap_diff(
                vec(a, "de20_delta_pearson"), vec(b, "de20_delta_pearson"))

    # common valid set for correlation metrics (no-change is undefined everywhere by construction)
    corr_methods = [m for m in methods if m != "no_change"]
    common = [c for c in test if all(not math.isnan(seed_avg[m][c]["delta_pearson"]) for m in corr_methods)]
    claim = _claim(comparisons, methods)

    # ---------------------------------------------------------------- write tables
    _write_csv(os.path.join(out_dir, "results_by_run.csv"), table)
    _write_csv(os.path.join(out_dir, "results_summary.csv"),
               [{**r, "delta_rmse_ci95": f"[{r['delta_rmse_ci95'][0]:.4f}, {r['delta_rmse_ci95'][1]:.4f}]"} for r in summary])
    long_rows = [{"method": m, "seed": s, **r} for m, s, rows, _, _ in entries for r in rows]
    _write_csv(os.path.join(out_dir, "per_intervention_all.csv"), long_rows)
    _write_csv(os.path.join(out_dir, "run_status.csv"), run_status)
    bench = _benchmark_table(exp_dir)
    report = {"track": "B", "split_hash": split_hash, "n_test_interventions": len(test),
              "summary": summary, "comparisons": comparisons, "claim": claim, "run_status": run_status,
              "correlation_common_set": {"n": len(common), "excluded": [c for c in test if c not in common],
                                         "note": "no_change delta Pearson is undefined (zero predicted delta)"},
              "benchmark": bench, "manifest": {k: ds.manifest[k] for k in
                                               ["dataset", "n_cells", "n_genes", "split_sizes", "split_seed", "h5ad_sha256"]}}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    _write_markdown(os.path.join(out_dir, "results.md"), report, table)

    # ---------------------------------------------------------------- demo data + plots
    demo = _demo_payload(ds, entries, seed_avg, methods, summary, ctrl, claim, report)
    with open(os.path.join(out_dir, "demo_data.json"), "w") as f:
        json.dump(demo, f, separators=(",", ":"))
    with open(os.path.join(os.path.dirname(__file__), "demo_template.html")) as f:
        html = f.read().replace("/*__DEMO_DATA__*/null", json.dumps(demo, separators=(",", ":")))
    with open(os.path.join(out_dir, "demo.html"), "w") as f:
        f.write(html)
    _plots(out_dir, summary, seed_avg, test, methods)
    return report


def _claim(comp, methods) -> dict:
    """Map measured evidence to the outline's permissible-claim table."""
    def better(key):
        c = comp.get(key)
        return c is not None and c["ci_high"] < 0  # lower RMSE, CI excludes 0

    if "jepa" not in methods or "scgpt" not in methods:
        return {"level": "baselines_only", "text": "Only baselines are complete; no model comparison can be claimed."}
    strong = [b for b in ("linear", "training_mean") if b in methods]
    if better("jepa_minus_scgpt_delta_rmse") and all(better(f"jepa_minus_{b}_delta_rmse") for b in strong) and strong:
        txt = ("Our JEPA adaptation has lower held-out delta RMSE than matched scGPT and the simple baselines on this "
               "Adamson split (bootstrap over interventions; 2 training seeds, one fixed split).")
        lvl = "beats_strong_baselines"
    elif better("jepa_minus_scgpt_delta_rmse"):
        txt = ("Preliminary: our JEPA adaptation improves delta RMSE over matched scGPT on this split, but not over all "
               "simple baselines.")
        lvl = "beats_matched_scgpt_only"
    else:
        txt = ("No supported improvement: the JEPA auxiliary objective did not reduce held-out delta RMSE versus matched "
               "scGPT with a bootstrap CI excluding zero. This is a valid negative/inconclusive result.")
        lvl = "no_improvement"
    if "linear" not in methods:
        txt += " Linear baseline missing — result is preliminary."
    return {"level": lvl, "text": txt + " Conditional on the fixed split and fitted models; Track B, not a Cell-JEPA reproduction."}


def _benchmark_table(exp_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(exp_dir, "benchmark", "*.json"))):
        r = _read_json(p)
        env = r["environment"]
        for method, m in r["methods"].items():
            rows.append({"config": os.path.basename(p)[:-5], "stack": env["stack"], "torch": env["torch"],
                         "cuda": env["cuda"], "gpu": env["gpu"], "method": method,
                         "sec_per_step": m["seconds_per_step"], "cells_per_sec": m["cells_per_second"],
                         "validation_sec": m["validation_seconds"], "peak_mem_gb": m["peak_gpu_mem_gb"],
                         "projected_15ep_min": m["projected_15_epoch_minutes"],
                         "fingerprint": m["fingerprint"]["pred_first16"], "gates_pass": r.get("all_gates_pass")})
    ref = next((r for r in rows if r["stack"] == "hopper"), None)
    for r in rows:
        if ref is not None:
            same = [x for x in rows if x["config"] == ref["config"] and x["method"] == r["method"]]
            if same:
                r["fingerprint_max_abs_diff_vs_hopper"] = float(np.max(np.abs(np.array(r["fingerprint"]) - np.array(same[0]["fingerprint"]))))
                r["speedup_vs_hopper"] = same[0]["sec_per_step"] / r["sec_per_step"]
        r.pop("fingerprint")
    return rows


def _demo_payload(ds, entries, seed_avg, methods, summary, ctrl, claim, report):
    test = ds.split["test"]
    by_method_preds = {}
    for m, s, _, preds, _ in entries:
        by_method_preds.setdefault(m, []).append(preds)
    conds = []
    for c in test:
        obs_d = ds.observed_mean(c) - ctrl
        top = np.argsort(-np.abs(obs_d))[:20]
        de = ds.de_idx[c]
        genes = sorted(set(top.tolist()) | set(de))
        item = {"condition": c, "target": c.split("+")[0], "n_cells": int(len(ds.cells_of(c))),
                "genes": [ds.gene_names[g] for g in genes], "observed": np.round(obs_d[genes], 4).tolist(),
                "de20": [ds.gene_names[g] for g in de], "pred": {}, "metrics": {}}
        # scatter uses every gene; quantise to keep the file small
        item["obs_all"] = np.round(obs_d, 3).tolist()
        item["pred_all"] = {}
        for m in methods:
            p = np.mean([pp[c] for pp in by_method_preds[m]], axis=0) - ctrl
            item["pred"][m] = np.round(p[genes], 4).tolist()
            item["pred_all"][m] = np.round(p, 3).tolist()
            item["metrics"][m] = {k: (None if math.isnan(v) else round(v, 4)) for k, v in seed_avg[m][c].items()}
        conds.append(item)
    ref = "scgpt" if "scgpt" in methods else methods[0]
    order = sorted(conds, key=lambda it: it["metrics"][ref]["delta_rmse"])
    return {"gene_names": list(ds.gene_names), "methods": methods, "labels": {m: METHOD_LABEL[m] for m in methods},
            "colors": {m: METHOD_COLOR[m] for m in methods},
            "summary": [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()} for r in summary],
            "conditions": conds, "typical": order[len(order) // 2]["condition"], "failure": order[-1]["condition"],
            "claim": claim, "split_hash": report["split_hash"], "run_status": report["run_status"]}


def _plots(out_dir, summary, seed_avg, test, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ys = np.arange(len(summary))
    for i, r in enumerate(summary):
        lo, hi = r["delta_rmse_ci95"]
        ax.barh(i, r["delta_rmse"], height=0.5, color=METHOD_COLOR[r["method"]][0])
        ax.plot([lo, hi], [i, i], color="#0b0b0b", lw=1.2)
        ax.text(hi, i, f"  {r['delta_rmse']:.4f}", va="center", color="#52514e", fontsize=9)
    ax.set_yticks(ys, [r["label"] for r in summary])
    ax.invert_yaxis()
    ax.set_xlabel("Mean per-intervention delta RMSE on held-out interventions (lower is better; 95% bootstrap CI)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "aggregate_delta_rmse.png"), dpi=160)
    plt.close(fig)

    show = [m for m in ("jepa", "scgpt", "linear") if m in methods]  # <=3 overlapping series
    if show:
        order = sorted(test, key=lambda c: seed_avg[show[-1]][c]["delta_rmse"])
        fig, ax = plt.subplots(figsize=(8, 0.22 * len(order) + 1.2))
        for m in show:
            ax.scatter([seed_avg[m][c]["delta_rmse"] for c in order], range(len(order)), s=36,
                       color=METHOD_COLOR[m][0], label=METHOD_LABEL[m], edgecolor="white", linewidth=1.5, zorder=3)
        ax.set_yticks(range(len(order)), [c.split("+")[0] for c in order], fontsize=8)
        ax.set_xlabel("Delta RMSE per held-out intervention (lower is better)")
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "per_intervention_delta_rmse.png"), dpi=160)
        plt.close(fig)


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _fmt(v, nd=4):
    return "undefined" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def _write_markdown(path, report, table):
    L = ["# Adamson perturbation benchmark — Track B (our JEPA adaptation; not a Cell-JEPA reproduction)", "",
         f"Split hash `{report['split_hash']}` · {report['n_test_interventions']} held-out test interventions · "
         "metrics averaged per intervention (equal weight); CIs bootstrap interventions and are conditional on the "
         "fixed split and fitted models.", "", f"**Claim supported:** {report['claim']['text']}", "",
         "## Summary (seed-averaged)", "",
         "| Method | seeds | delta RMSE ↓ | 95% CI | delta Pearson | DE20 RMSE ↓ | DE20 Pearson | DE20 direction | abs Pearson |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in report["summary"]:
        lo, hi = r["delta_rmse_ci95"]
        L.append(f"| {r['label']} | {r['n_seeds']} | {_fmt(r['delta_rmse'])} | [{lo:.4f}, {hi:.4f}] | "
                 f"{_fmt(r['delta_pearson'])} ({r['delta_pearson_n']}/{report['n_test_interventions']}) | "
                 f"{_fmt(r['de20_delta_rmse'])} | {_fmt(r['de20_delta_pearson'])} | {_fmt(r['de20_direction_acc'])} | "
                 f"{_fmt(r['abs_pearson'])} |")
    L += ["", "## Paired comparisons (mean A − B over interventions, 95% bootstrap CI)", "",
          "| Comparison | mean diff | CI | n | frac A lower |", "|---|---|---|---|---|"]
    for k, c in report["comparisons"].items():
        L.append(f"| {k} | {_fmt(c['mean_diff'])} | [{_fmt(c['ci_low'])}, {_fmt(c['ci_high'])}] | {c['n']} | {_fmt(c['frac_a_lower'], 2)} |")
    L += ["", "## Individual runs", "",
          "| method | seed | epochs | status | delta RMSE | CI | delta Pearson | minutes | peak GPU GB |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        L.append(f"| {r['method']} | {r['seed']} | {r['epochs_completed']} | {r['status']} | {_fmt(r['delta_rmse'])} | "
                 f"{r['delta_rmse_ci95']} | {_fmt(r['delta_pearson'])} | {r['elapsed_min']} | {r['peak_gpu_mem_gb']} |")
    L += ["", "## Run status", "", "| run | method | seed | status | epochs | minutes | gpu |", "|---|---|---|---|---|---|---|"]
    for r in report["run_status"]:
        L.append(f"| {r['run']} | {r.get('method')} | {r.get('seed')} | {r['status']} | {r.get('epochs', '-')} | "
                 f"{r.get('minutes', '-')} | {r.get('gpu', '-')} |")
    if report["benchmark"]:
        L += ["", "## Hardware / software-stack benchmark (pilot)", "",
              "| config | stack | torch | gpu | method | s/step | speedup vs hopper-H100 | val s | peak GB | proj. 15 ep (min) | fingerprint Δ |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for b in report["benchmark"]:
            L.append(f"| {b['config']} | {b['stack']} | {b['torch']} | {b['gpu']} | {b['method']} | {b['sec_per_step']:.3f} | "
                     f"{_fmt(b.get('speedup_vs_hopper'), 2)} | {b['validation_sec']:.1f} | {b['peak_mem_gb']:.1f} | "
                     f"{b['projected_15ep_min']:.0f} | {_fmt(b.get('fingerprint_max_abs_diff_vs_hopper'), 5)} |")
    L += ["", f"Correlation metrics common valid set: {report['correlation_common_set']['n']} interventions; "
          f"{report['correlation_common_set']['note']}.", "",
          "Limitations: one fixed intervention-level split; two training seeds (not independent split replicates); "
          "K562 CRISPRi knockdown RNA readout only; GEARS HVG selection inherited from the whole dataset; no claim "
          "about protein, drug response, or therapeutic benefit."]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
