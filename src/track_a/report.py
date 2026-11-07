"""Track A aggregation — Cell-JEPA reimplemented from the paper (unofficial).

Track A fine-tunes two kidney-pretrained checkpoints (JEPA- and scGPT-objective) with two
fine-tuning methods (`scgpt`, `celljepa`) and several seeds inside ONE experiment directory,
so runs must be keyed by (checkpoint, method, seed) — `src.report.collect_runs` keys by
(method, seed) and would silently collide. Everything here is numpy-only and reads saved
evaluation outputs; it never needs the dataset cache (the held-out condition set is taken from
the saved per-intervention rows and checked to be identical across every entry).

Track B numbers may appear ONLY in the separate "for context" section and never in a
matched comparison.
"""

from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

from ..evaluate import METRIC_KEYS, bootstrap_mean_ci, paired_bootstrap_diff
from ..report import METHOD_COLOR, _fmt, _read_json, _read_rows, _write_csv

TRACK_LABEL = "Track A — Cell-JEPA reimplemented from the paper (unofficial)"
DEFAULT_CKPT = "scGPT_human"  # train_blackwell's default `ckpt`
PAPER_ARM = "jepa_kidney/celljepa"      # paper: kidney-JEPA pretraining + Cell-JEPA fine-tuning
MATCHED_ARM = "scgpt_kidney/scgpt"      # paper: kidney-scGPT pretraining + scGPT fine-tuning
BASELINES = ["linear", "training_mean", "no_change"]
NON_METRIC = {"condition", "n_cells"}

ARM_ORDER = [PAPER_ARM, MATCHED_ARM, "jepa_kidney/scgpt", "scgpt_kidney/celljepa", *BASELINES]
ARM_LABEL = {
    PAPER_ARM: "Kidney-JEPA PT + Cell-JEPA FT (paper's method, reimpl.)",
    MATCHED_ARM: "Kidney-scGPT PT + scGPT FT (matched)",
    "jepa_kidney/scgpt": "Kidney-JEPA PT + scGPT FT (ablation)",
    "scgpt_kidney/celljepa": "Kidney-scGPT PT + Cell-JEPA FT (ablation)",
    "linear": "Linear (gene emb.)", "training_mean": "Training mean", "no_change": "No change",
}
# fixed entity -> categorical slot (dataviz reference palette, light / dark); never by rank
ARM_COLOR = {PAPER_ARM: METHOD_COLOR["jepa"], MATCHED_ARM: METHOD_COLOR["scgpt"],
             "jepa_kidney/scgpt": ("#e87ba4", "#d55181"), "scgpt_kidney/celljepa": ("#6250d6", "#9085e9"),
             **{b: METHOD_COLOR[b] for b in BASELINES}}
OTHER_COLOR = ("#e34948", "#e66767")

# Published Adamson numbers (Cell-JEPA, arXiv 2602.02093, Table 3/4) extracted from the arXiv HTML.
# Column correspondence to our GEARS-style columns is our reading of the table — spot-check vs PDF.
PUBLISHED_SOURCE = "arXiv 2602.02093 HTML (Tables 3-4); spot-check vs PDF"
PUBLISHED = {
    MATCHED_ARM: {"gears_pearson": 0.905, "gears_pearson_de": 0.699, "gears_top20_de_nd": 0.677,
                  "gears_delta_pearson": 0.173, "gears_delta_pearson_de": 0.101, "gears_delta_top20_de_nd": 0.120},
    PAPER_ARM: {"gears_pearson": 0.937, "gears_pearson_de": 0.741, "gears_top20_de_nd": 0.720,
                "gears_delta_pearson": 0.120, "gears_delta_pearson_de": -0.176, "gears_delta_top20_de_nd": -0.152},
}
PUBLISHED_LABEL = {MATCHED_ARM: "scGPT", PAPER_ARM: "Cell-JEPA"}

COMPARISONS = [  # (A, B, why) — A − B; negative delta RMSE difference favours A
    (PAPER_ARM, MATCHED_ARM, "paper's comparison (pretraining + fine-tuning both differ)"),
    ("jepa_kidney/scgpt", MATCHED_ARM, "pretraining effect, scGPT fine-tuning held fixed"),
    ("scgpt_kidney/celljepa", MATCHED_ARM, "fine-tuning effect, kidney-scGPT checkpoint held fixed"),
    (PAPER_ARM, "jepa_kidney/scgpt", "fine-tuning effect, kidney-JEPA checkpoint held fixed"),
    (PAPER_ARM, "linear", "vs strong simple baseline"),
    (MATCHED_ARM, "linear", "vs strong simple baseline"),
    (PAPER_ARM, "training_mean", "vs simple baseline"),
    (MATCHED_ARM, "training_mean", "vs simple baseline"),
]


# =============================================================================== keying


def ckpt_key(ckpt, default: str | None = DEFAULT_CKPT) -> str | None:
    """'/vol/trackA/checkpoints/jepa_kidney/' -> 'jepa_kidney'; missing -> `default`."""
    if not ckpt:
        return default
    return os.path.basename(str(ckpt).rstrip("/")) or default


def arm_id(ckpt: str, method: str) -> str:
    return f"{ckpt}/{method}"


def arm_label(arm: str) -> str:
    return ARM_LABEL.get(arm, arm.replace("/", " PT + ") + " FT")


def arm_color(arm: str):
    return ARM_COLOR.get(arm, OTHER_COLOR)


def collect_track_a_runs(exp_dir: str, split_hash: str | None, default_ckpt: str | None = None):
    """Latest completed run per (checkpoint, method, seed); every run's status is reported.

    Identity comes from each run's config.json (`ckpt`, `method`, `seed`), falling back to
    metrics.json — never from the directory name. A run without a recorded `ckpt` is excluded
    (unless `default_ckpt` is given, e.g. for single-checkpoint context experiments): falling back
    to a default would collapse the two kidney checkpoints onto one key.
    """
    status, chosen = [], {}
    for d in sorted(glob.glob(os.path.join(exp_dir, "runs", "*"))):
        if not os.path.isdir(d):
            continue
        sp, cp, mp = (os.path.join(d, n) for n in ("status.json", "config.json", "metrics.json"))
        st = _read_json(sp) if os.path.exists(sp) else {"status": "submitted"}
        cfg = _read_json(cp) if os.path.exists(cp) else {}
        m = _read_json(mp) if os.path.exists(mp) else None
        method = cfg.get("method", (m or {}).get("method"))
        seed = cfg.get("seed", (m or {}).get("seed"))
        ck = ckpt_key(cfg.get("ckpt", (m or {}).get("ckpt")), default_ckpt)
        state = st.get("status", "?")
        if state == "running" and m is None:
            state = "running_or_died"  # a container that hit its timeout leaves this behind
        row = {"run": os.path.basename(d), "checkpoint": ck, "method": method, "seed": seed,
               "input_transform": cfg.get("input_transform", "-"), "status": state}
        if m:
            env = m.get("environment") or {}
            row.update(epochs=m.get("epochs_completed"), minutes=round(m.get("elapsed_seconds", 0) / 60, 1),
                       gpu=env.get("gpu"), reload_ok=m.get("reload_ok"))
            missing = [f for f in ("per_intervention.csv", "test_predictions.npz")
                       if not os.path.exists(os.path.join(d, f))]
            if method is None or seed is None or ck is None:
                row["status"] = "excluded_unidentified"
            elif split_hash is not None and m.get("split_hash") != split_hash:
                row["status"] = "excluded_split_mismatch"
            elif missing:
                row["status"] = "excluded_missing_" + "+".join(missing)
            elif m.get("status") in ("completed", "stopped_by_time_budget"):
                key = (ck, method, int(seed))
                if key not in chosen or os.path.getmtime(mp) > os.path.getmtime(os.path.join(chosen[key], "metrics.json")):
                    chosen[key] = d
        status.append(row)
    return status, chosen


# =============================================================================== aggregation


def _numeric_keys(rows) -> list[str]:
    keys = []
    for r in rows:
        for k, v in r.items():
            if k not in NON_METRIC and isinstance(v, float) and k not in keys:
                keys.append(k)
    return keys


def _mean_defined(v):
    v = np.asarray(v, float)
    ok = ~np.isnan(v)
    return (float(v[ok].mean()) if ok.any() else float("nan")), int(ok.sum())


def _load_entries(exp_dir, split_hash, default_ckpt=None):
    """[(arm, ckpt, method, seed, rows, meta, run_dir)] for baselines + chosen Track A runs."""
    entries = []
    for bdir in sorted(glob.glob(os.path.join(exp_dir, "baselines", "*"))):
        mp = os.path.join(bdir, "metrics.json")
        if not os.path.exists(mp):
            continue
        meta = _read_json(mp)
        if split_hash is None:
            split_hash = meta["split_hash"]
        if meta["split_hash"] != split_hash:
            raise AssertionError(f"baseline {bdir} split mismatch ({meta['split_hash']} != {split_hash})")
        name = meta.get("method", os.path.basename(bdir))
        entries.append((name, "-", name, 0, _rows_with_percell(bdir), meta, bdir))
    run_status, chosen = collect_track_a_runs(exp_dir, split_hash, default_ckpt)
    for (ck, method, seed), d in sorted(chosen.items()):
        entries.append((arm_id(ck, method), ck, method, seed, _rows_with_percell(d),
                        _read_json(os.path.join(d, "metrics.json")), d))
    return entries, run_status, split_hash


def _rows_with_percell(d: str) -> list[dict]:
    """per_intervention.csv rows, merged (by condition) with per_intervention_percell.csv if present."""
    rows = _read_rows(os.path.join(d, "per_intervention.csv"))
    pc = os.path.join(d, "per_intervention_percell.csv")
    if os.path.exists(pc):
        extra = {r["condition"]: r for r in _read_rows(pc)}
        for r in rows:
            r.update({k: v for k, v in extra.get(r["condition"], {}).items()
                      if k.startswith("percell_")})
    return rows


def _summarise_arms(entries, test):
    """Seed-averaged per-intervention values per arm, then equal-weight mean over interventions."""
    per = {}  # arm -> condition -> key -> [values over seeds]
    keys = []
    for arm, _, _, _, rows, _, _ in entries:
        for k in _numeric_keys(rows):
            if k not in keys:
                keys.append(k)
        for r in rows:
            slot = per.setdefault(arm, {}).setdefault(r["condition"], {})
            for k, v in r.items():
                if k not in NON_METRIC and isinstance(v, float):
                    slot.setdefault(k, []).append(v)
    seed_avg = {a: {c: {k: _mean_defined(v)[0] for k, v in d.items()} for c, d in cs.items()} for a, cs in per.items()}
    arms = [a for a in ARM_ORDER if a in seed_avg] + sorted(a for a in seed_avg if a not in ARM_ORDER)
    summary = []
    for a in arms:
        rec = {"arm": a, "label": arm_label(a), "n_seeds": sum(1 for e in entries if e[0] == a)}
        for k in keys:
            rec[k], rec[f"{k}_n"] = _mean_defined([seed_avg[a][c].get(k, float("nan")) for c in test])
        rec["delta_rmse_ci95"] = bootstrap_mean_ci(np.array([seed_avg[a][c].get("delta_rmse", float("nan")) for c in test]))
        summary.append(rec)
    return seed_avg, arms, summary, keys


def _context(context_exp_dirs):
    """Other experiments (Track B; public-scGPT Cell-JEPA) summarised SEPARATELY, never compared."""
    out = []
    for d in context_exp_dirs or []:
        if not os.path.isdir(d):
            out.append({"exp_id": os.path.basename(d.rstrip("/")), "available": False})
            continue
        try:
            sh = _read_json(os.path.join(d, "experiment.json"))["split_hash"] \
                if os.path.exists(os.path.join(d, "experiment.json")) else None
            entries, run_status, sh = _load_entries(d, sh, default_ckpt=DEFAULT_CKPT)
            sets = {tuple(sorted(r["condition"] for r in e[4])) for e in entries}
            if len(sets) > 1:
                raise AssertionError("evaluation condition sets differ between entries")
            test = list(next(iter(sets), ()))
            rows = []
            if entries:
                _, arms, summ, _ = _summarise_arms(entries, test)
                rows = [{"arm": r["arm"], "n_seeds": r["n_seeds"], "delta_rmse": r["delta_rmse"],
                         "delta_rmse_ci95": list(r["delta_rmse_ci95"])} for r in summ]
            out.append({"exp_id": os.path.basename(d.rstrip("/")), "available": True, "split_hash": sh,
                        "n_test_interventions": len(test), "summary": rows,
                        "n_runs_seen": len(run_status)})
        except Exception as e:  # context must never break the Track A report — but say why
            out.append({"exp_id": os.path.basename(d.rstrip("/")), "available": False, "error": repr(e)[:500]})
    return out


def _pretraining_provenance(ckpt_root, ckpts):
    out = {}
    for ck in ckpts:
        if ck == DEFAULT_CKPT or ckpt_root is None:
            out[ck] = {"note": "public checkpoint" if ck == DEFAULT_CKPT else "ckpt_root not given"}
            continue
        d = os.path.join(ckpt_root, ck)
        rec = {"dir": d, "exists": os.path.isdir(d)}
        for f in ("args.json", "best_model.pt", "vocab.json"):
            rec[f] = os.path.exists(os.path.join(d, f))
        for f in ("pretrain_summary.json", "metrics.json"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                try:
                    rec[f] = _read_json(p)
                except Exception as e:
                    rec[f] = {"unreadable": repr(e)[:200]}
        out[ck] = rec
    return out


def _claim(comp, arms) -> dict:
    if PAPER_ARM not in arms or MATCHED_ARM not in arms:
        return {"level": "incomplete",
                "text": "The paper's matched comparison is incomplete (need both kidney-JEPA+Cell-JEPA and "
                        "kidney-scGPT+scGPT runs); no Track A claim."}
    c = comp.get(f"{PAPER_ARM} - {MATCHED_ARM}", {}).get("delta_rmse")
    if c and c["ci_high"] < 0:
        lvl, txt = "paper_arm_better", ("Our reimplementation of Cell-JEPA (kidney-JEPA pretraining + Cell-JEPA "
                                        "fine-tuning) has lower held-out delta RMSE than the matched kidney-scGPT arm.")
    elif c and c["ci_low"] > 0:
        lvl, txt = "matched_arm_better", ("The matched kidney-scGPT arm has lower held-out delta RMSE than our "
                                          "Cell-JEPA reimplementation.")
    else:
        lvl, txt = "inconclusive", ("No difference in held-out delta RMSE between our Cell-JEPA reimplementation "
                                    "and the matched kidney-scGPT arm (bootstrap CI includes zero).")
    for b in ("linear", "training_mean"):
        cb = comp.get(f"{PAPER_ARM} - {b}", {}).get("delta_rmse")
        if cb and not cb["ci_high"] < 0:
            txt += f" It does not beat the {arm_label(b)} baseline with a CI excluding zero."
    return {"level": lvl, "text": txt + " Unofficial reimplementation from the paper text (no official code or "
            "checkpoints); see the deviation log. Conditional on one fixed split and the fitted models."}


def aggregate_track_a(exp_dir: str, out_dir: str, context_exp_dirs=(), ckpt_root: str | None = None,
                      make_plots: bool = True) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    ep = os.path.join(exp_dir, "experiment.json")
    split_hash = _read_json(ep)["split_hash"] if os.path.exists(ep) else None
    entries, run_status, split_hash = _load_entries(exp_dir, split_hash)
    if not entries:
        raise RuntimeError(f"nothing to aggregate in {exp_dir} (no baselines, no completed runs)")
    test = sorted(r["condition"] for r in entries[0][4])
    for e in entries:
        if sorted(r["condition"] for r in e[4]) != test:
            raise AssertionError(f"evaluation set differs: {e[6]}")

    # ---------------------------------------------------------------- per-run table
    table = []
    for arm, ck, method, seed, rows, meta, d in entries:
        is_run = ck != "-"
        rec = {"arm": arm, "checkpoint": ck, "method": method, "seed": seed if is_run else "-",
               "run": os.path.basename(d) if is_run else "baseline", "split_hash": split_hash,
               "epochs_completed": meta.get("epochs_completed", "-"), "selected_epoch": meta.get("selected_epoch", "-"),
               "status": meta.get("status", "completed"), "elapsed_min": round(meta.get("elapsed_seconds", 0) / 60, 1),
               "peak_gpu_mem_gb": round(meta["peak_gpu_mem_gb"], 1) if meta.get("peak_gpu_mem_gb") else "-",
               "deviations": "; ".join(meta.get("deviations", []))}
        for k in _numeric_keys(rows):
            rec[k], rec[f"{k}_n"] = _mean_defined([r.get(k, float("nan")) for r in rows])
        lo, hi = bootstrap_mean_ci(np.array([r["delta_rmse"] for r in rows]))
        rec["delta_rmse_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
        table.append(rec)

    seed_avg, arms, summary, keys = _summarise_arms(entries, test)
    gears_keys = [k for k in keys if k.startswith("gears_")]

    def vec(a, k):
        return np.array([seed_avg[a][c].get(k, float("nan")) for c in test])

    comparisons = {}
    for a, b, why in COMPARISONS:
        if a in seed_avg and b in seed_avg:
            c = {"why": why, "delta_rmse": paired_bootstrap_diff(vec(a, "delta_rmse"), vec(b, "delta_rmse")),
                 "de20_delta_pearson": paired_bootstrap_diff(vec(a, "de20_delta_pearson"), vec(b, "de20_delta_pearson"))}
            for k in gears_keys:
                c[k] = paired_bootstrap_diff(vec(a, k), vec(b, k))
            comparisons[f"{a} - {b}"] = c

    published = []
    for arm, pub in PUBLISHED.items():
        ours = next((r for r in summary if r["arm"] == arm), None)
        published.append({"arm": arm, "paper_row": PUBLISHED_LABEL[arm],
                          "metrics": {k: {"published": v, "ours": (ours or {}).get(k),
                                          "ours_percell": (ours or {}).get(k.replace("gears_", "percell_"))}
                                      for k, v in pub.items()}})

    claim = _claim(comparisons, arms)
    ckpts = sorted({e[1] for e in entries if e[1] != "-"})
    report = {"track": "A", "label": TRACK_LABEL, "exp_dir": exp_dir, "split_hash": split_hash,
              "n_test_interventions": len(test), "summary": summary, "comparisons": comparisons,
              "gears_columns": gears_keys, "published": published, "published_source": PUBLISHED_SOURCE,
              "claim": claim, "run_status": run_status,
              "pretraining": _pretraining_provenance(ckpt_root, ckpts),
              "context_not_comparable": _context(context_exp_dirs)}

    _write_csv(os.path.join(out_dir, "results_by_run.csv"), table)
    _write_csv(os.path.join(out_dir, "results_summary.csv"),
               [{**r, "delta_rmse_ci95": f"[{r['delta_rmse_ci95'][0]:.4f}, {r['delta_rmse_ci95'][1]:.4f}]"} for r in summary])
    _write_csv(os.path.join(out_dir, "per_intervention_all.csv"),
               [{"arm": a, "checkpoint": ck, "method": m, "seed": s, **r} for a, ck, m, s, rows, _, _ in entries for r in rows])
    _write_csv(os.path.join(out_dir, "run_status.csv"), run_status)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    _write_markdown(os.path.join(out_dir, "results.md"), report, table)
    if make_plots:
        _plots(out_dir, summary)
    return report


# =============================================================================== outputs


def _plots(out_dir, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(summary) + 1.6))
    ax.set_axisbelow(True)
    for i, r in enumerate(summary):
        lo, hi = r["delta_rmse_ci95"]
        ax.barh(i, r["delta_rmse"], height=0.5, color=arm_color(r["arm"])[0])
        ax.plot([lo, hi], [i, i], color="#0b0b0b", lw=1.2)
        ax.text(hi, i, f"  {r['delta_rmse']:.4f}", va="center", color="#52514e", fontsize=9)
    ax.set_yticks(np.arange(len(summary)), [r["label"] + (f" (seeds={r['n_seeds']})" if "/" in r["arm"] else "")
                                            for r in summary])
    ax.invert_yaxis()
    his = [r["delta_rmse_ci95"][1] for r in summary if not math.isnan(r["delta_rmse_ci95"][1])]
    if his:
        ax.set_xlim(0, max(his) * 1.18)
    ax.set_xlabel("Mean delta RMSE, held-out interventions (lower is better; 95% CI)")
    ax.grid(axis="y", visible=False)
    fig.suptitle(TRACK_LABEL, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "track_a_delta_rmse.png"), dpi=160)
    plt.close(fig)


def _write_markdown(path, report, table):
    n = report["n_test_interventions"]
    L = [f"# Adamson perturbation benchmark — {TRACK_LABEL}", "",
         "No official Cell-JEPA code or checkpoints exist; both kidney checkpoints were pretrained by us from the "
         "paper's description (see docs/track_a.md deviation log). Not the authors' numbers.", "",
         f"Split hash `{report['split_hash']}` · {n} held-out test interventions · metrics averaged per intervention "
         "(equal weight), seed-averaged per arm; CIs bootstrap interventions and are conditional on the fixed split.",
         "", f"**Claim supported:** {report['claim']['text']}", "",
         "## Summary (seed-averaged)", "",
         "| Arm | seeds | delta RMSE ↓ | 95% CI | delta Pearson | DE20 RMSE ↓ | DE20 Pearson | DE20 direction |",
         "|---|---|---|---|---|---|---|---|"]
    for r in report["summary"]:
        lo, hi = r["delta_rmse_ci95"]
        L.append(f"| {r['label']} | {r['n_seeds']} | {_fmt(r['delta_rmse'])} | [{lo:.4f}, {hi:.4f}] | "
                 f"{_fmt(r.get('delta_pearson'))} ({r.get('delta_pearson_n', 0)}/{n}) | {_fmt(r.get('de20_delta_rmse'))} | "
                 f"{_fmt(r.get('de20_delta_pearson'))} | {_fmt(r.get('de20_direction_acc'))} |")
    L += ["", "## Paired comparisons (mean A − B over interventions, 95% bootstrap CI)", "",
          "Delta RMSE: negative favours A. Correlation metrics: positive favours A.", "",
          "| A − B | why | metric | mean diff | CI | n | frac A lower |", "|---|---|---|---|---|---|---|"]
    for k, c in report["comparisons"].items():
        for metric, v in c.items():
            if metric == "why":
                continue
            L.append(f"| {k} | {c['why']} | {metric} | {_fmt(v['mean_diff'])} | [{_fmt(v['ci_low'])}, "
                     f"{_fmt(v['ci_high'])}] | {v['n']} | {_fmt(v['frac_a_lower'], 2)} |")
    L += ["", "## GEARS-style metrics vs published (Adamson)", "",
          f"Published values: {report['published_source']}. Column correspondence is our reading of the paper's "
          "table; the paper's evaluation is ambiguous. We report two readings: GEARS mean-based (correlate "
          "population-mean prediction with population-mean truth) and per-cell (correlate each prediction with "
          "one paired treated cell, Δ against that cell's own control input, then average). Mean-based scores "
          "are near-saturated (no-change ≈ 0.99 Pearson), so the published values (~0.9) are more consistent "
          "with a per-cell protocol; neither reading is confirmed.", ""]
    if report["gears_columns"]:
        L += ["| Arm (paper row) | metric | ours (mean-based) | ours (per-cell) | published |", "|---|---|---|---|---|"]
        for p in report["published"]:
            for k, v in p["metrics"].items():
                L.append(f"| {arm_label(p['arm'])} ({p['paper_row']}) | {k.replace('gears_', '')} | "
                         f"{_fmt(v['ours'], 3)} | {_fmt(v.get('ours_percell'), 3)} | {v['published']:.3f} |")
    else:
        L.append("_No `gears_*` columns in per_intervention.csv — GEARS metrics not available for these runs._")
    L += ["", "## Individual runs", "",
          "| arm | seed | run | epochs | sel. epoch | status | delta RMSE | CI | minutes | peak GPU GB |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        L.append(f"| {arm_label(r['arm'])} | {r['seed']} | {r['run']} | {r['epochs_completed']} | {r['selected_epoch']} | "
                 f"{r['status']} | {_fmt(r['delta_rmse'])} | {r['delta_rmse_ci95']} | {r['elapsed_min']} | {r['peak_gpu_mem_gb']} |")
    L += ["", "## Run status", "", "| run | checkpoint | method | seed | input | status | epochs | minutes | gpu |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in report["run_status"]:
        L.append(f"| {r['run']} | {r['checkpoint']} | {r['method']} | {r['seed']} | {r['input_transform']} | "
                 f"{r['status']} | {r.get('epochs', '-')} | {r.get('minutes', '-')} | {r.get('gpu', '-')} |")
    L += ["", "## Pretrained checkpoints", "", "```json", json.dumps(report["pretraining"], indent=2, default=str)[:4000],
          "```"]
    L += ["", "## For context only — other experiments (NOT a matched comparison)", "",
          "These numbers come from different experiments (Track B: public whole-human scGPT; different "
          "checkpoints/recipes). They are shown for orientation only and must not be compared with Track A as if "
          "matched.", ""]
    for c in report["context_not_comparable"]:
        if not c.get("available"):
            L.append(f"- `{c['exp_id']}`: not available{(' — ' + c['error']) if c.get('error') else ''}")
            continue
        L.append(f"- `{c['exp_id']}` (split `{c['split_hash']}`, {c['n_test_interventions']} test interventions):")
        for r in c["summary"]:
            lo, hi = r["delta_rmse_ci95"]
            L.append(f"  - {r['arm']} (n={r['n_seeds']}): delta RMSE {_fmt(r['delta_rmse'])} [{_fmt(lo)}, {_fmt(hi)}]")
    L += ["", "Limitations: unofficial reimplementation (unspecified paper details resolved as documented in the "
          "deviation log); one fixed GEARS simulation split; two fine-tuning seeds; one pretraining run per "
          "objective; K562 CRISPRi RNA readout only."]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# =============================================================================== synthetic layout (smoke / tests)


def write_synthetic_layout(exp_dir: str, n_test: int = 6, seeds=(1, 2), with_gears: bool = True,
                           ckpt_root: str = "/vol/trackA/checkpoints", seed: int = 0) -> dict:
    """Fake baselines + 2 checkpoints x {scgpt, celljepa} x seeds, with colliding (method, seed)."""
    import csv

    rng = np.random.default_rng(seed)
    conds = [f"G{i}+ctrl" for i in range(n_test)]
    split_hash = "synthetic"
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "experiment.json"), "w") as f:
        json.dump({"exp_id": os.path.basename(exp_dir), "track": "A", "split_hash": split_hash}, f)

    def write(d, meta, level):
        os.makedirs(d, exist_ok=True)
        rows = []
        for c in conds:
            r = {"condition": c, "n_cells": 50}
            for k in METRIC_KEYS:
                r[k] = float(level + 0.01 * rng.normal()) if "rmse" in k else float(np.clip(0.5 + 0.1 * rng.normal(), -1, 1))
            if with_gears:
                for k in PUBLISHED[PAPER_ARM]:
                    r[k] = float(np.clip(0.7 + 0.05 * rng.normal(), -1, 1))
            rows.append(r)
        with open(os.path.join(d, "per_intervention.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        np.savez_compressed(os.path.join(d, "test_predictions.npz"), conditions=np.array(conds),
                            preds=rng.normal(size=(len(conds), 4)))
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump({"split_hash": split_hash, "status": "completed", "elapsed_seconds": 60.0, **meta}, f)

    for i, b in enumerate(BASELINES):
        write(os.path.join(exp_dir, "baselines", b), {"method": b}, 0.30 + 0.02 * i)
    levels = {("jepa_kidney", "celljepa"): 0.20, ("scgpt_kidney", "scgpt"): 0.22,
              ("jepa_kidney", "scgpt"): 0.21, ("scgpt_kidney", "celljepa"): 0.23}
    tags = {"jepa_kidney": "kidneyjepa_v1", "scgpt_kidney": "kidneyscgpt_v1"}
    for (ck, method), lvl in levels.items():
        for s in seeds:
            d = os.path.join(exp_dir, "runs", f"{method}_s{s}_{tags[ck]}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"method": method, "seed": s, "ckpt": os.path.join(ckpt_root, ck),
                           "input_transform": "bins50"}, f)
            write(d, {"method": method, "seed": s, "epochs_completed": 3, "selected_epoch": 2,
                      "environment": {"gpu": "synthetic"}, "reload_ok": True}, lvl)
            with open(os.path.join(d, "status.json"), "w") as f:
                json.dump({"status": "completed"}, f)
    return {"exp_dir": exp_dir, "conditions": conds, "n_runs": len(levels) * len(seeds)}
