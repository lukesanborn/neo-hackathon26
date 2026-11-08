"""Markdown tables for X1/X2 from diag/results/X_*_retrieval.json and X_program_probe.json.

  ../.venv/bin/python diag/xjepa_report.py > diag/results/X_report.md
"""
import json
import os
import sys

R = "diag/results"
MAIN = ["cinv_mrr", "cinv_top1", "cinv_top5", "inv_mrr", "inv_top1", "id_top1", "ref_spearman"]
DIFF = ["cinv_mrr", "cinv_top1", "inv_mrr", "id_top1", "ref_spearman"]


def f(x, nd=3):
    return "undef" if x is None or x != x else f"{x:.{nd}f}"


def cell(c):
    return f"{f(c['mean'])} [{f(c['lo'], 2)}, {f(c['hi'], 2)}]" if c["mean"] is not None else "undef"


def keep(arm):
    return ("/runs/" in arm or arm.startswith(("baselines/", "geneemb_linear/", "(upper")))


def table(res, title):
    n = res["n_test"]
    out = [f"### {title} — {n} held-out knockdowns (chance top-1 {1 / n:.3f}, chance MRR "
           f"{sum(1 / k for k in range(1, n + 1)) / n:.3f})", "",
           "| arm | seeds | " + " | ".join(MAIN) + " |", "|---" * (len(MAIN) + 2) + "|"]
    arms = sorted((a for a in res["arms"] if keep(a)),
                  key=lambda a: -(res["arms"][a]["ci"]["cinv_mrr"]["mean"] or -1))
    for a in arms:
        c = res["arms"][a]["ci"]
        out.append(f"| {a.split('/runs/')[-1]} | {len(res['arms'][a]['members'])} | "
                   + " | ".join(cell(c[m]) for m in MAIN) + " |")
    out += ["", "Paired differences A − B: diff [95% CI] (fraction of resamples > 0)", "",
            "| A | B | " + " | ".join(DIFF) + " |", "|---" * (len(DIFF) + 2) + "|"]
    for d in res["diffs"]:
        cs = [f"{f(d[m]['diff'])} [{f(d[m]['lo'], 2)}, {f(d[m]['hi'], 2)}] ({f(d[m]['frac_gt0'], 2)})"
              if d[m]["n_valid"] else "undef" for m in DIFF]
        out.append(f"| {d['a'].split('/runs/')[-1]} | {d['b'].split('/runs/')[-1]} | " + " | ".join(cs) + " |")
    if res.get("skipped"):
        out += ["", "Skipped: " + "; ".join(res["skipped"])]
    return out + [""]


def probe_table(pp):
    progs = ["HALLMARK_INTERFERON_GAMMA_RESPONSE", "HALLMARK_INTERFERON_ALPHA_RESPONSE", "HALLMARK_G2M_CHECKPOINT",
             "HALLMARK_E2F_TARGETS", "HALLMARK_HYPOXIA", "HALLMARK_OXIDATIVE_PHOSPHORYLATION"]
    out = ["### Program readout probe (kidney held-out cells; program genes removed from input; 5-fold CV R²)", "",
           "| embedding | n programs | mean R² | median R² | " + " | ".join(p.replace("HALLMARK_", "") for p in progs)
           + " |", "|---" * (len(progs) + 4) + "|"]
    for n, r in sorted(pp.items(), key=lambda kv: -kv[1]["mean_r2"]):
        out.append(f"| {n} | {r['n_programs_scored']} | {f(r['mean_r2'])} | {f(r['median_r2'])} | "
                   + " | ".join(f(r["per_program"].get(p)) for p in progs) + " |")
    return out + [""]


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "X"
    lines = []
    for exp, title in (("replogle_trackA_lr3e5", "Replogle K562 essential"), ("adamson_trackA_lr3e5", "Adamson")):
        p = f"{R}/{tag}_{exp}_retrieval.json"
        if os.path.exists(p):
            lines += table(json.load(open(p)), title)
    p = f"{R}/{tag}_program_probe.json"
    if os.path.exists(p):
        lines += probe_table(json.load(open(p)))
    print("\n".join(lines))
