"""Figure: the retrieval "ladder" on Replogle validation genes (A25). Each row adds one change to the row above.

Reads diag/results/A25_val_retrieval_ci.json, writes diag/results/figures/A25_ladder.png.
Run:  uv run --no-project --with matplotlib --with numpy python diag/fig_ladder.py [--bars]
"""
import json
import os
import sys

BARS = "--bars" in sys.argv   # horizontal bars from 0 (default: dots with CI lines)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = json.load(open("diag/results/A25_val_retrieval_ci.json"))
A = R["arms"]
E = "replogle_k562_v1/runs/"
N = R["n_test"]

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
GROUPS = {  # group: (color, legend label)
    "base": ("#8a8984", "Baseline (no deep model)"),
    "vec": ("#2a78d6", "scGPT knowledge, no fine-tuning"),
    "ft": ("#eb6834", "scGPT fine-tuned"),
    "jepa": ("#1baf7a", "+ JEPA"),
}
STEPS = [  # (label, arm, group)
    ("Training mean (same prediction for every gene)", "baselines/training_mean", "base"),
    ("Baseline: linear (PCA gene embedding)", "baselines/linear", "base"),
    ("scGPT gene vectors only (no model)", "baselines/ridge_scgpt_gene_emb", "vec"),
    ("Public scGPT, fine-tuned", E + "scgpt_paper_s2pub_r0", "ft"),
    ("  + contrastive retrieval loss", E + "scgpt_paper_s2pub_r1", "ft"),
    ("  + blood pretraining (blood-scGPT)", E + "scgpt_paper_s2bscgpt_r1", "ft"),
    ("  + JEPA in pretraining (blood-JEPA)", E + "scgpt_paper_s2bjepa_r1", "jepa"),
    ("  + JEPA in fine-tuning (Cell-JEPA)", E + "celljepa_s2bjepa_r1", "jepa"),
    ("  + average 2 seeds", E + "celljepa_s2bjepa_r1__ens2", "jepa"),
]
CHANCE = {"id_pct": 0.5, "id_top1": 1 / N, "id_top3": 3 / N, "id_top10": 10 / N}


def ci(arm, m):
    c = A[arm]["ci"][m]
    return c["mean"], c["lo"], c["hi"]


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK, "text.color": INK})
fig = plt.figure(figsize=(15, 7.2), facecolor=SURF)
gs = fig.add_gridspec(1, 5, width_ratios=[3.0, 0.45, 1, 1, 1], wspace=0.12)
ys = list(range(len(STEPS)))[::-1]

# ---- panel A: ID percentile ladder with jumps ----
ax = fig.add_subplot(gs[0, 0], facecolor=SURF)
jx = fig.add_subplot(gs[0, 1], sharey=ax, facecolor=SURF)
jx.set_xlim(0, 1)
jx.axis("off")
jx.set_title("Jump vs\nrow above", fontsize=10, color=INK2, pad=10)
pct = [ci(a, "id_pct") for _, a, _ in STEPS]
jumps = [None] + [pct[i][0] - pct[i - 1][0] for i in range(1, len(STEPS))]
model_jumps = {i: j for i, j in enumerate(jumps) if j is not None and i >= 2}
big = max(model_jumps, key=lambda i: model_jumps[i])
for y, (lab, arm, g), (m, lo, hi), j in zip(ys, STEPS, pct, jumps):
    col = GROUPS[g][0]
    if BARS:
        ax.barh(y, m - 0.5, left=0.5, height=0.62, color=col, edgecolor=SURF, linewidth=2, zorder=2)
        ax.plot([lo, hi], [y, y], color=INK, lw=1.2, zorder=3)
        ax.plot([lo, lo], [y - 0.12, y + 0.12], color=INK, lw=1.2, zorder=3)
        ax.plot([hi, hi], [y - 0.12, y + 0.12], color=INK, lw=1.2, zorder=3)
    else:
        ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([m], [y], s=90, color=col, edgecolor=SURF, linewidth=2, zorder=3)
    ax.text(hi + 0.008, y, f"{m:.3f}", va="center", ha="left", fontsize=10, color=INK)
    if j is not None:
        bold = (ys.index(y) == big)
        jx.text(0.5, y, f"{j:+.3f}", va="center", ha="center", fontsize=11 if bold else 10,
                fontweight="bold" if bold else "normal", color=INK if bold else INK2)
if BARS:
    ax.axvline(0.5, color=INK2, lw=1.2, zorder=1)
    ax.text(0.502, len(STEPS) - 0.35, "chance (0.5): bars start here", ha="left", va="bottom", fontsize=9, color=INK2)
else:
    ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.text(0.5, len(STEPS) - 0.35, "chance", ha="center", va="bottom", fontsize=9, color=INK2)
ax.set_yticks(ys)
ax.set_yticklabels([s[0] for s in STEPS])
ax.set_xlim((0.5, 0.95) if BARS else (0.40, 0.95))
ax.set_ylim(-0.6, len(STEPS) - 0.1)
ax.set_xlabel("ID percentile (primary metric; 1 = own gene ranked first)" + (" — axis starts at chance, 0.5" if BARS else ", 0.5 = chance"))
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_title("A. ID percentile, one change per row (whisker = 95% CI)" if BARS else "A. ID percentile: each row adds one change (95% CI)", loc="left", fontsize=11, color=INK, pad=22)

# ---- panels B-D: top-1 / top-3 / top-10, same row order ----
for k, (m, title) in enumerate((("id_top1", "B. Top-1"), ("id_top3", "C. Top-3"), ("id_top10", "D. Top-10"))):
    bx = fig.add_subplot(gs[0, k + 2], sharey=ax, facecolor=SURF)
    for y, (_, arm, g) in zip(ys, STEPS):
        v, lo, hi = ci(arm, m)
        col = GROUPS[g][0]
        if BARS:
            bx.barh(y, v, height=0.62, color=col, edgecolor=SURF, linewidth=2, zorder=2)
            bx.plot([lo, hi], [y, y], color=INK, lw=1, zorder=3)
        else:
            bx.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
            bx.scatter([v], [y], s=60, color=col, edgecolor=SURF, linewidth=2, zorder=3)
        bx.text(min(hi + 0.02, 0.97), y + 0.28, f"{v:.2f}", ha="left", va="center", fontsize=8.5, color=INK2)
    bx.axvline(CHANCE[m], color=INK2, lw=1, ls=(0, (3, 3)))
    bx.set_xlim(0, 1)
    bx.set_xticks([0, 0.5, 1])
    bx.set_xticklabels(["0", "0.5", "1"])
    bx.set_title(title + f"  (chance {CHANCE[m]:.2f})", loc="left", fontsize=11, color=INK, pad=22)
    bx.grid(axis="x", color=GRID, lw=0.8)
    bx.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        bx.spines[sp].set_visible(False)
    bx.tick_params(axis="y", length=0, labelleft=False)

from matplotlib.patches import Patch
handles = ([Patch(facecolor=c, label=l) for c, l in GROUPS.values()] if BARS else
           [Line2D([0], [0], marker="o", color=c, lw=2, markersize=8, markeredgecolor=SURF, label=l)
            for c, l in GROUPS.values()])
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=10)
fig.text(0.01, 0.005,
         f"Replogle K562 CRISPRi, {N} VALIDATION genes (test genes untouched). Fine-tuned rows: 2 seeds averaged per gene; "
         "the last row averages the 2 seeds' predictions. Gene-vector baseline tunes 1 setting on these genes (slightly optimistic). "
         "Bold jump = largest single step.", fontsize=8.5, color=INK2)
os.makedirs("diag/results/figures", exist_ok=True)
out = "diag/results/figures/A25_ladder" + ("_bars" if BARS else "") + ".png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURF)
print(out)
