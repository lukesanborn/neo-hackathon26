"""Replicates the team's ladder chart (ID percentile + top-1/3/10) on the 103 TEST genes of Replogle K562,
adding the dual-mask (X1) runs.   ../.venv/bin/python diag/plot_test_chart.py
Reads diag/results/XC_replogle_test_retrieval_ci.json -> diag/results/XC_test_chart.png"""
import json

import matplotlib.pyplot as plt

A = json.load(open("diag/results/XC_replogle_test_retrieval_ci.json"))["arms"]
N = 103
GRAY, BLUE, ORANGE, GREEN, PURPLE = "#8a8a85", "#3b74d4", "#d9622b", "#2b8a5e", "#8b5cc7"
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e3"


def arm(name):
    for k in A:
        if k == name or k.endswith("/" + name):
            return A[k]["ci"]
    raise KeyError(name)


# (label, arm, colour, jump reference index or None, section)
ROWS = [
    ("Training mean (same prediction for every gene)", "baselines/training_mean", None, None),
    ("Baseline: linear (PCA gene embedding)", "baselines/linear", GRAY, 0),
    ("scGPT gene vectors only (no model)", "baselines/ridge_scgpt_gene_emb", BLUE, 1),
    ("Public scGPT, fine-tuned", "scgpt_paper_s2pub_r0", ORANGE, 2),
    ("+ contrastive retrieval loss", "scgpt_paper_s2pub_r1", ORANGE, 3),
    ("+ blood pretraining (blood-scGPT)", "scgpt_paper_s2bscgpt_r1", ORANGE, 4),
    ("+ JEPA in pretraining (blood-JEPA)", "scgpt_paper_s2bjepa_r1", GREEN, 5),
    ("+ JEPA in fine-tuning (Cell-JEPA)", "celljepa_s2bjepa_r1", GREEN, 6),
    ("+ average 2 seeds", "celljepa_s2bjepa_r1__ens2", GREEN, 7),
    # ---- new runs (this session) ----
    ("Dual mask acc pretraining, blood (swap for row 8's JEPA)", "celljepa_s2bijepa_r1", PURPLE, 7),
    ("+ average 2 seeds", "celljepa_s2bijepa_r1__ens2", PURPLE, 9),
    ("Dual mask acc pretraining, whole-body, avg 2 seeds", "celljepa_s2wijepa_r1__ens2", PURPLE, 10),
    ("Public scGPT, bins50 inputs + retrieval loss, avg 2 seeds", "scgpt_paper_s2pubbins_r1__ens2", ORANGE, 8),
    ("Blood-scGPT + retrieval loss, avg 2 seeds (best on test)", "scgpt_paper_s2bscgpt_r1__ens2", ORANGE, 8),
]
SPLIT = 9  # rows >= SPLIT are the new runs
REF_NOTE = {9: "vs row 8", 12: "vs row 9", 13: "vs row 9"}

vals = [arm(a) for _, a, _, _ in ROWS]
n = len(ROWS)
ys = list(range(n))[::-1]

fig = plt.figure(figsize=(20, 0.62 * n + 2.4), facecolor="#fcfcfb")
gs = fig.add_gridspec(1, 5, width_ratios=[4.2, 0.75, 1.35, 1.35, 1.35], wspace=0.08,
                      left=0.235, right=0.985, top=0.88, bottom=0.1)
axA = fig.add_subplot(gs[0])
axJ = fig.add_subplot(gs[1], sharey=axA)
axs = [fig.add_subplot(gs[i], sharey=axA) for i in (2, 3, 4)]

for ax in [axA, axJ] + axs:
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)

# separator between the team's ladder and the new runs
for ax in [axA, axJ] + axs:
    ax.axhline(ys[SPLIT] + 0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)))
axA.text(0.94, ys[SPLIT] + 0.5, "new runs (this session) ↓", fontsize=9.5, color=MUTED, va="bottom", ha="right",
         backgroundcolor="#fcfcfb")

# Panel A: ID percentile, bars start at chance
for y, c, (_, _, col, _) in zip(ys, vals, ROWS):
    m, lo, hi = c["id_pct"]["mean"], c["id_pct"]["lo"], c["id_pct"]["hi"]
    if col is not None:
        axA.barh(y, m - 0.5, left=0.5, height=0.62, color=col, edgecolor="#fcfcfb", linewidth=2)
    axA.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=1.2, capsize=5)
    axA.text(hi + 0.004, y, f"{m:.3f}", va="center", fontsize=11, color=INK)
axA.set_xlim(0.5, 0.95)
axA.set_xticks([0.5 + 0.05 * i for i in range(10)])
axA.xaxis.grid(True, color=GRID, lw=0.8)
axA.set_axisbelow(True)
axA.set_yticks(ys)
axA.set_yticklabels([r[0] for r in ROWS], fontsize=11.5, color=INK)
axA.axvline(0.5, color=MUTED, lw=1)
axA.text(0.502, ys[0] + 0.55, "chance (0.5): bars start here", fontsize=9.5, color=MUTED)
axA.set_xlabel("ID percentile on 103 held-out TEST genes (1 = own gene ranked first); axis starts at chance, 0.5",
               fontsize=10.5, color=MUTED)
axA.set_title("A. ID percentile, test set (whisker = 95% CI)", fontsize=13, color=INK, loc="left")

# Jump column
plt.setp(axJ.get_yticklabels(), visible=False)
axJ.tick_params(axis="y", length=0)
axJ.set_xlim(0, 1)
axJ.set_xticks([])
axJ.spines["bottom"].set_visible(False)
axJ.set_title("Jump vs\nrow above", fontsize=11, color=MUTED)
for i, (y, (_, _, _, ref)) in enumerate(zip(ys, ROWS)):
    if ref is None:
        continue
    d = vals[i]["id_pct"]["mean"] - vals[ref]["id_pct"]["mean"]
    big = abs(d) >= 0.05
    axJ.text(0.5, y + (0.12 if i in REF_NOTE else 0), f"{d:+.3f}", ha="center", va="center", fontsize=12,
             color=INK, fontweight="bold" if big else "normal")
    if i in REF_NOTE:
        axJ.text(0.5, y - 0.26, REF_NOTE[i], ha="center", va="center", fontsize=8.5, color=MUTED)

# Panels B-D: top-k with chance lines at test-set levels
for ax, key, k, letter in zip(axs, ("id_top1", "id_top3", "id_top10"), (1, 3, 10), "BCD"):
    chance = k / N
    for y, c, (_, _, col, _) in zip(ys, vals, ROWS):
        m, lo, hi = c[key]["mean"], c[key]["lo"], c[key]["hi"]
        ax.barh(y, m, height=0.62, color=col or GRAY, edgecolor="#fcfcfb", linewidth=2,
                alpha=1.0 if col else 0.55)
        ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
        ax.text(hi + 0.02, y + 0.22, f"{m:.2f}", fontsize=9.5, color=MUTED)
    ax.axvline(chance, color=INK, lw=1, ls=(0, (3, 3)))
    ax.set_xlim(0, 0.8)
    ax.set_xticks([0, 0.25, 0.5, 0.75])
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_title(f"{letter}. Top-{k}  (chance {chance:.3f})", fontsize=12.5, color=INK, loc="left")

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GRAY, BLUE, ORANGE, GREEN, PURPLE)]
fig.legend(handles, ["Baseline (no deep model)", "scGPT knowledge, no fine-tuning", "scGPT fine-tuned",
                     "+ JEPA (whole-cell)", "+ dual mask acc (gene-level JEPA)"],
           loc="upper center", ncol=5, frameon=False, fontsize=11.5, bbox_to_anchor=(0.6, 0.96))
fig.text(0.235, 0.012, "Replogle K562 essential, GEARS split d4237826c68c: 103 test knockdowns, never used for any "
         "choice.\nFine-tune configs as in the team's validation chart; 2 seeds averaged per gene unless 'average "
         "2 seeds' (= ensemble of the two seeds' predictions).", fontsize=9, color=MUTED)
fig.savefig("diag/results/XC_test_chart.png", dpi=150)
print("saved diag/results/XC_test_chart.png")
