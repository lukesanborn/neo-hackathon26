"""Final-presentation figures 1-3 (Replogle K562, 103 held-out TEST genes; nothing here was used for selection).
  Fig 1  We beat PCA            -> diag/results/fig1_beat_pca.png
  Fig 2  What gets you there    -> diag/results/fig2_recipe.png
  Fig 3  JEPA does not help     -> diag/results/fig3_jepa.png
Palette (validated with dataviz validate_palette.js, light surface): teal #008a7e, purple #7d52ba,
terracotta #d9693f; neutral gray for baselines; teal / lavender tints from the method diagram.
  ../.venv/bin/python diag/make_figures.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

for f in ("Helvetica Neue", "Helvetica", "Arial"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break

TEAL, PURPLE, CORAL = "#008a7e", "#7d52ba", "#d9693f"
TEAL_TINT, LAV_TINT, CORAL_TINT = "#e3f1ee", "#efe9f6", "#fbeae3"
GRAY, GRAY_LIGHT = "#9aa1a3", "#d9dcdc"
INK, MUTED, GRID, BG = "#1d2b2e", "#5f6b6e", "#e8ecea", "#ffffff"
N = 103
RNG = np.random.default_rng(0)
IDX = RNG.integers(0, N, (5000, N))

T = json.load(open("diag/results/XC_replogle_test_retrieval_ci.json"))["arms"]
K = json.load(open("diag/results/X_id_percentile.json"))["replogle_trackA_lr3e5"]["res"]


def pg(arm, m="id_pct"):
    for k in T:
        if k == arm or k.endswith("/" + arm):
            return np.array(T[k]["per_gene"][m], float)
    raise KeyError(arm)


def ci_mean(x):
    b = x[IDX].mean(1)
    return x.mean(), np.quantile(b, 0.025), np.quantile(b, 0.975)


def style(ax, grid_axis="x"):
    ax.set_facecolor(BG)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=11, length=0)
    for a in ("x", "y") if grid_axis == "both" else (grid_axis,):
        getattr(ax, f"{a}axis").grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)


def title(fig, main, sub):
    fig.text(0.04, 0.955, main, fontsize=20, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.895, sub, fontsize=12.5, color=MUTED, va="top")


def footer(fig, text, y=0.02):
    import textwrap
    fig.text(0.04, y, "\n".join(textwrap.wrap(text, 175)), fontsize=9.5, color=MUTED, va="bottom")


BEST = "scgpt_paper_s2bscgpt_r1__ens2"
FOOT = ("Replogle K562 essential, 103 held-out test knockdowns (GEARS split d4237826c68c), never used for any "
        "model or setting choice. Whiskers/bands = 95% paired bootstrap CI over genes.")

# =============================================================================== Fig 1
models = [("PCA linear baseline", "baselines/linear", GRAY),
          ("scGPT gene vectors\n(no model)", "baselines/ridge_scgpt_gene_emb", PURPLE),
          ("Our model", BEST, TEAL)]
fig = plt.figure(figsize=(15, 6.6), facecolor=BG)
title(fig, "Our model identifies the knocked-down gene far better than PCA",
      "Given an unseen knockdown's measured response, rank all 103 candidate genes. Where does the true gene land?")
axA = fig.add_axes([0.13, 0.15, 0.36, 0.62])
axB = fig.add_axes([0.585, 0.15, 0.3, 0.62])

style(axA)
ys = np.arange(len(models))[::-1]
for y, (lab, arm, col) in zip(ys, models):
    m, lo, hi = ci_mean(pg(arm))
    axA.barh(y, m - 0.5, left=0.5, height=0.58, color=col, edgecolor=BG, lw=2)
    axA.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=1.3, capsize=5)
    axA.text(hi + 0.008, y, f"{m:.2f}", va="center", fontsize=15, fontweight="bold" if col == TEAL else "normal",
             color=INK)
axA.set_yticks(ys)
axA.set_yticklabels([m[0] for m in models], fontsize=13, color=INK)
axA.set_xlim(0.5, 0.9)
axA.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
axA.set_xticklabels(["0.5\nchance", "0.6", "0.7", "0.8", "0.9"])
axA.set_title("ID percentile  (1 = true gene ranked first)", fontsize=13, color=INK, loc="left", pad=12)
d, lo, hi = ci_mean(pg(BEST) - pg("baselines/linear"))
axA.text(0.5, -0.75, f"Our model − PCA: +{d:.2f}  [95% CI +{lo:.2f}, +{hi:.2f}]", fontsize=12, color=TEAL,
         fontweight="bold", transform=axA.get_yaxis_transform())

style(axB, "both")
ks = np.arange(1, 31)
axB.plot(ks, ks / N, color=GRAY_LIGHT, lw=2, ls=(0, (4, 3)))
axB.text(22, 22 / N - 0.045, "chance", color=MUTED, fontsize=10.5, ha="center", rotation=12)
for lab, arm, col in models:
    r = pg(arm, "id_rank")
    curve = np.array([(r <= k).mean() for k in ks])
    band = np.array([[(r[i] <= k).mean() for k in ks] for i in IDX[:1000]])
    axB.fill_between(ks, np.quantile(band, 0.025, 0), np.quantile(band, 0.975, 0), color=col, alpha=0.12, lw=0)
    axB.plot(ks, curve, color=col, lw=2.6 if col == TEAL else 2)
    axB.text(30.6, curve[-1], lab.split("\n")[0], color=col, fontsize=11, va="center", fontweight="bold")
for arm, col in ((BEST, TEAL), ("baselines/linear", GRAY)):
    v = (pg(arm, "id_rank") <= 10).mean()
    axB.plot([10], [v], "o", ms=8, color=col, mec=BG, mew=2, zorder=5)
    axB.text(10.6, v + (0.03 if col == TEAL else -0.055), f"{v:.0%}", color=col, fontsize=14, fontweight="bold")
axB.axvline(10, color=GRID, lw=1.2)
axB.set_xlim(1, 30)
axB.set_ylim(0, 0.8)
axB.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
axB.set_yticklabels(["0%", "20%", "40%", "60%", "80%"])
axB.set_xlabel("Shortlist size (top-k of 103 candidates)", fontsize=11.5, color=MUTED)
axB.set_title("True gene is on the shortlist", fontsize=13, color=INK, loc="left", pad=12)
footer(fig, FOOT + "  Our model = blood-adapted scGPT + contrastive retrieval loss, 2-seed ensemble.")
fig.savefig("diag/results/fig1_beat_pca.png", dpi=200)
plt.close(fig)

# =============================================================================== Fig 2
steps = [("PCA linear\nbaseline", "baselines/linear"),
         ("+ scGPT's\ngene knowledge", "baselines/ridge_scgpt_gene_emb"),
         ("+ fine-tune the\nfull model", "scgpt_paper_s2pub_r0"),
         ("+ contrastive\nretrieval loss", "scgpt_paper_s2pub_r1"),
         ("+ blood\npretraining", "scgpt_paper_s2bscgpt_r1"),
         ("+ average\n2 seeds", BEST)]
vals = [pg(a).mean() for _, a in steps]
fig = plt.figure(figsize=(15, 6.8), facecolor=BG)
title(fig, "What gets you there: one change at a time",
      "Each column adds one change to the column before it. ID percentile on the 103 test genes (0.5 = chance).")
ax = fig.add_axes([0.06, 0.2, 0.9, 0.6])
style(ax, "y")
xs = np.arange(len(steps) + 1)
w = 0.62
ax.bar(0, vals[0] - 0.5, bottom=0.5, width=w, color=GRAY, edgecolor=BG, lw=2)
ax.text(0, vals[0] + 0.006, f"{vals[0]:.3f}", ha="center", fontsize=13, color=INK)
for i in range(1, len(steps)):
    lo_, hi_ = sorted((vals[i - 1], vals[i]))
    up = vals[i] >= vals[i - 1]
    ax.bar(i, hi_ - lo_, bottom=lo_, width=w, color=TEAL_TINT if up else CORAL_TINT,
           edgecolor=TEAL if up else CORAL, lw=1.6)
    ax.plot([i - 1 + w / 2, i - w / 2], [vals[i - 1]] * 2, color=MUTED, lw=1, ls=(0, (2, 2)))
    ax.text(i, hi_ + 0.006, f"{vals[i] - vals[i - 1]:+.3f}", ha="center", fontsize=14, fontweight="bold",
            color=TEAL if up else CORAL)
m, lo, hi = ci_mean(pg(BEST))
xf = len(steps)
ax.plot([len(steps) - 1 + w / 2, xf - w / 2], [vals[-1]] * 2, color=MUTED, lw=1, ls=(0, (2, 2)))
ax.bar(xf, m - 0.5, bottom=0.5, width=w, color=TEAL, edgecolor=BG, lw=2)
ax.errorbar(xf, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=1.3, capsize=5)
ax.text(xf, hi + 0.008, f"{m:.3f}", ha="center", fontsize=15, fontweight="bold", color=INK)
ax.set_xticks(xs)
ax.set_xticklabels([s[0] for s in steps] + ["Our model"], fontsize=12, color=INK)
ax.get_xticklabels()[-1].set_fontweight("bold")
ax.set_ylim(0.5, 0.86)
ax.set_yticks([0.5, 0.6, 0.7, 0.8])
ax.set_yticklabels(["0.5 chance", "0.6", "0.7", "0.8"])
ax.annotate(f"+{vals[-1] - vals[0]:.2f} over PCA", xy=(xf - w / 2 - 0.05, 0.80), xytext=(xf - 1.35, 0.835),
            fontsize=13.5, color=TEAL, fontweight="bold", ha="center", va="center",
            arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.4))
footer(fig, FOOT.replace("Whiskers/bands", "Whisker") + "  Steps are single changes; individual steps are not "
       "each significant, the total is.")
fig.savefig("diag/results/fig2_recipe.png", dpi=200)
plt.close(fig)

# =============================================================================== Fig 3
CFG = [("scgpt_paper", 0), ("scgpt_paper", 1), ("celljepa", 0), ("celljepa", 1)]


def avg4(start):
    return np.mean([pg(f"{m}_s2{start}_r{r}") for m, r in CFG], axis=0)


def kid(name):
    return np.array(K[f"{name} | scgpt_paper | forward-raw"]["per_gene"], float)[:, 0] / 100


rows = [  # (label, diff per gene, colour, group)
    ("Whole-cell JEPA (blood)", avg4("bjepa") - avg4("bscgpt"), PURPLE, 0),
    ("Dual mask acc (blood)", avg4("bijepa") - avg4("bscgpt"), CORAL, 0),
    ("Cell-JEPA, as in the paper (kidney)", kid("kidney JEPA faithful") - kid("kidney scGPT"), PURPLE, 0),
    ("Dual mask acc (kidney)", kid("X1 ijepa") - kid("kidney scGPT"), CORAL, 0),
    ("JEPA term in fine-tuning", np.mean([pg(f"celljepa_s2{s}_r{r}") - pg(f"scgpt_paper_s2{s}_r{r}")
                                         for s in ("pub", "bscgpt") for r in (0, 1)], axis=0), PURPLE, 0),
    ("Dual mask acc vs Cell-JEPA (kidney)", kid("X1 ijepa") - kid("kidney JEPA faithful"), CORAL, 1),
    ("Dual mask acc vs whole-cell JEPA (blood)", avg4("bijepa") - avg4("bjepa"), CORAL, 1),
]
fig = plt.figure(figsize=(15, 7.4), facecolor=BG)
title(fig, "JEPA does not improve gene identification",
      "Matched comparisons: same starting model, data and training steps. Only the JEPA objective changes.")
ax = fig.add_axes([0.33, 0.14, 0.62, 0.66])
style(ax)
gap = 0.9
yv, yl = [], []
y = 0.0
for i, (lab, d, col, g) in enumerate(rows):
    if i and g != rows[i - 1][3]:
        y -= gap
    yv.append(y)
    y -= 1
ax.axvspan(-0.1, 0, color=CORAL_TINT, alpha=0.5, lw=0)
ax.axvspan(0, 0.1, color=TEAL_TINT, alpha=0.6, lw=0)
ax.axvline(0, color=INK, lw=1.2)
for yy, (lab, d, col, g) in zip(yv, rows):
    m, lo, hi = ci_mean(d)
    ax.plot([lo, hi], [yy, yy], color=col, lw=3, solid_capstyle="round")
    ax.plot(m, yy, "o", ms=12, color=col, mec=BG, mew=2.5, zorder=5)
    sig = lo > 0 or hi < 0
    ax.text(0.081, yy, f"{m:+.3f}" + ("" if sig else "  n.s."), va="center", fontsize=12.5, color=INK,
            fontweight="bold" if sig else "normal")
ax.set_yticks(yv)
ax.set_yticklabels([r[0] for r in rows], fontsize=12.5, color=INK)
ax.set_xlim(-0.1, 0.1)
ax.set_xticks([-0.08, -0.04, 0, 0.04, 0.08])
ax.set_xlabel("Change in ID percentile from adding JEPA  (test set, 103 genes)", fontsize=11.5, color=MUTED)
ax.text(-0.097, yv[0] + 0.75, "\u2190 JEPA hurts", color=CORAL, fontsize=12, fontweight="bold", family="DejaVu Sans")
ax.text(0.097, yv[0] + 0.75, "JEPA helps \u2192", color=TEAL, fontsize=12, fontweight="bold", ha="right",
        family="DejaVu Sans")
gy = [yv[0] + 0.55, yv[5] + 0.55]
fig.text(0.04, ax.transData.transform((0, yv[0] + 0.4))[1] / fig.dpi / fig.get_figheight(),
         "vs the same model without JEPA", fontsize=11, color=MUTED, fontstyle="italic")
fig.text(0.04, ax.transData.transform((0, yv[5] + 0.4))[1] / fig.dpi / fig.get_figheight(),
         "our JEPA vs the paper's JEPA", fontsize=11, color=MUTED, fontstyle="italic")
handles = [plt.Line2D([], [], marker="o", ls="", ms=11, color=c) for c in (PURPLE, CORAL)]
ax.legend(handles, ["Whole-cell JEPA (paper style)", "Dual mask acc (our gene-level JEPA)"], loc="center right",
          frameon=False, fontsize=11.5, ncol=2, bbox_to_anchor=(0.1, yv[4] - (1 + gap) / 2),
          bbox_transform=ax.transData)
footer(fig, FOOT.replace("Whiskers/bands", "Lines") + "  Blood rows: mean over 8 fine-tune variants; kidney rows: "
       "3 seeds; n.s. = CI includes 0.")
fig.savefig("diag/results/fig3_jepa.png", dpi=200)
plt.close(fig)
print("saved fig1_beat_pca.png, fig2_recipe.png, fig3_jepa.png")
