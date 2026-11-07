# Do our predictions carry functional information? (the clustering use case)

Measured 2026-11-07 on the Track B Adamson results (22 held-out perturbations, 5,060 genes,
split `5618e72828c1`). Reproduce with:

```bash
python scripts/clustering_structure.py reports/demo.html
```

Read-only analysis of predictions already saved by `aggregate`. No training was run.

---

## Why this test exists

The proposed application is **assigning function to genes with no literature**: predict a gene's
perturbation response, see which known genes it clusters with. Our existing metrics cannot tell us
whether that works.

The proof is the training-mean baseline. It has competitive delta RMSE (0.0588, better than either
fine-tuned model) **and predicts an identical profile for every perturbation.** Its clustering value
is exactly zero. So low error does not imply functional information — the two can be anti-correlated,
because hedging toward the average is rewarded by RMSE and fatal for clustering.

The property clustering needs is: *do predicted responses arrange perturbations the same way
measured responses do?*

---

## Results

| Method | shared component | raw *r* | centred *r* | *p* |
|---|---|---|---|---|
| scGPT + JEPA aux (ours) | 95% | −0.124 | **+0.259** | 0.003 |
| Linear (PCA baseline) | 99% | +0.002 | **+0.246** | 0.004 |
| scGPT (matched) | 94% | −0.119 | **+0.178** | 0.015 |
| Training mean | 100% | — | — | identical profile for every gene |
| No change | 100% | — | — | identical profile for every gene |
| **CEILING** (measured vs measured, disjoint gene halves) | — | — | **+0.965** | <0.0001 |

*r* = Pearson correlation between the predicted and measured perturbation-similarity matrices.
*p* = Mantel permutation test, 20,000 permutations of perturbation labels (the 231 pairwise
entries are not independent, so an ordinary Pearson p-value would be far too optimistic).

**Measured responses for reference:** mean pairwise cosine +0.514 (sd 0.464), with **40 of 231
pairs anti-correlated** — real perturbations produce genuinely diverse, sometimes opposing effects.
A shared "generic response" component accounts for 68% of a typical measured response.

---

## Three findings

### 1. Naive clustering on these predictions does not work
Raw *r* is ≈ 0 or slightly negative for every method. The reason is visible in the "shared
component" column: **94–99% of each predicted response is the same generic signature**, versus 68%
in reality. The models output "an average perturbation response, slightly tweaked." Cosine
similarity is then dominated by the part every prediction shares, and the gene-specific part is
swamped.

**Practical consequence: subtract the mean predicted response before clustering.** Without that
step the exercise returns nothing.

### 2. After centring, there is real but weak signal
All three models beat chance (*p* ≤ 0.015). But at *r* ≈ 0.18–0.26 against a ceiling of 0.965,
they capture roughly **a fifth to a quarter of the recoverable structure**. Enough to say the
information is there; not enough to claim a working function-assignment tool.

### 3. The method ranking flips — second independent evidence the metric is wrong
On delta RMSE: linear (0.0575) **beats** jepa (0.0723) and scgpt (0.0725).
On clustering structure: jepa (0.259) ≥ linear (0.246) > scgpt (0.178).

The ordering is not the same. A model can lose on error and carry more functional information.
This is a separate line of evidence from the saturated-Pearson finding, pointing the same way:
**delta RMSE is not measuring what the application needs.**

---

## Caveats

- **22 perturbations is small.** The gap between jepa (0.259) and linear (0.246) is almost
  certainly noise; jepa vs scgpt (0.178) may be too. Treat the ranking as suggestive, not
  established. What *is* established is that all three beat chance and all fall far below ceiling.
- All 22 are single-gene knockdowns from one pathway family in one cell line. Structure in a
  broader gene set may behave differently.
- The payload rounds values to 3 decimals. Effect sizes here are large enough that this cannot
  flip the conclusions, but exact figures may shift in the last digit.
- This measures agreement with *measured* structure. It does not yet compare against
  **co-essentiality clustering** (Wainberg 2021), which is the actual incumbent — see
  `docs/use_case_map.md` §4.

---

## What this means for the work

**For the pitch:** the clustering application is not dead, but it cannot be claimed as working.
The defensible statement is:

> Predicted responses do carry statistically significant functional information — but only after
> removing a generic component that makes up 95% of each prediction, and even then they recover
> about a quarter of what the measured data contains. We can measure the gap; we haven't closed it.

**For Track A (Vincent):** this is a cheap selection metric that complements RMSE. If a
kidney-pretrained Cell-JEPA checkpoint raises centred *r* toward the ceiling, that is a stronger
result than any RMSE improvement, and it directly supports the application story. The script runs
on any experiment's `demo_data.json`, including `adamson_trackA_v1`.

**The most promising lead:** the shared-component gap (94–99% predicted vs 68% measured) is a
concrete, specific failure mode. A model that produced responses with reality's diversity would
score far better here. That is a sharper target than "reduce RMSE."

---

# Addendum: precision@k — does the model pick the right perturbations to validate?

Measured 2026-11-07, same Track B predictions. Reproduce with:

```bash
python scripts/precision_at_k.py reports/demo.html
```

## Why this metric

Arrayed validation is robotics-bound: a lab follows up 20–100 perturbations, never thousands
(`docs/use_case_map.md` §8). So the decision-relevant question is not how accurate each predicted
value is, but **of the k perturbations ranked highest, how many are real hits?** Every wrong pick
is a wasted validation well. Perturbations are ranked by predicted effect magnitude against the
same quantity measured — *which knockdowns actually do something?*

## Results (22 held-out perturbations)

| Method | precision@5 (random 23%) | precision@10 (random 45%) | Spearman ρ |
|---|---|---|---|
| Linear (PCA baseline) | **4/5 = 80%**, p = 0.003 | 5/10 = 50%, p = 0.52 | **+0.289** |
| scGPT + JEPA aux | 1/5 = 20%, p = 0.77 | 5/10 = 50%, p = 0.52 | +0.182 |
| scGPT (matched) | 1/5 = 20%, p = 0.76 | 3/10 = 30%, p = 0.96 | −0.107 |
| Training mean | **cannot rank** | **cannot rank** | undefined |
| No change | **cannot rank** | **cannot rank** | undefined |

## Two findings

### 1. The baselines that beat us on RMSE cannot perform this task at all
Training-mean (delta RMSE 0.0588, better than either fine-tuned model) and no-change predict an
**identical profile for every perturbation**. Ranking them produces an all-tied order, which is not
a ranking. On the metric that maps to the lab's actual decision they score nothing.

This is the sharpest statement yet of the thesis: **delta RMSE rewarded methods that are useless
for the decision the application exists to support.** Three independent lines now point the same
way — the saturated Pearson, the clustering flip, and this.

### 2. The linear baseline wins again, and significantly
80% precision@5 at p = 0.003 is a real result — the PCA baseline puts 4 of the 5 biggest real
movers in its top 5. Both foundation models are at or below chance there.

Caveats: 22 perturbations is small; this is one of several metrics examined, so some multiple-
comparison discount applies; and the linear model's advantage fades by k = 10 (50%, not
significant), so it is good at the very top rather than across the whole ranking. The top is the
regime that matters, which makes the result more useful, not less.

**Untested alternative explanation:** the linear model's gene embedding may be ranking genes by
something as simple as their own expression level or variability, in which case an even cruder
heuristic would match it. Worth checking before crediting the model — it needs the control
expression profile, which is not in the demo payload.

## For Track A

The script is model-agnostic and runs on any experiment's `demo_data.json`. Whatever checkpoint
Vincent's fine-tuning produces can be scored on it immediately, and **a win here is worth more to
the application story than a win on delta RMSE** — it converts directly into validation wells saved.

The bar to beat is the linear baseline's 4/5, not the other foundation models.
