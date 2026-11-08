# Pretraining JEPA knobs (long tier) — adamson held-out genes

Every arm: continued pretraining from public `scGPT_human` on the kidney cache (4 epochs x the full corpus, lr 3e-05, bs 32, seed 0), then Cell-JEPA fine-tuning on Adamson (15 epochs, seeds 1). Only the knob differs.

`spread` is the teacher's `<cls>` spread at the end of pretraining — the collapse diagnostic. A19: ~0.003 means the JEPA term has nothing cell-specific left to teach; 0.03-0.04 is healthy.
`mask` is the realised fraction of gene tokens masked (unit masks overshoot the target ratio).

## Reference arms

| arm | what changed | mask | teacher spread | val ΔRMSE (7 genes) | **test ΔRMSE (22 genes)** | vs baseline | test Δ-Pearson |
|---|---|---|---|---|---|---|---|
| `base_center` | JEPA baseline: random 15%, full-cell centred target (A19's working setting) | 0.15 | 0.0275 | 0.0747 | **0.0709** | - | 0.618 |
| `no_jepa` | no JEPA term at all: matched reconstruction-only control | 0.15 | - | 0.0742 | **0.0684** | -0.0025 | 0.657 |

## 2a — what unit gets masked

| arm | what changed | mask | teacher spread | val ΔRMSE (7 genes) | **test ΔRMSE (22 genes)** | vs baseline | test Δ-Pearson |
|---|---|---|---|---|---|---|---|
| `unit_program` | whole co-expression programs | 0.21 | 0.0274 | 0.0763 | **0.0724** | +0.0015 | 0.614 |
| `unit_pathway` | whole curated pathways (MSigDB Hallmark) | 0.16 | 0.0276 | 0.0749 | **0.0739** | +0.0030 | 0.609 |
| `unit_topexpr` | the cell's highest-expressed genes | 0.15 | 0.0274 | 0.0817 | **0.0734** | +0.0025 | 0.605 |

## 2b — how much gets masked

| arm | what changed | mask | teacher spread | val ΔRMSE (7 genes) | **test ΔRMSE (22 genes)** | vs baseline | test Δ-Pearson |
|---|---|---|---|---|---|---|---|
| `ratio_30` | random 0.3 | 0.30 | 0.0270 | 0.0756 | **0.0745** | +0.0036 | 0.604 |
| `ratio_50` | random 0.5 | 0.50 | 0.0287 | 0.0752 | **0.0729** | +0.0020 | 0.626 |
| `ratio_80` | expected to be too hard; included to find the ceiling | 0.80 | 0.0284 | 0.0755 | **0.0712** | +0.0003 | 0.617 |

## 2c — what the target is

| arm | what changed | mask | teacher spread | val ΔRMSE (7 genes) | **test ΔRMSE (22 genes)** | vs baseline | test Δ-Pearson |
|---|---|---|---|---|---|---|---|
| `target_token` | predict only the masked genes' embeddings (I-JEPA style) | 0.15 | 0.0203 | 0.0726 | **0.0716** | +0.0006 | 0.614 |
| `target_faithful` | paper's full-cell target; A19 predicts collapse here | 0.15 | 0.0129 | 0.0750 | **0.0696** | -0.0013 | 0.642 |
| `target_token_program` | combines the two most promising: program masking + masked-part target | 0.21 | 0.0198 | 0.0761 | **0.0707** | -0.0002 | 0.627 |

## Reference points (not run here)

| predictor | delta RMSE | identification top-1 |
|---|---|---|
| linear | 0.0575 | 0.091 |
| training mean | 0.0588 | 0.045 |
| no change | 0.0915 | 0.000 |

Lower delta RMSE is better. No deep arm has yet beaten the linear baseline (0.0575) on this metric; `docs/clustering_finding.md` argues delta RMSE is not what the application needs, so treat identification as the metric that matches the CRISPR-targeting goal.

Identification scores for these arms: run

    .venv/bin/modal run diag/retrieval.py --exps adamson_knobs_v1 --tag model_knobs

which reads each run's saved predictions and reports top-1 / mean rank with bootstrap CIs (A17).
