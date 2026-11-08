# Pretraining JEPA knobs (screen tier) — Adamson 22 held-out genes

Every arm: continued pretraining from public `scGPT_human` on the kidney cache (3 epochs x 48,000 cells, lr 3e-05, bs 32, seed 0), then Cell-JEPA fine-tuning on Adamson (15 epochs, seeds 1). Only the knob differs.

`spread` is the teacher's `<cls>` spread at the end of pretraining — the collapse diagnostic. A19: ~0.003 means the JEPA term has nothing cell-specific left to teach; 0.03-0.04 is healthy.
`mask` is the realised fraction of gene tokens masked (unit masks overshoot the target ratio).

## Reference arms

| arm | what changed | mask | teacher spread | val ΔRMSE (7 genes) | **test ΔRMSE (22 genes)** | vs baseline | test Δ-Pearson |
|---|---|---|---|---|---|---|---|
| `base_center` | JEPA baseline: random 15%, full-cell centred target (A19's working setting) | 0.15 | 0.0177 | - | **-** | - | - |
| `no_jepa` | no JEPA term at all: matched reconstruction-only control | 0.15 | - | - | **-** | - | - |

## 2a — what unit gets masked

| arm | what changed | mask | teacher spread | val ΔRMSE (7 genes) | **test ΔRMSE (22 genes)** | vs baseline | test Δ-Pearson |
|---|---|---|---|---|---|---|---|
| `unit_program` | whole co-expression programs | 0.21 | 0.0197 | - | **-** | - | - |
| `unit_pathway` | whole curated pathways (MSigDB Hallmark) | 0.16 | 0.0167 | - | **-** | - | - |
| `unit_topexpr` | the cell's highest-expressed genes | 0.15 | 0.0177 | - | **-** | - | - |

## 2b — how much gets masked

| arm | what changed | mask | teacher spread | val ΔRMSE (7 genes) | **test ΔRMSE (22 genes)** | vs baseline | test Δ-Pearson |
|---|---|---|---|---|---|---|---|
| `ratio_30` | random 0.3 | 0.30 | 0.0190 | - | **-** | - | - |
| `ratio_50` | random 0.5 | 0.50 | 0.0206 | - | **-** | - | - |
| `ratio_80` | expected to be too hard; included to find the ceiling | 0.80 | 0.0230 | - | **-** | - | - |

## 2c — what the target is

| arm | what changed | mask | teacher spread | val ΔRMSE (7 genes) | **test ΔRMSE (22 genes)** | vs baseline | test Δ-Pearson |
|---|---|---|---|---|---|---|---|
| `target_token` | predict only the masked genes' embeddings (I-JEPA style) | 0.15 | 0.0066 | - | **-** | - | - |
| `target_faithful` | paper's full-cell target; A19 predicts collapse here | 0.15 | 0.0034 | - | **-** | - | - |
| `target_token_program` | combines the two most promising: program masking + masked-part target | 0.21 | 0.0058 | - | **-** | - | - |

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
