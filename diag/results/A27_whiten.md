# A27: is JEPA's knockdown signal squashed or gone? (real Replogle cells, split-half ceiling)

Readouts fit on control cells or TRAINING genes only; hyperparameter chosen on validation genes; test scored once.
Test: 103 genes (chance id_pct 0.5, top-1 0.010). CIs: bootstrap over test genes.

| Space | Readout | chosen | val id_pct | test id_pct [95% CI] | test top-1 | top-3 | top-10 |
|---|---|---|---|---|---|---|---|
| public scGPT | raw | - | 0.937 | 0.931 [0.897, 0.959] | 0.612 | 0.748 | 0.835 |
| public scGPT | ctrl_pcK | K=30 | 0.984 | 0.962 [0.936, 0.982] | 0.641 | 0.806 | 0.893 |
| public scGPT | ctrl_white | lam=0.01 | 0.996 | 0.986 [0.978, 0.992] | 0.689 | 0.835 | 0.942 |
| public scGPT | within_white | lam=0.001 | 0.992 | 0.991 [0.987, 0.995] | 0.718 | 0.854 | 0.990 |
| public scGPT | lda | lam=0.01 | 0.995 | 0.993 [0.989, 0.996] | 0.748 | 0.913 | 0.981 |
| blood-scGPT | raw | - | 0.901 | 0.901 [0.858, 0.938] | 0.534 | 0.680 | 0.786 |
| blood-scGPT | ctrl_pcK | K=10 | 0.982 | 0.968 [0.948, 0.983] | 0.631 | 0.757 | 0.932 |
| blood-scGPT | ctrl_white | lam=0.001 | 0.987 | 0.973 [0.956, 0.986] | 0.670 | 0.825 | 0.913 |
| blood-scGPT | within_white | lam=0.01 | 0.987 | 0.986 [0.978, 0.993] | 0.738 | 0.864 | 0.961 |
| blood-scGPT | lda | lam=0.1 | 0.986 | 0.992 [0.987, 0.996] | 0.718 | 0.913 | 0.981 |
| blood-JEPA (center) | raw | - | 0.682 | 0.677 [0.625, 0.728] | 0.068 | 0.126 | 0.291 |
| blood-JEPA (center) | ctrl_pcK | K=30 | 0.755 | 0.713 [0.653, 0.768] | 0.117 | 0.233 | 0.388 |
| blood-JEPA (center) | ctrl_white | lam=0.01 | 0.762 | 0.755 [0.699, 0.809] | 0.107 | 0.184 | 0.476 |
| blood-JEPA (center) | within_white | lam=0.01 | 0.770 | 0.758 [0.700, 0.813] | 0.155 | 0.262 | 0.456 |
| blood-JEPA (center) | lda | lam=0.01 | 0.790 | 0.778 [0.727, 0.830] | 0.155 | 0.272 | 0.456 |

## Paired differences in test id_pct (A − B) [95% CI]

| readout | A | B | diff |
|---|---|---|---|
| raw | blood-JEPA (center) | blood-scGPT | -0.224 [-0.283, -0.164] |
| raw | blood-JEPA (center) | public scGPT | -0.253 [-0.305, -0.203] |
| ctrl_pcK | blood-JEPA (center) | blood-scGPT | -0.255 [-0.310, -0.204] |
| ctrl_pcK | blood-JEPA (center) | public scGPT | -0.249 [-0.307, -0.195] |
| ctrl_white | blood-JEPA (center) | blood-scGPT | -0.218 [-0.272, -0.170] |
| ctrl_white | blood-JEPA (center) | public scGPT | -0.231 [-0.285, -0.180] |
| within_white | blood-JEPA (center) | blood-scGPT | -0.229 [-0.285, -0.175] |
| within_white | blood-JEPA (center) | public scGPT | -0.234 [-0.291, -0.179] |
| lda | blood-JEPA (center) | blood-scGPT | -0.213 [-0.263, -0.165] |
| lda | blood-JEPA (center) | public scGPT | -0.215 [-0.264, -0.165] |
| ctrl_white - raw | public scGPT | public scGPT | +0.055 [+0.029, +0.087] |
| within_white - raw | public scGPT | public scGPT | +0.061 [+0.033, +0.093] |
| lda - raw | public scGPT | public scGPT | +0.062 [+0.034, +0.095] |
| ctrl_white - raw | blood-scGPT | blood-scGPT | +0.072 [+0.039, +0.110] |
| within_white - raw | blood-scGPT | blood-scGPT | +0.085 [+0.050, +0.126] |
| lda - raw | blood-scGPT | blood-scGPT | +0.091 [+0.055, +0.133] |
| ctrl_white - raw | blood-JEPA (center) | blood-JEPA (center) | +0.078 [+0.022, +0.131] |
| within_white - raw | blood-JEPA (center) | blood-JEPA (center) | +0.081 [+0.023, +0.137] |
| lda - raw | blood-JEPA (center) | blood-JEPA (center) | +0.101 [+0.046, +0.154] |

## Geometry of each space (control cells; training knockdowns)

| Space | effective rank (controls) | top-10 PCs share of variance | between-knockdown / within-knockdown variance | per-cell spread |
|---|---|---|---|---|
| public scGPT | 12.6 | 0.84 | 0.259 | 0.0061 |
| blood-scGPT | 6.7 | 0.89 | 0.280 | 0.0021 |
| blood-JEPA (center) | 6.8 | 0.95 | **0.033** | **0.0247** |

Not dimensional collapse (same effective rank as blood-scGPT). JEPA spreads individual cells out ~10x more, while
knockdown means separate ~8x less relative to that cell-to-cell spread. Linear readouts recover part (+0.10) but
JEPA stays ~0.21 below both scGPT spaces under every readout. Both blood models started from public scGPT (0.931 raw):
2 epochs of plain scGPT continued pretraining keep most of it (0.901); 2 epochs with the JEPA `center` term drop it to 0.677.
