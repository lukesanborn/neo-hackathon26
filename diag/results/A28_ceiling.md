# A28: perturbation-pair JEPA views — does knockdown information survive? (real Replogle cells, split-half ceiling)

Four arms continued from public scGPT_human on the SAME cache (Replogle TRAINING knockdowns + controls, 50,550 cells),
8 epochs, lr 3e-5, bs 32, seed 0, identical order/masks. Only the objective or the JEPA views differ:
`pv_no_jepa` reconstruction only (the control) | `pv_same_cell` JEPA center, Cell-JEPA views (masked cell vs the same full cell)
| `pv_pair1` / `pv_pair4` JEPA center, teacher sees 1 / 4 OTHER cells of the same knockdown.

Readouts fit on control cells or TRAINING genes only; hyperparameter chosen on validation genes; test scored once.
Test: 103 genes (chance id_pct 0.5, top-1 0.010). CIs: bootstrap over test genes.

| Space | Readout | chosen | val id_pct | test id_pct [95% CI] | test top-1 | top-3 | top-10 |
|---|---|---|---|---|---|---|---|
| pv_no_jepa | raw | - | 0.956 | 0.958 [0.939, 0.975] | 0.612 | 0.709 | 0.864 |
| pv_no_jepa | ctrl_pcK | K=10 | 0.991 | 0.986 [0.978, 0.992] | 0.670 | 0.825 | 0.971 |
| pv_no_jepa | ctrl_white | lam=1 | 0.996 | 0.991 [0.982, 0.996] | 0.796 | 0.922 | 0.971 |
| pv_no_jepa | within_white | lam=1 | 0.995 | 0.991 [0.985, 0.996] | 0.796 | 0.922 | 0.971 |
| pv_no_jepa | lda | lam=0.001 | 0.995 | 0.995 [0.992, 0.997] | 0.786 | 0.932 | 1.000 |
| pv_same_cell | raw | - | 0.856 | 0.893 [0.854, 0.928] | 0.408 | 0.583 | 0.738 |
| pv_same_cell | ctrl_pcK | K=10 | 0.954 | 0.934 [0.907, 0.957] | 0.427 | 0.602 | 0.786 |
| pv_same_cell | ctrl_white | lam=0.001 | 0.965 | 0.955 [0.934, 0.972] | 0.553 | 0.718 | 0.874 |
| pv_same_cell | within_white | lam=0.1 | 0.955 | 0.974 [0.963, 0.984] | 0.592 | 0.767 | 0.913 |
| pv_same_cell | lda | lam=0.01 | 0.983 | 0.980 [0.969, 0.989] | 0.650 | 0.806 | 0.942 |
| pv_pair1 | raw | - | 0.930 | 0.936 [0.901, 0.967] | 0.524 | 0.738 | 0.845 |
| pv_pair1 | ctrl_pcK | K=10 | 0.977 | 0.981 [0.970, 0.989] | 0.670 | 0.806 | 0.951 |
| pv_pair1 | ctrl_white | lam=1 | 0.976 | 0.985 [0.977, 0.992] | 0.670 | 0.845 | 0.951 |
| pv_pair1 | within_white | lam=0.01 | 0.995 | 0.994 [0.991, 0.997] | 0.757 | 0.932 | 1.000 |
| pv_pair1 | lda | lam=0.01 | 0.994 | 0.990 [0.983, 0.996] | 0.796 | 0.932 | 0.961 |
| pv_pair4 | raw | - | 0.954 | 0.952 [0.926, 0.974] | 0.563 | 0.699 | 0.854 |
| pv_pair4 | ctrl_pcK | K=10 | 0.985 | 0.973 [0.953, 0.988] | 0.709 | 0.806 | 0.932 |
| pv_pair4 | ctrl_white | lam=0.1 | 0.988 | 0.991 [0.986, 0.994] | 0.748 | 0.854 | 0.971 |
| pv_pair4 | within_white | lam=0.01 | 0.994 | 0.995 [0.992, 0.997] | 0.806 | 0.942 | 1.000 |
| pv_pair4 | lda | lam=0.01 | 0.996 | 0.995 [0.993, 0.998] | 0.806 | 0.961 | 0.990 |

## Paired differences in test id_pct (A − B) [95% CI]

| readout | A | B | diff |
|---|---|---|---|
| raw | pv_same_cell | pv_no_jepa | -0.065 [-0.098, -0.035] |
| raw | pv_pair1 | pv_no_jepa | -0.023 [-0.056, +0.008] |
| raw | pv_pair4 | pv_no_jepa | -0.007 [-0.034, +0.017] |
| ctrl_pcK | pv_same_cell | pv_no_jepa | -0.052 [-0.077, -0.029] |
| ctrl_pcK | pv_pair1 | pv_no_jepa | -0.005 [-0.016, +0.005] |
| ctrl_pcK | pv_pair4 | pv_no_jepa | -0.012 [-0.033, +0.003] |
| ctrl_white | pv_same_cell | pv_no_jepa | -0.036 [-0.057, -0.018] |
| ctrl_white | pv_pair1 | pv_no_jepa | -0.006 [-0.015, +0.005] |
| ctrl_white | pv_pair4 | pv_no_jepa | -0.000 [-0.006, +0.008] |
| within_white | pv_same_cell | pv_no_jepa | -0.017 [-0.027, -0.009] |
| within_white | pv_pair1 | pv_no_jepa | +0.003 [-0.002, +0.009] |
| within_white | pv_pair4 | pv_no_jepa | +0.004 [-0.000, +0.009] |
| lda | pv_same_cell | pv_no_jepa | -0.015 [-0.025, -0.006] |
| lda | pv_pair1 | pv_no_jepa | -0.004 [-0.011, +0.002] |
| lda | pv_pair4 | pv_no_jepa | +0.001 [-0.002, +0.004] |
| ctrl_white - raw | pv_no_jepa | pv_no_jepa | +0.032 [+0.017, +0.049] |
| within_white - raw | pv_no_jepa | pv_no_jepa | +0.033 [+0.018, +0.050] |
| lda - raw | pv_no_jepa | pv_no_jepa | +0.036 [+0.020, +0.055] |
| ctrl_white - raw | pv_same_cell | pv_same_cell | +0.061 [+0.030, +0.096] |
| within_white - raw | pv_same_cell | pv_same_cell | +0.081 [+0.050, +0.116] |
| lda - raw | pv_same_cell | pv_same_cell | +0.087 [+0.054, +0.125] |
| ctrl_white - raw | pv_pair1 | pv_pair1 | +0.049 [+0.022, +0.081] |
| within_white - raw | pv_pair1 | pv_pair1 | +0.059 [+0.027, +0.094] |
| lda - raw | pv_pair1 | pv_pair1 | +0.055 [+0.027, +0.087] |
| ctrl_white - raw | pv_pair4 | pv_pair4 | +0.039 [+0.017, +0.065] |
| within_white - raw | pv_pair4 | pv_pair4 | +0.043 [+0.022, +0.070] |
| lda - raw | pv_pair4 | pv_pair4 | +0.044 [+0.023, +0.069] |
