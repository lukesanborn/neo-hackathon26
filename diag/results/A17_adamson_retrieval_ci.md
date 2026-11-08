# A17_adamson: retrieval metrics with 95% bootstrap CIs

Cache `/vol/data/adamson_gears002_simulation_seed1`; experiments adamson_trackA_lr3e5, adamson_celljepa_public_v1, adamson_trackB_v1. n_test = 22 held-out genes (chance id_top1 = 0.045, chance id_mean_rank = 11.5); 57 reference (training) perturbations; 500 genes.
Per-gene scores averaged over seeds, then paired bootstrap over test genes (2000 resamples, rng seed 0, percentile CIs). cluster_agreement recomputed on each resampled sub-matrix (duplicate-gene pairs dropped; undefined draws skipped).
**Small n caveat:** with only 22 test genes the CIs are wide; they cover resampling of these genes only (not other gene sets / datasets), and seed variation enters only through the seed average (2 seed(s) per deep arm). NaN per-seed scores (constant predictions) are skipped in the seed average.

## Per-arm means [95% CI]

| Arm | seeds | ref_spearman | ref_p@5 | cluster_agreement | id_top1 | id_mean_rank |
|---|---|---|---|---|---|---|
| (upper bound) true deltas | 1 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| adamson_celljepa_public_v1/runs/celljepa_v1 | 2 | 0.601 [0.419, 0.760] | 0.286 [0.182, 0.391] | 0.158 [-0.012, 0.350] | 0.205 [0.068, 0.341] | 8.591 [5.909, 11.477] |
| adamson_celljepa_public_v1/runs/scgpt_paper_v1 | 2 | 0.509 [0.321, 0.682] | 0.250 [0.164, 0.341] | -0.132 [-0.235, 0.161] | 0.227 [0.091, 0.386] | 9.159 [6.295, 12.136] |
| adamson_trackA_lr3e5/runs/celljepa_jepa_lr3e5 | 2 | 0.386 [0.246, 0.520] | 0.150 [0.086, 0.218] | 0.100 [-0.171, 0.404] | 0.136 [0.023, 0.273] | 11.227 [8.613, 13.819] |
| adamson_trackA_lr3e5/runs/celljepa_jepa_lr3e5_center | 2 | 0.433 [0.223, 0.624] | 0.177 [0.095, 0.268] | -0.187 [-0.273, -0.052] | 0.136 [0.000, 0.273] | 10.568 [7.659, 13.364] |
| adamson_trackA_lr3e5/runs/celljepa_jepa_lr3e5_token | 2 | 0.374 [0.234, 0.503] | 0.118 [0.059, 0.191] | 0.006 [-0.127, 0.165] | 0.114 [0.000, 0.250] | 11.432 [8.681, 14.091] |
| adamson_trackA_lr3e5/runs/celljepa_scgpt_lr3e5 | 2 | 0.357 [0.194, 0.513] | 0.127 [0.045, 0.232] | -0.206 [-0.379, -0.024] | 0.136 [0.000, 0.273] | 11.477 [8.795, 14.068] |
| adamson_trackA_lr3e5/runs/scgpt_paper_jepa_lr3e5 | 2 | 0.368 [0.218, 0.511] | 0.105 [0.055, 0.168] | 0.156 [-0.190, 0.381] | 0.068 [0.000, 0.136] | 11.636 [9.023, 14.114] |
| adamson_trackA_lr3e5/runs/scgpt_paper_jepa_lr3e5_center | 2 | 0.438 [0.243, 0.607] | 0.141 [0.082, 0.209] | -0.141 [-0.249, 0.109] | 0.136 [0.023, 0.273] | 10.886 [8.091, 13.477] |
| adamson_trackA_lr3e5/runs/scgpt_paper_jepa_lr3e5_token | 2 | 0.309 [0.160, 0.446] | 0.123 [0.064, 0.186] | -0.075 [-0.176, 0.094] | 0.114 [0.000, 0.250] | 11.136 [8.340, 13.841] |
| adamson_trackA_lr3e5/runs/scgpt_paper_scgpt_lr3e5 | 2 | 0.329 [0.188, 0.465] | 0.136 [0.059, 0.232] | -0.105 [-0.244, 0.062] | 0.136 [0.023, 0.273] | 10.955 [8.159, 13.773] |
| adamson_trackB_v1/runs/jepa_v1 | 2 | 0.565 [0.388, 0.719] | 0.273 [0.168, 0.395] | -0.050 [-0.199, 0.203] | 0.159 [0.045, 0.295] | 9.341 [6.545, 12.160] |
| adamson_trackB_v1/runs/scgpt_v1 | 2 | 0.531 [0.344, 0.712] | 0.255 [0.150, 0.377] | -0.046 [-0.227, 0.224] | 0.182 [0.068, 0.318] | 9.636 [6.591, 12.682] |
| baselines/linear | 1 | 0.559 [0.349, 0.727] | 0.273 [0.164, 0.391] | 0.024 [-0.179, 0.393] | 0.091 [0.000, 0.227] | 9.955 [7.364, 12.682] |
| baselines/no_change | 1 | undef | 0.055 [0.018, 0.091] | undef | 0.000 [0.000, 0.000] | 11.500 [11.500, 11.500] |
| baselines/training_mean | 1 | 0.496 [0.299, 0.663] | 0.164 [0.100, 0.236] | undef | 0.045 [0.000, 0.136] | 11.500 [8.863, 14.092] |

## Paired differences (A − B): diff [95% CI] (fraction of resamples with diff > 0)

For id_mean_rank lower is better, so a negative difference favours A.

| A | B | ref_spearman | ref_p@5 | cluster_agreement | id_top1 | id_mean_rank |
|---|---|---|---|---|---|---|
| adamson_celljepa_public_v1/runs/celljepa_v1 | baselines/linear | 0.042 [-0.018, 0.103] (0.91) | 0.014 [-0.055, 0.086] (0.65) | 0.134 [-0.260, 0.442] (0.68) | 0.114 [-0.068, 0.295] (0.87) | -1.364 [-3.182, 0.364] (0.06) |
| adamson_celljepa_public_v1/runs/scgpt_paper_v1 | baselines/linear | -0.049 [-0.163, 0.061] (0.21) | -0.023 [-0.086, 0.050] (0.25) | -0.156 [-0.455, 0.058] (0.08) | 0.136 [-0.045, 0.318] (0.90) | -0.795 [-2.432, 0.682] (0.15) |
| adamson_trackA_lr3e5/runs/celljepa_jepa_lr3e5 | baselines/linear | -0.172 [-0.294, -0.042] (0.01) | -0.123 [-0.214, -0.036] (0.00) | 0.076 [-0.461, 0.539] (0.56) | 0.045 [-0.091, 0.182] (0.69) | 1.273 [-0.909, 3.455] (0.88) |
| adamson_trackA_lr3e5/runs/celljepa_jepa_lr3e5_center | baselines/linear | -0.125 [-0.251, -0.013] (0.02) | -0.095 [-0.196, -0.005] (0.02) | -0.211 [-0.528, -0.025] (0.01) | 0.045 [-0.091, 0.182] (0.60) | 0.614 [-1.478, 2.977] (0.70) |
| adamson_trackA_lr3e5/runs/celljepa_jepa_lr3e5_token | baselines/linear | -0.184 [-0.295, -0.056] (0.00) | -0.155 [-0.255, -0.055] (0.00) | -0.018 [-0.371, 0.252] (0.43) | 0.023 [-0.159, 0.205] (0.53) | 1.477 [-0.750, 3.523] (0.91) |
| adamson_trackA_lr3e5/runs/celljepa_scgpt_lr3e5 | baselines/linear | -0.201 [-0.365, -0.030] (0.01) | -0.145 [-0.273, -0.018] (0.01) | -0.230 [-0.615, -0.055] (0.01) | 0.045 [-0.091, 0.182] (0.59) | 1.523 [-0.659, 3.682] (0.91) |
| adamson_trackA_lr3e5/runs/scgpt_paper_jepa_lr3e5 | baselines/linear | -0.191 [-0.331, -0.039] (0.01) | -0.168 [-0.268, -0.073] (0.00) | 0.132 [-0.504, 0.500] (0.62) | -0.023 [-0.136, 0.069] (0.29) | 1.682 [-0.523, 4.023] (0.93) |
| adamson_trackA_lr3e5/runs/scgpt_paper_jepa_lr3e5_center | baselines/linear | -0.120 [-0.196, -0.047] (0.00) | -0.132 [-0.227, -0.045] (0.00) | -0.165 [-0.504, 0.075] (0.10) | 0.045 [-0.136, 0.227] (0.63) | 0.932 [-1.137, 3.069] (0.81) |
| adamson_trackA_lr3e5/runs/scgpt_paper_jepa_lr3e5_token | baselines/linear | -0.249 [-0.365, -0.116] (0.00) | -0.150 [-0.241, -0.064] (0.00) | -0.099 [-0.430, 0.145] (0.23) | 0.023 [-0.159, 0.205] (0.53) | 1.182 [-1.045, 3.341] (0.85) |
| adamson_trackA_lr3e5/runs/scgpt_paper_scgpt_lr3e5 | baselines/linear | -0.229 [-0.419, -0.028] (0.01) | -0.136 [-0.268, -0.014] (0.01) | -0.129 [-0.527, 0.081] (0.11) | 0.045 [-0.045, 0.159] (0.70) | 1.000 [-0.864, 2.887] (0.85) |
| adamson_trackB_v1/runs/jepa_v1 | baselines/linear | 0.007 [-0.066, 0.078] (0.59) | 0.000 [-0.073, 0.073] (0.49) | -0.074 [-0.339, 0.080] (0.18) | 0.068 [-0.091, 0.250] (0.74) | -0.614 [-2.228, 1.023] (0.23) |
| adamson_trackB_v1/runs/scgpt_v1 | baselines/linear | -0.028 [-0.128, 0.081] (0.33) | -0.018 [-0.086, 0.055] (0.32) | -0.070 [-0.333, 0.111] (0.19) | 0.091 [-0.068, 0.273] (0.80) | -0.318 [-2.160, 1.455] (0.36) |
| adamson_trackA_lr3e5/runs/celljepa_jepa_lr3e5 | adamson_trackA_lr3e5/runs/scgpt_paper_scgpt_lr3e5 | 0.057 [-0.094, 0.199] (0.78) | 0.014 [-0.077, 0.086] (0.64) | 0.205 [-0.116, 0.538] (0.79) | 0.000 [-0.114, 0.091] (0.44) | 0.273 [-1.273, 1.864] (0.62) |
| adamson_celljepa_public_v1/runs/celljepa_v1 | adamson_celljepa_public_v1/runs/scgpt_paper_v1 | 0.091 [-0.019, 0.204] (0.95) | 0.036 [-0.014, 0.082] (0.92) | 0.290 [0.091, 0.435] (1.00) | -0.023 [-0.114, 0.068] (0.24) | -0.568 [-1.795, 0.591] (0.18) |
| adamson_trackB_v1/runs/jepa_v1 | adamson_trackB_v1/runs/scgpt_v1 | 0.034 [-0.058, 0.128] (0.75) | 0.018 [-0.041, 0.068] (0.73) | -0.004 [-0.161, 0.126] (0.54) | -0.023 [-0.068, 0.000] (0.00) | -0.295 [-1.545, 0.886] (0.34) |

## Point aggregates (per prediction set; seed aggregates averaged — A16 format)

| Predictions | n | ref_spearman | ref_p@5 | cluster_agreement | id_top1 | id_mean_rank |
|---|---|---|---|---|---|---|
| (upper bound) true deltas | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| adamson_celljepa_public_v1/runs/celljepa_v1 | 2 | 0.601 | 0.286 | 0.158 | 0.205 | 8.591 |
| adamson_celljepa_public_v1/runs/scgpt_paper_v1 | 2 | 0.509 | 0.250 | -0.132 | 0.227 | 9.159 |
| adamson_trackA_lr3e5/runs/celljepa_jepa_lr3e5 | 2 | 0.386 | 0.150 | 0.100 | 0.136 | 11.227 |
| adamson_trackA_lr3e5/runs/celljepa_jepa_lr3e5_center | 2 | 0.433 | 0.177 | -0.187 | 0.136 | 10.568 |
| adamson_trackA_lr3e5/runs/celljepa_jepa_lr3e5_token | 2 | 0.374 | 0.118 | 0.006 | 0.114 | 11.432 |
| adamson_trackA_lr3e5/runs/celljepa_scgpt_lr3e5 | 2 | 0.357 | 0.127 | -0.206 | 0.136 | 11.477 |
| adamson_trackA_lr3e5/runs/scgpt_paper_jepa_lr3e5 | 2 | 0.368 | 0.105 | 0.156 | 0.068 | 11.636 |
| adamson_trackA_lr3e5/runs/scgpt_paper_jepa_lr3e5_center | 2 | 0.438 | 0.141 | -0.141 | 0.136 | 10.886 |
| adamson_trackA_lr3e5/runs/scgpt_paper_jepa_lr3e5_token | 2 | 0.309 | 0.123 | -0.075 | 0.114 | 11.136 |
| adamson_trackA_lr3e5/runs/scgpt_paper_scgpt_lr3e5 | 2 | 0.329 | 0.136 | -0.105 | 0.136 | 10.955 |
| adamson_trackB_v1/runs/jepa_v1 | 2 | 0.565 | 0.273 | -0.050 | 0.159 | 9.341 |
| adamson_trackB_v1/runs/scgpt_v1 | 2 | 0.531 | 0.255 | -0.046 | 0.182 | 9.636 |
| baselines/linear | 1 | 0.559 | 0.273 | 0.024 | 0.091 | 9.955 |
| baselines/no_change | 1 | undef | 0.055 | undef | 0.000 | 11.500 |
| baselines/training_mean | 1 | 0.496 | 0.164 | undef | 0.045 | 11.500 |
