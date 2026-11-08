# knobs_adamson: retrieval metrics with 95% bootstrap CIs

Cache `/vol/data/adamson_gears002_simulation_seed1`; experiments adamson_knobs_v1. n_test = 22 held-out genes (chance id_top1 = 0.045, chance id_mean_rank = 11.5); 57 reference (training) perturbations; 500 genes.
Per-gene scores averaged over seeds, then paired bootstrap over test genes (2000 resamples, rng seed 0, percentile CIs). cluster_agreement recomputed on each resampled sub-matrix (duplicate-gene pairs dropped; undefined draws skipped).
**Small n caveat:** with only 22 test genes the CIs are wide; they cover resampling of these genes only (not other gene sets / datasets), and seed variation enters only through the seed average (1 seed(s) per deep arm). NaN per-seed scores (constant predictions) are skipped in the seed average.

## Per-arm means [95% CI]

| Arm | seeds | ref_spearman | ref_p@5 | cluster_agreement | id_top1 | id_mean_rank |
|---|---|---|---|---|---|---|
| (upper bound) true deltas | 1 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| adamson_knobs_v1/runs/celljepa_mk_base_center | 1 | 0.615 [0.411, 0.801] | 0.345 [0.227, 0.473] | 0.018 [-0.145, 0.459] | 0.136 [0.000, 0.273] | 8.591 [5.727, 11.591] |
| adamson_knobs_v1/runs/celljepa_mk_no_jepa | 1 | 0.590 [0.372, 0.781] | 0.327 [0.200, 0.473] | 0.054 [-0.095, 0.426] | 0.136 [0.000, 0.273] | 9.273 [6.455, 12.139] |
| adamson_knobs_v1/runs/celljepa_mk_ratio_30 | 1 | 0.496 [0.314, 0.667] | 0.264 [0.145, 0.391] | 0.015 [-0.176, 0.273] | 0.227 [0.091, 0.409] | 9.182 [6.181, 12.182] |
| adamson_knobs_v1/runs/celljepa_mk_ratio_50 | 1 | 0.578 [0.394, 0.740] | 0.255 [0.145, 0.373] | 0.015 [-0.149, 0.364] | 0.227 [0.091, 0.409] | 8.091 [5.363, 11.001] |
| adamson_knobs_v1/runs/celljepa_mk_ratio_80 | 1 | 0.604 [0.388, 0.788] | 0.300 [0.182, 0.427] | 0.067 [-0.151, 0.441] | 0.136 [0.000, 0.273] | 9.136 [6.273, 12.091] |
| adamson_knobs_v1/runs/celljepa_mk_target_faithful | 1 | 0.638 [0.423, 0.824] | 0.382 [0.255, 0.509] | 0.044 [-0.115, 0.482] | 0.182 [0.045, 0.364] | 8.545 [5.816, 11.502] |
| adamson_knobs_v1/runs/celljepa_mk_target_token | 1 | 0.645 [0.443, 0.825] | 0.336 [0.227, 0.455] | 0.035 [-0.164, 0.562] | 0.136 [0.000, 0.273] | 8.545 [5.682, 11.591] |
| adamson_knobs_v1/runs/celljepa_mk_target_token_program | 1 | 0.612 [0.399, 0.796] | 0.345 [0.227, 0.464] | 0.020 [-0.163, 0.493] | 0.091 [0.000, 0.227] | 8.500 [5.682, 11.547] |
| adamson_knobs_v1/runs/celljepa_mk_unit_pathway | 1 | 0.650 [0.446, 0.829] | 0.382 [0.255, 0.509] | 0.012 [-0.148, 0.548] | 0.091 [0.000, 0.227] | 8.091 [5.409, 11.182] |
| adamson_knobs_v1/runs/celljepa_mk_unit_program | 1 | 0.588 [0.369, 0.775] | 0.318 [0.200, 0.455] | 0.021 [-0.151, 0.427] | 0.136 [0.000, 0.273] | 8.955 [6.136, 12.045] |
| adamson_knobs_v1/runs/celljepa_mk_unit_topexpr | 1 | 0.610 [0.402, 0.798] | 0.345 [0.227, 0.464] | 0.090 [-0.085, 0.470] | 0.091 [0.000, 0.227] | 8.500 [5.636, 11.591] |
| baselines/linear | 1 | 0.559 [0.349, 0.727] | 0.273 [0.164, 0.391] | 0.024 [-0.179, 0.393] | 0.091 [0.000, 0.227] | 9.955 [7.364, 12.682] |
| baselines/no_change | 1 | undef | 0.073 [0.027, 0.127] | undef | 0.000 [0.000, 0.000] | 11.500 [11.500, 11.500] |
| baselines/training_mean | 1 | 0.496 [0.299, 0.663] | 0.164 [0.100, 0.236] | undef | 0.045 [0.000, 0.136] | 11.500 [8.863, 14.092] |

## Paired differences (A − B): diff [95% CI] (fraction of resamples with diff > 0)

For id_mean_rank lower is better, so a negative difference favours A.

| A | B | ref_spearman | ref_p@5 | cluster_agreement | id_top1 | id_mean_rank |
|---|---|---|---|---|---|---|
| adamson_knobs_v1/runs/celljepa_mk_base_center | baselines/linear | 0.056 [-0.042, 0.152] (0.88) | 0.073 [-0.045, 0.200] (0.87) | -0.006 [-0.285, 0.220] (0.54) | 0.045 [-0.136, 0.227] (0.58) | -1.364 [-3.545, 0.864] (0.11) |
| adamson_knobs_v1/runs/celljepa_mk_no_jepa | baselines/linear | 0.031 [-0.075, 0.131] (0.74) | 0.055 [-0.064, 0.182] (0.82) | 0.030 [-0.256, 0.283] (0.65) | 0.045 [-0.136, 0.227] (0.58) | -0.682 [-2.818, 1.591] (0.26) |
| adamson_knobs_v1/runs/celljepa_mk_ratio_30 | baselines/linear | -0.062 [-0.179, 0.045] (0.16) | -0.009 [-0.082, 0.073] (0.38) | -0.009 [-0.414, 0.279] (0.48) | 0.136 [-0.045, 0.364] (0.88) | -0.773 [-2.682, 1.091] (0.20) |
| adamson_knobs_v1/runs/celljepa_mk_ratio_50 | baselines/linear | 0.019 [-0.086, 0.117] (0.66) | -0.018 [-0.109, 0.073] (0.33) | -0.009 [-0.323, 0.223] (0.51) | 0.136 [-0.091, 0.364] (0.84) | -1.864 [-3.455, -0.273] (0.01) |
| adamson_knobs_v1/runs/celljepa_mk_ratio_80 | baselines/linear | 0.046 [-0.077, 0.159] (0.78) | 0.027 [-0.082, 0.155] (0.64) | 0.043 [-0.241, 0.320] (0.65) | 0.045 [-0.136, 0.227] (0.58) | -0.818 [-2.955, 1.591] (0.24) |
| adamson_knobs_v1/runs/celljepa_mk_target_faithful | baselines/linear | 0.079 [-0.035, 0.186] (0.92) | 0.109 [-0.009, 0.236] (0.95) | 0.020 [-0.252, 0.222] (0.61) | 0.091 [-0.136, 0.273] (0.73) | -1.409 [-3.501, 0.909] (0.10) |
| adamson_knobs_v1/runs/celljepa_mk_target_token | baselines/linear | 0.086 [-0.010, 0.185] (0.96) | 0.064 [-0.045, 0.191] (0.86) | 0.011 [-0.289, 0.294] (0.56) | 0.045 [-0.136, 0.227] (0.58) | -1.409 [-3.591, 0.728] (0.10) |
| adamson_knobs_v1/runs/celljepa_mk_target_token_program | baselines/linear | 0.054 [-0.054, 0.166] (0.85) | 0.073 [-0.036, 0.200] (0.89) | -0.004 [-0.290, 0.252] (0.51) | 0.000 [-0.182, 0.182] (0.39) | -1.455 [-3.455, 0.636] (0.08) |
| adamson_knobs_v1/runs/celljepa_mk_unit_pathway | baselines/linear | 0.092 [-0.001, 0.189] (0.97) | 0.109 [0.000, 0.236] (0.98) | -0.012 [-0.325, 0.246] (0.47) | 0.000 [-0.182, 0.182] (0.39) | -1.864 [-3.682, 0.045] (0.03) |
| adamson_knobs_v1/runs/celljepa_mk_unit_program | baselines/linear | 0.029 [-0.084, 0.138] (0.71) | 0.045 [-0.055, 0.164] (0.78) | -0.003 [-0.284, 0.227] (0.53) | 0.045 [-0.136, 0.227] (0.58) | -1.000 [-3.091, 1.273] (0.17) |
| adamson_knobs_v1/runs/celljepa_mk_unit_topexpr | baselines/linear | 0.051 [-0.043, 0.150] (0.86) | 0.073 [-0.036, 0.200] (0.90) | 0.066 [-0.261, 0.361] (0.66) | 0.000 [-0.182, 0.182] (0.39) | -1.455 [-3.683, 0.727] (0.09) |

## Point aggregates (per prediction set; seed aggregates averaged — A16 format)

| Predictions | n | ref_spearman | ref_p@5 | cluster_agreement | id_top1 | id_mean_rank |
|---|---|---|---|---|---|---|
| (upper bound) true deltas | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| adamson_knobs_v1/runs/celljepa_mk_base_center | 1 | 0.615 | 0.345 | 0.018 | 0.136 | 8.591 |
| adamson_knobs_v1/runs/celljepa_mk_no_jepa | 1 | 0.590 | 0.327 | 0.054 | 0.136 | 9.273 |
| adamson_knobs_v1/runs/celljepa_mk_ratio_30 | 1 | 0.496 | 0.264 | 0.015 | 0.227 | 9.182 |
| adamson_knobs_v1/runs/celljepa_mk_ratio_50 | 1 | 0.578 | 0.255 | 0.015 | 0.227 | 8.091 |
| adamson_knobs_v1/runs/celljepa_mk_ratio_80 | 1 | 0.604 | 0.300 | 0.067 | 0.136 | 9.136 |
| adamson_knobs_v1/runs/celljepa_mk_target_faithful | 1 | 0.638 | 0.382 | 0.044 | 0.182 | 8.545 |
| adamson_knobs_v1/runs/celljepa_mk_target_token | 1 | 0.645 | 0.336 | 0.035 | 0.136 | 8.545 |
| adamson_knobs_v1/runs/celljepa_mk_target_token_program | 1 | 0.612 | 0.345 | 0.020 | 0.091 | 8.500 |
| adamson_knobs_v1/runs/celljepa_mk_unit_pathway | 1 | 0.650 | 0.382 | 0.012 | 0.091 | 8.091 |
| adamson_knobs_v1/runs/celljepa_mk_unit_program | 1 | 0.588 | 0.318 | 0.021 | 0.136 | 8.955 |
| adamson_knobs_v1/runs/celljepa_mk_unit_topexpr | 1 | 0.610 | 0.345 | 0.090 | 0.091 | 8.500 |
| baselines/linear | 1 | 0.559 | 0.273 | 0.024 | 0.091 | 9.955 |
| baselines/no_change | 1 | undef | 0.073 | undef | 0.000 | 11.500 |
| baselines/training_mean | 1 | 0.496 | 0.164 | undef | 0.045 | 11.500 |
