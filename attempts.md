# Attempts log

A running record of every experiment we have run, what it showed, and what we concluded, kept so the final write-up can report what we learned and how the models behaved. Newest entries go at the bottom of each section. **Rule: every number here comes from a run on Modal** (volume `jepa-perturb`, workspace `ls9563`) unless it is marked *(paper)* or *(guess)*.

Common facts:
- **Adamson split:** GEARS simulation split, seed 1, hash `5618e72828c1`: 58 train / 7 val / 22 test perturbations. Every fine-tune and baseline below uses this split.
- **Primary fine-tune metric:** test mean per-perturbation **delta RMSE** (lower is better). Also reported: Δ-Pearson over all genes, and Δ-Pearson on the top-20 DE genes ("DE20").
- **Kidney pretraining data:** CELLxGENE Census 2023-05-15, 814,279 cells (806,136 train). 4 epochs at batch 32 = 100,768 steps.
- **"Recon"** = masked-gene reconstruction MSE on binned values (bins 1–50), measured on held-out validation cells. ~34–38 = the model predicts each gene's typical value and ignores the cell; ~20–26 = it uses the cell's context.

---

## Baselines on the Adamson split (Luke, Nov 6, Track B experiment `adamson_trackB_v1`)

| Method | Test delta RMSE | Δ-Pearson | DE20 Δ-Pearson |
|---|---|---|---|
| **Linear (gene embedding, Ahlmann-Eltze)** | **0.0575** | 0.720 | 0.738 |
| Training mean | 0.0588 | 0.711 | 0.733 |
| No change (control) | 0.0915 | undefined | undefined |

These are the numbers to beat. The two simple baselines win on delta RMSE against every deep model we have trained so far.

---

## A1. Track B: public scGPT ± Luke's population-level JEPA auxiliary loss (Luke, Nov 6)

Public `scGPT_human`, 15 epochs, B300, 2 seeds per arm (~25 min per run).

| Method | Test delta RMSE (2-seed mean) | Δ-Pearson | DE20 Δ-Pearson |
|---|---|---|---|
| scGPT + JEPA aux | 0.0723 | 0.625 | 0.784 |
| scGPT (matched, MSE only) | 0.0725 | 0.610 | 0.770 |

- JEPA − scGPT delta RMSE: −0.0003, 95% CI [−0.0020, 0.0014]. **No difference.**
- Both lose to linear by about 0.015 delta RMSE (CI excludes 0).
- DE20 Δ-Pearson favours the JEPA arm slightly (+0.014, CI [0.001, 0.029]). That is a weak signal on one metric.
- Our GEARS-style mean-based metrics (absolute Pearson ≈ 0.99) are on a different scale from the paper's (scGPT 0.905 *(paper)*). The paper probably computes per cell. **Unresolved.**

## A2. Kidney pretraining pilot (Luke, Nov 6)

300 steps per objective on B300. **0.050 s/step at batch 32** (641 cells/s, 7.6 GB). A full 4-epoch run is therefore ~1.4 h.
- JEPA (paper version): teacher `<cls>` spread fell from 0.002 to **0.00009** within 300 steps, and the JEPA loss to 0.0004. This was the first sign of collapse.

## A3. Short anti-collapse ablations (Luke, Nov 7)

3 epochs × 48k cells = **4,500 steps** (~4% of a full run), seed 0, clip 1.0, bf16. Values at the last epoch.

| Run | Val recon | Teacher spread | JEPA loss |
|---|---|---|---|
| plain scGPT | 36.0 | — | — |
| JEPA `faithful` (paper) | 23.2 | 0.030 | 0.027 |
| JEPA `center` (DINO-style centring) | 24.7 | 0.031 | 0.049 |
| JEPA `vicreg` (variance hinge) | 31.9 | 0.041 | 0.123 |
| JEPA `token` (I-JEPA-style per-token targets) | **21.3** | 0.030 | 0.035 |

In this short run, `faithful` recovered from the pilot's collapse. `vicreg` had the most spread but the worst reconstruction.

## A4. Fine-tuning on Adamson (Nov 7)

15 epochs, B300, best epoch chosen on validation, then tested once.

**(a) Public scGPT: Cell-JEPA fine-tuning recipe vs the paper-matched scGPT recipe (MSE + 0.8·ECS)**

| Recipe | Seed | Test delta RMSE | Δ-Pearson | DE20 Δ-Pearson |
|---|---|---|---|---|
| Cell-JEPA (`celljepa`) | 1 | 0.0736 | 0.621 | 0.785 |
| Cell-JEPA (`celljepa`) | 2 | 0.0719 | 0.594 | 0.779 |
| scGPT + ECS (`scgpt_paper`) | 1 | 0.0718 | 0.617 | 0.774 |
| scGPT + ECS (`scgpt_paper`) | 2 | 0.0769 | 0.557 | 0.738 |

2-seed means: Cell-JEPA 0.0728 vs scGPT+ECS 0.0744 delta RMSE. The DE20 Δ-Pearson means are 0.782 vs 0.756. The difference is within seed noise, and both are clearly worse than linear (0.0575).

**(b) Luke's short (4,500-step) kidney checkpoints from A3, seed 1, bins50 inputs**

| Pretrained checkpoint | FT recipe | Test delta RMSE | Δ-Pearson | DE20 Δ-Pearson |
|---|---|---|---|---|
| JEPA `faithful` | scGPT+ECS | **0.0733** | 0.581 | 0.761 |
| JEPA `token` | scGPT+ECS | 0.0746 | 0.607 | 0.773 |
| JEPA `faithful` | Cell-JEPA | 0.0772 | 0.516 | 0.678 |
| JEPA `token` | Cell-JEPA | 0.0787 | 0.499 | 0.727 |
| JEPA `vicreg` | scGPT+ECS | 0.0812 | 0.489 | 0.709 |
| JEPA `center` | scGPT+ECS | 0.0813 | 0.520 | 0.742 |
| JEPA `center` | Cell-JEPA | 0.0828 | 0.535 | 0.746 |
| plain scGPT | Cell-JEPA | 0.1411 | 0.043 | 0.276 |
| plain scGPT | scGPT+ECS | 0.1424 | 0.043 | 0.276 |
| JEPA `vicreg` | Cell-JEPA | 0.1424 | 0.043 | 0.276 |

- Checkpoints from pretraining runs that learned (JEPA variants) fine-tune to roughly public-scGPT level. They reach that after only 4,500 kidney steps, versus 33M-cell whole-human pretraining for public scGPT.
- The stuck plain-scGPT checkpoint fine-tunes to **worse than no-change** (0.142 vs 0.0915).
- **Degenerate output:** three runs from different checkpoints and recipes give *identical* metrics (0.1424 / 0.0432 / 0.2764). That points to a constant, input-independent prediction, i.e. collapse during fine-tuning. Not yet investigated.
- In this short-checkpoint setting, the scGPT+ECS recipe beat the Cell-JEPA recipe for every checkpoint (one seed).

## A5. Full 4-epoch kidney pretraining (Nov 7, `trackA/pretrain/runs/*_v1*`)

Paper schedule (4 epochs, batch 32, lr 1e-4), seed 0, clip 1.0, bf16, B300.

| Run | Val recon after epoch 1 | Teacher spread | Status |
|---|---|---|---|
| JEPA `faithful` (paper) | 37.6 | 0.000016 | **Fully collapsed.** JEPA loss ~4e-6; reconstruction identical to plain scGPT to 4 decimals. Cancelled after ~2 epochs. |
| plain scGPT | 37.6, then **50.8 after epoch 3** | — | **Stuck** at "predict each gene's typical value" from step ~200; got worse. Cancelled. |
| JEPA `token` | **19.8** | 0.038 | healthy; still running |
| JEPA `center` | **19.9** | 0.040 | healthy; still running |

Context-shuffle probe: we shuffled the values of the visible genes within each cell, which keeps *which genes are expressed* but scrambles *how much*. Val recon after vs before:
- `token` 19.34 vs 19.32
- `center` 19.57 vs 19.57
- the stuck models: unchanged

**No model uses the expression values of the other genes.** The learning models use *which genes are present*, which encodes cell type.

## A6. Why is plain scGPT stuck? One-seed variant test (Nov 7)

Plain scGPT objective, 3,000 steps, seed 0. One change per variant.

| Variant | Val recon | Escaped? |
|---|---|---|
| baseline (clip 1.0, bf16, post-norm, lr 1e-4) | 35.0 | no |
| + LR warmup (1k steps) | 34.5 | no |
| pre-norm | 26.4 | yes |
| pre-norm + warmup | 32.2 | partly |
| fp32 | 23.8 | yes |
| lr 3e-5 | 25.5 | yes |
| no clipping | 25.3 | yes |

We first concluded "the clipping/bf16 combination causes it". **A7 showed this was wrong:** all of these runs share one seed, so they cannot tell a cause apart from luck.

## A7. Clip-value sweep, 2 seeds (Nov 7)

Plain scGPT, 3,000 steps.

| Clip | Seed 1 val recon | Seed 2 val recon | Fraction of steps clipped |
|---|---|---|---|
| 1.0 | 39.0 | **24.8** | 100% |
| 10 | **26.3** | 38.6 | ~97% |
| 50 | 36.9 | 38.6 | 75% |
| 200 | 37.9 | 37.7 | ~16% |
| 1,000 | 36.9 | 37.7 | 1% |
| none | **23.8** | 37.9 | 0% |

- **3 of 12 escaped, with no relation to the clip value.** Escape from the plateau looks stochastic, typically at steps 1,500–2,500 when it happens.
- Median gradient norm is ~90–145; early spikes reach 2,000–21,000. No run diverged, even without clipping.
- **Conclusion:** clipping is neither the cause nor the fix. The plain scGPT objective has a plateau ("predict the per-gene median") that runs escape only by chance.

## A8. Escape rates over 4 seeds (Nov 7, running)

5,000 steps, clip 1.0, bf16; seeds 1–4 per configuration. "Escaped" = val recon well below the ~35 plateau. Val recon per seed:

| Config | s1 | s2 | s3 | s4 | Escaped |
|---|---|---|---|---|---|
| plain scGPT (lr 1e-4, post-norm) | 35.3 | 38.6 | 40.9 | 36.7 | **0/4** |
| plain scGPT, pre-norm | **20.2** | 45.2 | 39.8 | **25.1** | 2/4 |
| **plain scGPT, lr 3e-5** | **20.4** | **20.7** | **21.9** | **21.5** | **4/4** |
| JEPA `token` (lr 1e-4) | 35.3 | **20.9** | **23.1** | **22.5** | 3/4 |
| JEPA `faithful` (paper, lr 1e-4) | **23.0** | 36.2 | **27.5** | **22.0** | 3/4 |

- **Lowering the learning rate to 3e-5 fixes the plateau reliably.** All 4 seeds escape, and they start descending by step 1,000. They also reach the best reconstruction (~20.4–21.9).
- At the paper's lr 1e-4, the JEPA term helps (3/4 escape vs 0/4 for plain scGPT), but it is not reliable. When a JEPA run fails to escape, its teacher also collapses (spread ~2e-5, JEPA loss ~2e-5). **The reconstruction plateau and the JEPA collapse occur together.** That explains A5: the full `faithful` run (seed 0) was an unlucky seed.
- Pre-norm (Xiong et al.) helps only half the time.
- **Conclusion:** the stall is an optimisation problem at lr 1e-4 in our setup. A fair Track A comparison should run both objectives at a learning rate where plain scGPT trains reliably (3e-5), recorded as a deviation from the paper's 1e-4. The claim "JEPA fixes the plateau" does **not** survive: a simple LR change does better.

---

## A9. Full 4-epoch pretraining at lr 3e-5, all objectives (Nov 7)

Launched with `diag/launch_pretrain_lr3e5.py` (call ids in `diag/call_ids_A9.json`). Paper schedule except **lr 3e-5** (paper: 1e-4; deviation justified by A8). Seed 0, clip 1.0, bf16, B300.
- Run dirs: `trackA/pretrain/runs/{scgpt_lr3e5, jepa_lr3e5, jepa_lr3e5_token, jepa_lr3e5_center}`
- Checkpoints: `trackA/checkpoints/<objective>_kidney_<tag>`

Final results (validation masked-reconstruction MSE per epoch; teacher spread = collapse check, ~0 = collapsed):

| Run | Ep 1 | Ep 2 | Ep 3 | Ep 4 (best) | Teacher spread at end | Wall time |
|---|---|---|---|---|---|---|
| **plain scGPT** | 17.18 | 16.24 | 16.19 | **15.68** | — | 73 min |
| JEPA `faithful` (paper) | 19.25 | 18.63 | 18.13 | **17.73** | 0.039 | ~86 min |
| JEPA `token` | 19.76 | 19.31 | 18.77 | **18.40** | 0.037 | ~88 min |
| JEPA `center` | 21.82 | 19.90 | 20.24 | **18.73** | 0.040 | ~86 min |

- **First working kidney scGPT baseline.** It improved every epoch, unlike lr 1e-4 (stuck at 35–50).
- **The paper's JEPA did not collapse at lr 3e-5.** Spread held at 0.036–0.039, versus 0.00002 at lr 1e-4 (A5).
- JEPA reconstructs worse than plain scGPT (17.7–18.7 vs 15.7). This is the expected cost of the extra latent objective, and reconstruction is not the paper's evaluation. The comparison that matters is downstream (Tasks 1–3).
- Checkpoints: `trackA/checkpoints/{scgpt_kidney_lr3e5, jepa_kidney_lr3e5, jepa_kidney_lr3e5_token, jepa_kidney_lr3e5_center}`.
- The A5 `token`/`center` runs at lr 1e-4 finished at val recon 18.07 / 18.12. Their checkpoints are `jepa_kidney_v1_token` and `jepa_kidney_v1_center`.

## A10. Seed check: JEPA at lr 3e-5, and Xiong-style warmup (Nov 7, running)

`diag/seeds_check.py`, 4 seeds each:
- JEPA `faithful` at lr 3e-5, 5,000 steps
- JEPA `token` at lr 3e-5, 5,000 steps
- plain scGPT at lr 1e-4 with a 4,000-step linear warmup, 8,000 steps (post-norm, as Xiong et al. prescribe)

Raw output in `diag/results/A10_seeds.txt`. Val recon per seed:

| Config | Steps | s1 | s2 | s3 | s4 | Escaped | Teacher spread at end |
|---|---|---|---|---|---|---|---|
| JEPA `faithful`, lr 3e-5 | 5k | 24.6 | 23.7 | 24.5 | 24.0 | **4/4** | 0.026–0.030 (no collapse) |
| JEPA `token`, lr 3e-5 | 5k | 21.7 | 24.0 | 24.3 | 24.4 | **4/4** | 0.026–0.031 |
| plain scGPT, lr 1e-4 + 4k warmup | 8k | 18.3 | 18.8 | 23.7 | 21.3 | **4/4** | — |
| *(A8)* plain scGPT, lr 3e-5 | 5k | 20.4 | 20.7 | 21.9 | 21.5 | 4/4 | — |

- **At lr 3e-5, JEPA trains reliably too, and the paper's JEPA no longer collapses.** That holds for all 8 runs.
- At the same lr and step count, JEPA's reconstruction (~22–24.5) is **slightly worse** than plain scGPT's (~20.4–21.9). The JEPA term trades a little reconstruction for its latent objective. This is expected and is not the target metric, but the write-up should note it.
- **Xiong et al.'s fix works:** with a 4,000-step linear warmup, plain scGPT escapes 4/4 *at the paper's lr 1e-4*. Train recon at step 5k is ~21.8–22.2, similar to lr 3e-5. The 1-seed, 1k-step warmup in A6 was too short or too unlucky to show this.
- **Implication:** a warmup at lr 1e-4 is a *more paper-faithful* fix than lowering the LR. The paper gives lr 1e-4 and says nothing about warmup. Not yet tested: JEPA with warmup at lr 1e-4.
- **Decision (Nov 7):** use lr 3e-5 (A9) as the main Track A pretraining setting. It is simpler: already running, no code change, verified on 12 seeds. Keep "lr 1e-4 + 4k warmup" as the backup. Using it would need a `warmup_steps` option in `PretrainConfig` plus a redeploy.

---

## A11. Task 2: zero-shot PBMC-10K cell-type clustering (Nov 7)

`modal_pbmc.py` + `src/track_a/pbmc.py`.
- **Data:** scvi-tools 1.1.6 `pbmc_dataset`. It matches the paper exactly: **11,990 cells (7,982 + 4,008), 3,346 genes**, 9 labels; ~458 expressed genes per cell.
- **Embedding:** scGPT's own zero-shot procedure. Nonzero genes in the vocab, up to 1,200; per-cell bins 1–50; `<cls>` output, L2-normalised. The `<cls>` value follows each checkpoint's convention (kidney 0, scGPT_human −2).
- **Metrics:** scib-equivalent. NMI/ARI at the Leiden resolution (0.1–2.0) that maximises NMI; ASW = (silhouette + 1)/2; AvgBIO = their mean. Higher is better.

| Embedding | AvgBIO | NMI | ARI | ASW |
|---|---|---|---|---|
| **PCA-50 (log1p, no model; the paper has no such baseline)** | **0.773** | 0.838 | 0.881 | 0.600 |
| public `scGPT_human` (33M cells) | 0.709 | 0.799 | 0.699 | 0.627 |
| kidney JEPA `center`, lr 1e-4, full 4 ep (A5) | 0.413 | 0.459 | 0.266 | 0.514 |
| kidney JEPA `token`, lr 1e-4, full 4 ep (A5) | 0.342 | 0.378 | 0.144 | 0.505 |
| kidney JEPA `faithful`, short 4.5k steps (A3) | 0.241 | 0.180 | 0.122 | 0.421 |
| kidney JEPA `token`, short (A3) | 0.220 | 0.142 | 0.040 | 0.478 |
| kidney JEPA `center`, short (A3) | 0.188 | 0.086 | 0.043 | 0.435 |
| kidney plain scGPT, short, stuck (A3) | 0.167 | 0.004 | 0.001 | 0.496 |
| kidney JEPA `vicreg`, short (A3) | 0.068 | 0.063 | 0.003 | 0.137 |
| *(paper)* Cell-JEPA, kidney | 0.719 | 0.769 | 0.808 | 0.581 |
| *(paper)* scGPT, kidney | 0.529 | 0.559 | 0.495 | 0.533 |

- **PCA beats the paper's Cell-JEPA** (0.773 vs 0.719) and our public scGPT (0.709). Our PCA number agrees with non-deep baselines in the scGPT paper (~0.72–0.78, as reported by the experiment agent; not verified), which is a sanity check on our metric code.
- Our public scGPT (0.709) is far above the paper's *kidney* scGPT (0.529). That is expected: different pretraining data.
- Our kidney models are well below the paper so far. The best is `center` at 0.413. The stuck plain-scGPT checkpoint carries no cell-type information (NMI 0.004).
- **Pending:** the lr 3e-5 checkpoints (A9), including the first *working* kidney scGPT, are needed for the matched JEPA vs scGPT comparison.

## A12. Task 1: PBMC-10K fine-tuning, first runs (Nov 7)

`modal_pbmc.py --stage finetune` + `src/track_a/pbmc_finetune.py`. Paper App. E.2 recipe: 40% mask, 30 epochs, Adam 1e-4 ×0.9/epoch, batch 64, 9:1 split with seed 42. Loss GEP + GEPC + ECS (+1000·JEPA for JEPA checkpoints, same variant). Seed 0. Final epoch scored with the Task 2 metrics. ~17–20 min per run on B300.

| Fine-tuned from | FT objective | AvgBIO | NMI | ARI | ASW | vs zero-shot (A11) |
|---|---|---|---|---|---|---|
| kidney JEPA `center` (lr 1e-4) | Cell-JEPA | **0.653** | 0.662 | 0.676 | 0.621 | +0.240 |
| public `scGPT_human` | scGPT | 0.624 | 0.733 | 0.469 | 0.668 | **−0.085** |
| kidney JEPA `token` (lr 1e-4) | Cell-JEPA | 0.479 | 0.524 | 0.355 | 0.559 | +0.137 |
| *(A11)* PCA-50, no model | — | 0.773 | 0.838 | 0.881 | 0.600 | |
| *(paper)* Cell-JEPA | | 0.783 | 0.776 | 0.847 | 0.726 | |
| *(paper)* scGPT | | 0.753 | 0.765 | 0.784 | 0.710 | |

- Fine-tuning helps the kidney JEPA checkpoints a lot (`center` 0.41 → 0.65).
- **Fine-tuning hurts public scGPT** (0.709 → 0.624, mostly ARI). Possible causes: our unsupervised ECS (threshold 0.8) may pull embeddings together; we score the final epoch rather than the best one; or there is a recipe mismatch. The paper's fine-tuned numbers (0.75–0.78) are above all of ours. **Unresolved.**
- Nothing beats PCA-50 (0.773) yet.
- Single seed. The matched comparison awaits the lr 3e-5 checkpoints.

## A13. Why did PBMC fine-tuning lower public scGPT? (Nov 7)

`diag/launch_ecs_check.py`. Public `scGPT_human` → scGPT fine-tune (GEP + GEPC + ECS), seed 0, scored every 5 epochs. Raw output in `diag/results/A13_ecs_check.txt`.

| Variant | Ep 5 | Ep 10 | Ep 15 | Ep 20 | Ep 25 | Final (30) |
|---|---|---|---|---|---|---|
| same recipe as A12 (re-run) | 0.571 | 0.618 | 0.623 | 0.625 | 0.689 | **0.691** |
| ECS off | 0.567 | 0.616 | 0.613 | 0.628 | 0.693 | **0.684** |
| ECS threshold 0.3 | 0.564 | 0.606 | 0.678 | 0.685 | 0.623 | **0.680** |
| *(A12)* same recipe, same seed | | | | | | 0.624 |
| *(A11)* zero-shot, no fine-tune | | | | | | 0.709 |

- **Run-to-run noise is about ±0.07 AvgBIO.** The identical recipe and seed gave 0.624 (A12) and 0.691 (here); GPU kernels are not bitwise deterministic. Even within one run, the score moves by up to ~0.06 between epochs (0.685 → 0.623 in the ECS-0.3 run). ARI is the most volatile component.
- **ECS is not the cause.** On, off and threshold 0.3 all end at 0.68–0.69.
- **Fine-tuning does not improve public scGPT on PBMC.** It drops at first (~0.57 at epoch 5) and recovers to ~0.68–0.69 by epoch 25–30, just under its zero-shot 0.709. The "decrease" in A12 was mostly noise.
- **Implication for Task 1:** a single seed cannot resolve differences smaller than ~0.07. Use ≥ 3 seeds and report mean ± sd. The paper reports single numbers with no error bars (its gap is 0.783 vs 0.753 = 0.03, below this noise level).

## A14. Matched evaluation of the lr 3e-5 kidney checkpoints (Nov 7)

Launched with `diag/launch_tasks23_lr3e5.py` (ids in `diag/call_ids_A14.json`); status and results via `diag/status_A14.py` → `diag/results/A14_results.json`.

### Task 2: zero-shot PBMC-10K clustering (paper Table 2)

Same procedure and metrics as A11; one embedding pass per checkpoint, deterministic apart from Leiden.

| Embedding | AvgBIO | NMI | ARI | ASW |
|---|---|---|---|---|
| **kidney JEPA `faithful` (paper method)** | **0.606** | 0.642 | 0.535 | 0.641 |
| kidney JEPA `center` | 0.471 | 0.518 | 0.389 | 0.507 |
| kidney JEPA `token` | 0.428 | 0.463 | 0.286 | 0.536 |
| **kidney plain scGPT (matched baseline)** | **0.375** | 0.430 | 0.223 | 0.473 |
| *(A11)* PCA-50, no model | 0.773 | 0.838 | 0.881 | 0.600 |
| *(A11)* public `scGPT_human` | 0.709 | 0.799 | 0.699 | 0.627 |
| *(paper)* Cell-JEPA, kidney | 0.719 | 0.769 | 0.808 | 0.581 |
| *(paper)* scGPT, kidney | 0.529 | 0.559 | 0.495 | 0.533 |

- **The paper's headline result replicates in direction and size.** Kidney JEPA beats matched kidney scGPT by **+0.231 AvgBIO** (paper: +0.191). The win holds on all four metrics: NMI +0.21, ARI +0.31, ASW +0.17.
- Both of our numbers sit ~0.11–0.15 below the paper's. Likely reasons: our lr 3e-5 deviation, scheduling details, and 4 epochs on our cache (we can't match the paper's exact training).
- The paper's own JEPA objective (`faithful`) beats both anti-collapse variants (`center` 0.471, `token` 0.428). Every JEPA variant beats plain scGPT.
- **Caveat:** PCA-50 (0.773) still beats every model, including the paper's reported Cell-JEPA. The paper never compared against PCA.
- Not yet checked: seed variation. Each pretraining run is a single seed; the embedding step is deterministic.

### Task 3: Adamson perturbation, 16 runs (paper Tables 3–4)

All 16 completed (15 epochs, ~28 min for the scGPT recipe, ~38 min for the Cell-JEPA recipe, B300). Values are means over seeds 1, 2; the ± is the spread across those two seeds. Test set: 22 held-out genes. Same split as every earlier run. Full table in `diag/results/A14_task3_summary.json`.

| Pretrained ckpt → FT recipe | delta RMSE ↓ | GEARS abs Pearson / DE / top20 | GEARS Δ Pearson / DE / top20 | per-cell abs Pearson / DE | per-cell Δ Pearson / DE |
|---|---|---|---|---|---|
| **kidney JEPA → Cell-JEPA FT (paper's arm)** | **0.078 ± 0.002** | 0.989 / 0.971 / 0.967 | 0.573 / 0.753 / 0.757 | 0.846 / 0.919 | 0.636 / 0.713 |
| **kidney scGPT → scGPT FT (paper's baseline)** | **0.079 ± 0.000** | 0.988 / 0.969 / 0.966 | 0.555 / 0.741 / 0.741 | 0.845 / 0.918 | 0.635 / 0.708 |
| kidney JEPA → scGPT FT | 0.076 ± 0.002 | 0.989 / 0.972 / 0.969 | 0.575 / 0.748 / 0.751 | 0.846 / 0.920 | 0.637 / 0.712 |
| kidney scGPT → Cell-JEPA FT | 0.080 ± 0.001 | 0.988 / 0.968 / 0.965 | 0.555 / 0.744 / 0.746 | 0.845 / 0.917 | 0.634 / 0.704 |
| JEPA `token` → Cell-JEPA FT | 0.074 ± 0.002 | 0.990 / 0.969 / 0.965 | 0.577 / 0.748 / 0.751 | 0.846 / 0.917 | 0.637 / 0.708 |
| JEPA `token` → scGPT FT | 0.079 ± 0.000 | 0.988 / 0.965 / 0.960 | 0.528 / 0.733 / 0.733 | 0.845 / 0.913 | 0.635 / 0.697 |
| JEPA `center` → Cell-JEPA FT | 0.076 ± 0.002 | 0.989 / 0.967 / 0.963 | 0.572 / 0.737 / 0.741 | 0.846 / 0.916 | 0.637 / 0.705 |
| JEPA `center` → scGPT FT | 0.074 ± 0.001 | 0.990 / 0.968 / 0.964 | 0.562 / 0.738 / 0.739 | 0.847 / 0.917 | 0.638 / 0.709 |
| **Linear baseline** | **0.058** | 0.993 / 0.970 / 0.966 | **0.720** / 0.735 / 0.738 | **0.854** / 0.919 | **0.659** / 0.708 |
| Training mean | 0.059 | 0.993 / 0.969 / 0.965 | 0.711 / 0.729 / 0.733 | 0.854 / 0.918 | 0.658 / 0.704 |
| No change | 0.091 | 0.986 / 0.919 / 0.906 | undefined | 0.751 / 0.825 | undefined |
| *(paper)* Cell-JEPA | — | 0.937 / 0.741 / 0.720 | 0.120 / −0.176 / −0.152 | | |
| *(paper)* scGPT | — | 0.905 / 0.699 / 0.677 | 0.173 / 0.101 / 0.120 | | |

- **The paper's direction holds weakly on absolute metrics.** Kidney JEPA + Cell-JEPA FT beats kidney scGPT + scGPT FT on every column, but only by 0.001–0.016. The delta-RMSE gap (0.078 vs 0.079) is within seed spread.
- **The paper's delta collapse does not replicate.** The paper has Cell-JEPA *worse* on Δ metrics (−0.176 Δ-Pearson DE). Ours is slightly *better* on Δ (0.753 vs 0.741).
- **Pretraining matters slightly more than the fine-tune recipe.** JEPA-pretrained checkpoints are ~0.002–0.003 delta RMSE better than scGPT-pretrained ones under either recipe. Swapping the recipe on a fixed checkpoint changes almost nothing.
- **Every deep model loses to the simple baselines** (linear 0.058, training mean 0.059 vs best deep 0.074). Linear also wins on GEARS Δ-Pearson (0.72 vs ≤0.58) and per-cell metrics. This matches Ahlmann-Eltze et al. 2025.
- **The metric scale still does not match the paper.** Our mean-based absolute Pearson is ~0.99 (paper 0.90–0.94). Our per-cell absolute Pearson (~0.85) is closer, but our Δ metrics (0.55–0.76) are far above the paper's (−0.18 to 0.17). The paper's exact metric definition remains unknown.

## A16. Predicted-effect retrieval: scoring Task 3 the way a selection tool would use it (Nov 7)

`diag/retrieval.py` (CPU; reads the saved `test_predictions.npz` of every Adamson run and baseline). Output in `diag/results/A16_retrieval.md` / `A16_retrieval_raw.json`.
- **Deltas:** profile − mean of all control cells.
- **Genes:** the 500 with the largest variance across *training*-perturbation deltas (no test information).
- **22 held-out genes,** 58 measured training perturbations as the reference set. Seeds pooled (2 per run arm).

Metrics (higher is better except mean rank):
- **ref_spearman:** does the prediction rank the 58 measured perturbations like the true response does?
- **ref p@5:** overlap of the top-5 most similar measured perturbations.
- **cluster_agreement:** correlation of the predicted vs true similarity matrices among the 22 held-out genes.
- **id_top1 / id mean rank:** is each prediction closest to its *own* gene's true response among the 22? Chance top-1 = 0.045; chance mean rank = 11.5.

| Predictions | ref_spearman | ref p@5 | cluster agr. | id top-1 | id mean rank |
|---|---|---|---|---|---|
| (upper bound: true deltas) | 1.000 | 1.000 | 1.000 | 1.000 | 1.0 |
| **public scGPT → Cell-JEPA FT** | **0.601** | **0.286** | **0.158** | 0.205 | **8.6** |
| public scGPT → scGPT+ECS FT | 0.509 | 0.250 | −0.132 | **0.227** | 9.2 |
| public scGPT + Luke's JEPA aux (Track B) | 0.565 | 0.273 | −0.050 | 0.159 | 9.3 |
| public scGPT, MSE only (Track B) | 0.531 | 0.255 | −0.046 | 0.182 | 9.6 |
| kidney JEPA → Cell-JEPA FT (paper arm) | 0.386 | 0.150 | 0.100 | 0.136 | 11.2 |
| kidney scGPT → scGPT FT (paper baseline) | 0.329 | 0.136 | −0.105 | 0.136 | 11.0 |
| other kidney arms (token/center/cross) | 0.31–0.44 | 0.11–0.18 | −0.21–0.16 | 0.07–0.14 | 10.6–11.6 |
| **linear baseline** | 0.559 | 0.273 | 0.024 | 0.091 | 10.0 |
| training mean | 0.496 | 0.164 | undefined | 0.045 (chance) | 11.5 |
| no change | undefined | ~chance | undefined | 0.000 | 11.5 |

- **First metric where deep models beat linear: identification.** Public-scGPT-based models pick out the right held-out gene 16–23% of the time, versus linear 9% and chance 4.5%. Their mean rank is also better (8.6–9.6 vs 10.0). The models capture gene-specific structure that the RMSE-winning baselines lack.
- **Neighbour retrieval:** public scGPT + Cell-JEPA FT is best (0.601 vs linear 0.559). The training mean already scores 0.496 here, because all knockdowns share a common response component.
- **Clustering held-out genes against each other fails for everyone** (|r| ≤ 0.21). With only 22 genes, this metric is noisy.
- **The kidney checkpoints are clearly worse than public scGPT here,** and worse than linear and even the training mean on neighbour retrieval. Within the paper's pair, JEPA is slightly ahead on 3 of 4 metrics, but the differences are tiny.
- **Caveat:** 22 test genes × 2 seeds gives very low resolution (one gene = 0.045 top-1). There are no confidence intervals yet. Next: bootstrap over genes, and a larger screen (Norman, or Replogle K562 essential) before claiming anything.

## A17. Confidence intervals for the retrieval test (Nov 7)

`diag/retrieval.py` with `src/retrieval.py`. Per-gene scores, seeds averaged within each arm, paired bootstrap over the 22 test genes (2,000 resamples, seed 0, 95% percentile CIs). Full tables in `diag/results/A17_adamson_retrieval_ci.md` / `.json`. The reference set is 57 training perturbations (A16 said 58; the control is excluded).

Selected per-arm means [95% CI]:

| Arm | ref_spearman | id_top1 | id mean rank |
|---|---|---|---|
| public scGPT → Cell-JEPA FT | 0.601 [0.42, 0.76] | 0.205 [0.07, 0.34] | 8.6 [5.9, 11.5] |
| public scGPT → scGPT+ECS FT | 0.509 [0.32, 0.68] | 0.227 [0.09, 0.39] | 9.2 [6.3, 12.1] |
| kidney JEPA → Cell-JEPA FT | 0.386 [0.25, 0.52] | 0.136 [0.02, 0.27] | 11.2 [8.6, 13.8] |
| kidney scGPT → scGPT FT | 0.329 [0.19, 0.47] | 0.136 [0.02, 0.27] | 11.0 [8.2, 13.8] |
| linear | 0.559 [0.35, 0.73] | 0.091 [0.00, 0.23] | 10.0 [7.4, 12.7] |
| training mean | 0.496 [0.30, 0.66] | 0.045 [0.00, 0.14] | 11.5 |

Key paired differences, diff [95% CI] (fraction of resamples > 0):
- public Cell-JEPA FT − linear, id_top1: +0.114 [−0.07, 0.30] (0.87); id mean rank −1.36 [−3.2, 0.4]
- public scGPT+ECS FT − linear, id_top1: +0.136 [−0.05, 0.32] (0.90)
- kidney JEPA arm − kidney scGPT arm (paper's comparison): unresolved on every metric
- public Cell-JEPA FT − public scGPT+ECS FT, cluster_agreement: +0.29 [0.09, 0.44] (1.00). **The only robust JEPA-vs-scGPT difference.**
- all 8 kidney lr 3e-5 arms − linear, ref_spearman and ref_p@5: CIs entirely below 0 (kidney checkpoints are worse than linear at ranking measured perturbations)

**Conclusions:**
- **A16's "deep beats linear at identifying genes" is a trend, not a result.** ~87–90% of resamples favour deep models, but the CIs include 0.
- **The only CI-supported JEPA effect:** with public scGPT, the Cell-JEPA fine-tune recipe recovers the similarity structure among held-out genes better than the scGPT recipe (cluster agreement +0.29).
- **22 genes are too few.** The CIs only cover gene resampling, not seed variance, so the true uncertainty is larger. This motivates the Replogle evaluation (Step 2).

## A18. Continued pretraining from public scGPT: code + smoke test (Nov 7)

New `PretrainConfig.init_ck pt` (merged from `worker/init-ckpt`). It loads public scGPT_human weights (159 keys: encoder, value_encoder, transformer_encoder, decoder) into the student before training. The JEPA teacher is a copy of the loaded student. Both objectives start from the identical init hash `4aba96aff0547947`. There is a `--cache` override to pretrain on another corpus.

B300 smoke on the kidney cache (bs 32, lr 3e-5, 20 warmup + 200 steps), results in `/vol/trackA/pretrain_pilot/B300_scgpt_human_init.json`:

| | scGPT objective | JEPA objective |
|---|---|---|
| train recon, step 0 → 200 | 82.1 → 12.7 | 82.1 → 13.2 |
| JEPA loss | — | 1.04 → 0.015 |
| teacher `<cls>` spread | — | 0.0092 → 0.0085 (range 0.0085–0.0102) |
| s/step | 0.043 | 0.051 |

- Loading works: recon starts at 82 instead of ~1,500 from random init.
- **Concern: low teacher `<cls>` spread from step 0.** Public scGPT was pretrained *without* a `<cls>` objective (`no_cls: true`), so its `<cls>` output is nearly the same for every cell (spread ~0.009). For comparison: kidney JEPA runs sat at 0.03–0.04, and collapsed runs at 0.00002. The JEPA loss falls to 0.015 within 200 steps, which suggests the JEPA term may carry little per-cell signal at first. It is not training-induced collapse (it does not shrink), but it is a risk for the blood run.
- Deviation: we feed `<cls>` value 0, whereas public scGPT's own embedding code used −2.

## A19. `<cls>` spread pilot: does JEPA learn anything when continuing from public scGPT? (Nov 7)

`diag/cls_pilot.py`, raw output in `diag/results/A19_cls_pilot.jsonl`. JEPA objective, init from public scGPT_human, kidney cache, bs 32, lr 3e-5, 2,000 steps, seed 0. "Constant-guess ref" = the JEPA loss a predictor would get by always outputting the batch-average target. If the real JEPA loss is not far below it, the JEPA term carries no per-cell signal.

| Setting | Step | recon | JEPA loss | constant-guess ref | teacher spread |
|---|---|---|---|---|---|
| paper JEPA, `<cls>`=0 | 1 / 500 / 2000 | 82.1 / 15.7 / 15.2 | 1.04 / 0.0042 / **0.0004** | 0.023 / 0.011 / **0.0025** | 0.0092 / 0.0062 / **0.0029** |
| paper JEPA, `<cls>`=−2 | 1 / 500 / 2000 | 81.4 / 15.8 / 15.2 | 1.03 / 0.0037 / **0.0003** | 0.026 / 0.011 / **0.0018** | 0.0097 / 0.0060 / **0.0024** |
| `center` variant, `<cls>`=0 | 1 / 500 / 2000 | 82.1 / 16.3 / 15.8 | 0.99 / 0.072 / **0.037** | 0.82 / 0.82 / **0.83** | 0.0092 / 0.0205 / **0.0194** |

Validation at 2,000 steps: paper JEPA val recon 14.63 (`<cls>`=0) / 14.68 (−2); `center` 15.19, with target spread 0.043 (stable).

- **The paper's JEPA slowly collapses when continuing from public scGPT.** Teacher spread falls 0.009 → 0.003. Even a constant guess would score ~0.002, so the JEPA term has almost nothing cell-specific left to teach. Feeding `<cls>`=−2 changes nothing.
- **`center` carries real per-cell signal.** Its JEPA loss (0.037) is ~20× below the constant-guess reference (0.83), the centred target spread is stable at ~0.042, and the teacher's own spread *doubles* (0.009 → 0.019).
- **Cost:** slightly worse reconstruction (val 15.19 vs 14.63).
- **Implication for the blood run:** the paper's JEPA would likely reduce to plain scGPT. A meaningful JEPA-vs-scGPT test from public scGPT needs `center` (a deviation from the paper, recorded). Note: in the kidney from-scratch runs, `faithful` beat `center` on Task 2 (0.606 vs 0.471), so the best variant depends on the starting point.

## A20. Replogle K562 essential: data contract + baselines + timing pilot (Nov 7)

Merged from the Unit 2 branch. Experiment `replogle_k562_v1`, GEARS-processed (dataverse 7458695), cache `data/replogle_k562_essential_gears_simulation_seed1`, split hash **`d4237826c68c`** (GEARS 0.0.2 simulation, seed 1). The Adamson cache was rebuilt in a temp dir to confirm it is byte-identical (split hash 5618e72828c1 unchanged).

- **680 of 1,092 knockdowns were excluded:** their target gene is not among the 5,000 measured genes. Our scGPT perturbation models mark the target by its gene column, so they cannot represent these. This differs from published GEARS/scGPT Replogle splits, which keep all perturbations.
- **Kept:** 412 knockdowns, 68,716 cells (10,691 controls) × 5,000 genes. Split: **278 train / 31 val / 103 test** (~5× Adamson's 22 test genes).
- **12 targets aliased to current HGNC symbols.** 7 of these are new and Replogle-only (kept dataset-specific so Adamson's inputs do not change).

| Baseline | test delta RMSE | test Δ-Pearson |
|---|---|---|
| linear | 0.0755 | 0.423 |
| training mean | 0.0773 | 0.400 |
| no change | 0.0832 | undefined |

- Replogle is a harder benchmark than Adamson: Δ-Pearson ~0.4 vs ~0.7. Linear and training mean are closer to no-change here.
- **Timing pilot** (B300, `scgpt_paper`, public scGPT): 0.116 s/step at batch 64, 790 steps/epoch → **~30 min per 15-epoch fine-tune**; peak GPU 17.8 GB.

## A15. Task 1: PBMC-10K fine-tuning on the lr 3e-5 kidney checkpoints, 3 seeds (Nov 7)

`diag/launch_task1_lr3e5.py`, results in `diag/results/A15_task1_results.json`. Recipe as in A12 (paper App. E.2); each checkpoint is fine-tuned with its own objective; seeds 0, 1, 2. Values are mean ± sd over the 3 seeds.

| Fine-tuned from | AvgBIO | NMI | ARI | ASW | zero-shot (A14) |
|---|---|---|---|---|---|
| **kidney JEPA `faithful` (paper method)** | **0.695 ± 0.046** | 0.713 ± 0.023 | 0.718 ± 0.117 | 0.654 ± 0.003 | 0.606 |
| kidney JEPA `center` | 0.602 ± 0.033 | 0.629 ± 0.042 | 0.588 ± 0.052 | 0.588 ± 0.004 | 0.471 |
| kidney JEPA `token` | 0.531 ± 0.024 | 0.559 ± 0.011 | 0.447 ± 0.062 | 0.586 ± 0.007 | 0.428 |
| **kidney plain scGPT (matched baseline)** | **0.500 ± 0.003** | 0.573 ± 0.002 | 0.246 ± 0.010 | 0.681 ± 0.002 | 0.375 |
| *(A11)* PCA-50, no model | 0.773 | 0.838 | 0.881 | 0.600 | |
| *(paper)* Cell-JEPA / scGPT | 0.783 / 0.753 | 0.776 / 0.765 | 0.847 / 0.784 | 0.726 / 0.710 | |

- **Task 1 replicates in direction, with a larger gap than the paper.** JEPA − scGPT = **+0.195 AvgBIO** (paper +0.030). It is far outside the seed spread: the JEPA arm's worst seed (0.630) beats scGPT's best (0.504).
- The gain comes mostly from **ARI** (0.72 vs 0.25: JEPA embeddings form clusters that match cell types). scGPT has the slightly higher ASW (0.68 vs 0.65).
- Fine-tuning improves every kidney checkpoint over zero-shot (+0.09 to +0.13). The ordering is unchanged (`faithful` > `center` > `token` > scGPT).
- Seed spread: JEPA ±0.046 (one seed at 0.630, two at ~0.73), scGPT ±0.003. Consistent with A13's ~±0.07 single-run noise for some checkpoints.
- PCA-50 (0.773) still beats every fine-tuned model. The paper's absolute numbers are higher than ours (our scGPT baseline 0.50 vs 0.75).

## A21. Blood + bone-marrow pretraining cache, 2M cells (Nov 7)

Merged from the Unit 3 branch. `/vol/trackA/blood_marrow_census_2023-05-15_2M`, same shard contract as kidney (loads with the unchanged `KidneyCache`).
- **Filter:** Census 2023-05-15, `tissue_general in ['blood', 'bone marrow'] and disease == 'normal' and suspension_type != 'na'`. 5,166,736 matching cells from 55 datasets → seeded random sample of **2,000,000** (blood 1.77M, marrow 0.23M).
- **Data:** 25 shards; split 1,980,000 train / 20,000 val. Mean 1,438 expressed genes per cell (median 1,259; 88% > 600). 100% of genes in the scGPT vocab. RAM to load: ~14.4 GB (the pretrain container has 64 GB).
- **Leakage check for the PBMC-10K evaluation data** (10x pbmc4k/pbmc8k): no matching dataset by title/collection, nothing excluded. 17 broader PBMC/10x datasets were reviewed and kept (listed in the manifest). Limitation: this is a text-based screen only.
- **Build:** ~6 min (8 parallel shard containers). Kidney corpus defaults are unchanged and pinned by a test.

## A24. Step 0: retrieval-focused training and scoring options (Nov 7)

Goal of the scoped program: on Replogle's 103 held-out genes, find the setup that best identifies an untested gene's response, and test whether JEPA is what helps. Selection uses validation genes only; the test set is scored once.
- **Primary metric:** identification percentile `id_pct` (1 = own gene ranked first, 0.5 = chance). Secondary: top-1, top-3, mean rank, neighbour Spearman/p@5, cluster agreement.
- **Training (`src/train.py`), defaults unchanged:**
  - `selection="retrieval"` picks the epoch with the best validation `id_pct` (500-gene panel from training deltas). Validation `id_pct`/top-1/top-3 are now logged every epoch for every run.
  - `retr_weight>0` adds an InfoNCE loss: each perturbed training cell's predicted delta must be closest to its *own* training perturbation's mean delta (training conditions only).
  - `val_predictions.npz` is saved. Modal `train*` functions accept `cfg_overrides`.
- **Scoring (`diag/retrieval.py`):** `--split val|test`; seed ensembles (average of predicted deltas, `<arm>__ensN`); two stronger baselines from scGPT_human gene embeddings (ridge, kNN; hyperparameter chosen on validation): `src/retrieval_baselines.py`.
- **Check on Adamson** (existing runs, test split, 22 genes; `diag/results/A24_adamson_check_*`). id_pct / top-1 / top-3:
  - public scGPT → Cell-JEPA FT: single seeds 0.639 / 0.205 / 0.341; **2-seed ensemble 0.639 / 0.273 / 0.409**
  - public scGPT → scGPT+ECS FT: 0.611 / 0.227 / 0.364; ensemble 0.630 / 0.273 / 0.455
  - **scGPT gene vectors only, no model (ridge):** 0.604 / 0.045 / 0.227 (stronger than the PCA linear baseline on id_pct)
  - **Baseline: linear (PCA gene embedding):** 0.574 / 0.091 / 0.182; training mean 0.500 (chance)
  - kidney arms ≈ chance on id_pct (0.49–0.55)
- **Takeaways:** ensembling helps top-1/top-3. The scGPT-gene-vector reference is a tougher competitor than the PCA baseline. Kidney checkpoints carry little gene-specific signal.

## A22. Replogle K562: retrieval on 103 held-out genes (Nov 7)

8 fine-tunes launched 16:01 (`diag/launch_replogle_ft.py`), all completed (15 epochs, B300, ~30–40 min each). Scored on the **test split: 103 held-out genes**, with the 278 training knockdowns as the reference set and a 500-gene panel from training deltas. Values are mean [95% paired-bootstrap CI over genes, 2,000 resamples]; seeds are averaged per gene. Full tables: `diag/results/A22_replogle_retrieval_ci.md`.

**What the metrics mean.** For each held-out gene, compare its predicted response with the true responses of all 103 held-out genes:
- *ID percentile:* where its own gene ranks (1 = first, 0.5 = chance).
- *Top-1 / Top-3:* its own gene is first / in the top 3.
- *Neighbour Spearman:* does the prediction rank the 278 measured knockdowns in the same order as the true response?

| Predictions | What it is | Seeds | ID percentile ↑ (primary) | Top-1 ↑ | Top-3 ↑ | Neighbour Spearman ↑ |
|---|---|---|---|---|---|---|
| **public scGPT → scGPT fine-tune** | pretrained transformer, fine-tuned on 278 knockdowns (MSE + ECS) | 2 | 0.680 [0.62, 0.74] | 0.141 [0.08, 0.21] | 0.214 [0.15, 0.29] | 0.607 [0.55, 0.66] |
| **public scGPT → Cell-JEPA fine-tune** | same, plus the JEPA term during fine-tuning | 2 | 0.688 [0.63, 0.75] | 0.121 [0.07, 0.18] | 0.180 [0.12, 0.25] | 0.606 [0.54, 0.67] |
| kidney scGPT → scGPT fine-tune | our 814k-kidney-cell pretraining (lr 3e-5), paper's baseline arm | 2 | 0.590 [0.54, 0.65] | 0.092 [0.05, 0.15] | 0.121 [0.07, 0.18] | 0.448 [0.38, 0.52] |
| kidney JEPA → Cell-JEPA fine-tune | our kidney JEPA pretraining, paper's Cell-JEPA arm | 2 | 0.584 [0.53, 0.64] | 0.073 [0.03, 0.12] | 0.107 [0.06, 0.17] | 0.384 [0.32, 0.44] |
| **Baseline: linear (PCA gene embedding)** | Ahlmann-Eltze et al. 2025 standard: PCA of training responses → linear map | – | 0.542 [0.48, 0.60] | 0.010 [0.00, 0.03] | 0.049 [0.01, 0.10] | 0.548 [0.49, 0.61] |
| scGPT gene vectors only, no model (ridge) | scGPT's learned gene vector of the target → ridge regression; no transformer | – | 0.675 [0.62, 0.73] | 0.019 [0.00, 0.05] | 0.068 [0.02, 0.12] | 0.453 [0.37, 0.53] |
| scGPT gene vectors only, no model (kNN) | average response of the most similar training genes by scGPT gene vector | – | 0.624 [0.56, 0.69] | 0.058 [0.02, 0.11] | 0.165 [0.10, 0.23] | 0.505 [0.42, 0.59] |
| Training mean | same prediction for every gene | – | 0.500 [0.45, 0.56] | 0.010 [0.00, 0.03] | 0.029 [0.00, 0.07] | 0.513 [0.46, 0.57] |
| *chance* | random ranking | – | 0.500 | 0.010 | 0.029 | 0 |

**Paired differences, model − baseline** (CIs that exclude 0 are clear differences):

| Comparison | ID percentile | Top-1 | Top-3 | Neighbour Spearman |
|---|---|---|---|---|
| public scGPT → scGPT FT − Baseline: linear (PCA) | +0.138 [+0.10, +0.18] | +0.131 [+0.07, +0.20] | +0.165 [+0.11, +0.23] | +0.059 [+0.02, +0.10] |
| public scGPT → scGPT FT − scGPT gene vectors only (ridge) | +0.004 [-0.05, +0.06] | +0.121 [+0.06, +0.19] | +0.146 [+0.06, +0.23] | +0.154 [+0.09, +0.22] |
| public scGPT → Cell-JEPA FT − Baseline: linear (PCA) | +0.146 [+0.11, +0.19] | +0.112 [+0.06, +0.17] | +0.131 [+0.08, +0.19] | +0.058 [+0.02, +0.10] |
| public scGPT → Cell-JEPA FT − scGPT gene vectors only (ridge) | +0.013 [-0.04, +0.07] | +0.102 [+0.04, +0.17] | +0.112 [+0.03, +0.19] | +0.153 [+0.09, +0.22] |

**Top-10 (added after the fact; chance = 10/103 = 0.097):**

| Predictions | Top-10 ↑ [95% CI] | vs Baseline: linear (PCA) | vs scGPT gene vectors only (ridge) |
|---|---|---|---|
| public scGPT → Cell-JEPA fine-tune | **0.374** [0.29, 0.46] | +0.218 [+0.13, +0.31] | +0.092 [−0.01, +0.20] |
| public scGPT → scGPT fine-tune | 0.330 [0.25, 0.41] | +0.175 [+0.10, +0.26] | +0.049 [−0.06, +0.16] |
| kidney scGPT → scGPT fine-tune | 0.228 [0.16, 0.30] | | |
| kidney JEPA → Cell-JEPA fine-tune | 0.184 [0.13, 0.25] | | |
| scGPT gene vectors only, no model (ridge) | 0.282 [0.20, 0.38] | | |
| scGPT gene vectors only, no model (kNN) | 0.272 [0.19, 0.36] | | |
| **Baseline: linear (PCA gene embedding)** | 0.155 [0.09, 0.22] | | |
| Training mean | 0.097 [0.05, 0.16] | | |

At top-10, the advantage of the models over the gene-vector reference shrinks (CIs touch 0), and the Cell-JEPA fine-tune is nominally best (0.374). The "precision at the very top" result is strongest at top-1/top-3.

**Conclusions:**
1. **The fine-tuned public-scGPT models clearly beat the standard baseline** (linear, PCA gene embedding) on ID percentile (+0.14), top-1, and top-3. Every CI excludes 0. With 103 genes, the Adamson trend (A16/A17) is confirmed.
2. **scGPT's gene vectors alone (no model) already reach the same ID percentile** as the fine-tuned models (0.675 vs 0.68–0.69; difference ≈ 0). But the fine-tuned models put the right gene **first 6–7× more often** (top-1 0.12–0.14 vs 0.02) and in the **top 3 about 3× more often**. They also rank similar measured knockdowns much better (+0.15 neighbour Spearman). All of these CIs exclude 0. **The value the full model adds is precision at the top of the list,** which is what a shortlist-style selection tool needs.
3. **JEPA in fine-tuning does not help here:** Cell-JEPA FT vs scGPT FT on public scGPT, 0.688 vs 0.680 ID percentile; top-1 0.121 vs 0.141.
4. **Our kidney checkpoints are clearly weaker** than public scGPT (0.58–0.59), and below the gene-vector baseline on ID percentile.
5. 2-seed ensembles change little on Replogle.

Note: the experiment folder also contains single-seed `celljepa_mk_*` runs we did not launch (a masking-strategy sweep, most likely Luke's). They score 0.67–0.69 ID percentile and appear in the full tables.

**Baseline naming used from here on:** "Baseline: linear (PCA gene embedding)" is *the* baseline (Ahlmann-Eltze et al. 2025, no external knowledge). "scGPT gene vectors only, no model" is a second reference row showing how much of the result comes from scGPT's gene knowledge alone.

## A23. Step 1: blood continued pretraining from public scGPT (Nov 7)

`diag/launch_blood_pretrain.py`. Both arms start from public scGPT_human (identical weights) on the 2M blood + bone-marrow cache (A21): 2 epochs × 61,875 steps, batch 32, lr 3e-5, seed 0, B300. Both completed.

| Arm | Val recon after ep 1 | Val recon after ep 2 (best) | Teacher spread | Centred-target spread | Wall time |
|---|---|---|---|---|---|
| plain scGPT objective | 9.16 | **9.13** | — | — | 90 min |
| JEPA `center` | 10.46 | **10.25** | 0.027 → 0.031 | 0.041 | 106 min |

- **Both trained cleanly. JEPA `center` did not collapse:** teacher spread rose slightly (0.027 → 0.031), and the centred target stayed at ~0.041. This is the same healthy behaviour as the A19 pilot.
- JEPA reconstructs a little worse (10.25 vs 9.13), the expected cost of the extra objective (as in A9/A19). Both are far below public scGPT's starting point on kidney (~82 at step 0; different corpus, so only indicative).
- Checkpoints: `trackA/checkpoints/scgpt_kidney_blood_lr3e5` and `trackA/checkpoints/jepa_kidney_blood_lr3e5`. The `_kidney` in the name is only the export-naming convention; the tag marks the blood corpus.

## A25. Step 2 screen on Replogle (Nov 7, launched ~19:05, done 19:57)

`diag/launch_step2_screen.py`, 24 runs, call ids in `diag/call_ids_A25.json`. Factors:
- starting point {public scGPT, blood-scGPT, blood-JEPA `center`}
- fine-tune recipe {scGPT, Cell-JEPA}
- contrastive retrieval loss {off, on (weight 1)}
- seeds {1, 2}

All runs select their checkpoint by validation ID percentile. Experiment `replogle_k562_v1`; run dirs `<method>_s<seed>_s2<start>_r<0|1>`. Configs are compared on the **31 validation genes only**; the test split is scored in Step 3.
- **Metric decision (user, Nov 7):** ID percentile stays the single primary metric. Top-1, top-3 and **top-10** (added now that n = 103) are secondary.
- **Confound to note:** public scGPT runs use log1p inputs (as in all earlier public runs), while the blood checkpoints use bins50 (as pretrained). So blood-vs-public mixes pretraining and input format. The key comparison, blood-JEPA vs blood-scGPT, is fully matched.

**Full results (19:57; all 24 runs done).** Validation split, 31 genes; chance: ID percentile 0.5, top-1 0.032, top-3 0.097, top-10 0.32. Top-k columns are 2-seed means. CIs for single arms are about ±0.08–0.10 (see `diag/results/A25_val_retrieval_ci.md`).

| Starting point | Fine-tune | Contrastive | ID percentile (2-seed mean) | ID percentile (2-seed ens) | Top-1 | Top-3 | Top-10 |
|---|---|---|---|---|---|---|---|
| public scGPT | scGPT | off | 0.697 | 0.701 | 0.226 | 0.452 | 0.581 |
| public scGPT | scGPT | on | 0.774 | 0.795 | 0.339 | 0.468 | 0.677 |
| public scGPT | Cell-JEPA | off | 0.668 | 0.668 | 0.242 | 0.355 | 0.532 |
| public scGPT | Cell-JEPA | on | 0.745 | 0.775 | 0.339 | 0.403 | 0.613 |
| blood-scGPT | scGPT | off | 0.746 | 0.778 | 0.290 | 0.403 | 0.694 |
| blood-scGPT | scGPT | on | 0.769 | 0.792 | 0.258 | 0.419 | 0.677 |
| blood-scGPT | Cell-JEPA | off | 0.739 | 0.745 | 0.210 | 0.403 | 0.726 |
| blood-scGPT | Cell-JEPA | on | 0.775 | 0.809 | 0.258 | 0.468 | 0.694 |
| **blood-JEPA** | scGPT | off | 0.744 | 0.773 | 0.306 | 0.452 | 0.694 |
| **blood-JEPA** | scGPT | on | 0.774 | 0.805 | 0.258 | 0.403 | 0.710 |
| **blood-JEPA** | Cell-JEPA | off | 0.743 | 0.744 | 0.323 | 0.435 | 0.710 |
| **blood-JEPA** | Cell-JEPA | on | 0.778 | 0.816 | 0.339 | 0.468 | 0.694 |

**Factor effects** (ID percentile, 2-seed means; each compares otherwise-identical configs):
- **Contrastive retrieval loss, on − off:** public +0.077 / +0.077 (scGPT / Cell-JEPA FT); blood-scGPT +0.023 / +0.036; blood-JEPA +0.030 / +0.035. **The only consistent lever**, largest for public scGPT.
- **JEPA pretraining, blood-JEPA − blood-scGPT:** −0.002, +0.005, +0.004, +0.003 across the 4 recipe × loss cells. **≈ 0: no effect of JEPA continued pretraining on retrieval.**
- **Cell-JEPA fine-tune − scGPT fine-tune:** public −0.029 / −0.029 (worse); blood −0.007 to +0.006 (≈ 0).
- **Blood vs public:** +0.04–0.08 without the contrastive loss; ≈ 0–0.03 with it (part of this may be the bins50-vs-log1p input difference, see above).
- **2-seed ensembles:** +0.02–0.04. Best: blood-JEPA → Cell-JEPA FT + contrastive, ens 0.816; blood-scGPT → Cell-JEPA FT + contrastive, ens 0.809.

**Figure:** `diag/results/figures/A25_ladder.png` (script `diag/fig_ladder.py`). Each row adds one change to the row above; panel A shows ID percentile with jumps, panels B–D show top-1/3/10. Validation genes only. The PCA and training-mean baselines were fitted on training genes and predicted for the validation genes in `diag/retrieval.py --split val`. Largest single jumps: scGPT gene vectors over the PCA baseline (+0.098), the contrastive loss (+0.077), PCA over training mean (+0.078), and seed averaging (+0.038). The JEPA steps (+0.005, +0.004) and blood pretraining (−0.005, once the contrastive loss is on) are ≈ 0.

**Bottom line of the screen:** the contrastive loss, blood pretraining and ensembling each help; **JEPA (in pretraining or in fine-tuning) does not move retrieval** in this setup. No deep-config difference is resolved on 31 genes.

## A26. Latent-space retrieval (branch `dangelo/latent-retrieval`): stopped at the sanity checks (Nov 7)

**Idea:** score retrieval in a pretrained model's *embedding* space (mean L2-normalised `<cls>` embedding of a knockdown's cells minus that of control cells) instead of in expression space, hoping JEPA's better-organised space (Tasks 1/2) would make predictions easier to match. Code: `src/latent_retrieval.py`, `diag/latent_retrieval.py`. Predicted cells are turned into encoder input by keeping each predicted cell's top-k genes (k = genes expressed by its paired real control cell) and binning them like real data. Only the `refs` stage ran (3 GPU jobs, ~$2); validation split, 31 genes.

| Embedding space | **Ceiling:** real cells, split halves (ID pct / top-1) | **Oracle:** control + TRUE mean delta through the conversion (ID pct / top-1) |
|---|---|---|
| public scGPT | **0.935** / 0.74 | 0.687 / 0.10 |
| blood-scGPT | 0.910 / 0.55 | 0.518 / 0.03 |
| **blood-JEPA (`center`)** | **0.685** / **0.06** | 0.591 / 0.06 |

- **The predicted-cell → encoder conversion destroys most of the signal.** Even the perfect answer only reaches 0.52–0.69, below the expression-space models (~0.77, A25). Knockdown effects are small shifts across many genes, and these barely change a cell's top-expressed, rank-binned gene set. **Idea #1 cannot beat expression-space retrieval for any model; the 24 prediction jobs were not run.**
- **Real finding:** on *real* cells, blood-JEPA's embedding space separates knockdowns far worse than scGPT's (0.685 vs 0.91–0.94 ID percentile; top-1 0.06 vs 0.55–0.74). Yet JEPA embeddings cluster *cell types* much better (PBMC, A14/A15). **JEPA (center) embeddings are more invariant to within-cell-type perturbation changes.** This matches the Cell-JEPA paper's own explanation for its weak delta metrics ("features invariant across masked views ... may suppress sensitivity to perturbation-induced deviations").
- Caveats: one JEPA variant (`center`, continued from public scGPT, 2 epochs); 31 genes; the ceiling uses ≤256 cells per knockdown split in half.

## A27. Is the perturbation signal erased by JEPA's embedding, or only squashed? (Nov 7)

`diag/latent_transforms.py`, real cells only (no prediction models), ~$3. For each embedding space × readout (`<cls>` / mean of gene tokens) × transform (raw; ZCA whitening on control cells; remove top-10 control PCs; LDA fitted on training knockdowns), each knockdown's real cells are split in half, and we ask whether half A's latent shift identifies half B's (symmetric). Gene sets: 31 validation genes, and 278 training genes for the transforms that do not learn from them. Full table: `diag/results/A27_latent_transforms.md`.

Best transform per space (ID percentile, val / train; top-1 val / train):

| Space | Raw `<cls>` | Best transform | Best ID pct (val / train) | Best top-1 (val / train) |
|---|---|---|---|---|
| gene expression (no model) | 0.996 / 0.986 | – | **0.996 / 0.986** | 0.92 / 0.66 |
| public scGPT | 0.944 / 0.904 | whitening / LDA | 0.992 (LDA) / 0.970 (whiten) | 0.87 / 0.43 |
| blood-scGPT | 0.905 / 0.850 | whitening / LDA | 0.988 (LDA) / 0.963 (whiten, tok) | 0.82 / 0.44 |
| **blood-JEPA (`center`)** | **0.666 / 0.633** | whitening, token readout | **0.810 / 0.738** | 0.24 / 0.04 |

- **Partly squashed, mostly lost.** Whitening (rescaling every direction by its spread among control cells) recovers a lot for JEPA: +0.14 val, +0.11 train ID percentile; top-1 0.08 → 0.24. So some perturbation signal is hidden in low-variance directions, which is the user's "smoothing" intuition. But the same transforms lift the scGPT spaces to ~0.97–0.99. **The gap stays large (0.81 vs 0.98 val; 0.74 vs 0.96 train; train top-1 0.04 vs 0.42–0.44).** Most of the knockdown-specific information is genuinely absent from the blood-JEPA (`center`) embedding.
- **Gene-expression space is the best space of all** (0.996 / 0.986). Every learned embedding loses some perturbation information relative to the raw expression panel. Even whitened public scGPT is a little below it on top-1.
- The `<cls>` vs token readout makes little difference. Whitening ≥ remove-PCs. LDA helps the scGPT spaces most on val top-1.
- **Implications:**
  1. Retrieval for the tool should stay in expression space. That is what A22/A25 do.
  2. The JEPA trade-off is now quantified on real cells: blood-JEPA (`center`) embeddings group cell types well (A14/A15) but keep far less knockdown-specific information, even after un-smoothing.
  3. Untested: whether this is specific to `center`/continued pretraining or general to JEPA. Kidney `faithful`/`token`/`center` vs kidney scGPT would answer it cheaply.

## Literature context (for the write-up)

- **Released scGPT shows our "stuck" behaviour.** Without its cell embedding, it "predicts the median expression value for every single gene regardless of its true expression value" (Kedzierska et al., *Genome Biology* 2025, doi:10.1186/s13059-025-03574-x; MSR summary). We have read only the summary and figure captions, not the full paper.
- **Plateau followed by a sudden drop is a known transformer phenomenon:**
  - Slow attention learning plus representation collapse during the plateau; attention interventions shorten it (Gopalani & Hu, NeurIPS 2025, arXiv 2506.13688).
  - Masked-modelling matrix completion shows the same pattern (arXiv 2410.22244).
  - BERT's sudden drops coincide with attention heads acquiring structure (Chen et al., arXiv 2309.07311).
- **Post-norm transformers need warmup; pre-norm trains stably without it** (Xiong et al., ICML 2020, arXiv 2002.04745).
- **Public scGPT_human was itself pretrained with a 10,000-step LR warmup** (`warmup_ratio_or_step: 10000` in its args.json; lr 1e-4, decay 0.99 per 100 steps). Cell-JEPA's appendix does not mention warmup. Our backup fix (warmup at lr 1e-4) is what scGPT's authors did.
- **Hypothesis tested in A8:** "a JEPA term that rewards distinct cell representations helps escape the plateau." **Partly true:** at lr 1e-4, JEPA escapes 3/4 vs 0/4 for plain scGPT. But lowering the LR fixes plain scGPT 4/4, so it is not a JEPA-specific win.

## Open questions
1. ~~What is the escape rate per objective?~~ Answered in A8: lr 3e-5 fixes plain scGPT 4/4. Still to check: do the JEPA variants also train reliably at lr 3e-5?
2. ~~Why does the paper's JEPA collapse in the full run but not the short run?~~ Answered in A8: escape is seed-dependent at lr 1e-4, and collapse goes together with failing to escape.
3. Why do several fine-tunes produce identical, degenerate outputs (A4b)?
3b. ~~Why does PBMC fine-tuning lower public scGPT?~~ Answered in A13: mostly run-to-run noise (±0.07); ECS is not the cause. Fine-tuning does not help public scGPT on PBMC.
4. Metric scale: do we need per-cell metrics to match the paper's numbers?
5. Is any deep model ever better than linear on the Adamson test split?
