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

## A9. Full 4-epoch pretraining at lr 3e-5, all objectives (Nov 7, running)

Launched with `diag/launch_pretrain_lr3e5.py` (call ids in `diag/call_ids_A9.json`). Paper schedule except **lr 3e-5** (paper: 1e-4; deviation justified by A8). Seed 0, clip 1.0, bf16, B300.
- Run dirs: `trackA/pretrain/runs/{scgpt_lr3e5, jepa_lr3e5, jepa_lr3e5_token, jepa_lr3e5_center}`
- Checkpoints: `trackA/checkpoints/<objective>_kidney_<tag>`

Progress at ~19 min: plain scGPT val recon **17.18 after epoch 1** (better than any lr 1e-4 run). The JEPA runs had not logged an epoch yet.

The A5 `token`/`center` runs at lr 1e-4 **finished**: best val recon 18.07 (`token`) / 18.12 (`center`), at epoch 4. Teacher spread stayed at 0.032 / 0.042 (no collapse). Checkpoints: `trackA/checkpoints/jepa_kidney_v1_token`, `jepa_kidney_v1_center`.

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
4. Metric scale: do we need per-cell metrics to match the paper's numbers?
5. Is any deep model ever better than linear on the Adamson test split?
