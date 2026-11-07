# Easy wins: where JEPA can beat linear/PCA and scGPT (ranked)

Goal: beat **both** the linear/PCA baseline (`src/baselines.py::linear_gene_embedding`, Ahlmann-Eltze style) and scGPT on **one** pre-declared thing (a metric, a regime, a subset or a task), and argue that JEPA is more capable at that thing, with matched arms, the baselines in the same table, and no test-set tuning. **Replication of Cell-JEPA (Track A, `docs/track_a.md`) comes first**; everything below either reuses Track A/Track B outputs or runs alongside them.

Number labels: **(paper: X)** = taken from the cited paper; **(measured)** = from our own runs or saved split; **(guess)** = estimate, not measured. External claims carry their source. Nothing below has been run yet.

Honest prior: on the prediction task itself, delta-type metrics are unlikely to be won. Cell-JEPA lost on them (paper: Cell-JEPA Table 4, Adamson Δ-Pearson DE −0.176 vs scGPT 0.101), and two independent benchmarks found that deep models, scGPT included, do not beat simple linear/mean baselines on unseen single perturbations (paper: Ahlmann-Eltze, Huber & Anders, *Nat. Methods* 22:1657, 2025, doi:10.1038/s41592-025-02772-6; paper: Viñas Torné et al., "Systema", *Nat. Biotechnol.* 2025, doi:10.1038/s41587-025-02777-8, who attribute much of the apparent skill to systematic control-vs-perturbed shifts). The only published JEPA win is **representation quality** (paper: Cell-JEPA Table 2, zero-shot PBMC AvgBIO 0.719 vs scGPT 0.529). So the top picks test representations of *sparse, noisy cells*, which is the mechanism JEPA is supposed to improve, rather than mean-expression regression. Mean-based GEARS Pearson is close to its ceiling (no-change ≈ 0.99 Pearson, per the deviation log in `docs/track_a.md`; provenance not recorded), so none of the picks uses it as the headline.

## Ranked table

| Rank | ID | Experiment (one line) | The "JEPA is better at X" claim | Metric & baselines beaten | Ease (1–5) | Promise (1–5) | Score | Est. cost (GPU-h, guess) | Depends on |
|---|---|---|---|---|---|---|---|---|---|
| 1 | E9 | Few-cell perturbation identification from frozen cell embeddings | JEPA embeddings identify which perturbation a handful of sparse cells received better than PCA or scGPT | Top-1 accuracy / MRR of nearest-centroid retrieval at k = 5 cells; beats PCA-50 of log1p and matched kidney-scGPT | 4 | 3 | **12** | 0.5 | Kidney checkpoints for the headline; public `scGPT_human` arm + PCA can run now as calibration |
| 2 | E10 | Pre-declared 50/50 blend of linear prediction with each deep model | Cell-JEPA predictions carry held-out-perturbation signal the linear model lacks, more than scGPT's | Mean delta RMSE + DE20 Δ-Pearson; blend(linear, celljepa) beats linear, blend(linear, scGPT) and blend(linear, training-mean) | 5 | 2 | **10** | 0 (CPU) | Public `scGPT_human` runs (Track B `scgpt` + `celljepa`); also Track A runs |
| 3 | E11 | Per-cell / distribution-level scoring of predicted populations | JEPA generates more realistic single-cell post-perturbation populations | Energy distance (PCA space, DE genes) and per-cell Pearson; beats linear-per-cell (`ctrl_i + linear Δ`) and scGPT | 5 | 2 | **10** | 0.3 | Public `scGPT_human` runs (can run now); Track A runs later |
| 4 | E13 | Linear model with swapped perturbation embeddings (PCA vs scGPT vs JEPA encoder) | JEPA encoder's in-silico perturbation embedding is a better linear feature | Delta RMSE / DE20 Δ-Pearson of the *same* linear head; beats PCA-embedding linear and scGPT-embedding linear | 4 | 2 | **8** | 0.2 | Kidney checkpoints (headline); public FT encoders as a variant |
| 5 | E14 | Robustness to thinned control inputs (simulated lower depth) | JEPA's cell representation is more dropout-robust, so its predictions degrade less | Δ in delta RMSE vs thinning rate; beats scGPT and linear-per-cell (mean linear is invariant by construction) | 4 | 2 | **8** | 0.5 | Public `scGPT_human` runs (can run now) |
| 6 | E12 | Few-shot response estimation: k measured cells of a new perturbation → population Δ | JEPA embeddings denoise k sparse cells into a better estimate of the full response | Delta RMSE vs full-population Δ at k = 5; beats raw k-mean, PCA-projected k-mean, scGPT embedding | 3 | 2 | **6** | 0.5 | Kidney checkpoints |
| 7 | E15 | Zero-shot PBMC-10K clustering, adding the missing PCA/HVG baseline | JEPA pretraining gives better zero-shot cell-state embeddings than PCA, not only than scGPT | AvgBIO (NMI/ARI/ASW); beats PCA-50 on HVGs and kidney-scGPT | 3 | 2 | **6** | 0.3 | Kidney checkpoints; PBMC-10K download |
| 8 | E16 | Low-data regime: fine-tune on 25 % / 50 % of training perturbations | JEPA fine-tuning is more data-efficient | Delta RMSE on the fixed test set; beats scGPT and linear trained on the same subset | 3 | 2 | **6** | 8–12 | Public `scGPT_human` (can run now) |
| 9 | E5 | Delta-aware JEPA loss (existing plan) | A delta-aware JEPA objective fixes Cell-JEPA's delta regression | Delta metrics; beat linear and scGPT | 3 | 1 | **3** | 6 | Public `scGPT_human` or kidney |

Ease: 5 = under ~50 new lines on existing outputs, no new training. Promise: probability of a clean win over **both** baselines; 3 means roughly a coin flip, 2 means plausible against scGPT but unlikely against linear/PCA, 1 means unlikely. All are guesses.

---

### 1. E9 — Few-cell perturbation identification (top pick)

- **Hypothesis.** With k ≈ 5 cells per query, the frozen kidney-JEPA embedding identifies which Adamson perturbation the cells received more accurately than PCA of log1p expression and than the matched kidney-scGPT embedding.
- **Why JEPA should win.** The JEPA objective trains the student to predict the teacher's embedding of the complete cell from a masked or subsampled view, which is the denoising-of-sparse-cells mechanism the paper credits for its clustering gain (paper: Cell-JEPA §1/§4, "dropout-robust"). That should help most when there are few cells to average over. At large k, averaging removes the noise and PCA should catch up; we expect that, and the claim is limited to small k.
- **Setup on this codebase.** New `src/probe.py` (~120 lines):
  - `embed(ckpt_dir, input_transform, cells)`: build `Trainer(ds, ckpt_dir, TrainConfig(method="scgpt", seed=1, ckpt=..., input_transform=...))` and call `tr._encode_pooled(tr.model, cells, 64)` (mean-pooled gene outputs, no perturbation flags). Use `bins50` for kidney checkpoints and `log1p` for `scGPT_human`, as in Track A.
  - Whiten every arm's embedding with the mean and std of **control** cells.
  - PCA arm: 50 PCs of log1p expression fitted on control cells only, so it is unsupervised like the encoders.
  - Per perturbation (all ~86; no labels are used by the frozen encoders): split its cells once (seed 0) into a reference half and a query half. References are the mean embeddings of the reference half. Queries are the mean of k random query cells, 200 draws per perturbation, scored by cosine nearest centroid.
  - Report top-1, top-5 and MRR at k ∈ {1, 3, 5, 10, 25}.
  - Add a `probe` stage to `modal_app.py` (~30 lines; GPU for embedding, CPU for scoring).
  - Public variant, runnable now: `scGPT_human`, plus the fine-tuned Track B `scgpt` vs `celljepa` encoders (`tr.fresh_model_from(best.pt)`), restricted to the 22 test perturbations (measured: README split 58/7/22) because those encoders saw the training perturbations.
- **Matched controls.** Kidney-JEPA vs kidney-scGPT (identical schedule, Track A), the same cells, k-draws and whitening for every arm, plus PCA-50 and a random-init scGPT encoder (to show pretraining matters). Report the chance level (1/86).
- **Pass/fail (declared now).** Primary: top-1 accuracy at **k = 5**, over all perturbations. Pass = kidney-JEPA > PCA-50 **and** > kidney-scGPT, each with a paired bootstrap 95 % CI over perturbations excluding 0 (`evaluate.paired_bootstrap_diff`). Report the other k values as a curve without picking among them. Secondary (not headline): the same metric restricted to perturbations with ≥ 10 DE genes, computed from the reference half only.
- **Main risk.**
  - Kidney-only pretraining applied zero-shot to K562 is a domain shift, and in-domain PCA is a strong baseline.
  - Smoke tests showed near-constant `<cls>` embeddings (`docs/track_a.md`). Gate first: if `teacher_diagnostics.raw_centroid_mean_pairwise_cos` from existing `celljepa` runs is ≈ 1, expect failure. We use mean pooling and whitening to reduce this risk.
  - Many Adamson knockdowns have weak effects, so every arm may sit near chance on those perturbations.

### 2. E10 — Linear + deep blend

- **Hypothesis.** `0.5·linear + 0.5·celljepa` has lower delta RMSE than linear alone and than `0.5·linear + 0.5·scgpt`.
- **Why JEPA should win.** The linear model only sees the training-perturbation PCA subspace (K = 10). A deep model that conditions on the gene identity through pretrained token embeddings may add orthogonal signal. JEPA's per-cell latent targets could make that signal less noisy. This mechanism is weak, so the promise is low.
- **Setup.** New `src/blend.py` (~40 lines, CPU). Read `test_predictions.npz` from `baselines/linear`, `baselines/training_mean` and `runs/{scgpt,celljepa}_s{1,2}_*`. Form the fixed 50/50 blends and score them with `evaluate.evaluate_predictions` and `paired_bootstrap_diff`. The weight is **pre-declared at 0.5**, not tuned (validation has only 7 perturbations, measured), so no validation predictions are needed.
- **Matched controls.** `blend(linear, training_mean)` is mandatory, because any blend gains from shrinkage toward the mean. Also `blend(linear, scgpt)` and each arm alone. Use the same seeds and split.
- **Pass/fail.** Pass = `blend(linear, celljepa)` beats all three of `linear`, `blend(linear, training_mean)` and `blend(linear, scgpt)` on mean delta RMSE, with a paired-bootstrap CI excluding 0, in **both** seeds. DE20 Δ-Pearson is reported as secondary.
- **Main risk.** The gain is shrinkage, so the training-mean blend matches it. Cell-JEPA's known delta weakness (paper: Table 4) may make its blend worse than the scGPT blend.

### 3. E11 — Per-cell and distribution-level metrics

- **Hypothesis.** Cell-JEPA's predicted *populations* (one prediction per control input) are closer in distribution to the observed treated cells than scGPT's and than a linear shift model's.
- **Why JEPA should win.** The paper's only perturbation win was absolute-state fidelity (paper: Table 3, Adamson Pearson 0.937 vs 0.905), which `docs/track_a.md` suspects is a per-cell protocol. A model trained to land in a teacher's "realistic cell" embedding might produce more realistic cells.
- **Setup.** `Trainer.evaluate_per_cell` already computes per-cell Pearsons (`src/evaluate.py::percell_metrics`) but discards the per-cell predictions. Save them (~5 lines in `src/train.py`, or recompute from `best.pt`, ~0.3 GPU-h guess). Then:
  - Add linear-per-cell = `X[ctrl_i] + (linear_pred − ctrl_ref)` and training-mean-per-cell (~25 lines in `src/baselines.py`).
  - Add `energy_distance(pred_cells, obs_cells)` on the 20 DE genes (`ds.de_idx`) and in 50-PC space fitted on training cells (~30 lines in `src/evaluate.py`).
- **Matched controls.** The same 200 fixed `percell_pairs` per condition for every arm, plus linear-per-cell, training-mean-per-cell, scGPT and the observed-control population itself (no-change).
- **Pass/fail.** Primary = mean energy distance (DE genes) over test perturbations. Pass = celljepa < scGPT **and** < linear-per-cell, with a paired-bootstrap CI excluding 0, in both seeds.
- **Main risk.**
  - The shift model `ctrl_i + Δ` keeps real control heterogeneity and is a hard baseline.
  - Per-cell *absolute* Pearson rewards smoothing toward the mean, so a constant mean prediction would win it. That is why it is not the primary metric.

### 4. E13 — Same linear head, swapped embeddings

- **Hypothesis.** Ahlmann-Eltze's linear model with the perturbation embedding `P` taken from the JEPA encoder beats the same model with `P` from the matched scGPT encoder, and from training PCA.
- **Why JEPA should win.** Linear probing is the standard way to test JEPA representations. This isolates representation quality from fine-tuning dynamics.
- **Setup.** In `src/baselines.py`, generalise `linear_gene_embedding` to accept an external `P` (~20 lines). In `src/probe.py`, compute `P[g]` = whitened mean pooled embedding of 256 control cells with the perturbation flag on gene g minus the flag-off embedding (~30 lines). `K` and `λ` stay at the current defaults (10, 0.1); no tuning.
- **Matched controls.** The same `G` (training PCA), the same λ/K and the same training perturbations. Only `P` changes: PCA, kidney-scGPT, kidney-JEPA, random encoder.
- **Pass/fail.** Pass = kidney-JEPA-P beats both PCA-P and kidney-scGPT-P on mean delta RMSE (paired CI excluding 0).
- **Main risk.** In the source benchmark, foundation-model embeddings did not beat training-data PCA embeddings for unseen single perturbations (paper: Ahlmann-Eltze et al. 2025; recalled, **[VERIFY]** the exact figure). A related benchmark found prior-knowledge features beat foundation models (paper: Csendes et al., *BMC Genomics* 2025, doi:10.1186/s12864-025-11600-2). A win over scGPT-P alone is the realistic outcome.

### 5. E14 — Robustness to thinned control inputs

- **Hypothesis.** When control inputs are binomially thinned to 50 % / 25 % / 10 % of their counts, celljepa's delta RMSE rises less than scGPT's.
- **Why JEPA should win.** The robustness-to-sparsity argument (paper: Cell-JEPA §2, stochastic subsampling as "biological dropout").
- **Setup.** Thin `X_in` rows at evaluation only (~20 lines: an `input_thin` argument in `Trainer.predict` / `_forward_controls`, applied to the shared `eval_controls`). Inference only, from `best.pt` of existing runs. Adamson X is log1p-normalised, so thin on `expm1` values and re-log. This approximation is declared up front.
- **Matched controls.** The same thinned inputs (fixed seed) for every model, plus linear-per-cell under the same thinning. The mean linear model ignores the input, so it is invariant by construction. Report that plainly: this candidate can beat scGPT and linear-per-cell, never the mean linear model.
- **Pass/fail.** Primary = slope of delta RMSE vs thinning rate. Pass = the celljepa slope is smaller than scGPT's in both seeds, with a paired-bootstrap CI.
- **Main risk.** Predictions average 256 controls, which washes out input noise, so all slopes may be ≈ 0. It also does not beat linear in the usual sense.

### 6. E12 — Few-shot response estimation

- **Hypothesis.** Given k = 5 measured cells of a held-out perturbation, decoding their mean JEPA embedding shift back to expression estimates the full-population Δ better than the raw or PCA-denoised k-cell mean.
- **Why JEPA should win.** The same denoising argument as E9, now applied to a regression readout.
- **Setup.** Reuse `src/probe.py` embeddings (~60 lines). Fit a ridge map from embedding shift to expression Δ on the **58 training perturbations** (measured), then apply it to k query cells of each test perturbation. Truth = the mean of the *other* cells of that perturbation (disjoint from the k cells).
- **Matched controls.** Raw k-mean; k-mean projected onto the linear model's `G` (training PCA, the linear denoiser); the scGPT-embedding ridge; the linear model's zero-shot prediction (k = 0).
- **Pass/fail.** Pass = JEPA beats the PCA-projected k-mean and scGPT at k = 5 (paired CI).
- **Main risk.** A ridge decoder fitted on 58 points is the bottleneck. PCA projection is a very strong denoiser for pseudobulk.

### 7. E15 — Zero-shot PBMC clustering with a PCA baseline

- **Hypothesis.** Kidney-JEPA zero-shot AvgBIO beats PCA-50 on HVGs, not only scGPT.
- **Why JEPA should win.** This is the paper's headline result (paper: Cell-JEPA Table 2, 0.719 vs 0.529). It never compared against PCA.
- **Setup.** New script (~120 lines): PBMC-10K via the scvi-tools pipeline the paper names (paper: Cell-JEPA §3.2), embeddings from `src/probe.py`, and `scib-metrics` NMI/ARI/ASW, so AvgBIO matches the definition in paper App. D.1.
- **Matched controls.** Kidney-scGPT, PCA-50 on 2k HVGs, and a random encoder; the same cells and the same Leiden/KMeans procedure.
- **Pass/fail.** Pass = kidney-JEPA AvgBIO > PCA and > kidney-scGPT.
- **Main risk.** On the same PBMC 10k data, plain integration baselines already reach AvgBIO 0.72–0.78 (paper: scGPT, Cui et al. *Nat. Methods* 2024, Fig. 3a: Seurat 0.724, scVI 0.753, Harmony 0.784). A zero-shot benchmark found that HVG/PCA baselines often beat scGPT zero-shot (recalled: Kedzierska et al., *Genome Biol.* 2025, **[VERIFY]**). JEPA's 0.72 may therefore not beat PCA. This is also not a perturbation result.

### 8. E16 — Low-data fine-tuning

- **Hypothesis.** With 25 % or 50 % of the training perturbations, celljepa beats scGPT and the linear model fitted on the same subset.
- **Why JEPA should win.** The JEPA term adds per-cell latent supervision, which acts as a regulariser that might matter more when data is scarce.
- **Setup.**
  - Add a `train_fraction` field to `TrainConfig` (~15 lines in `src/train.py`) that subsamples `ds.split["train"]` with a fixed seed.
  - Make the linear baseline take the same subset (~5 lines).
  - Keep the test set fixed.
- **Matched controls.** The same subset for every arm; training-mean on the same subset.
- **Pass/fail.** Pass = celljepa beats both at 25 % in both seeds (paired CI).
- **Main risk.** This is the most expensive candidate (~12 fine-tunes, guess). The training-mean baseline is strongest when data is scarce.

### 9. E5 — Delta-aware JEPA loss (existing plan)

Kept for completeness. With 22 test perturbations (measured) and published evidence that linear/mean baselines win on delta metrics (paper: Ahlmann-Eltze 2025; Systema 2025), beating **linear** here is unlikely, even if E5 recovers Cell-JEPA's delta loss against scGPT. See §3 below for the setup. Treat a win over scGPT as the realistic ceiling.

---

### Do this first

**Top pick: E9 (few-cell perturbation identification).** It is the only candidate that tests the mechanism the paper actually demonstrated (robust embeddings of sparse cells) on the perturbation data, needs no training, and its public calibration arm can run while kidney pretraining is in progress. First concrete change:

1. Write `src/probe.py` with `embed()` (wrapping `Trainer._encode_pooled`), control-cell whitening, a PCA-50 arm and the k-cell nearest-centroid scorer.
2. Add a `probe` stage to `modal_app.py` and run it on `scGPT_human` + PCA + a random encoder for k ∈ {1, 3, 5, 10, 25}. This fixes the noise floor and chance level before any JEPA number exists.
3. Before relying on E9, check `teacher_diagnostics.raw_centroid_mean_pairwise_cos` in an existing `celljepa` run's `metrics.json`. If it is ≈ 1, embeddings barely separate perturbations; deprioritise E9/E12/E13 and lead with E10.

In parallel, E10 is a ~40-line CPU script over existing `test_predictions.npz` and should be run as soon as the Track B `scgpt` and `celljepa` runs finish. When the kidney checkpoints land, add the two kidney arms to the same `probe` stage.

---

# Cell-JEPA rematch: experiment plan

Goal: show whether a delta-aware JEPA beats Cell-JEPA's *effect-size* (delta) metrics on K562 perturbation data, and where it sits against trivial baselines. Every claim in the final table must come from numbers we re-ran on our own split, never from numbers copied between papers.

Status of this file: written without seeing the in-house Cell-JEPA code, so commands are placeholders (`<ENTRY>`). Items marked **[VERIFY]** are from memory or from a single unconfirmed source.

---

## 0. Ground rules

- **Data:** GEARS `PertData` (`norman`, `adamson`), K562. Controls are `condition == "ctrl"`.
- **Splits:** `prepare_split(split="simulation", seed=s)` for seeds 1..5. Norman also needs the combo subgroups (`combo_seen0/1/2`) from `gears.inference` analysis. Report mean ± std over seeds. Cell-JEPA's own seeds are unknown.
- **Metrics** (match `gears.inference.compute_metrics`, `deeper_analysis`, `non_dropout_analysis`):
  - absolute: `pearson`, `pearson_de`, `pearson_top20_de_non_dropout`
  - **delta (the headline):** `pearson_delta`, `pearson_delta_de`, `pearson_delta_top20_de_non_dropout`, where delta = (predicted mean − control mean) vs (true mean − control mean), per perturbation, then averaged.
- **Calibration gate (E1):** a single fetch of the scGPT tutorial showed Adamson `unseen_single` delta metrics of about 0.62 / 0.80 / 0.79 **[VERIFY]**, while the Cell-JEPA table lists its scGPT baseline at 0.17 / 0.10 / 0.12. Do not compare to the paper's table until our metric code reproduces the tutorial and we understand the gap (different pretraining, or different metric definition).
- **Controls:** every training run uses the identical split object and identical control-sampling rule. Log the rule.
- **Seeds:** 5 per config unless noted. Log git commit, config JSON, GPU type.

## 1. Numbers to beat (from the Cell-JEPA paper, arXiv 2602.02093)

Delta metrics, `unseen_single`, order = Δ-Pearson / Δ-Pearson DE / Δ-top20:

| | Norman | Adamson |
|---|---|---|
| Cell-JEPA | 0.059 / 0.041 / 0.115 | 0.120 / **−0.176** / **−0.152** |
| scGPT (their kidney-pretrained copy) | 0.103 / 0.105 / 0.169 | 0.173 / 0.101 / 0.120 |

Absolute metrics (Pearson / DE / top20): Norman Cell-JEPA 0.787 / 0.592 / 0.565 vs scGPT 0.631 / 0.276 / 0.278; Adamson 0.937 / 0.741 / 0.720 vs 0.905 / 0.699 / 0.677. The paper has no mean, additive or linear baseline, no ablations, no error bars.

## 2. Run contract (what the code must expose)

One entry point, so the same code runs locally and on Modal:

```
<ENTRY> --dataset {norman,adamson} --split-seed S --exp E --config path.yaml --out results/E/RUNID/
```

Each run writes `results/<exp>/<run_id>/metrics.json` containing: `dataset, split, seed, exp, config_hash, git_commit, metrics{...all 6 above...}, per_perturbation{pert: {delta_pearson, ...}}, train_time_s, gpu`. Per-perturbation numbers are required for the "which perturbations win" plots and the demo.

A separate `aggregate.py` reads all `metrics.json` and prints the table in section 6.

## 3. Experiments

Priority: **P0** run first, gates the rest. **P1** main contribution. **P2** only if time.

### P0

| ID | Experiment | Config | Output / pass condition |
|---|---|---|---|
| E0 | **Trivial baselines**, no training: (a) no-change = control mean; (b) control mean + mean of training deltas; (c) additive (Norman doubles); (d) ridge / PCA regression from perturbation features to delta | CPU, all seeds/splits | Table rows for every later run to sit next to. If (b) already beats Cell-JEPA's delta numbers, the headline becomes "delta-aware JEPA vs the strongest simple baseline." |
| E1 | **Reproduce the scGPT perturbation tutorial** and our metric code | scGPT tutorial config, Adamson, seed 1 | Our numbers match the tutorial's within noise. Then explain the gap to the paper's scGPT row. |
| E2 | **Teacher diagnostic**: cosine between frozen-teacher embeddings of matched control and perturbed cells | any trained/pretrained checkpoint | If cosine is ~1, the JEPA target barely sees perturbation signal. Report the distribution per perturbation. |
| E-data | **UPF1 data check** (gates E8): load Replogle K562 essential via `pertpy` (`replogle_2022_k562_essential`), search `UPF1, UPF2, SMG1, SMG5, SMG6, SMG7` | one-off notebook | Record cells per gene. If UPF1 is absent or tiny, drop E8. |

### P1

| ID | Experiment | Variants | Question |
|---|---|---|---|
| PT1 | **Pretraining ablation** on the same kidney set | (a) w_JEPA=0 (reconstruction only), (b) w_JEPA=1000 as published, 1 seed each | The ablation the paper never ran: does JEPA itself help, or does an extra loss at weight 1000 change training? Use CELLxGENE Census snapshot 2023-05-15 to match **[VERIFY]**. Run (a) and (b) concurrently. |
| E3 | **Fine-tune loss ablation** | MSE only; +JEPA; +ECS; full (paper's E.3 weights: JEPA 1.0, rec 1.0, ECS 0.8). Sweep w_JEPA ∈ {0, 0.1, 1, 10}. ECS off. | Which term helps delta metrics. Hypothesis: ECS may flatten structure on K562-only data (all cells share one cell type), so ECS-off could help. |
| E3b | **Collapse guard** | add VICReg-style variance hinge on the cell embedding (see `toy-jepa/toy_jepa.py::variance_hinge`), weights {0, 0.1, 1} | Cell-JEPA has only EMA/frozen-teacher + a reconstruction anchor. Does an explicit variance term change delta metrics? Also log embedding std per epoch. |
| E4 | **Frozen vs fine-tuned encoder** | pretrained encoder frozen (train perturbation head only) vs full fine-tune | Whether the pretrained encoder suppresses perturbation signal. |
| E5 | **Delta-aware loss** | add `L_delta = 1 − cos(ẑ_pert − z_ctrl, z_pert − z_ctrl)` with stop-gradient on the target side; also a DE-weighted delta MSE on expression; weights {0.1, 1, 10} | The main proposed improvement. Compare to E3 full. |
| E6 | **Norman combos** | `combo_seen0/1/2` for the best E3/E5 config vs additive from E0(c) | Do we beat additive on doubles? The paper never evaluated combos. |

### P2

| ID | Experiment | Notes |
|---|---|---|
| E7 | **Module masking** | Mask gene blocks (graph/coexpression modules, BioM-JEPA style) instead of random 15% values; ratios {0.15, 0.4}. |
| E8 | **UPF1 hold-out** (only if E-data passes) | Replogle K562 custom split holding out UPF1 (then each of UPF2/SMG1/5/6/7 separately). Score vs (i) training-mean delta and (ii) mean of the other NMD members' deltas. Also check whether known NMD targets move the right way in the predicted delta. Not a true mutation benchmark: it is knockdown, not variant data. |

## 4. Open questions to resolve in the code review (from the paper read-through)

1. Which view is masked: student masked, teacher full? (Paper text and equations disagree; equations and Fig. 2 say student masked.)
2. EMA momentum value and schedule (not reported). Teacher in `eval()` mode, no dropout?
3. Are perturbation-mode values binned or continuous? (Not stated.)
4. How controls are paired with perturbed cells (random vs matched) **[VERIFY]**: not stated in the paper.
5. What labels ECS sees on K562-only data.
6. Perturbation fine-tune: frozen teacher copy (App. E.3) vs EMA (Sec. 2.5); JEPA weight 1.0 there vs 1000 in pretraining.
7. Delta metric uses control *mean* (GEARS convention) or matched control?
8. No overlap between pretraining cells (kidney) and evaluation data; DE gene lists computed from training data only.

## 5. Modal skeleton

Written from memory of the Modal API **[VERIFY against current docs]**: GPU names, `add_local_dir` vs `Mount`, and whether B300 is offered.

```python
# modal_run.py
import json, itertools, modal

app = modal.App("cell-jepa-rematch")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "torch_geometric", "cell-gears", "scanpy",
                 "scikit-learn", "pandas", "matplotlib", "scipy", "pyyaml")
    .add_local_dir(".", remote_path="/root/code")   # your repo
)

data = modal.Volume.from_name("cj-data", create_if_missing=True)      # GEARS data, kidney tensors
out  = modal.Volume.from_name("cj-results", create_if_missing=True)   # checkpoints, metrics

@app.function(image=image, gpu="H100", timeout=6 * 3600,
              volumes={"/data": data, "/results": out})
def run(cfg: dict) -> dict:
    import subprocess, os
    os.chdir("/root/code")
    path = f"/results/{cfg['exp']}/{cfg['run_id']}"
    os.makedirs(path, exist_ok=True)
    json.dump(cfg, open(f"{path}/config.json", "w"))
    subprocess.run(["python", "<ENTRY>", "--dataset", cfg["dataset"],
                    "--split-seed", str(cfg["seed"]), "--exp", cfg["exp"],
                    "--config", f"{path}/config.json", "--out", path,
                    "--data-dir", "/data"], check=True)
    out.commit()
    return json.load(open(f"{path}/metrics.json"))

@app.local_entrypoint()
def main(exp: str = "E3", dataset: str = "adamson", seeds: int = 5):
    configs = [dict(exp=exp, dataset=dataset, seed=s, run_id=f"{exp}-{dataset}-s{s}")
               for s in range(1, seeds + 1)]
    for r in run.map(configs):
        print(r["metrics"])
```

Practical notes:
- Run `modal run modal_run.py --exp E0` for CPU-only baselines (drop the `gpu=` argument for those).
- Download datasets **once** into the `cj-data` volume, not per run, and commit the volume. Preprocess the kidney pretraining set (binning, tokenization) to tensors there.
- A 28M-parameter model will not saturate an H100: pack several seeds per GPU or use `.map` across containers. Measure throughput in the first 10 minutes before estimating wall time. My estimate, not measured: full pretraining 3-6 h, fine-tune 15-30 min.
- Keep one run exactly as published (batch 32, lr 1e-4) as the reference; treat large-batch runs as a separate variant.
- Pretrained scGPT_human checkpoint (`vocab.json`, `args.json`, `best_model.pt`) must also be on the data volume for E1.

## 6. Results template

One table per dataset, `unseen_single`, mean ± std over 5 seeds. Rows in this order:

| Model | Δ-Pearson | Δ-Pearson DE | Δ-top20 | Pearson | Pearson DE |
|---|---|---|---|---|---|
| no-change (E0a) | | | | | |
| mean delta (E0b) | | | | | |
| linear / PCA (E0d) | | | | | |
| scGPT re-run (E1) | | | | | |
| Cell-JEPA as published (paper table) | 0.059 / 0.041 / 0.115 (Norman) | | | | |
| in-house Cell-JEPA (PT1b + E3 full) | | | | | |
| + delta-aware loss (E5) | | | | | |
| best variant | | | | | |

Plus Norman doubles: additive vs ours, and per-perturbation scatter of delta-Pearson vs `corr_with_shared` (from `explore_perturb.py`).

## 7. Order of operations

1. Hours 0-2: E-data, E0, E1, E2. Decide whether the paper's table is beatable, and whether UPF1 is feasible.
2. Launch PT1 (a) and (b) as early as the data pipeline works, since they take longest.
3. While they run: E3, E3b, E4, E5 on the existing scGPT_human or published checkpoint.
4. Then E6; then E7/E8 if time.
5. Keep 4-6 hours at the end for the table, the demo, and writing up.
