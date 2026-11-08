# Baseline checklist: replicating Cell-JEPA and scGPT

The goal is to reproduce the paper's three tasks with our own matched kidney-pretrained models (Cell-JEPA vs scGPT), plus simple baselines. Once that works we can (1) run experiments that change the method, and (2) use the cell embeddings or the predictor for a downstream application.

Legend: ✅ done · 🔄 running / in progress · ⬜ to do · ⚠️ done but with a known issue. Run details and numbers are in `attempts.md`.

The paper (arXiv 2602.02093) has three evaluations, all comparing kidney-pretrained Cell-JEPA vs kidney-pretrained scGPT:
- **Task 1:** fine-tune on PBMC-10K, then cluster cell types (Table 1: AvgBIO 0.783 vs 0.753)
- **Task 2:** zero-shot embeddings on PBMC-10K, then cluster (Table 2: AvgBIO **0.719 vs 0.529**, the headline result)
- **Task 3:** perturbation prediction on Adamson and Norman (Tables 3–4)

---

## 0. Infrastructure
- ✅ Modal workspace `ls9563`, volume `jepa-perturb`, all 5 apps deployed from `track-a-fixes`
- ✅ Pinned images (B300 "blackwell" stack); 88 local tests pass
- ✅ Code on GitHub: `track-a-fixes` (fixes + Luke's variants), `dangelo/establishing-baseline` (log, diagnostics)
- ⬜ Fix the `modal_track_a.py` pretrain/pilot/census stages: they pass the wrong arguments (we launch directly with `diag/` scripts for now)
- ⬜ Merge `track-a-fixes` / `dangelo/establishing-baseline` into `main` once the team agrees

## 1. Kidney pretraining (shared by all three tasks)
- ✅ Kidney corpus: Census 2023-05-15, 814,279 cells, scGPT filter, per-cell 50-bin quantile binning
- ✅ JEPA objective (student/EMA teacher, value masking 15%, attention mask, 1000×cosine + recon) and matched scGPT objective
- ✅ Paper schedule fixed: **4 epochs** (code previously said 44)
- ✅ Diagnosed plain-scGPT stall and paper-JEPA collapse at lr 1e-4; both fixed by **lr 3e-5** (12/12 seeds); backup: lr 1e-4 + 4k-step warmup (`attempts.md` A6–A10)
- ✅ **Main pretraining runs at lr 3e-5** (A9): scGPT 15.68, JEPA `faithful` 17.73, `token` 18.40, `center` 18.73 val recon
- ✅ JEPA `token` and `center` at lr 1e-4 finished (val recon 18.07 / 18.12, no collapse). Kept as extra checkpoints.
- ✅ No collapse (teacher spread 0.037–0.040); checkpoints exported as `*_kidney_lr3e5*`
- ⬜ Record final lr and runtime in the `docs/track_a.md` deviation log

## 2. Task 2: zero-shot cell-type clustering on PBMC-10K (paper's headline; no training needed)
- ✅ Download PBMC-10K, processed as in the paper (scvi-tools: 3,346 genes, 7,982 + 4,008 cells, 9 labels; exact match)
- ✅ Embedding extraction (scGPT's zero-shot procedure; `src/track_a/pbmc.py`, `modal_pbmc.py`)
- ✅ Metrics: scib-equivalent NMI / ARI / ASW / AvgBIO
- ✅ Ran on PCA-50 (**0.773**, beats the paper's Cell-JEPA 0.719), public scGPT (0.709), all existing kidney checkpoints (best 0.413) (A11)
- ✅ Run on the lr 3e-5 kidney checkpoints (A14): **JEPA 0.606 vs scGPT 0.375** (+0.23)
- ✅ Compare with Table 2 (paper 0.719 vs 0.529, +0.19): **direction and size replicate**; both ~0.12 lower in absolute terms; PCA (0.773) beats all

## 3. Task 1: fine-tuning for cell-type clustering on PBMC-10K
- ✅ Implement the fine-tune loop: GEP + GEPC + ECS (+ JEPA for Cell-JEPA), 40% masking, Adam 1e-4 ×0.9/epoch, batch 64, 30 epochs, 9:1 split with seed 42 (App. E.2) (`src/track_a/pbmc_finetune.py`)
- ✅ GEPC loss (scGPT MVC-style head)
- ✅ First runs (A12): kidney JEPA `center` 0.653; public scGPT 0.624 (*lower* than its zero-shot 0.709, needs a look); `token` 0.479
- ✅ Run on the lr 3e-5 kidney checkpoints, 3 seeds (A15): **JEPA 0.695 ± 0.046 vs scGPT 0.500 ± 0.003** (paper 0.783 vs 0.753) → replicates in direction
- ✅ Investigated (A13): noise ±0.07 between identical runs; ECS not the cause → Task 1 needs ≥ 3 seeds
- ✅ Same metrics as Task 2; compared with Table 1: direction replicates (+0.195 vs paper +0.030); absolute values lower; PCA (0.773) beats all

## 4. Task 3: perturbation prediction
- ✅ Adamson data contract, GEARS simulation split (seed 1), shared evaluation controls
- ✅ Baselines: no-change (0.0915), training-mean (0.0588), **linear (0.0575)** test delta RMSE
- ✅ Fine-tune recipes: Cell-JEPA (MSE + 1.0·JEPA + 0.8·ECS) and paper-matched scGPT (MSE + 0.8·ECS)
- ✅ Pipeline tested end to end: public scGPT (4 runs) and Luke's short kidney checkpoints (10 runs) (A4)
- ✅ **Main Track A fine-tunes** (A14, 16 runs incl. token/center): kidney JEPA + Cell-JEPA FT 0.078 vs kidney scGPT + scGPT FT 0.079 delta RMSE. JEPA marginally better on all metrics; all lose to linear (0.058)
- ⚠️ Metric scale mismatch: ours are mean-based (absolute Pearson ~0.99); the paper reports 0.905. We save per-cell metrics (`test_percell`) but haven't checked them against the paper.
- ⚠️ Some fine-tunes produce identical, input-independent predictions (A4b). Cause unknown.
- ⬜ Norman dataset (CRISPRa, singles + combos; `combo_seen0/1/2` subgroups). Not implemented.
- ✅ Compare with Tables 3–4: small absolute gain replicates (within noise); the paper's Δ-metric drop does **not** replicate; metric scale still differs

## 5. Reporting
- ✅ `attempts.md` run log; Track B report on the volume
- ⬜ One results table per task: ours vs paper, both arms, simple baselines, seeds
- ⬜ Final deviation list (lr 3e-5, grad clip, bf16, unsupervised ECS, ...)

---

## After replication

### Phase 2: experiments that change the method (ranked list in `experiment.md`)
- ⬜ Pretraining objective variants: `token` / `center` vs `faithful` (checkpoints already exist at lr 1e-4)
- ⬜ JEPA weight ablation (paper: 1000; no ablation in the paper)
- ⬜ Delta-aware JEPA loss for perturbation (paper's own suggested future work)
- ⬜ Report the escape-rate finding: at the paper LR, the JEPA term helps escape the scGPT plateau (3/4 vs 0/4 seeds), but an LR change fixes it outright

### Phase 3: downstream use of the embedding or predictor
- ✅ A16 predicted-effect retrieval on saved Adamson predictions: deep models beat linear at **identifying** held-out genes (top-1 0.16–0.23 vs 0.09; chance 0.045); public scGPT + Cell-JEPA FT best at neighbour retrieval (0.60 vs 0.56). Kidney checkpoints weak here.
- ✅ Bootstrap CIs (A17): deep>linear identification is only a trend (CI includes 0); public Cell-JEPA FT > scGPT FT on cluster agreement (+0.29, CI excludes 0)
- ✅ Replogle K562 essential prepared (A20): 412 knockdowns kept (680 excluded: target not a measured gene), 103 test genes; linear 0.0755 / training mean 0.0773 delta RMSE; ~30 min per fine-tune
- ⬜ Replogle fine-tunes (public scGPT + kidney ckpts, both recipes, seeds) + retrieval on 103 test genes
- ✅ Continued-pretraining code from public scGPT (A18) and 2M blood/marrow cache (A21)
- ✅ `<cls>` pilot (A19): paper JEPA slowly collapses from public scGPT; `center` carries per-cell signal
- ✅ Blood continued pretraining (A23): scGPT 9.13 vs JEPA `center` 10.25 val recon; no collapse
- ⬜ Step 2 screen on Replogle (validation-selected): public / blood-scGPT / blood-JEPA × 2 FT recipes × contrastive on/off × 2 seeds
- ⬜ E9: few-cell perturbation identification from frozen embeddings (top-ranked in `experiment.md`; no training)
- ⬜ Cell-type / cell-state annotation via a linear probe on frozen embeddings (reuses the Task 2 code)
- ⬜ In-silico perturbation ranking with the fine-tuned predictor (prioritise knockdowns against a target expression profile)
