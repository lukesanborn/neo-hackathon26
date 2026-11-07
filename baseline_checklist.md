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
- 🔄 **Main pretraining runs at lr 3e-5** (A9): scGPT, JEPA `faithful` (paper), JEPA `token`, JEPA `center` (~1.5 h)
- ✅ JEPA `token` and `center` at lr 1e-4 finished (val recon 18.07 / 18.12, no collapse). Kept as extra checkpoints.
- ⬜ Confirm the A9 runs finish without collapse (teacher spread > 0.01) and export checkpoints
- ⬜ Record final lr and runtime in the `docs/track_a.md` deviation log

## 2. Task 2: zero-shot cell-type clustering on PBMC-10K (paper's headline; no training needed)
- ⬜ Download PBMC-10K, processed as in the paper (scvi-tools pipeline: 3,346 HVGs, 2 batches, 11,990 cells, cell-type labels)
- ⬜ Embedding extraction: run each frozen checkpoint, take the `<cls>` output (same binning/tokenisation as pretraining)
- ⬜ Metrics: AvgBIO = mean(NMI, ARI, ASW) on cell-type labels, scGPT / scIB conventions (Leiden-cluster resolution sweep for NMI/ARI)
- ⬜ Run on: kidney-JEPA, kidney-scGPT, public `scGPT_human` (reference), **PCA-50 on HVGs (baseline the paper omitted)**
- ⬜ Compare with Table 2 (0.719 vs 0.529)

## 3. Task 1: fine-tuning for cell-type clustering on PBMC-10K
- ⬜ Implement the fine-tune loop: GEP + GEPC + ECS (+ JEPA for Cell-JEPA), 40% masking, Adam 1e-4 ×0.9/epoch, batch 64, 30 epochs, 9:1 split with seed 42 (App. E.2)
- ⬜ GEPC loss (not implemented anywhere yet)
- ⬜ Same metrics as Task 2; compare with Table 1 (0.783 vs 0.753)

## 4. Task 3: perturbation prediction
- ✅ Adamson data contract, GEARS simulation split (seed 1), shared evaluation controls
- ✅ Baselines: no-change (0.0915), training-mean (0.0588), **linear (0.0575)** test delta RMSE
- ✅ Fine-tune recipes: Cell-JEPA (MSE + 1.0·JEPA + 0.8·ECS) and paper-matched scGPT (MSE + 0.8·ECS)
- ✅ Pipeline tested end to end: public scGPT (4 runs) and Luke's short kidney checkpoints (10 runs) (A4)
- ⬜ **Main Track A fine-tunes** from the A9 checkpoints: {kidney-JEPA, kidney-scGPT} × {Cell-JEPA FT, scGPT FT} × seeds {1,2} = 8 runs
- ⚠️ Metric scale mismatch: ours are mean-based (absolute Pearson ~0.99); the paper reports 0.905. We save per-cell metrics (`test_percell`) but haven't checked them against the paper.
- ⚠️ Some fine-tunes produce identical, input-independent predictions (A4b). Cause unknown.
- ⬜ Norman dataset (CRISPRa, singles + combos; `combo_seen0/1/2` subgroups). Not implemented.
- ⬜ Compare with Tables 3–4

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
- ⬜ E9: few-cell perturbation identification from frozen embeddings (top-ranked in `experiment.md`; no training)
- ⬜ Cell-type / cell-state annotation via a linear probe on frozen embeddings (reuses the Task 2 code)
- ⬜ In-silico perturbation ranking with the fine-tuned predictor (prioritise knockdowns against a target expression profile)
