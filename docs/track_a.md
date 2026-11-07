# Track A — Cell-JEPA reimplemented from the paper (unofficial)

Cell-JEPA ([arXiv 2602.02093](https://arxiv.org/abs/2602.02093)) has no public code or
checkpoints. We have asked the corresponding author for them; until they arrive, Track A is
**our reimplementation from the paper text**. Every run, table and plot carries the label
"Track A — Cell-JEPA reimplemented from the paper (unofficial)". Track A numbers are never put in
the same matched comparison as Track B (public scGPT + our JEPA auxiliary). The Track A report may
show Track B only in a separate "for context only" section.

Pipeline: CELLxGENE Census 2023-05-15 kidney cells → matched kidney pretraining (JEPA objective
and scGPT objective, identical schedule) → Adamson perturbation fine-tuning of each checkpoint with
the `scgpt_paper` and `celljepa` recipes (seeds 1, 2) → Track A report. `scgpt_paper` is the
paper's scGPT fine-tuning baseline (§2.5 / App. E.3: MSE + 0.8·ECS on `<cls>`) — the `celljepa`
objective without the JEPA term, on the identical code path, init, pairing and RNG streams — so
the two fine-tuning arms differ only in the JEPA loss. (Track B's `scgpt` method, MSE only and
no `<cls>`, is unchanged.) Separately, we run the Cell-JEPA
fine-tuning recipe on public `scGPT_human` (experiment `adamson_celljepa_public_v1`).

## Apps

| App | File | What it runs |
|---|---|---|
| `jepa-census` | `modal_census.py` | kidney Census cache → `/vol/trackA/kidney_census_2023-05-15/` |
| `jepa-pretrain` | `modal_pretrain.py` | pretraining pilot (`/vol/trackA/pretrain_pilot/<gpu>.json`) and pretraining → `/vol/trackA/checkpoints/<objective>_kidney/` |
| `jepa-perturb` | `modal_app.py` | `prepare`, `train_blackwell(..., ckpt, input_transform)`, `aggregate` |
| `jepa-reference` | `modal_reference.py` | GEARS-style metrics vs published numbers |
| `jepa-track-a` | `modal_track_a.py` | orchestration (this doc) + Track A report (CPU) |

All share the Volume `jepa-perturb` mounted at `/vol`. Function names the orchestrator calls are
constants at the top of `modal_track_a.py` (`CENSUS_FN`, `PILOT_FN`, `PRETRAIN_FN`, `TRAIN_FN`, …);
adjust there if a unit names a function differently. `--stage dry-run` reports every missing
function and checks our keyword arguments against the function's source in the checkout.

## Run book (in order)

```bash
source .venv/bin/activate

# 0. deploy every app (GPU apps need GPU billing on the workspace)
modal deploy modal_census.py
modal deploy modal_pretrain.py
modal deploy modal_app.py
modal deploy modal_reference.py
modal deploy modal_track_a.py          # optional: `modal run` works without deploying this one

# 1. dry run: lookups + argument checks + volume state; nothing executes
modal run modal_track_a.py --stage dry-run

# 2. kidney Census cache (CPU): smoke first, then full
modal run modal_track_a.py --stage census --census-stage smoke
modal run modal_track_a.py --stage census --census-stage full

# 3. pretraining pilot on B300 for both objectives: prints s/step, cells/s, projected hours and
#    cost for the paper schedule (44 ep, bs 32) and alternatives, then STOPS
modal run modal_track_a.py --stage pilot

# 4. decide ONE schedule for both objectives, record it in the deviation log below, then:
modal run --detach modal_track_a.py --stage pretrain --confirm-pretrain \
    --pt-batch-size <B> --pt-epochs <E> --pt-lr <LR> --pt-max-minutes <M>
#    prints the two FunctionCall ids and exits (use --wait yes to block); later:
modal run modal_track_a.py --stage collect --call-ids <id1>,<id2>

# 5. fine-tune: prepare(adamson_trackA_v1) then 2 checkpoints x {scgpt_paper, celljepa} x seeds {1,2}
#    = 8 runs on B300, max 4 concurrent; refuses until both checkpoints have args.json,
#    best_model.pt, vocab.json. Redeploy jepa-perturb from this checkout first: preflight checks
#    `scgpt_paper` against the LOCAL src/train.py METHODS, not the deployed app
#    (`modal deploy modal_app.py`)
modal run --detach modal_track_a.py --stage finetune

# 6. Cell-JEPA fine-tuning recipe on public scGPT_human (separate experiment, seeds 1,2)
modal run --detach modal_track_a.py --stage finetune-public

# 7. Track A report -> /vol/experiments/adamson_trackA_v1/report_track_a/
modal run modal_track_a.py --stage report
modal run modal_reference.py --exp-id adamson_trackA_v1          # GEARS-style metrics (unit 4)
modal volume get jepa-perturb experiments/adamson_trackA_v1/report_track_a ./reports/track_a
```

Smoke test of the report path (synthetic runs under `/vol/smoke/track_a/`, no GPU):
`modal run modal_track_a.py --stage report-smoke`.

Useful flags: `--gpu B300`, `--seeds 1,2`, `--ft-epochs 15`, `--ft-max-minutes 150`,
`--tag v1` (runs are immutable; a new tag makes new run dirs), `--pt-extra-kwargs '{"warmup":...}'`
and `--pilot-kwargs '{...}'` (passed through verbatim), `--usd-per-gpu-hour`, `--n-cells`.

### Layout

```text
/vol/trackA/kidney_census_2023-05-15/   manifest.json, shards/shard_XXX.npz, split.json
/vol/trackA/pretrain_pilot/<gpu>.json   pilot reports
/vol/trackA/checkpoints/{jepa,scgpt}_kidney/   args.json, best_model.pt, vocab.json
/vol/experiments/adamson_trackA_v1/
  baselines/{no_change,training_mean,linear}/   same split (GEARS simulation seed 1) as Track B
  runs/{scgpt_paper,celljepa}_s<seed>_kidneyjepa_v1/    ckpt=/vol/trackA/checkpoints/jepa_kidney, input bins50
  runs/{scgpt_paper,celljepa}_s<seed>_kidneyscgpt_v1/   ckpt=/vol/trackA/checkpoints/scgpt_kidney, input bins50
  report_track_a/   results.md, report.json, results_*.csv, per_intervention_all.csv, run_status.csv,
                    track_a_delta_rmse.png
/vol/experiments/adamson_celljepa_public_v1/runs/celljepa_s<seed>_v1/   public scGPT_human, log1p
```

The Track A report keys runs by **(checkpoint, method, seed)** read from each run's `config.json`
(`src/track_a/report.py`); `src/report.collect_runs` keys by (method, seed) and must not be used
for this experiment.

### Key comparison

The paper's comparison: **kidney-JEPA pretraining + Cell-JEPA fine-tuning** (`jepa_kidney/celljepa`)
vs **kidney-scGPT pretraining + scGPT fine-tuning** (`scgpt_kidney/scgpt_paper`; matched schedule),
plus baselines. As in the paper (§2.5), both fine-tuning objectives share MSE + 0.8·ECS on the
student `<cls>`; `celljepa` adds only the 1.0·JEPA term. The two cross arms
(`jepa_kidney/scgpt_paper`, `scgpt_kidney/celljepa`) separate the pretraining and fine-tuning
(JEPA-term) effects. We report our primary delta RMSE and, when `per_intervention.csv` has them, the GEARS
columns next to the published Adamson values:

| | Pearson | Pearson DE | top-20 DE (non-dropout) | Δ Pearson | Δ Pearson DE | Δ top-20 DE |
|---|---|---|---|---|---|---|
| scGPT (paper) | 0.905 | 0.699 | 0.677 | 0.173 | 0.101 | 0.120 |
| Cell-JEPA (paper) | 0.937 | 0.741 | 0.720 | 0.120 | −0.176 | −0.152 |

Extracted from the arXiv HTML (Tables 3–4); column correspondence is our reading — **spot-check
against the PDF** before quoting.

## Cost / time notes

* **Pretraining** (decided after the pilot): the paper's schedule is ≈ 814k cells × 44 epochs /
  batch 32 ≈ **1.12M optimiser steps per model**, two models. The paper reports "about one day on
  an A5000", which is hard to reconcile with 1.1M steps of a 12-layer scGPT; the pilot measures
  s/step on B300 and prints projected hours and cost (both models) for the paper schedule and
  shorter alternatives. Cost uses an *assumed* `--usd-per-gpu-hour` (default 6.25) — verify on
  modal.com/pricing. Pretraining is never launched automatically.
* **Fine-tuning**: 8 Track A runs + 2 public runs, ≤ 150 min each on B300 with ≤ 4 concurrent
  (Track B measurements: well under that budget for 15 epochs).
* **Report**: CPU only, minutes.

## Deviation log

| Item | Paper says | We do | Why |
|---|---|---|---|
| Code / checkpoints | Not released | Reimplement from the text; train both kidney checkpoints ourselves | No official artifacts; author contacted. All Track A numbers are "reimplemented (unofficial)" |
| EMA momentum (teacher) | Unspecified | 0.996 | I-JEPA / BYOL-style default |
| Predictor width / depth | Unspecified | 2-layer MLP, hidden = d_model (512) | Smallest standard choice; matches embedding width |
| Projector | Mentioned for fine-tuning | Projector used in fine-tuning only, not pretraining | Our reading of the method section |
| ECS term | Supervised contrastive with cell-type positives; weight 0.8 in both the Cell-JEPA and the scGPT fine-tuning objectives (§2.5, App. E.3) | scGPT's unsupervised ECS, threshold 0.3, weight 0.8 on student `<cls>` in both `celljepa` and the baseline `scgpt_paper` (as the paper, the baseline also uses ECS) | Adamson (K562 only) has no cell-type labels to form positives |
| Control ↔ perturbed pairing | Unspecified | Fixed per-seed random pairing (same in every arm) | Deterministic and matched across arms |
| Fine-tuning input scale | Unspecified (bins vs log1p) | `bins50` for kidney checkpoints; `log1p` for public scGPT_human | Kidney models were pretrained on binned values; public run matches Track B |
| 50 vs 51 bins | Ambiguous (n_bins 51 incl. zero vs 50 non-zero) | 50 non-zero bins + zero token (scGPT `n_bins=51` convention) | Matches scGPT preprocessing |
| Teacher in fine-tuning | Frozen (Appendix E.3) vs EMA (§2.2) — conflicting | Frozen | E.3 is the more specific setup description; avoids target drift on a small dataset |
| Checkpoint selection | Unspecified | Fine-tuning: best validation mean delta RMSE; pretraining: best validation reconstruction loss | Validation-only selection; test touched once |
| Census filter | ~800k kidney cells, filter not given | Census 2023-05-15, scGPT's cell filter, **no** `is_primary_data` filter | Reproduces scGPT's data recipe; resulting cell count is recorded in the census manifest |
| Attention mask details | Not fully specified | scGPT's standard pad masking; masked-value positions hidden from the student, visible to the teacher (confirm against `modal_pretrain.py`) | Closest standard JEPA choice |
| Δ / metric definition | Ambiguous | Two readings reported side by side: GEARS mean-based (prediction − control mean vs truth − control mean) and per-cell (each prediction vs one paired treated cell, Δ vs that cell's own control input; 200 fixed pairs per condition, shared by all methods), plus our primary delta RMSE | Mean-based scores saturate (no-change ≈ 0.99 Pearson) while the paper reports ~0.9, which fits a per-cell protocol; unconfirmed |
| Zero-valued genes in fine-tuning | Not stated (fine-tuning uses the official scGPT perturbation pipeline, which includes all genes) | Kept: all 5,060 Adamson genes are tokens; zeros stay value 0 under `bins50`; pretraining saw only nonzero genes (value 0 only at `<cls>`) | Faithful to the stated pipeline; logged because kidney models never saw zero-valued gene tokens |
| Gradient clipping | Not stated | Clip global norm 1.0 in pretraining and fine-tuning, both objectives | Stability with the 1000× JEPA weight; note it rescales most JEPA steps, applied identically to the baseline |
| JEPA target spread | Not discussed | Monitored: `teacher_cls_std` (pretraining) and `teacher_cls_pairwise_cos` / `jepa_meanpred_baseline` (fine-tuning); real-data pilot checked before full pretraining | Smoke tests showed near-constant teacher `<cls>` embeddings (risk that the JEPA term is trivially satisfied) |
| Split | Unspecified seed | GEARS `simulation` split, seed 1 (same as Track B) | Reproducible; baselines identical across tracks |
| Compute | ~1 day A5000 pretraining; 2 h vs 22 h fine-tuning reported in different places | Measured by our pilot on B300; reported actual runtimes | Paper's figures are inconsistent with ≈1.1M steps/model |
| Pretraining schedule | 44 epochs, batch 32 | **TBD after pilot** (identical for both objectives) — record chosen bs / epochs / lr / max minutes here | Cost; any reduction applies to both models equally |
| Seeds | Unspecified | Fine-tuning seeds 1, 2; one pretraining run per objective | Budget |

Fill in the "TBD after pilot" row (and the actual runtime) before quoting any Track A result.
