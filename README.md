# Where to point CRISPR: JEPA vs scGPT for perturbation retrieval

We **reimplemented Cell-JEPA** (arXiv 2602.02093: a JEPA student/EMA-teacher objective added to scGPT) on our own pretraining runs, **replicated its three claims**, and then asked a practical question: *given an untested gene, can a model tell a researcher which measured knockdowns it will resemble, and, in reverse, which gene to knock down to get a desired effect?*

Everything heavy runs on [Modal](https://modal.com) (B300 GPUs). The full experiment log, with every number, overturned conclusion and caveat, is in **`attempts.md`** (entries A1–A29).

## Headline results

**1. Cell-JEPA replicates (direction), with our own matched kidney pretraining.** AvgBIO cell-type clustering on PBMC-10K:

| | Zero-shot (Task 2) | Fine-tuned, 3 seeds (Task 1) |
|---|---|---|
| kidney JEPA (paper method) | **0.606** | **0.695 ± 0.046** |
| kidney scGPT (matched) | 0.375 | 0.500 ± 0.003 |
| *paper: Cell-JEPA vs scGPT* | *0.719 vs 0.529* | *0.783 vs 0.753* |
| PCA-50, no model | 0.773 | – |

On Adamson perturbation prediction (Task 3), JEPA is marginally ahead of matched scGPT and both lose to a linear baseline, consistent with Ahlmann-Eltze et al. 2025. The paper's drop on change-from-control metrics did not replicate. (A14, A15)

**2. A training fix the paper didn't need to mention.** At the paper's lr 1e-4, plain scGPT pretraining stalls on a "predict each gene's median" plateau (0/4 seeds escape) and paper-JEPA collapses. lr 3e-5 (or a 4k-step warmup) trains reliably (12/12 seeds). Public scGPT itself was trained with a 10k-step warmup. (A6–A10)

**3. Perturbation retrieval on Replogle K562** (CRISPRi, 103 held-out test genes; 278 training knockdowns). Fine-tuned scGPT identifies an untested gene's response far better than the standard linear baseline:
- top-1 0.12–0.14 vs 0.01;
- ID percentile 0.68 vs 0.54.

A **contrastive retrieval loss** is the biggest single lever. JEPA, in pretraining or fine-tuning, adds ≈ 0 to retrieval. See the ladder figure `diag/results/figures/A25_ladder_bars.png` (validation genes) and A22/A25/A28 in `attempts.md` (final 5-seed test numbers: see the latest entry).

**4. Why JEPA doesn't help retrieval: a measured trade-off.** On real cells, JEPA embeddings separate knockdowns within one cell type far worse than matched scGPT embeddings (blood: 0.67 vs 0.91 ID percentile; kidney: 0.59–0.66 vs 0.76), while clustering cell types much better. Whitening recovers part of the gap, but not all. This quantifies the Cell-JEPA paper's own hypothesis that JEPA features suppress perturbation-induced deviations. (A26, A27, A29)

## Repository map

```text
attempts.md                 full experiment log (start here)
baseline_checklist.md       replication checklist
experiment.md               ranked experiment ideas
docs/checkpoints.md         how to reuse our pretrained checkpoints (blood / kidney, JEPA vs scGPT)
docs/track_a.md             Cell-JEPA reimplementation notes + deviation log
docs/use_case_map.md        wet-lab use cases, costs, where prediction fits
docs/clustering_finding.md  do predictions carry functional information?

src/track_a/                Cell-JEPA reimplementation: Census corpora (kidney, blood), pretraining
                            objectives (JEPA variants, init from checkpoint), PBMC Tasks 1/2
src/train.py                perturbation fine-tuning (scgpt / scgpt_paper / celljepa), retrieval
                            selection + contrastive retrieval loss, random-init ablation
src/data.py                 GEARS datasets (Adamson, Replogle K562), splits, caches
src/retrieval.py            retrieval metrics (ID percentile, top-1/3/10, ...) + paired bootstrap CIs
src/retrieval_baselines.py  gene-vector ridge/kNN, linear + contrastive baselines
src/latent_retrieval.py     latent-space retrieval + embedding transforms (whitening, PCs, LDA)

modal_pretrain.py / modal_census.py / modal_pbmc.py / modal_app.py   Modal apps
diag/                       launch scripts, diagnostics, figures (diag/results/figures/)
scripts/                    precision@k, inverse retrieval, clustering analyses
tests/                      CPU test suite
```

## Running

```bash
uv venv && uv pip install modal && .venv/bin/modal setup     # Modal account; volume `jepa-perturb`
uv run --no-project --python 3.12 --managed-python --with numpy --with pytest --with torch \
    --with scipy --with scikit-learn python -m pytest -q tests                 # local tests

modal run modal_census.py --stage full --corpus blood_marrow       # pretraining corpus (CPU)
python diag/launch_blood_pretrain.py                               # continued pretraining (GPU)
modal run modal_app.py --stage prepare --dataset replogle_k562_essential --exp-id replogle_k562_v1
python diag/launch_step2_screen.py                                 # fine-tuning screen
modal run diag/retrieval.py --cache /vol/data/replogle_k562_essential_gears_simulation_seed1 \
    --exps replogle_k562_v1 --tag my_scores --split val            # retrieval metrics + CIs
uv run --no-project --with matplotlib --with numpy python diag/fig_ladder.py --bars
```

See `docs/checkpoints.md` for continued pretraining / fine-tuning recipes, and `.venv/bin/python diag/dashboard.py` for a live view of running jobs.

**Honest limits:** K562 CRISPRi knockdown only; targets must be among the 5,000 measured genes (680 of 1,092 Replogle knockdowns excluded); this is an unofficial reimplementation (no official Cell-JEPA code or checkpoints); deviations are listed in `docs/track_a.md` and `attempts.md`.

## Appendix: original Track B pipeline (Adamson, public scGPT)

```bash
source .venv/bin/activate                      # has the `modal` client; workspace auth required
modal run modal_app.py --stage prepare         # CPU: download, manifest, split, baselines
modal run modal_app.py --stage benchmark       # pilots: hopper/H100, blackwell-stack/H100, blackwell/B300
modal run modal_app.py --stage pilot           # execution + timing gate (H100)
modal run --detach modal_app.py --stage train --epochs 15   # 2 arms x seeds 1,2, max 4 containers
modal run modal_app.py --stage aggregate       # tables, plots, demo.html into the Volume
modal volume get jepa-perturb experiments/adamson_trackB_v1/report ./reports   # optional, small files
modal run modal_reference.py --exp-id adamson_trackB_v1   # adds de_rank20.json to the cache; writes reference/
```

Useful flags: `--seeds 1,2`, `--aux-weight 0.5`, `--max-minutes 150`, `--tag v2` (runs are
immutable; a new tag creates a new run directory), `--stack blackwell --gpu B300` to train on
Blackwell. `JEPA_CPU_ONLY=1` deploys only the CPU functions (for workspaces without GPU billing).

Cell-JEPA fine-tuning recipe (**reimplemented from the paper, unofficial**): `--methods celljepa`
(or `scgpt,jepa,celljepa`), `--teacher-update frozen|ema`, `--ckpt scGPT_human|/abs/ckpt_dir`,
`--input-transform log1p|bins50` (bins50 bins encoder inputs per cell; targets/predictions stay
log1p). Loss = MSE + 1.0·(1 − cos(predictor(projector(student cls)), sg(teacher cls))) + 0.8·scGPT-ECS;
the teacher encodes the paired *perturbed* cell. Deviations are listed in `src/train.py`
(`CELLJEPA_DEVIATIONS`) and recorded in every run's config/metrics. `--max-steps-per-epoch N` is
for smoke tests only.

Local tests: `uv run --no-project --with numpy --with pytest python -m pytest -q tests`
(+ `--with torch` to run `tests/test_celljepa_ft.py`)

## Volume layout (`jepa-perturb`)

```text
raw/                         adamson.zip + extracted GEARS files
checkpoints/scGPT_human/     public whole-human scGPT (args.json, best_model.pt, vocab.json)
data/adamson_gears002_simulation_seed1/   X.npy, split.json, manifest.json, eval_controls.json, de_idx.json, de_rank20.json
experiments/<exp_id>/
  experiment.json            binds the experiment to one split hash
  baselines/<name>/          metrics.json, per_intervention.csv, test_predictions.npz
  runs/<method>_s<seed>_<tag>/  config.json, manifest.json, environment.json, log.jsonl, status.json,
                             best.pt, metrics.json, per_intervention.csv, test_predictions.npz
  benchmark/<config>.json    pilot / hardware benchmark reports
  reference/                 GEARS-style metrics (recomputed from saved predictions) vs published Cell-JEPA
  report/                    results.md, results_*.csv, per_intervention_all.csv, demo.html, *.png
```

## Key design decisions (details in code docstrings)

* **Split**: GEARS `simulation` split (cell-gears 0.0.2 `DataSplitter`, seed 1): held-out *genes*;
  58 train / 7 val / 22 test interventions. Saved with a hash; every result checks it.
* **Control reference**: mean of all control cells. **Evaluation inputs**: 256 control cells per
  held-out condition, drawn once and shared by every method; predictions averaged per condition.
* **Primary metric**: mean per-intervention delta RMSE (equal to absolute RMSE under a common
  control — interpret against no-change / training-mean). Undefined correlations stay NaN with
  coverage counts.
* **Checkpoint selection**: validation mean delta RMSE only; test evaluated once after reload.
* **JEPA adaptation (v0)**: frozen copy of the pretrained-loaded student is the teacher. For each
  *training* intervention, target = mean teacher embedding of up to 128 treated training cells minus
  the control centroid (a population-level representation shift). A predictor maps the student's
  pooled, intervention-conditioned representation to that target; loss = MSE + w·(1 − cos).
  Teacher targets never touch val/test cells. w = 0.5 is a declared default, not tuned on test.
* **Matched arms**: identical seeded initialisation (state hash logged), data order, gene
  sub-sampling, schedule (15 epochs, Adam 1e-4, StepLR 0.9, batch 64, 1536 genes, bf16).
* **Vocabulary fix**: 11 Adamson targets use outdated symbols (AARS, ATP5B, SLMO2, SRPR, …) absent
  from scGPT's vocab; under the tutorial rule they collapse to `<pad>` and become indistinguishable.
  We map them to current HGNC symbols in both arms and fail loudly if any target is still lost.

## Blackwell

scGPT depends on torchtext, which has no releases past torch 2.3, so it cannot use CUDA 12.8+/13
wheels as-is. The `blackwell` image omits torchtext and puts a minimal import shim on
`PYTHONPATH` (scGPT only imports `torchtext.vocab.Vocab` to define `GeneVocab`, which we never
instantiate). `--stage benchmark` runs the identical pilot on hopper/H100, blackwell-stack/H100 and
blackwell/B300 so hardware speed-up and software-stack effects are separated and numerical
fingerprints can be compared.
