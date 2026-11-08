# Gene perturbation prediction with JEPA — Track B

Does adding a JEPA-style representation-prediction objective to scGPT perturbation
fine-tuning improve prediction of expression changes for **held-out** gene knockdowns
(Adamson, K562 CRISPRi), compared with matched scGPT and simple baselines?

This is **Track B: our JEPA adaptation** of public pretrained scGPT. It is *not* a
reproduction of Cell-JEPA (no kidney pretraining; see `outline.md` §4).

**Building on our pretrained checkpoints (blood / kidney, JEPA vs scGPT):** see `docs/checkpoints.md`. Run log: `attempts.md`.

## Layout

```text
outline.md                 project plan / scope
modal_app.py               Modal jobs: prepare, pilot(+_blackwell), train(+_blackwell), aggregate
modal_reference.py         CPU job: GEARS-style metrics from saved predictions vs published Cell-JEPA numbers
environment/image.py       two pinned images: "hopper" (torch 2.1.2/cu121 + torchtext) and
                           "blackwell" (torch 2.12.1/cu130, torchtext replaced by an import shim)
src/data.py                download, manifest, GEARS simulation split, eval-control draw, cache
src/baselines.py           no-change, training-mean, linear (Ahlmann-Eltze gene-embedding model)
src/models.py              scGPT build + pretrained load, HGNC alias fix, frozen teacher, predictor
src/train.py               matched training loop, validation selection, reload check, pilot
src/evaluate.py            shared metrics (per intervention, equal weight), bootstrap
src/report.py              aggregation, tables, PNG plots, demo export
src/demo_template.html     interactive demo (filled with saved predictions by `aggregate`)
tests/                     leakage / metric / baseline tests (numpy only)
```

## Running (everything heavy runs in Modal)

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
