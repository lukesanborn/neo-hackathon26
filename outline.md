# Gene perturbation prediction with JEPA

Implementation guideline, project outline, and scope clarification

Prepared November 6, 2026. Status: planning complete; implementation and GPU runs have not begun.

## Implementation starting point

**The first implementation uses Track B: load public pretrained scGPT weights and fine-tune on Adamson. There is no foundation-model or kidney-pretraining job in the initial workflow.**

The two experiment arms start from the same saved weights:

- Public pretrained scGPT → standard perturbation fine-tuning → evaluation.
- The same public pretrained scGPT → perturbation fine-tuning with our JEPA-style auxiliary objective → evaluation.

Fine-tuning updates the designated model parameters; using pretrained weights does not mean freezing the whole student model. The frozen teacher in the adaptation is a separate copy. Record any newly initialized prediction heads and keep shared initialization matched across arms.

Public scGPT weights do **not** contain Cell-JEPA's JEPA pretraining. Our default experiment therefore does not reproduce the paper's full training recipe. Exact reproduction remains a separate, conditional track.

All dataset and model downloads, preprocessing, and training run inside Modal. Large artifacts remain in Modal Volumes. The local computer holds code, documentation, and optional small reports—not a 100 GB data download.

## 1. What we are building

We want to predict how suppressing a gene changes a cell population's RNA-expression profile. The longer-term use case is helping researchers prioritize experiments: compare candidate gene suppressions against a defined biological objective, then test the most promising candidates in the lab.

Our first research question is narrower:

> Does adding a JEPA-style representation-prediction objective improve prediction of expression changes from held-out gene suppressions, compared with a matched scGPT baseline and simple predictors?

The first deliverable is a reproducible Adamson benchmark with saved predictions, a comparison table, and a small visual demo. A negative result is a valid deliverable. The demo must reflect the measured outcome.

## 2. Decisions and boundaries

| Item                         | Default decision                                                                                     |
| ---------------------------- | ---------------------------------------------------------------------------------------------------- |
| First dataset                | GEARS-processed Adamson; one fixed data version                                                      |
| Biological context           | K562 cells; gene suppression; RNA-expression readout                                                 |
| First model                  | Public pretrained scGPT checkpoint and official perturbation fine-tuning workflow                    |
| Foundation-model pretraining | Excluded from the initial implementation; reuse public weights                                       |
| JEPA experiment              | Add the objective during perturbation fine-tuning, not kidney pretraining                            |
| First compute platform       | Modal                                                                                                |
| Initial hardware             | One H100 pilot; A100-80GB fallback; benchmark B200 only if compatible                                |
| Initial parallel training    | Two scGPT runs and two JEPA-adaptation runs, once both implementations pass checks                   |
| First split                  | One fixed intervention-level split; save the actual assignments                                      |
| Repeats                      | Training seeds 1 and 2 on that same split; these are not independent split replications              |
| Primary extension metric     | Mean per-intervention change-from-control RMSE                                                       |
| Supporting metrics           | Change-from-control correlation, responsive-gene recovery, final-expression accuracy                 |
| Time constraint              | Aim for results within 2–3 hours after the runnable environment, data, and model pass the pilot      |
| Initial resource limit       | Maximum four concurrent training containers; proposed $100 first-batch budget within the $500 credit |
| Exact Cell-JEPA reproduction | Conditional on verified artifacts and sufficient configuration detail                                |

The resource limit is a planning default, not a reservation or a guarantee about platform billing controls. Confirm the workspace allowance and current rates before launching.

We are not yet predicting protein abundance, translation efficiency, drug binding, drug response, patient outcomes, or Huntington's progression. Suppressing HTT expression is not equivalent to modeling its disease-causing repeat expansion. A genetic intervention is not automatically a model of a drug targeting the corresponding protein.

## 3. What the prior work establishes

Cell-JEPA builds on scGPT. Its reported setup pretrains on approximately 800,000 kidney cells before downstream evaluation. It reports roughly one day of pretraining on an A5000 and two hours of perturbation fine-tuning on an A100. Those are reported compute measurements, not our end-to-end estimates. Its perturbation experiments use Adamson and Norman, and improved absolute-state prediction does not translate into better change-from-control correlation. [Cell-JEPA methods and Appendix E](https://arxiv.org/html/2602.02093v1)

Public scGPT code, checkpoint links, and a perturbation tutorial are available. GEARS provides model code, processed-data loaders, and split/evaluation utilities. [scGPT repository](https://github.com/bowang-lab/scGPT), [perturbation tutorial](https://scgpt.readthedocs.io/en/latest/tutorial_perturbation.html), [GEARS repository](https://github.com/snap-stanford/GEARS)

The public foundation checkpoint is not automatically an Adamson-fine-tuned perturbation predictor. If suitable fine-tuned weights become available, evaluate them directly for an initial reference result; retain matched fine-tuning runs to isolate the effect of our new objective.

Simple baselines are substantive competitors: a published benchmark found that its evaluated deep-learning models did not outperform simple linear baselines. Beating scGPT alone is insufficient evidence of progress. [Benchmark paper](https://doi.org/10.1038/s41592-025-02772-6), [reproduction code](https://github.com/const-ae/linear_perturbation_prediction-Paper)

An official Cell-JEPA implementation/checkpoint release was not verified in our search. The lab listing links to the preprint. A request for code, both pretrained checkpoints, and reproduction configurations has been sent to the corresponding author. Do not describe this as confirmation that no release exists. [Lab research listing](https://magics.cs.northwestern.edu/research.html)

## 4. Two tracks with different claims

### Track A: paper reproduction

Required artifacts: official code revision, environment, both matched pretrained checkpoints, gene vocabulary and preprocessing, intervention split assignments, training configuration, checkpoint-selection rule, and exact evaluation definitions.

First target only the perturbation results in Tables 3–4. Evaluate supplied fine-tuned checkpoints first if available; then rerun training. Do not silently substitute whole-human scGPT weights for the paper's starting checkpoint. Every substitution belongs in a deviation log.

Success means independently rerunning the specified experiment, reporting observed and published scores, and explaining discrepancies. Exact numerical agreement is not assumed without information about training variance and nondeterminism.

### Track B: hackathon experiment — recommended while artifacts are unresolved

Start both arms from the same public scGPT checkpoint. Compare matched fine-tuning with and without a new JEPA-style auxiliary objective. Label this **our JEPA adaptation**, not a reproduction of Cell-JEPA.

Reuse the downloaded public pretrained checkpoint across experiments; do not rerun pretraining for each seed or variant. The paper's day-long kidney-pretraining stage is excluded from this track. If a later idea changes pretraining itself, that becomes a separately scoped and budgeted experiment rather than an automatic extension of this workflow.

Track B can begin without Track A being unblocked. Do not combine their scores in one purportedly matched comparison.

## 5. Data contract and leakage controls

Use processed expression matrices, not raw sequencing reads. GEARS supplies processed Adamson and Norman downloads; its Replogle downloads are filtered subsets. [Loader source](https://github.com/snap-stanford/GEARS/blob/master/gears/pertdata.py)

Adamson has approximately 68,600 observations and 86 single-gene interventions in the documented processed version. Norman has approximately 91,200 observations and includes both single- and two-gene activation. Both use roughly 5,000 selected genes. Verify exact dimensions after loading rather than hard-coding these approximate counts. [CZI dataset documentation](https://virtualcellmodels.cziscience.com/model/scgenept)

For the first run, use Adamson only. Norman tests activation, so it does not directly expand the gene-suppression claim. Replogle is a later generalization experiment.

Record a data manifest containing:

- Source URL, download date, checksum, and GEARS revision.
- Cell and gene counts; ordered gene identifiers and vocabulary coverage.
- Expression layer, normalization, transformation, and feature-selection procedure.
- Intervention identities, control labels, cell context, and available batch/replicate metadata.
- Exact train/validation/test intervention lists and split-generator configuration.
- Separate split seed and training seed.

Controls and treated cells are generally separate measured cells. Do not claim paired before-and-after trajectories. Follow and document the source control-sampling procedure; use the same control pool and sampling rule for every comparison. The primary outcome is the average response of a population under an intervention.

No held-out **test** treated measurements may enter training, teacher-target construction, hyperparameter selection, or baseline fitting. Validation outcomes may be used for declared checkpoint and hyperparameter selection, but not for gradient updates or training-target construction. The identity of a held-out target gene can be known; its measured test intervention outcome cannot. Do not replace a perturbation-level split with a random cell split.

For paper reproduction, preserve source preprocessing and disclose any whole-dataset feature selection. For the new experiment, fit data-dependent transformations on training data when feasible; document inherited benchmark preprocessing rather than claiming an entirely untouched pipeline.

Define the control reference in the configuration and use it consistently. For our new experiment, derive it from the designated control pool without using held-out treated outcomes. Evaluation-only responsive-gene lists may be derived from test outcomes, but cannot become model inputs or tuning signals.

## 6. Baselines and the first JEPA adaptation

### Baselines

1. **No change:** predict the control reference for every intervention.
2. **Training mean:** average the mean profiles of training interventions, weighting interventions equally rather than weighting by cell count.
3. **scGPT:** initialize from the selected checkpoint and follow the pinned perturbation workflow.
4. **Linear baseline:** port an appropriate unseen-intervention baseline from the public benchmark. Document any gene embeddings or external knowledge it requires. A one-hot regressor with an untrained coefficient for each unseen intervention is not an adequate substitute.
5. **GEARS:** add after the first working pipeline, or concurrently if another implementation owner is available.

No-change and training-mean baselines must ship with the first result. If linear/GEARS comparisons are unfinished, call the result preliminary and limit claims accordingly.

### Proposed adaptation, not the paper's verified implementation

Preserve the scGPT inputs, output prediction head, supervised expression loss, control sampling, and checkpoint initialization. Add a predictor that learns a representation of the post-intervention population from the control profile plus intervention identity.

For an explicitly defined v0 experiment:

- Freeze a copy of the initial encoder as the teacher; disable its dropout.
- Encode treated **training** cells and average their embeddings within each training intervention to form a target centroid. This avoids pretending that arbitrary control/treated cells are matched trajectories.
- Use one documented gene-order, pooling, and normalization convention for all representations. Validate the teacher's input adapter; never merely assume that a perturbation generator exposes a suitable cell embedding.
- Train an auxiliary predictor from the student's intervention-conditioned cell representation to the corresponding target centroid. Use a stopped-gradient teacher target and a cosine-distance auxiliary loss.
- Combine supervised expression loss with the auxiliary loss using an explicit weight. Log both losses separately. Choose any weight using validation only.
- At test time, use only the control input and intervention identity. Teacher targets and treated test measurements are unavailable to prediction.

This population-centroid objective is a design choice for our adaptation. It is not a claim about Cell-JEPA's implementation and does not model within-population heterogeneity. Check that teacher embeddings vary meaningfully across training interventions before launching.

Keep all non-JEPA settings matched. If using effective batch size 64 or aiming for 15 epochs, both arms follow the same declared rule. Avoid adding unrelated loss changes, different checkpoints, or separate hyperparameter searches to only one arm. Establish a matched scGPT control if it differs from the untouched upstream tutorial.

**Important mathematical check:** replacing ordinary MSE with MSE of deltas against the same control is not a new objective:

`MSE(prediction - control, target - control) = MSE(prediction, target)`

A genuinely different objective would need a substantive change such as training-derived gene weights, a ranking term, or another response-sensitive criterion. Those are later experiments, not simultaneous changes to v0.

## 7. Evaluation protocol

For each test intervention, average predicted and observed profiles under the fixed evaluation procedure. Subtract the same control reference from both to obtain predicted and observed changes. Compute metrics per intervention, then average across interventions so conditions with more cells do not dominate.

| Metric                                        | Purpose and interpretation                                                                                                                                                    |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Delta RMSE over the fixed evaluation gene set | Primary new-experiment error metric; lower is better. With a common control it equals absolute RMSE, so it must be interpreted against no-change and training-mean baselines. |
| Delta Pearson correlation                     | Measures agreement in the pattern of changes; it does not establish correct effect magnitude.                                                                                 |
| Responsive-gene RMSE and correlation          | Reveals whether the model predicts effects beyond mostly unchanged genes. Follow a fixed gene-selection rule.                                                                 |
| Direction agreement                           | Score only genes exceeding a predefined meaningful-effect threshold; do not count near-zero noise as a direction.                                                             |
| Absolute-expression correlation               | Useful for reproducing the source benchmark; not evidence of accurate intervention effects by itself.                                                                         |

No-change predicts a zero delta vector, making delta Pearson correlation undefined. Record it as undefined, with coverage counts; do not silently assign zero or drop different interventions for different models. Declare a common valid comparison set and report exclusions.

Select checkpoints using the declared validation metric. For Track A, use the authors' rule; for Track B, use validation mean per-intervention RMSE. Never use test scores for early stopping or tuning. Evaluate the final selected checkpoint once for each predeclared run.

Report individual seed results, per-intervention scores, and the aggregate. Bootstrap uncertainty across interventions, not individual cells, and label it as conditional on the fixed split and fitted models. Two training seeds provide an initial variability check, not a robust estimate of all uncertainty.

Output columns: method, checkpoint, split hash, seed, epochs completed, primary error, delta correlation, responsive-gene metrics, absolute correlation, elapsed time, peak GPU memory, and deviations.

## 8. Modal implementation

Build a batch application with four functions: `prepare`, `pilot`, `train`, and `aggregate`. These are proposed interfaces; no executable launcher exists yet.

| Function    | Resources and responsibilities                                                                      |
| ----------- | --------------------------------------------------------------------------------------------------- |
| `prepare`   | CPU container; download once, validate manifests, generate splits/caches, run simple baselines      |
| `pilot`     | One GPU; check training, validation, checkpoint save/reload, target availability, speed, and memory |
| `train`     | One GPU per experiment; bounded runtime; save to a unique run directory                             |
| `aggregate` | CPU container; verify compatible manifests, combine metrics, export demo data and plots             |

Use one pinned container image after checking compatibility among Python, PyTorch, torchtext, PyTorch Geometric, scGPT, and GEARS. Do not install unbounded latest versions. Record CUDA, attention backend, precision, and any patches. A B200 may need a newer software stack; compare correctness before treating it as a drop-in replacement.

Use persistent Volumes for data/checkpoints/results. Downloads happen in the remote `prepare` function, not the local launcher. A single preparation job writes shared caches; commit them before consumers begin. Refresh mounted data where required, close open files before reload, and keep each worker's writes in a separate directory. Do not have every GPU independently build the same cache. Do not automatically synchronize datasets, caches, or checkpoints back to the local computer. [Modal Volumes](https://modal.com/docs/guide/volumes)

Submit independent jobs with Modal's parallel invocation facilities and detached execution. Each container handles one training job at a time. Bound concurrency and timeouts; distinguish submitted, running, failed, timed-out, and completed jobs in the report. Preserve checkpoints before the deadline rather than relying on a hard timeout to save state. [Batch execution](https://modal.com/docs/guide/batch-processing), [GPU configuration](https://modal.com/docs/guide/gpu)

Initial system-memory allocation: 64–128 GB per GPU worker, subject to measured usage. Allow roughly 100 GB of **cloud** persistent storage initially; this is headroom for data, caches, and outputs, not a verified dataset size or local-disk requirement. These are planning allocations, not verified requirements. CPU, memory, and storage charges must be included in the budget estimate.

Avoid distributed training in the first batch. If pretraining becomes necessary, separately benchmark multi-GPU scaling and preserve effective batch size, update count, and schedule. Independent runs and distributed training solve different timing problems.

## 9. Suggested repository layout

```text
PROJECT_PLAN.md
README.md                    # install, prepare, pilot, train, evaluate
modal_app.py                 # remote resource and job orchestration
environment/                 # pinned image, dependency lock, upstream revisions
configs/                     # baseline, adaptation, exact-reproduction configs
src/
  data.py                    # manifests, splits, control sampling, gene mapping
  baselines.py               # no-change, training-mean, linear adapters
  models.py                  # scGPT adapter, frozen teacher, latent predictor
  train.py                   # training, validation, checkpoint selection
  evaluate.py                # shared metrics and per-intervention predictions
  report.py                  # comparison tables and demo exports
tests/                       # meaningful leakage, loss, and resume checks
```

Remote outputs should be organized by immutable experiment ID, then method and seed. Save `config.json`, `manifest.json`, `metrics.json`, `per_intervention.csv`, predictions, checkpoint, environment details, and logs. Large data and weights belong in persistent storage, not the code repository.

## 10. Timing gates and fallback plan

The 2–3-hour target covers the initial fine-tuning/evaluation batch after the pilot passes. It does not include recreating foundation-model pretraining. Build/download/debug time is not measured, and faster hardware has not yet been benchmarked. The reported paper runtime does not guarantee our runtime. A new fine-tuning variant does not automatically add another day: independent variants can reuse the checkpoint and run concurrently.

| Gate            | Evidence required                                                                            | If it fails                                                      |
| --------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Artifact gate   | Verified official artifacts for Track A, or an explicitly labeled Track B configuration      | Continue public scGPT baseline; do not claim exact reproduction  |
| Data gate       | Manifest, fixed split, controls, vocabulary coverage, and baselines verified                 | Fix inputs before GPU training                                   |
| Execution gate  | Finite loss; one optimizer step; correct target isolation; successful save/reload/evaluation | Debug one GPU, not four identical failures                       |
| Timing gate     | Measured training and full validation speed fit the remaining time                           | Predeclare a shorter matched experiment, or finish baseline only |
| Comparison gate | Same data/split/evaluation and documented model differences                                  | Report runs separately rather than claim an improvement          |

At kickoff, spend at most about 30–45 minutes trying to establish the data and execution gates before reassessing scope. This is a decision deadline, not a promise that setup finishes then.

Once runnable, measure 100–200 steps after warmup, including the teacher overhead. Estimate total time from training steps, validation, loading, checkpoint writes, and final evaluation. Benchmark H100 and B200 in parallel only if the environment supports both.

If projected time fits, run baseline and adaptation seeds concurrently and reserve the final 20–30 minutes for evaluation and reporting. If it does not, shorten the experiment before test evaluation, apply the same schedule to both arms, and label it a pilot. A timed-out partial run is not a completed paper reproduction.

Do not allocate a full pretraining run until its measured throughput and available artifacts make the decision defensible. Public pretrained scGPT allows Track B to proceed without kidney pretraining.

## 11. Work allocation

| Workstream           | First responsibility                                                      | Dependency                        |
| -------------------- | ------------------------------------------------------------------------- | --------------------------------- |
| Data/evaluation      | Frozen manifest, splits, controls, simple baselines, shared scorer        | First priority                    |
| Infrastructure/scGPT | Working Modal image, checkpoint load, timed baseline pilot                | Data/evaluation contract          |
| JEPA adaptation      | Explicit representation adapter, frozen teacher targets, matched ablation | Same checkpoint and data contract |
| Demo/reporting       | Visualize exported predictions; surface limitations and run status        | Final evaluated predictions       |

Assign people to these roles at kickoff. This document does not assume additional agents or contributors are already running.

## 12. Demo and permissible conclusions

The first demo lets a viewer choose a **held-out, experimentally measured intervention**, see predicted expression changes, and reveal observed results. Show our adaptation, scGPT, and simple baselines side by side. Include a typical case and a failure alongside aggregate results. All displayed numbers must come from saved evaluation outputs.

Core visualizations: predicted-versus-observed change scatterplot, top changed genes, per-intervention error comparison, and an aggregate results table. Add pathway interpretation only when supported by a documented analysis. Embedding or UMAP distance is not itself a measure of therapeutic benefit.

The Adamson benchmark alone does not establish a healthy target state or validate treatment efficacy. A later prioritization study must predefine a biological objective and score rankings against withheld experimental outcomes. Retrospective ranking success would support experiment prioritization; actual wet-lab savings would still need demonstration.

| Outcome                                        | Claim supported                                                            |
| ---------------------------------------------- | -------------------------------------------------------------------------- |
| Baseline runs successfully                     | We have a reproducible baseline under the documented setup                 |
| Official setup reruns with comparable scores   | We reproduced the specified paper experiment within documented differences |
| Adaptation beats matched scGPT only            | Preliminary improvement over that baseline on this dataset/split           |
| Adaptation beats strong baselines consistently | Evidence supporting the method within the evaluated setting                |
| Ranking predicts useful held-out interventions | Retrospective evidence for prioritization under the chosen objective       |

## 13. Acceptance checklist

- [ ] Track A or Track B is explicit in every run and report.
- [ ] Initial jobs load public pretrained scGPT weights; no kidney/foundation pretraining is launched as part of Track B.
- [ ] Large downloads and outputs stay in Modal; no automatic local data or checkpoint synchronization is configured.
- [ ] Source revisions, environment, checkpoints, data, and splits are recorded.
- [ ] Intervention labels survive vocabulary mapping and sequence subsampling; missing targets are reported.
- [ ] Training never accesses held-out treated outcomes or their teacher representations.
- [ ] Target encoder is frozen, deterministic under its declared inputs, and receives no gradients.
- [ ] Prediction works without treated observations; changing the intervention can change the prediction.
- [ ] Checkpoint reload reproduces predictions within declared numerical tolerance.
- [ ] Baselines and models share the evaluation contract; undefined metrics are explicit.
- [ ] Selected checkpoints use validation only; test results are not used for tuning.
- [ ] Actual runtime, cost estimate, memory use, and completion status are reported.
- [ ] Final plots reproduce saved predictions; claim wording matches the evidence.

## 14. Unresolved items

Official Cell-JEPA artifacts and precise reproduction settings remain unresolved. Installed dependency compatibility, actual download sizes, memory use, GPU throughput, and workspace concurrency have not been tested. The auxiliary-loss weight and representation adapter require a documented pilot decision. No evidence currently establishes that our adaptation will improve results.

These uncertainties do not block preparing the public baseline. They do block guaranteeing exact reproduction or improvement within a fixed three-hour window.
