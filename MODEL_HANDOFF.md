# Best model: hand-off for using it

**Model:** blood-scGPT + scGPT fine-tune recipe + contrastive retrieval loss, **2-seed ensemble**.
**Task:** given a measured knockdown response in K562 cells, rank candidate genes by how likely their knockdown caused it.
**Test performance:** Replogle K562 essential, 103 held-out genes never used for any choice.
- ID percentile **0.782** (95% CI 0.73–0.83)
- true gene in the top 10 of 103 candidates **49%** of the time (PCA baseline: 16%)
- true gene ranked #1 19% of the time

Full results: `attempts.md` (sections A22–A25, XC) and `diag/results/XC_replogle_test_retrieval_ci.md`.

## 1. Where the files are

Everything is on Modal: workspace **`ls9563`**, volume **`jepa-perturb`**. Inside Modal functions the volume is mounted at `/vol`.

| What | Path on the volume | Notes |
|---|---|---|
| Fine-tuned model, seed 1 | `experiments/replogle_k562_v1/runs/scgpt_paper_s1_s2bscgpt_r1/best.pt` | `{"model": state_dict, "config": ...}` |
| Fine-tuned model, seed 2 | `experiments/replogle_k562_v1/runs/scgpt_paper_s2_s2bscgpt_r1/best.pt` | same format |
| Run config / metrics | same folders: `config.json`, `metrics.json`, `log.jsonl` | exact training settings |
| Saved predictions | same folders: `test_predictions.npz`, `val_predictions.npz` | already-computed profiles for held-out genes |
| Base checkpoint (architecture + vocab) | `trackA/checkpoints/scgpt_kidney_blood_lr3e5/` (`args.json`, `vocab.json`, `best_model.pt`) | needed to rebuild the network before loading `best.pt`. The name says "kidney" but it is the **blood** continued-pretraining export (A23). |
| Dataset cache (genes, controls, split) | `data/replogle_k562_essential_gears_simulation_seed1/` | provides the 5,000-gene order, control cells and split |
| Experiment binding | `experiments/replogle_k562_v1/experiment.json` | split hash `d4237826c68c` |

Download locally (needs `modal` CLI logged into `ls9563`):
```bash
modal volume get jepa-perturb experiments/replogle_k562_v1/runs/scgpt_paper_s1_s2bscgpt_r1 ./best_model_s1
modal volume get jepa-perturb experiments/replogle_k562_v1/runs/scgpt_paper_s2_s2bscgpt_r1 ./best_model_s2
modal volume get jepa-perturb trackA/checkpoints/scgpt_kidney_blood_lr3e5 ./base_ckpt
```

## 2. Code

GitHub `lukesanborn/neo-hackathon26`, branch **`dangelo/establishing-baseline`**. It contains the fine-tune code with `selection="retrieval"` and `retr_weight`. Relevant files:
- `src/train.py`: `TrainConfig`, `Trainer`, `Trainer.fresh_model_from()`, `Trainer.predict()`
- `src/models.py`: `build_scgpt`, gene-vocab mapping
- `src/data.py`: `load_cache`
- `src/retrieval.py`: ID percentile, scoring
- `modal_app.py`: `train_blackwell`, i.e. how runs are launched
- `environment/image.py`: the exact Modal image (torch, scGPT 0.2.x, pinned versions)

**Exact training settings** (from `config.json`):
- `method="scgpt_paper"`, `ckpt=/vol/trackA/checkpoints/scgpt_kidney_blood_lr3e5`, `input_transform="bins50"`
- `selection="retrieval"`, `retr_weight=1.0`, `retr_tau=0.1`
- `epochs=15`, `batch_size=64`, `lr=1e-4`, `aux_weight=0.5`
- `seed` ∈ {1, 2}

## 3. Loading and predicting (inside a Modal function with the volume at `/vol`)

```python
import json, numpy as np
from src.data import load_cache
from src.train import Trainer, TrainConfig

ds = load_cache(json.load(open("/vol/experiments/replogle_k562_v1/experiment.json"))["cache_dir"])
BASE = "/vol/trackA/checkpoints/scgpt_kidney_blood_lr3e5"
RUNS = "/vol/experiments/replogle_k562_v1/runs"

models = []
for s in (1, 2):
    cfg = TrainConfig(method="scgpt_paper", seed=s, ckpt=BASE, input_transform="bins50",
                      selection="retrieval", retr_weight=1.0)
    tr = Trainer(ds, BASE, cfg)                        # builds network + input transform
    models.append((tr, tr.fresh_model_from(f"{RUNS}/scgpt_paper_s{s}_s2bscgpt_r1/best.pt")))

def predict_profiles(conditions):                       # e.g. ["GENE+ctrl", ...]
    """2-seed ensemble: mean predicted post-knockdown expression per condition."""
    outs = [tr.predict(conditions, model=m) for tr, m in models]
    return {c: np.mean([o[c] for o in outs], axis=0) for c in conditions}
```

## 4. Using it for the product: "which knockdown caused this response?"

```python
from src.retrieval import unit

ctrl = ds.control_reference()                           # mean control expression
# 500-gene panel = highest-variance genes across TRAINING knockdown deltas (as in the benchmark)
train = [c for c in ds.split["train"] if c != "ctrl"]
R = np.stack([ds.observed_mean(c) - ctrl for c in train])
panel = np.argsort(-R.var(0))[:500]

def rank_candidates(observed_profile, candidate_conditions):
    """observed_profile: mean expression of the perturbed cells (same 5,000-gene order as ds.gene_names).
    Returns candidates sorted best-first by cosine(predicted delta, observed delta)."""
    P = predict_profiles(candidate_conditions)
    obs = (observed_profile - ctrl)[panel]
    scores = {c: float(unit((P[c] - ctrl)[panel][None])[0] @ unit(obs[None])[0]) for c in candidate_conditions}
    return sorted(scores.items(), key=lambda kv: -kv[1])
```

Candidates must be genes in the measured 5,000-gene panel, because the model marks the knocked-down gene by its column.

`Trainer.predict()` reads `ds.eval_controls[condition]` (a fixed set of 256 control cells), which exists only for the 412 knockdown conditions in the cache. To score a gene `G` outside that set, register it first:
```python
c = f"{G}+ctrl"
ds.eval_controls[c] = ds.eval_controls[ds.split["test"][0]]   # reuse the shared control cells
# then: tr.predict([c], model=m, gene_override={c: ds.gene_names.index(G)})
```

## 5. Before production

1. **Retrain on all 412 knockdowns** (train + val + test) with the same settings and 2 seeds. The benchmark model saw only 279. The test numbers above remain the honest estimate for unseen genes.
2. **Scope:** K562 cells, essential genes, CRISPRi knockdown. Other cell types or screens need fine-tuning on that screen.
3. **Equivalent alternatives** (within CI on test): public scGPT + bins50 + retrieval loss, ensemble (0.775); dual mask acc on blood, ensemble (0.777). The pick above is the simplest and highest.
