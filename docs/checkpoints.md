# Using our pretrained checkpoints

How to build on the checkpoints we have trained, for your own JEPA or fine-tuning experiments. Run history and results are in `attempts.md`.

**Branch:** starting a run from a checkpoint (`init_ckpt`) and pointing pretraining at another dataset (`--cache` / `cache=`) exist only on **`dangelo/establishing-baseline`** (not on `main` yet). The deployed Modal apps (`jepa-pretrain`, `jepa-perturb`) already run this code.

**Where everything lives:** Modal workspace `ls9563`, volume `jepa-perturb`, mounted at `/vol` inside every app.

## Available checkpoints

All of them use the checkpoint contract `args.json` / `best_model.pt` / `last.pt` / `vocab.json`. Use `best_model.pt`; fine-tuning and `init_ckpt` pick it up automatically. **Export names always contain `_kidney`** by convention; the tag (e.g. `_blood_lr3e5`) tells you the dataset.

| Path under `/vol/trackA/checkpoints/` | What it is | Trained on | Objective | Val recon | Log |
|---|---|---|---|---|---|
| **`scgpt_kidney_blood_lr3e5`** | public scGPT + 2 epochs on blood | 2M blood + bone-marrow cells | plain scGPT | 9.13 | A23 |
| **`jepa_kidney_blood_lr3e5`** | public scGPT + 2 epochs on blood, with JEPA (matched to the row above) | same 2M cells, same order and steps | scGPT + JEPA `center` | 10.25 | A23 |
| `scgpt_kidney_lr3e5` | from scratch on kidney (paper replication baseline) | 814k kidney cells, 4 epochs | plain scGPT | 15.68 | A9 |
| `jepa_kidney_lr3e5` | from scratch on kidney (paper replication, Cell-JEPA) | same | JEPA `faithful` (paper) | 17.73 | A9 |
| `jepa_kidney_lr3e5_token` / `_center` | kidney JEPA variants | same | JEPA `token` / `center` | 18.40 / 18.73 | A9 |

Public scGPT (`scGPT_human`, ~33M human cells, all tissues) is at `/vol/checkpoints/scGPT_human` and can be referred to by the name `"scGPT_human"`.

**Data caches** (shard format read by `src.track_a.pretrain.KidneyCache`):
- `/vol/trackA/blood_marrow_census_2023-05-15_2M`: 2M cells, 1.98M train / 20k val (A21)
- `/vol/trackA/kidney_census_2023-05-15`: 814k cells

## Input conventions (all of our checkpoints)

- Values are **per-cell quantile bins 1–50** of nonzero raw counts. For fine-tuning, use `input_transform="bins50"`.
- `<cls>` token value **0**, pad value **−2**, masked value **−1**.
- Vocabulary: scGPT_human's `vocab.json`.
- Architecture: 12 layers, d_model 512, 8 heads, d_hid 512. `init_ckpt` checks that these match and fails loudly if not.

## Recipe 1: continue pretraining with JEPA on top of a checkpoint

```python
import modal

pretrain = modal.Function.from_name("jepa-pretrain", "pretrain").with_options(gpu="B300")
overrides = {
    "init_ckpt": "/vol/trackA/checkpoints/scgpt_kidney_blood_lr3e5",  # or "scGPT_human"
    "jepa_variant": "center",   # see the warning below
    "epochs": 1, "batch_size": 32, "lr": 3e-5, "seed": 0, "max_minutes": 600.0,
}
call = pretrain.spawn("jepa", False, "<your_unique_tag>", overrides,
                      "/vol/trackA/blood_marrow_census_2023-05-15_2M")
print(call.object_id)   # keep this; check with modal.FunctionCall.from_id(id).get(timeout=0)
```

- Arguments are `(objective, smoke, tag, overrides, cache)`. `objective` is `"jepa"` or `"scgpt"`.
- **Use a new, unique tag.** Runs go to `/vol/trackA/pretrain/runs/<objective>_<tag>` and exports to `/vol/trackA/checkpoints/<objective>_kidney_<tag>`. Reusing a tag *resumes* that run, and refuses if the config differs. A non-kidney cache refuses the default tag `v1`.
- Spawn on the **deployed** app as above. With `modal run --detach`, only the last-started function survives if your terminal closes.
- Speed on B300 at batch 32: ~0.044 s/step (scGPT), ~0.051 s/step (JEPA). One epoch of the 2M blood cache is 61,875 steps, about 45–55 min.

**Warnings from our runs:**
- **The paper's JEPA (`faithful`) slowly collapses when starting from public scGPT** (A19). Public scGPT was trained without a `<cls>` objective, so its per-cell `<cls>` targets are almost identical. **`center` works.** Watch `teacher_cls_std` and `target_spread` in the log: ~0.03–0.04 is healthy; heading towards 0 means collapse.
- **lr 1e-4 from random init stalls** (plain scGPT) or collapses (JEPA) for most seeds; **3e-5 trains reliably** (A6–A10). Continued pretraining at 3e-5 was fine.
- **Fair comparison:** JEPA trained *on top of* blood-scGPT gets more total training than blood-scGPT itself. Compare against a control trained for the same extra steps *without* JEPA (same tag scheme, objective `"scgpt"`), or use the matched pair in the table above.

## Recipe 2: fine-tune a checkpoint for perturbation prediction

```python
train = modal.Function.from_name("jepa-perturb", "train_blackwell").with_options(gpu="B300")
call = train.spawn(
    exp_id="replogle_k562_v1",                 # or an Adamson experiment, e.g. "adamson_trackA_lr3e5"
    method="scgpt_paper",                      # "scgpt_paper" (MSE + ECS) or "celljepa" (+ JEPA term)
    seed=1, epochs=15, aux_weight=0.5, max_minutes=150.0,
    tag="<your_unique_tag>",
    ckpt="/vol/trackA/checkpoints/jepa_kidney_blood_lr3e5",
    input_transform="bins50",                  # "log1p" only for public scGPT_human
    cfg_overrides={"selection": "retrieval",   # pick the epoch by validation retrieval (default: delta RMSE)
                   "retr_weight": 1.0},        # optional contrastive retrieval loss (default 0 = off)
)
```

- Runs go to `/vol/experiments/<exp_id>/runs/<method>_s<seed>_<tag>/`, with `metrics.json` and `val_predictions.npz` / `test_predictions.npz`.
- The experiment must exist already (`modal run modal_app.py --stage prepare --dataset ... --exp-id ...`).
- Replogle fine-tunes take ~30–40 min on B300.
- Score retrieval (ID percentile, top-1/3/10, with CIs) with:
  `modal run diag/retrieval.py --cache <cache> --exps <exp_id> --tag <name> [--split val]`
  Choose configs on `--split val`, and score `--split test` once at the end.

## Watching runs

`.venv/bin/python diag/dashboard.py` shows the epoch progress of every run updated in the last few hours.
