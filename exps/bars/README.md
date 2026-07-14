# `exps/bars` — Reproducible BARS-CTR experiments (paper results)

This directory contains the **final results** and **everything needed to reproduce
them** for the comparison of DCNv2 (`scikit_rank.DCNClassifier`) against gradient-boosting
baselines (LightGBM / CatBoost / XGBoost) and against the reference DCNv2 implementation
from FuxiCTR, on the two pre-split datasets of the
**[BARS-CTR](https://openbenchmark.github.io/BARS/)** benchmark — `criteo_x1` and
`avazu_x1`. The FuxiCTR runs are **not** reproduced here; the published 5-seed FuxiCTR
results are available upstream:
[Criteo_x1](https://github.com/reczoo/BARS/tree/main/ranking/ctr/DCNv2/DCNv2_criteo_x1)
and [Avazu_x1](https://github.com/reczoo/BARS/tree/main/ranking/ctr/DCNv2/DCNv2_avazu_x1).

All runs go through a **single config-driven runner**, `train.py`: one YAML
configuration per (model × dataset), an identical evaluation protocol for every model
(ROC-AUC / LogLoss on `valid` and `test`), a split-integrity check by MD5, and on-disk
artifacts. Each experiment is run over **5 fixed seeds**; the aggregable `metrics.json`
files are stored under `results/`.

---

## Directory contents

```
exps/bars/
├── train.py            # single runner: --config <yaml> [--random-state N] [--output-dir ...]
├── _datasets.py        # dataset registry (name, label, reference split MD5s from BARS)
├── _data.py            # split loading, categorical vocab encoding, MD5 verification
├── _adapters.py        # per-model adapters (GBDT + DCN); uniform run() contract
├── _reporting.py       # ClearML tracking (optional) + on-disk artifacts
├── configs/            # 10 YAML configs (model × dataset × ablation)
└── results/            # published metrics: <dataset>/<group>/seed-<seed>/metrics.json
```

## Datasets

| Key | Features | Source |
|---|---|---|
| `criteo_x1` | 13 numeric (`I1..I13`) + 26 categorical (`C1..C26`) | [BARS/reczoo pre-split](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x1) `train/valid/test.csv` |
| `avazu_x1`  | 22 categorical (`feat_1..feat_22`), no numeric features | [BARS/reczoo pre-split](https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x1) `train/valid/test.csv` |

The data is **not** stored in the repository. The splits can be downloaded from the
reczoo Datasets registry
([CTR prediction](https://github.com/reczoo/Datasets?tab=readme-ov-file#ctr-prediction)),
and must be placed locally under
`data/Criteo_x1/` and `data/Avazu_x1/` (three files each: `train.csv` / `valid.csv` /
`test.csv`); the path is set by the `data_dir` key in each config. Split integrity is
verified against the reference MD5s from the registry on every run (this can be disabled
with `check_md5: false` in the config).

## Experiments (groups under `results/` ↔ configs)

The subfolder name in `results/<dataset>/<group>/` corresponds to a config:

| Group (`group`) | Dataset | Config | Model / what is tested |
|---|---|---|---|
| `bars_parity`  | criteo_x1, avazu_x1 | `config_dcn_<ds>.yaml` | **DCNv2, parity with FuxiCTR** — the BARS reference point |
| `num_encoding` | criteo_x1 | `config_dcn_criteo_x1_num_encoding.yaml` | DCNv2 + PLE numeric encoder (ablation; criteo only) |
| `cat_encoding` | avazu_x1  | `config_dcn_avazu_x1_multihash.yaml` | DCNv2 + multihash / Unified Embedding (ablation; avazu only) |
| `lgbm`         | criteo_x1, avazu_x1 | `config_lgbm_<ds>.yaml`     | LightGBM (baseline) |
| `catboost`     | criteo_x1, avazu_x1 | `config_catboost_<ds>.yaml` | CatBoost (baseline) |
| `xgboost`      | criteo_x1, avazu_x1 | `config_xgboost_<ds>.yaml`  | XGBoost (baseline, **requires a CUDA GPU**) |

The two DCN ablations are dataset-specific by construction: `num_encoding` (the PLE numeric
encoder) applies only to criteo (avazu has no numeric features), whereas `cat_encoding` (a
shared hashed embedding for two high-cardinality columns) applies only to avazu.

Seeds (5): `2021`, `27011`, `190034`, `948432`, `992817`.

## Results summary

Mean ± standard deviation over the 5 seeds (`test` split), computed with
[`exps/scripts/summarize_metrics.py`](../scripts/summarize_metrics.py) over
`results/<dataset>/<group>/seed-*/metrics.json` (the standard deviation is the **sample**
standard deviation, ddof=1):

| Dataset | Model / group | test AUC | test LogLoss |
|---|---|---:|---:|
| criteo_x1 | DCNv2 (`num_encoding`, PLE) | **0.8146 ± 0.0003** | **0.4374 ± 0.0003** |
| criteo_x1 | DCNv2 (`bars_parity`)       | 0.8138 ± 0.0002 | 0.4382 ± 0.0003 |
| criteo_x1 | CatBoost                    | 0.8135 ± 0.0001 | 0.4380 ± 0.0001 |
| criteo_x1 | XGBoost                     | 0.8115 ± 0.0000 | 0.4401 ± 0.0000 |
| criteo_x1 | LightGBM                    | 0.8102 ± 0.0000 | 0.4413 ± 0.0000 |
| avazu_x1  | DCNv2 (`bars_parity`)       | 0.7646 ± 0.0015 | 0.3669 ± 0.0007 |
| avazu_x1  | DCNv2 (`cat_encoding`, multihash) | **0.7646 ± 0.0004** | 0.3671 ± 0.0002 |
| avazu_x1  | LightGBM                    | 0.7624 ± 0.0005 | 0.3676 ± 0.0002 |
| avazu_x1  | CatBoost                    | 0.7605 ± 0.0017 | 0.3684 ± 0.0005 |
| avazu_x1  | XGBoost                     | 0.7581 ± 0.0004 | 0.3697 ± 0.0002 |

To recompute the aggregates for a single group (one row per seed plus the trailing
`Avg` / `Std` rows):

```bash
uv run python exps/scripts/summarize_metrics.py exps/bars/results/criteo_x1/bars_parity
```

The script takes a directory of per-run subfolders (`seed-*/metrics.json`), reads
`test_auc` / `test_log_loss` from each, and prints a table with the final `Avg` / `Std`.

## Reproduction

### 1. Environment

The project uses `uv` (Python ≥3.12). DCN requires the full environment
(torch/accelerate/scikit_rank); the GBDT baselines can be installed as lean groups:

```bash
uv sync                              # full environment — required for DCN
uv sync --group lgbm            # lean install: LightGBM only
uv sync --group catboost        # ... CatBoost only
uv sync --group xgboost         # ... XGBoost only (requires a CUDA GPU)
```

### 2. Data

Place the BARS splits under `data/Criteo_x1/` and `data/Avazu_x1/` (three `*.csv` files
each). At run time their MD5s are verified against the reference in `_datasets.py`; a
mismatch aborts the run.

### 3. Single run

DCN — in the full environment, with a plain `uv run`:

```bash
uv run python exps/bars/train.py \
    --config exps/bars/configs/config_dcn_criteo_x1.yaml \
    --random-state 2021 \
    --output-dir output/criteo_bars_parity/seed-2021
```

A GBDT baseline — with the corresponding group:

```bash
uv run python exps/bars/train.py \
    --config exps/bars/configs/config_lgbm_criteo_x1.yaml \
    --random-state 2021 \
    --output-dir output/criteo_lgbm/seed-2021
```

### 4. Artifacts and metrics

Each run writes the following to `--output-dir`:

- **`metrics.json`** — `val_auc`, `val_log_loss`, `test_auc`, `test_log_loss` (plus
  `best_iteration` for the GBDT models); these are exactly the files published under
  `results/`.
- `history.csv` — per-round training curves (`epoch`, `train_loss`, `val_loss`, `val_auc`).
- `predictions.parquet` — the `P(click=1)` predictions on `test` (`y_true`, `proba`).

The trained model is not saved. ClearML logging is optional: add
`--clearml [--clearml-project ...] [--clearml-task ...] [--clearml-tags ...]`.

## Reproducibility notes

- **DCN is not bit-for-bit across runs.** Even with a fixed seed, CUDA-kernel scheduling and
  data-loader ordering introduce small differences; reproduce the reported values to normal
  floating-point/training tolerance, not bit-for-bit. The `deterministic: true` flag in the
  DCN configs enables deterministic CUDA kernels (slower) but does not guarantee an exact
  match across different hardware.
- **The XGBoost configs require a CUDA GPU** (`device: cuda`, `tree_method: hist`,
  `enable_categorical: true`). LightGBM and CatBoost run on CPU.
- **Hardware used for the reported runs.** DCN was trained on a single H100 GPU; XGBoost on
  2× H100 GPUs; LightGBM and CatBoost on a 16-CPU machine.
- **A single evaluation protocol** is used for every model: model selection by AUC on
  `valid` (early stopping), then metrics on `valid` and `test`. The categorical encoding is
  identical for the GBDT models and DCN (all train uniques, OOV → 0), which makes the
  comparison directly head-to-head.
- **`bars_parity` is the reference point**: the DCNv2 implementation in `scikit_rank` is brought
  to parity with FuxiCTR DCNv2 (the model behind the BARS-CTR leaderboard).
