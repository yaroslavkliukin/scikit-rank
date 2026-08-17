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

## Hyperparameter selection

Hyperparameters were selected independently for `criteo_x1` and `avazu_x1` using a
stagewise block-coordinate grid search. At each stage, an exhaustive Cartesian grid
was evaluated for a group of related parameters, while the validation-AUC winner
from every preceding stage was held fixed. Library defaults were retained as
neutral candidates whenever applicable. LightGBM and XGBoost used a maximum of 10,000
boosting rounds, whereas CatBoost used 1,000 rounds. All three boosting searches used
early stopping with a patience of 200 rounds. DCN used a maximum of 100 epochs and an
early-stopping patience of two epochs. The selected configurations were subsequently
evaluated across the five fixed seeds listed above.

Preliminary backend comparisons were conducted for CatBoost and XGBoost. The GPU
implementation of CatBoost produced substantially lower validation AUC than its CPU
counterpart, so only the CPU branch was retained for hyperparameter selection.
For XGBoost, no meaningful difference in predictive quality was observed between CPU
and GPU. The GPU implementation was therefore selected for the subsequent
hyperparameter search.
Accordingly, only the selected CPU CatBoost and GPU XGBoost grids are reported below.
LightGBM was tuned and evaluated on CPU only.

### DCN

The initial architecture and training hyperparameters were taken from the
dataset-specific DCNv2 reference configurations published in the BARS benchmark. All
parameters not listed below were held fixed at their BARS values. Candidate
configurations were screened with random seed 2021, using validation AUC as the
selection criterion. The selected configurations were then evaluated across the five
fixed seeds.

#### `criteo_x1`

The numeric-feature encoder was varied. The encoder output dimension was fixed at
10 throughout the reported search. The BARS linear encoder
(`linear:embedding_dim=10`) served as the reference.

| Numeric encoder | Grid |
|---|---|
| PLE | `embedding_dim` = 10, with `n_bins` ∈ {16, 32, 48, 64, 128} × `activation` ∈ {false, true} × `feature_dropout` ∈ {0.0, 0.1} |
| PLR | `embedding_dim` = 10 and `activation` = `silu`, with `n_freq` ∈ {16, 24, 32, 48, 64} × `sigma` ∈ {0.05, 0.1, 0.2, 0.4, 0.8} |

The PLE and PLR grids were evaluated as full Cartesian products. The selected
configuration used PLE with `n_bins` = 32, activation enabled, and
`feature_dropout` = 0.1.

#### `avazu_x1`

The reported search was restricted to multihash configurations in which exactly two
high-cardinality fields, `feat_10` and `feat_9`, were mapped to a shared hash table. All
remaining fields retained their BARS settings.

| Parameter block | Grid |
|---|---|
| Multihash capacity | `cardinality` ∈ {10000, 30000, 100000, 300000, 1000000} × `n_hashes` ∈ {1, 2, 3}, with `embedding_dim` = 10 and `embedding_regularizer` = 0.05 |

`cardinality` and `n_hashes` were evaluated as a full Cartesian grid. The selected
multihash configuration used `cardinality` = 10000, `n_hashes` = 2,
`embedding_dim` = 10, and `embedding_regularizer` = 0.05.

### LightGBM

Parameters listed in the same row were varied jointly. The search was conducted on
CPU only.

#### `criteo_x1`

| Parameter block | Grid |
|---|---|
| Tree capacity | `num_leaves` ∈ {127, 255, 511, 1023, 2047} × `min_child_samples` ∈ {100, 200, 500, 2000, 5000, 10000, 20000} |
| Row and feature sampling | `colsample_bytree` ∈ {0.5, 0.7, 0.9, 1.0} × `subsample` ∈ {0.6, 0.8, 1.0} |
| Leaf regularization | `reg_lambda` ∈ {0, 0.1, 1, 5, 10, 50} × `reg_alpha` ∈ {0, 0.1, 1, 10} |
| Categorical-split regularization | `cat_smooth` ∈ {1, 10, 50, 100} × `cat_l2` ∈ {1, 10, 50} × `max_cat_threshold` ∈ {32, 64, 128} |
| Histogram resolution | `max_bin` ∈ {63, 127, 255, 511, 1023, 2047} |
| Final shrinkage search | `learning_rate` ∈ {0.005, 0.01, 0.02, 0.03, 0.05, 0.1} |

#### `avazu_x1`

| Parameter block | Grid |
|---|---|
| Tree capacity | `num_leaves` ∈ {63, 127, 255, 511, 1023} × `min_child_samples` ∈ {100, 500, 2000, 5000, 10000, 20000} |
| Row and feature sampling | `colsample_bytree` ∈ {0.5, 0.7, 0.9, 1.0} × `subsample` ∈ {0.6, 0.8, 1.0} |
| Leaf regularization | `reg_lambda` ∈ {0, 0.1, 1, 5, 10, 50} × `reg_alpha` ∈ {0, 0.1, 1, 10} |
| Categorical-split regularization | `cat_smooth` ∈ {1, 10, 50, 100} × `cat_l2` ∈ {1, 10, 50} × `max_cat_threshold` ∈ {32, 64, 128} |
| Histogram resolution | `max_bin` ∈ {255, 511, 1023, 2047, 4095} |
| Final shrinkage search | `learning_rate` ∈ {0.005, 0.01, 0.02, 0.03, 0.05, 0.1} |

### CatBoost

Only the selected CPU search is reported. Parameters listed in the same row were
varied jointly.

#### `criteo_x1`

| Parameter block | CPU grid |
|---|---|
| CTR controls and preliminary depth | `max_ctr_complexity` ∈ {1, 2, 4} × `one_hot_max_size` ∈ {2, 4, 10, 25, 100, 255} × `depth` ∈ {6, 8} |
| Tree capacity | `depth` ∈ {5, 6, 7, 8, 9, 10} × `l2_leaf_reg` ∈ {0.5, 1, 3, 5, 10, 20} |
| Learning rate | Coarse grid: {0.03, 0.05, 0.1, 0.2, 0.3, 0.41, 0.6}. Final grid: {0.02, 0.07, 0.15, 0.25, 0.3, 0.5} for a coarse winner below 0.25, or {0.1, 0.2, 0.25, 0.35, 0.5, 0.7} otherwise. The coarse winner was also included. |
| Additional regularization | The carried `l2_leaf_reg` winner together with {4, 7, 15}, crossed with `random_strength` ∈ {0, 1, 10} and `model_size_reg` ∈ {0, 0.5, 2} |
| Sampling | MVS used `subsample` ∈ {0.6, 0.8, 1.0} × `rsm` ∈ {0.6, 0.8, 1.0} × `mvs_reg` ∈ {0, 1, 10}. Bernoulli (`subsample` = 0.8) and Bayesian (`bagging_temperature` = 1.0) controls were also evaluated. |
| Numeric-feature bins | `border_count` ∈ {128, 254, 512, 1024}, evaluated jointly with the final learning-rate grid |

#### `avazu_x1`

| Parameter block | CPU grid |
|---|---|
| CTR controls and preliminary depth | `max_ctr_complexity` ∈ {1, 2, 4} × `one_hot_max_size` ∈ {2, 4, 10, 25, 100, 255} × `depth` ∈ {6, 8} |
| Tree capacity | `depth` ∈ {5, 6, 7, 8, 9, 10} × `l2_leaf_reg` ∈ {0.5, 1, 3, 5, 10, 20} |
| Learning rate | Coarse grid: {0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6}. Final grid: {0.02, 0.07, 0.15, 0.25, 0.3, 0.5} for a coarse winner below 0.25, or {0.1, 0.2, 0.25, 0.35, 0.5, 0.7} otherwise. The coarse winner was also included. |
| Additional regularization | The carried `l2_leaf_reg` winner together with {4, 7, 15}, crossed with `random_strength` ∈ {0, 1, 10} and `model_size_reg` ∈ {0, 0.5, 2} |
| Sampling | MVS used `subsample` ∈ {0.6, 0.8, 1.0} × `rsm` ∈ {0.6, 0.8, 1.0} × `mvs_reg` ∈ {0, 1, 10}. Bernoulli (`subsample` = 0.8) and Bayesian (`bagging_temperature` = 1.0) controls were also evaluated. |
| Numeric-feature bins | Not varied because `avazu_x1` contains no numeric features |

### XGBoost

Only the selected GPU search is reported. Parameters listed in the same row were
varied jointly.

#### `criteo_x1`

| Parameter block | GPU grid |
|---|---|
| Learning rate | Coarse grid: {0.03, 0.05, 0.1, 0.2, 0.3, 0.5}. Final grid: {0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5} |
| Tree capacity | `max_depth` ∈ {4, 6, 8, 10, 12} × `min_child_weight` ∈ {1, 5, 20, 100, 300, 1000} |
| Row and feature sampling | `sampling_method` ∈ {`uniform`, `gradient_based`} × `subsample` ∈ {0.3, 0.5, 0.7, 0.9} × `colsample_bytree` ∈ {0.5, 0.7, 0.9, 1.0}. The neutral (`uniform`, 1.0, 1.0) setting was also included. |
| Leaf and split regularization | `reg_lambda` ∈ {0, 1, 5, 10, 50} × `reg_alpha` ∈ {0, 1, 10} × `gamma` ∈ {0, 1, 5} |
| Categorical-split controls | `max_cat_to_onehot` ∈ {4, 16, 64, 256} × `max_cat_threshold` ∈ {16, 64, 256}. The library-default setting was also included. |
| Numeric-feature bins | `max_bin` ∈ {128, 256, 512, 1024} |

#### `avazu_x1`

| Parameter block | GPU grid |
|---|---|
| Learning rate | Coarse grid: {0.03, 0.05, 0.1, 0.2, 0.3, 0.5}. Final grid: {0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5} |
| Tree capacity | `max_depth` ∈ {4, 6, 8, 10, 12} × `min_child_weight` ∈ {1, 5, 20, 100, 300, 1000} |
| Row and feature sampling | `sampling_method` ∈ {`uniform`, `gradient_based`} × `subsample` ∈ {0.3, 0.5, 0.7, 0.9} × `colsample_bytree` ∈ {0.5, 0.7, 0.9, 1.0}. The neutral (`uniform`, 1.0, 1.0) setting was also included. |
| Leaf and split regularization | `reg_lambda` ∈ {0, 1, 5, 10, 50} × `reg_alpha` ∈ {0, 1, 10} × `gamma` ∈ {0, 1, 5} |
| Categorical-split controls | `max_cat_to_onehot` ∈ {4, 16, 64, 256} × `max_cat_threshold` ∈ {16, 64, 256}. The library-default setting was also included. |
| Numeric-feature bins | Not varied because `avazu_x1` contains no numeric features |

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
