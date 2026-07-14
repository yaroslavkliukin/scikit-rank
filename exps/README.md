# `exps/` — Experiments & benchmarks

This directory holds everything needed to reproduce the published `scikit-rank`
results: the benchmark runners, per-dataset configs and hyperparameters, dataset
preparation scripts, and the final metrics.

## Layout

```
exps/
├── bars/               # BARS-CTR benchmark (Criteo, Avazu) — configs, runner, results
├── temporal_article/   # Temporal-split experiments (Avazu-Time, MIND-small) — best models + hyperparameters
└── scripts/            # Shared data-prep, training, and evaluation scripts
```

- **BARS scripts and configs live in [`bars/`](bars/README.md).** That folder contains
  the single config-driven runner (`bars/train.py`), one YAML per (model × dataset),
  the published 5-seed `results/`, and full reproduction instructions.
- **The [`temporal_article/`](temporal_article/README.md) README contains the scripts
  and exact hyperparameters for training the best models from the paper** (Avazu-Time
  CTR and MIND-small ranking), including the final `scikit-rank` and FuxiCTR commands.

## Datasets

| Dataset | Task | Source |
|---|---|---|
| Criteo | CTR (classification) | BARS-CTR (`criteo_x1` pre-split) |
| MovieLens | CTR (classification) | BARS |
| Avazu | CTR (classification) | BARS (`avazu_x1`) + raw [Kaggle Avazu CTR](https://www.kaggle.com/competitions/avazu-ctr-prediction/overview) (temporal split) |
| MIND | Ranking | [MIND (Microsoft News)](https://msnews.github.io/) |

The **BARS** pre-split datasets (`criteo_x1`, `avazu_x1`) are used as-is for the
`bars/` benchmark — see [`bars/README.md`](bars/README.md) for their download and MD5
verification.

The **temporal-split** experiments recreate their own train/eval/test splits from the
raw sources, as described below.

## Recreating the temporal-split datasets

Temporal experiments split chronologically: the **final day** is the test set, the
**preceding day** is the eval set, and all earlier rows are train. Splits are produced
by [`scripts/temporal_split.py`](scripts/temporal_split.py), which parses a date column,
optionally filters by minimum user/item counts, and writes `train` / `eval` / `test`
under the output directory. Each output directory must contain `train`, `eval`, and
`test` files (`.csv` or `.parquet`).

### Avazu

1. Download Avazu [from here](https://www.kaggle.com/competitions/avazu-ctr-prediction/overview)

```bash
uv run python exps/scripts/temporal_split.py \
     data/avazu-ctr-prediction/train.gz \
     data/avazu-ctr-prediction/temporal \
     --user-idx device_id --date-idx hour --label-idx click \
     --test-days 1 --eval-days 1
```

### MIND

1. Download MIND Small [from here](https://msnews.github.io/)
2. Process Train Dataset

```bash
unzip MINDsmall_train.zip -d data/mind-small-ranking \
    && mv data/mind-small-ranking/MINDsmall_train/ data/mind-small-ranking/train
```

3. Process Eval Dataset

```bash
unzip MINDsmall_dev.zip -d data/mind-small-ranking \
    && mv data/mind-small-ranking/MINDsmall_dev/ data/mind-small-ranking/eval
```

4. Create parquet files

```bash
uv run python exps/scripts/mind_to_parquet.py \
     data/mind-small-ranking \
     data/mind-small-ranking/processed \
     --main-name mind-dataset.parquet \
     --attach-entity-embeddings
```

5. Create training files

```bash
uv run python exps/scripts/temporal_split.py \
     data/mind-small-ranking/processed/mind-dataset.parquet \
     data/mind-small-ranking/processed/temporal \
     --user-idx user_id --item-idx candidate_news_id --date-idx timestamp \
     --label-idx click --test-days 1 --eval-days 1 --format parquet
```

The `--attach-entity-embeddings` flag emits the MIND entity vectors alongside the main
parquet; they are passed to `DCNRanker` as the named external embedding stream `entity`
in the temporal-article runs.

## Scripts reference (`scripts/`)

| Script | Purpose |
|---|---|
| `temporal_split.py` | Chronological train/eval/test split (final day = test, previous day = eval). |
| `mind_to_parquet.py` | Convert raw MIND TSV behaviors/news into a flat parquet, optionally attaching entity embeddings. |
| `train_temporal_dcn.py` | Train `scikit-rank` DCNv2 (classifier / ranker) on temporal splits. |
| `train_temporal_tabular.py` | Train GBDT / RandomForest baselines (LightGBM, CatBoost, XGBoost, sklearn RF). |
| `train_temporal_fuxictr.py` | Train the reference FuxiCTR DCNv2 on the same splits. |
| `evaluate_ranking.py` | Offline grouped-ranking metrics (NDCG@k, MRR, Recall@k) from a saved predictions file. |
| `summarize_metrics.py` | Aggregate per-seed `metrics.json` into mean ± std tables. |
| `compare_predictions_auc.py` | DeLong paired significance test for the AUC difference between two prediction files. |

For the exact best-model commands and hyperparameters, see
[`temporal_article/README.md`](temporal_article/README.md).
