# Temporal article experiments

This directory contains platform-neutral commands for the final, single-seed
temporal experiments used in the Scikit-Rank article. The reported values are
recorded in [`final_metrics.json`](final_metrics.json), and every final command
is defined in [`run_final.sh`](run_final.sh).

## Protocol

- Seed: `2021`.
- Avazu-Time: row-level binary CTR; report AUC and LogLoss. `device_id` is used
  by preprocessing for temporal/user filtering, not as an evaluation group.
- MIND-small: macro-average ranking metrics by `impression_id`; report NDCG@5,
  NDCG@10, and full-list MRR.
- Split: final day for test, preceding day for eval, all earlier rows for train.
- The MIND entity vectors are produced by `mind_to_parquet.py` and passed to
  `DCNRanker` as the named external embedding stream `entity`.

Temporal results use one seed by design. CUDA kernels and data-loader scheduling
can cause small run-to-run differences even with a fixed seed; reproduce the
reported values to normal floating-point/training tolerance, not bit-for-bit.

The Avazu configuration was selected using `eval`, before its final test run.
The reported MIND BPR-plus-entity configuration is reproducible but exploratory:
the choice to promote BPR over the stronger BCE validation candidate considered
the test-period results. It must not be presented as a strictly validation-only
confirmatory estimate; a fresh held-out temporal period is required for that claim.

## Data preparation

Avazu:

```bash
uv run python exps/scripts/temporal_split.py \
  data/avazu-ctr-prediction/train.gz \
  data/avazu-temporal \
  --user-idx device_id --date-idx hour --label-idx click \
  --test-days 1 --eval-days 1
```

MIND-small:

```bash
uv run python exps/scripts/mind_to_parquet.py \
  data/mind-small-ranking \
  data/mind-small-ranking/processed \
  --main-name mind-dataset.parquet \
  --attach-entity-embeddings

uv run python exps/scripts/temporal_split.py \
  data/mind-small-ranking/processed/mind-dataset.parquet \
  data/mind-small-temporal \
  --user-idx user_id --item-idx candidate_news_id \
  --date-idx timestamp --label-idx click \
  --test-days 1 --eval-days 1 --format parquet
```

Each output directory must contain `train.parquet`, `eval.parquet`, and
`test.parquet`.

## Environment

Install Scikit-Rank and its dependencies as well as gradient boosting libraries:

```bash
uv sync --all-groups
```

FuxiCTR's package contains its runtime, but not the DCNv2 model-zoo class.
Install the pinned runtime and its compatibility dependencies; the exact official
DCNv2 source snapshot is included under `exps/vendor/fuxictr_v2_3_9/`:

```bash
uv pip install --python .venv/bin/python --no-deps fuxictr==2.3.9
uv pip install --python .venv/bin/python h5py==3.16.0 keras-preprocessing==1.1.2
```

## Final runs

By default, data is read from `data/avazu-temporal` and
`data/mind-small-temporal`, and outputs are written below
`output/temporal-article`. Override the roots with `DATA_ROOT` and
`OUTPUT_ROOT`.

```bash
# Headline Scikit-Rank results
exps/temporal_article/run_final.sh avazu-scikit-rank
exps/temporal_article/run_final.sh mind-scikit-rank

# Pinned FuxiCTR DCNv2 references
exps/temporal_article/run_final.sh avazu-fuxictr
exps/temporal_article/run_final.sh mind-fuxictr

# Final boosting baselines
exps/temporal_article/run_final.sh avazu-lightgbm
exps/temporal_article/run_final.sh avazu-catboost
exps/temporal_article/run_final.sh avazu-xgboost
exps/temporal_article/run_final.sh mind-lightgbm
exps/temporal_article/run_final.sh mind-xgboost
exps/temporal_article/run_final.sh mind-catboost
```

Example with custom locations and a CPU-only FuxiCTR run:

```bash
DATA_ROOT=/datasets OUTPUT_ROOT=/results \
  exps/temporal_article/run_final.sh avazu-scikit-rank

FUXICTR_GPU=-1 exps/temporal_article/run_final.sh avazu-fuxictr
```

Scikit-Rank runs write `metrics.json`, `run_config.json`, `history.csv`, test
predictions, and a model unless `--no-save-model` is added to the underlying
command. FuxiCTR writes its metrics, exact resolved config, eval/test predictions,
and checkpoint. Baselines write metrics, config, and predictions; the commands
skip their serialized models to reduce artifact size.

## Baseline caveat

Native XGBoost categorical splits are pathological on Avazu columns with very
high cardinality (`device_ip`, `device_id`, and related IDs). The final XGBoost
run therefore uses train-only frequency encoding. LightGBM and CatBoost retain
their native categorical handling. This preprocessing difference must be stated
when interpreting the lower XGBoost AUC.

## Expected headline metrics

| Dataset | Model | AUC | NDCG@5 | NDCG@10 | MRR |
|---|---|---:|---:|---:|---:|
| Avazu-Time | Scikit-Rank DCNv2 | 0.746084 | - | - | - |
| Avazu-Time | CatBoost | 0.745693 | - | - | - |
| Avazu-Time | FuxiCTR DCNv2 | 0.745474 | - | - | - |
| MIND-small | Scikit-Rank, BPR + entity embeddings | - | 0.285289 | 0.342282 | 0.300437 |
| MIND-small | FuxiCTR DCNv2 | - | 0.277166 | 0.336214 | 0.296182 |

The Scikit-Rank MIND configuration improves over the FuxiCTR reference by
`0.006068` NDCG@10 (1.80% relative) and `0.004256` MRR (1.44% relative). On
Avazu-Time, Scikit-Rank improves AUC by `0.000611` over FuxiCTR; with one seed,
this should be described as a small gain or comparable performance rather than
as statistically established superiority.
