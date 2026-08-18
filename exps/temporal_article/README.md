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

## Hyperparameter selection

Hyperparameters were explored independently on Avazu-Time and MIND-small with
seed `2021`. Candidate DCN configurations were evaluated in two stages. First,
related parameter families were screened independently around a dataset-specific
reference configuration. Second, compact Cartesian grids combined the strongest
architecture, encoder, objective, and regularization candidates. Avazu candidates
were compared by validation AUC; MIND candidates were compared primarily by
validation NDCG@10. The test split was disabled for screening and baseline tuning.
The documented exploratory exception for the reported MIND result is described in
the protocol section above.

Avazu DCN runs used at most 100 epochs, batch size 4096, and early stopping with a
patience of three epochs. MIND DCN runs used at most 50 epochs, batch size 4096,
and a patience of five epochs. The GBDT searches used seed `2021`, a maximum of
2,500 boosting rounds for LightGBM and XGBoost or 3,000 rounds for CatBoost, and
early stopping with a patience of 150 rounds.

### Scikit-Rank DCN

#### Avazu-Time

The reference configuration used BCE, a parallel DCNv2 with hidden units
`[512, 256]`, four cross layers, per-feature categorical embeddings of dimension
10, ReLU, batch normalization, dropout 0.2, learning rate `1e-3`, no weight
decay, and embedding regularization 0.05. Avazu-Time contains categorical
features only, so numerical encoders were not varied.

| Parameter block | Candidates |
|---|---|
| Deep tower | `hidden_units` in {[256, 128], [400, 400, 400], [512, 256], [512, 512, 256]} x `structure` in {stacked, parallel} x `dropout` in {0.1, 0.2} |
| Cross network | `structure` in {stacked, parallel} x `cross_layers` in {2, 3, 4, 5} x `cross_rank` in {full, 64, 128} |
| Per-feature embeddings | `embedding_dim` in {8, 16, 32} |
| Unified embedding | `embedding_dim` in {16, 32, 64, 128}, with `embedding_regularizer=1e-4` |
| Multi-hash | (`cardinality`, `n_hashes`, `embedding_dim`) in {(100000, 2, 16), (100000, 3, 16), (500000, 2, 16), (500000, 2, 32)} for `site_id`, `app_id`, `device_id`, `device_ip`, and `device_model` |
| Time representation | cyclical hour only, raw hour, day, or raw hour plus day |
| Optimization/regularization | (`learning_rate`, `weight_decay`, `embedding_regularizer`) in {(3e-4, 0, 0.05), (1e-3, 0, 0), (1e-3, 0, 1e-4), (1e-3, 0, 1e-3), (1e-3, 1e-4, 0.05), (1e-3, 1e-3, 0.05), (3e-3, 0, 0.05)} |

The combination stage crossed `dropout` in {0.1, 0.2}, `cross_rank` in
{full, 64}, raw-hour inclusion in {false, true}, and categorical encoding in
{per-feature, unified-128, multi-hash}. Exploratory screens also covered gated
and masked cross layers, GLU-family activations, and mixture-of-experts towers;
none replaced the selected standard DCNv2 configuration.

The selected Scikit-Rank configuration uses hidden units `[512, 256]`, a
parallel structure, four full-rank cross layers, dropout 0.1, per-feature
10-dimensional embeddings, BCE, batch normalization, learning rate `1e-3`,
and embedding regularization 0.05.

#### MIND-small

The reference configuration used a stacked DCNv2 with hidden units `[256, 128]`,
three full-rank cross layers, per-feature embeddings of dimension 32, ReLU,
dropout 0.1, learning rate `3e-3`, and no weight or embedding regularization.

| Parameter block | Candidates |
|---|---|
| Objective | BCE; BPR; margin pairwise; LambdaRank@10/@30; focal LambdaRank@10/@30; LambdaNDCG++@10/@30; listwise; ListMLE; ApproxNDCG (`alpha` 5/20); NeuralNDCG (`temperature` 3/10); BCE+listwise and BCE+LambdaNDCG++ composites (`alpha` 0.3/0.5) |
| External entity stream | no external stream, or normalized trainable projections with output dimension in {16, 32, 64, 128, 256, 512} and dropout in {0.0, 0.1} |
| Numerical encoding | identity with standard normalization; linear-16/32 on raw values; PLE-16 with 32 bins; PLE-32 with 64 bins; PLR-16/32 after quantile normalization |
| Categorical encoding | per-feature or unified embeddings with dimension in {16, 32, 64}; multi-hash capacities in {100000, 200000, 500000} with two hashes and dimension 32 |
| Architecture | `hidden_units` in {[256, 128], [512, 256]} x `structure` in {stacked, parallel} x `cross_layers` in {2, 3, 4} x `cross_rank` in {full, 64} x batch normalization in {false, true} |
| Optimization/regularization | learning rate in {3e-4, 1e-3, 3e-3}; weight decay in {0, 1e-4, 1e-3}; embedding regularization in {0, 1e-4, 1e-3}; plateau scheduler on/off, evaluated as a compact set of tuples rather than a full product |

The combination stage crossed four objectives (BCE, listwise, LambdaNDCG++@10,
and focal LambdaRank@10), external entity projection in {none, 32, 64},
categorical encoding in {per-feature-32, unified-16}, architecture in
{stacked/full-rank, parallel/rank-64}, and batch normalization in {false, true}.
BPR was evaluated separately with entity projection dimensions
{64, 128, 256, 512} across the same architecture and batch-normalization choices.

The reported exploratory Scikit-Rank configuration uses BPR with all-pairs
sampling, a normalized 64-dimensional external entity projection, per-feature
32-dimensional categorical embeddings, hidden units `[256, 128]`, a stacked
structure with three full-rank cross layers, no batch normalization, dropout
0.1, learning rate `3e-3`, and no weight or embedding regularization.

### FuxiCTR DCNv2 references

The FuxiCTR reference configurations were fixed comparators rather than another
hyperparameter search. Both used binary cross-entropy and Adam. Avazu used a
parallel DCNv2 with hidden units `[400, 400, 400]`, five cross layers,
10-dimensional embeddings, dropout 0.2, batch normalization, learning rate
`1e-3`, and embedding regularization 0.05. MIND used a stacked DCNv2 with hidden
units `[256, 128]`, three cross layers, 32-dimensional embeddings, dropout 0.1,
no batch normalization, learning rate `3e-3`, and no embedding or network
regularization. The respective early-stopping patience values were two and five
epochs.

### Gradient-boosting baselines

Parameters within a row were evaluated as a Cartesian grid. All other parameters
were kept fixed as shown in the final commands in [`run_final.sh`](run_final.sh).
Avazu used binary classification objectives. On MIND, LightGBM used LambdaRank,
XGBoost used `rank:ndcg`, and CatBoost used YetiRank.

| Dataset/model | Parameter block | Grid |
|---|---|---|
| Avazu LightGBM | Tree capacity | `num_leaves` in {31, 63, 127} x `min_child_samples` in {50, 200} x `reg_lambda` in {1, 10} |
| Avazu XGBoost | Tree capacity | `max_depth` in {4, 6, 8} x `min_child_weight` in {1, 10} x `reg_lambda` in {1, 10}; the frequency-encoded branch additionally used depths {4, 6, 8, 10}, followed by an extension to {12, 14, 16} with `reg_lambda=10` |
| MIND LightGBM | Tree capacity | `num_leaves` in {15, 31, 63} x `min_child_samples` in {50, 200} x `reg_lambda` in {1, 10} |
| MIND XGBoost | Tree capacity | `max_depth` in {3, 4, 6} x `min_child_weight` in {5, 20} x `reg_lambda` in {5, 20} |
| Both CatBoost | Tree capacity | `depth` in {3, 4, 6} x `learning_rate` in {0.01, 0.03} x `l2_leaf_reg` in {10, 30}; MIND used YetiRank |

The selected Avazu configurations were LightGBM with 127 leaves,
`min_child_samples=200`, and `reg_lambda=1`; frequency-encoded XGBoost with
depth 10, `min_child_weight=10`, and `reg_lambda=10`; and CatBoost with depth 6,
learning rate 0.03, and `l2_leaf_reg=10`. The selected MIND configurations were
LightGBM with 63 leaves, `min_child_samples=200`, and `reg_lambda=1`; XGBoost
with depth 6, `min_child_weight=20`, and `reg_lambda=5`; and CatBoost with depth
3, learning rate 0.03, `l2_leaf_reg=10`, and YetiRank.

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
