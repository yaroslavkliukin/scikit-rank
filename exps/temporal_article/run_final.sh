#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: DATA_ROOT=/path/to/data OUTPUT_ROOT=/path/to/output $0 <run-name>" >&2
    exit 2
fi

RUN_NAME=$1
DATA_ROOT=${DATA_ROOT:-data}
OUTPUT_ROOT=${OUTPUT_ROOT:-output/temporal-article}
PYTHON=${PYTHON:-.venv/bin/python}
SEED=2021
FUXICTR_GPU=${FUXICTR_GPU:-0}

run_dcn() {
    "$PYTHON" exps/scripts/train_temporal_dcn.py "$@"
}

run_fuxictr() {
    "$PYTHON" exps/scripts/train_temporal_fuxictr.py "$@"
}

run_tabular() {
    "$PYTHON" exps/scripts/train_temporal_tabular.py "$@"
}

case "$RUN_NAME" in
avazu-scikit-rank)
    run_dcn \
        --dataset avazu_temporal --task ctr \
        --data-dir "$DATA_ROOT/avazu-temporal" \
        --output-dir "$OUTPUT_ROOT/avazu-scikit-rank" \
        --evaluate-test --random-state "$SEED" --loss bce \
        --hidden-units 512 256 --cross-layers 4 --structure parallel \
        --embedding-dim 10 --dropout 0.1 --activation relu --batch-norm \
        --cat-encoder per_feature --lr 1e-3 --weight-decay 0 \
        --embedding-regularizer 0.05 --grad-clip-norm 10 \
        --lr-scheduler 'plateau:patience=0;factor=0.1;min_lr=1e-6' \
        --epochs 100 --batch-size 4096 --early-stopping-rounds 3 \
        --eval-metric auc --chunk-rows 50000000
    ;;
mind-scikit-rank)
    run_dcn \
        --dataset mind_small_temporal --task ranking \
        --data-dir "$DATA_ROOT/mind-small-temporal" \
        --output-dir "$OUTPUT_ROOT/mind-scikit-rank" \
        --evaluate-test --random-state "$SEED" --loss bpr:sampling=all_pairs \
        --hidden-units 256 128 --cross-layers 3 --structure stacked \
        --embedding-dim 32 --dropout 0.1 --activation relu --no-batch-norm \
        --cat-encoder per_feature --use-entity-embedding \
        --embedding-encoder 'tower:output_dim=64;dropout=0.0;normalize=true' \
        --lr 3e-3 --weight-decay 0 --embedding-regularizer 0 \
        --grad-clip-norm 10 --epochs 50 --batch-size 4096 \
        --early-stopping-rounds 5 --eval-metric ndcg@10
    ;;
avazu-fuxictr)
    run_fuxictr \
        --dataset avazu --data-dir "$DATA_ROOT/avazu-temporal" \
        --output-dir "$OUTPUT_ROOT/avazu-fuxictr" --evaluate-test \
        --model-structure parallel --hidden-units 400 400 400 \
        --cross-layers 5 --embedding-dim 10 --dropout 0.2 --batch-norm \
        --learning-rate 1e-3 --embedding-regularizer 0.05 --net-regularizer 0 \
        --epochs 100 --batch-size 4096 --early-stopping-rounds 2 \
        --random-state "$SEED" --gpu "$FUXICTR_GPU" --num-workers 4
    ;;
mind-fuxictr)
    run_fuxictr \
        --dataset mind --data-dir "$DATA_ROOT/mind-small-temporal" \
        --output-dir "$OUTPUT_ROOT/mind-fuxictr" --evaluate-test \
        --model-structure stacked --hidden-units 256 128 \
        --cross-layers 3 --embedding-dim 32 --dropout 0.1 --no-batch-norm \
        --learning-rate 3e-3 --embedding-regularizer 0 --net-regularizer 0 \
        --epochs 50 --batch-size 4096 --early-stopping-rounds 5 \
        --random-state "$SEED" --gpu "$FUXICTR_GPU" --num-workers 4
    ;;
avazu-lightgbm)
    run_tabular \
        --dataset avazu_temporal --task ctr --model lgbm_classifier \
        --data-dir "$DATA_ROOT/avazu-temporal" \
        --output-dir "$OUTPUT_ROOT/avazu-lightgbm" --evaluate-test --no-save-model \
        --random-state "$SEED" --num-threads -1 --early-stopping-rounds 150 \
        --n-estimators 2500 --learning-rate 0.03 --num-leaves 127 \
        --min-child-samples 200 --subsample 0.8 --subsample-freq 1 \
        --colsample-bytree 0.8 --reg-lambda 1
    ;;
avazu-catboost)
    run_tabular \
        --dataset avazu_temporal --task ctr --model catboost_classifier \
        --data-dir "$DATA_ROOT/avazu-temporal" \
        --output-dir "$OUTPUT_ROOT/avazu-catboost" --evaluate-test --no-save-model \
        --random-state "$SEED" --num-threads -1 --early-stopping-rounds 150 \
        --iterations 3000 --learning-rate 0.03 --depth 6 \
        --l2-leaf-reg 10 --random-strength 1
    ;;
avazu-xgboost)
    run_tabular \
        --dataset avazu_temporal --task ctr --model xgboost_classifier \
        --data-dir "$DATA_ROOT/avazu-temporal" \
        --output-dir "$OUTPUT_ROOT/avazu-xgboost" --evaluate-test --no-save-model \
        --random-state "$SEED" --num-threads -1 --early-stopping-rounds 150 \
        --n-estimators 2500 --learning-rate 0.03 --xgb-max-depth 10 \
        --min-child-weight 10 --subsample 0.8 --colsample-bytree 0.8 \
        --reg-lambda 10 --no-xgb-enable-categorical --xgb-frequency-encoding
    ;;
mind-lightgbm)
    run_tabular \
        --dataset mind_small_temporal --task ranking --model lgbm_ranker \
        --data-dir "$DATA_ROOT/mind-small-temporal" \
        --output-dir "$OUTPUT_ROOT/mind-lightgbm" --evaluate-test --no-save-model \
        --random-state "$SEED" --num-threads -1 --early-stopping-rounds 150 \
        --n-estimators 2500 --learning-rate 0.03 --num-leaves 63 \
        --min-child-samples 200 --subsample 0.8 --subsample-freq 1 \
        --colsample-bytree 0.8 --reg-lambda 1
    ;;
mind-xgboost)
    run_tabular \
        --dataset mind_small_temporal --task ranking --model xgboost_ranker \
        --data-dir "$DATA_ROOT/mind-small-temporal" \
        --output-dir "$OUTPUT_ROOT/mind-xgboost" --evaluate-test --no-save-model \
        --random-state "$SEED" --num-threads -1 --early-stopping-rounds 150 \
        --n-estimators 2500 --learning-rate 0.03 --xgb-max-depth 6 \
        --min-child-weight 20 --subsample 0.8 --colsample-bytree 0.8 --reg-lambda 5
    ;;
mind-catboost)
    run_tabular \
        --dataset mind_small_temporal --task ranking --model catboost_ranker \
        --data-dir "$DATA_ROOT/mind-small-temporal" \
        --output-dir "$OUTPUT_ROOT/mind-catboost" --evaluate-test --no-save-model \
        --random-state "$SEED" --num-threads -1 --early-stopping-rounds 150 \
        --iterations 3000 --learning-rate 0.03 --depth 3 \
        --l2-leaf-reg 10 --random-strength 1 --catboost-rank-loss YetiRank
    ;;
*)
    echo "Unknown run: $RUN_NAME" >&2
    echo "Available: avazu-scikit-rank mind-scikit-rank avazu-fuxictr mind-fuxictr" >&2
    echo "           avazu-lightgbm avazu-catboost avazu-xgboost" >&2
    echo "           mind-lightgbm mind-xgboost mind-catboost" >&2
    exit 2
    ;;
esac
