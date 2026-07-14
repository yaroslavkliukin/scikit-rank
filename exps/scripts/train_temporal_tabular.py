"""Train tabular baselines on temporal Avazu and MIND splits.

Supported models:

* LightGBM classifier / ranker
* CatBoost classifier / ranker (optional dependency)
* XGBoost classifier / ranker (optional dependency)
* sklearn RandomForest classifier

For MIND every model is evaluated by the same offline grouped-ranking evaluator.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from evaluate_ranking import evaluate_ranking_frame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

logger = logging.getLogger("temporal.tabular")

AVAZU_DROP_DEFAULT = {"id", "click", "hour"}
MIND_GROUP_COL = "impression_id"
MIND_TARGET_COL = "click"
MIND_ITEM_COL = "candidate_news_id"
MIND_CAT_CANDIDATES = ["user_id", "candidate_news_id", "category", "subcategory"]
MIND_NUM_CANDIDATES = ["history_len", "entity_embedding_count"]
RANKING_METRIC_KS = [5, 10, 100]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)  # noqa: NPY002 - baseline libraries use the legacy global RNG


def maybe_head(frame: pl.LazyFrame, max_rows: int | None) -> pl.LazyFrame:
    return frame.head(max_rows) if max_rows is not None else frame


def scan_split(data_dir: Path, split: str, file_format: str) -> pl.LazyFrame:
    path = data_dir / f"{split}.{file_format}"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    if file_format == "parquet":
        return pl.scan_parquet(path)
    if file_format == "csv":
        return pl.scan_csv(path, schema_overrides={"id": pl.String, "hour": pl.String})
    raise ValueError(f"Unsupported file format: {file_format!r}")


def infer_split_format(data_dir: Path, split: str = "train") -> str:
    for file_format in ("parquet", "csv"):
        if (data_dir / f"{split}.{file_format}").exists():
            return file_format
    raise FileNotFoundError(
        f"Missing split file in {data_dir}: expected {split}.parquet or {split}.csv",
    )


def avazu_with_time_features(
    frame: pl.LazyFrame,
    *,
    include_day_feature: bool,
    keep_raw_hour: bool,
) -> pl.LazyFrame:
    schema = frame.collect_schema()
    if "hour" not in schema:
        return frame
    exprs = [pl.col("hour").cast(pl.String).str.slice(-2).alias("hour_of_day")]
    if include_day_feature:
        exprs.append(pl.col("hour").cast(pl.String).str.slice(0, 6).alias("day"))
    frame = frame.with_columns(exprs)
    if not keep_raw_hour:
        frame = frame.drop("hour")
    return frame


def collect_labels(frame: pl.LazyFrame, target_col: str) -> np.ndarray:
    return frame.select(pl.col(target_col)).collect().to_series().to_numpy()


def collect_groups(frame: pl.LazyFrame, group_col: str | None) -> np.ndarray | None:
    if group_col is None:
        return None
    return frame.select(pl.col(group_col)).collect().to_series().to_numpy()


def existing_columns(frame: pl.DataFrame, candidates: list[str]) -> list[str]:
    return [column for column in candidates if column in frame.columns]


def prepare_avazu_frames(
    data_dir: Path,
    args: argparse.Namespace,
) -> tuple[
    pl.DataFrame | None,
    pl.DataFrame,
    pl.DataFrame,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
]:
    file_format = infer_split_format(data_dir) if args.file_format == "auto" else args.file_format
    train_lf = maybe_head(scan_split(data_dir, "train", file_format), args.max_train_rows)
    eval_lf = maybe_head(scan_split(data_dir, "eval", file_format), args.max_eval_rows)
    test_lf = (
        maybe_head(scan_split(data_dir, "test", file_format), args.max_test_rows)
        if args.evaluate_test
        else None
    )
    train_lf = avazu_with_time_features(
        train_lf,
        include_day_feature=args.include_day_feature,
        keep_raw_hour=args.keep_raw_hour,
    )
    eval_lf = avazu_with_time_features(
        eval_lf,
        include_day_feature=args.include_day_feature,
        keep_raw_hour=args.keep_raw_hour,
    )
    if test_lf is not None:
        test_lf = avazu_with_time_features(
            test_lf,
            include_day_feature=args.include_day_feature,
            keep_raw_hour=args.keep_raw_hour,
        )
    train = train_lf.collect()
    eval_df = eval_lf.collect()
    test = test_lf.collect() if test_lf is not None else None
    drop_cols = {args.target_col, *args.drop_cols}
    cat_features = [column for column in train.columns if column not in drop_cols]
    return (
        train.drop(args.target_col),
        eval_df.drop(args.target_col),
        test.drop(args.target_col) if test is not None else None,
        train[args.target_col].to_numpy(),
        eval_df[args.target_col].to_numpy(),
        test[args.target_col].to_numpy() if test is not None else None,
        [],
        cat_features,
    )


def prepare_mind_frame(frame: pl.LazyFrame) -> pl.LazyFrame:
    schema = frame.collect_schema()
    keep = [
        column
        for column in [
            MIND_GROUP_COL,
            "source_split",
            MIND_TARGET_COL,
            *MIND_CAT_CANDIDATES,
            *MIND_NUM_CANDIDATES,
        ]
        if column in schema
    ]
    frame = frame.select(keep)
    if "source_split" in keep:
        frame = frame.with_columns(
            pl.concat_str(
                [
                    pl.col("source_split").cast(pl.String),
                    pl.col(MIND_GROUP_COL).cast(pl.String),
                ],
                separator=":",
            ).alias(MIND_GROUP_COL),
        ).drop("source_split")
    if "entity_embedding_count" in keep:
        frame = frame.with_columns(pl.col("entity_embedding_count").fill_null(0))
    return frame


def prepare_mind_frames(
    data_dir: Path,
    args: argparse.Namespace,
) -> tuple[
    pl.DataFrame | None,
    pl.DataFrame,
    pl.DataFrame,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
]:
    train_lf = maybe_head(scan_split(data_dir, "train", "parquet"), args.max_train_rows)
    eval_lf = maybe_head(scan_split(data_dir, "eval", "parquet"), args.max_eval_rows)
    test_lf = (
        maybe_head(scan_split(data_dir, "test", "parquet"), args.max_test_rows)
        if args.evaluate_test
        else None
    )
    train = prepare_mind_frame(train_lf).collect()
    eval_df = prepare_mind_frame(eval_lf).collect()
    test = prepare_mind_frame(test_lf).collect() if test_lf is not None else None
    cat_features = existing_columns(train, MIND_CAT_CANDIDATES)
    num_features = existing_columns(train, MIND_NUM_CANDIDATES)
    return (
        train.drop([args.target_col, args.group_col]),
        eval_df.drop([args.target_col, args.group_col]),
        test.drop([args.target_col, args.group_col]) if test is not None else None,
        train[args.target_col].to_numpy(),
        eval_df[args.target_col].to_numpy(),
        test[args.target_col].to_numpy() if test is not None else None,
        train[args.group_col].to_numpy(),
        eval_df[args.group_col].to_numpy(),
        test[args.group_col].to_numpy() if test is not None else None,
        num_features,
        cat_features,
    )


def fit_categorical_vocabs(
    train: pl.DataFrame,
    cat_features: list[str],
    min_count: int,
) -> dict[str, dict[str, int]]:
    vocabs: dict[str, dict[str, int]] = {}
    for column in cat_features:
        values = (
            train.lazy()
            .select(pl.col(column).cast(pl.String))
            .group_by(column)
            .len()
            .filter(pl.col("len") >= min_count)
            .select(column)
            .collect()
            .to_series()
            .drop_nulls()
            .sort()
            .to_list()
        )
        vocabs[column] = {value: idx + 1 for idx, value in enumerate(values)}
    return vocabs


def apply_categorical_vocabs(
    frame: pl.DataFrame,
    vocabs: dict[str, dict[str, int]],
) -> pl.DataFrame:
    exprs = []
    for column, vocab in vocabs.items():
        exprs.append(
            pl.col(column)
            .cast(pl.String)
            .replace_strict(
                list(vocab.keys()),
                list(vocab.values()),
                default=0,
                return_dtype=pl.Int32,
            )
            .alias(column),
        )
    return frame.with_columns(exprs) if exprs else frame


def fit_numeric_fill(train: pl.DataFrame, num_features: list[str]) -> dict[str, float]:
    fills = {}
    for column in num_features:
        value = train.select(pl.col(column).cast(pl.Float64).median()).item()
        fills[column] = float(value or 0.0)
    return fills


def apply_numeric_fill(frame: pl.DataFrame, fills: dict[str, float]) -> pl.DataFrame:
    if not fills:
        return frame
    return frame.with_columns(
        [
            pl.col(column)
            .cast(pl.Float64)
            .fill_nan(None)
            .fill_null(value)
            .cast(pl.Float32)
            .alias(column)
            for column, value in fills.items()
        ],
    )


def encode_features(
    train: pl.DataFrame,
    eval_df: pl.DataFrame,
    test: pl.DataFrame | None,
    *,
    num_features: list[str],
    cat_features: list[str],
    min_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, int]]:
    vocabs = fit_categorical_vocabs(train, cat_features, min_count)
    fills = fit_numeric_fill(train, num_features)
    columns = [*num_features, *cat_features]
    encoded = []
    for frame in (train, eval_df, test):
        if frame is None:
            continue
        out = apply_categorical_vocabs(frame, vocabs)
        out = apply_numeric_fill(out, fills)
        encoded.append(out.select(columns).to_pandas())
    cardinalities = {column: len(vocab) + 1 for column, vocab in vocabs.items()}
    test_encoded = encoded[2] if test is not None else None
    return encoded[0], encoded[1], test_encoded, cardinalities


def apply_xgboost_categories(
    frames: tuple[pd.DataFrame | None, ...],
    cat_features: list[str],
    cardinalities: dict[str, int],
) -> tuple[pd.DataFrame | None, ...]:
    """Mark fixed train-fitted category domains for XGBoost native handling."""
    converted = []
    for frame in frames:
        if frame is None:
            converted.append(None)
            continue
        frame = frame.copy()
        for column in cat_features:
            frame[column] = pd.Categorical(
                frame[column],
                categories=range(cardinalities[column]),
            )
        converted.append(frame)
    return tuple(converted)


def apply_frequency_encoding(
    frames: tuple[pd.DataFrame | None, ...],
    cat_features: list[str],
) -> tuple[pd.DataFrame | None, ...]:
    """Encode categories with train-only occurrence counts for XGBoost."""
    train = frames[0]
    if train is None:
        raise ValueError("The first frame must be the training frame")
    frequencies = {column: train[column].value_counts(dropna=False) for column in cat_features}
    converted = []
    for frame in frames:
        if frame is None:
            converted.append(None)
            continue
        frame = frame.copy()
        for column in cat_features:
            frame[column] = frame[column].map(frequencies[column]).fillna(0).astype("float32")
        converted.append(frame)
    return tuple(converted)


def group_sorted(
    X: pd.DataFrame,
    y: np.ndarray,
    group: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[int]]:
    order = np.argsort(group, kind="stable")
    X_sorted = X.iloc[order].reset_index(drop=True)
    y_sorted = y[order]
    group_sorted_ = group[order]
    starts = np.flatnonzero(np.r_[True, group_sorted_[1:] != group_sorted_[:-1]])
    counts = np.diff(np.r_[starts, len(group_sorted_)]).astype(int).tolist()
    return X_sorted, y_sorted, group_sorted_, counts


def dense_qid(group: np.ndarray) -> np.ndarray:
    _, codes = np.unique(group, return_inverse=True)
    return codes.astype(np.int64)


def lightgbm_classifier(args: argparse.Namespace) -> Any:
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        subsample_freq=args.subsample_freq,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=args.random_state,
        n_jobs=args.num_threads,
        verbosity=-1,
        first_metric_only=True,
    )


def lightgbm_ranker(args: argparse.Namespace) -> Any:
    import lightgbm as lgb

    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        subsample_freq=args.subsample_freq,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=args.random_state,
        n_jobs=args.num_threads,
        verbosity=-1,
        first_metric_only=True,
    )


def catboost_classifier(args: argparse.Namespace) -> Any:
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("catboost is not installed; run `uv pip install catboost`") from exc
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_strength=args.random_strength,
        random_seed=args.random_state,
        od_type="Iter",
        od_wait=args.early_stopping_rounds or 200,
        verbose=args.verbose,
        allow_writing_files=False,
    )


def catboost_ranker(args: argparse.Namespace) -> Any:
    try:
        from catboost import CatBoostRanker
    except ImportError as exc:
        raise ImportError("catboost is not installed; run `uv pip install catboost`") from exc
    return CatBoostRanker(
        loss_function=args.catboost_rank_loss,
        eval_metric="NDCG:top=10",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_strength=args.random_strength,
        random_seed=args.random_state,
        od_type="Iter",
        od_wait=args.early_stopping_rounds or 200,
        verbose=args.verbose,
        allow_writing_files=False,
    )


def xgboost_classifier(args: argparse.Namespace) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is not installed; run `uv pip install xgboost`") from exc
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=args.xgb_enable_categorical,
        max_bin=args.xgb_max_bin,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        learning_rate=args.learning_rate,
        max_depth=args.xgb_max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        random_state=args.random_state,
        n_jobs=args.num_threads,
    )


def xgboost_ranker(args: argparse.Namespace) -> Any:
    try:
        from xgboost import XGBRanker
    except ImportError as exc:
        raise ImportError("xgboost is not installed; run `uv pip install xgboost`") from exc
    return XGBRanker(
        objective=args.xgb_rank_objective,
        eval_metric="ndcg@10",
        tree_method="hist",
        enable_categorical=True,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        learning_rate=args.learning_rate,
        max_depth=args.xgb_max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        random_state=args.random_state,
        n_jobs=args.num_threads,
    )


def random_forest_classifier(args: argparse.Namespace) -> Any:
    return RandomForestClassifier(
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        min_samples_leaf=args.rf_min_samples_leaf,
        max_features=args.rf_max_features,
        class_weight=args.rf_class_weight,
        random_state=args.random_state,
        n_jobs=args.num_threads,
    )


def is_ranker(model_name: str) -> bool:
    return model_name.endswith("_ranker")


def build_model(args: argparse.Namespace) -> Any:
    builders = {
        "lgbm_classifier": lightgbm_classifier,
        "lgbm_ranker": lightgbm_ranker,
        "catboost_classifier": catboost_classifier,
        "catboost_ranker": catboost_ranker,
        "xgboost_classifier": xgboost_classifier,
        "xgboost_ranker": xgboost_ranker,
        "random_forest": random_forest_classifier,
    }
    return builders[args.model](args)


def fit_classifier(
    model: Any,
    args: argparse.Namespace,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    cat_features: list[str],
) -> Any:
    if args.model == "lgbm_classifier":
        import lightgbm as lgb

        callbacks = []
        if args.early_stopping_rounds is not None:
            callbacks.append(lgb.early_stopping(args.early_stopping_rounds, verbose=args.verbose))
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            eval_metric="auc",
            categorical_feature=cat_features,
            callbacks=callbacks,
        )
    elif args.model == "catboost_classifier":
        model.fit(X_train, y_train, eval_set=(X_eval, y_eval), cat_features=cat_features)
    elif args.model == "xgboost_classifier":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            verbose=args.verbose,
        )
    else:
        model.fit(X_train, y_train)
    return model


def fit_ranker(
    model: Any,
    args: argparse.Namespace,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    group_train: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    group_eval: np.ndarray,
    cat_features: list[str],
) -> Any:
    X_train_s, y_train_s, group_train_s, train_counts = group_sorted(X_train, y_train, group_train)
    X_eval_s, y_eval_s, group_eval_s, eval_counts = group_sorted(X_eval, y_eval, group_eval)
    if args.model == "lgbm_ranker":
        import lightgbm as lgb

        callbacks = []
        if args.early_stopping_rounds is not None:
            callbacks.append(lgb.early_stopping(args.early_stopping_rounds, verbose=args.verbose))
        model.fit(
            X_train_s,
            y_train_s,
            group=train_counts,
            eval_set=[(X_eval_s, y_eval_s)],
            eval_group=[eval_counts],
            eval_at=[10],
            categorical_feature=cat_features,
            callbacks=callbacks,
        )
    elif args.model == "catboost_ranker":
        from catboost import Pool

        train_pool = Pool(
            X_train_s,
            y_train_s,
            group_id=group_train_s,
            cat_features=cat_features,
        )
        eval_pool = Pool(X_eval_s, y_eval_s, group_id=group_eval_s, cat_features=cat_features)
        model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
    elif args.model == "xgboost_ranker":
        model.fit(
            X_train_s,
            y_train_s,
            qid=dense_qid(group_train_s),
            eval_set=[(X_eval_s, y_eval_s)],
            eval_qid=[dense_qid(group_eval_s)],
            verbose=args.verbose,
        )
    else:
        raise ValueError(f"{args.model} is not a ranker model")
    return model


def predict_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)
    return np.asarray(model.predict(X), dtype=np.float64).reshape(-1)


def save_artifacts(
    output_dir: Path,
    *,
    model: Any,
    metrics: dict[str, float | int],
    run_config: dict[str, Any],
    predictions: pl.DataFrame | None,
    eval_predictions: pl.DataFrame | None = None,
    save_model: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_model:
        with (output_dir / "model.pkl").open("wb") as file:
            pickle.dump(model, file)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str) + "\n",
    )
    if predictions is not None:
        predictions.write_parquet(output_dir / "predictions.parquet")
    if eval_predictions is not None:
        eval_predictions.write_parquet(output_dir / "eval_predictions.parquet")


def run_avazu(args: argparse.Namespace) -> None:
    X_train_pl, X_eval_pl, X_test_pl, y_train, y_eval, y_test, num_features, cat_features = (
        prepare_avazu_frames(Path(args.data_dir), args)
    )
    X_train, X_eval, X_test, cardinalities = encode_features(
        X_train_pl,
        X_eval_pl,
        X_test_pl,
        num_features=num_features,
        cat_features=cat_features,
        min_count=args.min_categr_count,
    )
    if args.model == "xgboost_classifier":
        if args.xgb_enable_categorical and args.xgb_frequency_encoding:
            raise ValueError(
                "--xgb-frequency-encoding requires --no-xgb-enable-categorical",
            )
        if args.xgb_enable_categorical:
            X_train, X_eval, X_test = apply_xgboost_categories(
                (X_train, X_eval, X_test),
                cat_features,
                cardinalities,
            )
        elif args.xgb_frequency_encoding:
            X_train, X_eval, X_test = apply_frequency_encoding(
                (X_train, X_eval, X_test),
                cat_features,
            )
        assert X_train is not None
        assert X_eval is not None
    model = build_model(args)
    if is_ranker(args.model):
        raise ValueError("Avazu is CTR data; use classifier models, not rankers")
    fit_started = time.perf_counter()
    fit_classifier(model, args, X_train, y_train, X_eval, y_eval, cat_features)
    fit_seconds = time.perf_counter() - fit_started
    predict_started = time.perf_counter()
    eval_score = predict_scores(model, X_eval)
    eval_predict_seconds = time.perf_counter() - predict_started
    metrics = {
        "eval_auc": float(roc_auc_score(y_eval, eval_score)),
        "eval_log_loss": float(log_loss(y_eval, eval_score, labels=[0, 1])),
        "fit_seconds": fit_seconds,
        "eval_predict_seconds": eval_predict_seconds,
    }
    predictions = None
    if X_test is not None and y_test is not None:
        test_score = predict_scores(model, X_test)
        prediction = (test_score >= 0.5).astype(np.int64)
        metrics.update(
            {
                "test_auc": float(roc_auc_score(y_test, test_score)),
                "test_log_loss": float(log_loss(y_test, test_score, labels=[0, 1])),
                "test_accuracy": float(accuracy_score(y_test, prediction)),
            },
        )
        predictions = pl.DataFrame(
            {
                args.target_col: y_test.astype(np.int64),
                "score": test_score.astype(np.float32),
                "prediction": prediction,
            },
        )
    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is None and hasattr(model, "get_best_iteration"):
        best_iteration = model.get_best_iteration()
    if best_iteration is not None and int(best_iteration) >= 0:
        metrics["best_iteration"] = int(best_iteration)
    save_artifacts(
        Path(args.output_dir),
        model=model,
        metrics=metrics,
        run_config={
            **vars(args),
            "num_features": num_features,
            "cat_features": cat_features,
            "cardinalities": cardinalities,
        },
        predictions=predictions,
        save_model=args.save_model,
    )
    logger.info("Avazu metrics:\n%s", json.dumps(metrics, indent=2))


def run_mind(args: argparse.Namespace) -> None:
    (
        X_train_pl,
        X_eval_pl,
        X_test_pl,
        y_train,
        y_eval,
        y_test,
        group_train,
        group_eval,
        group_test,
        num_features,
        cat_features,
    ) = prepare_mind_frames(Path(args.data_dir), args)
    X_train, X_eval, X_test, cardinalities = encode_features(
        X_train_pl,
        X_eval_pl,
        X_test_pl,
        num_features=num_features,
        cat_features=cat_features,
        min_count=args.min_categr_count,
    )
    if args.model == "xgboost_ranker":
        X_train, X_eval, X_test = apply_xgboost_categories(
            (X_train, X_eval, X_test),
            cat_features,
            cardinalities,
        )
        assert X_train is not None
        assert X_eval is not None
    model = build_model(args)
    fit_started = time.perf_counter()
    if is_ranker(args.model):
        fit_ranker(
            model,
            args,
            X_train,
            y_train,
            group_train,
            X_eval,
            y_eval,
            group_eval,
            cat_features,
        )
    else:
        fit_classifier(model, args, X_train, y_train, X_eval, y_eval, cat_features)
    fit_seconds = time.perf_counter() - fit_started

    predict_started = time.perf_counter()
    eval_score = predict_scores(model, X_eval)
    eval_predict_seconds = time.perf_counter() - predict_started
    eval_predictions = pl.DataFrame(
        {
            args.group_col: group_eval,
            args.target_col: y_eval.astype(np.int64),
            "score": eval_score.astype(np.float32),
        },
    )
    if MIND_ITEM_COL in X_eval_pl.columns:
        eval_predictions = eval_predictions.with_columns(X_eval_pl[MIND_ITEM_COL])

    eval_metrics = evaluate_ranking_frame(
        eval_predictions,
        group_col=args.group_col,
        target_col=args.target_col,
        score_col="score",
        ks=RANKING_METRIC_KS,
    )
    metrics = {
        **{f"eval_{key}": value for key, value in eval_metrics.items()},
        "fit_seconds": fit_seconds,
        "eval_predict_seconds": eval_predict_seconds,
    }
    test_predictions = None
    if X_test is not None and y_test is not None and group_test is not None:
        test_score = predict_scores(model, X_test)
        test_predictions = pl.DataFrame(
            {
                args.group_col: group_test,
                args.target_col: y_test.astype(np.int64),
                "score": test_score.astype(np.float32),
            },
        )
        if X_test_pl is not None and MIND_ITEM_COL in X_test_pl.columns:
            test_predictions = test_predictions.with_columns(X_test_pl[MIND_ITEM_COL])
        test_metrics = evaluate_ranking_frame(
            test_predictions,
            group_col=args.group_col,
            target_col=args.target_col,
            score_col="score",
            ks=RANKING_METRIC_KS,
        )
        metrics.update({f"test_{key}": value for key, value in test_metrics.items()})
    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is None and hasattr(model, "get_best_iteration"):
        best_iteration = model.get_best_iteration()
    if best_iteration is not None and int(best_iteration) >= 0:
        metrics["best_iteration"] = int(best_iteration)
    save_artifacts(
        Path(args.output_dir),
        model=model,
        metrics=metrics,
        run_config={
            **vars(args),
            "num_features": num_features,
            "cat_features": cat_features,
            "cardinalities": cardinalities,
        },
        predictions=test_predictions,
        eval_predictions=eval_predictions,
        save_model=args.save_model,
    )
    logger.info("MIND metrics:\n%s", json.dumps(metrics, indent=2, default=str))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train temporal tabular baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group("data")
    data.add_argument("--dataset", choices=["avazu_temporal", "mind_small_temporal"], required=True)
    data.add_argument("--data-dir", required=True)
    data.add_argument("--task", choices=["ctr", "ranking"], required=True)
    data.add_argument(
        "--model",
        choices=[
            "lgbm_classifier",
            "lgbm_ranker",
            "catboost_classifier",
            "catboost_ranker",
            "xgboost_classifier",
            "xgboost_ranker",
            "random_forest",
        ],
        required=True,
    )
    data.add_argument("--target-col", default="click")
    data.add_argument("--group-col", default="impression_id")
    data.add_argument("--drop-cols", nargs="*", default=sorted(AVAZU_DROP_DEFAULT))
    data.add_argument("--include-day-feature", action="store_true")
    data.add_argument("--keep-raw-hour", action="store_true")
    data.add_argument("--file-format", choices=["auto", "csv", "parquet"], default="auto")
    data.add_argument("--max-train-rows", type=int, default=None)
    data.add_argument("--max-eval-rows", type=int, default=None)
    data.add_argument("--max-test-rows", type=int, default=None)
    data.add_argument(
        "--evaluate-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable during hyperparameter search; enable only for final seed reruns.",
    )

    pre = parser.add_argument_group("preprocessing")
    pre.add_argument("--min-categr-count", type=int, default=1)

    hp = parser.add_argument_group("boosting hyperparameters")
    hp.add_argument("--n-estimators", type=int, default=1000)
    hp.add_argument("--learning-rate", type=float, default=0.05)
    hp.add_argument("--num-leaves", type=int, default=31)
    hp.add_argument("--max-depth", type=int, default=-1)
    hp.add_argument("--min-child-samples", type=int, default=20)
    hp.add_argument("--subsample", type=float, default=1.0)
    hp.add_argument("--subsample-freq", type=int, default=0)
    hp.add_argument("--colsample-bytree", type=float, default=1.0)
    hp.add_argument("--reg-alpha", type=float, default=0.0)
    hp.add_argument("--reg-lambda", type=float, default=0.0)
    hp.add_argument("--early-stopping-rounds", type=int, default=200)

    cat = parser.add_argument_group("catboost hyperparameters")
    cat.add_argument("--iterations", type=int, default=5000)
    cat.add_argument("--depth", type=int, default=8)
    cat.add_argument("--l2-leaf-reg", type=float, default=10.0)
    cat.add_argument("--random-strength", type=float, default=1.0)
    cat.add_argument("--catboost-rank-loss", default="YetiRank")

    xgb = parser.add_argument_group("xgboost hyperparameters")
    xgb.add_argument("--xgb-max-depth", type=int, default=8)
    xgb.add_argument("--min-child-weight", type=float, default=1.0)
    xgb.add_argument("--xgb-rank-objective", default="rank:ndcg")
    xgb.add_argument(
        "--xgb-enable-categorical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Native categorical splits. Disable on high-cardinality data (Avazu).",
    )
    xgb.add_argument(
        "--xgb-frequency-encoding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace categories with train occurrence counts before XGBoost.",
    )
    xgb.add_argument("--xgb-max-bin", type=int, default=256)

    rf = parser.add_argument_group("random forest hyperparameters")
    rf.add_argument("--rf-n-estimators", type=int, default=300)
    rf.add_argument("--rf-max-depth", type=int, default=None)
    rf.add_argument("--rf-min-samples-leaf", type=int, default=1)
    rf.add_argument("--rf-max-features", default="sqrt")
    rf.add_argument("--rf-class-weight", default=None)

    run = parser.add_argument_group("run")
    run.add_argument("--num-threads", type=int, default=-1)
    run.add_argument("--random-state", type=int, default=2021)
    run.add_argument("--verbose", action="store_true")
    run.add_argument("--output-dir", required=True)
    run.add_argument(
        "--save-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args(argv)
    seed_everything(args.random_state)
    if args.dataset == "avazu_temporal":
        if args.task != "ctr":
            raise ValueError("avazu_temporal must be run with --task ctr")
        run_avazu(args)
    elif args.dataset == "mind_small_temporal":
        if args.task != "ranking":
            raise ValueError("mind_small_temporal must be run with --task ranking")
        run_mind(args)
    else:  # pragma: no cover - argparse choices prevent this
        raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()
