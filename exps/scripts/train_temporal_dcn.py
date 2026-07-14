"""Train dcn models on temporal Avazu and MIND splits.

This runner is deliberately separate from ``train_dcn.py`` because the existing
script follows the BARS layout (``valid.csv`` and target ``label``). Temporal
experiments use ``eval`` splits, target ``click``, and MIND additionally needs
``group=impression_id`` for pairwise/listwise losses.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from evaluate_ranking import evaluate_ranking_frame
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from scikit_rank import DCNClassifier, DCNRanker

logger = logging.getLogger("temporal.scikit_rank")

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
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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
    frame: pl.LazyFrame | pl.DataFrame,
    *,
    include_day_feature: bool,
    keep_raw_hour: bool,
) -> pl.LazyFrame | pl.DataFrame:
    schema = frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
    if "hour" not in schema:
        return frame
    exprs = [
        pl.col("hour").cast(pl.String).str.slice(-2).alias("hour_of_day"),
    ]
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


def drop_existing(
    frame: pl.LazyFrame | pl.DataFrame,
    columns: set[str] | list[str],
) -> pl.LazyFrame | pl.DataFrame:
    schema = frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
    return frame.drop([column for column in columns if column in schema])


def existing_columns(frame: pl.LazyFrame | pl.DataFrame, candidates: list[str]) -> list[str]:
    schema = frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
    return [column for column in candidates if column in schema]


def prepare_avazu(
    data_dir: Path,
    args: argparse.Namespace,
) -> tuple[
    pl.LazyFrame,
    pl.DataFrame | None,
    pl.DataFrame,
    np.ndarray | None,
    np.ndarray,
    list[str],
    list[str],
]:
    file_format = infer_split_format(data_dir) if args.file_format == "auto" else args.file_format
    train = maybe_head(scan_split(data_dir, "train", file_format), args.max_train_rows)
    eval_lf = maybe_head(scan_split(data_dir, "eval", file_format), args.max_eval_rows)
    test_lf = (
        maybe_head(scan_split(data_dir, "test", file_format), args.max_test_rows)
        if args.evaluate_test
        else None
    )

    train = avazu_with_time_features(
        train,
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

    y_eval = collect_labels(eval_lf, args.target_col)
    y_test = collect_labels(test_lf, args.target_col) if test_lf is not None else None
    drop_cols = {args.target_col, *args.drop_cols}
    multihash_cols = set(args.multihash_features or [])
    schema_names = train.collect_schema().names()
    cat_features = [
        column
        for column in schema_names
        if column not in drop_cols and column not in multihash_cols
    ]

    X_eval = drop_existing(eval_lf, drop_cols).collect()
    X_test = drop_existing(test_lf, drop_cols).collect() if test_lf is not None else None
    return train, X_eval, X_test, y_eval, y_test, [], cat_features


def zero_entity_embedding_expr(width: int = 100) -> pl.Expr:
    return pl.lit([0.0] * width, dtype=pl.List(pl.Float64))


def prepare_mind_frame(
    frame: pl.LazyFrame,
    *,
    use_entity_embedding: bool,
) -> pl.LazyFrame:
    schema = frame.collect_schema()
    keep = [
        column
        for column in [
            MIND_GROUP_COL,
            "source_split",
            MIND_TARGET_COL,
            *MIND_CAT_CANDIDATES,
            *MIND_NUM_CANDIDATES,
            "entity_embedding",
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
    if use_entity_embedding and "entity_embedding" in keep:
        frame = frame.with_columns(
            pl.when(pl.col("entity_embedding").is_null())
            .then(zero_entity_embedding_expr())
            .otherwise(pl.col("entity_embedding"))
            .alias("entity_embedding"),
        )
    elif "entity_embedding" in keep:
        frame = frame.drop("entity_embedding")
    return frame


def prepare_mind(
    data_dir: Path,
    args: argparse.Namespace,
) -> tuple[
    pl.LazyFrame,
    pl.DataFrame | None,
    pl.DataFrame,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    dict[str, str] | None,
]:
    train = maybe_head(scan_split(data_dir, "train", "parquet"), args.max_train_rows)
    eval_lf = maybe_head(scan_split(data_dir, "eval", "parquet"), args.max_eval_rows)
    test_lf = (
        maybe_head(scan_split(data_dir, "test", "parquet"), args.max_test_rows)
        if args.evaluate_test
        else None
    )
    train = prepare_mind_frame(train, use_entity_embedding=args.use_entity_embedding)
    eval_lf = prepare_mind_frame(eval_lf, use_entity_embedding=args.use_entity_embedding)
    if test_lf is not None:
        test_lf = prepare_mind_frame(test_lf, use_entity_embedding=args.use_entity_embedding)

    y_eval = collect_labels(eval_lf, args.target_col)
    y_test = collect_labels(test_lf, args.target_col) if test_lf is not None else None
    g_eval = collect_groups(eval_lf, args.group_col)
    g_test = collect_groups(test_lf, args.group_col) if test_lf is not None else None
    assert g_eval is not None

    multihash_cols = set(args.multihash_features or [])
    cat_features = [
        column
        for column in existing_columns(train, MIND_CAT_CANDIDATES)
        if column not in multihash_cols
    ]
    num_features = existing_columns(train, MIND_NUM_CANDIDATES)
    embedding_features = None
    if args.use_entity_embedding and "entity_embedding" in train.collect_schema():
        embedding_features = {"entity": "entity_embedding"}

    X_eval = eval_lf.drop([args.target_col, args.group_col]).collect()
    X_test = (
        test_lf.drop([args.target_col, args.group_col]).collect() if test_lf is not None else None
    )
    return (
        train,
        X_eval,
        X_test,
        y_eval,
        y_test,
        g_eval,
        g_test,
        num_features,
        cat_features,
        embedding_features,
    )


def auc_from_logits(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(roc_auc_score(np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)))


def ndcg10_from_grouped_logits(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
) -> float:
    """Impression-macro NDCG@10 used for MIND checkpoint selection."""
    k = 10
    frame = pl.DataFrame(
        {
            MIND_GROUP_COL: np.asarray(group).reshape(-1),
            MIND_TARGET_COL: np.asarray(y_true).reshape(-1),
            "score": np.asarray(y_pred).reshape(-1),
        },
    )
    ranked = frame.with_columns(
        pl.col("score").rank(method="ordinal", descending=True).over(MIND_GROUP_COL).alias("_rank"),
    )
    discount = 1.0 / pl.col("_rank").cast(pl.Float64).add(1.0).log(2.0)
    grouped = ranked.group_by(MIND_GROUP_COL).agg(
        pl.when(pl.col("_rank") <= k)
        .then(pl.col(MIND_TARGET_COL).cast(pl.Float64) * discount)
        .otherwise(0.0)
        .sum()
        .alias("_dcg"),
        pl.col(MIND_TARGET_COL).sum().alias("_positives"),
    )
    ideal = sum(
        pl.when(pl.col("_positives") >= rank).then(1.0 / np.log2(rank + 1.0)).otherwise(0.0)
        for rank in range(1, k + 1)
    )
    return float(grouped.select((pl.col("_dcg") / ideal).mean()).item())


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def save_common_artifacts(
    output_dir: Path,
    *,
    estimator: DCNClassifier | DCNRanker,
    metrics: dict[str, float | int],
    run_config: dict[str, Any],
    predictions: pl.DataFrame | None,
    save_model: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # The serialized model is optional because metrics and predictions are
    # sufficient for result verification and are much smaller to archive.
    if save_model:
        estimator.save(output_dir / "model.pkl")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str) + "\n",
    )
    if predictions is not None:
        predictions.write_parquet(output_dir / "predictions.parquet")
    pl.DataFrame(estimator.history_).write_csv(output_dir / "history.csv")


def dcn_kwargs(
    args: argparse.Namespace,
    *,
    num_features: list[str],
    cat_features: list[str],
    embedding_features: dict[str, str] | None = None,
) -> dict[str, Any]:
    accelerator_config: dict[str, Any] | None = None
    if args.cpu:
        accelerator_config = {"cpu": True}
    if args.mixed_precision is not None:
        accelerator_config = {
            **(accelerator_config or {}),
            "mixed_precision": args.mixed_precision,
        }
    eval_metrics = {
        "auc": auc_from_logits,
        "ndcg@10": ndcg10_from_grouped_logits,
        "loss": None,
    }
    return {
        "hidden_units": args.hidden_units,
        "cross_layers": args.cross_layers,
        "cross_rank": args.cross_rank,
        "embedding_dim": args.embedding_dim,
        "dropout": args.dropout,
        "structure": args.structure,
        "num_encoder": args.num_encoder,
        "cat_encoder": args.cat_encoder,
        "gated_cross": args.gated_cross,
        "cross_type": args.cross_type,
        "mask_ratio": args.mask_ratio,
        "activation": args.activation,
        "batch_norm": args.batch_norm,
        "use_moe": args.use_moe,
        "num_experts": args.num_experts,
        "moe_top_k": args.moe_top_k,
        "use_inner_cross_layers": args.use_inner_cross_layers,
        "loss": args.loss,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "early_stopping_rounds": args.early_stopping_rounds,
        "eval_metric": eval_metrics[args.eval_metric],
        "eval_metric_name": args.eval_metric,
        "eval_metric_direction": "min" if args.eval_metric == "loss" else "max",
        "eval_metric_group_aware": args.eval_metric == "ndcg@10",
        "num_features": num_features,
        "cat_features": cat_features,
        "multihash_features": args.multihash_features or None,
        "multihash_encoder": args.multihash_encoder,
        "embedding_features": embedding_features,
        "embedding_encoders": (
            {"entity": args.embedding_encoder}
            if embedding_features is not None and args.embedding_encoder is not None
            else None
        ),
        "normalize_numeric": args.normalize_numeric,
        "numeric_nan_fill": args.numeric_nan_fill,
        "ple_n_bins": args.ple_n_bins,
        "lr_scheduler": args.lr_scheduler,
        "grad_clip_norm": args.grad_clip_norm,
        "embedding_regularizer": args.embedding_regularizer,
        "ema_decay": args.ema_decay,
        "chunk_rows": args.chunk_rows,
        "random_state": args.random_state,
        "accelerator_config": accelerator_config,
        "verbose": args.verbose,
    }


def run_avazu(args: argparse.Namespace) -> None:
    train, X_eval, X_test, y_eval, y_test, num_features, cat_features = prepare_avazu(
        Path(args.data_dir),
        args,
    )
    params = dcn_kwargs(args, num_features=num_features, cat_features=cat_features)
    logger.info("Avazu features: num=%s cat=%s", num_features, cat_features)
    logger.info("dcn params:\n%s", json.dumps(params, indent=2, default=str))

    estimator = DCNClassifier(**params)
    fit_started = time.perf_counter()
    estimator.fit(train, y=args.target_col, eval_set=(X_eval, y_eval))
    fit_seconds = time.perf_counter() - fit_started

    predict_started = time.perf_counter()
    eval_score = estimator.predict_proba(X_eval)[:, 1]
    eval_predict_seconds = time.perf_counter() - predict_started
    metrics = {
        "eval_auc": float(roc_auc_score(y_eval, eval_score)),
        "eval_log_loss": float(log_loss(y_eval, eval_score, labels=[0, 1])),
        "epochs_trained": len(estimator.history_),
        "fit_seconds": fit_seconds,
        "eval_predict_seconds": eval_predict_seconds,
        "n_parameters": sum(parameter.numel() for parameter in estimator.model_.parameters()),
    }
    predictions = None
    if X_test is not None and y_test is not None:
        test_score = estimator.predict_proba(X_test)[:, 1]
        test_pred = (test_score >= 0.5).astype(np.int64)
        metrics.update(
            {
                "test_auc": float(roc_auc_score(y_test, test_score)),
                "test_log_loss": float(log_loss(y_test, test_score, labels=[0, 1])),
                "test_accuracy": float(accuracy_score(y_test, test_pred)),
            },
        )
        predictions = pl.DataFrame(
            {
                "click": y_test.astype(np.int64),
                "score": test_score.astype(np.float32),
                "prediction": test_pred,
            },
        )
    run_config = {
        **vars(args),
        "num_features": num_features,
        "cat_features": cat_features,
        "resolved_loss": type(estimator.loss_).__name__,
    }
    save_common_artifacts(
        Path(args.output_dir),
        estimator=estimator,
        metrics=metrics,
        run_config=run_config,
        predictions=predictions,
        save_model=args.save_model,
    )
    logger.info("Avazu metrics:\n%s", json.dumps(metrics, indent=2))


def run_mind(args: argparse.Namespace) -> None:
    (
        train,
        X_eval,
        X_test,
        y_eval,
        y_test,
        g_eval,
        g_test,
        num_features,
        cat_features,
        embedding_features,
    ) = prepare_mind(Path(args.data_dir), args)
    params = dcn_kwargs(
        args,
        num_features=num_features,
        cat_features=cat_features,
        embedding_features=embedding_features,
    )
    logger.info(
        "MIND features: num=%s cat=%s emb=%s",
        num_features,
        cat_features,
        embedding_features,
    )
    logger.info("dcn params:\n%s", json.dumps(params, indent=2, default=str))

    estimator = DCNRanker(**params)
    fit_started = time.perf_counter()
    estimator.fit(
        train,
        y=args.target_col,
        group=args.group_col,
        eval_set=(X_eval, y_eval, g_eval),
    )
    fit_seconds = time.perf_counter() - fit_started

    predict_started = time.perf_counter()
    eval_score = estimator.predict(X_eval).reshape(-1)
    eval_predict_seconds = time.perf_counter() - predict_started
    eval_predictions = pl.DataFrame(
        {
            args.group_col: g_eval,
            args.target_col: y_eval.astype(np.int64),
            "score": eval_score.astype(np.float32),
        },
    )
    if MIND_ITEM_COL in X_eval.columns:
        eval_predictions = eval_predictions.with_columns(X_eval[MIND_ITEM_COL])

    eval_metrics = evaluate_ranking_frame(
        eval_predictions,
        group_col=args.group_col,
        target_col=args.target_col,
        score_col="score",
        ks=RANKING_METRIC_KS,
    )
    metrics = {
        **{f"eval_{key}": value for key, value in eval_metrics.items()},
        "epochs_trained": len(estimator.history_),
        "fit_seconds": fit_seconds,
        "eval_predict_seconds": eval_predict_seconds,
        "n_parameters": sum(parameter.numel() for parameter in estimator.model_.parameters()),
    }
    test_predictions = None
    if X_test is not None and y_test is not None and g_test is not None:
        test_score = estimator.predict(X_test).reshape(-1)
        test_predictions = pl.DataFrame(
            {
                args.group_col: g_test,
                args.target_col: y_test.astype(np.int64),
                "score": test_score.astype(np.float32),
            },
        )
        if MIND_ITEM_COL in X_test.columns:
            test_predictions = test_predictions.with_columns(X_test[MIND_ITEM_COL])
        test_metrics = evaluate_ranking_frame(
            test_predictions,
            group_col=args.group_col,
            target_col=args.target_col,
            score_col="score",
            ks=RANKING_METRIC_KS,
        )
        metrics.update({f"test_{key}": value for key, value in test_metrics.items()})

    run_config = {
        **vars(args),
        "num_features": num_features,
        "cat_features": cat_features,
        "embedding_features": embedding_features,
        "resolved_loss": type(estimator.loss_).__name__,
    }
    output_dir = Path(args.output_dir)
    save_common_artifacts(
        output_dir,
        estimator=estimator,
        metrics=metrics,
        run_config=run_config,
        predictions=test_predictions,
        save_model=args.save_model,
    )
    eval_predictions.write_parquet(output_dir / "eval_predictions.parquet")
    logger.info("MIND metrics:\n%s", json.dumps(metrics, indent=2, default=str))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train dcn on temporal Avazu/MIND splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group("data")
    data.add_argument("--dataset", choices=["avazu_temporal", "mind_small_temporal"], required=True)
    data.add_argument("--data-dir", required=True)
    data.add_argument("--task", choices=["ctr", "ranking"], required=True)
    data.add_argument("--target-col", default="click")
    data.add_argument("--group-col", default="impression_id")
    data.add_argument("--drop-cols", nargs="*", default=sorted(AVAZU_DROP_DEFAULT))
    data.add_argument("--include-day-feature", action="store_true")
    data.add_argument("--keep-raw-hour", action="store_true")
    data.add_argument("--file-format", choices=["auto", "csv", "parquet"], default="auto")
    data.add_argument("--use-entity-embedding", action="store_true")
    data.add_argument("--max-train-rows", type=int, default=None)
    data.add_argument("--max-eval-rows", type=int, default=None)
    data.add_argument("--max-test-rows", type=int, default=None)
    data.add_argument(
        "--evaluate-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Load and evaluate test.*. Disable during hyperparameter search so the "
            "held-out temporal window remains untouched until final seed reruns."
        ),
    )

    arch = parser.add_argument_group("architecture")
    arch.add_argument("--hidden-units", type=int, nargs="+", default=[256, 128])
    arch.add_argument("--cross-layers", type=int, default=3)
    arch.add_argument("--cross-rank", type=int, default=None)
    arch.add_argument("--embedding-dim", type=int, default=32)
    arch.add_argument("--dropout", type=float, default=0.0)
    arch.add_argument("--structure", choices=["stacked", "parallel"], default="stacked")
    arch.add_argument("--num-encoder", default="identity")
    arch.add_argument("--cat-encoder", default="per_feature")
    arch.add_argument("--gated-cross", action="store_true")
    arch.add_argument("--cross-type", choices=["standard", "mldcn"], default="standard")
    arch.add_argument("--mask-ratio", type=float, default=0.5)
    arch.add_argument("--activation", default="relu")
    arch.add_argument("--batch-norm", action=argparse.BooleanOptionalAction, default=False)
    arch.add_argument("--use-moe", action="store_true")
    arch.add_argument("--num-experts", type=int, default=4)
    arch.add_argument("--moe-top-k", type=int, default=2)
    arch.add_argument("--use-inner-cross-layers", action="store_true")
    arch.add_argument("--multihash-features", nargs="*", default=None)
    arch.add_argument("--multihash-encoder", default="multihash")
    arch.add_argument(
        "--embedding-encoder",
        default=None,
        help=(
            "Dense external embedding encoder spec for MIND entity embeddings, "
            'e.g. "tower:output_dim=32;dropout=0.0;normalize=true". '
            "Ignored unless --use-entity-embedding is set."
        ),
    )

    opt = parser.add_argument_group("optimization")
    opt.add_argument("--loss", default="bce")
    opt.add_argument("--lr", type=float, default=1e-3)
    opt.add_argument("--weight-decay", type=float, default=0.0)
    opt.add_argument("--epochs", type=int, default=10)
    opt.add_argument("--batch-size", type=int, default=1024)
    opt.add_argument("--early-stopping-rounds", type=int, default=None)
    opt.add_argument("--eval-metric", choices=["loss", "auc", "ndcg@10"], default="loss")
    opt.add_argument("--lr-scheduler", default=None)
    opt.add_argument("--grad-clip-norm", type=float, default=None)
    opt.add_argument("--embedding-regularizer", type=float, default=0.0)
    opt.add_argument("--ema-decay", type=float, default=None)

    pre = parser.add_argument_group("preprocessing")
    pre.add_argument("--normalize-numeric", action=argparse.BooleanOptionalAction, default=True)
    pre.add_argument(
        "--numeric-normalization",
        choices=["standard", "quantile", "none"],
        default=None,
        help=(
            "Explicit numeric normalization mode for sweeps. Overrides "
            "--normalize-numeric/--no-normalize-numeric when set."
        ),
    )
    pre.add_argument("--numeric-nan-fill", choices=["median", "zero"], default="median")
    pre.add_argument("--ple-n-bins", type=int, default=42)
    pre.add_argument("--chunk-rows", type=int, default=100_000)

    comp = parser.add_argument_group("compute")
    comp.add_argument("--cpu", action="store_true")
    comp.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default=None)
    comp.add_argument("--verbose", action="store_true")
    comp.add_argument("--random-state", type=int, default=2021)

    run = parser.add_argument_group("run")
    run.add_argument("--output-dir", required=True)
    run.add_argument(
        "--save-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write model.pkl to --output-dir. Use --no-save-model to keep only "
        "metrics and predictions.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args(argv)
    if args.numeric_normalization is not None:
        args.normalize_numeric = {
            "standard": True,
            "quantile": "quantile",
            "none": False,
        }[args.numeric_normalization]
    seed_everything(args.random_state)
    if args.dataset == "avazu_temporal":
        if args.task != "ctr":
            raise ValueError("avazu_temporal must be run with --task ctr")
        if args.eval_metric == "ndcg@10":
            raise ValueError("Avazu has no impression groups; use --eval-metric auc")
        run_avazu(args)
    elif args.dataset == "mind_small_temporal":
        if args.task != "ranking":
            raise ValueError("mind_small_temporal must be run with --task ranking")
        if args.eval_metric == "auc":
            raise ValueError("MIND checkpoint selection must use loss or grouped ndcg@10")
        run_mind(args)
    else:  # pragma: no cover - argparse choices prevent this
        raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()
