"""Train the official FuxiCTR v2.3.9 DCNv2 on temporal article splits."""

from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import logging
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from evaluate_ranking import evaluate_ranking_frame
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

logger = logging.getLogger("temporal.fuxictr")

FUXICTR_VERSION = "2.3.9"
FUXICTR_COMMIT = "b7dff736885fdb8f59387d82d08219ad2e4cae50"
MIND_GROUP_COL = "impression_id"
MIND_TARGET_COL = "click"
MIND_ITEM_COL = "candidate_news_id"
MIND_CAT_FEATURES = ["user_id", "candidate_news_id", "category", "subcategory"]
MIND_NUM_FEATURES = ["history_len", "entity_embedding_count"]
RANKING_METRIC_KS = [5, 10, 100]


def pinned_dcnv2_source() -> Path:
    """Return the path to the exact upstream model-zoo source used in the article."""
    return Path(__file__).resolve().parents[1] / "vendor" / "fuxictr_v2_3_9" / "DCNv2.py"


def load_pinned_dcnv2() -> type[Any]:
    """Load the pinned upstream model-zoo class from its source snapshot."""
    source = pinned_dcnv2_source()
    if not source.is_file():
        raise FileNotFoundError(
            "The pinned FuxiCTR DCNv2 model-zoo snapshot is missing: "
            f"{source}. Restore exps/vendor/fuxictr_v2_3_9 before running the reference.",
        )
    spec = importlib.util.spec_from_file_location("fuxictr_v2_3_9_dcnv2", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load vendored FuxiCTR DCNv2 from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DCNv2


def scan_split(data_dir: Path, split: str) -> pl.LazyFrame:
    path = data_dir / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pl.scan_parquet(path)


def prepare_avazu(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.with_columns(
        pl.col("hour").cast(pl.String).str.slice(-2).alias("hour_of_day"),
    ).drop([column for column in ["id", "hour"] if column in frame.collect_schema()])


def prepare_mind(frame: pl.LazyFrame) -> pl.LazyFrame:
    schema = frame.collect_schema().names()
    required = {MIND_GROUP_COL, MIND_TARGET_COL, "source_split"}
    missing = sorted(required - set(schema))
    if missing:
        raise ValueError(f"MIND split is missing columns: {missing}")
    keep = [
        column
        for column in [
            MIND_GROUP_COL,
            "source_split",
            MIND_TARGET_COL,
            *MIND_CAT_FEATURES,
            *MIND_NUM_FEATURES,
        ]
        if column in schema
    ]
    frame = (
        frame.select(keep)
        .with_columns(
            pl.concat_str(
                [pl.col("source_split").cast(pl.String), pl.col(MIND_GROUP_COL).cast(pl.String)],
                separator=":",
            ).alias(MIND_GROUP_COL),
        )
        .drop("source_split")
    )
    if "entity_embedding_count" in keep:
        frame = frame.with_columns(pl.col("entity_embedding_count").fill_null(0))
    return frame


def fit_categorical_vocabularies(
    train: pl.LazyFrame,
    categorical_features: list[str],
) -> dict[str, list[str]]:
    if not categorical_features:
        return {}
    row = (
        train.select(
            [
                pl.col(name).cast(pl.String).fill_null("").unique().sort().implode().alias(name)
                for name in categorical_features
            ],
        )
        .collect()
        .row(0, named=True)
    )
    return {name: list(row[name]) for name in categorical_features}


def categorical_code(name: str, categories: list[str]) -> pl.Expr:
    encoded = (
        pl.col(name)
        .cast(pl.String)
        .fill_null("")
        .cast(pl.Enum(categories), strict=False)
        .to_physical()
    )
    return pl.when(encoded.is_null()).then(0).otherwise(encoded + 1).cast(pl.Int64).alias(name)


def encode_frame(
    frame: pl.LazyFrame,
    *,
    categorical_vocabularies: dict[str, list[str]],
    numeric_features: list[str],
    group_col: str | None,
    target_col: str,
) -> pl.LazyFrame:
    expressions: list[pl.Expr] = [
        categorical_code(name, categories) for name, categories in categorical_vocabularies.items()
    ]
    expressions.extend(
        pl.col(name).cast(pl.Float32).fill_null(0.0).alias(name) for name in numeric_features
    )
    if group_col is not None:
        expressions.append(
            pl.col(group_col).cast(pl.String).rank(method="dense").cast(pl.Int64).alias(group_col),
        )
    expressions.append(pl.col(target_col).cast(pl.Float32).alias(target_col))
    return frame.select(expressions)


def write_feature_map(
    path: Path,
    *,
    dataset_id: str,
    categorical_vocabularies: dict[str, list[str]],
    numeric_features: list[str],
    group_col: str | None,
    target_col: str,
) -> None:
    features: OrderedDict[str, dict[str, Any]] = OrderedDict()
    if group_col is not None:
        features[group_col] = {"type": "meta", "source": ""}
    for name, categories in categorical_vocabularies.items():
        features[name] = {
            "type": "categorical",
            "source": "",
            "vocab_size": len(categories) + 1,
        }
    for name in numeric_features:
        features[name] = {"type": "numeric", "source": ""}
    payload = {
        "dataset_id": dataset_id,
        "num_fields": len(categorical_vocabularies) + len(numeric_features),
        "total_features": sum(len(values) + 1 for values in categorical_vocabularies.values())
        + len(numeric_features),
        "input_length": len(features),
        "labels": [target_col],
        "features": [{name: spec} for name, spec in features.items()],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prepare_encoded_data(
    args: argparse.Namespace,
) -> tuple[Path, str, list[str], list[str], str | None, dict[str, pl.DataFrame]]:
    raw_frames: dict[str, pl.LazyFrame] = {
        "train": scan_split(args.data_dir, "train"),
        "valid": scan_split(args.data_dir, "eval"),
    }
    if args.evaluate_test:
        raw_frames["test"] = scan_split(args.data_dir, "test")

    if args.dataset == "avazu":
        frames = {name: prepare_avazu(frame) for name, frame in raw_frames.items()}
        categorical_features = [
            name for name in frames["train"].collect_schema().names() if name != args.target_col
        ]
        numeric_features: list[str] = []
        group_col = None
    else:
        frames = {name: prepare_mind(frame) for name, frame in raw_frames.items()}
        schema = frames["train"].collect_schema().names()
        categorical_features = [name for name in MIND_CAT_FEATURES if name in schema]
        numeric_features = [name for name in MIND_NUM_FEATURES if name in schema]
        group_col = args.group_col

    vocabularies = fit_categorical_vocabularies(frames["train"], categorical_features)
    dataset_id = f"{args.dataset}_temporal_fuxictr"
    encoded_root = args.encoded_root or (args.output_dir / "encoded")
    encoded_dir = encoded_root / dataset_id
    encoded_dir.mkdir(parents=True, exist_ok=True)
    write_feature_map(
        encoded_dir / "feature_map.json",
        dataset_id=dataset_id,
        categorical_vocabularies=vocabularies,
        numeric_features=numeric_features,
        group_col=group_col,
        target_col=args.target_col,
    )

    offline_frames: dict[str, pl.DataFrame] = {}
    for split, frame in frames.items():
        offline_columns = [args.target_col]
        if group_col is not None:
            offline_columns.append(group_col)
        if args.dataset == "mind" and MIND_ITEM_COL in frame.collect_schema():
            offline_columns.append(MIND_ITEM_COL)
        offline_frames[split] = frame.select(offline_columns).collect()
        encoded = encode_frame(
            frame,
            categorical_vocabularies=vocabularies,
            numeric_features=numeric_features,
            group_col=group_col,
            target_col=args.target_col,
        )
        encoded.sink_parquet(encoded_dir / f"{split}.parquet")

    return (
        encoded_dir,
        dataset_id,
        categorical_features,
        numeric_features,
        group_col,
        offline_frames,
    )


def source_sha256() -> str:
    return hashlib.sha256(pinned_dcnv2_source().read_bytes()).hexdigest()


def fuxictr_config(
    args: argparse.Namespace,
    *,
    dataset_id: str,
    encoded_dir: Path,
    group_col: str | None,
) -> dict[str, Any]:
    monitor = "AUC" if args.dataset == "avazu" else "NDCG(10)"
    metrics = [monitor]
    return {
        "dataset_id": dataset_id,
        "model_id": Path(args.output_dir).name,
        "model_root": str(args.output_dir / "checkpoints"),
        "train_data": str(encoded_dir / "train.parquet"),
        "valid_data": str(encoded_dir / "valid.parquet"),
        "test_data": str(encoded_dir / "test.parquet") if args.evaluate_test else None,
        "data_format": "parquet",
        "streaming": False,
        "num_workers": args.num_workers,
        "verbose": args.verbose,
        "early_stop_patience": args.early_stopping_rounds,
        "save_best_only": True,
        "eval_steps": None,
        "group_id": group_col,
        "model": "DCNv2",
        "loss": "binary_crossentropy",
        "metrics": metrics,
        "task": "binary_classification",
        "optimizer": "adam",
        "model_structure": args.model_structure,
        "use_low_rank_mixture": False,
        "low_rank": 32,
        "num_experts": 4,
        "learning_rate": args.learning_rate,
        "embedding_regularizer": args.embedding_regularizer,
        "net_regularizer": args.net_regularizer,
        "batch_size": args.batch_size,
        "embedding_dim": args.embedding_dim,
        "stacked_dnn_hidden_units": args.hidden_units,
        "parallel_dnn_hidden_units": args.hidden_units,
        "dnn_activations": "relu",
        "num_cross_layers": args.cross_layers,
        "net_dropout": args.dropout,
        "batch_norm": args.batch_norm,
        "epochs": args.epochs,
        "shuffle": True,
        "seed": args.random_state,
        "monitor": monitor,
        "monitor_mode": "max",
        "reduce_lr_on_plateau": True,
        "max_gradient_norm": 10.0,
        "gpu": args.gpu,
        "fuxictr_version": FUXICTR_VERSION,
        "fuxictr_commit": FUXICTR_COMMIT,
        "dcnv2_source_sha256": source_sha256(),
    }


def evaluate_predictions(
    args: argparse.Namespace,
    split: str,
    offline: pl.DataFrame,
    scores: np.ndarray,
) -> tuple[dict[str, float | int], pl.DataFrame]:
    prediction_frame = offline.with_columns(pl.Series("score", scores.astype(np.float32)))
    if args.dataset == "avazu":
        labels = offline.get_column(args.target_col).to_numpy()
        predictions = (scores >= 0.5).astype(np.int64)
        metrics: dict[str, float | int] = {
            f"{split}_auc": float(roc_auc_score(labels, scores)),
            f"{split}_log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        }
        if split == "test":
            metrics["test_accuracy"] = float(accuracy_score(labels, predictions))
        return metrics, prediction_frame

    ranking = evaluate_ranking_frame(
        prediction_frame,
        group_col=args.group_col,
        target_col=args.target_col,
        score_col="score",
        ks=RANKING_METRIC_KS,
    )
    return {f"{split}_{name}": value for name, value in ranking.items()}, prediction_frame


def train(args: argparse.Namespace) -> None:
    # PyPI provides the FuxiCTR runtime below, but not model-zoo classes such as
    # DCNv2. The model class is therefore loaded from the pinned official source
    # snapshot in exps/vendor; it still uses this same installed 2.3.9 runtime.
    try:
        import fuxictr  # noqa: PLC0415
        from fuxictr.features import FeatureMap  # noqa: PLC0415
        from fuxictr.pytorch.dataloaders import RankDataLoader  # noqa: PLC0415
        from fuxictr.pytorch.torch_utils import seed_everything  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "FuxiCTR runtime is missing. Install exactly fuxictr==2.3.9 with --no-deps.",
        ) from exc
    DCNv2 = load_pinned_dcnv2()

    if fuxictr.__version__ != FUXICTR_VERSION:
        raise RuntimeError(f"Expected FuxiCTR {FUXICTR_VERSION}, got {fuxictr.__version__}")
    np.__dict__.setdefault("Inf", np.inf)  # FuxiCTR v2.3.9 compatibility with NumPy 2

    random.seed(args.random_state)
    np.random.seed(args.random_state)  # noqa: NPY002 - required by FuxiCTR
    seed_everything(seed=args.random_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoded_dir, dataset_id, cat_features, num_features, group_col, offline_frames = (
        prepare_encoded_data(args)
    )
    config = fuxictr_config(
        args,
        dataset_id=dataset_id,
        encoded_dir=encoded_dir,
        group_col=group_col,
    )
    config.update({"categorical_features": cat_features, "numeric_features": num_features})

    feature_map = FeatureMap(dataset_id, str(encoded_dir))
    feature_map.load(str(encoded_dir / "feature_map.json"), config)
    model = DCNv2(feature_map, **config)
    Path(model.model_dir).mkdir(parents=True, exist_ok=True)
    n_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    train_gen, valid_gen = RankDataLoader(feature_map, stage="train", **config).make_iterator()
    fit_started = time.perf_counter()
    model.fit(train_gen, validation_data=valid_gen, **config)
    fit_seconds = time.perf_counter() - fit_started

    metrics: dict[str, float | int] = {
        "fit_seconds": fit_seconds,
        "n_parameters": n_parameters,
        "seed": args.random_state,
    }
    eval_started = time.perf_counter()
    eval_scores = model.predict(valid_gen)
    metrics["eval_predict_seconds"] = time.perf_counter() - eval_started
    eval_metrics, eval_predictions = evaluate_predictions(
        args,
        "eval",
        offline_frames["valid"],
        eval_scores,
    )
    metrics.update(eval_metrics)
    eval_predictions.write_parquet(args.output_dir / "eval_predictions.parquet")

    if args.evaluate_test:
        test_gen = RankDataLoader(feature_map, stage="test", **config).make_iterator()
        test_scores = model.predict(test_gen)
        test_metrics, test_predictions = evaluate_predictions(
            args,
            "test",
            offline_frames["test"],
            test_scores,
        )
        metrics.update(test_metrics)
        test_predictions.write_parquet(args.output_dir / "test_predictions.parquet")

    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    logger.info("FuxiCTR metrics:\n%s", json.dumps(metrics, indent=2, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["avazu", "mind"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--encoded-root",
        type=Path,
        default=None,
        help="Temporary encoded-data root; keep it outside the output artifacts directory.",
    )
    parser.add_argument("--target-col", default=MIND_TARGET_COL)
    parser.add_argument("--group-col", default=MIND_GROUP_COL)
    parser.add_argument(
        "--evaluate-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep disabled for validation search; enable only for the final run.",
    )
    parser.add_argument("--model-structure", choices=["stacked", "parallel"], required=True)
    parser.add_argument("--hidden-units", nargs="+", type=int, required=True)
    parser.add_argument("--cross-layers", type=int, required=True)
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--batch-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--embedding-regularizer", type=float, default=0.0)
    parser.add_argument("--net-regularizer", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--early-stopping-rounds", type=int, required=True)
    parser.add_argument("--random-state", type=int, default=2021)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--verbose", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
