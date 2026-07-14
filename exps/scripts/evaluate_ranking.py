"""Offline grouped-ranking metrics for temporal recommendation experiments.

The script intentionally evaluates a saved predictions file instead of relying on
trainer-internal metrics. That keeps MIND evaluation identical for scikit_rank,
LightGBM/CatBoost/XGBoost/RandomForest, and FuxiCTR baselines.

Expected input columns by default:

``impression_id`` | ``click`` | ``score``

where a larger ``score`` means a candidate should be ranked higher.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def dcg(labels: np.ndarray, k: int) -> float:
    labels = labels[:k].astype(np.float64, copy=False)
    if labels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, labels.size + 2, dtype=np.float64))
    return float((labels * discounts).sum())


def ndcg_at_k(labels: np.ndarray, k: int) -> float:
    denom = dcg(np.sort(labels)[::-1], k)
    if denom <= 0.0:
        return 0.0
    return dcg(labels, k) / denom


def recall_at_k(labels: np.ndarray, k: int) -> float:
    total_positives = float((labels > 0).sum())
    if total_positives <= 0.0:
        return 0.0
    topk_positives = float((labels[:k] > 0).sum())
    return topk_positives / total_positives


def mrr_at_k(labels: np.ndarray, k: int) -> float:
    for rank, label in enumerate(labels[:k], start=1):
        if label > 0:
            return 1.0 / rank
    return 0.0


def scan_predictions(path: Path) -> pl.LazyFrame:
    if path.suffix == ".parquet":
        return pl.scan_parquet(path)
    if path.suffix == ".csv":
        return pl.scan_csv(path)
    raise ValueError(f"Unsupported predictions format {path.suffix!r}; use .parquet or .csv")


def evaluate_ranking_frame(
    frame: pl.DataFrame,
    *,
    group_col: str,
    target_col: str,
    score_col: str,
    ks: list[int],
) -> dict[str, float | int]:
    if not ks or any(k <= 0 for k in ks):
        raise ValueError(f"ks must contain positive integers, got {ks!r}")
    ks = sorted(set(ks))
    required = {group_col, target_col, score_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Predictions file is missing required columns: {missing}")
    if frame.is_empty():
        raise ValueError("Cannot evaluate an empty predictions frame")
    null_counts = frame.select(
        [pl.col(column).null_count().alias(column) for column in required],
    ).row(0, named=True)
    columns_with_nulls = sorted(column for column, count in null_counts.items() if count)
    if columns_with_nulls:
        raise ValueError(f"Ranking columns contain nulls: {columns_with_nulls}")
    labels = frame.get_column(target_col).cast(pl.Float64)
    unique_labels = set(labels.unique().to_list())
    if not unique_labels <= {0.0, 1.0}:
        raise ValueError(
            f"Ranking metrics expect binary relevance labels, got {sorted(unique_labels)!r}",
        )
    scores = frame.get_column(score_col).cast(pl.Float64).to_numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Ranking scores contain NaN or infinite values")

    # Preserve source candidate order as a deterministic tie-breaker. Exact score
    # ties otherwise depend on the group-by implementation and can make reruns differ.
    frame = frame.with_row_index("_row_order")

    groups = frame.group_by(group_col).agg(
        pl.len().alias("_n_candidates"),
        pl.col(target_col).sum().alias("_n_positives"),
    )
    group_stats = groups.select(
        pl.len().alias("n_groups"),
        pl.col("_n_candidates").mean().alias("avg_candidates_per_group"),
        pl.col("_n_candidates").max().alias("max_candidates_per_group"),
        (pl.col("_n_positives") > 0).mean().alias("groups_with_positive_rate"),
    ).row(0, named=True)
    if float(group_stats["groups_with_positive_rate"]) < 1.0:
        raise ValueError(
            "Every MIND impression must contain at least one clicked candidate; "
            f"observed rate={group_stats['groups_with_positive_rate']:.6f}",
        )

    metrics: dict[str, list[float]] = {}
    for metric_name in ("recall", "ndcg", "mrr"):
        for k in ks:
            metrics[f"{metric_name}@{k}"] = []
    metrics["mrr"] = []
    for _, group_df in frame.group_by(group_col):
        labels = (
            group_df.sort(
                [score_col, "_row_order"],
                descending=[True, False],
            )
            .get_column(target_col)
            .cast(pl.Float64)
            .to_numpy()
        )
        for k in ks:
            metrics[f"recall@{k}"].append(recall_at_k(labels, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(labels, k))
            metrics[f"mrr@{k}"].append(mrr_at_k(labels, k))
        metrics["mrr"].append(mrr_at_k(labels, len(labels)))

    result: dict[str, float | int] = {
        "n_rows": len(frame),
        "n_groups": int(group_stats["n_groups"]),
        "avg_candidates_per_group": float(group_stats["avg_candidates_per_group"]),
        "max_candidates_per_group": int(group_stats["max_candidates_per_group"]),
        "groups_with_positive_rate": float(group_stats["groups_with_positive_rate"]),
    }
    result.update({name: float(np.mean(values)) for name, values in metrics.items()})
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate grouped ranking predictions with Recall/NDCG/MRR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--group-col", default="impression_id")
    parser.add_argument("--target-col", default="click")
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 100])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    frame = scan_predictions(args.predictions).collect()
    metrics = evaluate_ranking_frame(
        frame,
        group_col=args.group_col,
        target_col=args.target_col,
        score_col=args.score_col,
        ks=args.ks,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
