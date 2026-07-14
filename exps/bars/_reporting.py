"""ClearML tracking + on-disk artifacts for ``train.py``.

ClearML is optional — every function no-ops when tracking is off or unavailable. The
model is never persisted; only metrics/history/predictions are written to ``--output-dir``.
"""
from __future__ import annotations
import json
import logging
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

try:
    from clearml import Task
except ImportError:  # pragma: no cover
    Task = None

logger = logging.getLogger("boosting.experiment")


# --------------------------------------------------------------------------- #
# ClearML tracking — optional (no automatic task-name formation)
# --------------------------------------------------------------------------- #
def init_clearml(args: argparse.Namespace, params: dict[str, Any]) -> Any | None:
    """Initialise a ClearML task, or return None (and warn) if unavailable.

    Task name is ``args.clearml_task`` verbatim; ClearML auto-names when it is omitted.
    """
    if not args.clearml:
        return None
    if Task is None:
        logger.warning("--clearml requested but the `clearml` package is not installed; skipping.")
        return None
    try:
        task = Task.init(
            project_name=args.clearml_project,
            task_name=args.clearml_task,
            tags=args.clearml_tags,
        )
    except Exception as exc:  # noqa: BLE001 - never let tracking crash the run
        logger.warning("ClearML init failed (%s); continuing without tracking.", exc)
        return None
    task.connect({k: json.dumps(v, default=str) for k, v in params.items()})
    logger.info("ClearML tracking enabled: project=%s", args.clearml_project)
    return task


def report_history_to_clearml(task: Any | None, history: list[dict[str, float]]) -> None:
    """Per-round loss + val AUC curves (analog of train_dcn's eval callback)."""
    if task is None:
        return
    log = task.get_logger()
    for record in history:
        epoch = int(record["epoch"])
        if "train_loss" in record:
            log.report_scalar("loss", "train", value=record["train_loss"], iteration=epoch)
        if "val_loss" in record:
            log.report_scalar("loss", "val", value=record["val_loss"], iteration=epoch)
        if "val_auc" in record:
            log.report_scalar("auc", "val", value=record["val_auc"], iteration=epoch)


def report_metrics_to_clearml(task: Any | None, metrics: dict[str, float]) -> None:
    if task is None:
        return
    log = task.get_logger()
    for name, value in metrics.items():
        log.report_single_value(name, value)
    # Also as scalar plots pinned at iteration=0 so "best" scalars align across runs.
    for split in ("val", "test"):
        if f"{split}_auc" in metrics:
            log.report_scalar("auc_best", split, value=metrics[f"{split}_auc"], iteration=0)
        if f"{split}_log_loss" in metrics:
            log.report_scalar(
                "log_loss_best", split, value=metrics[f"{split}_log_loss"], iteration=0,
            )


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def save_artifacts(
    output_dir: Path,
    *,
    metrics: dict[str, float],
    history: list[dict[str, float]],
    predictions: pl.DataFrame,
    task: Any | None,
) -> None:
    """Persist metrics, per-round history and test predictions (the model is not saved)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    history_path = output_dir / "history.csv"
    if history:
        pl.DataFrame(history).write_csv(history_path)

    predictions_path = output_dir / "predictions.parquet"
    predictions.write_parquet(predictions_path)

    logger.info("Artifacts written to %s", output_dir.resolve())

    if task is not None:
        task.upload_artifact("metrics", artifact_object=metrics_path)
