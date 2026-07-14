"""Unified experiment runner: train a CTR model on a pre-split BARS dataset.

Config-driven runner for the three GBDT baselines (LightGBM/CatBoost/XGBoost) or DCNv2
(via ``scikit_rank.DCNClassifier``) on ``avazu_x1`` / ``criteo_x1``.

Everything lives in a YAML ``--config`` (model, dataset, feature lists, data_dir and a
nested ``model_params:`` of native library kwargs); the CLI carries only ``--output-dir``,
``--random-state`` and the ClearML flags. Each model is a ``ModelAdapter`` (``_adapters.py``)
that owns its whole pipeline. Helpers live in ``_data.py`` / ``_adapters.py`` / ``_reporting.py``.

Example:
-------
    uv run --group lgbm python exps/bars/train.py \
        --config exps/bars/configs/config_lgbm_criteo_x1.yaml
    uv run python exps/bars/train.py \
        --config exps/bars/configs/config_dcn_criteo_x1.yaml     # DCN: full env

"""

from __future__ import annotations
import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from sklearn.metrics import log_loss, roc_auc_score

# Put this dir on sys.path so the sibling helper modules import cleanly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _adapters import ADAPTERS
from _data import load_raw_splits, verify_split_md5
from _datasets import DATASETS
from _reporting import (
    init_clearml,
    report_history_to_clearml,
    report_metrics_to_clearml,
    save_artifacts,
)

logger = logging.getLogger("boosting.experiment")

# The BARS datasets this runner reproduces — each must have an entry in the registry.
SUPPORTED_DATASETS = ("avazu_x1", "criteo_x1")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# CLI + config
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    """Argparse parser — config path + run/tracking options.

    The whole run lives in the YAML ``--config``; ``--random-state`` overrides its seed so
    seed-sweeps reuse one config.
    """
    p = argparse.ArgumentParser(
        description="Train a CTR classifier (lgbm/catboost/xgboost/dcn) from a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", required=True, metavar="PATH",
        help="path to the YAML run-config (model, dataset, features, model_params, ...)",
    )
    p.add_argument("--output-dir", default="./output")
    p.add_argument(
        "--random-state", type=int, default=None,
        help="seed override: seeds the RNG AND the model seed, superseding the config's",
    )

    run = p.add_argument_group("tracking")
    run.add_argument("--clearml", action="store_true", help="log to ClearML if available")
    run.add_argument("--clearml-project", default="scikit_rank-ctr")
    run.add_argument(
        "--clearml-task", default=None,
        help="ClearML task name (if omitted, ClearML auto-names the task)",
    )
    run.add_argument(
        "--clearml-tags", nargs="+", default=None, metavar="TAG",
        help="space-separated tags for the ClearML task (e.g. --clearml-tags lgbm criteo_x1)",
    )
    return p


def load_config(path: Path) -> dict[str, Any]:
    """Load + validate the YAML run-config, failing fast with a clear message."""
    if not path.is_file():
        raise SystemExit(f"Config file not found: {path}")
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config must be a YAML mapping, got {type(cfg).__name__}: {path}")
    model = cfg.get("model")
    if model not in ADAPTERS:
        raise ValueError(f"config 'model' must be one of {tuple(sorted(ADAPTERS))}, got {model!r}")
    dataset = cfg.get("dataset")
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"config 'dataset' must be one of {SUPPORTED_DATASETS}, got {dataset!r}",
        )
    for key in ("num_features", "cat_features"):
        if not isinstance(cfg.get(key), list):
            raise ValueError(f"config '{key}' is required and must be a list of column names")
    if not isinstance(cfg.get("data_dir"), str) or not cfg["data_dir"]:
        raise ValueError("config 'data_dir' is required and must be a non-empty path string")
    model_params = cfg.get("model_params")
    if model_params is not None and not isinstance(model_params, dict):
        raise ValueError("config 'model_params' must be a mapping")
    return cfg


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(Path(args.config))
    model_name = cfg["model"]
    adapter = ADAPTERS[model_name]

    # --random-state, if passed, overrides the model's seed key (random_seed for CatBoost).
    params = dict(cfg.get("model_params") or {})
    if args.random_state is not None:
        params[adapter.seed_param] = args.random_state
    seed = int(params.get(adapter.seed_param, 2021))
    seed_everything(seed)

    # 1. Dataset -----------------------------------------------------------
    # Column types/data_dir come from the config; the registry supplies label + MD5s.
    ds = DATASETS[cfg["dataset"]]
    num_features = list(cfg["num_features"])
    cat_features = list(cfg["cat_features"])
    data_dir = Path(cfg["data_dir"])
    # Fail fast if the split isn't the BARS reference; skip with ``check_md5: false``.
    if cfg.get("check_md5", True):
        verify_split_md5(ds.md5s, data_dir)
    logger.info("Model=%s | loading %s from %s", model_name, ds.name, data_dir.resolve())
    train_lazy, x_val, y_val, x_test, y_test = load_raw_splits(
        data_dir, cfg.get("max_train_rows"), ds.label,
    )
    logger.info(
        "val=%d test=%d | click-rate val=%.3f test=%.3f",
        len(y_val), len(y_test), y_val.mean(), y_test.mean(),
    )
    logger.info("Numeric features (%d): %s", len(num_features), num_features)
    logger.info("Categorical features (%d): %s", len(cat_features), cat_features)

    # 2. Train + predict — the adapter owns preprocessing. -----------------
    logger.info("%s params:\n%s", model_name, json.dumps(params, indent=2, default=str))
    task = init_clearml(args, params)
    history, val_proba, test_proba, best_iter = adapter.run(
        cfg, train_lazy, x_val, y_val, x_test,
        label=ds.label, num_features=num_features, cat_features=cat_features,
        params=params, seed=seed, verbose=bool(cfg.get("verbose", False)),
    )
    if task is not None:
        task.get_logger().report_single_value("best_iteration", best_iter)
    report_history_to_clearml(task, history)

    # 3. Evaluate on val + test --------------------------------------------
    logger.info("Evaluating on the val and test splits...")
    metrics = {
        "val_auc": float(roc_auc_score(y_val, val_proba)),
        "val_log_loss": float(log_loss(y_val, val_proba, labels=[0, 1])),
        "test_auc": float(roc_auc_score(y_test, test_proba)),
        "test_log_loss": float(log_loss(y_test, test_proba, labels=[0, 1])),
        "best_iteration": best_iter,
    }
    logger.info("Best model on val — AUC=%.4f | log_loss=%.4f",
                metrics["val_auc"], metrics["val_log_loss"])
    logger.info("Test — AUC=%.4f | log_loss=%.4f",
                metrics["test_auc"], metrics["test_log_loss"])
    report_metrics_to_clearml(task, metrics)

    # 4. Artifacts ---------------------------------------------------------
    predictions = pl.DataFrame(
        {"y_true": y_test.astype(np.int64), "proba": test_proba.astype(np.float32)},
    )
    save_artifacts(
        Path(args.output_dir),
        metrics=metrics,
        history=history,
        predictions=predictions,
        task=task,
    )

    if task is not None:
        task.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
