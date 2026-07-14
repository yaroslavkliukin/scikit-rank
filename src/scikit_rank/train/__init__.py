"""Training subsystem: ignite-based ``Trainer`` and pluggable event handlers."""

from scikit_rank.train.optimizers import OPTIMIZER_TYPES, OptimizerConfig, build_optimizer
from scikit_rank.train.options import (
    BestStateSaver,
    EmaAverager,
    attach_early_stopping,
    attach_epoch_logger,
    attach_progress_bar,
)
from scikit_rank.train.trainer import ModelEvents, Trainer

__all__ = [
    "OPTIMIZER_TYPES",
    "BestStateSaver",
    "EmaAverager",
    "ModelEvents",
    "OptimizerConfig",
    "Trainer",
    "attach_early_stopping",
    "attach_epoch_logger",
    "attach_progress_bar",
    "build_optimizer",
]
