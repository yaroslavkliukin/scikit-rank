from __future__ import annotations
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from accelerate.utils import gather_object
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from ignite.handlers.tqdm_logger import ProgressBar
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from scikit_rank.train.trainer import ModelEvents

if TYPE_CHECKING:
    from accelerate import Accelerator
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    from scikit_rank.train.trainer import Trainer


logger = logging.getLogger(__name__)


# Validation-metric contract shared by ``attach_metric`` and ``TrainingRun``:
# ``metric_fn(y_true, y_pred[, group]) -> float`` where every argument is an
# ``np.ndarray``. ``group`` is passed only in the group-aware regime
# (``eval_metric_group_aware`` / ``group_aware=True``).
EvalMetricFn = (
    Callable[[np.ndarray, np.ndarray], float]
    | Callable[[np.ndarray, np.ndarray, np.ndarray | None], float]
)


class BestStateSaver:
    """Keeps the best (lowest eval loss) model state in memory and restores it."""

    def __init__(
        self,
        module: torch.nn.Module,
        accelerator: Accelerator,
        metric_name: str = "loss",
        direction: str = "min",
    ) -> None:
        assert direction in {"min", "max"}, (
            f"Direction is not valid (expected: 'min', 'max'), got: {direction!r})"
        )

        self.best_metric = float("inf") if direction == "min" else -float("inf")
        self._direction = direction
        self._metric_name = metric_name
        self._module = module
        self._accelerator = accelerator
        self._best_state: dict[str, torch.Tensor] | None = None

    def __call__(self, engine: Engine) -> None:
        metric = float(engine.state.metrics[self._metric_name])
        improvement = (
            metric < self.best_metric if self._direction == "min" else metric > self.best_metric
        )
        if improvement:
            self.best_metric = metric
            unwrapped = self._accelerator.unwrap_model(self._module)
            self._best_state = {
                k: v.detach().cpu().clone() for k, v in unwrapped.state_dict().items()
            }

    def restore(self) -> None:
        if self._best_state is None:
            return
        self._accelerator.unwrap_model(self._module).load_state_dict(self._best_state)


class EmaAverager:
    """Exponential moving average (EMA) of the module weights, opt-in.

    Ported from ``neural-ranking`` (``run/train.py``): wraps the trained module
    in :class:`torch.optim.swa_utils.AveragedModel` with
    :func:`~torch.optim.swa_utils.get_ema_multi_avg_fn`, so after every optimizer
    step the averaged weights follow
    ``theta_ema <- decay * theta_ema + (1 - decay) * theta_model`` (the first
    update is a plain copy of the model weights). With ``use_buffers=False`` (the
    default) the BatchNorm / other buffers are hard-copied from the live module
    on every update, so running stats stay current and ``update_bn`` is never
    needed.

    The swap discipline mirrors :class:`BestStateSaver`: :meth:`store` loads the
    averaged weights into the live module (backing up the raw ones) so evaluation
    and best-state selection observe the EMA model; :meth:`restore_live` swaps the
    raw weights back for continued training; and :meth:`copy_to` makes the average
    permanent when there is no validation set to select a best epoch from.

    ``update`` / ``store`` / ``restore_live`` take no arguments so ignite calls
    them without the engine (the events they hook carry no per-call state).
    """

    def __init__(
        self,
        module: torch.nn.Module,
        accelerator: Accelerator,
        decay: float,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {decay!r}")
        self._module = module
        self._accelerator = accelerator
        self._ema = AveragedModel(
            accelerator.unwrap_model(module),
            multi_avg_fn=get_ema_multi_avg_fn(decay),
        )
        self._backup: dict[str, torch.Tensor] | None = None

    def update(self) -> None:
        # Advance the average once per real optimizer step. Gating on
        # ``sync_gradients`` skips the gradient-accumulation micro-batches, the
        # same guard grad clipping uses in ``Trainer._train_step``.
        if self._accelerator.sync_gradients:
            self._ema.update_parameters(self._accelerator.unwrap_model(self._module))

    def store(self) -> None:
        unwrapped = self._accelerator.unwrap_model(self._module)
        self._backup = {k: v.detach().clone() for k, v in unwrapped.state_dict().items()}
        unwrapped.load_state_dict(self._ema.module.state_dict())

    def restore_live(self) -> None:
        if self._backup is None:
            return
        self._accelerator.unwrap_model(self._module).load_state_dict(self._backup)
        self._backup = None

    def copy_to(self) -> None:
        self._accelerator.unwrap_model(self._module).load_state_dict(
            self._ema.module.state_dict(),
        )


def attach_grad_clipping(
    trainer: Trainer,
    accelerator: Accelerator,
    max_norm: float,
) -> None:
    """Clip the global grad norm once per real optimizer step, opt-in.

    Registered on the train engine's ``BACKWARD_COMPLETED`` event so it runs
    after ``accelerator.backward`` and before ``optimizer.step`` (Accelerator
    handles AMP unscaling + DDP grad sync). Gating on ``sync_gradients`` skips
    the gradient-accumulation micro-batches -- the same guard
    :meth:`EmaAverager.update` uses -- so clipping runs once per optimizer step.
    The handler takes no arguments
    so ignite calls it without the engine.
    """

    def clip_gradients() -> None:
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(trainer.model.parameters(), max_norm)

    trainer.add_event("train", ModelEvents.BACKWARD_COMPLETED, clip_gradients)


def attach_early_stopping(
    trainer: Trainer,
    metric_name: str = "loss",
    patience: int = 10,
    direction: str = "min",
    min_delta: float = 1e-4,
) -> EarlyStopping:
    def score_function(engine: Engine) -> float:
        metric = engine.state.metrics[metric_name]
        sign = directions[direction]
        return sign * metric

    directions = {"min": -1.0, "max": 1.0}
    if direction not in directions:
        raise ValueError(
            f"Direction is not valid (expected: {list(directions)}, got: {direction})",
        )
    handler = EarlyStopping(
        patience,
        score_function,
        trainer=trainer.engines["train"],
        min_delta=min_delta,
    )
    trainer.add_event("eval", Events.COMPLETED, handler)
    return handler


def attach_lr_scheduler(
    trainer: Trainer,
    scheduler: ReduceLROnPlateau,
    metric_name: str = "loss",
) -> None:
    """Step a ``ReduceLROnPlateau`` on the eval ``metric_name`` each epoch.

    ``metric_name`` is the monitored value in ``engine.state.metrics`` -- the
    running eval loss, or a metric written there each epoch by :func:`attach_metric`
    -- so the same helper serves both the loss and the custom-metric paths.
    """

    def step(engine: Engine) -> None:
        scheduler.step(float(engine.state.metrics[metric_name]))

    trainer.add_event("eval", Events.COMPLETED, step)


def attach_metric(
    trainer: Trainer,
    accelerator: Accelerator,
    metric_name: str,
    metric_fn: EvalMetricFn,
    *,
    group_aware: bool = False,
) -> None:
    """Compute ``metric_fn(y_true, y_pred[, group])`` once per eval epoch into state.metrics.

    Generic over any sklearn-style metric: ``collect`` buffers the epoch's targets and
    raw model outputs; ``compute`` concatenates them at eval COMPLETED and scores the
    whole epoch once -- no per-metric assumptions (e.g. no sigmoid). The value lands in
    ``engine.state.metrics[metric_name]``.
    """
    logits_buf: list[torch.Tensor] = []
    targets_buf: list[torch.Tensor] = []
    group_buf: list[torch.Tensor] = []
    multi = accelerator.num_processes > 1

    def collect(engine: Engine) -> None:
        out = engine.state.output
        batch = engine.state.batch
        if "logits" not in out or batch.get("target") is None:
            return
        logits = out["logits"].detach()
        target = batch["target"].detach()
        if group_aware:
            # Buffer local shards; object-gather once at COMPLETED (below), so the
            # collective still runs on a rank whose shard is empty.
            logits_buf.append(logits.cpu())
            targets_buf.append(target.cpu())
            group_buf.append(batch["group"].detach().cpu())
        elif multi:
            # Drop the padding rows Accelerate added to even out the shards.
            logits_buf.append(accelerator.gather_for_metrics(logits).cpu())
            targets_buf.append(accelerator.gather_for_metrics(target).cpu())
        else:
            logits_buf.append(logits.cpu())
            targets_buf.append(target.cpu())

    def compute(engine: Engine) -> None:
        if group_aware and multi:
            # Gather the buffer lists (not a pre-cat tensor): an empty rank contributes
            # nothing and needs no dtype placeholder, and every rank runs the collective.
            logits_parts = gather_object(logits_buf)
            targets_parts = gather_object(targets_buf)
            group_parts = gather_object(group_buf)
        else:
            logits_parts = list(logits_buf)
            targets_parts = list(targets_buf)
            group_parts = list(group_buf)
        logits_buf.clear()
        targets_buf.clear()
        group_buf.clear()
        logits = torch.cat(logits_parts) if logits_parts else torch.empty(0)
        y_pred = logits.numpy()
        y_true = torch.cat(targets_parts).numpy()
        group = torch.cat(group_parts).numpy() if group_aware else None
        if group_aware and multi:
            # The in-memory ranking eval loader repeats whole group batches
            # (even_batches) to equalize per-rank counts, so object-gather can
            # surface a group twice; drop those exact-copy duplicates by group id.
            y_true, y_pred, group = _dedupe_by_group(y_true, y_pred, group)
        value = (
            float(metric_fn(y_true, y_pred, group))
            if group_aware
            else float(metric_fn(y_true, y_pred))
        )
        if np.isnan(value) and accelerator.is_main_process:
            logger.warning(
                "eval metric %r is NaN this epoch (degenerate eval, e.g. a single-class "
                "batch); passing NaN through to the monitors",
                metric_name,
            )
        engine.state.metrics[metric_name] = value

    trainer.add_event("eval", Events.ITERATION_COMPLETED, collect)
    trainer.add_event("eval", Events.COMPLETED, compute)


def _dedupe_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop ``even_batches``-duplicated groups from a gathered ranking eval set.

    Group ids are global and a group is never split across ranks, so a group id that
    appears in more than one contiguous run is an exact-copy duplicate the ranking
    sampler added to equalize per-rank batch counts. Keep the first run of each id and
    drop the rest. No-op when shards are disjoint (streaming dispatch, single process):
    every id then appears exactly once and the whole array is kept.
    """
    if group.size == 0:
        return y_true, y_pred, group
    starts = np.concatenate(([0], np.flatnonzero(group[1:] != group[:-1]) + 1))
    ends = np.concatenate((starts[1:], [group.size]))
    seen: set[object] = set()
    keep = np.ones(group.size, dtype=bool)
    for start, end in zip(starts, ends, strict=True):
        gid = group[start].item()
        if gid in seen:
            keep[start:end] = False
        else:
            seen.add(gid)
    return y_true[keep], y_pred[keep], group[keep]


def attach_embedding_regularizer(
    trainer: Trainer,
    embedding_params: list[torch.nn.Parameter],
    lam: float,
) -> Callable[[Engine], None]:
    """Add coupled embedding L2 to the training loss on FORWARD_COMPLETED.

    Keeps :class:`~scikit_rank.run.TrainingModule` a pure ``(model, loss)`` adapter:
    the penalty ``(lam / 2) * sum(w**2)`` over the embedding parameters is added
    to ``state.output["loss"]`` after the forward pass and before backward (see
    ``Trainer._train_step``), so it reaches the optimizer. Only the training
    engine is hooked -- the penalty is a training-time regularizer, so the eval
    loss remains unregularized. Returns the handler, mirroring
    :func:`attach_early_stopping`.
    """

    def add_penalty(engine: Engine) -> None:
        output = engine.state.output
        if "loss" not in output:
            return
        l2 = sum(p.pow(2).sum() for p in embedding_params)
        output["loss"] = output["loss"] + (lam / 2.0) * l2

    trainer.add_event("train", ModelEvents.FORWARD_COMPLETED, add_penalty)
    return add_penalty


def attach_epoch_logger(
    trainer: Trainer,
    history: list[dict[str, float]],
) -> None:
    pending: dict[str, float] = {}

    def log_eval(engine: Engine) -> None:
        for metric_name, metric_value in engine.state.metrics.items():
            pending[f"val_{metric_name}"] = float(metric_value)

    def log_train(engine: Engine) -> None:
        record = {
            "epoch": engine.state.epoch,
            "train_loss": float(engine.state.metrics["loss"]),
        }
        record.update(pending)
        pending.clear()
        history.append(record)

    trainer.add_event("train", Events.EPOCH_COMPLETED, log_train)
    trainer.add_event("eval", Events.COMPLETED, log_eval)


def attach_progress_bar(
    trainer: Trainer,
    metric_names: dict[str, str | list[str]] | None = None,
) -> None:
    metric_names = metric_names or {}
    for key, e in trainer.engines.items():
        pbar = ProgressBar(
            persist=True,
            bar_format=(
                "{desc} [{n_fmt}/{total_fmt}] "
                "{percentage:3.0f}%|{bar}|{postfix} "
                "({elapsed}<{remaining}, {rate_fmt})"
            ),
            desc="\033[33m" + key.capitalize() + "\033[00m",
        )
        pbar.attach(e, metric_names=metric_names.get(key))
