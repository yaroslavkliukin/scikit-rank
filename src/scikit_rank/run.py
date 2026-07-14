"""Training-run objects: a parameterised fit packaged as a runnable command."""

import bisect
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.data_loader import slice_tensors
from accelerate.utils import broadcast
from ignite.engine import Events
from torch.utils.data import DataLoader

from scikit_rank.data import GroupAwareBatchSampler, TensorDatasetSource
from scikit_rank.modules.losses import Loss
from scikit_rank.train.optimizers import (
    LRSchedulerConfig,
    OptimizerConfig,
    build_lr_scheduler,
    build_optimizer,
)
from scikit_rank.train.options import (
    BestStateSaver,
    EmaAverager,
    EvalMetricFn,
    attach_early_stopping,
    attach_embedding_regularizer,
    attach_epoch_logger,
    attach_grad_clipping,
    attach_lr_scheduler,
    attach_metric,
    attach_progress_bar,
)
from scikit_rank.train.trainer import ModelEvents, Trainer


class TrainingModule(torch.nn.Module):
    """Adapt a (model, loss) pair to the Trainer's batch-dict contract.

    ``forward(batch)`` returns ``{"logits"[, "loss"]}``; the loss is computed
    only when the batch carries a target, so the same module serves training,
    evaluation and inference.
    """

    def __init__(self, model: torch.nn.Module, loss_fn: Loss) -> None:
        super().__init__()
        self._model = model
        self._loss_fn = loss_fn

    def model(self) -> torch.nn.Module:
        return self._model

    def loss_fn(self) -> Loss:
        return self._loss_fn

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = self._model(batch)
        output: dict[str, torch.Tensor] = {"logits": logits}
        if batch.get("target") is not None:
            target = batch["target"]
            if logits.size(0) != target.size(0):
                raise ValueError(
                    "scores and target have inconsistent batch lengths: "
                    f"{logits.size(0)} != {target.size(0)}",
                )
            group = batch.get("group")
            if group is not None and group.size(0) != logits.size(0):
                raise ValueError(
                    "scores and group have inconsistent batch lengths: "
                    f"{logits.size(0)} != {group.size(0)}",
                )
            if self.training and self._loss_fn.requires_group and group is None:
                raise ValueError(
                    "The configured loss requires `group`, but the training batch "
                    "does not contain group ids. Pass group=... to fit() or "
                    "choose a pointwise loss.",
                )
            output["loss"] = self._loss_fn(logits, target, group)
        return output


class RunOutput:
    """Concrete output of :class:`TrainingRun`."""

    def __init__(
        self,
        module: TrainingModule,
        metrics: dict[str, Any],
        history: list[dict[str, float]],
    ) -> None:
        self._module = module
        self._metrics = metrics
        self._history = history

    def module(self) -> TrainingModule:
        return self._module

    def metrics(self) -> dict[str, Any]:
        return self._metrics | {"history": self._history}


class TrainingRun:
    """Training command.

    Construction wires the Accelerator, optimizer, prepared model + loss,
    loaders, trainer and handlers; :meth:`run` executes the fit and returns
    a :class:`RunOutput`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Loss,
        train_source: Iterable[dict[str, torch.Tensor]],
        val_source: Iterable[dict[str, torch.Tensor]] | None = None,
        *,
        lr: float,
        weight_decay: float,
        epochs: int,
        optimizer: OptimizerConfig | None = None,
        early_stopping_rounds: int | None = None,
        verbose: bool = False,
        accelerator_config: dict[str, Any] | None = None,
        eval_metric_fn: EvalMetricFn | None = None,
        eval_metric_name: str = "metric",
        eval_metric_direction: str = "max",
        eval_metric_group_aware: bool = False,
        lr_scheduler: LRSchedulerConfig | None = None,
        grad_clip_norm: float | None = None,
        embedding_regularizer: float = 0.0,
        ema_decay: float | None = None,
    ) -> None:
        self._epochs = epochs
        self._early_stopping_rounds = early_stopping_rounds
        self._verbose = verbose
        self._grad_clip_norm = grad_clip_norm

        self._eval_metric_fn = eval_metric_fn
        self._eval_metric_name = "loss" if eval_metric_fn is None else eval_metric_name
        self._eval_metric_direction = "min" if eval_metric_fn is None else eval_metric_direction
        self._eval_metric_group_aware = eval_metric_group_aware

        self._accelerator = Accelerator(**(accelerator_config or {}))
        self._history: list[dict[str, float]] = []

        self._embedding_regularizer = embedding_regularizer
        self._embedding_params = (
            list(model.embedding_parameters())
            if embedding_regularizer > 0.0 and hasattr(model, "embedding_parameters")
            else []
        )
        self._module = TrainingModule(
            model=self._accelerator.prepare_model(model),
            loss_fn=(
                self._accelerator.prepare_model(loss_fn)
                if any(p.requires_grad for p in loss_fn.parameters())
                else loss_fn.to(self._accelerator.device)
            ),
        )
        optimizer = optimizer or OptimizerConfig()
        optimizer = replace(optimizer, lr=lr, weight_decay=weight_decay)
        if ema_decay is not None and optimizer.optimizer_type == "schedulefree_adamw":
            raise ValueError(
                "schedulefree_adamw already maintains an internal weight average; "
                "enabling ema_decay on top double-averages -- pick one.",
            )
        self._optimizer = self._accelerator.prepare_optimizer(
            build_optimizer(self._module, optimizer),
        )
        self._lr_scheduler = (
            build_lr_scheduler(self._optimizer, lr_scheduler, self._eval_metric_direction)
            if lr_scheduler is not None
            else None
        )
        self._loaders: dict[str, DataLoader] = {
            "train": _prepare_source_loader(train_source, self._accelerator),
        }
        if val_source is not None:
            self._loaders["eval"] = _prepare_source_loader(val_source, self._accelerator)
        self._trainer = Trainer(self._module, self._optimizer, self._accelerator)
        self._ema = (
            EmaAverager(self._module, self._accelerator, ema_decay)
            if ema_decay is not None
            else None
        )
        self._saver = self._attach_handlers()

    def run(self) -> RunOutput:
        state = self._trainer.run(self._loaders, epochs=self._epochs)
        # Leave schedule-free optimizers in eval mode.
        self._trainer.set_optimizer_mode(train=False)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        self._accelerator.end_training()
        if self._saver is not None:
            # With an eval set the saver already holds the best (EMA-if-enabled)
            # snapshot; restoring it also lands the averaged weights.
            self._saver.restore()
        elif self._ema is not None:
            # No eval set to select a best epoch: make the running average the
            # final model (mirrors neural-ranking's end-of-training EMA load).
            self._ema.copy_to()
        return RunOutput(
            module=TrainingModule(
                self._accelerator.unwrap_model(self._module.model()),
                self._accelerator.unwrap_model(self._module.loss_fn()),
            ),
            metrics=state.metrics,
            history=self._history,
        )

    def _attach_handlers(self) -> BestStateSaver | None:
        saver: BestStateSaver | None = None
        if self._grad_clip_norm is not None:
            attach_grad_clipping(self._trainer, self._accelerator, self._grad_clip_norm)
        if self._embedding_regularizer > 0.0 and self._embedding_params:
            attach_embedding_regularizer(
                self._trainer,
                self._embedding_params,
                self._embedding_regularizer,
            )
        if self._ema is not None:
            self._trainer.add_event("train", ModelEvents.OPTIMIZER_COMPLETED, self._ema.update)
            if "eval" in self._loaders:
                self._trainer.add_event("eval", Events.STARTED, self._ema.store)
                self._trainer.add_event("train", Events.EPOCH_COMPLETED, self._ema.restore_live)
        if "eval" in self._loaders:
            if self._eval_metric_fn is not None:
                attach_metric(
                    self._trainer,
                    self._accelerator,
                    self._eval_metric_name,
                    self._eval_metric_fn,
                    group_aware=self._eval_metric_group_aware,
                )
            saver = BestStateSaver(
                self._trainer.model,
                self._accelerator,
                metric_name=self._eval_metric_name,
                direction=self._eval_metric_direction,
            )
            self._trainer.add_event("eval", Events.COMPLETED, saver)
            if self._early_stopping_rounds is not None:
                attach_early_stopping(
                    self._trainer,
                    metric_name=self._eval_metric_name,
                    patience=self._early_stopping_rounds,
                    direction=self._eval_metric_direction,
                    min_delta=1e-6 if self._eval_metric_fn is not None else 1e-4,
                )
            if self._lr_scheduler is not None:
                attach_lr_scheduler(
                    self._trainer,
                    self._lr_scheduler,
                    metric_name=self._eval_metric_name,
                )
        if self._verbose and self._accelerator.is_main_process:
            attach_progress_bar(self._trainer, metric_names={"train": ["loss"], "eval": ["loss"]})
        attach_epoch_logger(self._trainer, self._history)
        return saver


def _prepare_source_loader(
    source: Iterable[dict[str, torch.Tensor]],
    accelerator: Accelerator,
) -> DataLoader:
    """Build the ``DataLoader`` for a batch source.

    In-memory data is sharded differently for the two loss regimes, while
    streaming uses Accelerate's dispatch path:

    * In-memory pointwise (:class:`TensorDatasetSource`, integer ``batch_size``)
      is a map-style ``Dataset`` batched by PyTorch and sharded across ranks by
      ``Accelerator.prepare_data_loader``.
    * In-memory ranking uses ``batch_sampler=GroupAwareBatchSampler`` so
      pairwise / listwise losses always see whole query groups.
    * Streaming (:class:`LazyArrowBatchSource`) is an ``IterableDataset`` that
      yields ready-made batches, so automatic batching is disabled with
      ``batch_size=None``
    """
    if isinstance(source, TensorDatasetSource):
        if source.group_np is None:
            generator = torch.Generator()
            generator.manual_seed(int(source.rng.integers(0, 2**63 - 1)))
            return accelerator.prepare_data_loader(
                DataLoader(
                    source,
                    batch_size=source.batch_size,
                    shuffle=source.shuffle,
                    generator=generator,
                ),
            )
        sampler = GroupAwareBatchSampler(
            source.group_np,
            source.batch_size,
            source.shuffle,
            seed=_shared_shuffle_seed(source.rng, accelerator),
        )
        return accelerator.prepare_data_loader(
            DataLoader(source, batch_sampler=sampler),
        )

    # Streaming IterableDataset: it already emits whole (group-contiguous)
    # batches, so DataLoader must not re-batch. PyTorch forbids sampler /
    # batch_sampler on an IterableDataset, and an integer batch_size would split
    # groups by row count, so batch_size=None (automatic batching disabled) is
    # the only API that preserves ranking groups for out-of-core data.
    return accelerator.prepare_data_loader(
        DataLoader(source, batch_size=None),
        slice_fn_for_dispatch=_slice_batch_for_process,
    )


def _shared_shuffle_seed(rng: Any, accelerator: Accelerator) -> int:
    """Draw a shuffle seed that is identical on every rank.

    The :class:`GroupAwareBatchSampler` must permute its shared batch list the
    same way on every rank or the per-rank shards would overlap. Each rank draws
    a seed from its own ``rng`` (identical when ``random_state`` is set, but not
    guaranteed otherwise), so under DDP we broadcast rank 0's value to all ranks.
    """
    seed = int(rng.integers(0, 2**31 - 1))
    if accelerator.num_processes > 1:
        seed = int(broadcast(torch.tensor(seed, device=accelerator.device)).item())
    return seed


def _balanced_group_cuts(starts: list[int], n_rows: int, num_processes: int) -> list[int]:
    """Choose ``num_processes + 1`` non-decreasing cut points snapped to group starts.

    If there are fewer groups than processes, later ranks receive empty
    slices: losses are required to handle empty batches (see
    ``_zero_loss_if_empty`` in :mod:`scikit_rank.losses`).
    """
    cuts = [0]
    for rank in range(1, num_processes):
        target = rank * n_rows / num_processes
        idx = bisect.bisect_left(starts, target)
        # snap to the nearest group start at or after the previous cut
        candidates = [
            s for s in (starts[max(idx - 1, 0)], starts[min(idx, len(starts) - 1)]) if s >= cuts[-1]
        ]
        cuts.append(min(candidates, key=lambda s: abs(s - target), default=cuts[-1]))
    cuts.append(n_rows)
    return cuts


def _slice_batch_for_process(
    batch: dict[str, torch.Tensor],
    tensor_slice: slice,
    process_index: int | None = None,
    num_processes: int | None = None,
) -> dict[str, torch.Tensor]:
    """Slice a dispatched batch while keeping ranking groups intact.

    Accelerate's default dispatcher concatenates ``num_processes`` already-made
    scikit_rank batches and slices by row count. That is fine for pointwise losses
    but can split a query group for pairwise / listwise losses. When ``group``
    is present we ignore ``tensor_slice`` and snap cuts to group starts instead.
    """
    if "group" not in batch or process_index is None or num_processes is None:
        return slice_tensors(batch, tensor_slice)

    group = batch["group"]
    if group.size(0) == 0 or num_processes <= 1:
        return slice_tensors(batch, tensor_slice)

    starts = (
        []
        if group.numel() == 0
        else [
            0,
            *((group[1:] != group[:-1]).nonzero(as_tuple=False).flatten() + 1).cpu().tolist(),
        ]
    )
    cuts = _balanced_group_cuts(starts, group.size(0), num_processes)
    return slice_tensors(batch, slice(cuts[process_index], cuts[process_index + 1]))
