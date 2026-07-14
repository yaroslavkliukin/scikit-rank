"""Ignite/accelerate training loop with a ``model(batch) -> dict`` contract.

Works across CPU, single-GPU (cuda/mps) and DDP.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

import torch
from ignite.engine import Engine, EventEnum, Events, State

if TYPE_CHECKING:
    from collections.abc import Callable

    from accelerate import Accelerator
    from torch.utils.data import DataLoader


class ModelEvents(EventEnum):
    FORWARD_STARTED = "forward_started"
    FORWARD_COMPLETED = "forward_completed"
    BACKWARD_STARTED = "backward_started"
    BACKWARD_COMPLETED = "backward_completed"
    OPTIMIZER_STARTED = "optimizer_started"
    OPTIMIZER_COMPLETED = "optimizer_completed"


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        accelerator: Accelerator,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.engines = {
            "train": Engine(self._train_step),
            "eval": Engine(self._eval_step),
        }
        self._accelerator = accelerator
        self._add_events()
        for key, e in self.engines.items():
            e.state.name = key
            e.state.epoch_iteration = 0
            e.state_dict_user_keys.append("name")
            e.state_dict_user_keys.append("forward_iteration")
            e.state_dict_user_keys.append("backward_iteration")
            e.state_dict_user_keys.append("optimizer_iteration")
            e.state_dict_user_keys.append("epoch_iteration")

    def add_event(
        self,
        engine: str,
        event_name: Any,
        handler: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.engines[engine].add_event_handler(event_name, handler, *args, **kwargs)

    def run(
        self,
        loaders: dict[str, DataLoader],
        max_iters: dict[str, int] | None = None,
        epochs: int | None = None,
    ) -> State:
        self._loaders = loaders
        self._max_iters = max_iters or {}
        self.engines["train"].run(
            self._loaders["train"],
            epoch_length=self._max_iters.get("train"),
            max_epochs=epochs,
        )
        return self.engines["eval"].state if "eval" in loaders else self.engines["train"].state

    def _train_step(
        self,
        engine: Engine,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        self.model.train()
        self.set_optimizer_mode(train=True)
        with self._accelerator.accumulate(self.model):
            state = engine.state
            state.forward_iteration += 1
            engine.fire_event(ModelEvents.FORWARD_STARTED)
            with self._accelerator.autocast():
                output = state.output = self.model(batch)
            engine.fire_event(ModelEvents.FORWARD_COMPLETED)
            if "loss" not in output:
                return output
            state.backward_iteration += 1
            engine.fire_event(ModelEvents.BACKWARD_STARTED)
            self._accelerator.backward(output["loss"])
            engine.fire_event(ModelEvents.BACKWARD_COMPLETED)
            state.optimizer_iteration += 1
            engine.fire_event(ModelEvents.OPTIMIZER_STARTED)
            self.optimizer.step()
            engine.fire_event(ModelEvents.OPTIMIZER_COMPLETED)
            self.optimizer.zero_grad()
            state.metrics["_loss"] += output["loss"].detach()
            return output

    def _eval_step(
        self,
        engine: Engine,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        self.model.eval()
        self.set_optimizer_mode(train=False)
        with torch.no_grad():
            state = engine.state
            state.forward_iteration += 1
            engine.fire_event(ModelEvents.FORWARD_STARTED)
            with self._accelerator.autocast():
                output = state.output = self.model(batch)
            engine.fire_event(ModelEvents.FORWARD_COMPLETED)
            if "loss" in output:
                state.metrics["_loss"] += output["loss"].detach()
            return output

    def set_optimizer_mode(self, *, train: bool) -> None:
        # Schedule-free optimizers (e.g. ``schedulefree.AdamWScheduleFree``) keep
        # separate train / eval weights and require an explicit mode toggle
        # around steps and evaluation.
        mode = getattr(self.optimizer, "train" if train else "eval", None)
        if callable(mode):
            mode()

    def _add_events(self) -> None:
        for e in self.engines.values():
            e.register_events(
                *ModelEvents,
                event_to_attr={
                    ModelEvents.FORWARD_STARTED: "forward_iteration",
                    ModelEvents.FORWARD_COMPLETED: "forward_iteration",
                    ModelEvents.BACKWARD_STARTED: "backward_iteration",
                    ModelEvents.BACKWARD_COMPLETED: "backward_iteration",
                    ModelEvents.OPTIMIZER_STARTED: "optimizer_iteration",
                    ModelEvents.OPTIMIZER_COMPLETED: "optimizer_iteration",
                },
            )
        events = (
            (Events.EPOCH_STARTED, self._reset_epoch),
            (Events.ITERATION_COMPLETED, self._update_iteration),
            (Events.ITERATION_COMPLETED, self._update_loss),
        )
        for e in self.engines:
            for args in events:
                self.add_event(e, *args)
        self.add_event("train", Events.EPOCH_COMPLETED, self._run_eval)
        self.add_event("eval", Events.COMPLETED, self._finalize_eval_loss)

    def _run_eval(self) -> None:
        eval_loader = self._loaders.get("eval")
        if eval_loader is None:
            return
        self.engines["eval"].run(eval_loader, epoch_length=self._max_iters.get("eval"))

    def _reset_epoch(self, engine: Engine) -> None:
        state = engine.state
        state.metrics["_loss"] = torch.tensor(0.0, device=self._accelerator.device)
        state.epoch_iteration = 0

    def _update_iteration(self, engine: Engine) -> None:
        engine.state.epoch_iteration += 1

    def _update_loss(self, engine: Engine) -> None:
        state = engine.state
        state.metrics["loss"] = state.metrics["_loss"] / state.epoch_iteration

    def _finalize_eval_loss(self, engine: Engine) -> None:
        if self._accelerator.num_processes == 1:
            return
        state = engine.state
        loss_sum = state.metrics.get("_loss")
        if loss_sum is None:
            return
        total_sum = self._accelerator.reduce(loss_sum.detach().clone(), reduction="sum")
        total_count = self._accelerator.reduce(
            torch.tensor(float(state.epoch_iteration), device=self._accelerator.device),
            reduction="sum",
        )
        if float(total_count) > 0:
            state.metrics["loss"] = total_sum / total_count
