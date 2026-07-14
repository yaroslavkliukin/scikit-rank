"""Optimizer and LR-scheduler construction from string-named configs.

:func:`build_optimizer` dispatches on ``optimizer_type`` via
:data:`_OPTIMIZER_BUILDERS`. Biases, norm-layer and numeric-encoder parameters
are excluded from weight decay; Muon is applied only to the 2-D weight matrices
of the deep tower, with an auxiliary AdamW for everything else. Module detection
is structural (``isinstance``), so it survives arbitrary nesting/wrapping.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from muon import SingleDeviceMuonWithAuxAdam
from schedulefree import AdamWScheduleFree

from scikit_rank.modules.dcn import (
    DeepNetwork,
    NumericEncoder,
    PiecewiseLinearEncoder,
    PLREncoder,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.optim.optimizer import ParamsT

_NUMERIC_ENCODERS = (NumericEncoder, PiecewiseLinearEncoder, PLREncoder)


@dataclass(frozen=True)
class OptimizerConfig:
    """All hyperparameters needed to build an optimizer via :func:`build_optimizer`.

    A single, fully-typed config object shared by every builder. ``lr`` and
    ``weight_decay`` apply to all optimizers; the remaining fields are only
    consulted by the optimizers that use them (e.g. ``muon_lr`` for muon,
    ``ademamix_*`` for ademamix).
    """

    optimizer_type: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 3e-1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    muon_lr: float = 0.02
    amsgrad: bool = False
    ademamix_beta3: float = 0.9999
    ademamix_alpha: float = 5.0
    lion_lr_scale: float = 0.33


@dataclass(frozen=True)
class LRSchedulerConfig:
    """Hyperparameters to build an LR scheduler via :func:`build_lr_scheduler`.

    Mirrors :class:`OptimizerConfig`. Only ``ReduceLROnPlateau``
    (``scheduler_type="plateau"``) is supported today; its defaults use factor
    0.1, decay on the first non-improving epoch, a 1e-6 floor, and an absolute
    1e-6 improvement margin.
    """

    scheduler_type: str = "plateau"
    factor: float = 0.1
    patience: int = 0
    threshold: float = 1e-6
    min_lr: float = 1e-6


def _iter_zero_wd_params(model: torch.nn.Module) -> set[torch.nn.Parameter]:
    """Collect parameters that should not be weight-decayed.

    Biases, normalization-layer parameters and numeric-encoder parameters are
    excluded from weight decay.
    """
    zero_wd: set[torch.nn.Parameter] = set()
    for module in model.modules():
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm1d, *_NUMERIC_ENCODERS)):
            for p in module.parameters():
                zero_wd.add(p)
        for name, p in module.named_parameters(recurse=False):
            if name.endswith("bias"):
                zero_wd.add(p)
    return zero_wd


def _split_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, object]]:
    zero_wd = _iter_zero_wd_params(model)
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if p in zero_wd else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _build_adamw(model: torch.nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        _split_param_groups(model, cfg.weight_decay),
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        amsgrad=cfg.amsgrad or cfg.optimizer_type == "adamw_amsgrad",
    )


def _build_schedulefree_adamw(
    model: torch.nn.Module,
    cfg: OptimizerConfig,
) -> torch.optim.Optimizer:
    return AdamWScheduleFree(
        _split_param_groups(model, cfg.weight_decay),
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        warmup_steps=0,
    )


def _build_ademamix(model: torch.nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    return AdEMAMix(
        _split_param_groups(model, cfg.weight_decay),
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2, cfg.ademamix_beta3),
        alpha=cfg.ademamix_alpha,
        eps=cfg.adam_eps,
    )


def _build_lion(model: torch.nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    return Lion(
        _split_param_groups(model, cfg.weight_decay),
        lr=cfg.lr * cfg.lion_lr_scale,
        betas=(0.9, 0.99),
    )


def _build_cautious_adamw(model: torch.nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    return CautiousAdamW(
        _split_param_groups(model, cfg.weight_decay),
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )


def _iter_muon_params(model: torch.nn.Module) -> set[torch.nn.Parameter]:
    """The 2-D weight matrices of the deep tower(s), which Muon optimizes.

    Muon targets matrix-shaped parameters; vectors (norms, biases, 1-D weights)
    and non-deep-network parameters are left to the auxiliary AdamW.
    """  # noqa: D401
    muon_params: set[torch.nn.Parameter] = set()
    for deep in model.modules():
        if not isinstance(deep, DeepNetwork):
            continue
        for module in deep.modules():
            if isinstance(module, torch.nn.Linear) and 1 not in module.weight.shape[-2:]:
                muon_params.add(module.weight)
    return muon_params


class _MuonWithAuxAdam(SingleDeviceMuonWithAuxAdam):
    """``SingleDeviceMuonWithAuxAdam`` whose ``step`` accepts a ``closure``.

    Muon's ``step()`` takes no arguments, but :class:`accelerate.optimizer.
    AcceleratedOptimizer` always forwards a (possibly ``None``) closure. This
    thin override restores the standard :class:`torch.optim.Optimizer.step`
    signature so muon works through Accelerate.
    """

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = closure() if closure is not None else None
        super().step()
        return loss


def _build_muon(model: torch.nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    muon_params = _iter_muon_params(model)
    zero_wd = _iter_zero_wd_params(model)
    aux: dict[str, object] = {
        "betas": (cfg.adam_beta1, cfg.adam_beta2),
        "eps": cfg.adam_eps,
        "use_muon": False,
    }
    return _MuonWithAuxAdam(
        [
            {
                "params": list(muon_params),
                "lr": cfg.muon_lr,
                "use_muon": True,
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for p in zero_wd if p not in muon_params],
                "lr": cfg.lr,
                "weight_decay": 0.0,
                **aux,
            },
            {
                "params": [
                    p for p in model.parameters() if p not in muon_params and p not in zero_wd
                ],
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                **aux,
            },
        ],
    )


# Registry: optimizer_type -> builder(model, cfg) -> torch.optim.Optimizer.
_OPTIMIZER_BUILDERS: dict[
    str,
    Callable[[torch.nn.Module, OptimizerConfig], torch.optim.Optimizer],
] = {
    "adamw": _build_adamw,
    "adamw_amsgrad": _build_adamw,
    "muon": _build_muon,
    "schedulefree_adamw": _build_schedulefree_adamw,
    "ademamix": _build_ademamix,
    "lion": _build_lion,
    "cautious_adamw": _build_cautious_adamw,
}

OPTIMIZER_TYPES = tuple(_OPTIMIZER_BUILDERS)


def build_optimizer(
    model: torch.nn.Module,
    config: OptimizerConfig | None = None,
) -> torch.optim.Optimizer:
    """Build an optimizer for ``model`` from an :class:`OptimizerConfig`.

    ``model`` may be the :class:`~scikit_rank.model.DCNv2` itself or any module that
    contains it (e.g. :class:`~scikit_rank.run.TrainingModule`); the decay split and
    muon detection walk the full module tree.
    """
    config = config or OptimizerConfig()
    try:
        builder = _OPTIMIZER_BUILDERS[config.optimizer_type]
    except KeyError:
        raise ValueError(
            f"unknown optimizer_type={config.optimizer_type!r}, expected one of {OPTIMIZER_TYPES}",
        ) from None
    return builder(model, config)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: LRSchedulerConfig,
    mode: str,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Build an LR scheduler for ``optimizer`` from an :class:`LRSchedulerConfig`.

    Mirrors :func:`build_optimizer`. ``mode`` is the eval-metric monitor
    direction (AUC -> "max", loss -> "min"); the scheduler steps on the same
    monitored value as best-checkpoint selection / early stopping.
    """
    if config.scheduler_type != "plateau":
        raise ValueError(
            f"unknown scheduler_type={config.scheduler_type!r}, expected 'plateau'",
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=config.factor,
        patience=config.patience,
        threshold=config.threshold,
        threshold_mode="abs",
        min_lr=config.min_lr,
    )


class AdEMAMix(torch.optim.Optimizer):
    """AdEMAMix (Pagliardini et al., 2024, arXiv:2409.03137).

    Two EMAs of the gradient: fast m1 (β1) and slow m2 (β3).
    update = (m1 / (1 - β1ᵗ) + a · m2) / (sqrt(v) + eps).
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be > 0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "alpha": alpha,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2, b3 = group["betas"]
            alpha = group["alpha"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(p)
                    state["m2"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                m1, m2, v = state["m1"], state["m2"], state["v"]
                t = state["step"]
                m1.mul_(b1).add_(g, alpha=1 - b1)
                m2.mul_(b3).add_(g, alpha=1 - b3)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                bias_c1 = 1 - b1**t
                bias_c2 = 1 - b2**t
                denom = (v / bias_c2).sqrt().add_(eps)
                update = (m1 / bias_c1 + alpha * m2) / denom
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


class Lion(torch.optim.Optimizer):
    """Lion (Chen et al., 2023). Sign-based momentum.

    update = sign(β1 · m + (1-β1) · g); m ← β2 · m + (1-β2) · g.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                update = (b1 * m + (1 - b1) * g).sign_()
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
                m.mul_(b2).add_(g, alpha=1 - b2)
        return loss


class CautiousAdamW(torch.optim.Optimizer):
    """Cautious AdamW (Liang et al., 2024, arXiv:2411.16085).

    Masks updates where the step direction disagrees with the gradient sign:
        mask = (update · g > 0).float()
        update_cautious = mask · update / max(mean(mask), eps)
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                m, v = state["m"], state["v"]
                t = state["step"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                bc1 = 1 - b1**t
                bc2 = 1 - b2**t
                m_hat = m / bc1
                v_hat = v / bc2
                update = m_hat / (v_hat.sqrt() + eps)
                mask = (update * g > 0).float()
                mask_scale = mask.mean().clamp(min=eps)
                update = update * mask / mask_scale
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss
