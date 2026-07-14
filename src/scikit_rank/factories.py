"""Factories that turn user-facing hyperparams into composed ``DCNv2`` Modules.

``DCNv2`` itself stays free of construction logic: this module is where string
encoder specs are resolved, default heads are sized, and the
:class:`~scikit_rank.reducers.Concat` default is wired in. Callers (typically
:mod:`scikit_rank.sklearn`) supply scalars and string specs; the factory returns
a ready, parameter-initialised :class:`~scikit_rank.model.DCNv2`.
"""

from __future__ import annotations
import copy
import math
from typing import TYPE_CHECKING, Literal

import torch
from einops.layers.torch import EinMix

from scikit_rank.modules.dcn import (
    CategoricalEmbeddings,
    CoralLayer,
    CrossLayer,
    CrossNetwork,
    DCNv2,
    DeepNetwork,
    EmbeddingTower,
    LinearNumericEncoder,
    MultiHashEmbeddings,
    NumericEncoder,
    ParallelCrossDeep,
    PiecewiseLinearEncoder,
    PLREncoder,
    StackedCrossDeep,
    UnifiedEmbeddings,
)
from scikit_rank.modules.reducers import Concat
from scikit_rank.train.optimizers import LRSchedulerConfig
from scikit_rank.utils import ModuleParserSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

_PLE_DEFAULTS = {
    "embedding_dim": None,
    "activation": False,
    "feature_dropout": 0.0,
}

_PLR_DEFAULTS = {
    "n_freq": 32,
    "sigma": 1.0,
    "embedding_dim": 16,
    "activation": "relu",
    "feature_dropout": 0.0,
}

_LINEAR_DEFAULTS = {
    "embedding_dim": 16,
}

_UNIFIED_EMB_DEFAULTS = {
    "embedding_dim": 16,
}

_NUM_REGISTRY: dict[str, tuple[type[torch.nn.Module], dict]] = {
    "identity": (NumericEncoder, {}),
    "ple": (PiecewiseLinearEncoder, _PLE_DEFAULTS),
    "plr": (PLREncoder, _PLR_DEFAULTS),
    "linear": (LinearNumericEncoder, _LINEAR_DEFAULTS),
}

_CAT_REGISTRY: dict[str, tuple[type[torch.nn.Module], dict]] = {
    "per_feature": (CategoricalEmbeddings, {}),
    "unified": (UnifiedEmbeddings, _UNIFIED_EMB_DEFAULTS),
}

_MULTIHASH_DEFAULTS = {
    "cardinality": 100_000,
    "n_hashes": 2,
    "embedding_dim": 16,
}

_MULTIHASH_REGISTRY: dict[str, tuple[type[torch.nn.Module], dict]] = {
    "multihash": (MultiHashEmbeddings, _MULTIHASH_DEFAULTS),
}

_EMBEDDING_TOWER_DEFAULTS = {
    "output_dim": 64,
    "dropout": 0.1,
    "normalize": True,
}

_EMBEDDING_REGISTRY: dict[str, tuple[type[torch.nn.Module], dict]] = {
    "tower": (EmbeddingTower, _EMBEDDING_TOWER_DEFAULTS),
}

_SCHED_REGISTRY: dict[str, tuple[type[LRSchedulerConfig], dict]] = {
    "plateau": (
        LRSchedulerConfig,
        {"factor": 0.1, "patience": 0, "min_lr": 1e-6, "threshold": 1e-6},
    ),
}


def _require_output_dim(module: torch.nn.Module, role: str) -> None:
    if not hasattr(module, "output_dim"):
        raise TypeError(f"{role} Module must expose an integer `output_dim` attribute")


def build_numeric_encoder(
    spec: str | torch.nn.Module,
    *,
    n_features: int,
    bins: list[torch.Tensor] | None = None,
) -> torch.nn.Module:
    """Build a numeric encoder from a spec string or pass through a Module.

    ``bins`` is required when ``spec`` resolves to the data-driven
    ``PiecewiseLinearEncoder`` (``num_encoder='ple'``); it is ignored
    otherwise. Use
    :meth:`scikit_rank.preprocessing.TabularPreprocessor.fit_ple_bins` to obtain
    bin edges from training data.
    """
    if isinstance(spec, torch.nn.Module):
        _require_output_dim(spec, "numeric encoder")
        return copy.deepcopy(spec)
    parsed = ModuleParserSpec(spec, allowed=_NUM_REGISTRY)
    name = parsed.module_name()
    cls, defaults = _NUM_REGISTRY[name]
    kwargs = {k: parsed.get(k, default=v) for k, v in defaults.items()}
    if name == "ple":
        if bins is None:
            raise ValueError(
                "num_encoder='ple' requires `bins`; pass them via"
                " build_numeric_encoder(..., bins=...) or rely on the"
                " sklearn estimator's ple_n_bins parameter.",
            )
        return cls(bins=bins, **kwargs)
    return cls(n_features=n_features, **kwargs)


def build_categorical_encoder(
    spec: str | torch.nn.Module,
    *,
    cardinalities: Sequence[int],
    embedding_dims: Sequence[int] | None = None,
) -> torch.nn.Module | None:
    """Build a categorical encoder; returns ``None`` if no categorical features."""
    if not cardinalities:
        return None
    if isinstance(spec, torch.nn.Module):
        _require_output_dim(spec, "categorical encoder")
        return copy.deepcopy(spec)
    parsed = ModuleParserSpec(spec, allowed=_CAT_REGISTRY)
    name = parsed.module_name()
    if name == "per_feature":
        if embedding_dims is None:
            raise ValueError("cat_encoder='per_feature' requires embedding_dims")
        return CategoricalEmbeddings(list(cardinalities), list(embedding_dims))
    # unified
    return UnifiedEmbeddings(
        list(cardinalities),
        **{k: parsed.get(k, default=v) for k, v in _UNIFIED_EMB_DEFAULTS.items()},
    )


def build_multihash_encoder(
    spec: str | torch.nn.Module,
    *,
    n_inputs: int,
) -> torch.nn.Module | None:
    """Build the shared-table multihash encoder, or ``None`` when there is none.

    ``cardinality`` and ``embedding_dim`` are read from the spec. ``n_hashes``
    may also live in the same spec for the estimator/preprocessor, while
    ``n_inputs`` stays data-derived (``n_features * n_hashes``) and is passed by
    the caller. Returns ``None`` when ``n_inputs == 0`` so a default spec with no
    multihash features wires nothing in.
    """
    if n_inputs <= 0:
        return None
    if isinstance(spec, torch.nn.Module):
        _require_output_dim(spec, "multihash encoder")
        return copy.deepcopy(spec)
    parsed = ModuleParserSpec(spec, allowed=_MULTIHASH_REGISTRY)
    cls, _defaults = _MULTIHASH_REGISTRY[parsed.module_name()]
    config = multihash_encoder_config(parsed)
    return cls(
        cardinality=config["cardinality"],
        n_inputs=n_inputs,
        embedding_dim=config["embedding_dim"],
    )


def multihash_encoder_config(spec: str | torch.nn.Module | ModuleParserSpec) -> dict[str, int]:
    """Resolve hashing/model config carried by ``multihash_encoder`` spec."""
    if isinstance(spec, torch.nn.Module):
        config = dict(_MULTIHASH_DEFAULTS)
    else:
        parsed = (
            spec
            if isinstance(spec, ModuleParserSpec)
            else ModuleParserSpec(spec, allowed=_MULTIHASH_REGISTRY)
        )
        _, defaults = _MULTIHASH_REGISTRY[parsed.module_name()]
        config = {k: parsed.get(k, default=v) for k, v in defaults.items()}
    out = {
        "cardinality": int(config["cardinality"]),
        "n_hashes": int(config["n_hashes"]),
        "embedding_dim": int(config["embedding_dim"]),
    }
    for key, value in out.items():
        if value <= 0:
            raise ValueError(f"multihash_encoder {key} must be positive, got {value}")
    return out


def build_embedding_encoder(
    spec: str | torch.nn.Module,
    *,
    input_dim: int,
) -> torch.nn.Module:
    """Build a dense external-embedding projection tower from a spec or Module.

    ``input_dim`` (the incoming vector width) is data-derived and passed as a
    scalar, mirroring ``n_features`` for the numeric encoder; the spec carries
    the projection ``output_dim`` / ``dropout`` / ``normalize`` tunables.
    """
    if isinstance(spec, torch.nn.Module):
        _require_output_dim(spec, "embedding encoder")
        return copy.deepcopy(spec)
    parsed = ModuleParserSpec(spec, allowed=_EMBEDDING_REGISTRY)
    cls, defaults = _EMBEDDING_REGISTRY[parsed.module_name()]
    kwargs = {k: parsed.get(k, default=v) for k, v in defaults.items()}
    return cls(input_dim=input_dim, **kwargs)


def _build_reference_layers(
    *,
    multihash_encoder: str | torch.nn.Module,
    multihash_n_inputs: int | None,
    embedding_encoders: dict[str, str | torch.nn.Module] | None,
    embedding_input_dims: dict[str, int] | None,
) -> dict[str, torch.nn.Module]:
    """Build the multihash + dense external-embedding ("reference") input streams.

    ``multihash_encoder`` carries cardinality/n_hashes/embedding_dim in one
    spec. The model only needs the parsed cardinality/embedding_dim plus the
    data-derived ``multihash_n_inputs`` supplied by the preprocessor.
    ``embedding_encoders`` maps one spec per dense stream, with its incoming
    vector width supplied in ``embedding_input_dims``. Returns the streams keyed
    by model input-layer name.
    """
    layers: dict[str, torch.nn.Module] = {}
    if multihash_n_inputs is not None:
        multihash = build_multihash_encoder(
            multihash_encoder,
            n_inputs=multihash_n_inputs,
        )
        if multihash is not None:
            layers["multihash"] = multihash
    if embedding_encoders:
        dims = embedding_input_dims or {}
        for name, embedding_spec in embedding_encoders.items():
            if name in layers:
                raise ValueError(f"embedding_encoders duplicate built-in stream: {name!r}")
            if name not in dims:
                raise ValueError(
                    f"embedding_input_dims is missing an input_dim for {name!r}",
                )
            layers[name] = build_embedding_encoder(embedding_spec, input_dim=dims[name])
    return layers


def build_dcnv2(
    *,
    n_num_features: int,
    cardinalities: Sequence[int],
    embedding_dims: Sequence[int] | None = None,
    hidden_units: Sequence[int] = (256, 128),
    cross_layers: int = 3,
    cross_rank: int | None = None,
    structure: Literal["stacked", "parallel"] = "stacked",
    num_encoder: str | torch.nn.Module = "identity",
    cat_encoder: str | torch.nn.Module = "per_feature",
    reducer: torch.nn.Module | None = None,
    n_outputs: int = 1,
    dropout: float = 0.0,
    activation: str = "relu",
    batch_norm: bool = False,
    gated_cross: bool = False,
    cross_type: str = "standard",
    mask_ratio: float = 0.5,
    use_moe: bool = False,
    num_experts: int = 4,
    moe_top_k: int = 2,
    use_inner_cross_layers: bool = False,
    use_coral_head: bool = False,
    num_encoder_bins: list[torch.Tensor] | None = None,
    multihash_encoder: str | torch.nn.Module = "multihash",
    multihash_n_inputs: int | None = None,
    embedding_encoders: dict[str, str | torch.nn.Module] | None = None,
    embedding_input_dims: dict[str, int] | None = None,
    extra_layers: dict[str, torch.nn.Module] | None = None,
) -> DCNv2:
    """Compose a :class:`DCNv2` from scalar / string hyperparams.

    The head is built internally against the computed post-cross/deep
    dimension: ``torch.nn.Linear(final_dim, n_outputs)`` by default, or
    :class:`~scikit_rank.model.CoralLayer(final_dim, n_outputs)` when
    ``use_coral_head=True`` (``n_outputs`` is interpreted as ``num_classes``
    in that case).
    """
    layers = {}
    if n_num_features > 0:
        layers["num"] = build_numeric_encoder(
            num_encoder,
            n_features=n_num_features,
            bins=num_encoder_bins,
        )
    if (
        cat := build_categorical_encoder(
            cat_encoder,
            cardinalities=cardinalities,
            embedding_dims=embedding_dims,
        )
    ) is not None:
        layers["cat"] = cat
    reference_layers = _build_reference_layers(
        multihash_encoder=multihash_encoder,
        multihash_n_inputs=multihash_n_inputs,
        embedding_encoders=embedding_encoders,
        embedding_input_dims=embedding_input_dims,
    )
    overlap = set(layers).intersection(reference_layers)
    if overlap:
        raise ValueError(f"reference streams duplicate built-in streams: {sorted(overlap)}")
    layers.update(reference_layers)
    if extra_layers:
        overlap = set(layers).intersection(extra_layers)
        if overlap:
            raise ValueError(f"extra_layers duplicate built-in streams: {sorted(overlap)}")
        for name, module in extra_layers.items():
            _require_output_dim(module, f"extra layer {name!r}")
            layers[name] = copy.deepcopy(module)
    if not layers:
        raise ValueError(
            "DCNv2 needs at least one input stream.",
        )

    reducer = reducer if reducer is not None else Concat(dim=-1)
    if not hasattr(reducer, "compute_output_dim"):
        raise TypeError("reducer must expose compute_output_dim(input_dims) -> int")

    input_dims = {name: m.output_dim() for name, m in layers.items()}
    rep_dim = reducer.compute_output_dim(input_dims)

    cross_network = CrossNetwork(
        rep_dim,
        cross_layers,
        rank=cross_rank,
        gated=gated_cross,
        cross_type=cross_type,
        mask_ratio=mask_ratio,
    )

    if structure not in ("stacked", "parallel"):
        raise ValueError(f"structure must be 'stacked' or 'parallel', got {structure!r}")

    if use_inner_cross_layers:
        if structure != "stacked":
            raise ValueError("use_inner_cross_layers requires structure='stacked'")
        if not hidden_units or len(hidden_units) < 2:
            raise ValueError(
                "use_inner_cross_layers requires hidden_units with len >= 2",
            )
        adapter_out_dim = hidden_units[0]
        inner_total = sum(cross_network.inner_dims())
        adapter = torch.nn.Linear(rep_dim, adapter_out_dim, bias=False)
        deep_network = DeepNetwork(
            adapter_out_dim + inner_total,
            list(hidden_units[1:]),
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
            use_moe=use_moe,
            num_experts=num_experts,
            moe_top_k=moe_top_k,
        )
        body = StackedCrossDeep(
            cross_network,
            deep_network,
            use_inner_cross_layers=True,
            adapter=adapter,
        )
        final_dim = deep_network.output_dim()
    else:
        deep_network = (
            DeepNetwork(
                rep_dim,
                list(hidden_units),
                dropout=dropout,
                activation=activation,
                batch_norm=batch_norm,
                use_moe=use_moe,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
            )
            if hidden_units
            else None
        )
        deep_out = deep_network.output_dim() if deep_network is not None else rep_dim
        if structure == "stacked":
            body = StackedCrossDeep(cross_network, deep_network)
            final_dim = deep_out
        else:
            body = ParallelCrossDeep(cross_network, deep_network)
            final_dim = rep_dim + deep_out

    head = (
        CoralLayer(in_features=final_dim, num_classes=n_outputs)
        if use_coral_head
        else torch.nn.Linear(final_dim, n_outputs)
    )

    model = DCNv2(layers=layers, reducer=reducer, body=body, head=head)
    _init_weights(model)
    return model


def _xavier_normal_per_feature_(weight: torch.Tensor, gain: float = 1.0) -> None:
    """Xavier-normal init for a packed per-feature ``EinMix`` weight.

    The weight of an ``EinMix("b n i -> b n o", weight_shape="n i o")`` is a 3-D
    tensor ``(n_features, i, o)`` holding one independent ``i -> o`` linear map per
    feature. ``torch.nn.init.xavier_normal_`` would compute fan over the *packed*
    shape (folding ``n_features`` into ``fan_out``), which under-scales the init and
    makes it depend on ``n_features``. Instead, init each slice like an independent
    ``nn.Linear(i, o)`` -- ``std = gain * sqrt(2 / (i + o))`` -- so the scale is
    ``n_features``-independent. The last two dims are the linear map ``(i, o)``;
    ``sqrt(2 / (fan_in + fan_out))`` is symmetric in the two, so the ordering does
    not matter (also correct for a 2-D weight).
    """
    fan_in, fan_out = weight.shape[-2], weight.shape[-1]
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    with torch.no_grad():
        weight.normal_(0.0, std)


def _init_weights(model: torch.nn.Module) -> None:
    """Re-initialize model weights in place (post-composition).

    Embeddings ~ ``N(0, 1e-4)``; ``Linear`` weights via Xavier-normal with zero
    bias; per-feature ``EinMix`` numeric encoders (``LinearNumericEncoder``,
    ``ple``, ``plr``) via :func:`_xavier_normal_per_feature_` -- each ``(i -> o)``
    slice is initialized like an independent ``nn.Linear(i, o)`` so its std is
    ``n_features``-independent -- with zero bias; the cross-layer additive bias is
    zeroed (overriding :class:`CrossLayer`'s uniform default).
    """
    for mod in model.modules():
        if isinstance(mod, torch.nn.Embedding):
            torch.nn.init.normal_(mod.weight, mean=0.0, std=1e-4)
        elif isinstance(mod, EinMix):
            if getattr(mod, "weight", None) is not None:
                _xavier_normal_per_feature_(mod.weight)
            if getattr(mod, "bias", None) is not None:
                torch.nn.init.zeros_(mod.bias)
        elif isinstance(mod, torch.nn.Linear):
            if getattr(mod, "weight", None) is not None:
                torch.nn.init.xavier_normal_(mod.weight)
            if getattr(mod, "bias", None) is not None:
                torch.nn.init.zeros_(mod.bias)
        elif isinstance(mod, CrossLayer):
            torch.nn.init.zeros_(mod._bias)  # noqa: SLF001


def build_lr_scheduler_config(spec: str | None) -> LRSchedulerConfig | None:
    """Build an :class:`~scikit_rank.train.optimizers.LRSchedulerConfig` from a spec string.

    Mirrors the encoder builders (parse ``name[:k=v;...]`` against a registry), but
    returns a *config* rather than a Module: the scheduler needs the optimizer,
    which ``TrainingRun`` owns, so it is constructed there via
    :func:`~scikit_rank.train.optimizers.build_lr_scheduler`. ``None`` -> no scheduler.
    """
    if spec is None:
        return None
    parsed = ModuleParserSpec(spec, allowed=_SCHED_REGISTRY)
    name = parsed.module_name()
    cls, defaults = _SCHED_REGISTRY[name]
    kwargs = {k: parsed.get(k, default=v) for k, v in defaults.items()}
    return cls(scheduler_type=name, **kwargs)
