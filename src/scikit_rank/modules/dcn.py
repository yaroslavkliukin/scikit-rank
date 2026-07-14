"""DCNv2 model components.

Includes PLE/PLR-style numeric encoders, gated and ML-DCN cross layers,
GLU/GeGLU/SwiGLU FFNs, an MoE first deep layer, and a CORAL output head.
"""

import math
from collections.abc import Callable, Iterator, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import EinMix


class _PerFeatureSwiGLU(torch.nn.Module):
    """Per-feature SwiGLU block for :class:`PLREncoder`."""

    def __init__(
        self,
        n: int,
        in_dim: int,
        out_dim: int,
        with_down: bool = False,
    ) -> None:
        super().__init__()
        hidden = int(out_dim * 4 / 3) if with_down else out_dim
        self._w = EinMix(
            "b n i -> b n o",
            weight_shape="n i o",
            bias_shape="n o",
            n=n,
            i=in_dim,
            o=hidden,
        )
        self._v = EinMix(
            "b n i -> b n o",
            weight_shape="n i o",
            bias_shape="n o",
            n=n,
            i=in_dim,
            o=hidden,
        )
        self._down = (
            EinMix(
                "b n i -> b n o",
                weight_shape="n i o",
                bias_shape="n o",
                n=n,
                i=hidden,
                o=out_dim,
            )
            if with_down
            else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._down(self._w(x) * F.silu(self._v(x)))


class NumericEncoder(torch.nn.Module):
    """Identity numeric encoder."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self._output_dim = n_features

    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearNumericEncoder(torch.nn.Module):
    """Per-feature learnable linear numeric embedding.

    Each scalar feature ``x_f`` is projected to a vector by a feature-specific
    ``Linear(1, embedding_dim, bias=False)`` (``x_f -> x_f * w_f``). The
    per-feature weights are packed into a single
    :class:`~einops.layers.torch.EinMix`; the output is flattened to
    ``n_features * embedding_dim`` to match the other numeric encoders.

    This encoder is sensitive to input scale; use ``normalize_numeric=False``
    only when the numeric feature scale is already appropriate.
    """

    def __init__(self, n_features: int, embedding_dim: int) -> None:
        super().__init__()
        self._linear = EinMix(
            "b n i -> b n o",
            weight_shape="n i o",
            n=n_features,
            i=1,
            o=embedding_dim,
        )
        self._output_dim = n_features * embedding_dim

    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._linear(x[..., None])  # (B, n_features, embedding_dim)
        return rearrange(h, "b f d -> b (f d)")


class _PiecewiseLinearEncodingImpl(torch.nn.Module):
    """Piecewise-linear encoding kernel from Gorishniy et al. 2022.

    See ``rtdl-num-embeddings`` (``_PiecewiseLinearEncodingImpl``). The output
    layout aligns each feature's *last* bin component at the trailing index so
    a single vectorised clamp can be applied across features that have
    different numbers of bins.

    The weight and bias are non-trainable buffers derived from the bin edges.
    """

    weight: torch.Tensor
    bias: torch.Tensor
    single_bin_mask: torch.Tensor | None
    mask: torch.Tensor | None

    def __init__(self, bins: list[torch.Tensor]) -> None:
        super().__init__()
        if not bins:
            raise ValueError("PLE bins must not be empty")

        n_features = len(bins)
        n_bins = [len(x) - 1 for x in bins]
        max_n_bins = max(n_bins)

        self.register_buffer("weight", torch.zeros(n_features, max_n_bins))
        self.register_buffer("bias", torch.zeros(n_features, max_n_bins))

        single_bin_mask = torch.tensor(n_bins) == 1
        self.register_buffer(
            "single_bin_mask",
            single_bin_mask if bool(single_bin_mask.any()) else None,
        )

        self.register_buffer(
            "mask",
            None
            if all(len(x) == len(bins[0]) for x in bins)
            else torch.row_stack(
                [
                    torch.cat(
                        [
                            torch.ones((len(x) - 1) - 1, dtype=torch.bool),
                            torch.zeros(max_n_bins - (len(x) - 1), dtype=torch.bool),
                            torch.ones(1, dtype=torch.bool),
                        ],
                    )
                    for x in bins
                ],
            ),
        )

        for i, bin_edges in enumerate(bins):
            bin_width = bin_edges.diff()
            w = 1.0 / bin_width
            b = -bin_edges[:-1] / bin_width
            # last bin component is always stored at the trailing index
            self.weight[i, -1] = w[-1]
            self.bias[i, -1] = b[-1]
            # leading n_bins - 1 components stored at the head
            self.weight[i, : n_bins[i] - 1] = w[:-1]
            self.bias[i, : n_bins[i] - 1] = b[:-1]

    def get_max_n_bins(self) -> int:
        return self.weight.size(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.addcmul(self.bias, self.weight, x[..., None])
        if x.size(-1) <= 1:
            return x
        # single-bin features behave like min-max scaling: do not clamp at 0
        last = (
            x[..., -1:].clamp_min(0.0)
            if self.single_bin_mask is None
            else torch.where(
                self.single_bin_mask[..., None],
                x[..., -1:],
                x[..., -1:].clamp_min(0.0),
            )
        )
        return torch.cat(
            [x[..., :1].clamp_max(1.0), x[..., 1:-1].clamp(0.0, 1.0), last],
            dim=-1,
        )


class PiecewiseLinearEncoder(torch.nn.Module):
    """Piecewise-linear encoder for continuous features (Gorishniy et al. 2022).

    Uses data-driven per-feature bin edges (computed by
    :meth:`scikit_rank.preprocessing.TabularPreprocessor.fit_ple_bins` from training
    quantiles). With ``embedding_dim=None`` the encoder emits the raw
    piecewise-linear encoding flattened across features (padded components
    removed when features have different numbers of bins). With
    ``embedding_dim`` set, a per-feature linear (:class:`einops.layers.torch.EinMix`)
    projects each feature's encoding to ``embedding_dim`` and the output is
    flattened to ``(B, n_features * embedding_dim)``.
    """

    def __init__(
        self,
        bins: list[torch.Tensor],
        embedding_dim: int | None = None,
        activation: bool = False,
        feature_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not bins:
            raise ValueError("PiecewiseLinearEncoder requires non-empty bins")
        for i, b in enumerate(bins):
            if b.ndim != 1 or len(b) < 2:
                raise ValueError(
                    f"bins[{i}] must be a 1-D tensor with >=2 edges, got shape {tuple(b.shape)}",
                )
        self._impl = _PiecewiseLinearEncodingImpl(bins)
        self._n_features = len(bins)
        self._feature_dropout = feature_dropout
        self._embedding_dim = embedding_dim
        if embedding_dim is None:
            self._linear: torch.nn.Module | None = None
            self._activation: torch.nn.Module | None = None
            self._output_dim = (
                int(self._impl.weight.numel())
                if self._impl.mask is None
                else int(self._impl.mask.long().sum().item())
            )
        else:
            self._linear = EinMix(
                "b n i -> b n o",
                weight_shape="n i o",
                bias_shape="n o",
                n=self._n_features,
                i=self._impl.get_max_n_bins(),
                o=embedding_dim,
            )
            self._activation = torch.nn.ReLU() if activation else None
            self._output_dim = self._n_features * embedding_dim

    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._impl(x)
        if self._linear is None:
            mask = self._impl.mask
            return encoded.flatten(-2) if mask is None else encoded[:, mask]
        h = self._linear(encoded)
        if self._activation is not None:
            h = self._activation(h)
        if self.training and self._feature_dropout > 0:
            keep = torch.bernoulli(
                torch.full(
                    (h.size(0), h.size(1), 1),
                    1 - self._feature_dropout,
                    device=h.device,
                ),
            )
            h = h * keep / (1 - self._feature_dropout)
        return rearrange(h, "b f d -> b (f d)")


# Registry for PLREncoder activation blocks.
# Each factory receives (n_features, in_dim, embedding_dim) and returns a Module.
_PLR_ACT_BLOCKS: dict[str, Callable[[int, int, int], torch.nn.Module]] = {
    "relu": lambda n, d, e: torch.nn.Sequential(
        EinMix("b n i -> b n o", weight_shape="n i o", bias_shape="n o", n=n, i=d, o=e),
        torch.nn.ReLU(),
    ),
    "silu": lambda n, d, e: torch.nn.Sequential(
        EinMix("b n i -> b n o", weight_shape="n i o", bias_shape="n o", n=n, i=d, o=e),
        torch.nn.SiLU(),
    ),
    "swiglu": lambda n, d, e: _PerFeatureSwiGLU(n, d, e, with_down=False),
    "swiglu_x": lambda n, d, e: _PerFeatureSwiGLU(n, d, e, with_down=True),
}


class PLREncoder(torch.nn.Module):
    """Periodic + Linear + Activation encoding (Gorishniy et al. 2022)."""

    def __init__(
        self,
        n_features: int,
        n_freq: int = 32,
        sigma: float = 1.0,
        embedding_dim: int = 16,
        activation: str = "relu",
        feature_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if activation not in _PLR_ACT_BLOCKS:
            raise ValueError(
                f"activation must be one of {list(_PLR_ACT_BLOCKS)}, got {activation!r}",
            )
        self._feature_dropout = feature_dropout
        self._coeffs = torch.nn.Parameter(torch.empty(n_features, n_freq))
        torch.nn.init.normal_(self._coeffs, mean=0.0, std=sigma)
        self._act_block = _PLR_ACT_BLOCKS[activation](n_features, 2 * n_freq, embedding_dim)
        self._output_dim = n_features * embedding_dim

    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = 2 * math.pi * torch.einsum("f k, b f -> b f k", self._coeffs, x)
        h = torch.cat([torch.cos(h), torch.sin(h)], dim=-1)
        h = self._act_block(h)
        if self.training and self._feature_dropout > 0:
            mask = torch.bernoulli(
                torch.full((h.size(0), h.size(1), 1), 1 - self._feature_dropout, device=h.device),
            )
            h = h * mask / (1 - self._feature_dropout)
        return rearrange(h, "b f d -> b (f d)")


class CategoricalEmbeddings(torch.nn.Module):
    """One embedding table per categorical feature (index 0 = unknown/missing)."""

    def __init__(self, cardinalities: list[int], embedding_dims: list[int]) -> None:
        super().__init__()
        self._embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(card, dim)
                for card, dim in zip(cardinalities, embedding_dims, strict=True)
            ],
        )
        self._output_dim = sum(embedding_dims)

    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.cat([emb(ids[:, i]) for i, emb in enumerate(self._embeddings)], dim=-1)


class UnifiedEmbeddings(torch.nn.Module):
    """Single embedding table shared by categorical features.

    Each feature's ids are offset into one large shared table, so every feature
    keeps its own id space while sharing one embedding matrix and dimension.
    """

    def __init__(self, cardinalities: list[int], embedding_dim: int) -> None:
        super().__init__()
        offsets = torch.tensor([0, *cardinalities[:-1]], dtype=torch.long).cumsum(0)
        self.register_buffer("_offsets", offsets)
        self._embeddings = torch.nn.Embedding(
            num_embeddings=sum(cardinalities),
            embedding_dim=embedding_dim,
        )
        self._output_dim = len(cardinalities) * embedding_dim

    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        embeddings = self._embeddings(ids + self._offsets)
        return rearrange(embeddings, "b f d -> b (f d)")


class MultiHashEmbeddings(torch.nn.Module):
    """Shared embedding table for already-hashed sparse categorical ids.

    ``ids`` has shape ``[batch, n_features * n_hashes]``. Unlike
    :class:`UnifiedEmbeddings`, every position shares the same hash space.
    """

    def __init__(self, cardinality: int, n_inputs: int, embedding_dim: int) -> None:
        super().__init__()
        if cardinality <= 0 or n_inputs <= 0 or embedding_dim <= 0:
            raise ValueError("cardinality, n_inputs and embedding_dim must be positive")
        self._embeddings = torch.nn.Embedding(cardinality, embedding_dim)
        self._output_dim = n_inputs * embedding_dim

    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return rearrange(self._embeddings(ids), "b f d -> b (f d)")


class EmbeddingTower(torch.nn.Module):
    """Normalize and project a dense external embedding stream."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        self._normalize = normalize
        self._output_dim = output_dim
        layers: list[torch.nn.Module] = [
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        self._net = torch.nn.Sequential(*layers)

    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._normalize:
            x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return self._net(x)


class CrossLayer(torch.nn.Module):
    """x_{l+1} = x_0 * (U(V(x_l)) + b) + x_l.

    Full-rank: ``V = Linear(d, d)``, ``U = Identity``.
    Low-rank:  ``V = Linear(d, r)``, ``U = Linear(r, d)``.

    ``forward`` returns ``(output, inner)`` where ``inner = V(x_l)`` is the
    compressed cross projection. :class:`StackedCrossDeep` reuses ``inner`` to
    feed the deep tower when ``use_inner_cross_layers=True``.
    """

    def __init__(self, input_dim: int, rank: int | None, gated: bool = False) -> None:
        super().__init__()
        self._inner_dim = rank if rank is not None else input_dim
        self._V = torch.nn.Linear(input_dim, self._inner_dim, bias=False)
        self._U = (
            torch.nn.Linear(self._inner_dim, input_dim, bias=False)
            if rank is not None
            else torch.nn.Identity()
        )
        self._gated = gated
        self._gate_linear = torch.nn.Linear(input_dim, input_dim) if gated else torch.nn.Identity()
        self._bias = torch.nn.Parameter(torch.zeros(input_dim))
        torch.nn.init.uniform_(
            self._bias,
            -1 / math.sqrt(self._inner_dim),
            1 / math.sqrt(self._inner_dim),
        )

    def inner_dim(self) -> int:
        return self._inner_dim

    def forward(self, x_0: torch.Tensor, x_l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.ones_like(x_l)
        if self._gated:
            gate = torch.sigmoid(self._gate_linear(x_l))
        inner = self._V(x_l)
        return x_0 * (gate * self._U(inner) + self._bias) + x_l, inner


class MLDCNLayer(torch.nn.Module):
    """ML-DCN block: low-rank cross + instance-guided mask + LayerNorm.

    ``forward`` returns ``(output, inner)`` where ``inner = V(x_l)`` is the
    pre-mask compressed projection.
    """

    def __init__(self, input_dim: int, rank: int, mask_ratio: float = 0.5) -> None:
        super().__init__()
        self._V = torch.nn.Linear(input_dim, rank, bias=False)
        self._U = torch.nn.Linear(rank, input_dim, bias=False)
        self._inner_dim = rank
        self._bias = torch.nn.Parameter(torch.zeros(input_dim))
        hidden = max(1, int(rank * mask_ratio))
        self._mask_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, rank),
        )
        self._layer_norm = torch.nn.LayerNorm(input_dim)

    def inner_dim(self) -> int:
        return self._inner_dim

    def forward(
        self,
        x_0: torch.Tensor,
        x_l: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self._V(x_l)
        masked = projected * self._mask_net(x_l)
        out = self._layer_norm(x_0 * (self._U(masked) + self._bias) + x_l)
        return out, projected


class CrossNetwork(torch.nn.Module):
    """Stack of cross layers.

    ``forward(x)`` returns ``(output, inners)``: the final cross output and a
    list of every layer's compressed inner projection. Body modules that don't
    need ``inners`` (e.g. :class:`ParallelCrossDeep`) simply discard it.
    """

    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        rank: int | None = None,
        gated: bool = False,
        cross_type: str = "standard",
        mask_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        if cross_type == "mldcn":
            if rank is None:
                raise ValueError("cross_rank is required for cross_type='mldcn'")
            self._layers = torch.nn.ModuleList(
                [MLDCNLayer(input_dim, rank, mask_ratio) for _ in range(n_layers)],
            )
        elif cross_type == "standard":
            self._layers = torch.nn.ModuleList(
                [CrossLayer(input_dim, rank, gated=gated) for _ in range(n_layers)],
            )
        else:
            raise ValueError("cross_type must be 'standard' or 'mldcn'")

    def inner_dims(self) -> list[int]:
        """Per-layer compressed inner-projection dimensions."""
        return [layer.inner_dim() for layer in self._layers]

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x_l = x
        inners = []
        for layer in self._layers:
            x_l, inner = layer(x, x_l)
            inners.append(inner)
        return x_l, inners


# Registry for DeepNetwork layer blocks.
# Each factory receives ([i]n_dim, [o]ut_dim, dropout_[p]rob, [b]atch_norm) and
# returns a Module. When ``b`` is True a ``BatchNorm1d`` is inserted right after
# the Linear and before the activation (Linear -> BN -> activation -> Dropout).
# batch_norm is only meaningful for the plain-activation "relu" block; the
# gated/FFN variants accept the flag but ignore it, and DeepNetwork forbids
# batch_norm=True with them.
# With b=False the "relu" block is byte-identical to its pre-batch_norm form.
_DEEPN_LAYER_BUILDERS: dict[str, Callable[[int, int, float, bool], torch.nn.Module]] = {
    "relu": lambda i, o, p, b: torch.nn.Sequential(
        torch.nn.Linear(i, o),
        torch.nn.BatchNorm1d(o) if b else torch.nn.Identity(),
        torch.nn.ReLU(),
        torch.nn.Dropout(p) if p > 0 else torch.nn.Identity(),
    ),
    "glu": lambda i, o, p, b: torch.nn.Sequential(  # noqa: ARG005
        GLU(i, o),
        torch.nn.Dropout(p) if p > 0 else torch.nn.Identity(),
    ),
    "geglu": lambda i, o, p, b: torch.nn.Sequential(  # noqa: ARG005
        GLU(i, o, activation=F.gelu),
        torch.nn.Dropout(p) if p > 0 else torch.nn.Identity(),
    ),
    "swiglu_ffn": lambda i, o, p, b: SwiGLUFFN(i, o, dropout=p, sigmoid_gate=False),  # noqa: ARG005
    "glu_ffn": lambda i, o, p, b: SwiGLUFFN(i, o, dropout=p, sigmoid_gate=True),  # noqa: ARG005
}


class DeepNetwork(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: list[int],
        dropout: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = False,
        use_moe: bool = False,
        num_experts: int = 4,
        moe_top_k: int = 2,
    ) -> None:
        super().__init__()
        if activation not in _DEEPN_LAYER_BUILDERS:
            raise ValueError(
                f"activation must be one of {list(_DEEPN_LAYER_BUILDERS)}, got {activation!r}",
            )
        if batch_norm and activation != "relu":
            raise ValueError(
                "batch_norm=True is only supported with plain activations "
                f"relu, got activation={activation!r}",
            )
        build = _DEEPN_LAYER_BUILDERS[activation]
        layers = []
        for i, units in enumerate(hidden_units):
            block = (
                MOELayer(input_dim, units, num_experts, moe_top_k)
                if use_moe and i == 0
                else build(input_dim, units, dropout, batch_norm)
            )
            layers.append(block)
            input_dim = units
        self._network = torch.nn.Sequential(*layers)
        self._output_dim = input_dim

    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._network(x)


class StackedCrossDeep(torch.nn.Module):
    """Stacked cross→deep body for DCNv2.

    Replaces ``nn.Sequential(CrossNetwork, DeepNetwork)`` for the stacked
    path. When ``use_inner_cross_layers=True``, the cross network's
    compressed inner projections are concatenated with ``adapter(cross_out)``
    and fed to the deep tower.
    """

    def __init__(
        self,
        cross: CrossNetwork,
        deep: DeepNetwork | None = None,
        *,
        use_inner_cross_layers: bool = False,
        adapter: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        if use_inner_cross_layers and (deep is None or adapter is None):
            raise ValueError(
                "use_inner_cross_layers=True requires both a deep tower and an adapter",
            )
        self._cross = cross
        self._deep = deep
        self._adapter = adapter if adapter is not None else torch.nn.Identity()
        self._use_inner_cross_layers = use_inner_cross_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cross_out, inners = self._cross(x)
        if self._deep is None:
            return cross_out
        if self._use_inner_cross_layers:
            return self._deep(torch.cat([self._adapter(cross_out), *inners], dim=-1))
        return self._deep(cross_out)


class ParallelCrossDeep(torch.nn.Module):
    """Parallel cross/deep body for DCNv2: concat cross and deep outputs."""

    def __init__(self, cross: CrossNetwork, deep: DeepNetwork | None = None) -> None:
        super().__init__()
        self._cross = cross
        self._deep = deep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cross_out, _ = self._cross(x)
        if self._deep is None:
            return cross_out
        return torch.cat([cross_out, self._deep(x)], dim=-1)


class GLU(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.sigmoid,
    ) -> None:
        super().__init__()
        self._linear = torch.nn.Linear(in_features, out_features * 2)
        self._activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self._linear(x).chunk(2, dim=-1)
        return x * self._activation(gate)


class SwiGLUFFN(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
        sigmoid_gate: bool = False,
    ) -> None:
        super().__init__()
        hidden = max(1, int(out_features * 2 / 3))
        self._ln = torch.nn.LayerNorm(in_features)
        self._gate_proj = torch.nn.Linear(in_features, hidden, bias=False)
        self._value = torch.nn.Linear(in_features, hidden, bias=False)
        self._down = torch.nn.Linear(hidden, out_features, bias=False)
        self._dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self._proj = (
            torch.nn.Identity()
            if in_features == out_features
            else torch.nn.Linear(in_features, out_features, bias=False)
        )
        self._gate_fn = torch.sigmoid if sigmoid_gate else F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._ln(x)
        gate = self._gate_fn(self._gate_proj(h))
        h = self._dropout(self._down(gate * self._value(h)))
        return self._proj(x) + h


class MOELayer(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self._top_k = top_k
        self._gate = torch.nn.Linear(input_dim, num_experts)
        self._experts = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(input_dim, output_dim),
                    torch.nn.ReLU(),
                )
                for _ in range(num_experts)
            ],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._gate(x)
        top_logits, top_idx = logits.topk(self._top_k, dim=-1)
        weights = torch.softmax(top_logits, dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self._experts], dim=1)
        selected = expert_outputs.gather(
            1,
            repeat(top_idx, "b k -> b k d", d=expert_outputs.size(-1)),
        )
        return torch.einsum("b k d, b k -> b d", selected, weights)


class CoralLayer(torch.nn.Module):
    """CORAL output head: shared weight + K-1 ordered-logit biases."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        preinit_bias: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("CoralLayer requires at least two classes")
        self._num_classes = num_classes
        k = num_classes - 1
        self._w = torch.nn.Linear(in_features, 1, bias=False)
        init = torch.arange(k, 0, -1).float() / k if preinit_bias else torch.zeros(k)
        self._biases = torch.nn.Parameter(init)

    def output_dim(self) -> int:
        return self._num_classes - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._w(x) + rearrange(self._biases, "k -> 1 k")


class Parallel(torch.nn.Module):
    """Apply each branch to the same input ``x``; concatenate along ``dim``.

    Mirrors :class:`torch.nn.ModuleList`-based composition: pass a sequence
    of branches, each producing a tensor; ``forward`` returns their
    concatenation along the chosen dimension (last dim by default).
    """

    def __init__(self, modules: Sequence[torch.nn.Module], dim: int = -1) -> None:
        super().__init__()
        self._branches = torch.nn.ModuleList(modules)
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([m(x) for m in self._branches], dim=self._dim)


class DCNv2(torch.nn.Module):
    """DCNv2 composition root.

    Pure dependency injection: every submodule is supplied already built. Use
    :func:`scikit_rank.factories.build_dcnv2` to construct one from string specs /
    scalars; the model class itself contains no construction logic.

    Forward contract
    ----------------
    ``forward(inputs)`` reads each key in ``layers`` from ``inputs`` (a
    superset is fine; extra keys are ignored), applies the matching layer,
    fuses the encoded streams through ``reducer``, passes the result through
    ``body`` (typically :class:`torch.nn.Sequential` for stacked DCN or
    :class:`Parallel` for parallel DCN), and projects with ``head``.

    Layer / reducer contracts
    -------------------------
    * Every layer in ``layers`` must expose ``output_dim: int``.
    * ``reducer`` must expose ``compute_output_dim(input_dims) -> int`` so
      downstream sizing can happen without a dry-run forward.
    """

    def __init__(
        self,
        *,
        layers: dict[str, torch.nn.Module],
        reducer: torch.nn.Module,
        body: torch.nn.Module,
        head: torch.nn.Module,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("layers must contain at least one entry")
        self._layers = torch.nn.ModuleDict(layers)
        self._reducer = reducer
        self._body = body
        self._head = head

    def layers(self) -> torch.nn.ModuleDict:
        return self._layers

    def reducer(self) -> torch.nn.Module:
        return self._reducer

    def head(self) -> torch.nn.Module:
        return self._head

    def embedding_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Parameters of the feature encoders (numeric + categorical).

        Includes the per-feature numeric projection (none for ``identity``) and
        categorical embedding tables so an embedding-only L2 penalty can target
        exactly these tensors. The cross/deep body and head are excluded.
        """
        return self._layers.parameters()

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = {name: layer(inputs[name]) for name, layer in self._layers.items()}
        x = self._body(self._reducer(encoded))
        out = self._head(x)
        return out.squeeze(-1) if out.ndim > 1 and out.size(-1) == 1 else out
