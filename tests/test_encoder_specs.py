import numpy as np
import polars as pl
import pytest
import torch

from scikit_rank import (
    Concat,
    DCNClassifier,
    DCNv2,
    ModuleParserSpec,
    UnifiedEmbeddings,
    build_dcnv2,
)
from scikit_rank.modules.dcn import PiecewiseLinearEncoder, PLREncoder


class _LinearNumeric(torch.nn.Module):
    """Custom numeric encoder module exposing the required ``output_dim``."""

    def __init__(self, n_features: int, out: int = 6) -> None:
        super().__init__()
        self._lin = torch.nn.Linear(n_features, out)
        self._output_dim = out

    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self._lin(x))


class _SumEmbeddings(torch.nn.Module):
    """Custom categorical encoder summing per-feature embeddings."""

    def __init__(self, cardinalities: list[int], dim: int = 5) -> None:
        super().__init__()
        self._embs = torch.nn.ModuleList(
            [torch.nn.Embedding(c + 1, dim) for c in cardinalities],
        )
        self._output_dim = dim

    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, cat: torch.Tensor) -> torch.Tensor:
        return sum(emb(cat[:, i]) for i, emb in enumerate(self._embs))


def _cat_batch(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.long)


def test_build_dcnv2_accepts_nn_module_encoders() -> None:
    num_enc = _LinearNumeric(2, out=6)
    cat_enc = _SumEmbeddings([4, 5], dim=5)
    model = build_dcnv2(
        n_num_features=2,
        cardinalities=[4, 5],
        embedding_dims=[3, 4],
        hidden_units=[8],
        num_encoder=num_enc,
        cat_encoder=cat_enc,
    )

    # the modules are used as-is (by type) but deepcopied so external state is
    # not shared/trained in place
    assert isinstance(model.layers()["num"], _LinearNumeric)
    assert isinstance(model.layers()["cat"], _SumEmbeddings)
    assert model.layers()["num"] is not num_enc
    assert model.layers()["cat"] is not cat_enc
    assert model.layers()["num"].output_dim() == 6
    assert isinstance(model.reducer(), Concat)

    out = model(
        {
            "num": torch.randn(5, 2),
            "cat": _cat_batch([[0, 1], [1, 2], [2, 3], [3, 4], [0, 0]]),
        },
    )
    assert out.shape == (5,)


def test_module_encoder_requires_output_dim() -> None:
    class NoOutputDim(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
            return x

    with pytest.raises(TypeError, match="output_dim"):
        build_dcnv2(
            n_num_features=2,
            cardinalities=[],
            hidden_units=[8],
            num_encoder=NoOutputDim(),
        )


def test_estimator_accepts_module_encoder() -> None:
    rng = np.random.default_rng(0)
    n = 200
    X = pl.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        },
    )
    y = (X["x1"].to_numpy() + X["x2"].to_numpy() > 0).astype(int)

    enc = _LinearNumeric(2, out=6)
    clf = DCNClassifier(
        epochs=2,
        batch_size=64,
        hidden_units=[8],
        cross_layers=1,
        num_encoder=enc,
        cat_encoder=_SumEmbeddings([3], dim=5),
        accelerator_config={"cpu": True},
        random_state=0,
    ).fit(X, y)

    assert isinstance(clf.model_.layers()["num"], _LinearNumeric)
    assert isinstance(clf.model_.layers()["cat"], _SumEmbeddings)
    assert clf.model_.layers()["num"] is not enc  # deepcopied
    assert clf.predict_proba(X.head(4)).shape == (4, 2)


def test_encoder_specs_without_kwargs_use_defaults() -> None:
    plr_spec = ModuleParserSpec("PLR")
    ple_spec = ModuleParserSpec("PLE")
    unified_spec = ModuleParserSpec("unified")
    assert (plr_spec.module_name(), dict(plr_spec.kwargs())) == ("plr", {})
    assert (ple_spec.module_name(), dict(ple_spec.kwargs())) == ("ple", {})
    assert (unified_spec.module_name(), dict(unified_spec.kwargs())) == ("unified", {})

    plr_model = build_dcnv2(
        n_num_features=2,
        cardinalities=[4, 5],
        embedding_dims=[3, 4],
        hidden_units=[8],
        num_encoder="PLR",
        cat_encoder="unified",
    )
    assert isinstance(plr_model.layers()["num"], PLREncoder)
    assert isinstance(plr_model.layers()["cat"], UnifiedEmbeddings)
    assert plr_model.layers()["num"].output_dim() == 2 * 16  # default PLR embedding_dim
    assert plr_model.layers()["cat"].output_dim() == 2 * 16  # default unified embedding_dim

    ple_bins = [torch.linspace(-2.0, 2.0, 17), torch.linspace(-3.0, 3.0, 17)]
    ple_model = build_dcnv2(
        n_num_features=2,
        cardinalities=[4],
        embedding_dims=[3],
        hidden_units=[8],
        num_encoder="PLE",
        num_encoder_bins=ple_bins,
    )
    assert isinstance(ple_model.layers()["num"], PiecewiseLinearEncoder)
    # No projection (embedding_dim default=None): output = sum of per-feature n_bins
    assert ple_model.layers()["num"].output_dim() == 2 * 16

    keyed_model = build_dcnv2(
        n_num_features=2,
        cardinalities=[4],
        embedding_dims=[3],
        hidden_units=[8],
        num_encoder="PLR:n_freq=4;embedding_dim=3;activation=silu",
        cat_encoder="per_feature",
    )
    assert isinstance(keyed_model.layers()["num"], PLREncoder)

    out = plr_model(
        {
            "num": torch.randn(5, 2),
            "cat": _cat_batch([[0, 1], [1, 2], [2, 3], [3, 4], [0, 0]]),
        },
    )
    assert out.shape == (5,)


def test_dcnv2_forward_with_explicit_construction() -> None:
    """``DCNv2`` itself only does composition; constructors take Modules only."""
    from scikit_rank.modules.dcn import (
        CrossNetwork,
        DeepNetwork,
        ParallelCrossDeep,
        StackedCrossDeep,
    )

    layers = {
        "num": _LinearNumeric(2, out=6),
        "cat": _SumEmbeddings([4, 5], dim=5),
    }
    reducer = Concat(dim=-1)
    rep_dim = reducer.compute_output_dim({k: m.output_dim() for k, m in layers.items()})

    # Stacked body wires cross -> deep through the dedicated body module.
    stacked_model = DCNv2(
        layers=layers,
        reducer=reducer,
        body=StackedCrossDeep(
            CrossNetwork(rep_dim, n_layers=1),
            DeepNetwork(rep_dim, hidden_units=[8]),
        ),
        head=torch.nn.Linear(8, 1),
    )
    out = stacked_model(
        {
            "num": torch.randn(5, 2),
            "cat": _cat_batch([[0, 1], [1, 2], [2, 3], [3, 4], [0, 0]]),
        },
    )
    assert out.shape == (5,)

    # Parallel body cats cross and deep outputs along dim=-1.
    parallel_model = DCNv2(
        layers=layers,
        reducer=reducer,
        body=ParallelCrossDeep(
            CrossNetwork(rep_dim, n_layers=1),
            DeepNetwork(rep_dim, hidden_units=[8]),
        ),
        head=torch.nn.Linear(rep_dim + 8, 1),
    )
    out = parallel_model(
        {
            "num": torch.randn(5, 2),
            "cat": _cat_batch([[0, 1], [1, 2], [2, 3], [3, 4], [0, 0]]),
        },
    )
    assert out.shape == (5,)


def test_estimator_accepts_default_only_specs() -> None:
    rng = np.random.default_rng(0)
    n = 128
    X = pl.DataFrame(
        {
            "x": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        },
    )
    y = (X["x"].to_numpy() > 0).astype(int)

    clf = DCNClassifier(
        epochs=1,
        batch_size=64,
        hidden_units=[8],
        num_encoder="PLR",
        cat_encoder="unified",
        random_state=0,
    ).fit(X, y)

    assert isinstance(clf.model_.layers()["num"], PLREncoder)
    assert isinstance(clf.model_.layers()["cat"], UnifiedEmbeddings)
    assert clf.predict_proba(X.head(4)).shape == (4, 2)
