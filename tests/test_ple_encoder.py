"""Tests for the data-driven piecewise-linear numeric encoder."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch

from scikit_rank import DCNClassifier
from scikit_rank.modules.dcn import PiecewiseLinearEncoder, _PiecewiseLinearEncodingImpl
from scikit_rank.preprocessing import TabularPreprocessor


def _toy_frame(n: int = 256, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.exponential(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        },
    )


def test_ple_impl_clamps_within_unit_interval_for_interior_bins() -> None:
    bins = [torch.tensor([-1.0, 0.0, 1.0, 2.0])]  # 3 bins for 1 feature
    impl = _PiecewiseLinearEncodingImpl(bins)
    x = torch.tensor([[-5.0], [0.5], [10.0]])
    out = impl(x)
    # impl returns (batch, n_features, max_n_bins).
    assert out.shape == (3, 1, 3)
    # First encoding component is clamped at upper 1.0 across all rows.
    assert (out[:, 0, 0] <= 1.0 + 1e-6).all()
    # Interior components live in [0, 1].
    assert ((out[:, 0, 1] >= -1e-6) & (out[:, 0, 1] <= 1.0 + 1e-6)).all()


def test_piecewise_linear_encoder_output_dim_without_projection() -> None:
    # Uniform-width bins: 4 features each with 5 edges -> 4 bins per feature.
    bins = [torch.linspace(-2.0, 2.0, 5) for _ in range(4)]
    enc = PiecewiseLinearEncoder(bins)
    assert enc.output_dim() == 4 * 4
    out = enc(torch.randn(8, 4))
    assert out.shape == (8, 4 * 4)


def test_piecewise_linear_encoder_with_projection() -> None:
    bins = [torch.linspace(-2.0, 2.0, 9) for _ in range(3)]  # 8 bins per feature
    enc = PiecewiseLinearEncoder(bins, embedding_dim=6, activation=True)
    assert enc.output_dim() == 3 * 6
    out = enc(torch.randn(5, 3))
    assert out.shape == (5, 3 * 6)


def test_piecewise_linear_encoder_handles_non_uniform_bin_counts() -> None:
    # 1st feature: 3 bins (4 edges); 2nd feature: 5 bins (6 edges). The impl
    # should drop padding components when reporting output_dim.
    bins = [torch.linspace(-1.0, 1.0, 4), torch.linspace(-1.0, 1.0, 6)]
    enc = PiecewiseLinearEncoder(bins)
    assert enc.output_dim() == 3 + 5
    out = enc(torch.randn(4, 2))
    assert out.shape == (4, 3 + 5)


def test_piecewise_linear_encoder_rejects_degenerate_bins() -> None:
    with pytest.raises(ValueError, match=">=2 edges"):
        PiecewiseLinearEncoder([torch.tensor([0.0])])


def test_preprocessor_fit_ple_bins_returns_sorted_unique_edges() -> None:
    pre = TabularPreprocessor().fit(_toy_frame())
    bins = pre.fit_ple_bins(_toy_frame(), n_bins=16)
    assert len(bins) == len(pre.num_cols_) == 2
    for edges in bins:
        assert edges.ndim == 1
        assert edges.numel() >= 2
        # sorted, unique
        assert bool((edges[:-1] < edges[1:]).all())


def test_preprocessor_fit_ple_bins_works_on_lazy_frame() -> None:
    df = _toy_frame()
    pre = TabularPreprocessor().fit(df.lazy())
    bins = pre.fit_ple_bins(df.lazy(), n_bins=8)
    assert len(bins) == 2
    assert all(b.numel() >= 2 for b in bins)


def test_preprocessor_fit_ple_bins_rejects_constant_feature() -> None:
    df = pl.DataFrame({"const": [1.0] * 64, "y": [0, 1] * 32})
    pre = TabularPreprocessor().fit(df, exclude=("y",))
    with pytest.raises(ValueError, match="constant"):
        pre.fit_ple_bins(df, n_bins=8)


def test_dcn_classifier_uses_ple_end_to_end() -> None:
    rng = np.random.default_rng(0)
    n = 512
    X = pl.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.exponential(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        },
    )
    y = ((X["x1"].to_numpy() + 0.5 * X["x2"].to_numpy()) > 1.0).astype(np.int64)
    clf = DCNClassifier(
        epochs=1,
        batch_size=64,
        hidden_units=[16, 8],
        num_encoder="ple:embedding_dim=4",
        ple_n_bins=16,
        accelerator_config={"cpu": True},
        random_state=0,
    ).fit(X, y)
    assert clf.predict_proba(X.head(10)).shape == (10, 2)
