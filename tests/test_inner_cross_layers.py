"""Tests for the ``use_inner_cross_layers`` feature."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scikit_rank import DCNClassifier, build_dcnv2
from scikit_rank.modules.dcn import (
    CrossLayer,
    CrossNetwork,
    MLDCNLayer,
    StackedCrossDeep,
)


def test_cross_layer_forward_returns_output_and_inner() -> None:
    layer = CrossLayer(input_dim=8, rank=None)
    x = torch.randn(4, 8)
    out, inner = layer(x, x)
    assert out.shape == (4, 8)
    # Full-rank: inner dim equals input dim and equals V(x_l).
    assert inner.shape == (4, 8)
    assert layer.inner_dim() == 8


def test_low_rank_cross_layer_inner_has_rank_dim() -> None:
    layer = CrossLayer(input_dim=8, rank=3)
    x = torch.randn(4, 8)
    out, inner = layer(x, x)
    assert out.shape == (4, 8)
    assert inner.shape == (4, 3)
    assert layer.inner_dim() == 3


def test_gated_cross_layer_returns_inner() -> None:
    layer = CrossLayer(input_dim=6, rank=2, gated=True)
    x = torch.randn(3, 6)
    out, inner = layer(x, x)
    assert out.shape == (3, 6)
    assert inner.shape == (3, 2)


def test_mldcn_layer_returns_inner() -> None:
    layer = MLDCNLayer(input_dim=6, rank=3)
    x = torch.randn(3, 6)
    out, inner = layer(x, x)
    assert out.shape == (3, 6)
    assert inner.shape == (3, 3)


def test_cross_network_returns_inners_one_per_layer() -> None:
    net = CrossNetwork(input_dim=8, n_layers=3, rank=2)
    out, inners = net(torch.randn(5, 8))
    assert out.shape == (5, 8)
    assert len(inners) == 3
    assert all(t.shape == (5, 2) for t in inners)
    assert net.inner_dims() == [2, 2, 2]


def test_stacked_cross_deep_default_ignores_inners() -> None:
    cross = CrossNetwork(input_dim=4, n_layers=2)
    body = StackedCrossDeep(cross)  # no deep tower -> returns cross output
    out = body(torch.randn(3, 4))
    assert out.shape == (3, 4)


def test_build_dcnv2_with_inner_cross_layers_runs() -> None:
    model = build_dcnv2(
        n_num_features=4,
        cardinalities=[],
        embedding_dims=None,
        hidden_units=[16, 8],
        cross_layers=2,
        cross_rank=3,
        structure="stacked",
        num_encoder="identity",
        use_inner_cross_layers=True,
    )
    out = model({"num": torch.randn(5, 4)})
    assert out.shape == (5,)


def test_build_dcnv2_inner_cross_layers_rejects_parallel_structure() -> None:
    with pytest.raises(ValueError, match="structure='stacked'"):
        build_dcnv2(
            n_num_features=4,
            cardinalities=[],
            embedding_dims=None,
            hidden_units=[16, 8],
            cross_layers=2,
            structure="parallel",
            num_encoder="identity",
            use_inner_cross_layers=True,
        )


def test_build_dcnv2_inner_cross_layers_requires_two_hidden_units() -> None:
    with pytest.raises(ValueError, match="hidden_units"):
        build_dcnv2(
            n_num_features=4,
            cardinalities=[],
            embedding_dims=None,
            hidden_units=[16],
            cross_layers=2,
            structure="stacked",
            num_encoder="identity",
            use_inner_cross_layers=True,
        )


def test_dcn_classifier_uses_inner_cross_layers() -> None:
    rng = np.random.default_rng(0)
    n = 256
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(np.int64)
    clf = DCNClassifier(
        epochs=1,
        batch_size=32,
        hidden_units=[8, 4],
        cross_layers=2,
        cross_rank=2,
        use_inner_cross_layers=True,
        accelerator_config={"cpu": True},
        random_state=0,
    ).fit(X, y)
    assert clf.predict_proba(X[:5]).shape == (5, 2)
