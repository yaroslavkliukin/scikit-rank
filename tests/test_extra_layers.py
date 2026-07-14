from __future__ import annotations

import numpy as np
import pytest
import torch

from scikit_rank.data import TensorDatasetSource
from scikit_rank.factories import (
    build_dcnv2,
    build_embedding_encoder,
    build_multihash_encoder,
)
from scikit_rank.modules.dcn import EmbeddingTower, MultiHashEmbeddings


def test_multihash_embeddings_shape_and_shared_table() -> None:
    layer = MultiHashEmbeddings(cardinality=17, n_inputs=6, embedding_dim=4)
    x = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    assert layer(x).shape == (2, 24)
    assert layer.output_dim() == 24


def test_embedding_tower_normalizes_and_projects() -> None:
    tower = EmbeddingTower(8, 3, dropout=0.0)
    assert tower(torch.randn(5, 8)).shape == (5, 3)
    assert tower.output_dim() == 3


def test_build_dcnv2_accepts_all_reference_streams() -> None:
    model = build_dcnv2(
        n_num_features=2,
        cardinalities=[5],
        embedding_dims=[2],
        hidden_units=[16, 8],
        cross_layers=2,
        cross_rank=4,
        num_encoder="identity",
        multihash_encoder="multihash:cardinality=31;n_hashes=2;embedding_dim=3",
        multihash_n_inputs=6,
        embedding_encoders={
            "user_embeddings": "tower:output_dim=4;dropout=0.0",
            "item_embeddings": "tower:output_dim=4;dropout=0.0",
        },
        embedding_input_dims={"user_embeddings": 7, "item_embeddings": 9},
    )
    assert model.layers()["multihash"].output_dim() == 18  # 6 * 3
    out = model(
        {
            "num": torch.randn(3, 2),
            "cat": torch.randint(0, 5, (3, 1)),
            "multihash": torch.randint(0, 31, (3, 6)),
            "user_embeddings": torch.randn(3, 7),
            "item_embeddings": torch.randn(3, 9),
        },
    )
    assert out.shape == (3,)


def test_multihash_encoder_spec_defaults_and_override() -> None:
    default = build_multihash_encoder("multihash", n_inputs=6)
    assert default.output_dim() == 6 * 16  # _MULTIHASH_DEFAULTS embedding_dim=16
    keyed = build_multihash_encoder(
        "multihash:cardinality=31;n_hashes=3;embedding_dim=4",
        n_inputs=6,
    )
    assert keyed.output_dim() == 6 * 4
    # no stream -> no module
    assert build_multihash_encoder("multihash", n_inputs=0) is None


def test_multihash_encoder_spec_validates_positive_config() -> None:
    with pytest.raises(ValueError, match="cardinality must be positive"):
        build_dcnv2(
            n_num_features=1,
            cardinalities=[],
            multihash_encoder="multihash:cardinality=0",
            multihash_n_inputs=1,
        )


def test_embedding_encoder_spec_defaults_and_override() -> None:
    default = build_embedding_encoder("tower", input_dim=8)
    assert default.output_dim() == 64  # _EMBEDDING_TOWER_DEFAULTS output_dim=64
    keyed = build_embedding_encoder("tower:output_dim=5;dropout=0.0", input_dim=8)
    assert keyed(torch.randn(2, 8)).shape == (2, 5)


def test_reference_encoders_accept_nn_modules() -> None:
    mh = MultiHashEmbeddings(cardinality=31, n_inputs=6, embedding_dim=3)
    built = build_multihash_encoder(mh, n_inputs=6)
    assert built is not mh  # deepcopied
    assert built.output_dim() == mh.output_dim()
    tower = EmbeddingTower(8, 5, dropout=0.0)
    assert build_embedding_encoder(tower, input_dim=8) is not tower


def test_embedding_encoders_require_input_dim() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        build_dcnv2(
            n_num_features=1,
            cardinalities=[],
            embedding_encoders={"user_embeddings": "tower"},
        )


def test_tensor_dataset_source_carries_extra_features() -> None:
    source = TensorDatasetSource(
        np.zeros((2, 1), np.float32),
        np.zeros((2, 0), np.int64),
        np.zeros(2, np.float32),
        np.arange(2),
        batch_size=2,
        shuffle=False,
        rng=np.random.default_rng(0),
        extra_features={"multihash": np.ones((2, 3), np.int64)},
    )
    assert source[0]["multihash"].tolist() == [1, 1, 1]
