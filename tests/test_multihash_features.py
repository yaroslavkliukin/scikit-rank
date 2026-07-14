from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scikit_rank import DCNClassifier
from scikit_rank.preprocessing import _MISSING, TabularPreprocessor


def _frame(n: int = 240) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    X = pl.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
            "user_id": rng.integers(0, 5000, size=n).astype(str),
            "item_id": rng.integers(0, 9000, size=n).astype(str),
        },
    )
    y = (X["x1"].to_numpy() + X["x2"].to_numpy() > 0).astype(int)
    return X, y


def test_multihash_preprocessor_shapes_and_determinism() -> None:
    X, _ = _frame()
    pre = TabularPreprocessor(
        multihash_features=["user_id", "item_id"],
        multihash_cardinality=257,
        multihash_n_hashes=3,
    ).fit(X)

    # multihash columns are peeled out of num/cat inference
    assert pre.multihash_cols_ == ["user_id", "item_id"]
    assert "user_id" not in pre.num_cols_
    assert "user_id" not in pre.cat_cols_
    assert pre.num_cols_ == ["x1", "x2"]
    assert pre.cat_cols_ == ["cat"]

    # one int id per (feature, hash function)
    assert pre.multihash_n_inputs_ == 2 * 3
    assert len(pre.multihash_out_cols_) == 6

    _, _, extra = pre.transform(X)
    mh = extra["multihash"]
    assert mh.dtype == np.int64
    assert mh.shape == (len(X), 6)
    assert mh.min() >= 0
    assert mh.max() < 257

    # deterministic: a second transform of the same frame buckets identically
    _, _, extra2 = pre.transform(X)
    assert np.array_equal(mh, extra2["multihash"])


def test_multihash_overlap_with_cat_raises() -> None:
    X, _ = _frame()
    with pytest.raises(ValueError, match="overlap"):
        TabularPreprocessor(
            cat_features=["cat", "user_id"],
            multihash_features=["user_id"],
        ).fit(X)


def test_multihash_n_hashes_must_be_positive() -> None:
    with pytest.raises(ValueError, match="multihash_n_hashes"):
        TabularPreprocessor(multihash_features=["u"], multihash_n_hashes=0)


def test_dcn_classifier_multihash_end_to_end_eager() -> None:
    X, y = _frame()
    clf = DCNClassifier(
        epochs=2,
        batch_size=64,
        hidden_units=[8],
        cross_layers=1,
        multihash_features=["user_id", "item_id"],
        multihash_encoder="multihash:cardinality=512;n_hashes=3;embedding_dim=4",
        accelerator_config={"cpu": True},
        random_state=0,
    ).fit(X, y)

    assert "multihash" in clf.model_.layers()
    assert clf.model_.layers()["multihash"].output_dim() == 2 * 3 * 4
    # raw high-cardinality columns count as input features (predict validation)
    assert clf.n_features_in_ == 5
    assert "user_id" in clf.feature_names_in_
    assert clf.predict_proba(X.head(4)).shape == (4, 2)


def test_dcn_classifier_multihash_end_to_end_lazy() -> None:
    X, y = _frame()
    lazy = X.with_columns(pl.Series("y", y)).lazy()
    clf = DCNClassifier(
        epochs=2,
        batch_size=64,
        hidden_units=[8],
        cross_layers=1,
        multihash_features=["user_id", "item_id"],
        multihash_encoder="multihash:cardinality=512;n_hashes=2",
        accelerator_config={"cpu": True},
        random_state=0,
    ).fit(lazy, "y")

    assert "multihash" in clf.model_.layers()
    assert clf.predict_proba(X.head(4)).shape == (4, 2)


def _shared_value_frame() -> pl.DataFrame:
    """Two multihash features sharing raw string values row-by-row (+ a null)."""
    return pl.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 0.4],
            "user_id": ["0", "777", "abc", None],
            "item_id": ["0", "777", "xyz", None],
        },
    )


def _multihash_ids(pre: TabularPreprocessor, X: pl.DataFrame) -> np.ndarray:
    return pre.transform(X)[2]["multihash"]


def test_multihash_no_cross_feature_collision() -> None:
    # Per-feature salting (always on): the same raw value in different features
    # no longer maps to the same bucket, and probes within a feature stay
    # independent.
    X = _shared_value_frame()
    pre = TabularPreprocessor(
        multihash_features=["user_id", "item_id"],
        multihash_n_hashes=2,
    ).fit(X)
    mh = _multihash_ids(pre, X)
    # columns: user_id::h0, user_id::h1, item_id::h0, item_id::h1
    # rows 0 ('0'), 1 ('777'), 3 (null) share values across the two features
    for row in (0, 1, 3):
        assert mh[row, 0] != mh[row, 2]  # user probe0 != item probe0
        assert mh[row, 1] != mh[row, 3]  # user probe1 != item probe1
    # probes within one feature stay independent
    assert mh[0, 0] != mh[0, 1]


def test_multihash_seed_is_salted_per_feature() -> None:
    # The seed is feature_index * n_hashes + k, so feature 0 uses seeds {0, 1}
    # and feature 1 uses seeds {2, 3}.
    X = _shared_value_frame()
    pre = TabularPreprocessor(
        multihash_features=["user_id", "item_id"],
        multihash_n_hashes=2,
        multihash_cardinality=100_000,
    ).fit(X)
    mh = _multihash_ids(pre, X)

    def expected(col: str, seed: int) -> np.ndarray:
        e = (
            pl.col(col)
            .cast(pl.String)
            .fill_null(_MISSING)
            .hash(seed=seed)
            .mod(100_000)
            .cast(pl.Int64)
        )
        return X.select(e)[col].to_numpy()

    assert np.array_equal(mh[:, 0], expected("user_id", 0))
    assert np.array_equal(mh[:, 1], expected("user_id", 1))
    assert np.array_equal(mh[:, 2], expected("item_id", 2))
    assert np.array_equal(mh[:, 3], expected("item_id", 3))
