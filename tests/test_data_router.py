"""Unit tests for :class:`scikit_rank.sklearn._data_router.DataRouter`.

The router was extracted from ``sklearn/estimator.py`` (which was doing too
much). These tests exercise it in isolation -- without constructing a full
estimator -- to lock in the eager/lazy routing decision, the column-name vs
array resolution, group code densification, eval-set group inheritance, and the
temp-file cleanup contract.
"""

from __future__ import annotations
import contextlib
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scikit_rank.data import LazyArrowBatchSource, TensorDatasetSource
from scikit_rank.preprocessing import TabularPreprocessor
from scikit_rank.sklearn._data_router import (
    DataRouter,
    _pop_column,
    _require_column_name,
)


def _frame(n: int = 40) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "y": (rng.normal(size=n) > 0).astype(np.float32),
            "g": rng.integers(0, 5, size=n),
        },
    )


def _router(frame: pl.DataFrame | pl.LazyFrame, *, exclude: tuple[str, ...]) -> DataRouter:
    pre = TabularPreprocessor(num_features=["f0", "f1"], normalize=True).fit(
        frame,
        exclude=exclude,
    )
    return DataRouter(
        pre,
        batch_size=8,
        chunk_rows=16,
        rng=np.random.default_rng(1),
        encode_target=lambda a: a.astype(np.float32),
        target_expr=lambda col: pl.col(col).cast(pl.Float32),
    )


# -- helpers -----------------------------------------------------------------


def test_pop_column_by_name_drops_and_returns_array() -> None:
    df = _frame(10)
    out, arr = _pop_column(df, "y", name="y")
    assert "y" not in out.columns
    assert arr is not None
    assert arr.shape == (10,)


def test_pop_column_by_array_keeps_frame() -> None:
    df = _frame(10)
    vals = np.arange(10)
    out, arr = _pop_column(df, vals, name="group")
    assert out.columns == df.columns
    np.testing.assert_array_equal(arr, vals)


def test_pop_column_length_mismatch_raises() -> None:
    df = _frame(10)
    with pytest.raises(ValueError, match="inconsistent lengths"):
        _pop_column(df, np.arange(9), name="group")


def test_pop_column_missing_name_raises() -> None:
    df = _frame(10)
    with pytest.raises(ValueError, match="not found in X columns"):
        _pop_column(df, "nope", name="y")


def test_require_column_name_rejects_array() -> None:
    with pytest.raises(TypeError, match="must be a column name"):
        _require_column_name(np.arange(3), name="y")


def test_require_column_name_passthrough_and_none() -> None:
    assert _require_column_name("col", name="y") == "col"
    assert _require_column_name(None, name="group") is None


# -- eager routing -----------------------------------------------------------


def test_eager_build_returns_tensor_source_with_group_codes() -> None:
    df = _frame()
    router = _router(df, exclude=("y", "g"))
    with contextlib.ExitStack() as cleanup:
        src = router.build_train_source(df, "y", "g", cleanup=cleanup)
    assert isinstance(src, TensorDatasetSource)
    assert src.group_np is not None
    # group ids are densified to a contiguous 0..k-1 range
    assert set(np.unique(src.group_np)) == set(range(len(np.unique(df["g"].to_numpy()))))


def test_eager_build_missing_y_raises() -> None:
    df = _frame()
    router = _router(df, exclude=("y", "g"))
    with contextlib.ExitStack() as cleanup, pytest.raises(ValueError, match="y is required"):
        router.build_train_source(df, None, None, cleanup=cleanup)


def test_eager_encode_target_callable_is_applied() -> None:
    df = _frame()
    pre = TabularPreprocessor(num_features=["f0", "f1"], normalize=True).fit(
        df,
        exclude=("y", "g"),
    )
    router = DataRouter(
        pre,
        batch_size=8,
        chunk_rows=16,
        rng=np.random.default_rng(1),
        encode_target=lambda a: a.astype(np.float32) + 100.0,  # tag the target
        target_expr=lambda col: pl.col(col).cast(pl.Float32),
    )
    with contextlib.ExitStack() as cleanup:
        src = router.build_train_source(df, "y", None, cleanup=cleanup)
    batch = next(iter(src))  # type: ignore[arg-type]
    assert float(batch["target"]) >= 100.0


# -- lazy routing ------------------------------------------------------------


def test_lazy_build_returns_streaming_source_and_cleans_up() -> None:
    df = _frame()
    router = _router(df.lazy(), exclude=("y", "g"))
    with contextlib.ExitStack() as cleanup:
        src = router.build_train_source(df.lazy(), "y", "g", cleanup=cleanup)
        assert isinstance(src, LazyArrowBatchSource)
        path = Path(src._path)
        assert path.exists()
    # ExitStack callback removed the temp Arrow file on exit
    assert not path.exists()


def test_lazy_build_rejects_array_target() -> None:
    df = _frame()
    router = _router(df.lazy(), exclude=("y", "g"))
    with contextlib.ExitStack() as cleanup, pytest.raises(TypeError, match="must be a column name"):
        router.build_train_source(df.lazy(), np.arange(len(df)), None, cleanup=cleanup)


def test_lazy_build_missing_column_raises() -> None:
    df = _frame()
    router = _router(df.lazy(), exclude=("y", "g"))
    with contextlib.ExitStack() as cleanup, pytest.raises(ValueError, match="not found in LazyFrame"):
        router.build_train_source(df.lazy(), "y", "missing_group", cleanup=cleanup)


# -- eval-set routing --------------------------------------------------------


def test_eval_source_none_returns_none() -> None:
    df = _frame()
    router = _router(df, exclude=("y", "g"))
    with contextlib.ExitStack() as cleanup:
        assert router.build_eval_source(None, group_col=None, cleanup=cleanup) is None


def test_eval_source_inherits_group_column_name() -> None:
    df = _frame()
    router = _router(df, exclude=("y", "g"))
    # eval tuple omits group; router should inherit the training group column.
    with contextlib.ExitStack() as cleanup:
        src = router.build_eval_source((df, "y"), group_col="g", cleanup=cleanup)
    assert isinstance(src, TensorDatasetSource)
    assert src.group_np is not None


def test_eval_source_explicit_group_tuple_overrides() -> None:
    df = _frame()
    router = _router(df, exclude=("y", "g"))
    explicit = np.zeros(len(df), dtype=np.int64)
    with contextlib.ExitStack() as cleanup:
        src = router.build_eval_source((df, "y", explicit), group_col=None, cleanup=cleanup)
    assert isinstance(src, TensorDatasetSource)
    assert src.group_np is not None
    assert set(np.unique(src.group_np)) == {0}


def test_eval_source_unshuffled() -> None:
    df = _frame()
    router = _router(df, exclude=("y", "g"))
    with contextlib.ExitStack() as cleanup:
        src = router.build_eval_source((df, "y"), group_col=None, cleanup=cleanup)
    assert isinstance(src, TensorDatasetSource)
    assert src.shuffle is False
