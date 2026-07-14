"""Input validation helpers used by the sklearn estimators."""

from __future__ import annotations
import warnings
from typing import Any

import numpy as np
import polars as pl
from sklearn.exceptions import DataConversionWarning


def peek_shape(X: Any) -> tuple[int | None, int | None]:
    """Return ``(n_rows, n_cols)`` for any supported X without materialising it.

    ``n_rows`` is ``None`` for a :class:`polars.LazyFrame` (avoids a forced
    collect just to count rows; predict-time row count is not validated).
    """
    if isinstance(X, pl.LazyFrame):
        return None, len(X.collect_schema().names())
    if isinstance(X, pl.DataFrame):
        return X.height, X.width
    if hasattr(X, "shape") and hasattr(X, "columns"):  # pandas.DataFrame
        return int(X.shape[0]), int(X.shape[1])
    arr = np.asarray(X) if not isinstance(X, np.ndarray) else X
    if arr.ndim == 1:
        return int(arr.shape[0]), None
    if arr.ndim >= 2:
        return int(arr.shape[0]), int(arr.shape[1])
    return None, None


def validate_X(  # noqa: D417
    X: Any,
    *,
    expected_features: int | None = None,
    estimator_name: str = "estimator",
) -> None:
    """Reject inputs that sklearn's estimator checks expect to be rejected.

    Parameters
    ----------
    X : input array-like / frame.
    expected_features : if not ``None``, raise when the column count differs
        (used at predict time against the fitted ``n_features_in_``).

    """
    if isinstance(X, np.ndarray) and X.ndim == 1:
        raise ValueError(
            f"Expected 2D array, got 1D array instead:\narray={X}.\n"
            "Reshape your data either using array.reshape(-1, 1) if your data "
            "has a single feature or array.reshape(1, -1) if it contains a single sample.",
        )

    n_rows, n_cols = peek_shape(X)
    if n_cols == 0:
        rows = n_rows if n_rows is not None else 0
        raise ValueError(
            f"0 feature(s) (shape=({rows}, 0)) while a minimum of 1 is required.",
        )
    if expected_features is not None and n_cols is not None and n_cols != expected_features:
        raise ValueError(
            f"X has {n_cols} features, but {estimator_name} is expecting "
            f"{expected_features} features as input.",
        )


def validate_y(y: Any, *, allow_2d_column_vector: bool = True) -> np.ndarray:
    """Normalise ``y`` to a 1D numpy array and reject NaN / inf values.

    Column-vector inputs ``(n, 1)`` are squeezed with a
    :class:`sklearn.exceptions.DataConversionWarning`, matching sklearn's
    convention for estimators that take a 1D target.
    """
    if y is None:
        raise ValueError(
            "requires y to be passed, but the target y is None",
        )
    arr = y if isinstance(y, np.ndarray) else np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1 and allow_2d_column_vector:
        warnings.warn(
            "A column-vector y was passed when a 1d array was expected. "
            "Please change the shape of y to (n_samples,), for example using ravel().",
            DataConversionWarning,
            stacklevel=2,
        )
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(
            f"y should be a 1d array, got an array of shape {arr.shape} instead.",
        )
    if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
        raise ValueError("Input y contains NaN or infinity.")
    return arr
