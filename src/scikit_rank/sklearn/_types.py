"""Shared type aliases for the sklearn estimator surface.

Kept in a tiny leaf module so both :mod:`scikit_rank.sklearn.estimator` and
:mod:`scikit_rank.sklearn._data_router` can import them without a circular
dependency.
"""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

# numeric/categorical array-like for an in-memory target or group vector
ArrayLike = np.ndarray | pl.Series | Sequence[Any]
# accepted feature matrices (pandas.DataFrame is also accepted via duck typing)
XLike = np.ndarray | pl.DataFrame | pl.LazyFrame
# target / group: an array-like, a column name (str), or None
YLike = ArrayLike | str | None
GroupLike = ArrayLike | str | None
# one (X_val, y_val[, group_val]) tuple, or a list of such tuples
EvalTuple = tuple[Any, ...]
EvalSet = EvalTuple | Sequence[EvalTuple]
