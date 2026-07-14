"""Data routing for the sklearn estimators.

This module owns the single decision the estimator used to make inline: given a
feature frame plus a target/group reference, build the right batch source.

* eager (``polars.DataFrame`` / numpy / pandas) -> :class:`TensorDatasetSource`
  (in-memory tensors, batched by the ``DataLoader``).
* lazy (``polars.LazyFrame``) -> :class:`LazyArrowBatchSource`
  (preprocessed once to a temp Arrow IPC file, then streamed).

The estimator stays focused on the sklearn API (hyperparameters, target
metadata, inference, persistence) and delegates all batch-source construction
to :class:`DataRouter`. The router depends only on a fitted preprocessor plus
two small callables (target encoder / target polars-expression), so it is
unit-testable in isolation without constructing a full estimator.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from scikit_rank.data import (
    LazyArrowBatchSource,
    TensorDatasetSource,
    materialize_lazy,
    to_numpy_1d,
    to_polars,
)

if TYPE_CHECKING:
    import contextlib
    from collections.abc import Callable

    from scikit_rank.preprocessing import TabularPreprocessor
    from scikit_rank.sklearn._types import EvalSet, GroupLike, YLike

_TARGET_COL = "__scikit_rank_y__"
_GROUP_COL = "__scikit_rank_group__"

BatchSource = TensorDatasetSource | LazyArrowBatchSource


def _pop_column(
    df: pl.DataFrame,
    ref: YLike,
    *,
    name: str,
) -> tuple[pl.DataFrame, np.ndarray | None]:
    """Resolve y/group as either a column name (dropped from ``df``) or an array."""
    if ref is None:
        return df, None
    if isinstance(ref, str):
        if ref not in df.columns:
            raise ValueError(f"{name}={ref!r} not found in X columns")
        return df.drop(ref), df[ref].to_numpy()
    arr = to_numpy_1d(ref)
    if len(arr) != len(df):
        raise ValueError(
            f"X and {name} have inconsistent lengths: {len(df)} != {len(arr)}",
        )
    return df, arr


def _require_column_name(ref: YLike, *, name: str) -> str | None:
    """Lazy path: y/group must be column names, not arrays."""
    if ref is None:
        return None
    if not isinstance(ref, str):
        raise TypeError(
            f"With a LazyFrame, {name} must be a column name (str), got {type(ref).__name__}",
        )
    return ref


class DataRouter:
    """Builds batch sources from fitted estimator state.

    Parameters
    ----------
    preprocessor:
        A *fitted* :class:`~scikit_rank.preprocessing.TabularPreprocessor`. Supplies
        the numeric/categorical column lists plus the eager (``transform``) and
        lazy (``transform_exprs``) feature pipelines.
    batch_size, chunk_rows, rng:
        Forwarded to the batch sources (mini-batch size, streaming chunk size,
        and the shared numpy RNG used for shuffling).
    encode_target:
        Maps a raw in-memory target array to the float32 array the loss sees.
        This is the estimator's ``_prepare_y`` hook (e.g. label-encoding for
        the classifier), so subclass target semantics are preserved.
    target_expr:
        The lazy counterpart of ``encode_target``: given the target column
        name, returns the polars expression that produces the float32 target.
        This is the estimator's ``_y_expr`` hook.

    """

    def __init__(
        self,
        preprocessor: TabularPreprocessor,
        *,
        batch_size: int,
        chunk_rows: int,
        rng: np.random.Generator,
        encode_target: Callable[[np.ndarray], np.ndarray],
        target_expr: Callable[[str], pl.Expr],
    ) -> None:
        self._pre = preprocessor
        self._batch_size = batch_size
        self._chunk_rows = chunk_rows
        self._rng = rng
        self._encode_target = encode_target
        self._target_expr = target_expr

    def build_train_source(
        self,
        frame: pl.DataFrame | pl.LazyFrame,
        y: YLike,
        group: GroupLike,
        *,
        cleanup: contextlib.ExitStack,
    ) -> BatchSource:
        """Build the (shuffled) training source for ``frame``."""
        return self._build(frame, y, group, shuffle=True, cleanup=cleanup)

    def build_eval_source(
        self,
        eval_set: EvalSet | None,
        *,
        group_col: str | None,
        cleanup: contextlib.ExitStack,
        require_group: bool = False,
    ) -> BatchSource | None:
        """Build the (unshuffled) validation source from an ``eval_set``.

        ``eval_set`` is one ``(X_val, y_val[, group_val])`` tuple or a list of
        such tuples (only the first is used). When the tuple omits a group and
        training used a group *column name*, the eval frame inherits that
        column if present. With ``require_group`` (a group-aware eval metric),
        the eval batches must carry a group or a clear error is raised.
        """
        if eval_set is None:
            return None
        ev = eval_set[0] if isinstance(eval_set, list) else eval_set
        x_val, y_val = ev[0], ev[1]
        val_frame = to_polars(x_val)
        if len(ev) > 2:
            group_val: GroupLike = ev[2]
        else:  # inherit the group column name if the eval frame carries it
            schema = (
                val_frame.collect_schema()
                if isinstance(val_frame, pl.LazyFrame)
                else val_frame.schema
            )
            group_val = group_col if group_col is not None and group_col in schema.names() else None
        if require_group and group_val is None:
            raise ValueError(
                "eval_metric_group_aware=True requires a group for the eval set: pass it as "
                "the 3rd element of eval_set (X_val, y_val, group_val), or train with a group "
                "column name that is also present in the eval frame.",
            )
        return self._build(val_frame, y_val, group_val, shuffle=False, cleanup=cleanup)

    # -- internals -----------------------------------------------------------

    def _build(
        self,
        frame: pl.DataFrame | pl.LazyFrame,
        y: YLike,
        group: GroupLike,
        *,
        shuffle: bool,
        cleanup: contextlib.ExitStack,
    ) -> BatchSource:
        if isinstance(frame, pl.LazyFrame):
            return self._build_lazy(frame, y, group, shuffle=shuffle, cleanup=cleanup)
        return self._build_inmem(frame, y, group, shuffle=shuffle)

    def _build_inmem(
        self,
        df: pl.DataFrame,
        y: YLike,
        group: GroupLike,
        *,
        shuffle: bool,
    ) -> TensorDatasetSource:
        df, y_arr = _pop_column(df, y, name="y")
        if y_arr is None:
            raise ValueError("y is required")
        df, group_arr = _pop_column(df, group, name="group")

        num, cat, extra = self._pre.transform(df)
        y_enc = self._encode_target(y_arr)
        group_codes: np.ndarray | None = None
        if group_arr is not None:
            _, codes = np.unique(group_arr, return_inverse=True)
            group_codes = codes.astype(np.int64)
        return TensorDatasetSource(
            num=num,
            cat=cat,
            target=y_enc,
            group=group_codes,
            batch_size=self._batch_size,
            shuffle=shuffle,
            rng=self._rng,
            extra_features=extra,
        )

    def _build_lazy(
        self,
        lf: pl.LazyFrame,
        y: YLike,
        group: GroupLike,
        *,
        shuffle: bool,
        cleanup: contextlib.ExitStack,
    ) -> LazyArrowBatchSource:
        target_col = _require_column_name(y, name="y")
        group_col = _require_column_name(group, name="group")
        if target_col is None:
            raise ValueError("y is required")

        schema_names = lf.collect_schema().names()
        for col, label in ((target_col, "y"), (group_col, "group")):
            if col is not None and col not in schema_names:
                raise ValueError(f"{label}={col!r} not found in LazyFrame columns")

        exprs = [
            *self._pre.transform_exprs(),
            self._target_expr(target_col).alias(_TARGET_COL),
        ]
        if group_col is not None:
            exprs.append(
                pl.col(group_col).rank("dense").cast(pl.Int64).alias(_GROUP_COL),
            )
        path = materialize_lazy(
            lf=lf,
            exprs=exprs,
            group_col=_GROUP_COL if group_col is not None else None,
            chunk_rows=self._chunk_rows,
        )
        cleanup.callback(lambda p=path: Path(p).exists() and Path(p).unlink())
        return LazyArrowBatchSource(
            path=path,
            num_cols=self._pre.num_cols_,
            cat_cols=self._pre.cat_cols_,
            target_col=_TARGET_COL,
            group_col=_GROUP_COL if group_col is not None else None,
            batch_size=self._batch_size,
            shuffle=shuffle,
            rng=self._rng,
            chunk_rows=self._chunk_rows,
            extra_cols=self._pre.extra_out_cols_ or None,
        )
