"""Input adapters and batch sources.

pandas / numpy / polars / polars.LazyFrame inputs are all converted to a polars
frame, the single internal representation. Batch sources hide the difference
between in-memory tensors and lazily streamed parquet chunks from the trainer.
"""

import tempfile
from collections.abc import Iterator
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler

from scikit_rank.utils import IpcMaterializer


class BatchColumns(NamedTuple):
    """Decoded numpy columns for a batch: features plus optional target/group.

    ``extra`` holds additional named streams (e.g. the hashed ``"multihash"``
    columns or dense external embeddings) keyed by the model input-layer name
    they feed.
    """

    num: np.ndarray
    cat: np.ndarray
    target: np.ndarray | None
    group: np.ndarray | None
    extra: dict[str, np.ndarray]


def to_polars(X: Any) -> pl.DataFrame | pl.LazyFrame:
    """Convert any supported input to a polars (Lazy)Frame.

    Accepts polars (Lazy)Frame, pandas DataFrame, numpy ndarray, and any
    array-like (lists, sklearn's ``_NotAnArray`` wrapper, etc.) which is
    first coerced through :func:`numpy.asarray`. Complex dtypes are rejected
    explicitly; object dtypes are routed via pandas so column-wise mixed
    types are preserved.
    """
    if isinstance(X, (pl.DataFrame, pl.LazyFrame)):
        return X
    if isinstance(X, pd.DataFrame):
        return pl.from_pandas(X)
    # Detect scipy.sparse without importing scipy at module import time.
    if type(X).__module__.startswith("scipy.sparse"):
        raise TypeError(
            "sparse input is not supported by this estimator; "
            "densify with `.toarray()` or wrap your sparse data in a DataFrame.",
        )
    # Coerce array-likes (Python lists, sklearn's _NotAnArray, etc.) to ndarray.
    if not isinstance(X, np.ndarray):
        try:
            X = np.asarray(X)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Unsupported input type {type(X)!r}. "
                "Use numpy.ndarray, pandas.DataFrame, polars.DataFrame or polars.LazyFrame.",
            ) from e
    if np.iscomplexobj(X):
        raise ValueError("Complex data not supported")
    if X.ndim == 1:
        raise ValueError(
            f"Expected 2D array, got 1D array instead:\narray={X}.\n"
            "Reshape your data either using array.reshape(-1, 1) if your data "
            "has a single feature or array.reshape(1, -1) if it contains a single sample.",
        )
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D")
    if X.dtype == object:
        # Route object arrays through pandas so per-column dtype inference
        # handles mixed string/numeric/None cells correctly.
        return pl.from_pandas(
            pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]),
        )
    return pl.DataFrame(X, schema=[f"f{i}" for i in range(X.shape[1])])


def to_numpy_1d(y: Any) -> np.ndarray:
    """Convert a target / group vector to a 1D numpy array."""
    if isinstance(y, pl.Series):
        return y.to_numpy()
    if isinstance(y, (pd.DataFrame, pd.Series)):
        return np.asarray(y)
    arr = np.asarray(y)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


class GroupAwareBatchSampler(Sampler[list[int]]):
    """Batch sampler that never splits rows from the same group.

    Whole-group batches are packed once, deterministically (groups sorted by id),
    so the batch list is identical on every rank. With accelerate >= 1.14 the
    normal training path delegates DDP sharding to
    ``Accelerator.prepare_data_loader``, whose variable-size
    ``BatchSamplerShard`` distributes whole batches round-robin and pads by
    repeating whole batches so every rank yields the same number of batches.

    The sampler deliberately exposes no public ``batch_size`` attribute so that
    accelerate's ``BatchSamplerShard`` takes its variable-size code path.

    With ``shuffle`` the *order* of the (fixed) batches is permuted each epoch by
    a deterministic ``numpy.random.default_rng(seed + epoch)``. The seed is shared
    across ranks (see :func:`scikit_rank.run._shared_shuffle_seed`) so every rank
    permutes identically and the round-robin shards stay disjoint. Call
    :meth:`set_epoch` before each epoch (Accelerate's ``DataLoaderShard`` does
    this automatically).
    """

    def __init__(
        self,
        group: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        *,
        seed: int = 0,
    ) -> None:
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

        order = np.argsort(group, kind="stable")
        sorted_group = group[order]
        starts = np.flatnonzero(
            np.concatenate(([True], sorted_group[1:] != sorted_group[:-1])),
        )
        ends = np.concatenate((starts[1:], [len(sorted_group)]))
        self._batches: list[list[int]] = []
        batch: list[int] = []
        for start, end in zip(starts, ends, strict=True):
            group_indices = order[start:end].tolist()
            if batch and len(batch) + len(group_indices) > batch_size:
                self._batches.append(batch)
                batch = []
            batch.extend(group_indices)
            if len(batch) >= batch_size:
                self._batches.append(batch)
                batch = []
        if batch:
            self._batches.append(batch)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._batches
        if self._shuffle:
            order = np.random.default_rng(self._seed + self._epoch).permutation(len(batches))
            batches = [batches[i] for i in order]
        yield from batches

    def __len__(self) -> int:
        return len(self._batches)


class TensorDatasetSource(Dataset[dict[str, torch.Tensor]]):
    """In-memory row-level dataset batched by PyTorch ``DataLoader``."""

    def __init__(
        self,
        num: np.ndarray,
        cat: np.ndarray,
        target: np.ndarray | None,
        group: np.ndarray | None,
        batch_size: int,
        shuffle: bool,
        rng: np.random.Generator,
        extra_features: dict[str, np.ndarray] | None = None,
    ) -> None:
        self._num = torch.from_numpy(num)
        self._cat = torch.from_numpy(cat)
        self._target = None if target is None else torch.from_numpy(target)
        self._group = None if group is None else torch.from_numpy(group)
        self._group_np = None if group is None else np.asarray(group)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._rng = rng
        self._extra_features = {
            name: torch.from_numpy(np.array(values, copy=True))
            for name, values in (extra_features or {}).items()
        }
        for name, values in self._extra_features.items():
            if values.size(0) != self._num.size(0):
                raise ValueError(
                    f"extra feature {name!r} has {values.size(0)} rows; "
                    f"expected {self._num.size(0)}",
                )

    @property
    def group_np(self) -> np.ndarray | None:
        return self._group_np

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def __len__(self) -> int:
        return self._num.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = {"num": self._num[idx], "cat": self._cat[idx]}
        row.update({name: values[idx] for name, values in self._extra_features.items()})
        if self._target is not None:
            row["target"] = self._target[idx]
        if self._group is not None:
            row["group"] = self._group[idx]
        return row


class LazyArrowBatchSource(IterableDataset[dict[str, torch.Tensor]]):
    """Streams batches from a preprocessed Arrow IPC file (see ``materialize_lazy``).

    Each record batch is the unit of random access: the IPC file footer stores
    every batch's offset, so ``RecordBatchFileReader.get_record_batch(i)`` is an
    O(1) seek with no scanning and no per-row boundary probing.

    * Group mode (``group_col`` set): one record batch == one (sorted) query
      group. Group order is shuffled by permuting record-batch indices, and
      consecutive groups are packed into mini-batches up to ``batch_size`` rows
      without ever splitting a group, so pairwise / listwise losses always see
      whole groups.
    * Pointwise mode: one record batch == one fixed-size chunk. Chunk order is
      shuffled by permuting indices, with row-level shuffling inside each chunk.
    """

    def __init__(
        self,
        path: str,
        num_cols: list[str],
        cat_cols: list[str],
        target_col: str | None,
        group_col: str | None,
        batch_size: int,
        shuffle: bool,
        rng: np.random.Generator,
        chunk_rows: int = 100_000,
        extra_cols: dict[str, list[str]] | None = None,
    ) -> None:
        self._path = path
        self._num_cols = num_cols
        self._cat_cols = cat_cols
        self._target_col = target_col
        self._group_col = group_col
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._rng = rng
        self._chunk_rows = chunk_rows
        # name -> ordered materialized column names of an extra model input stream
        self._extra_cols = extra_cols or {}

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        reader = pa.ipc.open_file(pa.memory_map(self._path, "r"))
        order = np.arange(reader.num_record_batches)
        if self._shuffle:
            order = self._rng.permutation(order)

        if self._group_col is None:
            for i in order:
                chunk = pl.from_arrow(reader.get_record_batch(int(i)))
                yield from self._pointwise_batches(chunk)
            return

        # Group mode: each record batch is one whole group. Pack consecutive
        # groups (in shuffled order) up to batch_size rows; a group larger than
        # batch_size becomes its own batch. Groups are never split.
        pending = []
        pending_rows = 0
        for i in order:
            group_df = pl.from_arrow(reader.get_record_batch(int(i)))
            if pending and pending_rows + len(group_df) > self._batch_size:
                yield self._make_batch(pl.concat(pending))
                pending, pending_rows = [], 0
            pending.append(group_df)
            pending_rows += len(group_df)
        if pending:
            yield self._make_batch(pl.concat(pending))

    def _pointwise_batches(
        self,
        chunk: pl.DataFrame,
    ) -> Iterator[dict[str, torch.Tensor]]:
        cols = self._columns(chunk)
        source = TensorDatasetSource(
            num=cols.num,
            cat=cols.cat,
            target=cols.target,
            group=cols.group,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            rng=self._rng,
            extra_features=cols.extra,
        )
        generator = torch.Generator()
        generator.manual_seed(int(self._rng.integers(0, 2**63 - 1)))
        yield from DataLoader(
            source,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            generator=generator,
        )

    def _make_batch(self, df: pl.DataFrame) -> dict[str, torch.Tensor]:
        cols = self._columns(df)
        batch = {"num": torch.from_numpy(cols.num), "cat": torch.from_numpy(cols.cat)}
        batch.update({name: torch.from_numpy(arr) for name, arr in cols.extra.items()})
        if cols.target is not None:
            batch["target"] = torch.from_numpy(cols.target)
        if cols.group is not None:
            batch["group"] = torch.from_numpy(cols.group)
        return batch

    def _columns(self, df: pl.DataFrame) -> BatchColumns:
        num = (
            df.select(self._num_cols).to_numpy().astype(np.float32)
            if self._num_cols
            else np.zeros((len(df), 0), dtype=np.float32)
        )
        cat = (
            df.select(self._cat_cols).to_numpy().astype(np.int64)
            if self._cat_cols
            else np.zeros((len(df), 0), dtype=np.int64)
        )
        target = df[self._target_col].to_numpy(writable=True) if self._target_col else None
        group = df[self._group_col].to_numpy(writable=True) if self._group_col else None
        extra = {}
        for name, cols in self._extra_cols.items():
            stream = df.select(cols)
            dtype = np.int64 if all(dtype.is_integer() for dtype in stream.dtypes) else np.float32
            extra[name] = stream.to_numpy().astype(dtype)
        return BatchColumns(num, cat, target, group, extra)


def materialize_lazy(
    lf: pl.LazyFrame,
    exprs: list[pl.Expr],
    group_col: str | None = None,
    chunk_rows: int = 100_000,
    *,
    compression: str | None = "zstd",
) -> str:
    """Transform a LazyFrame and stream it to a temp Arrow IPC file; return the path.

    The frame is read in ``chunk_rows`` slices and written to Arrow IPC in a
    single sequential pass so :class:`LazyArrowBatchSource` can random-access
    batches:

    * Pointwise mode: rows are written as fixed-size ``chunk_rows`` batches.
    * Group mode (``group_col`` set): rows are sorted by group and written as
      one record batch per group.

    ``compression`` is forwarded to :class:`~scikit_rank.utils.IpcMaterializer`
    (``None`` disables compression).
    """
    out = lf.select(exprs)
    if group_col is not None:
        out = out.sort(group_col)

    with (
        tempfile.NamedTemporaryFile(
            suffix=".arrow",
            prefix="scikit_rank_",
            delete=False,
        ) as tmp,
        IpcMaterializer(tmp.name, compression=compression) as writer,
    ):
        if group_col is None:
            _stream_pointwise(out, writer, chunk_rows)
        else:
            _stream_grouped(out, writer, group_col, chunk_rows)
        writer.ensure_schema(out)  # empty input → schema-only IPC file
    return tmp.name


def _stream_pointwise(
    out: pl.LazyFrame,
    writer: IpcMaterializer,
    chunk_rows: int,
) -> None:
    n_rows = out.select(pl.len()).collect().item()
    for start in range(0, n_rows, chunk_rows):
        writer.append(out.slice(start, chunk_rows).collect())


def _stream_grouped(
    out: pl.LazyFrame,
    writer: IpcMaterializer,
    group_col: str,
    chunk_rows: int,
) -> None:
    """Emit one record batch per group; trailing group is carried forward."""
    n_rows = out.select(pl.len()).collect().item()
    carry: pl.DataFrame | None = None
    for start in range(0, n_rows, chunk_rows):
        chunk = out.slice(start, chunk_rows).collect()
        buf = pl.concat([carry, chunk]) if carry is not None else chunk
        parts = buf.partition_by(group_col, maintain_order=True)
        carry = parts.pop() if parts else None
        for part in parts:
            writer.append(part)
    if carry is not None:
        writer.append(carry)
