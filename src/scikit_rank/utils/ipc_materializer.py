"""Streaming Arrow IPC writer.

:class:`IpcMaterializer` owns the lifecycle of a ``RecordBatchFileWriter``
so callers can append polars frames without juggling a writer handle:

* The file/writer is opened lazily on the first non-empty append, using the
  schema of that first batch.
* :meth:`ensure_schema` lets the caller force a schema-only file when no
  batches were ever appended (so readers can still open the file).
* The writer is closed on context-manager exit, including on exceptions.

This module has no scikit_rank-internal dependencies and can be reused by any
preprocessing path that needs to stream polars frames to a single IPC file.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Self

import pyarrow as pa

if TYPE_CHECKING:
    from types import TracebackType

    import polars as pl


class IpcMaterializer:
    """Lazy-open, exception-safe Arrow IPC file writer."""

    def __init__(self, path: str, *, compression: str | None = "zstd") -> None:
        """Create a materializer for ``path``.

        Parameters
        ----------
        path:
            Destination Arrow IPC file path.
        compression:
            Codec name passed to :class:`pyarrow.Codec` (e.g. ``"zstd"``,
            ``"lz4"``). ``None`` disables compression.

        """
        self._path = path
        self._compression = compression
        self._writer: pa.ipc.RecordBatchFileWriter | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None

    def path(self) -> str:
        return self._path

    def append(self, df: pl.DataFrame) -> None:
        """Append ``df`` as a single record batch.

        Empty frames are skipped once the writer is initialized. Before the
        writer is initialized, the first call (empty or not) seeds the schema.
        """
        if not len(df) and self._writer is not None:
            return
        table = df.to_arrow().combine_chunks()
        if self._writer is None:
            options = (
                pa.ipc.IpcWriteOptions(compression=pa.Codec(self._compression))
                if self._compression is not None
                else pa.ipc.IpcWriteOptions()
            )
            self._writer = pa.ipc.new_file(self._path, table.schema, options=options)
        # A combined table holds one chunk per column → exactly one record batch.
        self._writer.write_table(table)

    def ensure_schema(self, template: pl.LazyFrame) -> None:
        """Write a schema-only file when no batch has been appended yet."""
        if self._writer is None:
            self.append(template.head(0).collect())
