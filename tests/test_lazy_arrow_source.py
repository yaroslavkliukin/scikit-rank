from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa

from scikit_rank.data import LazyArrowBatchSource, materialize_lazy


def _frame() -> tuple[pl.LazyFrame, np.ndarray]:
    # group sizes: 3, 1, 4, 2, 5 (intentionally uneven, unsorted input order)
    sizes = {10: 3, 11: 1, 12: 4, 13: 2, 14: 5}
    rows = []
    for g, n in sizes.items():
        for j in range(n):
            rows.append({"x1": float(g + j), "x2": float(-j), "g": g, "y": float(j)})
    np.random.default_rng(0)
    df = pl.DataFrame(rows)
    df = df.sample(fraction=1.0, shuffle=True, seed=7)  # unsorted on disk
    return df.lazy(), np.array([3, 1, 4, 2, 5])


def _exprs() -> list[pl.Expr]:
    return [pl.col("x1"), pl.col("x2"), pl.col("g"), pl.col("y")]


def test_materialize_lazy_writes_one_record_batch_per_group() -> None:
    lf, sizes = _frame()
    path = materialize_lazy(lf, _exprs(), group_col="g", chunk_rows=4)

    reader = pa.ipc.open_file(pa.memory_map(path, "r"))
    assert reader.num_record_batches == len(sizes)  # one batch per group
    # Each record batch holds exactly one group, sorted ascending by g.
    seen_groups = []
    for i in range(reader.num_record_batches):
        batch = pl.from_arrow(reader.get_record_batch(i))
        assert batch["g"].n_unique() == 1
        seen_groups.append(batch["g"][0])
        assert len(batch) == int(sizes[seen_groups[-1] - 10])
    assert seen_groups == [10, 11, 12, 13, 14]


def test_group_source_never_splits_groups_and_covers_all_rows() -> None:
    lf, _ = _frame()
    path = materialize_lazy(lf, _exprs(), group_col="g", chunk_rows=4)
    source = LazyArrowBatchSource(
        path, ["x1", "x2"], [], "y", "g",
        batch_size=4, shuffle=False, rng=np.random.default_rng(0),
    )

    total = 0
    for batch in source:
        groups = batch["group"].numpy()
        total += groups.shape[0]
        # every group present in this batch is present in full
        for g in np.unique(groups):
            assert np.sum(groups == g) == {10: 3, 11: 1, 12: 4, 13: 2, 14: 5}[int(g)]
    assert total == 15


def test_group_source_shuffles_group_order() -> None:
    lf, _ = _frame()
    path = materialize_lazy(lf, _exprs(), group_col="g", chunk_rows=4)

    def first_group(seed: int) -> int:
        source = LazyArrowBatchSource(
            path, ["x1", "x2"], [], "y", "g",
            batch_size=1, shuffle=True, rng=np.random.default_rng(seed),
        )
        batch = next(iter(source))
        return int(np.unique(batch["group"].numpy())[0])

    # With batch_size=1 each batch is a single group; across seeds the leading
    # group id should not be constant -> group order is actually shuffled.
    leads = {first_group(s) for s in range(8)}
    assert len(leads) > 1


def test_pointwise_source_uses_fixed_size_chunks_and_covers_all_rows() -> None:
    n = 25
    df = pl.DataFrame(
        {"x1": np.arange(n, dtype=np.float64), "y": np.arange(n, dtype=np.float64)},
    ).lazy()
    path = materialize_lazy(df, [pl.col("x1"), pl.col("y")], group_col=None, chunk_rows=10)

    reader = pa.ipc.open_file(pa.memory_map(path, "r"))
    assert [reader.get_record_batch(i).num_rows for i in range(reader.num_record_batches)] == [10, 10, 5]

    source = LazyArrowBatchSource(
        path, ["x1"], [], "y", None,
        batch_size=4, shuffle=False, rng=np.random.default_rng(0),
    )
    seen = np.concatenate([b["target"].numpy() for b in source])
    np.testing.assert_array_equal(np.sort(seen), np.arange(n, dtype=np.float64))
