from __future__ import annotations

import numpy as np
import pytest
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import DataLoader

from scikit_rank.data import GroupAwareBatchSampler, TensorDatasetSource


def test_tensor_dataset_source_uses_normal_dataloader_batch_size() -> None:
    n = 10
    source = TensorDatasetSource(
        np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        np.zeros((n, 0), dtype=np.int64),
        np.arange(n, dtype=np.float32),
        None,
        batch_size=4,
        shuffle=False,
        rng=np.random.default_rng(0),
    )

    batches = list(DataLoader(source, batch_size=source.batch_size, shuffle=source.shuffle))

    assert [batch["num"].shape[0] for batch in batches] == [4, 4, 2]
    assert "group" not in batches[0]
    np.testing.assert_array_equal(batches[0]["target"].numpy(), np.arange(4, dtype=np.float32))


def test_group_aware_batch_sampler_never_splits_groups() -> None:
    group = np.array([3, 1, 1, 2, 2, 2, 4, 4, 5])
    source = TensorDatasetSource(
        np.arange(len(group), dtype=np.float32).reshape(-1, 1),
        np.zeros((len(group), 0), dtype=np.int64),
        np.ones(len(group), dtype=np.float32),
        group,
        batch_size=4,
        shuffle=False,
        rng=np.random.default_rng(0),
    )

    seen = []
    loader = DataLoader(
        source,
        batch_sampler=GroupAwareBatchSampler(
            source.group_np, source.batch_size, source.shuffle, seed=0,
        ),
    )
    for batch in loader:
        batch_groups = batch["group"].numpy()
        seen.extend(batch["num"].squeeze(1).numpy().astype(int).tolist())
        for g in np.unique(batch_groups):
            assert np.sum(group == g) == np.sum(batch_groups == g)

    assert sorted(seen) == list(range(len(group)))


def test_group_aware_batch_sampler_does_not_overfill_with_multiple_groups() -> None:
    # Matches the original 1.14-compatible sampler contract: a batch may exceed
    # batch_size only when a single group is oversized, not by combining several
    # smaller groups.
    group = np.repeat(np.arange(3), [20, 20, 10]).astype(np.int64)
    batches = list(GroupAwareBatchSampler(group, 30, shuffle=False))
    assert [len(batch) for batch in batches] == [20, 30]
    assert [np.unique(group[batch]).tolist() for batch in batches] == [[0], [1, 2]]


# --- DDP sharding contract (pure-Python, no accelerate launch) ---------------
#
# These exercise accelerate's variable-size BatchSamplerShard path that the
# in-memory ranking path relies on, without spawning a multi-process job (so
# they run everywhere, unlike the launch-based integration test in
# tests/test_ddp_sharding.py).


def _rank_batches(
    group: np.ndarray,
    batch_size: int,
    world_size: int,
    *,
    epoch: int = 0,
    **kw: object,
) -> list[list[list[int]]]:
    """Whole batches each rank would consume, in order."""
    out: list[list[list[int]]] = []
    for rank in range(world_size):
        sampler = GroupAwareBatchSampler(group, batch_size, **kw)
        sampler.set_epoch(epoch)
        shard = BatchSamplerShard(
            sampler,
            num_processes=world_size,
            process_index=rank,
            split_batches=False,
            even_batches=True,
        )
        out.append([list(batch) for batch in shard])
    return out


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_accelerate_shard_gives_equal_steps_and_bounded_padding(world_size: int) -> None:
    # 20 groups of 10 rows, batch_size 30 -> 3 groups per batch -> 7 batches.
    group = np.repeat(np.arange(20), 10).astype(np.int64)
    base_batches = list(GroupAwareBatchSampler(group, 30, shuffle=False))
    expected_padding_batches = (-len(base_batches)) % world_size

    # Requirement: every rank runs the same number of optimizer steps.
    rank_batches = _rank_batches(group, 30, world_size, shuffle=False)
    assert len({len(batches) for batches in rank_batches}) == 1

    rows = [[idx for batch in batches for idx in batch] for batches in rank_batches]

    # accelerate >= 1.14 preserves coverage and pads by circling back to the
    # first whole batches instead of dropping the tail.
    sets = [set(row_ids) for row_ids in rows]
    union = set().union(*sets)
    assert union == set(range(len(group)))
    duplicated_rows = sum(len(row_ids) for row_ids in rows) - len(union)
    assert duplicated_rows <= expected_padding_batches * 30

    # Requirement: no query group is split inside a batch. Padding may duplicate
    # a whole batch on another rank, but never slices a group by row count.
    for batches in rank_batches:
        for batch in batches:
            batch_groups = group[batch]
            for gid in np.unique(batch_groups):
                assert np.sum(group == gid) == np.sum(batch_groups == gid)


def test_set_epoch_reshuffles_while_keeping_full_coverage() -> None:
    # Standard DDP convention (as in torch's DistributedSampler and the
    # reference repo): the batch list is shuffled with seed+epoch and *then*
    # strided per rank by accelerate, so every epoch keeps full coverage while
    # the assignment rotates across epochs (broadening per-rank coverage over
    # time). even_batches padding may duplicate a whole batch.
    group = np.repeat(np.arange(20), 10).astype(np.int64)

    def shards(epoch: int) -> list[list[int]]:
        return [
            [idx for batch in batches for idx in batch]
            for batches in _rank_batches(group, 30, 2, epoch=epoch, shuffle=True, seed=7)
        ]

    e0, e1 = shards(0), shards(1)
    # Each epoch covers every row; padding is bounded by one whole batch.
    assert set(e0[0]) | set(e0[1]) == set(range(len(group)))
    assert set(e1[0]) | set(e1[1]) == set(range(len(group)))
    assert len(e0[0]) + len(e0[1]) - len(group) <= 30
    assert len(e1[0]) + len(e1[1]) - len(group) <= 30
    # ... but the per-rank assignment changes between epochs (reshuffled).
    assert e0[0] != e1[0]


def test_batch_count_not_divisible_by_world_size_circles_tail() -> None:
    # 7 batches, world size 2 -> 4 batches per rank, with the first batch
    # repeated once for even_batches padding.
    group = np.repeat(np.arange(20), 10).astype(np.int64)
    rank_batches = _rank_batches(group, 30, 2, shuffle=False)
    assert [len(batches) for batches in rank_batches] == [4, 4]
    rows = [idx for batches in rank_batches for batch in batches for idx in batch]
    assert set(rows) == set(range(len(group)))
    assert len(rows) == len(group) + 30


def test_group_larger_than_batch_size_is_its_own_batch() -> None:
    group = np.concatenate([np.zeros(100), np.repeat(np.arange(1, 7), 10)]).astype(np.int64)
    batches = list(GroupAwareBatchSampler(group, 30, shuffle=False))
    assert len(batches[0]) == 100  # the oversized group, never split


def test_fewer_batches_than_world_size_are_padded_by_accelerate() -> None:
    group = np.repeat(np.arange(2), 10).astype(np.int64)  # 2 groups -> 1 batch
    rank_batches = _rank_batches(group, 30, 4, shuffle=False)
    assert [len(batches) for batches in rank_batches] == [1, 1, 1, 1]
    assert all(batch == list(range(20)) for batches in rank_batches for batch in batches)


def test_sampler_exposes_no_public_batch_size_for_accelerate_variable_path() -> None:
    group = np.repeat(np.arange(20), 10).astype(np.int64)
    sampler = GroupAwareBatchSampler(group, 30, shuffle=False)
    assert not hasattr(sampler, "batch_size")
