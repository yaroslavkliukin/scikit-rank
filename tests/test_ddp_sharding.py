"""DDP data-sharding regression tests.

These tests assert that, under ``accelerate launch --num_processes N``, each
rank trains on a sharded slice of the data rather than every rank iterating the
full stream. They cover all four batch-source regimes:

* in-memory pointwise  (TensorDatasetSource, integer batch_size)
* in-memory ranking    (TensorDatasetSource + GroupAwareBatchSampler)
* streaming pointwise  (LazyArrowBatchSource, dispatch)
* streaming ranking    (LazyArrowBatchSource + group-aware dispatch slicing)

The child script counts the rows each rank sees and gathers a fingerprint of
the actual rows (so we can prove sharding happened, not merely equal counts).
Running under one process must see every row exactly once; running under two
processes must shard the rows across ranks with no rank seeing the whole
dataset. The in-memory ranking path may duplicate up to one whole batch because
accelerate's ``even_batches`` padding circles the tail back to keep equal DDP
step counts.

Each child probe builds *only* the single case under test (passed as argv[2]),
so a failure in one regime cannot make the other three appear failed too.
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROBE_SCRIPT = r"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
from accelerate import Accelerator

from scikit_rank.data import LazyArrowBatchSource, TensorDatasetSource, materialize_lazy
from scikit_rank.run import _prepare_source_loader

N = 200
GROUP_SIZE = 10


def _data():
    rng = np.random.default_rng(0)
    num = rng.normal(size=(N, 2)).astype(np.float32)
    cat = np.zeros((N, 1), dtype=np.int64)
    y = (rng.normal(size=N) > 0).astype(np.float32)
    group = np.repeat(np.arange(N // GROUP_SIZE), GROUP_SIZE).astype(np.int64)
    # A unique per-row id smuggled in as the first numeric feature so we can
    # fingerprint exactly which rows a rank consumed.
    num[:, 0] = np.arange(N, dtype=np.float32)
    return num, cat, y, group


def _ids_from_loader(loader):
    ids = []
    for b in loader:
        ids.extend(b["num"][:, 0].round().to(torch.int64).cpu().tolist())
    return ids


def _inmem(acc, num, cat, y, group, *, batch_size):
    src = TensorDatasetSource(
        num, cat, y, group, batch_size=batch_size, shuffle=False,
        rng=np.random.default_rng(1),
    )
    return _ids_from_loader(_prepare_source_loader(src, acc))


def _streaming(acc, num, cat, y, group, *, batch_size, grouped):
    cols = {"f0": num[:, 0], "f1": num[:, 1], "t": y}
    exprs = [pl.col("f0"), pl.col("f1"), pl.col("t")]
    group_col = None
    if grouped:
        cols["g"] = group
        exprs.append(pl.col("g"))
        group_col = "g"
    lf = pl.DataFrame(cols).lazy()
    path = materialize_lazy(lf, exprs, group_col=group_col, chunk_rows=50)
    src = LazyArrowBatchSource(
        path, ["f0", "f1"], [], "t", group_col,
        batch_size=batch_size, shuffle=False, rng=np.random.default_rng(2),
    )
    return _ids_from_loader(_prepare_source_loader(src, acc))


def _gather(acc, ids):
    obj = [None] * acc.num_processes
    if dist.is_available() and dist.is_initialized():
        dist.all_gather_object(obj, ids)
    else:
        obj = [ids]
    return obj


def _build_case(case, acc, num, cat, y, group):
    if case == "inmem_pointwise":
        return _inmem(acc, num, cat, y, None, batch_size=20)
    if case == "inmem_ranking":
        return _inmem(acc, num, cat, y, group, batch_size=30)
    if case == "stream_pointwise":
        return _streaming(acc, num, cat, y, group, batch_size=20, grouped=False)
    if case == "stream_ranking":
        return _streaming(acc, num, cat, y, group, batch_size=30, grouped=True)
    raise ValueError(f"unknown case {case!r}")


def main() -> None:
    out_path = Path(sys.argv[1])
    case = sys.argv[2]
    acc = Accelerator(cpu=True)
    num, cat, y, group = _data()

    # Build ONLY the requested case so an error in one regime does not taint the
    # other three parametrized tests.
    ids = _build_case(case, acc, num, cat, y, group)

    result = {
        "num_processes": acc.num_processes,
        "process_index": acc.process_index,
        "shards": {case: _gather(acc, ids)},
    }
    if acc.is_main_process:
        out_path.write_text(json.dumps(result), encoding="utf-8")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
"""

_TOTAL_ROWS = 200


def _main_process_port(output: Path, case: str) -> str:
    key = f"{output}:{case}"
    return str(20000 + sum(key.encode("utf-8")) % 20000)


def _run(script: Path, output: Path, world_size: int, case: str) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "ACCELERATE_USE_CPU": "true",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "WORLD_SIZE": str(world_size),
            "PYTHONUNBUFFERED": "1",
        },
    )
    if world_size == 1:
        cmd = [sys.executable, str(script), str(output), case]
    else:
        cmd = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--multi-gpu",
            "--num_processes",
            str(world_size),
            "--num_machines",
            "1",
            "--main_process_port",
            _main_process_port(output, case),
            "--mixed_precision",
            "no",
            "--dynamo_backend",
            "no",
            str(script),
            str(output),
            case,
        ]
    completed = subprocess.run(  # noqa: S603
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    if "failed to bind" in completed.stderr and "EPERM" in completed.stderr:
        pytest.skip("torch distributed TCPStore cannot bind a local port in this environment")
    assert completed.returncode == 0, (
        "sharding probe subprocess failed\n"
        f"cmd: {' '.join(cmd)}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert output.exists(), f"probe did not write {output}"
    return json.loads(output.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def probe_script(tmp_path_factory: pytest.TempPathFactory) -> Path:
    script = tmp_path_factory.mktemp("ddp_shard") / "probe.py"
    script.write_text(_PROBE_SCRIPT, encoding="utf-8")
    return script


@pytest.mark.parametrize(
    "case",
    ["inmem_pointwise", "inmem_ranking", "stream_pointwise", "stream_ranking"],
)
def test_single_process_sees_every_row_once(
    probe_script: Path,
    tmp_path: Path,
    case: str,
) -> None:
    result = _run(probe_script, tmp_path / "single.json", world_size=1, case=case)
    assert result["num_processes"] == 1
    (shard,) = result["shards"][case]
    assert sorted(shard) == list(range(_TOTAL_ROWS))


# Maximum rows that may be shared across ranks. Pointwise/streaming divide
# evenly or dispatch disjointly. In-memory ranking is sharded by accelerate's
# variable-size BatchSamplerShard; with even_batches padding it may repeat one
# whole batch for world size 2.
_MAX_SHARED_ROWS = {
    "inmem_pointwise": 0,
    "inmem_ranking": 30,
    "stream_pointwise": 0,
    "stream_ranking": 0,
}

# Maximum rows that may go *uncovered*. accelerate >= 1.14 keeps in-memory
# ranking coverage complete and pads by repeating whole batches instead of
# dropping the tail.
_MAX_DROPPED_ROWS = {
    "inmem_pointwise": 0,
    "inmem_ranking": 0,
    "stream_pointwise": 0,
    "stream_ranking": 0,
}


@pytest.mark.parametrize(
    "case",
    ["inmem_pointwise", "inmem_ranking", "stream_pointwise", "stream_ranking"],
)
def test_two_processes_shard_data_with_bounded_overlap(
    probe_script: Path,
    tmp_path: Path,
    case: str,
) -> None:
    result = _run(probe_script, tmp_path / "multi.json", world_size=2, case=case)
    assert result["num_processes"] == 2

    shards = result["shards"][case]
    assert len(shards) == 2
    rank0, rank1 = set(shards[0]), set(shards[1])

    # Each rank must train on a strict subset (sharding actually happened, i.e.
    # no rank iterates the whole dataset the way the old non-sharded path did).
    assert 0 < len(rank0) < _TOTAL_ROWS
    assert 0 < len(rank1) < _TOTAL_ROWS

    # Shards must be disjoint except for documented even_batches padding.
    shared = rank0 & rank1
    assert len(shared) <= _MAX_SHARED_ROWS[case], (
        f"{case}: ranks share {len(shared)} rows "
        f"(> {_MAX_SHARED_ROWS[case]}): {sorted(shared)}"
    )

    # Together the ranks must cover the dataset, up to documented padding.
    covered = rank0 | rank1
    assert covered <= set(range(_TOTAL_ROWS))
    assert _TOTAL_ROWS - len(covered) <= _MAX_DROPPED_ROWS[case], (
        f"{case}: {_TOTAL_ROWS - len(covered)} rows uncovered "
        f"(> {_MAX_DROPPED_ROWS[case]})"
    )
