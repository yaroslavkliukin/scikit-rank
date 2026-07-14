"""DDP correctness for a group-aware eval metric.

``attach_metric`` (``scikit_rank.train.options``) monitors the eval metric for
best-checkpoint / early-stopping / LR decisions. Under ``accelerate launch
--num_processes N`` the eval set is sharded across ranks, so a group-aware
ranking metric MUST be gathered to a single global value before scoring --
otherwise each rank scores only its own shard and the main process reports a
partial, wrong number.

This probe drives the *real* ``Trainer`` eval engine with ``attach_metric(...,
group_aware=True)`` over a dataset whose groups (of unequal size) are sharded
whole across ranks -- rank row counts and batch counts differ, exercising the
object-gather path that tolerates ragged / uneven per-rank tensors. The metric
returns a value that depends on the *whole* global set (sum of scores + number
of groups), so a regression to per-shard computation changes the number.

Running under one process (the global reference) and under two processes must
report the SAME metric value.
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
import torch
import torch.distributed as dist
from accelerate import Accelerator

from scikit_rank.train.options import attach_metric
from scikit_rank.train.trainer import Trainer

# 7 groups of unequal size -> 35 rows. Score = global row index, so the sum of
# scores over the whole set is a fixed global fingerprint (595). Sharding by
# group id % world keeps whole groups on one rank and gives ranks unequal row
# AND batch counts (world=2: 4 groups vs 3).
GROUP_SIZES = [2, 3, 4, 5, 6, 7, 8]


def _global_data():
    groups = np.concatenate(
        [np.full(size, i, dtype=np.int64) for i, size in enumerate(GROUP_SIZES)],
    )
    scores = np.arange(groups.size, dtype=np.float32)
    targets = (scores % 2).astype(np.float32)
    return scores, targets, groups


class PassThrough(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))  # unused; keeps the optimizer happy

    def forward(self, batch: dict) -> dict:
        return {"logits": batch["num"][:, 0]}


def _rank_batches(scores, targets, groups, rank, world):
    # Assign whole group `g` to rank `g % world`; one batch per assigned group.
    batches = []
    for g in np.unique(groups):
        if int(g) % world != rank:
            continue
        mask = groups == g
        batches.append(
            {
                "num": torch.from_numpy(scores[mask]).reshape(-1, 1),
                "target": torch.from_numpy(targets[mask]),
                "group": torch.from_numpy(groups[mask]),
            },
        )
    return batches


def _metric(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> float:
    # Depends on the WHOLE global set: sum of scores + a large multiple of the
    # group count. A per-shard computation would change both terms.
    return float(y_pred.sum()) + 1e6 * float(np.unique(group).size)


def main() -> None:
    out_path = Path(sys.argv[1])
    acc = Accelerator(cpu=True)
    scores, targets, groups = _global_data()
    batches = _rank_batches(scores, targets, groups, acc.process_index, acc.num_processes)

    model = PassThrough()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer = Trainer(model, optimizer, acc)
    attach_metric(trainer, acc, "gauc", _metric, group_aware=True)

    trainer.engines["eval"].run(batches)
    value = float(trainer.engines["eval"].state.metrics["gauc"])

    if acc.is_main_process:
        out_path.write_text(
            json.dumps({"num_processes": acc.num_processes, "gauc": value}),
            encoding="utf-8",
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
"""

# Global reference: sum(range(35)) + 1e6 * 7 groups.
_EXPECTED = float(sum(range(sum([2, 3, 4, 5, 6, 7, 8])))) + 1e6 * 7


def _main_process_port(output: Path) -> str:
    return str(20000 + sum(str(output).encode("utf-8")) % 20000)


def _run(script: Path, output: Path, world_size: int) -> dict:
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
        cmd = [sys.executable, str(script), str(output)]
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
            _main_process_port(output),
            "--mixed_precision",
            "no",
            "--dynamo_backend",
            "no",
            str(script),
            str(output),
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
        "metric-gather probe subprocess failed\n"
        f"cmd: {' '.join(cmd)}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert output.exists(), f"probe did not write {output}"
    return json.loads(output.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def probe_script(tmp_path_factory: pytest.TempPathFactory) -> Path:
    script = tmp_path_factory.mktemp("ddp_metric") / "probe.py"
    script.write_text(_PROBE_SCRIPT, encoding="utf-8")
    return script


def test_single_process_metric_is_global(probe_script: Path, tmp_path: Path) -> None:
    result = _run(probe_script, tmp_path / "ws1.json", world_size=1)
    assert result["num_processes"] == 1
    assert result["gauc"] == pytest.approx(_EXPECTED)


def test_two_process_metric_equals_global(probe_script: Path, tmp_path: Path) -> None:
    # The group-aware metric must be gathered to the global value: two sharded
    # processes report the same number as the single-process reference. A
    # regression to per-shard scoring would report only the main rank's shard.
    result = _run(probe_script, tmp_path / "ws2.json", world_size=2)
    assert result["num_processes"] == 2
    assert result["gauc"] == pytest.approx(_EXPECTED)


# The probe above feeds manually-built disjoint batches, so it never exercises the
# real in-memory ranking eval loader (GroupAwareBatchSampler + prepare_data_loader
# with even_batches). This second probe drives that loader end-to-end: even_batches
# repeats a whole group batch to equalize per-rank counts, which the object-gather
# would double-count without dedup-by-group-id.
_RANKING_PROBE_SCRIPT = r"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator

from scikit_rank.data import TensorDatasetSource
from scikit_rank.run import _prepare_source_loader
from scikit_rank.train.options import attach_metric
from scikit_rank.train.trainer import Trainer

# 5 groups of unequal size -> 20 rows. batch_size=1 forces one whole group per
# batch, so there are 5 batches; under 2 ranks (5 % 2 != 0) even_batches repeats a
# whole group batch to equalize counts -> that group is double-counted without
# dedup. Score = global row index, so the sum of scores over the whole set is a
# fixed fingerprint (190) that a duplicated group would inflate.
GROUP_SIZES = [2, 3, 4, 5, 6]


class PassThrough(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))  # unused; keeps the optimizer happy

    def forward(self, batch: dict) -> dict:
        return {"logits": batch["num"][:, 0]}


def _sum_metric(y_true, y_pred, group) -> float:
    # Sensitive to row multiplicity: a duplicated group inflates the sum.
    return float(y_pred.sum())


def main() -> None:
    out_path = Path(sys.argv[1])
    acc = Accelerator(cpu=True)

    groups = np.concatenate(
        [np.full(size, i, dtype=np.int64) for i, size in enumerate(GROUP_SIZES)],
    )
    scores = np.arange(groups.size, dtype=np.float32).reshape(-1, 1)
    cat = np.zeros((groups.size, 1), dtype=np.int64)
    target = (np.arange(groups.size) % 2).astype(np.float32)

    source = TensorDatasetSource(
        num=scores,
        cat=cat,
        target=target,
        group=groups,
        batch_size=1,  # one whole group per batch
        shuffle=False,
        rng=np.random.default_rng(0),
    )
    loader = _prepare_source_loader(source, acc)

    model = PassThrough()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer = Trainer(model, optimizer, acc)
    attach_metric(trainer, acc, "gauc", _sum_metric, group_aware=True)

    trainer.engines["eval"].run(loader)
    value = float(trainer.engines["eval"].state.metrics["gauc"])

    if acc.is_main_process:
        out_path.write_text(
            json.dumps({"num_processes": acc.num_processes, "gauc": value}),
            encoding="utf-8",
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
"""

_RANKING_EXPECTED = float(sum(range(sum([2, 3, 4, 5, 6]))))  # sum(range(20)) = 190


@pytest.fixture(scope="module")
def ranking_probe_script(tmp_path_factory: pytest.TempPathFactory) -> Path:
    script = tmp_path_factory.mktemp("ddp_metric_rank") / "probe.py"
    script.write_text(_RANKING_PROBE_SCRIPT, encoding="utf-8")
    return script


def test_single_process_ranking_metric_is_global(
    ranking_probe_script: Path,
    tmp_path: Path,
) -> None:
    result = _run(ranking_probe_script, tmp_path / "rank_ws1.json", world_size=1)
    assert result["num_processes"] == 1
    assert result["gauc"] == pytest.approx(_RANKING_EXPECTED)


def test_two_process_ranking_metric_no_double_count(
    ranking_probe_script: Path,
    tmp_path: Path,
) -> None:
    # even_batches repeats a whole group batch across ranks; without dedup-by-group-id
    # the object-gathered set counts that group twice and the sum inflates. The fix
    # must reproduce the single-process global value.
    result = _run(ranking_probe_script, tmp_path / "rank_ws2.json", world_size=2)
    assert result["num_processes"] == 2
    assert result["gauc"] == pytest.approx(_RANKING_EXPECTED)
