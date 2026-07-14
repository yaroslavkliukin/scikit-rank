"""DDP correctness for the eval-LOSS monitor (``eval_metric=None``).

When no custom eval metric is set, the monitor is the running eval loss
(``Trainer._update_loss``). Under ``accelerate launch --num_processes N`` the eval
set is sharded across ranks, so a per-rank shard mean would make ``BestStateSaver`` /
``attach_early_stopping`` / ``attach_lr_scheduler`` act on divergent numbers -- ranks
could stop / decay LR on different epochs (deadlock / param divergence). So the loss
MUST be reduced to a single global value before those handlers read it.

``Trainer._finalize_eval_loss`` all-reduces the running sum + batch count ONCE at eval
COMPLETED (robust to unequal per-rank batch counts, unlike a per-iteration collective).
This probe drives the *real* ``Trainer`` eval engine with NO custom metric over a
dataset whose equal-size batches are sharded with UNEQUAL per-rank batch counts (rank 0
gets 2 low-value batches, rank 1 gets 3 high-value batches), so each rank's shard mean
differs from the global mean. Running under one process (the global reference) and under
two processes must report the SAME loss value.
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

import torch
import torch.distributed as dist
from accelerate import Accelerator

from scikit_rank.train.trainer import Trainer

# 30 rows, values 0..29, split into 5 equal batches of 6. Per-batch loss = mean of the
# batch values, so with equal batch sizes the mean-of-batch-means equals the global mean
# (14.5) regardless of how batches are distributed across ranks.
N = 30
BATCH = 6


class LossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))  # unused; keeps the optimizer happy

    def forward(self, batch: dict) -> dict:
        x = batch["num"][:, 0]
        return {"logits": x, "loss": x.mean()}


def _all_batches():
    return [
        {"num": torch.arange(s, s + BATCH, dtype=torch.float32).reshape(-1, 1)}
        for s in range(0, N, BATCH)
    ]


def _rank_batches(rank, world):
    batches = _all_batches()
    if world == 1:
        return batches
    # world == 2: rank 0 gets the 2 low-value batches (shard mean 5.5), rank 1 the 3
    # high-value batches (shard mean 20.5) -> unequal batch counts AND per-rank means
    # that differ from the global 14.5, so a per-shard (un-reduced) monitor is caught.
    return batches[:2] if rank == 0 else batches[2:]


def main() -> None:
    out_path = Path(sys.argv[1])
    acc = Accelerator(cpu=True)
    batches = _rank_batches(acc.process_index, acc.num_processes)

    model = LossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer = Trainer(model, optimizer, acc)

    trainer.engines["eval"].run(batches)
    value = float(trainer.engines["eval"].state.metrics["loss"])

    if acc.is_main_process:
        out_path.write_text(
            json.dumps({"num_processes": acc.num_processes, "loss": value}),
            encoding="utf-8",
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
"""

# Global reference: mean of all 30 values 0..29.
_EXPECTED = float(sum(range(30)) / 30)  # 14.5


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
        "loss-reduce probe subprocess failed\n"
        f"cmd: {' '.join(cmd)}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert output.exists(), f"probe did not write {output}"
    return json.loads(output.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def probe_script(tmp_path_factory: pytest.TempPathFactory) -> Path:
    script = tmp_path_factory.mktemp("ddp_loss") / "probe.py"
    script.write_text(_PROBE_SCRIPT, encoding="utf-8")
    return script


def test_single_process_loss_is_global(probe_script: Path, tmp_path: Path) -> None:
    result = _run(probe_script, tmp_path / "ws1.json", world_size=1)
    assert result["num_processes"] == 1
    assert result["loss"] == pytest.approx(_EXPECTED)


def test_two_process_loss_equals_global(probe_script: Path, tmp_path: Path) -> None:
    # The eval loss must be reduced to the global value: two sharded processes report
    # the same number as the single-process reference. A regression to per-shard
    # monitoring would report only the main rank's shard mean (5.5, not 14.5).
    result = _run(probe_script, tmp_path / "ws2.json", world_size=2)
    assert result["num_processes"] == 2
    assert result["loss"] == pytest.approx(_EXPECTED)
