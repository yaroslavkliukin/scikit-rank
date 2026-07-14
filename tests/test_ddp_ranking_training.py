"""End-to-end DDP regression for the in-memory ranking path.

This is the test the ``GroupAwareBatchSampler`` DDP fix is really about: a
two-process ``DCNRanker.fit(..., group=...)`` with a group-requiring listwise
loss, so every rank runs real forward/backward/optimizer steps on whole query
groups.

Why it matters: before the fix the variable-size group sampler could not pass
through ``Accelerator.prepare_data_loader`` at all, and the naive
``even_batches=False`` workaround gave ranks *different* numbers of optimizer
steps -- one rank finished the epoch while the other was still in
``backward()``, hanging the NCCL all-reduce until the watchdog killed it. The
sampler now delegates sharding to accelerate >= 1.14, whose variable-size
``BatchSamplerShard`` distributes whole batches round-robin and pads by
repeating whole batches so all ranks run the *same* number of steps. If that
invariant ever regresses, the two-process subprocess below hangs and the
``timeout`` trips this test rather than CI.
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_TRAIN_SCRIPT = r"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

from scikit_rank import DCNRanker


def main() -> None:
    out_path = Path(sys.argv[1])
    expected_world_size = int(sys.argv[2])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size != expected_world_size:
        raise RuntimeError(f"Expected WORLD_SIZE={expected_world_size}, got {world_size}")

    rng = np.random.default_rng(7)
    # 90 queries x 8 docs. An odd number of packed batches per epoch is the
    # interesting case: it forces even_batches padding, the exact spot the old
    # path produced unequal step counts and hung.
    n_q, docs = 90, 8
    qid = np.repeat(np.arange(n_q), docs)
    n = qid.size
    f0 = rng.normal(size=n)
    f1 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    # A learnable relevance signal so the model has something to fit.
    score = 1.5 * f0 - 0.8 * f1 + np.select([cat == "a", cat == "b", cat == "c"], [0.5, 0.0, -0.5])
    rel = np.clip(np.round(score - score.min()), 0, 3).astype(np.float32)
    X = pl.DataFrame({"f0": f0, "f1": f1, "cat": cat, "rel": rel, "qid": qid})
    feats = X.select(["f0", "f1", "cat"])

    model = DCNRanker(
        loss="lambdarank",
        epochs=5,
        batch_size=64 if world_size == 1 else 32,
        hidden_units=[24, 12],
        cross_layers=1,
        dropout=0.0,
        lr=3e-3,
        accelerator_config={"cpu": True},
        random_state=11,
    ).fit(X, "rel", group="qid")

    scores = model.predict(feats)
    result = {
        "world_size": world_size,
        "history_len": len(model.history_),
        "final_train_loss": float(model.history_[-1]["train_loss"]),
        "scores_head": scores[:240].astype(float).tolist(),
        "n_scores": int(scores.shape[0]),
    }
    if rank == 0:
        out_path.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
"""


def _main_process_port(output: Path, world_size: int) -> str:
    key = f"{output}:{world_size}"
    return str(20000 + sum(key.encode("utf-8")) % 20000)


def _run_training(script: Path, output: Path, world_size: int) -> dict:
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
        cmd = [sys.executable, str(script), str(output), "1"]
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
            _main_process_port(output, world_size),
            "--mixed_precision",
            "no",
            "--dynamo_backend",
            "no",
            str(script),
            str(output),
            str(world_size),
        ]
    completed = subprocess.run(  # noqa: S603
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        # A regression in equal-step sharding manifests as a hang, so the
        # timeout is the actual assertion for "ranks ran the same #steps".
        timeout=240,
        check=False,
    )
    if "failed to bind" in completed.stderr and "EPERM" in completed.stderr:
        pytest.skip("torch distributed TCPStore cannot bind a local port in this environment")
    assert completed.returncode == 0, (
        "ranking training subprocess failed\n"
        f"cmd: {' '.join(cmd)}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert output.exists(), f"training subprocess did not write {output}"
    return json.loads(output.read_text(encoding="utf-8"))


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="torch 2.12 CPU DDP segfaults in DistributedDataParallel.__init__ on macOS",
)
def test_two_worker_ranking_training_matches_single_worker(tmp_path: Path) -> None:
    script = tmp_path / "train_ranker.py"
    script.write_text(_TRAIN_SCRIPT, encoding="utf-8")

    single = _run_training(script, tmp_path / "single.json", world_size=1)
    multi = _run_training(script, tmp_path / "multi.json", world_size=2)

    # Both runs completed every epoch -- the two-process run did not hang on the
    # all-reduce, which is only possible if both ranks ran the same #steps.
    assert single["world_size"] == 1
    assert multi["world_size"] == 2
    assert single["history_len"] == 5
    assert multi["history_len"] == 5
    assert np.isfinite(single["final_train_loss"])
    assert np.isfinite(multi["final_train_loss"])
    assert single["n_scores"] == multi["n_scores"]

    # Both learned the same ranking signal (ranking cares about order, so
    # compare with rank correlation rather than absolute score values).
    single_scores = np.asarray(single["scores_head"])
    multi_scores = np.asarray(multi["scores_head"])
    single_rank = single_scores.argsort().argsort()
    multi_rank = multi_scores.argsort().argsort()
    assert np.corrcoef(single_rank, multi_rank)[0, 1] >= 0.7
