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

from scikit_rank import DCNClassifier


def main() -> None:
    out_path = Path(sys.argv[1])
    expected_world_size = int(sys.argv[2])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size != expected_world_size:
        raise RuntimeError(f"Expected WORLD_SIZE={expected_world_size}, got {world_size}")

    rng = np.random.default_rng(2026)
    n = 768
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    cat_effect = np.select([cat == "a", cat == "b", cat == "c"], [0.9, -0.4, 0.15])
    margin = 2.2 * x1 - 1.4 * x2 + cat_effect
    y = (margin > 0).astype(np.int64)
    X = pl.DataFrame({"x1": x1, "x2": x2, "cat": cat})

    clf = DCNClassifier(
        epochs=8,
        batch_size=192 if world_size == 1 else 96,
        hidden_units=[24, 12],
        cross_layers=1,
        dropout=0.0,
        lr=3e-3,
        accelerator_config={"cpu": True},
        random_state=11,
    ).fit(X, y)

    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(np.int64)
    result = {
        "world_size": world_size,
        "accuracy": float((pred == y).mean()),
        "proba_head": proba[:128].astype(float).tolist(),
        "history_len": len(clf.history_),
        "final_train_loss": float(clf.history_[-1]["train_loss"]),
    }
    if rank == 0:
        out_path.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
"""


def _run_training(script: Path, output: Path, world_size: int) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "ACCELERATE_USE_CPU": "true",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "WORLD_SIZE": str(world_size),
            # Avoid accidental inherited distributed env from an outer launcher.
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
            "--mixed_precision",
            "no",
            "--dynamo_backend",
            "no",
            str(script),
            str(output),
            str(world_size),
        ]
    completed = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, (
        "training subprocess failed\n"
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
def test_accelerate_two_worker_training_matches_single_worker(tmp_path: Path) -> None:
    script = tmp_path / "train_with_accelerate.py"
    script.write_text(_TRAIN_SCRIPT, encoding="utf-8")

    single = _run_training(script, tmp_path / "single.json", world_size=1)
    multi = _run_training(script, tmp_path / "multi.json", world_size=2)

    assert single["world_size"] == 1
    assert multi["world_size"] == 2
    assert single["history_len"] == 8
    assert multi["history_len"] == 8
    assert single["accuracy"] >= 0.80
    assert multi["accuracy"] >= 0.80
    assert abs(single["accuracy"] - multi["accuracy"]) <= 0.12

    single_proba = np.asarray(single["proba_head"])
    multi_proba = np.asarray(multi["proba_head"])
    assert np.mean(np.abs(single_proba - multi_proba)) <= 0.15
    assert np.corrcoef(single_proba, multi_proba)[0, 1] >= 0.80
    assert np.isfinite(single["final_train_loss"])
    assert np.isfinite(multi["final_train_loss"])
