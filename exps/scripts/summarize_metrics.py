r"""Summarize per-run test metrics from a folder of ``metrics.json`` files.

Reads each ``<subfolder>/metrics.json`` under the given directory, pulls ``test_auc`` /
``test_log_loss``, and prints one row per run plus trailing ``Avg`` / ``Std`` rows.

Usage
-----
    uv run python exps/scripts/summarize_metrics.py \\
        artifacts/criteo_x1/criteo_DCNv2_bars_parity_5seeds
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import polars as pl


def collect_metrics(root: Path) -> pl.DataFrame:
    """Return a DataFrame of (Runs, AUC, logloss), one row per subfolder."""
    rows: list[dict[str, object]] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        metrics_path = sub / "metrics.json"
        if not metrics_path.is_file():
            msg = f"missing metrics.json in {sub}"
            raise FileNotFoundError(msg)
        metrics = json.loads(metrics_path.read_text())
        rows.append(
            {
                "Runs": sub.name,
                "AUC": metrics["test_auc"],
                "logloss": metrics["test_log_loss"],
            },
        )
    if not rows:
        msg = f"no subfolder with metrics.json found under {root}"
        raise SystemExit(msg)
    return pl.DataFrame(rows)


def with_summary_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Append Avg / Std rows (Std formatted with a ± prefix) to ``df``."""
    avg = df.select(pl.col("AUC", "logloss").mean())
    std = df.select(pl.col("AUC", "logloss").std())
    summary = pl.DataFrame(
        {
            "Runs": ["Avg", "Std"],
            "AUC": [
                f"{avg['AUC'][0]:.7f}",
                f"±{std['AUC'][0]:.8f}",
            ],
            "logloss": [
                f"{avg['logloss'][0]:.7f}",
                f"±{std['logloss'][0]:.8f}",
            ],
        },
    )
    body = df.with_columns(
        pl.col("AUC").map_elements(lambda v: f"{v:.7f}", return_dtype=pl.String),
        pl.col("logloss").map_elements(lambda v: f"{v:.7f}", return_dtype=pl.String),
    )
    return pl.concat([body, summary])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="directory holding per-run subfolders")
    args = parser.parse_args()

    with_summary_rows(collect_metrics(args.folder))
    with pl.Config(tbl_rows=-1, tbl_hide_dataframe_shape=True):
        pass


if __name__ == "__main__":
    main()
