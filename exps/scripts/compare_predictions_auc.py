r"""Statistical significance of the AUC difference between two prediction files.

Runs DeLong's paired test (Sun & Xu 2014, in :mod:`compare_auc_delong_xu`) on two
prediction parquets (columns: ``y_true`` 0/1, ``proba``). The test is *paired* — both
files must share an identical ``y_true``.

Usage
-----
    uv run python exps/scripts/compare_predictions_auc.py \\
        predictions/criteo_x1/scikit_rank_parity.pq predictions/criteo_x1/scikit_rank_num.pq
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path

import compare_auc_delong_xu
import numpy as np
import polars as pl
import scipy.stats
import sklearn.metrics

ALPHA = 0.05


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, proba) as numpy arrays for a single prediction parquet."""
    df = pl.read_parquet(path, columns=["y_true", "proba"])
    y_true = df["y_true"].to_numpy().astype(np.int64)
    proba = df["proba"].to_numpy().astype(np.float64)

    if not np.isfinite(proba).all():
        msg = f"proba in {path} contains non-finite values (NaN/inf)"
        raise ValueError(msg)
    unique = np.unique(y_true)
    if not np.array_equal(unique, [0, 1]):
        msg = (
            f"y_true in {path} must contain exactly the two labels {{0, 1}} "
            f"(both classes present); got {unique.tolist()}"
        )
        raise ValueError(msg)
    return y_true, proba


def compare(path_a: Path, path_b: Path) -> None:
    """Print AUCs and the DeLong p-value for two prediction files."""
    y_true, proba_a = load_predictions(path_a)
    y_true_b, proba_b = load_predictions(path_b)

    if not np.array_equal(y_true, y_true_b):
        msg = (
            f"y_true in {path_b} differs from {path_a}; "
            "the DeLong test requires an identical ground truth across files"
        )
        raise ValueError(msg)


    auc_a = sklearn.metrics.roc_auc_score(y_true, proba_a)
    auc_b = sklearn.metrics.roc_auc_score(y_true, proba_b)

    name_a, name_b = path_a.stem, path_b.stem
    max(len(name_a), len(name_b))

    # One DeLong pass yields both the p-value and the AUC covariance (reused for the CI).
    # fastDeLong wants positives first; aucs/cov are ordered [a, b].
    order, label_1_count = compare_auc_delong_xu.compute_ground_truth_statistics(y_true)
    preds_sorted = np.vstack((proba_a, proba_b))[:, order]
    with np.errstate(invalid="ignore", divide="ignore"):
        aucs, cov = compare_auc_delong_xu.fastDeLong(preds_sorted, label_1_count)
        # calc_pvalue returns log10(p); 0/0 -> nan when Var(ΔAUC)==0.
        log10_p = compare_auc_delong_xu.calc_pvalue(aucs, cov).item()

    delta = auc_a - auc_b
    # Var(ΔAUC) = Var(a) + Var(b) - 2 Cov(a, b); max(., 0) guards a tiny negative fp variance.
    contrast = np.array([[1.0, -1.0]])
    var_delta = (contrast @ cov @ contrast.T).item()
    se_delta = math.sqrt(max(var_delta, 0.0))
    z = scipy.stats.norm.ppf(1.0 - ALPHA / 2.0)
    _ci_lo, _ci_hi = delta - z * se_delta, delta + z * se_delta
    f"{round((1.0 - ALPHA) * 100)}"

    # Keep the decision on the log scale: at large n, 10**log10_p underflows to 0.0.
    if math.isnan(log10_p):
        # Var(ΔAUC) == 0: indistinguishable scores, no evidence of a difference.
        return
    10**log10_p


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file_a", type=Path, help="first prediction parquet")
    parser.add_argument("file_b", type=Path, help="second prediction parquet")
    args = parser.parse_args()

    compare(args.file_a, args.file_b)


if __name__ == "__main__":
    main()
