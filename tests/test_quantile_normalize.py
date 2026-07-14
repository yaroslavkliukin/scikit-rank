"""Tests for the quantile->normal numeric normalization mode.

Matches ``sklearn.preprocessing.QuantileTransformer(output_distribution='normal')``
while working for both eager and lazy/streaming polars frames.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.preprocessing import QuantileTransformer

from scikit_rank.data import to_polars
from scikit_rank.preprocessing import TabularPreprocessor


def _frame(seed: int = 0, n: int = 5000) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "a": rng.lognormal(size=n),  # skewed
            "b": rng.normal(loc=10, scale=3, size=n),
            "c": rng.integers(0, 5, size=n).astype(np.float64),  # discrete/ties
            "k": rng.integers(0, 3, size=n),  # categorical
        },
    )


def test_mode_resolution() -> None:
    assert TabularPreprocessor(normalize=True)._normalize_mode == "standard"
    assert TabularPreprocessor(normalize="standard")._normalize_mode == "standard"
    assert TabularPreprocessor(normalize="quantile")._normalize_mode == "quantile"
    assert TabularPreprocessor(normalize=False)._normalize_mode is None
    assert TabularPreprocessor(normalize=None)._normalize_mode is None
    with pytest.raises(ValueError, match="normalize must be one of"):
        TabularPreprocessor(normalize="bogus")


def test_quantile_output_is_approximately_normal() -> None:
    df = _frame()
    pre = TabularPreprocessor(
        num_features=["a", "b", "c"],
        cat_features=["k"],
        normalize="quantile",
        n_quantiles=1000,
    ).fit(df)
    num, _, _ = pre.transform(df)
    # heavy skew of column "a" should be gone: mean ~0, std ~1.
    assert np.all(np.isfinite(num))
    assert abs(num[:, 0].mean()) < 0.05
    assert abs(num[:, 0].std() - 1.0) < 0.1


def test_matches_sklearn_quantile_transformer() -> None:
    df = _frame(seed=3)
    cols = ["a", "b", "c"]
    n = len(df)

    pre = TabularPreprocessor(
        num_features=cols,
        cat_features=["k"],
        normalize="quantile",
        n_quantiles=1000,
    ).fit(df)
    ours, _, _ = pre.transform(df)

    qt = QuantileTransformer(
        n_quantiles=min(1000, n),
        output_distribution="normal",
        subsample=n,  # use all rows so the reference table matches
        random_state=0,
    )
    ref = qt.fit_transform(df.select(cols).to_numpy())

    # Same monotone empirical-CDF -> inverse-normal map; allow small slack from
    # interpolation-grid and subsample differences.
    assert np.allclose(ours, ref.astype(np.float32), atol=1e-3)


def test_eager_lazy_equivalence() -> None:
    df = _frame(seed=7)
    cols = ["a", "b", "c"]
    pre = TabularPreprocessor(
        num_features=cols,
        cat_features=["k"],
        normalize="quantile",
    ).fit(df)

    eager, _, _ = pre.transform(df)
    lazy = (
        df.lazy()
        .select(pre.numeric_transform_exprs())
        .collect()
        .select(cols)
        .to_numpy()
        .astype(np.float32)
    )
    assert np.allclose(eager, lazy, atol=1e-6)


def test_handles_nulls_and_constant_columns() -> None:
    df = pl.DataFrame(
        {
            "with_nulls": [1.0, 2.0, None, 4.0, 5.0, None, 7.0, 8.0],
            "constant": [3.0] * 8,
            "k": [0, 1, 0, 1, 0, 1, 0, 1],
        },
    )
    pre = TabularPreprocessor(
        num_features=["with_nulls", "constant"],
        cat_features=["k"],
        normalize="quantile",
    ).fit(df)
    num, _, _ = pre.transform(df)
    assert np.all(np.isfinite(num))  # no inf/nan at the tails or from constants
    # A constant feature collapses to a single finite value (sklearn pins it to
    # the lower tail clip, -ppf(1 - 1e-7) ~= -5.199); the key guarantee is that
    # it stays finite and constant rather than producing inf/nan.
    assert len(np.unique(num[:, 1])) == 1


def test_quantile_fit_on_lazyframe() -> None:
    df = _frame(seed=11)
    cols = ["a", "b", "c"]
    pre = TabularPreprocessor(
        num_features=cols,
        cat_features=["k"],
        normalize="quantile",
    ).fit(df.lazy())
    num, _, _ = pre.transform(df)
    assert np.all(np.isfinite(num))
    assert set(pre.num_quantiles_) == set(cols)


@pytest.mark.parametrize("mode", ["standard", "quantile"])
def test_nan_numpy_input_stays_finite_and_matches_null(mode: str) -> None:
    # numpy inputs carry missing as float NaN (not null). The fit-side
    # aggregations must ignore NaN like null, otherwise stats/quantiles get
    # poisoned and normalization emits non-finite values. The NaN frame must
    # behave identically to the same data with null missings.
    rng = np.random.default_rng(0)
    col = rng.normal(50, 5, size=400)
    col[:120] = np.nan  # 30% missing
    other = rng.normal(size=400)

    nan_frame = to_polars(np.column_stack([col, other]))  # preserves NaN as NaN
    null_frame = pl.DataFrame(
        {
            "f0": [None if np.isnan(v) else float(v) for v in col],
            "f1": other,
        },
    )

    kw = {"num_features": ["f0", "f1"], "normalize": mode}
    nan_out = TabularPreprocessor(**kw).fit(nan_frame).transform(nan_frame)[0]
    null_out = TabularPreprocessor(**kw).fit(null_frame).transform(null_frame)[0]

    assert np.isfinite(nan_out).all()
    assert np.allclose(nan_out, null_out, atol=1e-6)
