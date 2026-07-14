"""Data loading, categorical encoding and split-integrity helpers for ``train.py``.

Streaming split load, the categorical vocab (matching
:class:`scikit_rank.preprocessing.TabularPreprocessor`) and the BARS MD5 check. Leaf module,
library-free (imported by ``train.py`` and ``_adapters.py``).
"""
from __future__ import annotations
import hashlib
import logging
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

logger = logging.getLogger("boosting.experiment")


# --------------------------------------------------------------------------- #
# Dataset loading (streaming train LazyFrame + eager val/test)
# --------------------------------------------------------------------------- #
def load_raw_splits(
    data_dir: Path,
    max_train_rows: int | None,
    label: str,
) -> tuple[pl.LazyFrame, pl.DataFrame, np.ndarray, pl.DataFrame, np.ndarray]:
    """Load pre-split CSVs: a streaming train ``LazyFrame`` (label kept, optionally capped
    to ``max_train_rows``) + eager val/test read whole with the label dropped.

    Returns ``(train_lazy, X_val, y_val, X_test, y_test)``. Dataset-agnostic (assumes the
    three CSVs and a ``label`` column).
    """
    train_lazy = pl.scan_csv(data_dir / "train.csv")
    if max_train_rows is not None:
        train_lazy = train_lazy.head(max_train_rows)
    val_df = pl.read_csv(data_dir / "valid.csv")
    test_df = pl.read_csv(data_dir / "test.csv")
    return (
        train_lazy,
        val_df.drop(label), val_df[label].to_numpy(),
        test_df.drop(label), test_df[label].to_numpy(),
    )


# --------------------------------------------------------------------------- #
# Split integrity — MD5 vs the BARS reference hashes
# --------------------------------------------------------------------------- #
_MD5_CHUNK = 1 << 20  # 1 MiB — the splits are multi-GB, so stream them.


def _md5_of(path: Path) -> str:
    """Streaming MD5 hexdigest of a file (read in 1 MiB chunks)."""
    h = hashlib.md5()  # noqa: S324 - integrity check, not security
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_MD5_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_split_md5(expected: dict[str, str], data_dir: Path) -> None:
    """Fail fast unless every split file under ``data_dir`` matches its reference MD5.

    ``expected`` maps filename -> hash (the registry's ``Dataset.md5s``). Raises on a
    mismatch so a corrupt split can't silently train a non-reproducible model.
    """
    for name, exp_md5 in expected.items():
        path = data_dir / name
        if not path.is_file():
            raise SystemExit(f"MD5 check: missing split file {path}")
        logger.info("Verifying MD5 of %s ...", path)
        actual = _md5_of(path)
        if actual != exp_md5:
            raise SystemExit(
                f"MD5 mismatch for {name}: expected {exp_md5}, got {actual} — the split "
                f"under {data_dir} is not the BARS reference.",
            )
        logger.info("MD5 OK: %s  %s", name, actual)


# --------------------------------------------------------------------------- #
# Categorical encoding (mirrors scikit_rank.TabularPreprocessor: all uniques, 0 = OOV)
# --------------------------------------------------------------------------- #
def fit_categorical_vocabs(
    train_x: pl.DataFrame,
    cat_cols: list[str],
) -> dict[str, dict[str, int]]:
    """Build per-column value->index vocabs from train (sorted train-uniques -> ``1..N``,
    ``0`` = OOV/missing).

    No frequency filtering (BARS' ``min_categr_count: 1``): byte-for-byte the encoding of
    :class:`scikit_rank.preprocessing.TabularPreprocessor`, so GBDT and DCN share one vocab.
    """
    vocabs: dict[str, dict[str, int]] = {}
    for c in cat_cols:
        values = (
            train_x.select(pl.col(c).cast(pl.String).drop_nulls().unique().sort())
            .to_series()
            .to_list()
        )
        vocabs[c] = {v: i + 1 for i, v in enumerate(values)}
    return vocabs


def apply_categorical_vocabs(
    x: pl.DataFrame,
    vocabs: dict[str, dict[str, int]],
) -> pl.DataFrame:
    """Map categorical columns to int32 codes via the fitted vocabs (OOV -> 0)."""
    return x.with_columns(
        [
            pl.col(c)
            .cast(pl.String)
            .replace_strict(
                list(vocab.keys()),
                list(vocab.values()),
                default=0,
                return_dtype=pl.Int32,
            )
            .alias(c)
            for c, vocab in vocabs.items()
        ],
    )
