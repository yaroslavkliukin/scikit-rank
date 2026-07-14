"""Feature preprocessing on top of polars.

Numerical features are imputed and optionally standardized; categorical
features are index-encoded for embedding lookup, with index 0 reserved for
unknown/missing values. The transform is expressed as polars expressions, so it
works identically for eager DataFrames and LazyFrames.
"""

import math
from functools import partial

import numpy as np
import polars as pl
import torch
from scipy.special import ndtri

_MISSING = "__missing__"
_BOUNDS_THRESHOLD = 1e-7


def _quantile_to_normal(
    series: pl.Series,
    quantiles: np.ndarray,
    references: np.ndarray,
) -> pl.Series:
    """Map values to a standard-normal distribution via their quantile rank.

    Faithfully reproduces ``QuantileTransformer(output_distribution='normal')``
    from scikit-learn (``_transform_col``): the empirical CDF is estimated by
    linear interpolation against the fitted ``quantiles`` -> ``references``
    table, evaluated forward and backward and averaged so *interior* repeated
    values land on the midpoint of their flat region. Values at/below the
    smallest fitted quantile are pinned to rank 0 and at/above the largest to
    rank 1, then pushed through the inverse normal CDF with finite tails.
    """
    x = series.to_numpy()
    rank = 0.5 * (
        np.interp(x, quantiles, references) - np.interp(-x, -quantiles[::-1], -references[::-1])
    )
    rank = np.where(x + _BOUNDS_THRESHOLD > quantiles[-1], 1.0, rank)
    rank = np.where(x - _BOUNDS_THRESHOLD < quantiles[0], 0.0, rank)
    tail = ndtri(1.0 - _BOUNDS_THRESHOLD)
    return pl.Series(np.clip(ndtri(rank), -tail, tail))


class TabularPreprocessor:
    """Fits feature statistics and produces polars transform expressions.

    Parameters
    ----------
    num_features, cat_features : explicit column lists, or None to infer from
        dtypes (numeric dtypes -> numerical, string/categorical/bool -> categorical).
    normalize : numeric normalization mode. Accepts:

        * ``True`` / ``"standard"`` -- standardize with mean/std (default).
        * ``"quantile"`` -- quantile -> standard-normal mapping, equivalent to
          ``sklearn.preprocessing.QuantileTransformer(output_distribution='normal')``
          but expressed natively in polars so it works for both eager and
          lazy/streaming frames.
        * ``False`` / ``None`` -- no normalization (median-impute only).
    n_quantiles : number of reference quantiles fitted per feature when
        ``normalize="quantile"`` (capped at the number of training rows).
    multihash_features : explicit list of raw high-cardinality categorical
        columns fed as a hashed embedding stream (``None`` -> no such stream).
        Unlike ``cat_features`` these are not vocab-encoded; each value is hashed
        in-library with ``multihash_n_hashes`` per-feature hash functions into
        ``[0, multihash_cardinality)`` (the "multihash" trick), so the user
        passes raw values and no pre-hashing is required.
    multihash_cardinality : size of the shared hash space (and of the embedding
        table the model builds for this stream).
    multihash_n_hashes : number of hash functions (probes) per multihash
        feature; the stream has ``len(multihash_features) * multihash_n_hashes``
        int ids.
    embedding_features : mapping ``stream_name -> vector_column`` for dense
        external embeddings (for example ALSO/Perseus vectors). Each column must
        be a polars ``List`` or fixed-size ``Array`` of numbers with a fixed
        width inferred at fit time. The stream name must match the corresponding
        model input-layer / ``embedding_encoders`` key.
    numeric_nan_fill : imputation value for missing numerical entries --
        ``"median"`` (per-column median, the default) or ``"zero"`` (constant 0).

    """

    def __init__(
        self,
        num_features: list[str] | None = None,
        cat_features: list[str] | None = None,
        normalize: bool | str | None = True,
        n_quantiles: int = 1000,
        multihash_features: list[str] | None = None,
        multihash_cardinality: int = 100_000,
        multihash_n_hashes: int = 2,
        embedding_features: dict[str, str] | None = None,
        numeric_nan_fill: str = "median",
    ) -> None:
        if multihash_n_hashes < 1:
            raise ValueError(f"multihash_n_hashes must be >= 1, got {multihash_n_hashes}")
        if multihash_cardinality < 1:
            raise ValueError(f"multihash_cardinality must be >= 1, got {multihash_cardinality}")
        if numeric_nan_fill not in {"median", "zero"}:
            raise ValueError(
                f"numeric_nan_fill must be 'median' or 'zero', got {numeric_nan_fill!r}",
            )
        self._num_features = num_features
        self._cat_features = cat_features
        self._normalize = normalize
        self._normalize_mode = self._resolve_normalize_mode(normalize)
        self._n_quantiles = n_quantiles
        self._multihash_features = multihash_features
        self._multihash_cardinality = multihash_cardinality
        self._multihash_n_hashes = multihash_n_hashes
        self._embedding_features = dict(embedding_features or {})
        self._numeric_nan_fill = numeric_nan_fill

    @staticmethod
    def _resolve_normalize_mode(normalize: bool | str | None) -> str | None:
        """Map the public ``normalize`` argument to an internal mode string."""
        if normalize is True:
            return "standard"
        if normalize is False or normalize is None:
            return None
        if normalize in ("standard", "quantile"):
            return normalize
        raise ValueError(
            "normalize must be one of True, False, None, 'standard', 'quantile'; "
            f"got {normalize!r}",
        )

    # -- fitting -----------------------------------------------------------

    def fit(
        self,
        frame: pl.DataFrame | pl.LazyFrame,
        exclude: tuple[str, ...] = (),
    ) -> "TabularPreprocessor":
        schema = frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
        (
            self.multihash_cols_,
            self.embedding_cols_,
            self.num_cols_,
            self.cat_cols_,
        ) = self._resolve_columns(
            schema,
            exclude,
        )
        lf = frame.lazy()

        # multihash stream: stateless hashing, so fit only records the derived
        # output column names and the shared-table sizing the model needs.
        self.multihash_cardinality_ = self._multihash_cardinality
        self.multihash_out_cols_ = [
            f"{c}::h{k}" for c in self.multihash_cols_ for k in range(self._multihash_n_hashes)
        ]
        self.multihash_n_inputs_ = len(self.multihash_out_cols_)
        self.embedding_input_dims_, self.embedding_kinds_ = self._fit_embedding_streams(
            lf,
            schema,
        )
        self.embedding_out_cols_ = {
            name: [f"{name}::{i}" for i in range(dim)]
            for name, dim in self.embedding_input_dims_.items()
        }
        self.extra_out_cols_ = {}
        if self.multihash_out_cols_:
            self.extra_out_cols_["multihash"] = self.multihash_out_cols_
        self.extra_out_cols_.update(self.embedding_out_cols_)

        if self.num_cols_:
            stat_exprs = []
            for c in self.num_cols_:
                col = pl.col(c).cast(pl.Float64).fill_nan(None)
                stat_exprs += [
                    col.median().alias(f"{c}::median"),
                    col.mean().alias(f"{c}::mean"),
                    col.std().alias(f"{c}::std"),
                ]
            stats = lf.select(stat_exprs).collect().row(0, named=True)
            self.num_stats_ = {
                c: {
                    "median": stats[f"{c}::median"] or 0.0,
                    "mean": stats[f"{c}::mean"] or 0.0,
                    "std": stats[f"{c}::std"] or 1.0,
                }
                for c in self.num_cols_
            }
            for s in self.num_stats_.values():
                if not np.isfinite(s["std"]) or s["std"] == 0.0:
                    s["std"] = 1.0
        else:
            self.num_stats_ = {}

        self.num_quantiles_: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if self.num_cols_ and self._normalize_mode == "quantile":
            self._fit_quantiles(lf)

        self.cat_vocabs_ = {}
        for c in self.cat_cols_:
            values = (
                lf.select(pl.col(c).cast(pl.String).drop_nulls().unique().sort())
                .collect()
                .to_series()
                .to_list()
            )
            self.cat_vocabs_[c] = {v: i + 1 for i, v in enumerate(values)}

        self.cardinalities_ = [len(v) + 1 for v in self.cat_vocabs_.values()]
        return self

    def _resolve_columns(
        self,
        schema: pl.Schema,
        exclude: tuple[str, ...],
    ) -> tuple[list[str], dict[str, str], list[str], list[str]]:
        multihash = list(self._multihash_features or [])
        embedding = dict(self._embedding_features)
        self._validate_embedding_features(embedding, schema)
        special_cols = set(multihash) | set(embedding.values())
        overlap = special_cols & (set(self._num_features or []) | set(self._cat_features or []))
        if overlap:
            raise ValueError(
                f"multihash/embedding features overlap with num/cat features: {sorted(overlap)}",
            )
        mh_emb_overlap = set(multihash) & set(embedding.values())
        if mh_emb_overlap:
            raise ValueError(
                f"multihash_features overlap with embedding_features: {sorted(mh_emb_overlap)}",
            )
        cols = [c for c in schema.names() if c not in exclude and c not in special_cols]
        if self._num_features is not None or self._cat_features is not None:
            num = list(self._num_features or [])
            cat = list(self._cat_features or [])
            if self._num_features is None:  # infer the rest as numerical
                num = [c for c in cols if c not in cat]
            if self._cat_features is None:
                cat = [c for c in cols if c not in num]
            return multihash, embedding, num, cat
        num, cat = [], []
        for c in cols:
            dtype = schema[c]
            if dtype.is_numeric() or dtype == pl.Boolean:  # booleans become 0/1 numericals
                num.append(c)
            else:
                cat.append(c)
        return multihash, embedding, num, cat

    @staticmethod
    def _validate_embedding_features(
        embedding: dict[str, str],
        schema: pl.Schema,
    ) -> None:
        reserved = {"num", "cat", "multihash"}
        bad_names = sorted(set(embedding) & reserved)
        if bad_names:
            raise ValueError(f"embedding stream names are reserved: {bad_names}")
        bad_items = {
            name: col
            for name, col in embedding.items()
            if not isinstance(name, str) or not name or not isinstance(col, str) or not col
        }
        if bad_items:
            raise ValueError(
                "embedding_features must map non-empty stream names to non-empty column names",
            )
        duplicate_cols = sorted(
            {col for col in embedding.values() if list(embedding.values()).count(col) > 1},
        )
        if duplicate_cols:
            raise ValueError(f"embedding_features contain duplicate columns: {duplicate_cols}")
        missing = sorted(col for col in embedding.values() if col not in schema)
        if missing:
            raise ValueError(f"embedding_features columns not found: {missing}")

    def _fit_embedding_streams(
        self,
        lf: pl.LazyFrame,
        schema: pl.Schema,
    ) -> tuple[dict[str, int], dict[str, str]]:
        input_dims: dict[str, int] = {}
        kinds: dict[str, str] = {}
        for name, col in self.embedding_cols_.items():
            dtype = schema[col]
            if isinstance(dtype, pl.Array):
                if len(dtype.shape) != 1:
                    raise ValueError(
                        f"embedding_features[{name!r}] must be a 1-D Array/List column",
                    )
                dim = int(dtype.size)
                kind = "array"
            elif isinstance(dtype, pl.List):
                dim = self._infer_list_embedding_dim(lf, name=name, col=col)
                kind = "list"
            else:
                raise TypeError(
                    f"embedding_features[{name!r}] column {col!r} must be a List or Array, "
                    f"got {dtype}",
                )
            if dim <= 0:
                raise ValueError(f"embedding_features[{name!r}] must have positive width")
            input_dims[name] = dim
            kinds[name] = kind
        return input_dims, kinds

    @staticmethod
    def _infer_list_embedding_dim(lf: pl.LazyFrame, *, name: str, col: str) -> int:
        lengths = (
            lf.select(
                pl.col(col).list.len().min().alias("min_len"),
                pl.col(col).list.len().max().alias("max_len"),
            )
            .collect()
            .row(0, named=True)
        )
        min_len = lengths["min_len"]
        max_len = lengths["max_len"]
        if min_len is None or max_len is None:
            raise ValueError(
                f"Cannot infer embedding width for embedding_features[{name!r}] "
                f"from empty/all-null column {col!r}",
            )
        if min_len != max_len:
            raise ValueError(
                f"embedding_features[{name!r}] column {col!r} must contain fixed-length "
                f"vectors, got lengths in [{min_len}, {max_len}]",
            )
        return int(max_len)

    def _fit_quantiles(self, lf: pl.LazyFrame) -> None:
        """Fit per-feature quantile -> standard-normal lookup tables.

        Computes ``n_quantiles`` (capped at the row count) evenly spaced
        reference quantiles per numeric feature in a single streaming pass,
        mirroring ``QuantileTransformer``'s ``quantiles_`` / ``references_``.
        Nulls and NaNs are ignored by ``quantile`` (``fill_nan(None)`` maps NaN
        to null; matches sklearn's NaN handling at fit time). Constant or
        all-missing columns get a degenerate flat table,
        which (as in scikit-learn) collapses every value onto a single finite
        tail value rather than producing inf/nan.
        """
        n_rows = lf.select(pl.len()).collect().item()
        n_q = max(2, min(self._n_quantiles, n_rows)) if n_rows else 2
        references = np.linspace(0.0, 1.0, n_q)
        aggs = [
            pl.col(c)
            .cast(pl.Float64)
            .fill_nan(None)
            .quantile(float(q), "linear")
            .alias(f"{c}::{i}")
            for c in self.num_cols_
            for i, q in enumerate(references)
        ]
        row = lf.select(aggs).collect().row(0, named=True)
        for c in self.num_cols_:
            raw = [row[f"{c}::{i}"] for i in range(n_q)]
            q = np.array(
                [v if v is not None and math.isfinite(float(v)) else np.nan for v in raw],
                dtype=np.float64,
            )
            if np.all(np.isnan(q)):
                q = np.zeros(n_q)
            else:
                # carry the nearest finite value across any null gaps, then
                # enforce the monotonic non-decreasing invariant numpy.interp needs.
                fill = self.num_stats_[c]["median"]
                q = np.where(np.isnan(q), fill, q)
                q = np.maximum.accumulate(q)
            self.num_quantiles_[c] = (q, references)

    # -- transforming ------------------------------------------------------

    def numeric_transform_exprs(self) -> list[pl.Expr]:
        """Polars expressions implementing the numeric portion of the transform."""
        exprs = []
        for c in self.num_cols_:
            s = self.num_stats_[c]
            fill_value = 0.0 if self._numeric_nan_fill == "zero" else s["median"]
            e = pl.col(c).cast(pl.Float64).fill_nan(None).fill_null(fill_value)
            if self._normalize_mode == "standard":
                e = (e - s["mean"]) / s["std"]
            elif self._normalize_mode == "quantile":
                quantiles, references = self.num_quantiles_[c]
                e = e.map_batches(
                    partial(_quantile_to_normal, quantiles=quantiles, references=references),
                    return_dtype=pl.Float64,
                )
            exprs.append(e.cast(pl.Float32).alias(c))
        return exprs

    def categorical_transform_exprs(self) -> list[pl.Expr]:
        """Polars expressions implementing the categorical portion of the transform."""
        exprs = []
        for c in self.cat_cols_:
            vocab = self.cat_vocabs_[c]
            exprs.append(
                pl.col(c)
                .cast(pl.String)
                .fill_null(_MISSING)
                .replace_strict(
                    list(vocab.keys()),
                    list(vocab.values()),
                    default=0,
                    return_dtype=pl.Int64,
                )
                .alias(c),
            )
        return exprs

    def multihash_transform_exprs(self) -> list[pl.Expr]:
        """Polars expressions hashing raw columns into the shared multihash space.

        Each multihash feature is hashed with ``multihash_n_hashes`` hash
        functions into ``[0, multihash_cardinality)``; the resulting int64
        columns feed :class:`~scikit_rank.modules.dcn.MultiHashEmbeddings`. The
        ``Int64`` cast is required -- polars ``hash`` yields ``UInt64``, which
        ``torch.from_numpy`` rejects. Hashing here (not in user code) is what
        lets callers pass raw high-cardinality values. The hash is deterministic
        within a polars version, so train and predict bucket identically.

        The seed is ``feature_index * n_hashes + k`` -- a distinct seed per
        *(feature, probe)*, so every feature gets its own independent hash
        functions. This is the paper's multiplexed hashing trick (Coleman et al.
        2023: "a different hash seed for each feature"): the same raw value in
        different features maps to different buckets, so inter-feature collisions
        are random rather than structural.
        """
        card = self._multihash_cardinality
        n_hashes = self._multihash_n_hashes
        return [
            pl.col(c)
            .cast(pl.String)
            .fill_null(_MISSING)
            .hash(seed=feature_index * n_hashes + k)
            .mod(card)
            .cast(pl.Int64)
            .alias(f"{c}::h{k}")
            for feature_index, c in enumerate(self.multihash_cols_)
            for k in range(n_hashes)
        ]

    def embedding_transform_exprs(self) -> list[pl.Expr]:
        """Polars expressions flattening dense vector columns into float streams."""
        exprs = []
        for name, col in self.embedding_cols_.items():
            getter = pl.col(col).arr if self.embedding_kinds_[name] == "array" else pl.col(col).list
            for i, out_col in enumerate(self.embedding_out_cols_[name]):
                exprs.append(
                    getter.get(i, null_on_oob=True)
                    .cast(pl.Float32)
                    .fill_nan(None)
                    .fill_null(0.0)
                    .alias(out_col),
                )
        return exprs

    def transform_exprs(self) -> list[pl.Expr]:
        """Polars expressions implementing the full transform (eager or lazy)."""
        return [
            *self.numeric_transform_exprs(),
            *self.categorical_transform_exprs(),
            *self.multihash_transform_exprs(),
            *self.embedding_transform_exprs(),
        ]

    def fit_ple_bins(
        self,
        frame: pl.DataFrame | pl.LazyFrame,
        n_bins: int = 48,
    ) -> list[torch.Tensor]:
        """Compute per-feature piecewise-linear bin edges from training quantiles.

        Mirrors ``compute_bins`` from Gorishniy et al. 2022 (rtdl-num-embeddings)
        but expressed in polars so it works for both eager DataFrames and
        LazyFrames. Bins are computed on the *transformed* numeric values so
        downstream :class:`~scikit_rank.modules.dcn.PiecewiseLinearEncoder` sees
        inputs in the same scale.

        Returns one 1-D tensor per numeric feature (sorted unique bin edges).
        Empty list if no numeric features were fit.
        """
        if not self.num_cols_:
            return []
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        lf = frame.lazy().select(self.numeric_transform_exprs())
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        aggs = [
            pl.col(c).quantile(float(q), "linear").alias(f"{c}::{i}")
            for c in self.num_cols_
            for i, q in enumerate(quantiles)
        ]
        row = lf.select(aggs).collect().row(0, named=True)
        bins: list[torch.Tensor] = []
        for c in self.num_cols_:
            raw = [row[f"{c}::{i}"] for i in range(len(quantiles))]
            finite = [float(v) for v in raw if v is not None and math.isfinite(float(v))]
            if not finite:
                raise ValueError(
                    f"Column {c!r}: all quantile values are null/non-finite; cannot fit PLE bins",
                )
            edges = torch.tensor(finite, dtype=torch.float32).unique()
            if edges.numel() < 2:
                raise ValueError(
                    f"Column {c!r} has fewer than 2 distinct quantile values; "
                    "cannot fit PLE bins (feature is effectively constant)",
                )
            bins.append(edges)
        return bins

    def transform(
        self,
        df: pl.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Eager transform to (num [n, F_num] f32, cat [n, F_cat] i64, extra streams).

        ``extra`` maps stream name -> array. It holds ``"multihash"``
        ([n, n_inputs] int64) and/or dense external embedding streams
        ([n, input_dim] float32) when configured. Each key matches a model
        input-layer name (see
        :func:`scikit_rank.factories.build_dcnv2`).
        """
        out = df.select(self.transform_exprs())
        n = len(out)
        num = (
            out.select(self.num_cols_).to_numpy().astype(np.float32)
            if self.num_cols_
            else np.zeros((n, 0), dtype=np.float32)
        )
        cat = (
            out.select(self.cat_cols_).to_numpy().astype(np.int64)
            if self.cat_cols_
            else np.zeros((n, 0), dtype=np.int64)
        )
        extra: dict[str, np.ndarray] = {}
        if self.multihash_out_cols_:
            extra["multihash"] = out.select(self.multihash_out_cols_).to_numpy().astype(np.int64)
        for name, cols in self.embedding_out_cols_.items():
            extra[name] = out.select(cols).to_numpy().astype(np.float32)
        return num, cat, extra
