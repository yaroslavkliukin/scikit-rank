from datetime import timedelta
from pathlib import Path

import click
import polars as pl


def build_datetime_expr(
    col: str,
    date_format: str,
    epoch_unit: str,
    source_dtype: pl.DataType,
) -> pl.Expr:
    """Return an expression parsing ``col`` into a ``Datetime``.

    Handles four shapes, in priority order:

    * existing parquet ``Date``/``Datetime`` columns -> pass through/cast.
    * ``epoch_unit`` set (``s``/``ms``/``us``/``ns``) -> integer epoch offset.
    * Avazu-style ``YYMMDDHH`` (a ``%H`` with no ``%M``) -> pad ``"00"`` minutes,
      because ``strptime`` refuses an hour without a minute.
    * any other ``strptime`` format string (e.g. ``%Y-%m-%d %H:%M:%S``).
    """
    if source_dtype == pl.Date:
        return pl.col(col).cast(pl.Datetime)
    if isinstance(source_dtype, pl.Datetime):
        return pl.col(col)
    if epoch_unit:
        return pl.from_epoch(pl.col(col).cast(pl.Int64), time_unit=epoch_unit)
    expr = pl.col(col).cast(pl.String)
    use_format = date_format
    if "%H" in date_format and "%M" not in date_format:
        expr = expr + pl.lit("00")
        use_format = date_format + "%M"
    return expr.str.to_datetime(use_format)


def scan_input(dataset_path: Path, date_idx: str) -> pl.LazyFrame:
    """Scan CSV/CSV.GZ or parquet input without materializing it."""
    if dataset_path.suffix == ".parquet":
        return pl.scan_parquet(dataset_path)
    overrides = {date_idx: pl.String, "id": pl.String}
    probe = pl.scan_csv(dataset_path, schema_overrides=overrides)
    schema_cols = probe.collect_schema().names()
    overrides = {column: dtype for column, dtype in overrides.items() if column in schema_cols}
    return pl.scan_csv(dataset_path, schema_overrides=overrides)


def filter_min_count(frame: pl.LazyFrame, column: str, min_count: int) -> pl.LazyFrame:
    """Keep only rows whose ``column`` value occurs at least ``min_count`` times."""
    if not column or min_count <= 0:
        return frame
    keep = frame.group_by(column).len().filter(pl.col("len") >= min_count).select(column)
    return frame.join(keep, on=column, how="inner")


def filter_to_fixpoint(
    frame: pl.LazyFrame,
    user_idx: str,
    item_idx: str,
    min_user_count: int,
    min_item_count: int,
) -> pl.LazyFrame:
    """Apply the user/item min-count filters until the row count is stable.

    Removing sparse items can push some users below their threshold and vice
    versa, so a single pass is not enough when both filters are active. When at
    most one filter is active this converges after one pass; when neither is
    active it is a no-op (no scan is triggered).
    """
    user_active = bool(user_idx) and min_user_count > 0
    item_active = bool(item_idx) and min_item_count > 0
    if not (user_active or item_active):
        return frame
    prev_rows = -1
    while True:
        frame = filter_min_count(frame, item_idx, min_item_count)
        frame = filter_min_count(frame, user_idx, min_user_count)
        if not (user_active and item_active):
            return frame
        rows = frame.select(pl.len()).collect(engine="streaming").item()
        if rows == prev_rows:
            return frame
        prev_rows = rows


@click.command(
    help="Create 4-way temporal split parts (full_train/train/eval/test).",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("dataset_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument(
    "dst_dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--user-idx",
    type=str,
    default="user_id",
    show_default=True,
    help="Column used as the 'user' entity for --min-user-count filtering.",
)
@click.option(
    "--item-idx",
    type=str,
    default="",
    show_default=True,
    help="Optional 'item' entity column for --min-item-count filtering (empty = skip).",
)
@click.option(
    "--date-idx",
    type=str,
    default="dttm",
    show_default=True,
    help="Timestamp column driving the temporal split.",
)
@click.option(
    "--label-idx",
    type=str,
    default="label",
    show_default=True,
    help="Binary label column; used only to print per-split positive rates.",
)
@click.option(
    "--date-format",
    type=str,
    default="%y%m%d%H",
    show_default=True,
    help="strptime format for --date-idx (Avazu YYMMDDHH by default).",
)
@click.option(
    "--epoch-unit",
    type=str,
    default="",
    show_default=True,
    help="If set (s/ms/us/ns) parse --date-idx as an integer epoch instead.",
)
@click.option("--test-days", type=click.INT, default=1, show_default=True)
@click.option("--eval-days", type=click.INT, default=1, show_default=True)
@click.option("--min-user-count", type=click.INT, default=0, show_default=True)
@click.option("--min-item-count", type=click.INT, default=0, show_default=True)
@click.option(
    "--drop-duplicates",
    is_flag=True,
    default=False,
    show_default=True,
    help="Drop duplicate (user, item) pairs, keeping the chronologically last.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["csv", "parquet"]),
    default="csv",
    show_default=True,
)
def main(
    dataset_path: Path,
    dst_dir: Path,
    user_idx: str,
    item_idx: str,
    date_idx: str,
    label_idx: str,
    date_format: str,
    epoch_unit: str,
    test_days: int,
    eval_days: int,
    min_user_count: int,
    min_item_count: int,
    drop_duplicates: bool,
    output_format: str,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    base = scan_input(dataset_path, date_idx)
    schema = base.collect_schema()
    if date_idx not in schema:
        raise click.ClickException(f"Date column {date_idx!r} is not present in {dataset_path}")

    date_col = "__dt__"
    base = base.with_columns(
        build_datetime_expr(date_idx, date_format, epoch_unit, schema[date_idx]).alias(date_col),
    )

    if drop_duplicates and item_idx:
        base = base.unique(subset=[user_idx, item_idx], keep="last", maintain_order=True)

    base = filter_to_fixpoint(base, user_idx, item_idx, min_user_count, min_item_count)

    max_date = base.select(pl.col(date_col).max()).collect(engine="streaming").item()
    min_date = base.select(pl.col(date_col).min()).collect(engine="streaming").item()
    test_split = max_date - timedelta(days=test_days)
    eval_split = test_split - timedelta(days=eval_days)
    click.echo(f"min date: {min_date} max date: {max_date}")
    click.echo(f"test cutoff (> {test_split}) eval cutoff (> {eval_split})")

    full_train = base.filter(pl.col(date_col) <= test_split)
    splits: dict[str, pl.LazyFrame] = {
        "full_train": full_train,
        "train": full_train.filter(pl.col(date_col) <= eval_split),
        "eval": full_train.filter(pl.col(date_col) > eval_split),
        "test": base.filter(pl.col(date_col) > test_split),
    }
    drop_cols = [date_col]
    for key, frame in splits.items():
        if key in ("full_train", "train"):
            frame = filter_to_fixpoint(frame, user_idx, item_idx, min_user_count, min_item_count)
        stats_cols: list[pl.Expr] = [pl.len().alias("rows")]
        if label_idx:
            stats_cols.append(pl.col(label_idx).cast(pl.Float64).mean().alias("pos_rate"))
        stats = frame.select(stats_cols).collect(engine="streaming").row(0, named=True)
        pos = f" pos_rate={stats['pos_rate']:.4f}" if label_idx else ""
        click.echo(f"{key:>10}: rows={stats['rows']:>10,}{pos}")

        dst = dst_dir / f"{key}.{output_format}"
        out = frame.drop(drop_cols)
        sink_func = out.sink_parquet if output_format == "parquet" else out.sink_csv
        sink_func(dst)

    click.echo(f"Wrote {len(splits)} splits to {dst_dir}")


if __name__ == "__main__":
    main()
