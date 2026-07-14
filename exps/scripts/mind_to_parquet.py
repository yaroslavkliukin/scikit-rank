from pathlib import Path
from typing import Final

import click
import polars as pl

BEHAVIOR_COLUMNS: Final = ["impression_id", "user_id", "time", "history", "impressions"]
NEWS_COLUMNS: Final = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]
MIND_TIME_FORMAT: Final = "%m/%d/%Y %I:%M:%S %p"
EMBEDDING_DIM: Final = 100
ENTITY_ID_PATTERN: Final = r'"WikidataId"\s*:\s*"[^"]+"'


def split_has_labels(split_dir: Path) -> bool:
    """Return whether the first impression list contains click labels."""
    behavior_path = split_dir / "behaviors.tsv"
    if not behavior_path.exists():
        return False
    with behavior_path.open(encoding="utf-8") as file:
        first = file.readline().rstrip("\n").split("\t")
    if len(first) < len(BEHAVIOR_COLUMNS):
        return False
    impressions = first[4].split()
    return bool(impressions) and all("-" in item for item in impressions)


def discover_labeled_splits(dataset_dir: Path) -> list[tuple[str, str, Path]]:
    """Return ``(source_split, output_split, split_dir)`` for labeled MIND splits."""
    discovered: list[tuple[str, str, Path]] = []
    for source_split in ("train", "eval", "test"):
        split_dir = dataset_dir / source_split
        if not split_has_labels(split_dir):
            if split_dir.exists():
                click.echo(f"Skipping unlabeled split: {source_split}")
            continue
        output_split = "test" if source_split == "eval" else source_split
        discovered.append((source_split, output_split, split_dir))
    return discovered


def scan_news(split_dir: Path) -> pl.LazyFrame:
    """Scan ``news.tsv`` from one split."""
    return pl.scan_csv(
        split_dir / "news.tsv",
        separator="\t",
        has_header=False,
        new_columns=NEWS_COLUMNS,
        schema_overrides=dict.fromkeys(NEWS_COLUMNS, pl.String),
        null_values=[""],
        quote_char=None,
    )


def scan_behavior_candidates(source_split: str, output_split: str, split_dir: Path) -> pl.LazyFrame:
    """Scan and explode one MIND ``behaviors.tsv`` into per-candidate rows."""
    behaviors = pl.scan_csv(
        split_dir / "behaviors.tsv",
        separator="\t",
        has_header=False,
        new_columns=BEHAVIOR_COLUMNS,
        schema_overrides=dict.fromkeys(BEHAVIOR_COLUMNS, pl.String),
        null_values=[""],
        quote_char=None,
    )
    return (
        behaviors.with_columns(
            pl.col("time").str.to_datetime(MIND_TIME_FORMAT).alias("timestamp"),
            pl.when(pl.col("history").is_null() | (pl.col("history") == ""))
            .then(0)
            .otherwise(pl.col("history").str.count_matches(" ") + 1)
            .cast(pl.UInt32)
            .alias("history_len"),
            pl.col("impressions").str.split(" ").alias("candidate"),
        )
        .explode("candidate")
        .with_columns(
            pl.lit(source_split).alias("source_split"),
            pl.lit(output_split).alias("split"),
            pl.col("candidate").str.extract(r"^([^-]+)", 1).alias("candidate_news_id"),
            pl.col("candidate").str.extract(r"-(\d)$", 1).cast(pl.Int8).alias("click"),
        )
        .filter(pl.col("click").is_not_null())
        .drop("candidate")
    )


def scan_news_entity_embeddings(split_dir: Path) -> pl.LazyFrame:
    """Mean-pool title/abstract entity embeddings per news article.

    MIND stores entity mentions as JSON strings in ``news.tsv`` and TransE vectors
    separately in ``entity_embedding.vec``. This returns one row per news article
    with a single parquet-native list column: ``entity_embedding``.
    """
    news_entities = (
        scan_news(split_dir)
        .select(
            "news_id",
            (
                pl.col("title_entities").fill_null("")
                + pl.lit(" ")
                + pl.col("abstract_entities").fill_null("")
            )
            .str.extract_all(ENTITY_ID_PATTERN)
            .list.eval(pl.element().str.extract(r'"([^"]+)"$', 1))
            .alias("entity_id"),
        )
        .explode("entity_id")
        .filter(pl.col("entity_id").is_not_null())
        .unique(subset=["news_id", "entity_id"])
    )
    vector_columns = [f"v{idx}" for idx in range(EMBEDDING_DIM)]
    pooled = (
        news_entities.join(
            scan_embedding_file(split_dir / "entity_embedding.vec"),
            left_on="entity_id",
            right_on="id",
            how="inner",
        )
        .group_by("news_id")
        .agg(
            pl.len().cast(pl.UInt16).alias("entity_embedding_count"),
            *[pl.col(column).mean().cast(pl.Float32).alias(column) for column in vector_columns],
        )
    )
    return pooled.select(
        "news_id",
        "entity_embedding_count",
        pl.concat_list(vector_columns).alias("entity_embedding"),
    )


def scan_joined_split(
    source_split: str,
    output_split: str,
    split_dir: Path,
    attach_entity_embeddings: bool,
) -> pl.LazyFrame:
    """Explode one split and left-join candidate news metadata."""
    candidates = scan_behavior_candidates(source_split, output_split, split_dir)
    news = scan_news(split_dir)
    if attach_entity_embeddings:
        news = news.join(scan_news_entity_embeddings(split_dir), on="news_id", how="left")
    return candidates.join(news, left_on="candidate_news_id", right_on="news_id", how="left")


def scan_embedding_file(path: Path) -> pl.LazyFrame:
    """Scan one ``*.vec`` file and normalize it to id + 100 float columns.

    The files often have a trailing tab, so polars sees an extra empty column.
    We deliberately keep only the id plus the first 100 vector dimensions.
    """
    raw = pl.scan_csv(path, separator="\t", has_header=False, quote_char=None)
    selected = [pl.col("column_1").cast(pl.String).alias("id")]
    selected.extend(
        pl.col(f"column_{idx + 2}").cast(pl.Float32).alias(f"v{idx}")
        for idx in range(EMBEDDING_DIM)
    )
    return raw.select(selected)


def write_embeddings(
    split_dirs: list[Path],
    filename: str,
    dst: Path,
) -> None:
    """Combine and deduplicate split-level embedding files into one parquet."""
    scans = [scan_embedding_file(split_dir / filename) for split_dir in split_dirs]
    if not scans:
        return
    pl.concat(scans).unique(subset=["id"], keep="first").sink_parquet(dst)


@click.command(
    help="Explode MIND behaviors.tsv + join news.tsv into combined parquet.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("dst_dir", type=click.Path(exists=False, file_okay=False, path_type=Path))
@click.option("--main-name", default="mind.parquet", show_default=True)
@click.option("--skip-embeddings", is_flag=True, default=False, show_default=True)
@click.option(
    "--attach-entity-embeddings",
    is_flag=True,
    default=False,
    show_default=True,
    help="Attach a mean-pooled entity_embedding list column to the main parquet.",
)
def main(
    dataset_dir: Path,
    dst_dir: Path,
    main_name: str,
    skip_embeddings: bool,
    attach_entity_embeddings: bool,
) -> None:
    """Convert one MIND dataset directory to parquet files."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    labeled_splits = discover_labeled_splits(dataset_dir)
    if not labeled_splits:
        raise click.ClickException(f"No labeled MIND behavior splits found under {dataset_dir}")

    for source_split, output_split, _ in labeled_splits:
        click.echo(f"Including split: {source_split} -> {output_split}")

    split_frames = [
        scan_joined_split(source, output, split_dir, attach_entity_embeddings)
        for source, output, split_dir in labeled_splits
    ]
    combined = pl.concat(split_frames)
    main_path = dst_dir / main_name
    combined.sink_parquet(main_path)
    rows = pl.scan_parquet(main_path).select(pl.len()).collect().item()
    click.echo(f"Wrote {rows:,} candidate rows to {main_path}")

    if skip_embeddings:
        return

    split_dirs = [split_dir for _, _, split_dir in labeled_splits]
    entity_path = dst_dir / "entity_embedding.parquet"
    relation_path = dst_dir / "relation_embedding.parquet"
    write_embeddings(split_dirs, "entity_embedding.vec", entity_path)
    write_embeddings(split_dirs, "relation_embedding.vec", relation_path)
    click.echo(f"Wrote entity embeddings to {entity_path}")
    click.echo(f"Wrote relation embeddings to {relation_path}")


if __name__ == "__main__":
    main()
