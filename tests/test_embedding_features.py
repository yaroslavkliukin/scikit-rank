from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scikit_rank import DCNRanker
from scikit_rank.preprocessing import TabularPreprocessor


def _rank_frame(n: int = 160) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(7)
    user_vec = rng.normal(size=(n, 4)).astype(np.float32)
    item_vec = rng.normal(size=(n, 3)).astype(np.float32)
    x = rng.normal(size=n)
    cat = rng.choice(["fresh", "frozen", "home"], size=n)
    qid = np.arange(n) // 8
    score = (
        0.7 * x + 0.2 * user_vec[:, 0] - 0.3 * item_vec[:, 1] + np.where(cat == "fresh", 0.4, 0.0)
    )
    y = np.clip(np.floor(score + 2.0), 0, 3).astype(np.float32)
    return (
        pl.DataFrame(
            {
                "score": x,
                "cat": cat,
                "user_vec": user_vec.tolist(),
                "item_vec": item_vec.tolist(),
                "qid": qid,
            },
        ),
        y,
    )


def test_embedding_preprocessor_flattens_vector_columns() -> None:
    X, _ = _rank_frame()
    X = X.with_columns(pl.col("item_vec").cast(pl.Array(pl.Float32, 3)))
    pre = TabularPreprocessor(
        embedding_features={
            "user_embeddings": "user_vec",
            "item_embeddings": "item_vec",
        },
    ).fit(X, exclude=("qid",))

    assert pre.embedding_cols_ == {
        "user_embeddings": "user_vec",
        "item_embeddings": "item_vec",
    }
    assert pre.embedding_input_dims_ == {"user_embeddings": 4, "item_embeddings": 3}
    assert "user_vec" not in pre.num_cols_
    assert "user_vec" not in pre.cat_cols_

    _, _, extra = pre.transform(X)
    assert extra["user_embeddings"].dtype == np.float32
    assert extra["user_embeddings"].shape == (len(X), 4)
    assert extra["item_embeddings"].shape == (len(X), 3)


def test_embedding_preprocessor_requires_fixed_length_vectors() -> None:
    X = pl.DataFrame({"vec": [[1.0, 2.0], [3.0]]})
    with pytest.raises(ValueError, match="fixed-length"):
        TabularPreprocessor(embedding_features={"dense": "vec"}).fit(X)


def test_dcn_ranker_embedding_features_end_to_end_eager() -> None:
    X, y = _rank_frame()
    ranker = DCNRanker(
        epochs=2,
        batch_size=40,
        hidden_units=[12],
        cross_layers=1,
        loss="listwise",
        num_features=["score"],
        cat_features=["cat"],
        embedding_features={
            "user_embeddings": "user_vec",
            "item_embeddings": "item_vec",
        },
        embedding_encoders={
            "user_embeddings": "tower:output_dim=5;dropout=0.0",
            "item_embeddings": "tower:output_dim=5;dropout=0.0",
        },
        accelerator_config={"cpu": True},
        random_state=0,
    ).fit(X, y, group="qid")

    assert ranker.model_.layers()["user_embeddings"].output_dim() == 5
    assert ranker.model_.layers()["item_embeddings"].output_dim() == 5
    assert ranker.n_features_in_ == 4  # score, cat, user_vec, item_vec
    assert "user_vec" in ranker.feature_names_in_
    scores = ranker.predict(X.drop("qid").head(6))
    assert scores.shape == (6,)
    assert np.isfinite(scores).all()


def test_dcn_ranker_embedding_features_end_to_end_lazy() -> None:
    X, y = _rank_frame()
    lf = X.with_columns(pl.Series("target", y)).lazy()
    ranker = DCNRanker(
        epochs=2,
        batch_size=40,
        hidden_units=[12],
        cross_layers=1,
        loss="listwise",
        chunk_rows=37,
        num_features=["score"],
        cat_features=["cat"],
        embedding_features={
            "user_embeddings": "user_vec",
            "item_embeddings": "item_vec",
        },
        embedding_encoders={
            "user_embeddings": "tower:output_dim=5;dropout=0.0",
            "item_embeddings": "tower:output_dim=5;dropout=0.0",
        },
        accelerator_config={"cpu": True},
        random_state=1,
    ).fit(lf, y="target", group="qid")

    scores = ranker.predict(lf.drop(["target", "qid"]).slice(0, 6))
    assert scores.shape == (6,)
    assert np.isfinite(scores).all()
