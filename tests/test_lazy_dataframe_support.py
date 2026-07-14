from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scikit_rank import DCNClassifier, DCNRanker, DCNRegressor


def _lazy_frame(n: int = 320) -> pl.LazyFrame:
    rng = np.random.default_rng(2027)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c", None], size=n).tolist()
    cat_arr = np.asarray(["__missing__" if v is None else v for v in cat])
    cat_effect = np.select([cat_arr == "a", cat_arr == "b", cat_arr == "c"], [0.8, -0.4, 0.2], default=0.0)
    score = 1.8 * x1 - 1.2 * x2 + cat_effect
    y_bin = (score > 0).astype(np.int64)
    y_reg = score.astype(np.float32)
    y_rank = np.clip(np.floor(score + 2.0), 0, 3).astype(np.float32)
    group = np.arange(n) // 8
    return pl.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "cat": cat,
            "target": y_bin,
            "reg_target": y_reg,
            "rank_target": y_rank,
            "qid": group,
        },
    ).lazy()


def test_classifier_fit_predict_and_lazy_inference_from_lazyframe() -> None:
    lf = _lazy_frame()
    clf = DCNClassifier(
        epochs=2,
        batch_size=64,
        hidden_units=[12],
        cross_layers=1,
        chunk_rows=97,
        num_features=["x1", "x2"],
        cat_features=["cat"],
        accelerator_config={"cpu": True},
        random_state=0,
    ).fit(lf, y="target")

    proba = clf.predict_proba(lf.drop(["target", "reg_target", "rank_target", "qid"]))

    assert proba.shape == (320, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(320), atol=1e-6)
    assert np.isfinite(proba).all()
    assert list(clf.classes_) == [0, 1]
    assert "target" not in clf.feature_names_in_
    assert "qid" not in clf.feature_names_in_


def test_regressor_fit_from_lazyframe_with_eval_set() -> None:
    lf = _lazy_frame(256)
    train_lf = lf.slice(0, 192)
    val_lf = lf.slice(192, 64)
    reg = DCNRegressor(
        epochs=2,
        batch_size=48,
        hidden_units=[10],
        cross_layers=1,
        early_stopping_rounds=2,
        chunk_rows=51,
        num_features=["x1", "x2"],
        cat_features=["cat"],
        accelerator_config={"cpu": True},
        random_state=1,
    ).fit(train_lf, y="reg_target", eval_set=(val_lf, "reg_target"))

    pred = reg.predict(val_lf.drop(["target", "reg_target", "rank_target", "qid"]))

    assert pred.shape == (64,)
    assert np.isfinite(pred).all()
    assert len(reg.history_) == 2
    assert all("val_loss" in row for row in reg.history_)


def test_ranker_fit_from_lazyframe_with_group_column() -> None:
    lf = _lazy_frame(256)
    ranker = DCNRanker(
        epochs=2,
        batch_size=40,
        hidden_units=[12],
        cross_layers=1,
        loss="listwise",
        chunk_rows=55,
        num_features=["x1", "x2"],
        cat_features=["cat"],
        accelerator_config={"cpu": True},
        random_state=2,
    ).fit(lf, y="rank_target", group="qid")

    scores = ranker.predict(lf.drop(["target", "reg_target", "rank_target", "qid"]))

    assert scores.shape == (256,)
    assert np.isfinite(scores).all()
    assert "qid" not in ranker.feature_names_in_


def test_lazyframe_requires_target_and_group_column_names() -> None:
    lf = _lazy_frame(32)
    y = lf.select("target").collect().to_series().to_numpy()
    group = lf.select("qid").collect().to_series().to_numpy()

    with pytest.raises(ValueError, match="`y` must be a column name"):
        DCNClassifier(epochs=1, accelerator_config={"cpu": True}).fit(lf, y=y)

    with pytest.raises(ValueError, match="`group` must be a column name"):
        DCNRanker(epochs=1, accelerator_config={"cpu": True}).fit(
            lf, y="rank_target", group=group,
        )
