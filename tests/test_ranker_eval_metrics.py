from __future__ import annotations

import numpy as np
import polars as pl

from scikit_rank import DCNRanker


def test_ranker_eval_metric_receives_group_ids() -> None:
    seen_groups: list[np.ndarray] = []

    def grouped_metric(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> float:
        assert y_true.shape == y_pred.shape == group.shape
        seen_groups.append(group.copy())
        return float(np.unique(group).size)

    train = pl.DataFrame(
        {
            "x": [0.0, 1.0, 0.2, 1.2, 0.4, 1.4, 0.6, 1.6],
            "click": [0, 1, 0, 1, 0, 1, 0, 1],
            "impression_id": [0, 0, 1, 1, 2, 2, 3, 3],
        },
    )
    eval_x = train.drop(["click", "impression_id"])
    eval_y = train["click"].to_numpy()
    eval_group = train["impression_id"].to_numpy()
    model = DCNRanker(
        loss="listwise",
        hidden_units=[8],
        cross_layers=1,
        epochs=2,
        batch_size=4,
        dropout=0.0,
        eval_metric=grouped_metric,
        eval_metric_name="group_count",
        eval_metric_group_aware=True,
        eval_metric_direction="max",
        accelerator_config={"cpu": True},
        random_state=7,
    )
    model.fit(
        train,
        y="click",
        group="impression_id",
        eval_set=(eval_x, eval_y, eval_group),
    )

    assert len(seen_groups) == 2
    assert all(np.unique(group).size == 4 for group in seen_groups)
    assert [row["val_group_count"] for row in model.history_] == [4.0, 4.0]


def test_ranker_keeps_two_argument_sklearn_metric_compatible() -> None:
    calls = 0

    def pointwise_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        assert y_true.shape == y_pred.shape
        return 0.5

    frame = pl.DataFrame(
        {
            "x": [0.0, 1.0, 0.2, 1.2],
            "click": [0, 1, 0, 1],
            "impression_id": [0, 0, 1, 1],
        },
    )
    model = DCNRanker(
        loss="bce",
        hidden_units=[4],
        cross_layers=1,
        epochs=1,
        batch_size=4,
        eval_metric=pointwise_metric,
        eval_metric_name="pointwise",
        random_state=3,
    )
    model.fit(
        frame,
        y="click",
        group="impression_id",
        eval_set=(
            frame.drop(["click", "impression_id"]),
            frame["click"].to_numpy(),
            frame["impression_id"].to_numpy(),
        ),
    )
    assert calls == 1
    assert model.history_[0]["val_pointwise"] == 0.5
