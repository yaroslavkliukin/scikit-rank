import numpy as np
import polars as pl

from scikit_rank import DCNClassifier, DCNRanker
from scikit_rank.modules.losses import BPRLoss, SoftOrdinalBCELoss


def _frame(n: int = 160):
    rng = np.random.default_rng(42)
    X = pl.DataFrame(
        {
            "x": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        },
    )
    y = (X["x"].to_numpy() > 0).astype(int)
    return X, y


def test_loss_spec_key_value_pairs_and_accelerator_config() -> None:
    X, y = _frame()
    clf = DCNClassifier(
        epochs=1,
        batch_size=64,
        hidden_units=[8],
        loss="soft_ordinal_bce:num_classes=2",
        accelerator_config={"cpu": True},
        random_state=0,
    ).fit(X, y)

    assert isinstance(clf.loss_, SoftOrdinalBCELoss)
    assert clf.loss_._num_classes == 2
    assert clf.predict_proba(X.head(3)).shape == (3, 2)


def test_ranker_loss_spec() -> None:
    X, y = _frame()
    ranker = DCNRanker(
        epochs=1,
        batch_size=64,
        hidden_units=[8],
        loss="bpr:sampling='all_pairs'",
        accelerator_config={"cpu": True},
        random_state=0,
    ).fit(X, y, group=np.arange(len(y)) // 8)

    assert isinstance(ranker.loss_, BPRLoss)
    assert ranker.loss_.sampling == "all_pairs"
    assert ranker.predict(X.head(3)).shape == (3,)


def test_ranker_eval_set_does_not_require_group() -> None:
    X, y = _frame()
    ranker = DCNRanker(
        epochs=1,
        batch_size=32,
        hidden_units=[8],
        loss="lambdarank",
        accelerator_config={"cpu": True},
        random_state=0,
    )
    group = np.arange(len(y)) // 8
    ranker.fit(X, y, group=group, eval_set=(X, y))
    assert ranker.history_


def test_ranker_eval_set_inherits_group_column_name() -> None:
    X, y = _frame()
    X_train = X.with_columns(pl.Series("qid", np.arange(len(y)) // 8))
    ranker = DCNRanker(
        epochs=1,
        batch_size=32,
        hidden_units=[8],
        loss="lambdarank",
        accelerator_config={"cpu": True},
        random_state=0,
    ).fit(X_train, y, group="qid", eval_set=(X_train, y))

    assert ranker.history_
