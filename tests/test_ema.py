"""Tests for the opt-in EMA (AveragedModel) weight-averaging feature.

EMA is ported from ``neural-ranking`` (``run/train.py``): the averaged weights
follow ``theta_ema <- decay*theta_ema + (1-decay)*theta_model`` per optimizer
step, are evaluated / persisted in place of the raw weights, and default off
(``ema_decay=None``) so every prior run reproduces unchanged.
"""

import numpy as np
import polars as pl
import pytest
import torch
from accelerate import Accelerator

from scikit_rank import DCNClassifier
from scikit_rank.train.options import EmaAverager


def _toy() -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 200
    X = pl.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        },
    )
    y = (X["x1"].to_numpy() + X["x2"].to_numpy() > 0).astype(int)
    return X, y


_CPU_KW = {
    "hidden_units": [16],
    "cross_layers": 1,
    "epochs": 3,
    "batch_size": 64,
    "random_state": 0,
    "accelerator_config": {"cpu": True},
}


# -- 1. EmaAverager unit behaviour -----------------------------------------


def test_ema_averager_first_update_copies_then_lerps() -> None:
    torch.manual_seed(0)
    accelerator = Accelerator(cpu=True)
    model = torch.nn.Linear(3, 2)
    ema = EmaAverager(model, accelerator, decay=0.9)

    # First update (n_averaged == 0) is a plain copy of the current weights.
    w0 = model.weight.detach().clone()
    ema.update()
    assert torch.allclose(ema._ema.module.weight, w0)

    # A second update applies the lerp with the *new* model weights.
    with torch.no_grad():
        model.weight.add_(1.0)
    w1 = model.weight.detach().clone()
    ema.update()
    expected = 0.9 * w0 + 0.1 * w1
    assert torch.allclose(ema._ema.module.weight, expected)


def test_ema_averager_store_restore_roundtrip() -> None:
    torch.manual_seed(0)
    accelerator = Accelerator(cpu=True)
    model = torch.nn.Linear(3, 2)
    ema = EmaAverager(model, accelerator, decay=0.5)
    ema.update()  # copies initial weights into the average
    with torch.no_grad():
        model.weight.add_(2.0)
    ema.update()  # average now differs from the live weights

    live = model.weight.detach().clone()
    ema_w = ema._ema.module.weight.detach().clone()
    assert not torch.allclose(live, ema_w)

    ema.store()  # swap the averaged weights into the live module
    assert torch.allclose(model.weight, ema_w)
    ema.restore_live()  # swap the raw weights back for continued training
    assert torch.allclose(model.weight, live)


def test_ema_averager_syncs_buffers_without_averaging() -> None:
    # use_buffers=False: BN running stats are hard-copied from the live model on
    # every update (not averaged), which is why update_bn is never needed.
    accelerator = Accelerator(cpu=True)
    model = torch.nn.BatchNorm1d(4)
    ema = EmaAverager(model, accelerator, decay=0.9)
    with torch.no_grad():
        model.running_mean.add_(3.0)
    ema.update()
    assert torch.allclose(ema._ema.module.running_mean, model.running_mean)


def test_ema_averager_rejects_decay_out_of_range() -> None:
    accelerator = Accelerator(cpu=True)
    model = torch.nn.Linear(3, 2)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="ema_decay"):
            EmaAverager(model, accelerator, decay=bad)


# -- 2. end-to-end through the estimator -----------------------------------


def test_ema_fit_predict_with_eval_set() -> None:
    X, y = _toy()
    clf = DCNClassifier(**_CPU_KW, ema_decay=0.9)
    clf.fit(X, y, eval_set=(X, y))
    assert clf.predict_proba(X.head(5)).shape == (5, 2)


def test_ema_fit_predict_without_eval_set() -> None:
    # No eval set -> copy_to() makes the running average the final model.
    X, y = _toy()
    clf = DCNClassifier(**_CPU_KW, ema_decay=0.9)
    clf.fit(X, y)
    assert clf.predict_proba(X.head(5)).shape == (5, 2)


def test_ema_changes_fitted_weights() -> None:
    # Same seed + data + architecture: the only difference is EMA averaging, so
    # the persisted weights must differ from the un-averaged reference.
    X, y = _toy()
    torch.manual_seed(0)
    base = DCNClassifier(**_CPU_KW).fit(X, y)
    torch.manual_seed(0)
    ema = DCNClassifier(**_CPU_KW, ema_decay=0.5).fit(X, y)

    base_sd = base.model_.state_dict()
    ema_sd = ema.model_.state_dict()
    assert base_sd.keys() == ema_sd.keys()
    assert any(not torch.allclose(base_sd[k], ema_sd[k]) for k in base_sd)


# -- 3. validation / mutual exclusion --------------------------------------


def test_ema_decay_out_of_range_raises_on_fit() -> None:
    X, y = _toy()
    clf = DCNClassifier(ema_decay=1.5, epochs=1, accelerator_config={"cpu": True})
    with pytest.raises(ValueError, match="ema_decay"):
        clf.fit(X, y)


def test_ema_with_schedulefree_optimizer_raises() -> None:
    X, y = _toy()
    clf = DCNClassifier(
        optimizer="schedulefree_adamw",
        ema_decay=0.9,
        epochs=1,
        accelerator_config={"cpu": True},
    )
    with pytest.raises(ValueError, match="schedulefree"):
        clf.fit(X, y)
