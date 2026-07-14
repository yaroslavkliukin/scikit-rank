import numpy as np
import polars as pl
import pytest
import torch

from scikit_rank import DCNClassifier, OptimizerConfig, build_optimizer
from scikit_rank.factories import build_dcnv2
from scikit_rank.train.optimizers import (
    OPTIMIZER_TYPES,
    AdEMAMix,
    CautiousAdamW,
    Lion,
    _iter_muon_params,
    _split_param_groups,
)


def _model():
    return build_dcnv2(
        n_num_features=4,
        cardinalities=[3, 5],
        embedding_dims=[2, 2],
        hidden_units=[16, 8],
        num_encoder="ple",
        num_encoder_bins=[torch.linspace(-2, 2, 5) for _ in range(4)],
    )


def _frame(n: int = 200):
    rng = np.random.default_rng(0)
    X = pl.DataFrame(
        {
            "x": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        },
    )
    y = (X["x"].to_numpy() > 0).astype(int)
    return X, y


def test_optimizer_types_registry_is_complete():
    assert set(OPTIMIZER_TYPES) == {
        "adamw",
        "adamw_amsgrad",
        "muon",
        "schedulefree_adamw",
        "ademamix",
        "lion",
        "cautious_adamw",
    }


@pytest.mark.parametrize("optimizer_type", OPTIMIZER_TYPES)
def test_build_optimizer_runs_a_step(optimizer_type):
    model = _model()
    opt = build_optimizer(model, OptimizerConfig(optimizer_type=optimizer_type, lr=1e-3))
    assert isinstance(opt, torch.optim.Optimizer)
    batch = {
        "num": torch.randn(8, 4),
        "cat": torch.randint(0, 3, (8, 2)),
    }
    out = model(batch).sum()
    out.backward()
    before = next(model.parameters()).detach().clone()
    if hasattr(opt, "train"):
        opt.train()
    opt.step()
    assert not torch.equal(before, next(model.parameters()))


def test_split_param_groups_puts_bias_and_norm_and_numeric_in_zero_wd():
    # PLR numeric encoder carries learnable parameters (unlike identity / PLE),
    # so it exercises the numeric-encoder zero-wd path.
    model = build_dcnv2(
        n_num_features=4,
        cardinalities=[3, 5],
        embedding_dims=[2, 2],
        hidden_units=[16, 8],
        num_encoder="plr",
    )
    decay, no_decay = _split_param_groups(model, weight_decay=0.1)
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    no_decay_set = set(no_decay["params"])
    # numeric encoder params land in the no-decay group
    numeric_params = set(model.layers()["num"].parameters())
    assert numeric_params
    assert numeric_params.issubset(no_decay_set)
    # every bias is in the no-decay group too
    biases = [p for n, p in model.named_parameters() if n.endswith("bias")]
    assert biases
    assert all(b in no_decay_set for b in biases)


def test_muon_targets_only_2d_deep_weights():
    model = _model()
    muon_params = _iter_muon_params(model)
    assert muon_params
    for p in muon_params:
        assert p.ndim == 2
        assert 1 not in p.shape


def test_unknown_optimizer_type_raises():
    with pytest.raises(ValueError, match="unknown optimizer_type"):
        build_optimizer(_model(), OptimizerConfig(optimizer_type="nope"))


def test_amsgrad_variant_enables_amsgrad():
    opt = build_optimizer(_model(), OptimizerConfig(optimizer_type="adamw_amsgrad"))
    assert all(g["amsgrad"] for g in opt.param_groups)


@pytest.mark.parametrize("cls", [AdEMAMix, Lion, CautiousAdamW])
def test_vendored_optimizers_decrease_a_quadratic(cls):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.tensor([5.0, -3.0]))
    opt = cls([p], lr=0.1)
    first = None
    for _ in range(50):
        opt.zero_grad()
        loss = (p**2).sum()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.detach())
    assert float((p**2).sum().detach()) < first


@pytest.mark.parametrize("optimizer_type", OPTIMIZER_TYPES)
def test_estimator_fits_with_each_optimizer(optimizer_type):
    X, y = _frame()
    clf = DCNClassifier(
        epochs=2,
        batch_size=64,
        hidden_units=[8],
        optimizer=optimizer_type,
        accelerator_config={"cpu": True},
        random_state=0,
    ).fit(X, y)
    assert clf.predict_proba(X.head(3)).shape == (3, 2)


def test_estimator_optimizer_kwargs_forwarded():
    X, y = _frame()
    clf = DCNClassifier(
        epochs=1,
        batch_size=64,
        hidden_units=[8],
        optimizer="muon",
        optimizer_kwargs={"muon_lr": 0.01, "adam_beta2": 0.95},
        accelerator_config={"cpu": True},
        random_state=0,
    ).fit(X, y)
    assert clf.predict_proba(X.head(2)).shape == (2, 2)
