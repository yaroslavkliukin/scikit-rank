import numpy as np
import polars as pl
import pytest
import torch

from scikit_rank import DCNClassifier, DCNRanker
from scikit_rank.modules.losses import BCELoss, CombinedLoss, CORALLayerLoss, LambdaRankLoss
from scikit_rank.run import TrainingModule


def _frame(n: int = 96):
    rng = np.random.default_rng(123)
    X = pl.DataFrame(
        {
            "x": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        },
    )
    y_ord = rng.integers(0, 4, size=n)
    group = np.arange(n) // 8
    return X, y_ord, group


def test_coral_classifier_with_coral_loss_returns_k_probabilities() -> None:
    X, y, _ = _frame()
    clf = DCNClassifier(
        epochs=1,
        batch_size=32,
        hidden_units=[8],
        loss="coral_layer",
        accelerator_config={"cpu": True},
        random_state=0,
    ).fit(X, y)

    proba = clf.predict_proba(X.head(7))
    assert isinstance(clf.loss_, CORALLayerLoss)
    assert proba.shape == (7, 4)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(7), atol=1e-6)
    assert np.all(proba >= -1e-7)


def test_coral_layer_loss_is_classifier_only() -> None:
    X, y, group = _frame()
    ranker = DCNRanker(
        epochs=1,
        batch_size=32,
        hidden_units=[8],
        loss="coral_layer",
        accelerator_config={"cpu": True},
        random_state=0,
    )
    with pytest.raises(ValueError, match="only supported by DCNClassifier"):
        ranker.fit(X, y, group=group)


def test_ranker_combined_loss_requires_group_from_instance_not_registry_class() -> None:
    X, y, _ = _frame()
    ranker = DCNRanker(
        epochs=1,
        batch_size=32,
        hidden_units=[8],
        loss="combined:loss1='bce';loss2='lambdarank'",
        accelerator_config={"cpu": True},
        random_state=0,
    )
    with pytest.raises(ValueError, match="requires `group`"):
        ranker.fit(X, y)

    loss = ranker._make_loss()
    assert isinstance(loss, CombinedLoss)
    assert loss.requires_group


class _LengthMismatchModel(torch.nn.Module):
    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        num = inputs["num"]
        return torch.zeros(num.shape[0] + 1, dtype=num.dtype, device=num.device)


def test_training_module_validates_score_target_batch_length() -> None:
    module = TrainingModule(_LengthMismatchModel(), BCELoss())
    batch = {
        "num": torch.zeros(3, 1),
        "cat": torch.zeros(3, 0, dtype=torch.long),
        "target": torch.zeros(3),
    }
    with pytest.raises(ValueError, match="scores and target have inconsistent"):
        module(batch)


def test_training_module_validates_score_group_batch_length() -> None:
    class _Model(torch.nn.Module):
        def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
            num = inputs["num"]
            return torch.zeros(num.shape[0], dtype=num.dtype, device=num.device)

    module = TrainingModule(_Model(), LambdaRankLoss())
    batch = {
        "num": torch.zeros(3, 1),
        "cat": torch.zeros(3, 0, dtype=torch.long),
        "target": torch.zeros(3),
        "group": torch.zeros(2, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="scores and group have inconsistent"):
        module(batch)


def test_grouped_losses_validate_direct_length_mismatch() -> None:
    scores = torch.zeros(3, requires_grad=True)
    target = torch.zeros(2)
    group = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError, match="scores and target have inconsistent"):
        LambdaRankLoss()(scores, target, group)


def test_empty_pointwise_loss_returns_differentiable_zero() -> None:
    scores = torch.empty(0, requires_grad=True)
    target = torch.empty(0)
    out = BCELoss()(scores, target)
    out.backward()
    assert out.item() == 0.0
    assert scores.grad is not None
    assert torch.equal(scores.grad, torch.zeros_like(scores))
