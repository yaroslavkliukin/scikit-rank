"""Lock in sklearn estimator-compliance checks for DCNClassifier / DCNRegressor.

Runs ``sklearn.utils.estimator_checks.estimator_checks_generator`` against both
estimators. SkipTest results are treated as passes (sklearn auto-skips checks
that aren't applicable to non-deterministic estimators or that require optional
environment variables like ``SCIPY_ARRAY_API``).
"""

from __future__ import annotations

import pytest
from sklearn.utils.estimator_checks import estimator_checks_generator

from scikit_rank import DCNClassifier, DCNRegressor


def _conformance_cases():
    for est in [
        DCNClassifier(
            hidden_units=(32, 16),
            cross_rank=2,
            epochs=30,
            batch_size=128,
            accelerator_config={"cpu": True},
            ple_n_bins=4,
        ),
        DCNRegressor(
            hidden_units=(32, 16, 8),
            cross_rank=2,
            epochs=30,
            batch_size=128,
            accelerator_config={"cpu": True},
            ple_n_bins=4,
        ),
    ]:
        for estimator, check in estimator_checks_generator(est):
            check_name = (
                getattr(check, "func", check).__name__
                if hasattr(check, "func")
                else check.__name__
            )
            yield pytest.param(estimator, check, id=f"{type(est).__name__}-{check_name}")


@pytest.mark.parametrize(("estimator", "check"), list(_conformance_cases()))
def test_sklearn_estimator_check(estimator, check) -> None:
    """Every sklearn ``estimator_checks_generator`` check must pass or skip."""
    try:
        check(estimator)
    except Exception as exc:
        # sklearn raises SkipTest for non-applicable checks; treat as pass.
        if type(exc).__name__ == "SkipTest":
            pytest.skip(str(exc))
        raise
