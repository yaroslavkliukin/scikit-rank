"""sklearn-compatible estimators for scikit_rank.

Public API:

- :class:`DCNClassifier`
- :class:`DCNRegressor`
- :class:`DCNRanker`
- :class:`DCNBase` (shared base class for custom estimators)

Input-validation helpers live in :mod:`scikit_rank.sklearn.input_validation`.
"""

from scikit_rank.sklearn.estimator import (
    DCNBase,
    DCNClassifier,
    DCNRanker,
    DCNRegressor,
)

__all__ = [
    "DCNBase",
    "DCNClassifier",
    "DCNRanker",
    "DCNRegressor",
]
