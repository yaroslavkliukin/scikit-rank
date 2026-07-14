"""scikit_rank — sklearn-compatible DCNv2 estimators.

Neural drop-in alternatives to LightGBM/CatBoost/XGBoost for tabular ranking
and classification.
"""

from scikit_rank.factories import (
    build_categorical_encoder,
    build_dcnv2,
    build_embedding_encoder,
    build_multihash_encoder,
    build_numeric_encoder,
)
from scikit_rank.modules.dcn import DCNv2, EmbeddingTower, MultiHashEmbeddings, UnifiedEmbeddings
from scikit_rank.modules.losses import LOSSES, Loss, make_loss
from scikit_rank.modules.reducers import Concat, PassThrough, Reduce
from scikit_rank.preprocessing import TabularPreprocessor
from scikit_rank.run import RunOutput, TrainingModule, TrainingRun
from scikit_rank.sklearn import DCNClassifier, DCNRanker, DCNRegressor
from scikit_rank.train import OptimizerConfig, Trainer, build_optimizer
from scikit_rank.utils import ModuleParserSpec

__all__ = [
    "LOSSES",
    "Concat",
    "DCNClassifier",
    "DCNRanker",
    "DCNRegressor",
    "DCNv2",
    "EmbeddingTower",
    "Loss",
    "ModuleParserSpec",
    "MultiHashEmbeddings",
    "OptimizerConfig",
    "PassThrough",
    "Reduce",
    "RunOutput",
    "TabularPreprocessor",
    "Trainer",
    "TrainingModule",
    "TrainingRun",
    "UnifiedEmbeddings",
    "build_categorical_encoder",
    "build_dcnv2",
    "build_embedding_encoder",
    "build_multihash_encoder",
    "build_numeric_encoder",
    "build_optimizer",
    "make_loss",
]
