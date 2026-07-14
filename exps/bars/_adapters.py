"""Per-model adapters (Strategy) for ``train.py`` — the only model-specific code.

Each adapter owns its whole pipeline from raw splits to predictions. Model libraries
(lightgbm/catboost/xgboost/torch+scikit_rank) are imported lazily inside methods;
keep module-level imports library-free (numpy/pandas/polars/sklearn only).
"""
from __future__ import annotations
import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pandas as pd
from _data import apply_categorical_vocabs, fit_categorical_vocabs
from sklearn.metrics import roc_auc_score

if TYPE_CHECKING:
    import numpy as np
    import polars as pl

logger = logging.getLogger("boosting.experiment")


# --------------------------------------------------------------------------- #
# Generic per-round history helper
# --------------------------------------------------------------------------- #
def _zip_history(
    train_ll: list[float],
    valid_ll: list[float],
    valid_auc: list[float],
) -> list[dict[str, float]]:
    """Zip parallel per-round curves into ``{epoch, train_loss, val_loss, val_auc}``."""
    n_rounds = max(len(train_ll), len(valid_ll), len(valid_auc))
    history: list[dict[str, float]] = []
    for i in range(n_rounds):
        record: dict[str, float] = {"epoch": i + 1}
        if i < len(train_ll):
            record["train_loss"] = float(train_ll[i])
        if i < len(valid_ll):
            record["val_loss"] = float(valid_ll[i])
        if i < len(valid_auc):
            record["val_auc"] = float(valid_auc[i])
        history.append(record)
    return history


# --------------------------------------------------------------------------- #
# Adapter contract
# --------------------------------------------------------------------------- #
class ModelAdapter(ABC):
    """One instance per model; ``main()`` drives it purely through ``run``."""

    name: str
    #: constructor kwarg carrying the RNG seed (``random_seed`` for CatBoost).
    seed_param: str = "random_state"

    @abstractmethod
    def run(
        self,
        cfg: dict[str, Any],
        train_lazy: pl.LazyFrame,
        x_val: pl.DataFrame,
        y_val: np.ndarray,
        x_test: pl.DataFrame,
        *,
        label: str,
        num_features: list[str],
        cat_features: list[str],
        params: dict[str, Any],
        seed: int,
        verbose: bool,
    ) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray, int]:
        """Train + predict; return ``(history, val_proba, test_proba, best_iteration)``.

        ``val_proba``/``test_proba`` are P(class=1) on val/test; ``y_val`` feeds early stopping.
        """

    @staticmethod
    def _predict_proba(model: Any, x: Any) -> np.ndarray:
        """P(class=1) for any sklearn-API classifier (incl. DCNClassifier)."""
        return model.predict_proba(x)[:, 1]


# --------------------------------------------------------------------------- #
# Gradient-boosting adapters
# --------------------------------------------------------------------------- #
class GBDTAdapter(ModelAdapter):
    """Shared pipeline for the GBDT models (lgbm/catboost/xgboost).

    Owns the min-count categorical encoding (OOV ``0``, mirroring scikit_rank/FuxiCTR);
    subclasses supply the per-library dtype prep, the fit call and history extraction.
    ``model_params`` reach the library as-is (native kwargs).
    """

    @abstractmethod
    def prepare_features(
        self,
        frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        cat_cols: list[str],
        cardinalities: dict[str, int],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Cast int-coded categoricals to the library dtype (same map for train/val/test)."""

    @abstractmethod
    def fit(
        self,
        params: dict[str, Any],
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_val: pd.DataFrame,
        y_val: np.ndarray,
        *,
        cat_cols: list[str],
        verbose: bool,
    ) -> tuple[Any, list[dict[str, float]], int]:
        """Train and return ``(model, history, best_iteration)``."""

    def run(self, cfg, train_lazy, x_val, y_val, x_test, *,
            label, num_features, cat_features, params, seed, verbose):
        # GBDT libraries need the train split in memory — collect the lazy frame.
        train_df = train_lazy.collect()
        y_train = train_df[label].to_numpy()
        x_train = train_df.drop(label)
        logger.info("train=%d (collected) | click-rate train=%.3f", len(y_train), y_train.mean())

        # Categorical encoding (all train uniques -> 1..N, OOV -> 0), then dtype prep.
        vocabs = fit_categorical_vocabs(x_train, cat_features)
        cardinalities = {c: len(v) + 1 for c, v in vocabs.items()}  # +1 for the OOV slot
        logger.info("cardinalities (incl. OOV): %s", cardinalities)
        x_tr = apply_categorical_vocabs(x_train, vocabs).to_pandas()
        x_va = apply_categorical_vocabs(x_val, vocabs).to_pandas()
        x_te = apply_categorical_vocabs(x_test, vocabs).to_pandas()
        x_tr, x_va, x_te = self.prepare_features((x_tr, x_va, x_te), cat_features, cardinalities)

        model, history, best_iter = self.fit(
            params, x_tr, y_train, x_va, y_val, cat_cols=cat_features, verbose=verbose,
        )
        logger.info("Trained model (best_iteration=%d)", best_iter)
        val_proba = self._predict_proba(model, x_va)
        test_proba = self._predict_proba(model, x_te)
        return history, val_proba, test_proba, best_iter


class LGBMAdapter(GBDTAdapter):
    name = "lgbm"

    def prepare_features(self, frames, cat_cols, cardinalities):
        # LightGBM accepts the int-coded frames directly (categorical_feature at fit).
        return frames

    def fit(self, params, x_train, y_train, x_val, y_val, *, cat_cols, verbose):
        import lightgbm as lgb  # noqa: PLC0415 - lazy per-model import

        # eval_metric / early_stopping_rounds are fit-time args, not ctor kwargs.
        ctor = dict(params)
        eval_metric = ctor.pop("eval_metric", ["binary_logloss", "auc"])
        esr = ctor.pop("early_stopping_rounds", None)

        evals_result: dict[str, dict[str, list[float]]] = {}
        callbacks = [
            lgb.log_evaluation(period=10 if verbose else 0),
            lgb.record_evaluation(evals_result),
        ]
        if esr is not None:
            # first_metric_only: early stopping keys off the FIRST metric in eval_metric.
            callbacks.insert(
                0,
                lgb.early_stopping(stopping_rounds=esr, first_metric_only=True, verbose=verbose),
            )

        model = lgb.LGBMClassifier(**ctor)
        model.fit(
            x_train, y_train,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            eval_names=["train", "valid"],
            eval_metric=eval_metric,
            categorical_feature=cat_cols,
            callbacks=callbacks,
        )
        best_iter = int(model.best_iteration_ or model.booster_.num_trees())
        return model, self._build_history(evals_result), best_iter

    @staticmethod
    def _build_history(evals_result):
        """LightGBM ``evals_result`` -> per-round records; AUC only if ``auc`` in eval_metric."""
        train_ll = evals_result.get("train", {}).get("binary_logloss", [])
        valid_ll = evals_result.get("valid", {}).get("binary_logloss", [])
        valid_auc = evals_result.get("valid", {}).get("auc", [])
        return _zip_history(train_ll, valid_ll, valid_auc)


class CatBoostAdapter(GBDTAdapter):
    name = "catboost"
    seed_param = "random_seed"

    def prepare_features(self, frames, cat_cols, cardinalities):
        # CatBoost requires categorical columns to be int/str (not nullable/float).
        dtypes = dict.fromkeys(cat_cols, "int64")
        return tuple(f.astype(dtypes) for f in frames)  # type: ignore[return-value]

    def fit(self, params, x_train, y_train, x_val, y_val, *, cat_cols, verbose):
        from catboost import CatBoostClassifier, Pool  # noqa: PLC0415 - lazy per-model import

        train_pool = Pool(x_train, y_train, cat_features=cat_cols)
        val_pool = Pool(x_val, y_val, cat_features=cat_cols)
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, verbose=10 if verbose else False)
        best_iter = int(model.get_best_iteration() or model.tree_count_)
        return model, self._build_history(model.get_evals_result()), best_iter

    @staticmethod
    def _build_history(evals_result):
        """CatBoost ``get_evals_result()`` -> per-round records (keys learn/validation, AUC opt)."""
        learn = evals_result.get("learn", {})
        valid = evals_result.get("validation", {})
        return _zip_history(
            learn.get("Logloss", []), valid.get("Logloss", []), valid.get("AUC", []),
        )


class XGBoostAdapter(GBDTAdapter):
    name = "xgboost"

    def prepare_features(self, frames, cat_cols, cardinalities):
        return tuple(  # type: ignore[return-value]
            self._to_categorical(f, cat_cols, cardinalities) for f in frames
        )

    def fit(self, params, x_train, y_train, x_val, y_val, *, cat_cols, verbose):
        import xgboost as xgb  # noqa: PLC0415 - lazy per-model import

        model = xgb.XGBClassifier(**params)
        # eval_set holds ONLY the val set (adding train would keep a second multi-GB
        # eval DMatrix); early stopping keys off its last metric.
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=10 if verbose else False)
        best_iter = int(model.best_iteration) if model.best_iteration is not None else 0
        return model, self._build_history(model.evals_result()), best_iter

    @staticmethod
    def _to_categorical(x, cat_cols, cardinalities):
        """Cast int-coded categoricals to an explicit ``CategoricalDtype`` (0..cardinality-1).

        The explicit range keeps the category code equal to the int value on every split
        (a bare ``'category'`` would let pandas infer different orders per split).
        """
        dtypes = {
            c: pd.CategoricalDtype(categories=list(range(cardinalities[c]))) for c in cat_cols
        }
        return x.astype(dtypes)

    @staticmethod
    def _build_history(evals_result):
        """XGBoost ``evals_result()`` -> per-round records (val only, keyed ``validation_0``)."""
        valid = evals_result.get("validation_0", {})
        return _zip_history([], valid.get("logloss", []), valid.get("auc", []))


# --------------------------------------------------------------------------- #
# DCNv2 adapter
# --------------------------------------------------------------------------- #
def _subtract_multihash(
    multihash_features: list[str] | None,
    num_features: list[str],
    cat_features: list[str],
) -> tuple[list[str], list[str]]:
    """Remove multihash columns from the (num, cat) streams — they are hashed, not
    vocab-encoded, and scikit_rank's preprocessor raises on the overlap. No-op when unset.
    """
    if not multihash_features:
        return list(num_features), list(cat_features)
    mh = set(multihash_features)
    missing = mh - set(num_features) - set(cat_features)
    if missing:
        raise SystemExit(f"multihash_features not in the dataset features: {sorted(missing)}")
    return (
        [c for c in num_features if c not in mh],
        [c for c in cat_features if c not in mh],
    )


class DCNAdapter(ModelAdapter):
    """DCNv2 via ``scikit_rank.DCNClassifier`` — feeds RAW columns and streams the train
    ``LazyFrame`` (the estimator runs its own ``TabularPreprocessor``). The adapter only
    maps the ``eval_metric`` shorthand, subtracts multihash columns, injects the feature
    lists and seeds torch; scikit_rank/torch are imported lazily (full env only).
    """

    name = "dcn"

    def run(self, cfg, train_lazy, x_val, y_val, x_test, *,
            label, num_features, cat_features, params, seed, verbose):
        import torch  # noqa: PLC0415 - lazy per-model import

        from scikit_rank import DCNClassifier  # noqa: PLC0415 - lazy per-model import

        self._seed_torch(torch, seed, deterministic=bool(cfg.get("deterministic", False)))
        est_kwargs = self._build_estimator_kwargs(
            params, num_features, cat_features, verbose=verbose,
        )
        logger.info(
            "Training DCNv2 (%s epoch(s), streaming lazy train)...", est_kwargs.get("epochs", 10),
        )
        estimator = DCNClassifier(**est_kwargs)
        estimator.fit(train_lazy, y=label, eval_set=(x_val, y_val))

        history = list(estimator.history_)
        best_iter = len(history)  # DCN has no boosting rounds; report epochs trained.
        logger.info("Trained DCNv2 (%d epoch(s))", best_iter)
        return (
            history,
            self._predict_proba(estimator, x_val),
            self._predict_proba(estimator, x_test),
            best_iter,
        )

    @staticmethod
    def _seed_torch(torch: Any, seed: int, *, deterministic: bool) -> None:
        """Seed torch's CUDA RNGs + cuDNN determinism (numpy/random already seeded).

        ``deterministic`` pins bit-for-bit CUDA kernels (slower); CUBLAS_WORKSPACE_CONFIG
        is set here, before the first cuBLAS call.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        if deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)  # noqa: FBT003

    @staticmethod
    def _build_estimator_kwargs(
        params: dict[str, Any], num_features: list[str], cat_features: list[str], *, verbose: bool,
    ) -> dict[str, Any]:
        """Map ``model_params`` onto ``DCNClassifier`` kwargs.

        ``eval_metric: auc`` -> ``roc_auc_score`` (+ name/direction); multihash cols are
        subtracted from the feature lists before injecting them. Everything else is native.
        """
        kwargs = dict(params)
        metric = kwargs.pop("eval_metric", None)  # "auc" | "loss" | None
        kwargs["eval_metric"] = roc_auc_score if metric == "auc" else None
        kwargs["eval_metric_name"] = metric or "metric"
        kwargs["eval_metric_direction"] = "max"
        num, cat = _subtract_multihash(kwargs.get("multihash_features"), num_features, cat_features)
        kwargs["num_features"] = num
        kwargs["cat_features"] = cat
        kwargs.setdefault("verbose", verbose)
        return kwargs


ADAPTERS: dict[str, ModelAdapter] = {
    "lgbm": LGBMAdapter(),
    "catboost": CatBoostAdapter(),
    "xgboost": XGBoostAdapter(),
    "dcn": DCNAdapter(),
}
