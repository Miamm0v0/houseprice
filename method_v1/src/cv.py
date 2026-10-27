"""Out-of-fold training that wraps feature engineering and preprocessing."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from .config import CVConfig, FeatureConfig, PreprocessingConfig
from .features import engineer_features
from .models import build_model
from .preprocessing import (
    CATBOOST_MODELS,
    build_preprocessor,
    infer_categorical_columns,
    normalize_model_name,
)


EARLY_STOPPING_MODELS = {
    "xgboost",
    "xgb",
    "lightgbm",
    "lgbm",
    "catboost",
    "cat",
}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error without scikit-learn version-specific options."""

    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.shape != prediction.shape:
        raise ValueError(
            f"RMSE shape mismatch: y_true={truth.shape}, y_pred={prediction.shape}"
        )
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


@dataclass(frozen=True)
class CVResult:
    model_name: str
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    fold_scores: tuple[float, ...]
    fold_assignments: np.ndarray
    oof_score: float
    best_iterations: tuple[int | None, ...] = ()
    fitted_pipelines: tuple[Any, ...] = ()

    def metrics_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "model": self.model_name,
                "fold": np.arange(1, len(self.fold_scores) + 1),
                "rmse": self.fold_scores,
                "best_iteration": self.best_iterations,
            }
        )

    def oof_frame(
        self,
        ids: pd.Series | np.ndarray,
        *,
        id_column: str = "Id",
    ) -> pd.DataFrame:
        values = np.asarray(ids)
        if len(values) != len(self.oof_predictions):
            raise ValueError("OOF IDs and predictions have different lengths")
        return pd.DataFrame(
            {
                id_column: values,
                "fold": self.fold_assignments,
                "prediction": self.oof_predictions,
            }
        )

    def test_frame(
        self,
        ids: pd.Series | np.ndarray,
        *,
        id_column: str = "Id",
    ) -> pd.DataFrame:
        values = np.asarray(ids)
        if len(values) != len(self.test_predictions):
            raise ValueError("Test IDs and predictions have different lengths")
        return pd.DataFrame(
            {id_column: values, "prediction": self.test_predictions}
        )


def _validate_training_inputs(
    X: pd.DataFrame,
    y_log: pd.Series | np.ndarray,
    X_test: pd.DataFrame,
    n_splits: int,
) -> np.ndarray:
    if not isinstance(X, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        raise TypeError("X and X_test must be pandas DataFrames")
    if len(X) < n_splits:
        raise ValueError("The number of CV folds exceeds the number of training rows")

    missing_from_test = [column for column in X.columns if column not in X_test]
    extra_in_test = [column for column in X_test.columns if column not in X]
    if missing_from_test or extra_in_test:
        raise ValueError(
            f"Train/test feature mismatch; missing={missing_from_test}, extra={extra_in_test}"
        )

    target = np.asarray(y_log, dtype=float)
    if target.ndim != 1 or len(target) != len(X):
        raise ValueError("y_log must be one-dimensional and match the training rows")
    if not np.isfinite(target).all():
        raise ValueError("y_log contains NaN or infinite values")
    return target


def _fit_with_fold_early_stopping(
    normalized_name: str,
    model: Any,
    X_train: Any,
    y_train: np.ndarray,
    X_valid: Any,
    y_valid: np.ndarray,
    stopping_rounds: int,
) -> None:
    """Fit a supported booster using only the current fold as validation."""

    if normalized_name in {"xgboost", "xgb"}:
        model.set_params(early_stopping_rounds=stopping_rounds)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
        return

    if normalized_name in {"lightgbm", "lgbm"}:
        from lightgbm import early_stopping, log_evaluation

        fit_kwargs: dict[str, Any] = {
            "eval_metric": "rmse",
            "callbacks": [
                early_stopping(stopping_rounds, verbose=False),
                log_evaluation(period=0),
            ],
        }
        if "eval_X" in inspect.signature(model.fit).parameters:
            fit_kwargs["eval_X"] = X_valid
            fit_kwargs["eval_y"] = y_valid
        else:
            fit_kwargs["eval_set"] = [(X_valid, y_valid)]

        model.fit(X_train, y_train, **fit_kwargs)
        return

    if normalized_name in CATBOOST_MODELS:
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=stopping_rounds,
            use_best_model=True,
            verbose=False,
        )
        return

    raise ValueError(
        f"Fold early stopping is not defined for model {normalized_name!r}"
    )


def _best_iteration(model: Any, normalized_name: str) -> int | None:
    """Return a human-readable one-based best tree count when available."""

    if normalized_name in {"xgboost", "xgb"}:
        try:
            return int(model.best_iteration) + 1
        except (AttributeError, TypeError, ValueError):
            return None

    if normalized_name in {"lightgbm", "lgbm"}:
        value = getattr(model, "best_iteration_", None)
        return int(value) if value is not None and int(value) > 0 else None

    if normalized_name in CATBOOST_MODELS:
        value = getattr(model, "tree_count_", None)
        return int(value) if value is not None and int(value) > 0 else None

    return None


def run_oof_cv(
    model_name: str,
    X: pd.DataFrame,
    y_log: pd.Series | np.ndarray,
    X_test: pd.DataFrame,
    *,
    model_params: Mapping[str, Any] | None = None,
    cv_config: CVConfig | None = None,
    feature_config: FeatureConfig | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
    retain_fitted_pipelines: bool = False,
    fold_callback: Callable[[int, float], None] | None = None,
) -> CVResult:
    """Train one model with leakage-safe OOF predictions.

    ``y_log`` is intentionally explicit: callers should pass ``log1p(SalePrice)``.
    Test predictions are averaged across the fitted fold models.
    """

    cv_config = cv_config or CVConfig()
    feature_config = feature_config or FeatureConfig()
    preprocessing_config = preprocessing_config or PreprocessingConfig()
    target = _validate_training_inputs(
        X, y_log, X_test, cv_config.n_splits
    )

    X_features = engineer_features(X, feature_config)
    X_test_features = engineer_features(X_test, feature_config)
    X_test_features = X_test_features.loc[:, X_features.columns]

    splitter = KFold(
        n_splits=cv_config.n_splits,
        shuffle=cv_config.shuffle,
        random_state=cv_config.random_state if cv_config.shuffle else None,
    )
    oof_predictions = np.empty(len(X_features), dtype=float)
    fold_assignments = np.empty(len(X_features), dtype=np.int16)
    fold_test_predictions = np.empty(
        (len(X_test_features), cv_config.n_splits), dtype=float
    )
    fold_scores: list[float] = []
    best_iterations: list[int | None] = []
    fitted_pipelines: list[Any] = []
    normalized_name = normalize_model_name(model_name)

    for fold_index, (train_index, valid_index) in enumerate(
        splitter.split(X_features), start=1
    ):
        X_train = X_features.iloc[train_index]
        X_valid = X_features.iloc[valid_index]
        y_train = target[train_index]
        y_valid = target[valid_index]

        preprocessor = build_preprocessor(
            model_name,
            X_train,
            skew_threshold=preprocessing_config.skew_threshold,
            forced_categorical=preprocessing_config.forced_categorical,
        )
        categorical_features = None
        if normalized_name in CATBOOST_MODELS:
            categorical_features = infer_categorical_columns(
                X_train,
                forced_categorical=preprocessing_config.forced_categorical,
            )

        model = build_model(
            model_name,
            params=model_params,
            random_state=cv_config.random_state + fold_index,
            categorical_features=categorical_features,
        )
        pipeline = Pipeline(steps=(("preprocess", preprocessor), ("model", model)))
        use_early_stopping = (
            cv_config.early_stopping
            and normalized_name in EARLY_STOPPING_MODELS
        )

        if use_early_stopping:
            X_train_transformed = preprocessor.fit_transform(X_train, y_train)
            X_valid_transformed = preprocessor.transform(X_valid)
            X_test_transformed = preprocessor.transform(X_test_features)
            _fit_with_fold_early_stopping(
                normalized_name,
                model,
                X_train_transformed,
                y_train,
                X_valid_transformed,
                y_valid,
                cv_config.early_stopping_rounds,
            )
            valid_predictions = np.asarray(
                model.predict(X_valid_transformed), dtype=float
            )
            test_predictions = np.asarray(
                model.predict(X_test_transformed), dtype=float
            )
        else:
            pipeline.fit(X_train, y_train)
            valid_predictions = np.asarray(pipeline.predict(X_valid), dtype=float)
            test_predictions = np.asarray(
                pipeline.predict(X_test_features), dtype=float
            )
        if not np.isfinite(valid_predictions).all():
            raise ValueError(f"{model_name} produced invalid validation predictions")
        if not np.isfinite(test_predictions).all():
            raise ValueError(f"{model_name} produced invalid test predictions")

        oof_predictions[valid_index] = valid_predictions
        fold_assignments[valid_index] = fold_index
        fold_test_predictions[:, fold_index - 1] = test_predictions
        score = rmse(y_valid, valid_predictions)
        fold_scores.append(score)
        best_iterations.append(_best_iteration(model, normalized_name))

        if retain_fitted_pipelines:
            fitted_pipelines.append(pipeline)
        if fold_callback is not None:
            fold_callback(fold_index, score)

    return CVResult(
        model_name=normalized_name,
        oof_predictions=oof_predictions,
        test_predictions=fold_test_predictions.mean(axis=1),
        fold_scores=tuple(fold_scores),
        fold_assignments=fold_assignments,
        oof_score=rmse(target, oof_predictions),
        best_iterations=tuple(best_iterations),
        fitted_pipelines=tuple(fitted_pipelines),
    )

