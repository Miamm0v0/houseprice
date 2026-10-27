"""OOF-based blending diagnostics, constrained weights, and ridge stacking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cv import rmse


@dataclass(frozen=True)
class BlendResult:
    weights: pd.Series
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    oof_score: float


@dataclass(frozen=True)
class StackingResult:
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    oof_score: float
    fold_scores: tuple[float, ...]


def _validated_prediction_frames(
    oof_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(oof_predictions, pd.DataFrame):
        raise TypeError("oof_predictions must be a pandas DataFrame")
    if not isinstance(test_predictions, pd.DataFrame):
        raise TypeError("test_predictions must be a pandas DataFrame")
    if oof_predictions.empty or test_predictions.empty:
        raise ValueError("Prediction frames cannot be empty")
    if list(oof_predictions.columns) != list(test_predictions.columns):
        raise ValueError("OOF and test prediction columns must match in the same order")

    oof_values = oof_predictions.to_numpy(dtype=float)
    test_values = test_predictions.to_numpy(dtype=float)
    if not np.isfinite(oof_values).all() or not np.isfinite(test_values).all():
        raise ValueError("Prediction frames contain NaN or infinite values")
    return oof_values, test_values


def _validated_target(y_log: pd.Series | np.ndarray, expected_length: int) -> np.ndarray:
    target = np.asarray(y_log, dtype=float)
    if target.ndim != 1 or len(target) != expected_length:
        raise ValueError("y_log must be one-dimensional and match OOF rows")
    if not np.isfinite(target).all():
        raise ValueError("y_log contains NaN or infinite values")
    return target


def weighted_average(
    predictions: pd.DataFrame | np.ndarray,
    weights: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Compute a row-wise weighted average after validating normalized weights."""

    values = np.asarray(predictions, dtype=float)
    weight_values = np.asarray(weights, dtype=float)
    if values.ndim != 2 or weight_values.ndim != 1:
        raise ValueError("Predictions must be 2-D and weights must be 1-D")
    if values.shape[1] != len(weight_values):
        raise ValueError("The number of weights does not match prediction columns")
    if not np.isfinite(values).all() or not np.isfinite(weight_values).all():
        raise ValueError("Predictions and weights must be finite")
    if (weight_values < 0).any():
        raise ValueError("Blend weights cannot be negative")
    total = weight_values.sum()
    if total <= 0:
        raise ValueError("At least one blend weight must be positive")
    normalized = weight_values / total
    return values @ normalized


def equal_weight_blend(
    oof_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    y_log: pd.Series | np.ndarray,
) -> BlendResult:
    """Create a transparent equal-weight baseline blend."""

    oof_values, test_values = _validated_prediction_frames(
        oof_predictions, test_predictions
    )
    target = _validated_target(y_log, len(oof_values))
    weights = np.full(oof_values.shape[1], 1.0 / oof_values.shape[1])
    oof_blend = weighted_average(oof_values, weights)
    test_blend = weighted_average(test_values, weights)
    return BlendResult(
        weights=pd.Series(weights, index=oof_predictions.columns, name="weight"),
        oof_predictions=oof_blend,
        test_predictions=test_blend,
        oof_score=rmse(target, oof_blend),
    )


def optimize_blend_weights(
    oof_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    y_log: pd.Series | np.ndarray,
) -> BlendResult:
    """Fit nonnegative OOF weights constrained to sum to one."""

    oof_values, test_values = _validated_prediction_frames(
        oof_predictions, test_predictions
    )
    target = _validated_target(y_log, len(oof_values))
    model_count = oof_values.shape[1]
    initial = np.full(model_count, 1.0 / model_count)

    result = minimize(
        fun=lambda weights: rmse(target, oof_values @ weights),
        x0=initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * model_count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 1_000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Blend weight optimization failed: {result.message}")

    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    weights /= weights.sum()
    oof_blend = oof_values @ weights
    test_blend = test_values @ weights
    return BlendResult(
        weights=pd.Series(weights, index=oof_predictions.columns, name="weight"),
        oof_predictions=oof_blend,
        test_predictions=test_blend,
        oof_score=rmse(target, oof_blend),
    )


def prediction_correlation(predictions: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between base-model prediction columns."""

    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise ValueError("predictions must be a non-empty DataFrame")
    return predictions.corr(method="pearson")


def residual_correlation(
    oof_predictions: pd.DataFrame,
    y_log: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Pearson correlation between base-model OOF residuals."""

    values = oof_predictions.to_numpy(dtype=float)
    target = _validated_target(y_log, len(values))
    residuals = pd.DataFrame(
        target[:, None] - values,
        columns=oof_predictions.columns,
        index=oof_predictions.index,
    )
    return residuals.corr(method="pearson")


def ridge_stack(
    oof_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    y_log: pd.Series | np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> StackingResult:
    """Train a conservative Ridge meta-model on base-model OOF columns."""

    oof_values, test_values = _validated_prediction_frames(
        oof_predictions, test_predictions
    )
    target = _validated_target(y_log, len(oof_values))
    if n_splits < 2 or n_splits > len(target):
        raise ValueError("n_splits must be between 2 and the number of OOF rows")

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    meta_oof = np.empty(len(target), dtype=float)
    meta_test = np.empty((len(test_values), n_splits), dtype=float)
    scores: list[float] = []

    for fold_index, (train_index, valid_index) in enumerate(splitter.split(oof_values)):
        meta_model = Pipeline(
            steps=(
                ("scale", StandardScaler()),
                (
                    "ridge",
                    RidgeCV(alphas=np.asarray(alphas, dtype=float), scoring="neg_mean_squared_error"),
                ),
            )
        )
        meta_model.fit(oof_values[train_index], target[train_index])
        valid_prediction = meta_model.predict(oof_values[valid_index])
        meta_oof[valid_index] = valid_prediction
        meta_test[:, fold_index] = meta_model.predict(test_values)
        scores.append(rmse(target[valid_index], valid_prediction))

    return StackingResult(
        oof_predictions=meta_oof,
        test_predictions=meta_test.mean(axis=1),
        oof_score=rmse(target, meta_oof),
        fold_scores=tuple(scores),
    )

