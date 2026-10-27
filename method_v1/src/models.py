"""Model factory with optional third-party estimators loaded lazily."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso

from .preprocessing import normalize_model_name


class ModelDependencyError(ImportError):
    """Raised when a selected optional model library is not installed."""


def _merged(defaults: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    values = dict(defaults)
    if overrides:
        values.update(dict(overrides))
    return values


def build_model(
    model_name: str,
    *,
    params: Mapping[str, Any] | None = None,
    random_state: int = 42,
    categorical_features: Sequence[str] | None = None,
) -> Any:
    """Construct an unfitted regressor for a supported model name."""

    name = normalize_model_name(model_name)

    if name == "lasso":
        defaults = {
            "alpha": 0.0005,
            "max_iter": 20_000,
            "random_state": random_state,
            "selection": "cyclic",
        }
        return Lasso(**_merged(defaults, params))

    if name in {"elasticnet", "elastic_net"}:
        defaults = {
            "alpha": 0.0005,
            "l1_ratio": 0.9,
            "max_iter": 20_000,
            "random_state": random_state,
            "selection": "cyclic",
        }
        return ElasticNet(**_merged(defaults, params))

    if name in {"gradient_boosting", "gradientboosting", "gbr"}:
        defaults = {
            "n_estimators": 1_000,
            "learning_rate": 0.03,
            "max_depth": 4,
            "max_features": "sqrt",
            "min_samples_leaf": 15,
            "min_samples_split": 10,
            "loss": "huber",
            "random_state": random_state,
        }
        return GradientBoostingRegressor(**_merged(defaults, params))

    if name in {"xgboost", "xgb"}:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ModelDependencyError(
                "XGBoost is selected but not installed: pip install xgboost"
            ) from exc
        defaults = {
            "n_estimators": 2_000,
            "learning_rate": 0.02,
            "max_depth": 4,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.001,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": random_state,
            "n_jobs": -1,
        }
        return XGBRegressor(**_merged(defaults, params))

    if name in {"lightgbm", "lgbm"}:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ModelDependencyError(
                "LightGBM is selected but not installed: pip install lightgbm"
            ) from exc
        defaults = {
            "n_estimators": 2_000,
            "learning_rate": 0.02,
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.001,
            "reg_lambda": 0.1,
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": -1,
        }
        return LGBMRegressor(**_merged(defaults, params))

    if name in {"catboost", "cat"}:
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ModelDependencyError(
                "CatBoost is selected but not installed: pip install catboost"
            ) from exc
        defaults = {
            "iterations": 2_000,
            "learning_rate": 0.03,
            "depth": 6,
            "loss_function": "RMSE",
            "random_seed": random_state,
            "verbose": False,
            "allow_writing_files": False,
            "cat_features": list(categorical_features or ()),
        }
        return CatBoostRegressor(**_merged(defaults, params))

    raise ValueError(f"Unsupported model name: {model_name!r}")

