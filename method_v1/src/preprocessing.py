"""Leakage-safe preprocessing branches for linear, tree, and CatBoost models."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DEFAULT_FORCED_CATEGORICAL


STRUCTURAL_CATEGORICAL_COLUMNS = (
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MasVnrType",
)

STRUCTURAL_NUMERIC_COLUMNS = (
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
    "MasVnrArea",
)

LINEAR_MODELS = {"lasso", "elasticnet", "elastic_net"}
ONE_HOT_TREE_MODELS = {
    "gradient_boosting",
    "gradientboosting",
    "gbr",
    "xgboost",
    "xgb",
    "lightgbm",
    "lgbm",
}
CATBOOST_MODELS = {"catboost", "cat"}


def normalize_model_name(model_name: str) -> str:
    return model_name.strip().lower().replace("-", "_").replace(" ", "_")


def _categorical_dtype(series: pd.Series) -> bool:
    dtype = series.dtype
    return bool(
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(dtype)
    )


class SemanticMissingTransformer(BaseEstimator, TransformerMixin):
    """Apply Ames-specific missing semantics while preserving a DataFrame."""

    def __init__(
        self,
        *,
        forced_categorical: Sequence[str] = DEFAULT_FORCED_CATEGORICAL,
        categorical_missing_token: str = "__NONE__",
    ) -> None:
        self.forced_categorical = forced_categorical
        self.categorical_missing_token = categorical_missing_token

    @staticmethod
    def _require_frame(X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SemanticMissingTransformer expects a pandas DataFrame")
        return X

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SemanticMissingTransformer":
        del y
        frame = self._require_frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        forced = set(self.forced_categorical)
        self.categorical_columns_ = [
            column
            for column in frame.columns
            if _categorical_dtype(frame[column]) or column in forced
        ]
        self.numeric_columns_ = [
            column for column in frame.columns if column not in self.categorical_columns_
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._require_frame(X)
        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError("SemanticMissingTransformer must be fitted first")

        expected = list(self.feature_names_in_)
        missing = [column for column in expected if column not in frame.columns]
        extra = [column for column in frame.columns if column not in expected]
        if missing or extra:
            raise ValueError(
                f"Preprocessing column mismatch; missing={missing}, extra={extra}"
            )

        result = frame.loc[:, expected].copy()
        categorical = set(self.categorical_columns_)
        forced = set(self.forced_categorical)

        for column in STRUCTURAL_CATEGORICAL_COLUMNS:
            if column in result.columns:
                values = result[column].astype(object)
                result[column] = values.where(
                    pd.notna(values), self.categorical_missing_token
                )

        for column in STRUCTURAL_NUMERIC_COLUMNS:
            if column in result.columns and column not in categorical:
                result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)

        for column in self.categorical_columns_:
            values = result[column].astype(object)
            if column in forced:
                result[column] = values.map(
                    lambda value: str(value) if pd.notna(value) else np.nan
                )
            else:
                result[column] = values.where(pd.notna(values), np.nan)

        return result

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        return self.feature_names_in_.copy()


class NonnegativeLog1pTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p to features selected as nonnegative on a training fold."""

    def fit(self, X: Any, y: Any = None) -> "NonnegativeLog1pTransformer":
        del y
        values = np.asarray(X, dtype=float)
        self.n_features_in_ = 1 if values.ndim == 1 else values.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        return np.log1p(np.maximum(values, 0.0))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            return np.asarray(
                [f"x{i}" for i in range(self.n_features_in_)], dtype=object
            )
        return np.asarray(input_features, dtype=object)


class CatBoostPreprocessor(BaseEstimator, TransformerMixin):
    """Fold-fitted imputation that keeps named categorical columns for CatBoost."""

    def __init__(
        self,
        *,
        forced_categorical: Sequence[str] = DEFAULT_FORCED_CATEGORICAL,
        missing_token: str = "__MISSING__",
    ) -> None:
        self.forced_categorical = forced_categorical
        self.missing_token = missing_token

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CatBoostPreprocessor":
        del y
        self.semantic_transformer_ = SemanticMissingTransformer(
            forced_categorical=self.forced_categorical
        ).fit(X)
        prepared = self.semantic_transformer_.transform(X)
        self.feature_names_in_ = np.asarray(prepared.columns, dtype=object)
        self.categorical_columns_ = list(
            self.semantic_transformer_.categorical_columns_
        )
        self.numeric_columns_ = [
            column
            for column in prepared.columns
            if column not in self.categorical_columns_
        ]

        numeric = prepared[self.numeric_columns_].apply(
            pd.to_numeric, errors="coerce"
        )
        self.numeric_medians_ = numeric.median().fillna(0.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "semantic_transformer_"):
            raise RuntimeError("CatBoostPreprocessor must be fitted first")
        result = self.semantic_transformer_.transform(X)

        for column in self.categorical_columns_:
            values = result[column].astype(object)
            result[column] = values.where(
                pd.notna(values), self.missing_token
            ).astype(str)

        if self.numeric_columns_:
            numeric = result[self.numeric_columns_].apply(
                pd.to_numeric, errors="coerce"
            )
            result.loc[:, self.numeric_columns_] = numeric.fillna(
                self.numeric_medians_
            )

        return result.loc[:, list(self.feature_names_in_)]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        return self.feature_names_in_.copy()


def _make_one_hot_encoder(*, sparse: bool) -> OneHotEncoder:
    parameters = {"handle_unknown": "ignore", "dtype": np.float64}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        parameters["sparse_output"] = sparse
    else:  # scikit-learn < 1.2
        parameters["sparse"] = sparse
    return OneHotEncoder(**parameters)


def _skewed_nonnegative_columns(
    frame: pd.DataFrame,
    numeric_columns: Sequence[str],
    threshold: float,
) -> list[str]:
    selected: list[str] = []
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        non_missing = values.dropna()
        if len(non_missing) < 3 or non_missing.nunique() < 3:
            continue
        if non_missing.min() < 0:
            continue
        skewness = non_missing.skew()
        if pd.notna(skewness) and abs(float(skewness)) > threshold:
            selected.append(column)
    return selected


def _linear_preprocessor(
    X_train: pd.DataFrame,
    *,
    skew_threshold: float,
    forced_categorical: Sequence[str],
) -> Pipeline:
    semantic = SemanticMissingTransformer(
        forced_categorical=forced_categorical
    ).fit(X_train)
    prepared = semantic.transform(X_train)
    categorical = list(semantic.categorical_columns_)
    numeric = list(semantic.numeric_columns_)
    skewed = _skewed_nonnegative_columns(prepared, numeric, skew_threshold)
    regular_numeric = [column for column in numeric if column not in skewed]

    numeric_regular_pipeline = Pipeline(
        steps=(
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        )
    )
    numeric_skewed_pipeline = Pipeline(
        steps=(
            ("impute", SimpleImputer(strategy="median")),
            ("log1p", NonnegativeLog1pTransformer()),
            ("scale", StandardScaler()),
        )
    )
    categorical_pipeline = Pipeline(
        steps=(
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", _make_one_hot_encoder(sparse=True)),
        )
    )
    columns = ColumnTransformer(
        transformers=(
            ("numeric", numeric_regular_pipeline, regular_numeric),
            ("skewed_numeric", numeric_skewed_pipeline, skewed),
            ("categorical", categorical_pipeline, categorical),
        ),
        remainder="drop",
        sparse_threshold=1.0,
    )
    return Pipeline(steps=(("semantic_missing", semantic), ("columns", columns)))


def _one_hot_tree_preprocessor(
    X_train: pd.DataFrame,
    *,
    forced_categorical: Sequence[str],
) -> Pipeline:
    semantic = SemanticMissingTransformer(
        forced_categorical=forced_categorical
    ).fit(X_train)
    categorical = list(semantic.categorical_columns_)
    numeric = list(semantic.numeric_columns_)

    numeric_pipeline = Pipeline(
        steps=(("impute", SimpleImputer(strategy="median")),)
    )
    categorical_pipeline = Pipeline(
        steps=(
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", _make_one_hot_encoder(sparse=False)),
        )
    )
    columns = ColumnTransformer(
        transformers=(
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ),
        remainder="drop",
        sparse_threshold=0.0,
    )
    return Pipeline(steps=(("semantic_missing", semantic), ("columns", columns)))


def build_preprocessor(
    model_name: str,
    X_train: pd.DataFrame,
    *,
    skew_threshold: float = 0.75,
    forced_categorical: Sequence[str] = DEFAULT_FORCED_CATEGORICAL,
) -> BaseEstimator:
    """Create a fresh preprocessor using only the current training fold."""

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame")
    name = normalize_model_name(model_name)

    if name in LINEAR_MODELS:
        return _linear_preprocessor(
            X_train,
            skew_threshold=skew_threshold,
            forced_categorical=forced_categorical,
        )
    if name in ONE_HOT_TREE_MODELS:
        return _one_hot_tree_preprocessor(
            X_train,
            forced_categorical=forced_categorical,
        )
    if name in CATBOOST_MODELS:
        return CatBoostPreprocessor(forced_categorical=forced_categorical)

    raise ValueError(f"No preprocessing branch is defined for model {model_name!r}")


def infer_categorical_columns(
    X_train: pd.DataFrame,
    *,
    forced_categorical: Sequence[str] = DEFAULT_FORCED_CATEGORICAL,
) -> list[str]:
    """Return the named categorical columns that CatBoost will receive."""

    transformer = SemanticMissingTransformer(
        forced_categorical=forced_categorical
    ).fit(X_train)
    return list(transformer.categorical_columns_)

