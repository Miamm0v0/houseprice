"""Raw data loading, validation, and submission-frame construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ProjectConfig


class DataValidationError(ValueError):
    """Raised when the competition files do not satisfy the expected schema."""


@dataclass(frozen=True)
class RawDataBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame


@dataclass(frozen=True)
class SupervisedData:
    X: pd.DataFrame
    y: pd.Series
    X_test: pd.DataFrame
    train_ids: pd.Series
    test_ids: pd.Series


def _read_csv(path: Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")
    return pd.read_csv(path, **read_csv_kwargs)


def load_raw_data(config: ProjectConfig, **read_csv_kwargs: Any) -> RawDataBundle:
    """Read the three official competition CSV files without modifying them."""

    return RawDataBundle(
        train=_read_csv(config.paths.train_csv, **read_csv_kwargs),
        test=_read_csv(config.paths.test_csv, **read_csv_kwargs),
        sample_submission=_read_csv(
            config.paths.sample_submission_csv, **read_csv_kwargs
        ),
    )


def validate_raw_data(bundle: RawDataBundle, config: ProjectConfig) -> None:
    """Validate important Kaggle schema and identifier invariants."""

    train = bundle.train
    test = bundle.test
    sample = bundle.sample_submission
    target = config.target_column
    id_column = config.id_column
    errors: list[str] = []

    for name, frame in (("train", train), ("test", test), ("sample", sample)):
        if id_column not in frame.columns:
            errors.append(f"{name} is missing ID column {id_column!r}")
        elif frame[id_column].isna().any():
            errors.append(f"{name}.{id_column} contains missing values")
        elif frame[id_column].duplicated().any():
            errors.append(f"{name}.{id_column} contains duplicate values")

    if target not in train.columns:
        errors.append(f"train is missing target column {target!r}")
    else:
        if train[target].isna().any():
            errors.append(f"train.{target} contains missing values")
        if not pd.api.types.is_numeric_dtype(train[target]):
            errors.append(f"train.{target} must be numeric")

    if target in test.columns:
        errors.append(f"test must not contain target column {target!r}")
    if target not in sample.columns:
        errors.append(f"sample submission is missing target column {target!r}")

    train_features = [c for c in train.columns if c != target]
    test_features = list(test.columns)
    missing_from_test = sorted(set(train_features) - set(test_features))
    extra_in_test = sorted(set(test_features) - set(train_features))
    if missing_from_test:
        errors.append(f"test is missing feature columns: {missing_from_test}")
    if extra_in_test:
        errors.append(f"test has unexpected feature columns: {extra_in_test}")

    if len(test) != len(sample):
        errors.append("test and sample submission have different row counts")
    elif id_column in test.columns and id_column in sample.columns:
        if not test[id_column].reset_index(drop=True).equals(
            sample[id_column].reset_index(drop=True)
        ):
            errors.append("sample submission IDs are not in the same order as test IDs")

    if errors:
        details = "\n- ".join(errors)
        raise DataValidationError(f"Data validation failed:\n- {details}")


def prepare_supervised_data(
    bundle: RawDataBundle,
    config: ProjectConfig,
) -> SupervisedData:
    """Validate files and split identifiers, features, and the raw target."""

    validate_raw_data(bundle, config)
    target = config.target_column
    id_column = config.id_column

    feature_columns = [
        column
        for column in bundle.train.columns
        if column not in {target, id_column}
    ]
    X = bundle.train.loc[:, feature_columns].copy()
    X_test = bundle.test.loc[:, feature_columns].copy()

    return SupervisedData(
        X=X,
        y=bundle.train[target].copy(),
        X_test=X_test,
        train_ids=bundle.train[id_column].copy(),
        test_ids=bundle.test[id_column].copy(),
    )


def log_target(y: pd.Series | np.ndarray) -> np.ndarray:
    """Transform a strictly positive sale-price target with log1p."""

    values = np.asarray(y, dtype=float)
    if values.ndim != 1:
        raise ValueError("Target must be one-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("Target contains NaN or infinite values")
    if (values < 0).any():
        raise ValueError("log1p target transformation does not accept negative values")
    return np.log1p(values)


def make_submission_frame(
    sample_submission: pd.DataFrame,
    test_ids: pd.Series | np.ndarray,
    log_predictions: pd.Series | np.ndarray,
    *,
    id_column: str = "Id",
    target_column: str = "SalePrice",
    clip_lower: float | None = 0.0,
) -> pd.DataFrame:
    """Convert log predictions to a validated, sample-aligned submission frame."""

    ids = np.asarray(test_ids)
    predictions = np.asarray(log_predictions, dtype=float)
    if ids.ndim != 1 or predictions.ndim != 1:
        raise ValueError("IDs and predictions must be one-dimensional")
    if len(ids) != len(predictions):
        raise ValueError("IDs and predictions have different lengths")
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain NaN or infinite values")

    original_scale = np.expm1(predictions)
    if clip_lower is not None:
        original_scale = np.maximum(original_scale, clip_lower)

    predicted = pd.DataFrame({id_column: ids, target_column: original_scale})
    if predicted[id_column].duplicated().any():
        raise ValueError("Test IDs contain duplicates")

    if id_column not in sample_submission or target_column not in sample_submission:
        raise DataValidationError(
            f"Sample submission must contain {id_column!r} and {target_column!r}"
        )

    result = sample_submission[[id_column]].merge(
        predicted,
        on=id_column,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if result[target_column].isna().any():
        raise DataValidationError("Some sample submission IDs have no prediction")
    return result[[id_column, target_column]]

