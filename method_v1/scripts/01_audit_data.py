"""Audit the official competition files and write compact diagnostic reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from src.config import load_config  # noqa: E402
from src.data import load_raw_data, log_target, validate_raw_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate House Prices data and save audit reports."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML configuration file; project defaults are used otherwise.",
    )
    return parser.parse_args()


def missing_value_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    columns = (
        "dataset",
        "column",
        "dtype",
        "missing_count",
        "missing_rate",
    )
    rows: list[dict[str, object]] = []
    for dataset, frame in (("train", train), ("test", test)):
        features = frame.drop(columns=[target_column], errors="ignore")
        for column in features.columns:
            count = int(features[column].isna().sum())
            if count == 0:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "dtype": str(features[column].dtype),
                    "missing_count": count,
                    "missing_rate": count / len(features),
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["dataset", "missing_rate"], ascending=[True, False]
    )


def categorical_level_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> pd.DataFrame:
    columns = (
        "column",
        "train_unique",
        "test_unique",
        "test_only_levels",
        "train_only_levels",
    )
    rows: list[dict[str, object]] = []
    feature_columns = [
        column
        for column in train.columns
        if column not in {target_column, id_column}
    ]
    for column in feature_columns:
        train_values = train[column]
        test_values = test[column]
        dtype = train_values.dtype
        if not (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            continue
        train_levels = set(train_values.dropna().astype(str).unique())
        test_levels = set(test_values.dropna().astype(str).unique())
        rows.append(
            {
                "column": column,
                "train_unique": len(train_levels),
                "test_unique": len(test_levels),
                "test_only_levels": len(test_levels - train_levels),
                "train_only_levels": len(train_levels - test_levels),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["test_only_levels", "column"], ascending=[False, True]
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.ensure_output_directories()
    bundle = load_raw_data(config)
    validate_raw_data(bundle, config)

    train = bundle.train
    test = bundle.test
    target = config.target_column
    id_column = config.id_column
    report_dir = config.paths.output_path("reports")

    dataset_summary = pd.DataFrame(
        [
            {
                "dataset": "train",
                "rows": len(train),
                "columns": train.shape[1],
                "duplicate_ids": int(train[id_column].duplicated().sum()),
                "total_missing": int(train.isna().sum().sum()),
            },
            {
                "dataset": "test",
                "rows": len(test),
                "columns": test.shape[1],
                "duplicate_ids": int(test[id_column].duplicated().sum()),
                "total_missing": int(test.isna().sum().sum()),
            },
        ]
    )
    dataset_summary.to_csv(report_dir / "dataset_summary.csv", index=False)

    missing_value_report(train, test, target).to_csv(
        report_dir / "missing_values.csv", index=False
    )
    categorical_level_report(train, test, target, id_column).to_csv(
        report_dir / "categorical_levels.csv", index=False
    )

    raw_target = train[target].to_numpy(dtype=float)
    transformed_target = log_target(raw_target)
    target_summary = {
        "count": int(len(raw_target)),
        "minimum": float(raw_target.min()),
        "maximum": float(raw_target.max()),
        "mean": float(raw_target.mean()),
        "median": float(np.median(raw_target)),
        "standard_deviation": float(raw_target.std(ddof=1)),
        "skewness": float(pd.Series(raw_target).skew()),
        "log1p_skewness": float(pd.Series(transformed_target).skew()),
    }
    with (report_dir / "target_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(target_summary, handle, ensure_ascii=False, indent=2)

    print(f"Data audit completed: {report_dir}")
    print(dataset_summary.to_string(index=False))


if __name__ == "__main__":
    main()
