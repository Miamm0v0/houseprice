"""Convert saved log predictions into a validated Kaggle submission CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from src.config import load_config  # noqa: E402
from src.data import (  # noqa: E402
    load_raw_data,
    make_submission_frame,
    validate_raw_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build submission.csv from ensemble predictions on log scale."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=None,
        help="Defaults to outputs/ensemble/test_blend.csv.",
    )
    parser.add_argument(
        "--output-name",
        default="submission.csv",
        help="Filename written inside outputs/submissions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.ensure_output_directories()
    bundle = load_raw_data(config)
    validate_raw_data(bundle, config)

    prediction_file = (
        args.prediction_file.expanduser().resolve()
        if args.prediction_file is not None
        else config.paths.output_path("ensemble", "test_blend.csv")
    )
    if not prediction_file.is_file():
        raise FileNotFoundError(f"Prediction file not found: {prediction_file}")

    predictions = pd.read_csv(prediction_file)
    required = {config.id_column, "prediction_log"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            f"Prediction file is missing required columns: {sorted(missing)}"
        )

    submission = make_submission_frame(
        bundle.sample_submission,
        predictions[config.id_column],
        predictions["prediction_log"],
        id_column=config.id_column,
        target_column=config.target_column,
    )

    output_name = Path(args.output_name).name
    if not output_name.lower().endswith(".csv"):
        output_name = f"{output_name}.csv"
    output_path = config.paths.output_path("submissions", output_name)
    submission.to_csv(output_path, index=False)

    print(f"Submission written: {output_path}")
    print(f"Rows: {len(submission)}")
    print(
        f"Predicted {config.target_column} range: "
        f"{submission[config.target_column].min():.2f} - "
        f"{submission[config.target_column].max():.2f}"
    )


if __name__ == "__main__":
    main()

