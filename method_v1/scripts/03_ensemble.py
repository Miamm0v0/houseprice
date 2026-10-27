"""Combine saved base-model OOF predictions and select an ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from src.config import load_config  # noqa: E402
from src.cv import rmse  # noqa: E402
from src.data import load_raw_data, log_target, prepare_supervised_data  # noqa: E402
from src.ensembling import (  # noqa: E402
    equal_weight_blend,
    optimize_blend_weights,
    prediction_correlation,
    residual_correlation,
    ridge_stack,
)
from src.preprocessing import normalize_model_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend or stack saved House Prices OOF predictions."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Base models to combine; available configured models are used otherwise.",
    )
    parser.add_argument(
        "--method",
        choices=("equal", "optimized", "stacking"),
        default="optimized",
    )
    return parser.parse_args()


def _read_prediction(
    path: Path,
    expected_ids: pd.Series,
    id_column: str,
) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {id_column, "prediction_log"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if not frame[id_column].reset_index(drop=True).equals(
        expected_ids.reset_index(drop=True)
    ):
        raise ValueError(f"ID order mismatch in {path}")
    return frame["prediction_log"]


def _available_models(config, requested: list[str] | None) -> list[str]:
    candidates = requested or list(config.enabled_models)
    available: list[str] = []
    missing: list[str] = []
    for candidate in candidates:
        model = normalize_model_name(candidate)
        oof_path = config.paths.output_path("oof", f"{model}.csv")
        test_path = config.paths.output_path("test_predictions", f"{model}.csv")
        if oof_path.is_file() and test_path.is_file():
            if model not in available:
                available.append(model)
        elif requested is not None:
            missing.append(model)
    if missing:
        raise FileNotFoundError(
            f"Missing OOF or test prediction files for: {', '.join(missing)}"
        )
    if not available:
        raise FileNotFoundError("No matching base-model prediction files were found")
    return available


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.ensure_output_directories()
    bundle = load_raw_data(config)
    supervised = prepare_supervised_data(bundle, config)
    y_log = log_target(supervised.y)
    models = _available_models(config, args.models)

    oof_predictions = pd.DataFrame(index=supervised.train_ids.index)
    test_predictions = pd.DataFrame(index=supervised.test_ids.index)
    fold_values = None

    for model in models:
        oof_path = config.paths.output_path("oof", f"{model}.csv")
        test_path = config.paths.output_path("test_predictions", f"{model}.csv")
        oof_frame = pd.read_csv(oof_path)
        oof_predictions[model] = _read_prediction(
            oof_path, supervised.train_ids, config.id_column
        ).to_numpy()
        test_predictions[model] = _read_prediction(
            test_path, supervised.test_ids, config.id_column
        ).to_numpy()
        if "fold" in oof_frame.columns:
            current_folds = oof_frame["fold"].to_numpy()
            if fold_values is None:
                fold_values = current_folds
            elif not (fold_values == current_folds).all():
                raise ValueError(f"Fold assignments differ for model {model}")

    ensemble_dir = config.paths.output_path("ensemble")
    metrics_dir = config.paths.output_path("metrics")
    prediction_correlation(oof_predictions).to_csv(
        ensemble_dir / "prediction_correlation.csv"
    )
    residual_correlation(oof_predictions, y_log).to_csv(
        ensemble_dir / "residual_correlation.csv"
    )

    equal_result = equal_weight_blend(oof_predictions, test_predictions, y_log)
    if args.method == "equal":
        selected_oof = equal_result.oof_predictions
        selected_test = equal_result.test_predictions
        selected_score = equal_result.oof_score
        selected_weights = equal_result.weights
    elif args.method == "optimized":
        optimized = optimize_blend_weights(
            oof_predictions, test_predictions, y_log
        )
        selected_oof = optimized.oof_predictions
        selected_test = optimized.test_predictions
        selected_score = optimized.oof_score
        selected_weights = optimized.weights
    else:
        stacked = ridge_stack(
            oof_predictions,
            test_predictions,
            y_log,
            n_splits=config.cv.n_splits,
            random_state=config.cv.random_state,
        )
        selected_oof = stacked.oof_predictions
        selected_test = stacked.test_predictions
        selected_score = stacked.oof_score
        selected_weights = None

    oof_output = pd.DataFrame(
        {
            config.id_column: supervised.train_ids.to_numpy(),
            "prediction_log": selected_oof,
        }
    )
    if fold_values is not None:
        oof_output.insert(1, "fold", fold_values)
    test_output = pd.DataFrame(
        {
            config.id_column: supervised.test_ids.to_numpy(),
            "prediction_log": selected_test,
        }
    )
    oof_output.to_csv(ensemble_dir / "oof_blend.csv", index=False)
    test_output.to_csv(ensemble_dir / "test_blend.csv", index=False)

    score_rows = [
        {"model": model, "rmse": rmse(y_log, oof_predictions[model].to_numpy())}
        for model in models
    ]
    score_rows.append({"model": "equal_blend", "rmse": equal_result.oof_score})
    if args.method != "equal":
        score_rows.append(
            {"model": f"selected_{args.method}", "rmse": selected_score}
        )
    pd.DataFrame(score_rows).sort_values("rmse").to_csv(
        metrics_dir / "ensemble_scores.csv", index=False
    )

    metadata: dict[str, object] = {
        "method": args.method,
        "oof_rmse": selected_score,
        "models": models,
    }
    if selected_weights is not None:
        metadata["weights"] = {
            model: float(weight) for model, weight in selected_weights.items()
        }
    with (ensemble_dir / "blend_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"Selected ensemble: {args.method}")
    print(f"Models: {', '.join(models)}")
    print(f"OOF RMSE: {selected_score:.6f}")
    if selected_weights is not None:
        print(selected_weights.to_string())


if __name__ == "__main__":
    main()

