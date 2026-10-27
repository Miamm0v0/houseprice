"""Train configured base models and save OOF/test predictions on log scale."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from src.config import load_config  # noqa: E402
from src.cv import run_oof_cv  # noqa: E402
from src.data import load_raw_data, log_target, prepare_supervised_data  # noqa: E402
from src.models import ModelDependencyError  # noqa: E402
from src.preprocessing import normalize_model_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train House Prices base models with leakage-safe OOF CV."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to train; defaults to models.enabled in the config.",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Persist all fitted fold pipelines; disabled by default to save space.",
    )
    return parser.parse_args()


def _parameter_values(config, requested_name: str) -> dict:
    normalized = normalize_model_name(requested_name)
    direct = config.model_params.get(requested_name)
    if direct is None:
        direct = config.model_params.get(normalized, {})
    return dict(direct)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.ensure_output_directories()
    bundle = load_raw_data(config)
    supervised = prepare_supervised_data(bundle, config)
    y_log = log_target(supervised.y)
    requested_models = tuple(args.models or config.enabled_models)
    if not requested_models:
        raise ValueError("No models were selected")

    metric_rows: list[dict[str, object]] = []
    trained_models: list[str] = []
    skipped_models: list[str] = []

    for requested_name in requested_models:
        model_name = normalize_model_name(requested_name)
        print(f"\nTraining {model_name}")

        def report_fold(fold: int, score: float, *, name: str = model_name) -> None:
            print(f"  {name} fold {fold}: RMSE={score:.6f}")

        try:
            result = run_oof_cv(
                model_name,
                supervised.X,
                y_log,
                supervised.X_test,
                model_params=_parameter_values(config, requested_name),
                cv_config=config.cv,
                feature_config=config.features,
                preprocessing_config=config.preprocessing,
                retain_fitted_pipelines=args.save_models,
                fold_callback=report_fold,
            )
        except ModelDependencyError as exc:
            print(f"  Skipped: {exc}")
            skipped_models.append(model_name)
            continue

        oof = result.oof_frame(supervised.train_ids, id_column=config.id_column)
        oof = oof.rename(columns={"prediction": "prediction_log"})
        test_prediction = result.test_frame(
            supervised.test_ids, id_column=config.id_column
        ).rename(columns={"prediction": "prediction_log"})

        oof.to_csv(
            config.paths.output_path("oof", f"{model_name}.csv"), index=False
        )
        test_prediction.to_csv(
            config.paths.output_path("test_predictions", f"{model_name}.csv"),
            index=False,
        )

        for fold, score in enumerate(result.fold_scores, start=1):
            metric_rows.append(
                {
                    "model": model_name,
                    "level": "fold",
                    "fold": fold,
                    "rmse": score,
                    "best_iteration": result.best_iterations[fold - 1],
                }
            )
        metric_rows.append(
            {
                "model": model_name,
                "level": "oof",
                "fold": 0,
                "rmse": result.oof_score,
                "best_iteration": None,
            }
        )

        if args.save_models:
            import joblib

            joblib.dump(
                result.fitted_pipelines,
                config.paths.output_path("models", f"{model_name}_folds.joblib"),
            )

        trained_models.append(model_name)
        print(f"  {model_name} overall OOF RMSE={result.oof_score:.6f}")

    if not trained_models:
        raise RuntimeError("No model was trained successfully")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(config.paths.output_path("metrics", "cv_scores.csv"), index=False)
    print(f"\nCompleted models: {', '.join(trained_models)}")
    if skipped_models:
        print(f"Skipped optional models: {', '.join(skipped_models)}")


if __name__ == "__main__":
    main()

