"""Configuration objects and path handling for the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MODELS = (
    "lasso",
    "elasticnet",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
)

DEFAULT_FORCED_CATEGORICAL = ("MSSubClass", "MoSold", "YrSold")

OUTPUT_SUBDIRECTORIES = (
    "reports",
    "metrics",
    "oof",
    "test_predictions",
    "ensemble",
    "models",
    "submissions",
)


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved input and output locations."""

    project_root: Path
    data_dir: Path
    output_dir: Path

    @property
    def train_csv(self) -> Path:
        return self.data_dir / "train.csv"

    @property
    def test_csv(self) -> Path:
        return self.data_dir / "test.csv"

    @property
    def sample_submission_csv(self) -> Path:
        return self.data_dir / "sample_submission.csv"

    def output_path(self, group: str, filename: str | None = None) -> Path:
        if group not in OUTPUT_SUBDIRECTORIES:
            allowed = ", ".join(OUTPUT_SUBDIRECTORIES)
            raise ValueError(f"Unknown output group {group!r}; expected one of: {allowed}")
        directory = self.output_dir / group
        return directory if filename is None else directory / filename


@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    early_stopping: bool = True
    early_stopping_rounds: int = 100


@dataclass(frozen=True)
class FeatureConfig:
    add_aggregate_features: bool = True
    add_ratio_features: bool = True
    add_age_features: bool = True
    add_indicator_features: bool = True
    add_quality_interactions: bool = True
    add_quality_area_features: bool = True
    add_ordinal_features: bool = True
    drop_original_ordinal: bool = True


@dataclass(frozen=True)
class PreprocessingConfig:
    skew_threshold: float = 0.75
    forced_categorical: tuple[str, ...] = DEFAULT_FORCED_CATEGORICAL


@dataclass(frozen=True)
class ProjectConfig:
    paths: ProjectPaths
    target_column: str = "SalePrice"
    id_column: str = "Id"
    cv: CVConfig = field(default_factory=CVConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    enabled_models: tuple[str, ...] = DEFAULT_MODELS
    model_params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def ensure_output_directories(self) -> None:
        """Create only the project-owned output directories."""

        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        for name in OUTPUT_SUBDIRECTORIES:
            (self.paths.output_dir / name).mkdir(parents=True, exist_ok=True)


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _section(mapping: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = mapping.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration section {name!r} must be a mapping")
    return value


def config_from_mapping(
    raw: Mapping[str, Any] | None = None,
    *,
    config_file: Path | None = None,
) -> ProjectConfig:
    """Build a validated configuration from a YAML-like mapping."""

    raw = raw or {}
    path_values = _section(raw, "paths")

    default_root = _default_project_root()
    if "project_root" in path_values:
        relative_to = config_file.parent if config_file is not None else Path.cwd()
        project_root = _resolve_path(path_values["project_root"], relative_to)
    else:
        project_root = default_root

    data_dir = _resolve_path(
        path_values.get(
            "data_dir",
            project_root.parent / "house-prices-advanced-regression-techniques",
        ),
        project_root,
    )
    output_dir = _resolve_path(path_values.get("output_dir", "outputs"), project_root)

    cv_values = _section(raw, "cv")
    cv = CVConfig(
        n_splits=int(cv_values.get("n_splits", 5)),
        shuffle=bool(cv_values.get("shuffle", True)),
        random_state=int(cv_values.get("random_state", 42)),
        early_stopping=bool(cv_values.get("early_stopping", True)),
        early_stopping_rounds=int(
            cv_values.get("early_stopping_rounds", 100)
        ),
    )
    if cv.n_splits < 2:
        raise ValueError("cv.n_splits must be at least 2")
    if cv.early_stopping and cv.early_stopping_rounds < 1:
        raise ValueError(
            "cv.early_stopping_rounds must be at least 1 when early stopping is enabled"
        )

    feature_values = _section(raw, "features")
    features = FeatureConfig(
        add_aggregate_features=bool(
            feature_values.get("add_aggregate_features", True)
        ),
        add_ratio_features=bool(feature_values.get("add_ratio_features", True)),
        add_age_features=bool(feature_values.get("add_age_features", True)),
        add_indicator_features=bool(
            feature_values.get("add_indicator_features", True)
        ),
        add_quality_interactions=bool(
            feature_values.get("add_quality_interactions", True)
        ),
        add_quality_area_features=bool(
            feature_values.get("add_quality_area_features", True)
        ),
        add_ordinal_features=bool(
            feature_values.get("add_ordinal_features", True)
        ),
        drop_original_ordinal=bool(
            feature_values.get("drop_original_ordinal", True)
        ),
    )

    preprocessing_values = _section(raw, "preprocessing")
    forced_categorical = tuple(
        preprocessing_values.get("forced_categorical", DEFAULT_FORCED_CATEGORICAL)
    )
    preprocessing = PreprocessingConfig(
        skew_threshold=float(preprocessing_values.get("skew_threshold", 0.75)),
        forced_categorical=forced_categorical,
    )
    if preprocessing.skew_threshold < 0:
        raise ValueError("preprocessing.skew_threshold cannot be negative")

    model_values = _section(raw, "models")
    enabled_models = tuple(model_values.get("enabled", DEFAULT_MODELS))
    params = model_values.get("params", {})
    if not isinstance(params, Mapping):
        raise TypeError("models.params must be a mapping keyed by model name")

    paths = ProjectPaths(
        project_root=project_root,
        data_dir=data_dir,
        output_dir=output_dir,
    )
    return ProjectConfig(
        paths=paths,
        target_column=str(raw.get("target_column", "SalePrice")),
        id_column=str(raw.get("id_column", "Id")),
        cv=cv,
        features=features,
        preprocessing=preprocessing,
        enabled_models=enabled_models,
        model_params=params,
    )


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load YAML configuration, or return project defaults when path is omitted."""

    if path is None:
        return config_from_mapping()

    config_file = Path(path).expanduser().resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency message only
        raise ImportError(
            "Reading YAML configuration requires PyYAML: pip install pyyaml"
        ) from exc

    with config_file.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("The top level of the configuration file must be a mapping")
    return config_from_mapping(raw, config_file=config_file)
