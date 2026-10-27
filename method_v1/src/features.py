"""Deterministic, row-wise feature engineering for Ames House Prices."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .config import FeatureConfig


QUALITY_MAP = {
    "Po": 1,
    "Fa": 2,
    "TA": 3,
    "Gd": 4,
    "Ex": 5,
}

ORDINAL_MAPPINGS: dict[str, dict[str, int]] = {
    "ExterQual": QUALITY_MAP,
    "ExterCond": QUALITY_MAP,
    "BsmtQual": QUALITY_MAP,
    "BsmtCond": QUALITY_MAP,
    "HeatingQC": QUALITY_MAP,
    "KitchenQual": QUALITY_MAP,
    "FireplaceQu": QUALITY_MAP,
    "GarageQual": QUALITY_MAP,
    "GarageCond": QUALITY_MAP,
    "PoolQC": QUALITY_MAP,
    "BsmtExposure": {"No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {
        "Unf": 1,
        "LwQ": 2,
        "Rec": 3,
        "BLQ": 4,
        "ALQ": 5,
        "GLQ": 6,
    },
    "BsmtFinType2": {
        "Unf": 1,
        "LwQ": 2,
        "Rec": 3,
        "BLQ": 4,
        "ALQ": 5,
        "GLQ": 6,
    },
    "GarageFinish": {"Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {
        "Sal": 0,
        "Sev": 1,
        "Maj2": 2,
        "Maj1": 3,
        "Mod": 4,
        "Min2": 5,
        "Min1": 6,
        "Typ": 7,
    },
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "Utilities": {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "Street": {"Grvl": 0, "Pave": 1},
    "CentralAir": {"N": 0, "Y": 1},
}

STRUCTURAL_ORDINAL_COLUMNS = {
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
}

ABSENCE_TOKENS = {"None", "NA", "__NONE__", "__MISSING__"}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _sum_available(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series | None:
    present = [column for column in columns if column in frame.columns]
    if not present:
        return None
    numeric = frame[present].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return numeric.sum(axis=1)


def _add_aggregate_features(result: pd.DataFrame) -> None:
    total_sf = _sum_available(result, ("TotalBsmtSF", "1stFlrSF", "2ndFlrSF"))
    if total_sf is not None:
        result["TotalSF"] = total_sf

    total_porch = _sum_available(
        result,
        ("OpenPorchSF", "3SsnPorch", "EnclosedPorch", "ScreenPorch", "WoodDeckSF"),
    )
    if total_porch is not None:
        result["TotalPorchSF"] = total_porch

    total_bath = pd.Series(0.0, index=result.index)
    bath_columns = {
        "FullBath": 1.0,
        "HalfBath": 0.5,
        "BsmtFullBath": 1.0,
        "BsmtHalfBath": 0.5,
    }
    present_bath = False
    for column, weight in bath_columns.items():
        if column in result.columns:
            total_bath = total_bath.add(
                _numeric(result, column).fillna(0.0) * weight,
                fill_value=0.0,
            )
            present_bath = True
    if present_bath:
        result["TotalBathrooms"] = total_bath

    total_sqr_footage = _sum_available(
        result, ("BsmtFinSF1", "BsmtFinSF2", "1stFlrSF", "2ndFlrSF")
    )
    if total_sqr_footage is not None:
        result["TotalSqrFootage"] = total_sqr_footage

    total_finished_sf = _sum_available(
        result, ("GrLivArea", "BsmtFinSF1", "BsmtFinSF2")
    )
    if total_finished_sf is not None:
        result["TotalFinishedSF"] = total_finished_sf

    finished_basement_sf = _sum_available(result, ("BsmtFinSF1", "BsmtFinSF2"))
    if finished_basement_sf is not None:
        result["FinishedBsmtSF"] = finished_basement_sf


def _safe_ratio(
    frame: pd.DataFrame,
    numerator_column: str,
    denominator_column: str,
) -> pd.Series | None:
    if numerator_column not in frame.columns or denominator_column not in frame.columns:
        return None

    numerator = _numeric(frame, numerator_column)
    denominator = _numeric(frame, denominator_column)
    safe_denominator = denominator.mask(denominator == 0)
    ratio = numerator / safe_denominator
    return ratio.mask(denominator == 0, 0.0)


def _add_ratio_features(result: pd.DataFrame) -> None:
    ratio_specs = (
        ("FinishedBsmtSF", "TotalBsmtSF", "FinishedBsmtRatio"),
        ("GrLivArea", "TotalSF", "AboveGroundRatio"),
        ("GrLivArea", "TotRmsAbvGrd", "LivingAreaPerRoom"),
        ("GrLivArea", "BedroomAbvGr", "LivingAreaPerBedroom"),
        ("GarageArea", "GarageCars", "GarageAreaPerCar"),
        ("TotalBathrooms", "BedroomAbvGr", "BathPerBedroom"),
        ("1stFlrSF", "LotArea", "LotCoverageRatio"),
        ("GrLivArea", "LotArea", "LivingLotRatio"),
        ("TotalPorchSF", "GrLivArea", "PorchAreaRatio"),
    )
    for numerator, denominator, output in ratio_specs:
        ratio = _safe_ratio(result, numerator, denominator)
        if ratio is not None:
            result[output] = ratio


def _nonnegative_difference(
    frame: pd.DataFrame,
    later_column: str,
    earlier_column: str,
) -> pd.Series | None:
    if later_column not in frame.columns or earlier_column not in frame.columns:
        return None
    difference = _numeric(frame, later_column) - _numeric(frame, earlier_column)
    return difference.clip(lower=0)


def _add_age_features(result: pd.DataFrame) -> None:
    age_specs = {
        "HouseAge": ("YrSold", "YearBuilt"),
        "RemodAge": ("YrSold", "YearRemodAdd"),
        "GarageAge": ("YrSold", "GarageYrBlt"),
    }
    for output, (later, earlier) in age_specs.items():
        value = _nonnegative_difference(result, later, earlier)
        if value is not None:
            result[output] = value

    if "YearBuilt" in result and "YearRemodAdd" in result:
        built = _numeric(result, "YearBuilt")
        remodeled = _numeric(result, "YearRemodAdd")
        valid = built.notna() & remodeled.notna()
        result["IsRemodeled"] = ((remodeled != built) & valid).astype("int8")


def _add_indicator_features(result: pd.DataFrame) -> None:
    indicator_sources = {
        "HasPool": "PoolArea",
        "HasGarage": "GarageArea",
        "HasBsmt": "TotalBsmtSF",
        "HasFireplace": "Fireplaces",
        "HasSecondFloor": "2ndFlrSF",
        "HasMasVnr": "MasVnrArea",
        "HasWoodDeck": "WoodDeckSF",
    }
    for output, source in indicator_sources.items():
        if source in result.columns:
            result[output] = (_numeric(result, source).fillna(0.0) > 0).astype("int8")

    porch = _sum_available(
        result, ("OpenPorchSF", "3SsnPorch", "EnclosedPorch", "ScreenPorch")
    )
    if porch is not None:
        result["HasPorch"] = (porch > 0).astype("int8")


def _add_quality_interactions(result: pd.DataFrame) -> None:
    if "OverallQual" not in result.columns:
        return
    quality = _numeric(result, "OverallQual")

    if "OverallCond" in result.columns:
        result["OverallGrade"] = quality * _numeric(result, "OverallCond")

    for source, output in (
        ("GrLivArea", "QualLivArea"),
        ("TotalSF", "QualTotalSF"),
        ("TotalFinishedSF", "OverallQual_TotalFinishedSF"),
        ("GarageArea", "OverallQual_GarageArea"),
    ):
        if source in result.columns:
            result[output] = quality * _numeric(result, source)


def _add_quality_area_features(result: pd.DataFrame) -> None:
    feature_specs = (
        ("ExterQual_Ordinal", "GrLivArea", "ExterQual_GrLivArea"),
        ("KitchenQual_Ordinal", "GrLivArea", "KitchenQual_GrLivArea"),
        ("BsmtQual_Ordinal", "TotalBsmtSF", "BsmtQual_TotalBsmtSF"),
        ("GarageQual_Ordinal", "GarageArea", "GarageQual_GarageArea"),
        ("FireplaceQu_Ordinal", "Fireplaces", "FireplaceQual_Count"),
        ("HeatingQC_Ordinal", "GrLivArea", "HeatingQual_GrLivArea"),
        ("PoolQC_Ordinal", "PoolArea", "PoolQual_PoolArea"),
    )
    for quality_column, area_column, output_column in feature_specs:
        if quality_column in result.columns and area_column in result.columns:
            result[output_column] = _numeric(
                result, quality_column
            ) * _numeric(result, area_column)


def _add_ordinal_features(
    result: pd.DataFrame,
    *,
    drop_original: bool,
) -> None:
    encoded_columns: list[str] = []
    original_columns: list[str] = []

    for column, mapping in ORDINAL_MAPPINGS.items():
        if column not in result.columns:
            continue

        original = result[column]
        text = original.astype("string")
        encoded = pd.to_numeric(text.map(mapping), errors="coerce")

        if column in STRUCTURAL_ORDINAL_COLUMNS:
            absence = text.isna() | text.isin(ABSENCE_TOKENS)
            encoded = encoded.mask(absence, 0.0)

        output_column = f"{column}_Ordinal"
        result[output_column] = encoded.astype(float)
        encoded_columns.append(output_column)
        original_columns.append(column)

    if drop_original and encoded_columns:
        result.drop(columns=original_columns, inplace=True)


def engineer_features(
    frame: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Return a copy with stable features that use only values from each row."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("engineer_features expects a pandas DataFrame")

    config = config or FeatureConfig()
    result = frame.copy()

    if config.add_aggregate_features:
        _add_aggregate_features(result)
    if config.add_ratio_features:
        _add_ratio_features(result)
    if config.add_age_features:
        _add_age_features(result)
    if config.add_indicator_features:
        _add_indicator_features(result)
    if config.add_ordinal_features:
        _add_ordinal_features(
            result,
            drop_original=config.drop_original_ordinal,
        )
    if config.add_quality_interactions:
        _add_quality_interactions(result)
    if config.add_quality_area_features:
        _add_quality_area_features(result)

    return result
