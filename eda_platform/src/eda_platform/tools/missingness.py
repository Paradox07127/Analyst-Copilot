"""Deterministic diagnostics for structured missingness.

The tool reports observable associations only. No test over observed data can
rule out MNAR, so that limitation is a literal field in every result rather
than optional prose an agent may omit.
"""

from __future__ import annotations

import math
from typing import Any, Literal, TypedDict, cast

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats
from statsmodels.stats.multitest import multipletests

from eda_platform.core.ids import make_artifact_id
from eda_platform.core.provenance import code_ref
from eda_platform.schemas.artifacts import AnalysisTable, Artifact, ArtifactType

_MAX_AUTO_GROUP_COLUMNS = 10
_MAX_GROUP_LEVELS = 50
_MAX_AUTO_GROUP_LEVELS = 20
_MAX_CATEGORICAL_TARGET_LEVELS = 50
_MAX_GROUP_UNIQUE_SHARE = 0.2
_MAX_PUBLISHED_ASSOCIATIONS = 50


class _TargetCandidate(TypedDict):
    missing_column: str
    test_name: Literal["point_biserial", "fisher_exact", "chi_square"]
    sample_size: int
    effect_size: float
    p_value: float


class MissingIndicatorAssociation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    column_a: str
    column_b: str
    phi: float
    rows: int


class GroupMissingnessRange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    missing_column: str
    group_column: str
    groups_compared: int
    minimum_group: str
    maximum_group: str
    minimum_missing_percent: float
    maximum_missing_percent: float
    range_percentage_points: float


class TargetMissingnessAssociation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    missing_column: str
    target_column: str
    test_name: Literal["point_biserial", "fisher_exact", "chi_square"]
    sample_size: int
    effect_size: float
    p_value: float
    adjusted_p_value: float


class MissingnessDiagnosticResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    rows_total: int
    columns_analyzed: int
    columns_with_missing: int
    missing_percent: dict[str, float]
    group_columns: list[str] = Field(default_factory=list)
    target_column: str | None = None
    indicator_correlations: list[MissingIndicatorAssociation] = Field(default_factory=list)
    group_rate_ranges: list[GroupMissingnessRange] = Field(default_factory=list)
    target_associations: list[TargetMissingnessAssociation] = Field(default_factory=list)
    mnar_ruled_out: Literal[False] = False
    limitations: list[str] = Field(default_factory=list)
    table: AnalysisTable


def diagnose_missingness(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    dataset_name: str,
    target_column: str | None = None,
    group_columns: list[str] | None = None,
    top_k: int = 20,
) -> MissingnessDiagnosticResult:
    """Measure observable missingness structure without inferring MCAR/MAR/MNAR."""
    if frame.empty:
        raise ValueError("diagnose_missingness requires at least one row.")
    columns = [str(column) for column in frame.columns]
    if not columns:
        raise ValueError("diagnose_missingness requires at least one column.")
    if target_column is not None and target_column not in frame.columns:
        raise ValueError(f"Target column not found: {target_column}")
    if not 1 <= top_k <= _MAX_PUBLISHED_ASSOCIATIONS:
        raise ValueError(f"top_k must be between 1 and {_MAX_PUBLISHED_ASSOCIATIONS}.")

    resolved_groups = _resolve_group_columns(
        frame,
        target_column=target_column,
        requested=group_columns,
    )
    missing_percent = {
        column: round(float(cast(Any, _series(frame, column).isna().mean())) * 100.0, 6)
        for column in columns
    }
    missing_columns = [column for column in columns if missing_percent[column] > 0.0]
    indicator_correlations = _indicator_correlations(frame, missing_columns, top_k=top_k)
    group_ranges = _group_missingness_ranges(
        frame,
        missing_columns=missing_columns,
        group_columns=resolved_groups,
        top_k=top_k,
    )
    target_associations, association_limitations = _target_associations(
        frame,
        missing_columns=missing_columns,
        target_column=target_column,
    )

    correlations_by_column: dict[str, float] = {}
    for association in indicator_correlations:
        for column in (association.column_a, association.column_b):
            correlations_by_column[column] = max(
                correlations_by_column.get(column, 0.0), abs(association.phi)
            )
    ranges_by_column: dict[str, float] = {}
    for item in group_ranges:
        ranges_by_column[item.missing_column] = max(
            ranges_by_column.get(item.missing_column, 0.0),
            item.range_percentage_points,
        )
    target_p_by_column = {
        item.missing_column: item.adjusted_p_value for item in target_associations
    }
    table = AnalysisTable(
        dataset_id=dataset_id,
        title=f"Missingness diagnostics — {dataset_name}",
        kind="missingness_diagnostic",
        description=(
            "Observed missingness rates and associations. These diagnostics do not "
            "establish a missing-data mechanism and cannot rule out MNAR."
        ),
        rows=[
            {
                "column": column,
                "missing_count": int(cast(Any, _series(frame, column).isna().sum())),
                "missing_percent": missing_percent[column],
                "strongest_indicator_phi": correlations_by_column.get(column),
                "max_group_range_percentage_points": ranges_by_column.get(column),
                "target_adjusted_p": target_p_by_column.get(column),
            }
            for column in columns
        ],
    )
    return MissingnessDiagnosticResult(
        dataset_id=dataset_id,
        rows_total=int(len(frame)),
        columns_analyzed=len(columns),
        columns_with_missing=len(missing_columns),
        missing_percent=missing_percent,
        group_columns=resolved_groups,
        target_column=target_column,
        indicator_correlations=indicator_correlations,
        group_rate_ranges=group_ranges,
        target_associations=target_associations,
        mnar_ruled_out=False,
        limitations=[
            "Observed data cannot rule out MNAR or prove MCAR/MAR.",
            "Associations identify where missingness is structured, not why it occurs.",
            *association_limitations,
        ],
        table=table,
    )


def create_missingness_artifact(
    result: MissingnessDiagnosticResult,
    *,
    project_id: str,
    session_id: str,
) -> Artifact:
    payload = {
        **result.table.model_dump(mode="json"),
        "rows_total": result.rows_total,
        "columns_analyzed": result.columns_analyzed,
        "columns_with_missing": result.columns_with_missing,
        "group_columns": result.group_columns,
        "target_column": result.target_column,
        "indicator_correlations": [
            item.model_dump(mode="json") for item in result.indicator_correlations
        ],
        "group_rate_ranges": [item.model_dump(mode="json") for item in result.group_rate_ranges],
        "target_associations": [
            item.model_dump(mode="json") for item in result.target_associations
        ],
        "mnar_ruled_out": result.mnar_ruled_out,
        "limitations": result.limitations,
    }
    return Artifact(
        id=make_artifact_id("table", payload),
        type=ArtifactType.TABLE,
        project_id=project_id,
        session_id=session_id,
        payload=payload,
        warnings=list(result.limitations),
        code_ref=code_ref(diagnose_missingness),
        plain_language=result.table.description,
    )


def _resolve_group_columns(
    frame: pd.DataFrame,
    *,
    target_column: str | None,
    requested: list[str] | None,
) -> list[str]:
    if requested is not None:
        if len(requested) > _MAX_AUTO_GROUP_COLUMNS:
            raise ValueError(f"At most {_MAX_AUTO_GROUP_COLUMNS} group columns may be requested.")
        if len(requested) != len(set(requested)):
            raise ValueError("group_columns must not contain duplicates.")
        for column in requested:
            if column not in frame.columns:
                raise ValueError(f"Group column not found: {column}")
            _validate_group_cardinality(_series(frame, column), column=column)
        return list(requested)

    candidates: list[str] = []
    for raw_column in frame.columns:
        column = str(raw_column)
        if column == target_column:
            continue
        series = _series(frame, raw_column)
        non_null = cast(pd.Series, series.dropna())
        levels = int(non_null.nunique())
        if not 2 <= levels <= _MAX_AUTO_GROUP_LEVELS:
            continue
        categorical = (
            is_bool_dtype(non_null.dtype)
            or is_object_dtype(non_null.dtype)
            or is_string_dtype(non_null.dtype)
            or (is_numeric_dtype(non_null.dtype) and levels <= 10)
        )
        if categorical and _group_cardinality_is_safe(non_null, levels=levels):
            candidates.append(column)
        if len(candidates) == _MAX_AUTO_GROUP_COLUMNS:
            break
    return candidates


def _validate_group_cardinality(series: pd.Series, *, column: str) -> None:
    non_null = cast(pd.Series, series.dropna())
    levels = int(non_null.nunique())
    if levels < 2:
        raise ValueError(f"Group column `{column}` needs at least two observed levels.")
    if levels > _MAX_GROUP_LEVELS or not _group_cardinality_is_safe(non_null, levels=levels):
        raise ValueError(
            f"Group column `{column}` is too high-cardinality for aggregate "
            "missingness diagnostics; use a coarser grouping."
        )


def _group_cardinality_is_safe(series: pd.Series, *, levels: int) -> bool:
    if len(series) < 20:
        return levels <= 10 and levels < len(series)
    return levels / len(series) <= _MAX_GROUP_UNIQUE_SHARE


def _indicator_correlations(
    frame: pd.DataFrame,
    missing_columns: list[str],
    *,
    top_k: int,
) -> list[MissingIndicatorAssociation]:
    usable = [
        column
        for column in missing_columns
        if 0 < int(cast(Any, _series(frame, column).isna().sum())) < len(frame)
    ]
    rows: list[MissingIndicatorAssociation] = []
    for left_index, left in enumerate(usable):
        left_values = cast(pd.Series, _series(frame, left).isna().astype(float))
        for right in usable[left_index + 1 :]:
            right_values = cast(pd.Series, _series(frame, right).isna().astype(float))
            phi = float(cast(Any, left_values.corr(right_values)))
            if not math.isfinite(phi):
                continue
            rows.append(
                MissingIndicatorAssociation(
                    column_a=left,
                    column_b=right,
                    phi=round(phi, 6),
                    rows=int(len(frame)),
                )
            )
    return sorted(rows, key=lambda item: (-abs(item.phi), item.column_a, item.column_b))[:top_k]


def _group_missingness_ranges(
    frame: pd.DataFrame,
    *,
    missing_columns: list[str],
    group_columns: list[str],
    top_k: int,
) -> list[GroupMissingnessRange]:
    rows: list[GroupMissingnessRange] = []
    for missing_column in missing_columns:
        indicator = cast(pd.Series, _series(frame, missing_column).isna().astype(float))
        for group_column in group_columns:
            if missing_column == group_column:
                continue
            groups = cast(
                pd.Series,
                _series(frame, group_column).astype("string").fillna("<MISSING>"),
            )
            rates = cast(
                pd.Series,
                cast(pd.Series, indicator.groupby(groups, dropna=False).mean()) * 100.0,
            )
            if len(rates) < 2:
                continue
            minimum_key = cast(Any, rates.idxmin())
            maximum_key = cast(Any, rates.idxmax())
            minimum_group = str(minimum_key)
            maximum_group = str(maximum_key)
            minimum = float(cast(Any, rates.loc[minimum_key]))
            maximum = float(cast(Any, rates.loc[maximum_key]))
            rows.append(
                GroupMissingnessRange(
                    missing_column=missing_column,
                    group_column=group_column,
                    groups_compared=int(len(rates)),
                    minimum_group=minimum_group,
                    maximum_group=maximum_group,
                    minimum_missing_percent=round(minimum, 6),
                    maximum_missing_percent=round(maximum, 6),
                    range_percentage_points=round(maximum - minimum, 6),
                )
            )
    return sorted(
        rows,
        key=lambda item: (
            -item.range_percentage_points,
            item.missing_column,
            item.group_column,
        ),
    )[:top_k]


def _target_associations(
    frame: pd.DataFrame,
    *,
    missing_columns: list[str],
    target_column: str | None,
) -> tuple[list[TargetMissingnessAssociation], list[str]]:
    if target_column is None:
        return [], []
    target = _series(frame, target_column)
    candidates: list[_TargetCandidate] = []
    limitations: list[str] = []
    numeric_target = (
        is_numeric_dtype(target.dtype) and int(cast(Any, target.nunique(dropna=True))) > 10
    )
    target_levels = int(cast(Any, target.nunique(dropna=True)))
    if not numeric_target and target_levels > _MAX_CATEGORICAL_TARGET_LEVELS:
        return [], [
            f"Target association tests were skipped because categorical target "
            f"`{target_column}` has {target_levels} observed levels; the maximum is "
            f"{_MAX_CATEGORICAL_TARGET_LEVELS}. Use a defensible coarser outcome."
        ]
    for missing_column in missing_columns:
        if missing_column == target_column:
            continue
        valid = target.notna()
        observed_target = cast(pd.Series, target.loc[valid])
        indicator = cast(pd.Series, _series(frame, missing_column).loc[valid].isna().astype(int))
        if len(indicator) < 8 or int(cast(Any, indicator.nunique())) < 2:
            continue
        if numeric_target:
            numeric = cast(pd.Series, pd.to_numeric(observed_target, errors="coerce"))
            numeric = cast(pd.Series, numeric.replace([np.inf, -np.inf], np.nan).dropna())
            binary = cast(pd.Series, indicator.loc[numeric.index])
            if len(binary) < 8 or int(cast(Any, binary.nunique())) < 2:
                continue
            test_result = cast(Any, stats.pointbiserialr(binary, numeric))
            effect = float(test_result.statistic)
            p_value = float(test_result.pvalue)
            test_name: Literal["point_biserial", "fisher_exact", "chi_square"] = "point_biserial"
            sample_size = int(len(binary))
        else:
            table = pd.crosstab(indicator, observed_target.astype("string"))
            table = table.loc[:, table.sum(axis=0) > 0]
            if table.shape[0] != 2 or table.shape[1] < 2:
                continue
            chi_result = cast(Any, stats.chi2_contingency(table, correction=False))
            chi2 = float(chi_result.statistic)
            chi_p = float(chi_result.pvalue)
            expected = np.asarray(chi_result.expected_freq, dtype=float)
            sample_size = int(table.to_numpy().sum())
            denominator = max(1, min(table.shape[0] - 1, table.shape[1] - 1))
            effect = math.sqrt(chi2 / (sample_size * denominator))
            if table.shape == (2, 2):
                fisher_result = cast(Any, stats.fisher_exact(table.to_numpy()))
                p_value = float(fisher_result.pvalue)
                test_name = "fisher_exact"
            elif bool(np.all(expected >= 5.0)):
                p_value = chi_p
                test_name = "chi_square"
            else:
                limitations.append(
                    f"Target association for `{missing_column}` was not tested because "
                    "the categorical contingency table had expected counts below 5."
                )
                continue
        if not math.isfinite(float(effect)) or not math.isfinite(float(p_value)):
            continue
        candidates.append(
            {
                "missing_column": missing_column,
                "test_name": test_name,
                "sample_size": sample_size,
                "effect_size": float(effect),
                "p_value": float(p_value),
            }
        )

    if not candidates:
        return [], limitations
    adjusted = np.asarray(
        multipletests(
            [item["p_value"] for item in candidates],
            method="holm",
        )[1],
        dtype=float,
    )
    rows = [
        TargetMissingnessAssociation(
            missing_column=str(item["missing_column"]),
            target_column=target_column,
            test_name=item["test_name"],
            sample_size=item["sample_size"],
            effect_size=round(item["effect_size"], 6),
            p_value=round(item["p_value"], 12),
            adjusted_p_value=round(float(adjusted[index]), 12),
        )
        for index, item in enumerate(candidates)
    ]
    return sorted(
        rows,
        key=lambda item: (item.adjusted_p_value, item.missing_column),
    ), limitations


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])
