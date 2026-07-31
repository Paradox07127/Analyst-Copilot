from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from eda_platform.core.ids import make_artifact_id
from eda_platform.core.provenance import code_ref
from eda_platform.schemas.artifacts import (
    AnalysisTable,
    Artifact,
    ArtifactType,
    ColumnProfile,
    DatasetProfile,
)
from eda_platform.schemas.charts import ChartSpec
from eda_platform.tools.frame_stats import histogram_bin_count, prefers_log_scale
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.value_parsing import parse_numeric_like

logger = logging.getLogger(__name__)

_MAX_HIST_BINS = 30
_MAX_CATEGORIES = 10
_MAX_TIME_BUCKETS = 60
# A time-series line requires at least two periods.
_MIN_TIME_PERIODS = 2
_MAX_CORRELATION_COLUMNS = 15
_MAX_SCATTER_CHARTS = 3
_MAX_SCATTER_POINTS = 200
_MIN_SCATTER_CORRELATION = 0.5
_MAX_NUMERIC_DISTRIBUTIONS = 4
_MAX_CATEGORICAL_DISTRIBUTIONS = 4
_MAX_TIME_SERIES = 2
_MAX_MISSING_COLUMNS = 20


def create_chart_specs(
    loaded: LoadedDataset,
    profile_artifact: Artifact,
    *,
    project_id: str,
    session_id: str,
) -> list[Artifact]:
    """Build chart specs whose ``data`` is a full-column deterministic aggregate."""
    profile = DatasetProfile.model_validate(profile_artifact.payload)
    frame = loaded.frame
    # Each entry pairs the spec with its provenance: (spec, code_ref, plain_language).
    specs: list[tuple[ChartSpec, str, str]] = []

    missing_rows = _missingness_rows(profile)
    if missing_rows:
        specs.append(
            (
                ChartSpec(
                    dataset_id=profile.dataset_id,
                    title="Missing values by column",
                    description=(
                        "Missing-value counts and rates for affected columns, ordered by rate."
                    ),
                    category="quality",
                    mark="bar",
                    data={"values": missing_rows},
                    encoding={
                        "x": {"field": "column", "type": "nominal", "sort": "-y"},
                        "y": {"field": "missing_percent", "type": "quantitative"},
                        "tooltip": [
                            {"field": "missing_count", "type": "quantitative"},
                            {"field": "missing_percent", "type": "quantitative"},
                        ],
                    },
                ),
                code_ref(_missingness_rows),
                f"Missingness overview for {len(missing_rows)} affected column(s).",
            )
        )

    nullity_rows = _nullity_correlation_rows(frame, profile)
    if nullity_rows:
        specs.append(
            (
                ChartSpec(
                    dataset_id=profile.dataset_id,
                    title="Missingness association",
                    description=(
                        "Phi/Pearson association between binary missingness indicators; "
                        "values near ±1 show missing fields that co-occur or alternate."
                    ),
                    category="quality",
                    mark="rect",
                    data={"values": nullity_rows},
                    encoding={
                        "x": {"field": "column_a", "type": "nominal"},
                        "y": {"field": "column_b", "type": "nominal"},
                        "color": {
                            "field": "association",
                            "type": "quantitative",
                            "scale": {"domain": [-1, 1], "scheme": "redblue"},
                        },
                    },
                ),
                code_ref(_nullity_correlation_rows),
                "Missingness-association heatmap over columns with variable null patterns.",
            )
        )

    # Only genuinely continuous columns are binned into ranges. Flags and small
    # integer level sets fall through to the value-count path below.
    for numeric_column in _selected_columns(
        profile,
        {"numeric"},
        limit=_MAX_NUMERIC_DISTRIBUTIONS,
        distribution_kinds={"continuous"},
    ):
        if numeric_column.name not in frame.columns:
            continue
        histogram, log_scale = _histogram_rows(frame, numeric_column.name)
        if histogram:
            non_null = sum(int(row["count"]) for row in histogram)
            x_encoding: dict[str, Any] = {
                # bin:"binned" keeps zero-baseline stacking; without it a ranged
                # bar treats y as position and renders floating segments.
                "field": "bin_start",
                "type": "quantitative",
                "bin": {"binned": True},
                "title": numeric_column.name,
            }
            if log_scale:
                x_encoding["scale"] = {"type": "log"}
            specs.append(
                (
                    ChartSpec(
                        dataset_id=profile.dataset_id,
                        title=f"Distribution of {numeric_column.name}",
                        description=(
                            f"Histogram of {numeric_column.name} over all non-null rows"
                            + (
                                ", binned on a log scale because the values span "
                                "several orders of magnitude."
                                if log_scale
                                else "."
                            )
                        ),
                        category="distribution",
                        mark="bar",
                        data={"values": histogram},
                        encoding={
                            "x": x_encoding,
                            "x2": {"field": "bin_end"},
                            "y": {"field": "count", "type": "quantitative"},
                            "tooltip": [
                                {"field": "bin_label", "type": "nominal"},
                                {"field": "count", "type": "quantitative"},
                            ],
                        },
                    ),
                    code_ref(_histogram_rows),
                    f"Histogram of {numeric_column.name}: {len(histogram)} bins over "
                    f"{non_null:,} non-null rows.",
                )
            )

    for categorical_column in _selected_columns(
        profile,
        {"categorical", "boolean", "numeric"},
        limit=_MAX_CATEGORICAL_DISTRIBUTIONS,
        distribution_kinds={"binary", "discrete"},
    ):
        if categorical_column.name not in frame.columns:
            continue
        counts = _value_count_rows(frame, categorical_column.name)
        if counts:
            specs.append(
                (
                    ChartSpec(
                        dataset_id=profile.dataset_id,
                        title=f"Top values in {categorical_column.name}",
                        description=(
                            f"Top values for {categorical_column.name}; categories beyond the "
                            "display cap are combined as (other), preserving the full count."
                        ),
                        category="distribution",
                        mark="bar",
                        data={"values": counts},
                        encoding={
                            "x": {
                                "field": categorical_column.name,
                                "type": "nominal",
                                "sort": "-y",
                            },
                            "y": {"field": "count", "type": "quantitative"},
                        },
                    ),
                    code_ref(_value_count_rows),
                    f"Bar chart of the top {len(counts)} most frequent values of "
                    f"{categorical_column.name} across the full dataset.",
                )
            )

    for datetime_column in _selected_columns(
        profile,
        {"datetime"},
        limit=_MAX_TIME_SERIES,
    ):
        if datetime_column.name not in frame.columns:
            continue
        timeline = _timeline_rows(frame, datetime_column.name)
        if 0 < len(timeline) < _MIN_TIME_PERIODS:
            # DI10-W5: fewer than two effective periods cannot show a trend —
            # skip instead of emitting a single-point "line".
            logger.info(
                "timeline chart skipped for %s.%s: only %d effective time "
                "period(s) (< %d)",
                profile.name,
                datetime_column.name,
                len(timeline),
                _MIN_TIME_PERIODS,
            )
        elif timeline:
            specs.append(
                (
                    ChartSpec(
                        dataset_id=profile.dataset_id,
                        title=f"Records over {datetime_column.name}",
                        description=f"Record counts resampled over {datetime_column.name}.",
                        category="time",
                        mark="line",
                        data={"values": timeline},
                        encoding={
                            "x": {"field": "period", "type": "temporal"},
                            "y": {"field": "count", "type": "quantitative"},
                        },
                    ),
                    code_ref(_timeline_rows),
                    f"Line chart of record counts over {datetime_column.name} across "
                    f"{len(timeline)} time periods.",
                )
            )

    artifacts: list[Artifact] = []
    for spec, ref, plain_language in specs:
        payload = spec.model_dump(mode="json")
        artifacts.append(
            Artifact(
                id=make_artifact_id("chart", payload),
                type=ArtifactType.CHART_SPEC,
                project_id=project_id,
                session_id=session_id,
                parents=[profile_artifact.id],
                payload=payload,
                code_ref=ref,
                plain_language=plain_language,
            )
        )
    return artifacts


def create_association_chart_specs(
    association_artifact: Artifact,
    *,
    project_id: str,
    session_id: str,
) -> list[Artifact]:
    """One bar chart ranking categorical associations by strength."""
    table = AnalysisTable.model_validate(association_artifact.payload)
    if table.kind != "association" or not table.rows:
        return []
    values = [
        {
            "pair": f"{row['column_a']} ~ {row['column_b']}",
            "association": float(row["association"]),
            "method": str(row["method"]),
            "n": int(row["pairwise_complete_n"]),
        }
        for row in table.rows
    ]
    spec = ChartSpec(
        dataset_id=table.dataset_id,
        title=f"{table.title} strength",
        description=(
            "Cramér's V between categories and the correlation ratio between a category "
            "and a measure, both on 0-1. Association is symmetric, not causal."
        ),
        category="relationship",
        mark="bar",
        data={"values": values},
        encoding={
            "y": {"field": "pair", "type": "nominal", "sort": "-x"},
            "x": {
                "field": "association",
                "type": "quantitative",
                "scale": {"domain": [0, 1]},
            },
            "color": {"field": "method", "type": "nominal"},
            "tooltip": [
                {"field": "method", "type": "nominal"},
                {"field": "association", "type": "quantitative"},
                {"field": "n", "type": "quantitative"},
            ],
        },
    )
    payload = spec.model_dump(mode="json")
    return [
        Artifact(
            id=make_artifact_id("chart", payload),
            type=ArtifactType.CHART_SPEC,
            project_id=project_id,
            session_id=session_id,
            parents=[association_artifact.id],
            payload=payload,
            code_ref=code_ref(create_association_chart_specs),
            plain_language=f"Association strength for {len(values)} field pair(s).",
        )
    ]


def create_correlation_chart_specs(
    loaded: LoadedDataset,
    correlation_artifact: Artifact,
    *,
    project_id: str,
    session_id: str,
) -> list[Artifact]:
    """Create bounded heatmap and scatter charts from a correlation table."""
    table = AnalysisTable.model_validate(correlation_artifact.payload)
    if table.kind != "correlation":
        return []
    rows = _non_trivial_correlation_rows(table.rows)
    heatmap_rows = _heatmap_rows(rows)
    specs: list[tuple[ChartSpec, str, str]] = []
    if heatmap_rows:
        columns = {str(row["column_a"]) for row in heatmap_rows} | {
            str(row["column_b"]) for row in heatmap_rows
        }
        specs.append(
            (
                ChartSpec(
                    dataset_id=table.dataset_id,
                    title=f"{table.title} heatmap",
                    description=(
                        "Pearson correlations for the strongest non-trivial numeric "
                        f"relationships across {len(columns)} columns."
                    ),
                    category="relationship",
                    mark="rect",
                    data={"values": heatmap_rows},
                    encoding={
                        "x": {"field": "column_a", "type": "nominal"},
                        "y": {"field": "column_b", "type": "nominal"},
                        "color": {
                            "field": "pearson",
                            "type": "quantitative",
                            "scale": {"domain": [-1, 1], "scheme": "redblue"},
                        },
                    },
                ),
                code_ref(_heatmap_rows),
                f"Correlation heatmap of {len(columns)} numeric columns; known trivial "
                "pairs are excluded and color runs from -1 to 1.",
            )
        )

    for row in rows[:_MAX_SCATTER_CHARTS]:
        pearson = float(row["pearson"])
        if abs(pearson) < _MIN_SCATTER_CORRELATION:
            continue
        column_a = str(row["column_a"])
        column_b = str(row["column_b"])
        points = _scatter_rows(loaded.frame, column_a, column_b)
        if not points:
            continue
        observation = (
            f"Pearson r={pearson:.4g}; this correlation is an observation, not causation."
        )
        specs.append(
            (
                ChartSpec(
                    dataset_id=table.dataset_id,
                    title=f"{column_a} vs {column_b} ({observation})",
                    description=(
                        f"Scatter plot of {len(points)} paired values. {observation}"
                    ),
                    category="relationship",
                    mark="point",
                    data={"values": points},
                    encoding={
                        "x": {"field": column_a, "type": "quantitative"},
                        "y": {"field": column_b, "type": "quantitative"},
                    },
                ),
                code_ref(_scatter_rows),
                f"{len(points)} paired observations for {column_a} and {column_b}; "
                f"{observation}",
            )
        )

    artifacts: list[Artifact] = []
    for spec, ref, plain_language in specs:
        payload = spec.model_dump(mode="json")
        artifacts.append(
            Artifact(
                id=make_artifact_id("chart", payload),
                type=ArtifactType.CHART_SPEC,
                project_id=project_id,
                session_id=session_id,
                parents=[correlation_artifact.id],
                payload=payload,
                code_ref=ref,
                plain_language=plain_language,
            )
        )
    return artifacts


def _non_trivial_correlation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if not bool(row.get("is_trivial_pair"))
        and row.get("pearson") is not None
        and row.get("column_a") is not None
        and row.get("column_b") is not None
    ]
    return sorted(
        eligible,
        key=lambda row: (
            -abs(float(row["pearson"])),
            str(row["column_a"]),
            str(row["column_b"]),
        ),
    )


def _heatmap_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    involvement: dict[str, float] = {}
    for row in rows:
        strength = abs(float(row["pearson"]))
        for key in ("column_a", "column_b"):
            column = str(row[key])
            involvement[column] = involvement.get(column, 0.0) + strength
    selected = {
        column
        for column, _ in sorted(
            involvement.items(), key=lambda item: (-item[1], item[0])
        )[:_MAX_CORRELATION_COLUMNS]
    }
    return [
        {
            "column_a": str(row["column_a"]),
            "column_b": str(row["column_b"]),
            "pearson": float(row["pearson"]),
        }
        for row in rows
        if str(row["column_a"]) in selected and str(row["column_b"]) in selected
    ]


def _scatter_rows(frame: pd.DataFrame, column_a: str, column_b: str) -> list[dict[str, float]]:
    paired = pd.DataFrame(
        {
            column_a: _numeric_series(frame, column_a),
            column_b: _numeric_series(frame, column_b),
        }
    ).dropna()
    if len(paired) > _MAX_SCATTER_POINTS:
        paired = paired.sample(
            n=_MAX_SCATTER_POINTS,
            random_state=0,
            replace=False,
        ).sort_index()
    records = cast(list[dict[str, Any]], paired.to_dict(orient="records"))
    return [
        {column_a: float(row[column_a]), column_b: float(row[column_b])}
        for row in records
    ]


def _selected_columns(
    profile: DatasetProfile,
    semantic_types: set[str],
    *,
    limit: int,
    distribution_kinds: set[str] | None = None,
) -> list[ColumnProfile]:
    # Fully empty columns render nothing; letting them occupy slots left whole
    # tables without a single distribution chart (World Cup teams/players).
    eligible = [
        column
        for column in profile.columns_detail
        if column.semantic_type in semantic_types
        and column.missing_percent < 100.0
        and (distribution_kinds is None or column.distribution_kind in distribution_kinds)
    ]
    return sorted(
        eligible,
        key=lambda column: (
            -(column.missing_count > 0),
            -(column.outlier_count > 0),
            -column.missing_percent,
            profile.column_names.index(column.name),
        ),
    )[:limit]


def _missingness_rows(profile: DatasetProfile) -> list[dict[str, Any]]:
    rows = [
        {
            "column": column.name,
            "missing_count": column.missing_count,
            "missing_percent": column.missing_percent,
        }
        for column in profile.columns_detail
        if column.missing_count > 0
    ]
    rows.sort(key=lambda row: (-float(row["missing_percent"]), str(row["column"])))
    return rows[:_MAX_MISSING_COLUMNS]


def _nullity_correlation_rows(
    frame: pd.DataFrame,
    profile: DatasetProfile,
) -> list[dict[str, Any]]:
    columns = [
        column.name
        for column in profile.columns_detail
        if 0 < column.missing_count < profile.rows and column.name in frame.columns
    ][:_MAX_CORRELATION_COLUMNS]
    if len(columns) < 2:
        return []
    indicators = cast(pd.DataFrame, frame[columns]).isna().astype(float)
    correlation = indicators.corr()
    rows: list[dict[str, Any]] = []
    for column_a in columns:
        for column_b in columns:
            value = correlation.loc[column_a, column_b]
            if pd.isna(value):
                continue
            rows.append(
                {
                    "column_a": column_a,
                    "column_b": column_b,
                    "association": round(float(value), 4),
                }
            )
    return rows


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    series = cast(pd.Series, frame[column])
    if is_numeric_dtype(series):
        return cast(pd.Series, pd.to_numeric(series, errors="coerce")).dropna()
    parsed = pd.Series(
        [parse_numeric_like(value, column_name=column) for value in series],
        dtype="float64",
    )
    return parsed.dropna()


def _histogram_rows(
    frame: pd.DataFrame,
    column: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Adaptive-width histogram rows, plus whether they were binned in log space.

    Bin count follows Freedman-Diaconis rather than a fixed 30: the same
    constant over-resolves a 50-row column and under-resolves a million-row one.
    """
    values = _numeric_series(frame, column)
    if values.empty:
        return [], False
    log_scale = prefers_log_scale(values)
    binning_input = (
        cast(pd.Series, np.log10(values)) if log_scale else values
    )
    bins = min(histogram_bin_count(binning_input), max(1, int(values.nunique())))
    binned = cast(pd.Series, pd.cut(binning_input, bins=max(1, bins)))
    counts = binned.value_counts(sort=False)
    rows: list[dict[str, Any]] = []
    for interval, count in counts.items():
        interval = cast(pd.Interval, interval)
        start = float(interval.left)
        end = float(interval.right)
        if log_scale:
            start, end = float(10.0**start), float(10.0**end)
        rows.append(
            {
                "bin_start": start,
                "bin_end": end,
                "bin_label": f"{start:.6g}–{end:.6g}",
                "count": int(count),
            }
        )
    return rows, log_scale


def _value_count_rows(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    series = cast(pd.Series, frame[column])
    counts = series.astype("object").where(series.notna(), "(missing)").value_counts()
    if len(counts) <= _MAX_CATEGORIES:
        selected = counts
        other_count = 0
    else:
        selected = counts.head(_MAX_CATEGORIES - 1)
        other_count = int(counts.iloc[_MAX_CATEGORIES - 1 :].sum())
    rows = [{column: str(value), "count": int(count)} for value, count in selected.items()]
    if other_count:
        rows.append({column: "(other)", "count": other_count})
    return rows


def _timeline_rows(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    parsed = pd.to_datetime(cast(pd.Series, frame[column]), errors="coerce", format="mixed")
    parsed = parsed.dropna()
    if parsed.empty:
        return []
    span_days = (parsed.max() - parsed.min()).days
    if span_days <= _MAX_TIME_BUCKETS:
        freq = "D"
    elif span_days <= _MAX_TIME_BUCKETS * 31:
        freq = "MS"
    else:
        freq = "YS"
    periods = parsed.dt.to_period(freq[0])
    counts = periods.value_counts().sort_index()
    complete_index = pd.period_range(periods.min(), periods.max(), freq=freq[0])
    counts = counts.reindex(complete_index, fill_value=0)
    if len(counts) > _MAX_TIME_BUCKETS:
        step = max(1, (len(counts) + _MAX_TIME_BUCKETS - 1) // _MAX_TIME_BUCKETS)
        bucket = pd.Series(range(len(counts)), index=counts.index) // step
        counts = counts.groupby(bucket).sum()
        labels = [
            str(complete_index[min(index * step, len(complete_index) - 1)])
            for index in counts.index
        ]
        return [
            {"period": label, "count": int(count)}
            for label, count in zip(labels, counts.tolist(), strict=True)
        ]
    return [{"period": str(period), "count": int(count)} for period, count in counts.items()]
