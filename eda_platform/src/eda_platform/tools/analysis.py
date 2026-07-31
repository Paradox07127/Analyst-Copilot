from __future__ import annotations

import re
from heapq import heappush, heapreplace
from itertools import combinations
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.random_projection import SparseRandomProjection

from eda_platform.core.ids import make_artifact_id
from eda_platform.schemas.artifacts import (
    AnalysisTable,
    Artifact,
    ArtifactType,
    DatasetProfile,
)
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.value_parsing import parse_numeric_like

_EXACT_CORRELATION_MAX_COLUMNS = 128
_CORRELATION_SCREEN_ROWS = 4_096
_CORRELATION_PROJECTION_DIMENSIONS = 64
_CORRELATION_CANDIDATE_PAIRS = 256
_CORRELATION_BLOCK_COLUMNS = 256
_MAX_ASSOCIATION_LEVELS = 30
_MAX_ASSOCIATION_ROWS = 15


def create_analysis_tables(
    loaded: LoadedDataset,
    profile_artifact: Artifact,
    *,
    project_id: str,
    session_id: str,
) -> list[Artifact]:
    profile = DatasetProfile.model_validate(profile_artifact.payload)
    numeric_columns = [
        column.name
        for column in profile.columns_detail
        if column.semantic_type == "numeric" and column.name in loaded.frame.columns
    ]

    categorical_columns = [
        column.name
        for column in profile.columns_detail
        if column.semantic_type in {"categorical", "boolean"}
        and column.name in loaded.frame.columns
        and 1 < column.unique_count <= _MAX_ASSOCIATION_LEVELS
    ]

    tables = [
        _numeric_summary_table(loaded, profile, numeric_columns),
        _correlation_table(loaded, profile, numeric_columns),
        _association_table(loaded, profile, categorical_columns, numeric_columns),
    ]
    artifacts: list[Artifact] = []
    for table in tables:
        if table is None or not table.rows:
            continue
        payload = table.model_dump(mode="json")
        artifacts.append(
            Artifact(
                id=make_artifact_id("table", payload),
                type=ArtifactType.TABLE,
                project_id=project_id,
                session_id=session_id,
                parents=[profile_artifact.id],
                payload=payload,
            )
        )
    return artifacts


def _association_table(
    loaded: LoadedDataset,
    profile: DatasetProfile,
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> AnalysisTable | None:
    """Strength of association for pairs a Pearson correlation cannot see.

    Cramér's V for category-category and the correlation ratio (eta) for
    category-measure. Both are on 0-1, so the rows sort together, and both are
    symmetric measures of association only — never of direction or cause.
    """
    if not categorical_columns:
        return None
    frame = loaded.frame
    rows: list[dict[str, Any]] = []
    for column_a, column_b in combinations(categorical_columns, 2):
        paired = frame.loc[:, [column_a, column_b]].dropna()
        if len(paired) < 3:
            continue
        table = pd.crosstab(paired[column_a], paired[column_b])
        if min(table.shape) < 2:
            continue
        value = _cramers_v_from_table(table)
        if value is None:
            continue
        rows.append(
            {
                "column_a": column_a,
                "column_b": column_b,
                "dataset": profile.name,
                "method": "cramers_v",
                "association": _round(value, digits=4),
                "pairwise_complete_n": int(len(paired)),
                "missing_policy": "pairwise_complete",
            }
        )
    for category in categorical_columns:
        for measure in numeric_columns:
            paired = pd.DataFrame(
                {
                    category: _column_series(frame, category),
                    measure: _numeric_series(
                        _column_series(frame, measure), column_name=measure
                    ),
                }
            ).dropna()
            if len(paired) < 3:
                continue
            value = _correlation_ratio(
                cast(pd.Series, paired[category]),
                cast(pd.Series, paired[measure]),
            )
            if value is None:
                continue
            rows.append(
                {
                    "column_a": category,
                    "column_b": measure,
                    "dataset": profile.name,
                    "method": "correlation_ratio",
                    "association": _round(value, digits=4),
                    "pairwise_complete_n": int(len(paired)),
                    "missing_policy": "pairwise_complete",
                }
            )
    if not rows:
        return None
    rows.sort(key=lambda row: (-float(row["association"]), str(row["column_a"])))
    return AnalysisTable(
        dataset_id=profile.dataset_id,
        title=f"{profile.name} - Categorical associations",
        kind="association",
        description=(
            f"Strongest categorical associations in {profile.name}: Cramér's V between "
            "categories and the correlation ratio between a category and a measure. "
            "Association is symmetric and is not evidence of causation."
        ),
        rows=rows[:_MAX_ASSOCIATION_ROWS],
    )


def _cramers_v_from_table(table: pd.DataFrame) -> float | None:
    """Bias-corrected Cramér's V; None when the table cannot support one."""
    observed = table.to_numpy(dtype="float64")
    total = float(observed.sum())
    if total <= 0:
        return None
    row_totals = observed.sum(axis=1, keepdims=True)
    column_totals = observed.sum(axis=0, keepdims=True)
    expected = row_totals @ column_totals / total
    if not np.all(expected > 0):
        return None
    chi_square = float(((observed - expected) ** 2 / expected).sum())
    rows, columns = observed.shape
    # Bergsma-Wicher correction: raw V is biased upward on small samples and
    # would report a strong association between two near-random columns.
    phi_squared = max(0.0, chi_square / total - (rows - 1) * (columns - 1) / (total - 1))
    corrected_rows = rows - (rows - 1) ** 2 / (total - 1)
    corrected_columns = columns - (columns - 1) ** 2 / (total - 1)
    denominator = min(corrected_rows - 1, corrected_columns - 1)
    if denominator <= 0:
        return None
    return float(np.sqrt(phi_squared / denominator))


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float | None:
    """Eta: the share of a measure's variance explained by group membership."""
    numeric = pd.to_numeric(values, errors="coerce")
    frame = pd.DataFrame({"group": categories.astype(str), "value": numeric}).dropna()
    if frame["group"].nunique() < 2:
        return None
    grand_mean = float(cast(Any, frame["value"].mean()))
    total = float(cast(Any, ((frame["value"] - grand_mean) ** 2).sum()))
    if total <= 0:
        return None
    between = 0.0
    for _, group in frame.groupby("group", sort=False):
        group_mean = float(cast(Any, cast(pd.Series, group["value"]).mean()))
        between += len(group) * (group_mean - grand_mean) ** 2
    return float(np.sqrt(max(0.0, min(1.0, between / total))))


def _numeric_summary_table(
    loaded: LoadedDataset,
    profile: DatasetProfile,
    numeric_columns: list[str],
) -> AnalysisTable | None:
    rows: list[dict[str, Any]] = []
    for column in numeric_columns:
        numeric = _numeric_series(_column_series(loaded.frame, column), column_name=column)
        non_null = numeric.dropna()
        if non_null.empty:
            continue
        # p1/p95/p99 are what reveal capping and long tails; quartiles hide both.
        p1, q1, median, q3, p95, p99 = (
            float(value)
            for value in non_null.quantile([0.01, 0.25, 0.5, 0.75, 0.95, 0.99])
        )
        rows.append(
            {
                "column": column,
                "dataset": profile.name,
                "count": int(non_null.count()),
                "missing": int(numeric.isna().sum()),
                # NumPy's sort-based exact cardinality releases its temporary
                # workspace per column; repeated pandas hash tables retained
                # hundreds of MiB of allocator arenas on high-cardinality data.
                "distinct": int(np.unique(non_null.to_numpy()).size),
                "mean": _round(non_null.mean()),
                "std": _round(non_null.std()),
                "median": _round(median),
                "p1": _round(p1),
                "q1": _round(q1),
                "q3": _round(q3),
                "p95": _round(p95),
                "p99": _round(p99),
                "iqr": _round(q3 - q1),
                "skew": _round(non_null.skew()),
                "kurtosis": _round(non_null.kurtosis()),
                "min": _round(non_null.min()),
                "max": _round(non_null.max()),
                "zero_count": int((non_null == 0).sum()),
                "negative_count": int((non_null < 0).sum()),
            }
        )
    if not rows:
        return None
    return AnalysisTable(
        dataset_id=profile.dataset_id,
        title=f"{profile.name} - Numeric summary",
        kind="numeric_summary",
        description=f"Summary statistics for numeric fields in {profile.name}.",
        rows=rows,
    )


def _correlation_table(
    loaded: LoadedDataset,
    profile: DatasetProfile,
    numeric_columns: list[str],
) -> AnalysisTable | None:
    if len(numeric_columns) < 2:
        return None
    if len(numeric_columns) > _EXACT_CORRELATION_MAX_COLUMNS:
        return _wide_correlation_table(loaded, profile, numeric_columns)
    numeric_series = {
        column: series
        for column in numeric_columns
        for series in [
            _numeric_series(_column_series(loaded.frame, column), column_name=column)
        ]
        if series.dropna().nunique() > 1
    }
    numeric_columns = [column for column in numeric_columns if column in numeric_series]
    if len(numeric_columns) < 2:
        return None
    # Pearson stays exact, but is evaluated one pair at a time. A complete
    # DataFrame.corr call materializes another dense n×p float buffer; on the
    # 310 MiB credit-card table that erased the profiler's memory savings.
    # Spearman is still computed only for the ten rows the artifact publishes.
    strongest: list[tuple[float, int, str, str, float]] = []
    for pair_order, (column_a, column_b) in enumerate(
        combinations(numeric_columns, 2)
    ):
        # A pair can still be constant on its pairwise-complete rows; NaN from
        # the suppressed divide is skipped below instead of warning.
        with np.errstate(invalid="ignore", divide="ignore"):
            pearson = float(
                numeric_series[column_a].corr(
                    numeric_series[column_b],
                    method="pearson",
                    min_periods=3,
                )
            )
        if pd.isna(pearson):
            continue
        # -pair_order keeps the earlier input-column pair on exact ties,
        # matching the previous stable full-list sort.
        candidate = (abs(pearson), -pair_order, column_a, column_b, pearson)
        if len(strongest) < 10:
            heappush(strongest, candidate)
        elif candidate[:2] > strongest[0][:2]:
            heapreplace(strongest, candidate)

    rows: list[dict[str, Any]] = []
    for _, _, column_a, column_b, pearson in sorted(
        strongest,
        key=lambda item: (-item[0], -item[1]),
    ):
        paired = pd.DataFrame(
            {
                column_a: numeric_series[column_a],
                column_b: numeric_series[column_b],
            }
        ).dropna()
        pairwise_count = int(len(paired))
        paired_a = cast(pd.Series, paired[column_a])
        paired_b = cast(pd.Series, paired[column_b])
        with np.errstate(invalid="ignore", divide="ignore"):
            spearman_value = paired_a.corr(
                paired_b,
                method="spearman",
            )
        spearman = None if pd.isna(spearman_value) else float(spearman_value)
        rows.append(
            {
                "column_a": column_a,
                "column_b": column_b,
                "dataset": profile.name,
                "pearson": _round(pearson, digits=4),
                "spearman": _round(spearman, digits=4),
                "abs_pearson": _round(abs(pearson), digits=4),
                "is_trivial_pair": _is_trivial_pair(
                    column_a,
                    column_b,
                    paired,
                    pearson=pearson,
                ),
                "sample_size": pairwise_count,
                "pairwise_complete_n": pairwise_count,
                "excluded_pair_n": int(len(loaded.frame) - pairwise_count),
                "missing_policy": "pairwise_complete",
            }
        )
    if not rows:
        return None
    return AnalysisTable(
        dataset_id=profile.dataset_id,
        title=f"{profile.name} - Numeric correlations",
        kind="correlation",
        description=f"Top Pearson correlations between numeric fields in {profile.name}.",
        rows=rows,
    )


def _wide_correlation_table(
    loaded: LoadedDataset,
    profile: DatasetProfile,
    numeric_columns: list[str],
) -> AnalysisTable | None:
    """Bound ultra-wide correlation discovery with a deterministic two-stage search.

    A sparse random projection screens pair candidates on bounded rows. Pearson
    and Spearman values published in the artifact are then recomputed exactly on
    all pairwise-complete rows; only pair *selection* is approximate.
    """
    frame = loaded.frame
    row_count = len(frame)
    sample_size = min(row_count, _CORRELATION_SCREEN_ROWS)
    if sample_size < 3:
        return None
    positions = np.linspace(0, row_count - 1, num=sample_size, dtype=np.int64)
    sampled = pd.DataFrame(
        {
            column: _numeric_series(
                _column_series(frame, column).iloc[positions],
                column_name=column,
            )
            for column in numeric_columns
        }
    )
    values = sampled.to_numpy(dtype="float64", na_value=np.nan)
    means = np.nanmean(values, axis=0)
    centered = values - means
    centered[~np.isfinite(centered)] = 0.0
    vectors = centered.T
    norms = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(norms) & (norms > 0)
    if int(valid.sum()) < 2:
        return None
    valid_indices = np.flatnonzero(valid)
    vectors = vectors[valid] / norms[valid, None]
    dimensions = min(_CORRELATION_PROJECTION_DIMENSIONS, max(2, vectors.shape[1] - 1))
    projected = SparseRandomProjection(
        n_components=cast(Any, dimensions),
        dense_output=True,
        random_state=0,
    ).fit_transform(vectors)
    projected_norms = np.linalg.norm(projected, axis=1)
    projected = projected / np.where(projected_norms > 0, projected_norms, 1.0)[:, None]

    candidates: list[tuple[float, int, int]] = []
    valid_count = len(valid_indices)
    for start in range(0, valid_count, _CORRELATION_BLOCK_COLUMNS):
        stop = min(start + _CORRELATION_BLOCK_COLUMNS, valid_count)
        scores = np.abs(projected[start:stop] @ projected.T)
        for local_index, global_index in enumerate(range(start, stop)):
            scores[local_index, : global_index + 1] = -np.inf
        finite_positions = np.flatnonzero(np.isfinite(scores))
        if finite_positions.size == 0:
            continue
        take = min(_CORRELATION_CANDIDATE_PAIRS, finite_positions.size)
        selected = finite_positions[
            np.argpartition(scores.ravel()[finite_positions], -take)[-take:]
        ]
        for flat_index in selected:
            local_index, other_index = np.unravel_index(flat_index, scores.shape)
            column_index_a = int(valid_indices[start + local_index])
            column_index_b = int(valid_indices[other_index])
            pair_order = _pair_order(
                column_index_a,
                column_index_b,
                len(numeric_columns),
            )
            candidate = (
                float(scores[local_index, other_index]),
                -pair_order,
                pair_order,
            )
            if len(candidates) < _CORRELATION_CANDIDATE_PAIRS:
                heappush(candidates, candidate)
            elif candidate[:2] > candidates[0][:2]:
                heapreplace(candidates, candidate)

    exact_strongest: list[tuple[float, int, str, str, float]] = []
    series_cache: dict[str, pd.Series] = {}
    for _, _, pair_order in candidates:
        column_index_a, column_index_b = _pair_from_order(
            pair_order,
            len(numeric_columns),
        )
        column_a = numeric_columns[column_index_a]
        column_b = numeric_columns[column_index_b]
        series_a = series_cache.setdefault(
            column_a,
            _numeric_series(_column_series(frame, column_a), column_name=column_a),
        )
        series_b = series_cache.setdefault(
            column_b,
            _numeric_series(_column_series(frame, column_b), column_name=column_b),
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            pearson = float(series_a.corr(series_b, method="pearson", min_periods=3))
        if pd.isna(pearson):
            continue
        candidate = (abs(pearson), -pair_order, column_a, column_b, pearson)
        if len(exact_strongest) < 10:
            heappush(exact_strongest, candidate)
        elif candidate[:2] > exact_strongest[0][:2]:
            heapreplace(exact_strongest, candidate)

    rows: list[dict[str, Any]] = []
    for _, _, column_a, column_b, pearson in sorted(
        exact_strongest,
        key=lambda item: (-item[0], -item[1]),
    ):
        paired = pd.DataFrame(
            {
                column_a: series_cache[column_a],
                column_b: series_cache[column_b],
            }
        ).dropna()
        pairwise_count = int(len(paired))
        paired_a = cast(pd.Series, paired[column_a])
        paired_b = cast(pd.Series, paired[column_b])
        with np.errstate(invalid="ignore", divide="ignore"):
            spearman_value = paired_a.corr(paired_b, method="spearman")
        pair_frame = cast(pd.DataFrame, paired[[column_a, column_b]])
        rows.append(
            {
                "column_a": column_a,
                "column_b": column_b,
                "dataset": profile.name,
                "pearson": _round(pearson, digits=4),
                "spearman": _round(spearman_value, digits=4),
                "abs_pearson": _round(abs(pearson), digits=4),
                "is_trivial_pair": _is_trivial_pair(
                    column_a,
                    column_b,
                    pair_frame,
                    pearson=pearson,
                ),
                "sample_size": pairwise_count,
                "pairwise_complete_n": pairwise_count,
                "excluded_pair_n": int(row_count - pairwise_count),
                "missing_policy": "pairwise_complete",
                "selection_method": "sparse_random_projection_then_exact",
                "selection_is_approximate": True,
                "screening_rows": sample_size,
                "candidate_pairs": len(candidates),
                "numeric_column_count": len(numeric_columns),
            }
        )
    if not rows:
        return None
    return AnalysisTable(
        dataset_id=profile.dataset_id,
        title=f"{profile.name} - Numeric correlations",
        kind="correlation",
        description=(
            f"Top exact Pearson correlations selected by bounded screening across "
            f"{len(numeric_columns)} numeric fields in {profile.name}; selection is approximate."
        ),
        rows=rows,
    )


def _pair_order(column_index_a: int, column_index_b: int, column_count: int) -> int:
    """Zero-based position of (a, b) in itertools.combinations(range(p), 2)."""
    return (
        column_index_a * (2 * column_count - column_index_a - 1) // 2
        + column_index_b
        - column_index_a
        - 1
    )


def _pair_from_order(pair_order: int, column_count: int) -> tuple[int, int]:
    remaining = pair_order
    for column_index_a in range(column_count - 1):
        row_width = column_count - column_index_a - 1
        if remaining < row_width:
            return column_index_a, column_index_a + 1 + remaining
        remaining -= row_width
    raise ValueError(f"Invalid pair order {pair_order} for {column_count} columns")


def _numeric_series(series: pd.Series, *, column_name: str) -> pd.Series:
    if is_numeric_dtype(series):
        return cast(pd.Series, pd.to_numeric(series, errors="coerce"))
    return pd.Series(
        [parse_numeric_like(value, column_name=column_name) for value in series],
        index=series.index,
        dtype="float64",
    )


def _column_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def _is_trivial_pair(
    column_a: str,
    column_b: str,
    numeric_frame: pd.DataFrame,
    *,
    pearson: float,
) -> bool:
    abs_r = abs(pearson)
    if abs_r >= 0.999:
        return True
    if abs_r < 0.97:
        return False
    if _complement_names(column_a, column_b):
        return True
    if _sum_is_near_constant(
        cast(pd.Series, numeric_frame[column_a]),
        cast(pd.Series, numeric_frame[column_b]),
    ):
        return True
    return _same_stem_rescale(column_a, column_b)


def _complement_names(column_a: str, column_b: str) -> bool:
    left = _name_tokens(column_a)
    right = _name_tokens(column_b)
    if len(left) >= 2 and len(right) >= 2:
        if left[0] == "home" and right[0] == "away" and left[1:] == right[1:]:
            return True
        if left[0] == "away" and right[0] == "home" and left[1:] == right[1:]:
            return True
    return _strip_against_suffix(left) == right or _strip_against_suffix(right) == left


def _strip_against_suffix(tokens: list[str]) -> list[str]:
    return tokens[:-1] if tokens[-1:] == ["against"] else tokens


def _sum_is_near_constant(series_a: pd.Series, series_b: pd.Series) -> bool:
    summed = cast(pd.Series, pd.DataFrame({"a": series_a, "b": series_b}).dropna().sum(axis=1))
    if len(summed) < 3:
        return False
    std = float(cast(float, summed.std()))
    mean = abs(float(summed.mean()))
    scale = max(mean, 1.0)
    return std / scale < 0.01


def _same_stem_rescale(column_a: str, column_b: str) -> bool:
    left = _rescale_token_set(column_a)
    right = _rescale_token_set(column_b)
    if not left or not right:
        return False
    return left == right


def _rescale_token_set(value: str) -> set[str]:
    normalized = re.sub(r"\bper[^0-9a-z]*90\b", "per90", value.lower())
    normalized = re.sub(r"\bp[^0-9a-z]*90\b", "p90", normalized)
    tokens = _name_tokens(normalized)
    while tokens and tokens[-1] in {"90", "90s", "rate", "ratio"}:
        tokens = tokens[:-1]
    return {
        token
        for token in tokens
        if token not in {"per90", "p90", "pct", "percent"}
    }


def _name_tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[^0-9a-z]+", value.lower()) if token]


def _round(value: Any, *, digits: int = 2) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)
