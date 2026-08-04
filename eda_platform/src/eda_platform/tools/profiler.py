from __future__ import annotations

import math
import re
from collections.abc import Callable
from itertools import combinations
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from pandas.core.util.hashing import hash_pandas_object as _hash_pandas_object

from eda_platform.core.ids import make_artifact_id
from eda_platform.schemas.artifacts import Artifact, ArtifactType, ColumnProfile, DatasetProfile
from eda_platform.tools.frame_stats import distribution_kind, iqr_bounds
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.value_parsing import numeric_parse_success_percent, parse_numeric_like

SemanticType = Literal["numeric", "categorical", "datetime", "id", "boolean", "text", "unknown"]
hash_pandas_object = cast(
    Callable[..., pd.Series],
    _hash_pandas_object,
)

# Optional domain hint.
_DEFAULT_ENTITY_IDENTIFIER_NAMES = frozenset(
    {
        "name",
        "code",
        "country",
        "email",
    }
)

# Above this uniqueness (and with short values), an object column is treated as an
# identifier/key rather than a free-form category.
_GENERIC_ID_UNIQUE_PERCENT = 95.0
_GENERIC_ID_MAX_AVG_LENGTH = 40.0
_GENERIC_ID_MIN_ROWS = 20

# Integer surrogate keys: near-unique AND near-contiguous. PCA-style float
# columns are also near-100% unique, so this stays integer-only, and sparse
# unique measures (e.g. amounts in cents) fail the range check.
_INTEGER_ID_UNIQUE_PERCENT = 99.5
_INTEGER_ID_MAX_RANGE_FACTOR = 2.0


def profile_dataset(
    loaded: LoadedDataset,
    *,
    project_id: str,
    session_id: str,
    entity_identifier_names: frozenset[str] | None = None,
    parents: list[str] | None = None,
) -> Artifact:
    frame = loaded.frame
    entity_names = entity_identifier_names or _DEFAULT_ENTITY_IDENTIFIER_NAMES
    columns_detail = [
        _profile_column(frame, str(column), entity_names=entity_names) for column in frame.columns
    ]
    numeric_columns = [
        column.name for column in columns_detail if column.semantic_type == "numeric"
    ]
    categorical_columns = [
        column.name
        for column in columns_detail
        if column.semantic_type in {"categorical", "boolean", "id", "text"}
    ]
    semantic_type_counts: dict[str, int] = {}
    for column in columns_detail:
        semantic_type_counts[column.semantic_type] = (
            semantic_type_counts.get(column.semantic_type, 0) + 1
        )
    primary_key_candidates = [
        column.name
        for column in columns_detail
        if column.semantic_type == "id"
        and column.missing_count == 0
        and column.unique_count == frame.shape[0]
    ]
    composite_key_candidates = (
        [] if primary_key_candidates else _composite_key_candidates(frame, columns_detail)
    )
    grain = _grain_statement(primary_key_candidates, composite_key_candidates)
    exact_duplicate_rows = _exact_duplicate_count(frame)
    duplicate_rows, duplicate_scope_columns = _duplicate_row_summary(
        frame,
        columns_detail,
        exact_duplicate_rows=exact_duplicate_rows,
    )
    profile = DatasetProfile(
        dataset_id=loaded.record.dataset_id,
        name=loaded.record.name,
        content_hash=loaded.record.content_hash,
        encoding=loaded.record.encoding,
        delimiter=loaded.record.delimiter,
        rows=int(frame.shape[0]),
        columns=int(frame.shape[1]),
        column_names=[str(column) for column in frame.columns],
        dtypes={str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        missing_values={
            column.name: column.missing_count for column in columns_detail
        },
        missing_percent={
            column.name: column.missing_percent for column in columns_detail
        },
        numeric_columns=[str(column) for column in numeric_columns],
        categorical_columns=[str(column) for column in categorical_columns],
        sample_rows=_sample_rows(frame),
        duplicate_rows=duplicate_rows,
        exact_duplicate_rows=exact_duplicate_rows,
        duplicate_scope_columns=duplicate_scope_columns,
        columns_detail=columns_detail,
        semantic_type_counts=semantic_type_counts,
        primary_key_candidates=primary_key_candidates,
        composite_key_candidates=composite_key_candidates,
        grain=grain,
    )
    payload = profile.model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("prof", payload),
        type=ArtifactType.DATASET_PROFILE,
        project_id=project_id,
        session_id=session_id,
        parents=list(parents) if parents else [],
        payload=payload,
    )


_MAX_COMPOSITE_KEY_WIDTH = 3
_MAX_COMPOSITE_KEY_COLUMNS = 12
_MAX_COMPOSITE_KEY_RESULTS = 3
# One surviving column cannot distinguish rows; two already can.
_MIN_DUPLICATE_SCOPE_COLUMNS = 2


def _composite_key_candidates(
    frame: pd.DataFrame,
    columns_detail: list[ColumnProfile],
) -> list[list[str]]:
    """Smallest column combinations that uniquely identify a row.

    Real fact tables are keyed by (entity, period), so single-column detection
    reports "no key" on exactly the tables whose grain matters most. Search is
    bounded to keep a wide table from exploding combinatorially, and a
    combination is only reported if no subset of it is already a key.
    """
    row_count = int(frame.shape[0])
    if row_count == 0:
        return []
    eligible = [
        column.name
        for column in columns_detail
        if column.semantic_type in {"id", "categorical", "datetime", "boolean"}
        and column.missing_count == 0
        and 1 < column.unique_count < row_count
        and column.name in frame.columns
    ]
    # Fewer distinct values first: a key is more likely to include the coarse
    # entity column than a near-unique one, and this keeps the search cheap.
    column_order = {str(name): index for index, name in enumerate(frame.columns)}
    eligible.sort(key=lambda name: frame[name].nunique())
    eligible = eligible[:_MAX_COMPOSITE_KEY_COLUMNS]
    found: list[list[str]] = []
    for width in range(2, _MAX_COMPOSITE_KEY_WIDTH + 1):
        for combination in combinations(eligible, width):
            if any(set(key).issubset(combination) for key in found):
                continue
            if not frame.loc[:, list(combination)].duplicated().any():
                # Cardinality order drives the search; report in table order,
                # which is how a key is written and read.
                found.append(sorted(combination, key=column_order.__getitem__))
                if len(found) >= _MAX_COMPOSITE_KEY_RESULTS:
                    return found
        if found:
            break
    return found


def _grain_statement(
    primary_key_candidates: list[str],
    composite_key_candidates: list[list[str]],
) -> str:
    if primary_key_candidates:
        return f"One row per {primary_key_candidates[0]}."
    if composite_key_candidates:
        joined = " + ".join(composite_key_candidates[0])
        return f"One row per unique ({joined}) combination."
    return (
        "No column combination checked identifies a row uniquely; the grain is "
        "not established and must be confirmed before joining or aggregating."
    )


def _duplicate_row_summary(
    frame: pd.DataFrame,
    columns_detail: list[ColumnProfile],
    *,
    exact_duplicate_rows: int,
) -> tuple[int, list[str]]:
    """Count duplicates over non-id columns: a surrogate key differs on every
    row and would otherwise mask true payload duplicates.

    A single surviving column is not a duplicate signal: a 7-row table whose
    only non-id column is a boolean reports 5 "duplicate rows", which is the
    definition of a boolean rather than a defect. Three such false alarms
    reached the Limitations section of the 2026-08-04 World Cup report. The
    floor is two columns and no higher -- ``(region, amount)`` and
    ``(amount, category)`` are the two-column scopes that catch the real
    payload duplicates these tests encode.
    """
    id_names = {column.name for column in columns_detail if column.semantic_type == "id"}
    payload_columns = [column for column in frame.columns if str(column) not in id_names]
    if (
        len(payload_columns) < _MIN_DUPLICATE_SCOPE_COLUMNS
        or len(payload_columns) == len(frame.columns)
    ):
        return exact_duplicate_rows, [str(column) for column in frame.columns]
    return (
        _exact_duplicate_count(frame, subset=payload_columns),
        [str(column) for column in payload_columns],
    )


def _exact_duplicate_count(
    frame: pd.DataFrame,
    *,
    subset: list[Any] | None = None,
) -> int:
    """Count exact duplicates without materializing pandas' wide row hashtable.

    A compact 64-bit row hash finds the only rows that can possibly be
    duplicates. The second pass compares those candidate rows exactly, so hash
    collisions cannot inflate the result.
    """
    selected = frame if subset is None else frame.loc[:, subset]
    if selected.empty:
        return 0
    row_hashes = hash_pandas_object(selected, index=False, categorize=True)
    candidates = row_hashes.duplicated(keep=False)
    if not bool(candidates.any()):
        return 0
    return int(selected.loc[candidates].duplicated().sum())


def _sample_rows(frame: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    sample = frame.head(limit).astype(object).where(pd.notna(frame.head(limit)), None)
    return [{str(key): value for key, value in row.items()} for row in sample.to_dict("records")]


def _profile_column(
    frame: pd.DataFrame,
    column_name: str,
    *,
    entity_names: frozenset[str],
) -> ColumnProfile:
    series = cast(pd.Series, frame[column_name])
    row_count = max(int(len(series)), 1)
    missing_count = int(series.isna().sum())
    missing_percent = round(missing_count / row_count * 100, 2)
    unique_count = _unique_count(series)
    unique_percent = round(unique_count / row_count * 100, 2)
    non_null = series.dropna()
    numeric_dtype = is_numeric_dtype(series)
    temporal_name = _has_temporal_name_token(column_name.lower().strip())
    # Native numeric columns need neither Python-level numeric parsing nor a
    # speculative full-column datetime conversion. Only temporal-name numeric
    # columns need the bounded range/date check; all string-like columns still
    # receive full parse-failure accounting after semantic inference.
    datetime_parse_success_percent = (
        _datetime_parse_success_percent(series)
        if not numeric_dtype or temporal_name
        else None
    )
    numeric_string_success_percent = (
        None
        if numeric_dtype
        else numeric_parse_success_percent(
            non_null,
            column_name=column_name,
        )
    )
    semantic_type, warnings, parse_success_percent = _infer_semantic_type(
        column_name=column_name,
        series=series,
        unique_percent=unique_percent,
        unique_count=unique_count,
        row_count=row_count,
        datetime_parse_success_percent=datetime_parse_success_percent,
        numeric_string_success_percent=numeric_string_success_percent,
        entity_names=entity_names,
    )
    parse_failure_count = _parse_failure_count(
        series,
        column_name=column_name,
        semantic_type=semantic_type,
    )
    non_finite_count = _non_finite_count(
        series,
        column_name=column_name,
        semantic_type=semantic_type,
    )
    whitespace_count = _whitespace_count(series)
    if parse_failure_count > 0:
        warnings.append("parse_failure")
    if non_finite_count > 0:
        warnings.append("non_finite")
    if whitespace_count > 0:
        warnings.append("surrounding_whitespace")
    outlier_count = _iqr_outlier_count(series, column_name) if semantic_type == "numeric" else 0
    if outlier_count > 0:
        warnings.append("has_outliers")
    return ColumnProfile(
        name=column_name,
        dtype=str(series.dtype),
        semantic_type=semantic_type,
        missing_count=missing_count,
        missing_percent=missing_percent,
        unique_count=unique_count,
        unique_percent=unique_percent,
        sample_values=_sample_values(series),
        category_levels=_category_levels(series, semantic_type=semantic_type),
        distribution_kind=distribution_kind(
            _shape_input_series(
                series,
                column_name=column_name,
                semantic_type=semantic_type,
                numeric_dtype=numeric_dtype,
            )
        ),
        parse_success_percent=parse_success_percent,
        parse_failure_count=parse_failure_count,
        non_finite_count=non_finite_count,
        whitespace_count=whitespace_count,
        outlier_count=outlier_count,
        warnings=warnings,
    )


_MAX_CATEGORY_LEVELS = 30


def _category_levels(
    series: pd.Series,
    *,
    semantic_type: SemanticType,
) -> list[dict[str, Any]]:
    """Observed values of a coded column, most frequent first.

    Bounded: past a few dozen levels this is a free-text field, and listing it
    for value-by-value confirmation would be busywork rather than knowledge.
    """
    if semantic_type not in {"categorical", "boolean"}:
        return []
    non_null = series.dropna()
    if non_null.empty:
        return []
    counts = non_null.astype(str).value_counts()
    if len(counts) > _MAX_CATEGORY_LEVELS:
        return []
    return [
        {"value": str(value), "count": int(count)} for value, count in counts.items()
    ]


def _shape_input_series(
    series: pd.Series,
    *,
    column_name: str,
    semantic_type: SemanticType,
    numeric_dtype: bool,
) -> pd.Series:
    """The values distribution shape should be judged on.

    A column stored as text but read as numeric ("27-003" -> 27) must be shaped
    on its parsed values, or every such column looks like an unbinnable category.
    """
    if semantic_type != "numeric" or numeric_dtype:
        return series
    return pd.Series(
        [parse_numeric_like(value, column_name=column_name) for value in series.dropna()],
        dtype="float64",
    )


def _unique_count(series: pd.Series) -> int:
    """Exact cardinality with bounded temporary memory for native numerics.

    pandas' hash table is fast, but repeated high-cardinality numeric columns
    retain large allocator arenas. NumPy's sort-based path is slightly slower
    per column and returns its temporary workspace promptly.
    """
    non_null = series.dropna()
    if is_numeric_dtype(series) and not is_object_dtype(series):
        return int(np.unique(non_null.to_numpy()).size)
    return int(non_null.nunique(dropna=False))


def _parse_failure_count(
    series: pd.Series,
    *,
    column_name: str,
    semantic_type: SemanticType,
) -> int:
    non_null = series.dropna()
    if non_null.empty:
        return 0
    if semantic_type == "datetime":
        parsed = _parse_datetime_values(non_null)
        return int(parsed.isna().sum())
    if semantic_type == "numeric" and not is_numeric_dtype(series):
        parsed = pd.Series(
            [parse_numeric_like(value, column_name=column_name) for value in non_null],
            dtype="float64",
        )
        return int(parsed.isna().sum())
    return 0


def _non_finite_count(
    series: pd.Series,
    *,
    column_name: str,
    semantic_type: SemanticType,
) -> int:
    if semantic_type != "numeric":
        return 0
    if is_numeric_dtype(series):
        numeric = cast(pd.Series, pd.to_numeric(series, errors="coerce")).dropna()
    else:
        numeric = pd.Series(
            [parse_numeric_like(value, column_name=column_name) for value in series.dropna()],
            dtype="float64",
        ).dropna()
    return int(sum(not math.isfinite(float(value)) for value in numeric))


def _whitespace_count(series: pd.Series) -> int:
    if not (
        is_object_dtype(series)
        or is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return 0
    non_null = series.dropna().astype(str)
    return int((non_null != non_null.str.strip()).sum())


def _iqr_outlier_count(series: pd.Series, column_name: str) -> int:
    if is_numeric_dtype(series):
        numeric = cast(pd.Series, pd.to_numeric(series, errors="coerce")).dropna()
    else:
        numeric = pd.Series(
            [parse_numeric_like(value, column_name=column_name) for value in series.dropna()],
            dtype="float64",
        ).dropna()
    bounds = iqr_bounds(numeric)
    if bounds is None:
        return 0
    lower, upper = bounds
    return int(((numeric < lower) | (numeric > upper)).sum())


def _infer_semantic_type(
    *,
    column_name: str,
    series: pd.Series,
    unique_percent: float,
    unique_count: int,
    row_count: int,
    datetime_parse_success_percent: float | None,
    numeric_string_success_percent: float | None,
    entity_names: frozenset[str],
) -> tuple[SemanticType, list[str], float | None]:
    normalized_name = column_name.lower().strip()
    warnings: list[str] = []
    non_null = series.dropna()

    if _looks_like_id_name(normalized_name) and unique_percent >= 60:
        return "id", warnings, None

    if _looks_like_entity_identifier(normalized_name, entity_names) and unique_percent >= 80:
        return "id", warnings, None

    if is_bool_dtype(series) or _is_boolean_like(non_null):
        return "boolean", warnings, None

    temporal_name = _has_temporal_name_token(normalized_name)

    if is_numeric_dtype(series) and not (
        temporal_name and _numeric_values_look_temporal(non_null)
    ):
        # Naming alone must never beat the values: "TrainingTimesLastYear" holds
        # counts 0-6, and reading it as a date produced fake trend questions and
        # a bogus time-coverage metric (2026-07-22 audit).
        if _looks_like_integer_id_sequence(
            series,
            non_null,
            unique_percent=unique_percent,
            unique_count=unique_count,
            row_count=row_count,
        ):
            return "id", warnings, None
        return "numeric", warnings, None

    if (
        numeric_string_success_percent is not None
        and numeric_string_success_percent >= 90
        and not temporal_name
    ):
        warnings.append("numeric_string")
        return "numeric", warnings, numeric_string_success_percent

    if datetime_parse_success_percent is not None and (
        temporal_name or datetime_parse_success_percent >= 90
    ):
        if datetime_parse_success_percent < 90:
            warnings.append("date_parse_failure")
        return "datetime", warnings, datetime_parse_success_percent

    if is_numeric_dtype(series):
        return "numeric", warnings, None

    if non_null.empty:
        return "unknown", warnings, None

    average_length = non_null.astype(str).map(len).mean()
    if _looks_mixed_type(non_null):
        warnings.append("mixed_type_string")

    # Generic (domain-agnostic) identifier heuristic: a near-unique column of
    # short values behaves like a key/identifier regardless of its column name.
    if (
        unique_percent >= _GENERIC_ID_UNIQUE_PERCENT
        and average_length <= _GENERIC_ID_MAX_AVG_LENGTH
        and row_count >= _GENERIC_ID_MIN_ROWS
    ):
        return "id", warnings, None

    if average_length >= 24:
        return "text", warnings, None

    if unique_count <= max(20, int(row_count * 0.5)):
        return "categorical", warnings, None

    if (
        is_object_dtype(series)
        or is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return ("text" if average_length >= 16 else "categorical"), warnings, None

    return "unknown", warnings, None


def _looks_like_integer_id_sequence(
    series: pd.Series,
    non_null: pd.Series,
    *,
    unique_percent: float,
    unique_count: int,
    row_count: int,
) -> bool:
    """Near-unique integers covering a near-contiguous range read as a surrogate
    key (e.g. a shuffled 1..N respondent id) that averaging would corrupt."""
    if not is_integer_dtype(series):
        return False
    if row_count < _GENERIC_ID_MIN_ROWS or unique_count <= 0:
        return False
    if unique_percent < _INTEGER_ID_UNIQUE_PERCENT:
        return False
    low = int(non_null.min())
    high = int(non_null.max())
    return (high - low + 1) <= _INTEGER_ID_MAX_RANGE_FACTOR * unique_count


def _looks_like_id_name(normalized_name: str) -> bool:
    return (
        normalized_name == "id"
        or normalized_name.endswith("_id")
        or normalized_name.endswith(" id")
    )


def _looks_like_entity_identifier(normalized_name: str, entity_names: frozenset[str]) -> bool:
    name = normalized_name.replace(" ", "_")
    for prefix in ("home_", "away_"):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
    if name in entity_names:
        return True
    return name.endswith("_name") or name.endswith("_code")


def _is_boolean_like(series: pd.Series) -> bool:
    if series.empty:
        return False
    values = {str(value).strip().lower() for value in series.unique()}
    boolean_values = {"true", "false", "yes", "no", "y", "n", "0", "1"}
    return len(values) <= 2 and values.issubset(boolean_values)


_TEMPORAL_NAME_TOKENS = frozenset({"date", "time", "datetime", "timestamp", "ts"})
# Plausible date-like numerics: 4-digit years, yyyymmdd, and epoch seconds/ms.
_TEMPORAL_NUMERIC_RANGES = ((1900, 2200), (19000101, 22001231), (10**8, 10**11), (10**11, 10**14))


def _has_temporal_name_token(normalized_name: str) -> bool:
    """Whole-token match, so "Times"/"timeline"/"datetime_id" don't all collide."""
    tokens = re.split(r"[^a-z0-9]+|(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])", normalized_name)
    return any(token in _TEMPORAL_NAME_TOKENS for token in tokens if token)


def _numeric_values_look_temporal(non_null: pd.Series) -> bool:
    """Whether numeric values fall in a range a real date encoding would occupy."""
    if non_null.empty:
        return False
    numeric = pd.Series(pd.to_numeric(non_null, errors="coerce")).dropna()
    if numeric.empty:
        return False
    low = float(cast(Any, numeric.min()))
    high = float(cast(Any, numeric.max()))
    return any(start <= low and high <= end for start, end in _TEMPORAL_NUMERIC_RANGES)


def _datetime_parse_success_percent(series: pd.Series) -> float | None:
    non_null = series.dropna()
    if non_null.empty:
        return None
    parsed = _parse_datetime_values(non_null)
    success_percent = round(float(parsed.notna().mean()) * 100, 2)
    return success_percent


def _parse_datetime_values(series: pd.Series) -> pd.Series:
    if is_numeric_dtype(series):
        numeric = cast(pd.Series, pd.to_numeric(series, errors="coerce"))
        valid = numeric.dropna()
        if valid.empty:
            return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        low = float(cast(Any, valid.min()))
        high = float(cast(Any, valid.max()))
        if 1900 <= low and high <= 2200:
            return pd.to_datetime(
                numeric.astype("Int64").astype("string"),
                errors="coerce",
                format="%Y",
            )
        if 19000101 <= low and high <= 22001231:
            return pd.to_datetime(
                numeric.astype("Int64").astype("string"),
                errors="coerce",
                format="%Y%m%d",
            )
        if 10**8 <= low and high <= 10**11:
            return pd.to_datetime(numeric, errors="coerce", unit="s", utc=True)
        if 10**11 <= low and high <= 10**14:
            return pd.to_datetime(numeric, errors="coerce", unit="ms", utc=True)
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    return pd.Series(pd.to_datetime(series, errors="coerce", format="mixed"), index=series.index)


def _looks_mixed_type(series: pd.Series) -> bool:
    values = series.astype(str).str.strip()
    numeric_mask = pd.Series(pd.to_numeric(values, errors="coerce")).notna()
    return bool(numeric_mask.any() and (~numeric_mask).any())


def _sample_values(series: pd.Series, limit: int = 5) -> list[str]:
    values = series.dropna().head(limit).tolist()
    return [str(value) for value in values]
