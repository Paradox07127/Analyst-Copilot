from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)

from eda_platform.core.column_roles import ColumnRoleName, ColumnRoleSet
from eda_platform.core.query import DuckDBQueryEngine
from eda_platform.core.semantic import JoinWhitelistEntry
from eda_platform.schemas.relations import (
    Cardinality,
    Confidence,
    RelationshipCandidate,
    RelationshipCandidateSet,
    RelationshipColumnPair,
    RelationshipSignals,
    RelationshipValidation,
    RelationshipValidationSet,
)
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.sql_names import quote_identifier as _quote_identifier

DEFAULT_WEIGHTS: dict[str, float] = {
    "name_similarity": 0.20,
    "type_compatible": 0.15,
    "overlap_left_in_right": 0.30,
    "overlap_right_in_left": 0.05,
    "right_unique_rate": 0.20,
    "null_quality": 0.05,
    "format_fingerprint_match": 0.05,
}

_SAMPLE_HASH_MODULUS = 1_000_000
_KEY_TOKENS = {"id", "key", "code", "uuid", "name"}

logger = logging.getLogger(__name__)

# strict id-naming tokens used by the join-proposal quality gate and the auto-confirmation
# criterion.
_ID_NAME_TOKENS = frozenset({"id", "uuid", "guid", "key"})

# Role provenances that count as *verified* for gating (unverified hypotheses
# must never gate anything — DI8-B red line).
_VERIFIED_PROVENANCES = ("inferred", "seeded")


@dataclass(frozen=True)
class _ColumnStats:
    dataset_id: str
    dataset_name: str
    relation_name: str
    column: str
    kind: str
    format_fingerprint: str
    row_count: int
    non_null_count: int
    distinct_count: int
    unique_rate: float
    null_rate: float
    key_like: bool
    semantic_key_like: bool


def discover_relationship_candidates(
    datasets: Sequence[LoadedDataset],
    engine: DuckDBQueryEngine | None = None,
    *,
    weights: Mapping[str, float] | None = None,
    high_overlap_threshold: float = 0.95,
    high_unique_threshold: float = 0.99,
    medium_overlap_threshold: float = 0.60,
    medium_unique_threshold: float = 0.90,
    max_null_rate: float = 0.50,
    max_candidates_per_dataset_pair: int = 10,
    max_overlap_checks_per_dataset_pair: int = 4,
    sample_threshold_rows: int = 50_000,
    sample_size: int = 20_000,
) -> RelationshipCandidateSet:
    query_engine = engine or DuckDBQueryEngine()
    for dataset in datasets:
        query_engine.register_frame(_relation_name(dataset.record.dataset_id), dataset.frame)

    stats_by_dataset = {
        dataset.record.dataset_id: _eligible_columns(
            dataset,
            max_null_rate=max_null_rate,
        )
        for dataset in datasets
    }
    configured_weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    candidates: list[RelationshipCandidate] = []
    truncated_pairs = 0
    overlap_pairs_evaluated = 0
    overlap_pairs_prefiltered = 0

    for left_index, left_dataset in enumerate(datasets):
        for right_dataset in datasets[left_index + 1 :]:
            pair_candidates, evaluated, prefiltered = _candidate_pairs(
                stats_by_dataset[left_dataset.record.dataset_id],
                stats_by_dataset[right_dataset.record.dataset_id],
                query_engine,
                weights=configured_weights,
                high_overlap_threshold=high_overlap_threshold,
                high_unique_threshold=high_unique_threshold,
                medium_overlap_threshold=medium_overlap_threshold,
                medium_unique_threshold=medium_unique_threshold,
                sample_threshold_rows=sample_threshold_rows,
                sample_size=sample_size,
                max_overlap_checks=max_overlap_checks_per_dataset_pair,
            )
            overlap_pairs_evaluated += evaluated
            overlap_pairs_prefiltered += prefiltered
            pair_candidates.sort(key=_candidate_sort_key)
            if len(pair_candidates) > max_candidates_per_dataset_pair:
                truncated_pairs += len(pair_candidates) - max_candidates_per_dataset_pair
                pair_candidates = pair_candidates[:max_candidates_per_dataset_pair]
            candidates.extend(pair_candidates)

    candidates = _demote_competing_left_columns(candidates)
    candidates.sort(key=_candidate_sort_key)
    return RelationshipCandidateSet(
        dataset_ids=[dataset.record.dataset_id for dataset in datasets],
        candidates=candidates,
        thresholds={
            "high_overlap_threshold": high_overlap_threshold,
            "high_unique_threshold": high_unique_threshold,
            "medium_overlap_threshold": medium_overlap_threshold,
            "medium_unique_threshold": medium_unique_threshold,
            "max_null_rate": max_null_rate,
            "max_candidates_per_dataset_pair": float(max_candidates_per_dataset_pair),
            "max_overlap_checks_per_dataset_pair": float(
                max_overlap_checks_per_dataset_pair
            ),
            "sample_threshold_rows": float(sample_threshold_rows),
            "sample_size": float(sample_size),
        },
        truncated_pairs=truncated_pairs,
        overlap_pairs_evaluated=overlap_pairs_evaluated,
        overlap_pairs_prefiltered=overlap_pairs_prefiltered,
        coverage_status="limited" if overlap_pairs_prefiltered else "complete",
        coverage_reason=(
            "Structural candidates exceeded the bounded overlap-query budget."
            if overlap_pairs_prefiltered
            else "Every structurally eligible column pair was evaluated."
        ),
    )


def validate_relationships(
    candidates: RelationshipCandidateSet | Sequence[RelationshipCandidate],
    engine: DuckDBQueryEngine,
    *,
    join_multiplier_warning_threshold: float = 1.05,
) -> RelationshipValidationSet:
    candidate_list = (
        candidates.candidates
        if isinstance(candidates, RelationshipCandidateSet)
        else list(candidates)
    )
    validations: list[RelationshipValidation] = []
    for candidate in sorted(candidate_list, key=lambda item: item.pair.label()):
        if candidate.confidence not in {"medium", "high"}:
            continue
        sql = _validation_sql(candidate.pair)
        frame = engine.execute_select(sql)
        row = frame.to_dict("records")[0]
        left_max = int(row["left_max_group_size"] or 0)
        right_max = int(row["right_max_group_size"] or 0)
        cardinality = _cardinality(left_max, right_max)
        join_multiplier = _float(row["join_row_multiplier"])
        warnings = _validation_warnings(
            cardinality,
            join_multiplier,
            threshold=join_multiplier_warning_threshold,
        )
        validations.append(
            RelationshipValidation(
                pair=candidate.pair,
                join_row_multiplier=join_multiplier,
                orphan_rate_left=_float(row["orphan_rate_left"]),
                orphan_rate_right=_float(row["orphan_rate_right"]),
                cardinality=cardinality,
                verified=True,
                verification_sql=sql,
                sampled=candidate.signals.sampled,
                warnings=warnings,
            )
        )
    return RelationshipValidationSet(validations=validations)


def eager_validation_candidates(
    candidates: RelationshipCandidateSet,
) -> list[RelationshipCandidate]:
    """Return only edges that can plausibly unlock safe same-run auto-confirmation."""
    return [
        candidate
        for candidate in candidates.candidates
        if candidate.confidence == "high"
        and _columns_id_named(candidate.pair.left_columns)
        and _columns_id_named(candidate.pair.right_columns)
    ]


def propose_join_candidates(
    candidates: RelationshipCandidateSet,
    validations: RelationshipValidationSet | None = None,
    *,
    role_sets: Mapping[str, ColumnRoleSet] | None = None,
) -> list[JoinWhitelistEntry]:
    """DI8-C ①: turn M2 relationship output into join whitelist *proposals*."""
    validation_map = {
        validation.pair.label(): validation
        for validation in (validations.validations if validations is not None else [])
    }
    proposals: list[JoinWhitelistEntry] = []
    rejected_by_role = 0
    for candidate in sorted(candidates.candidates, key=lambda item: item.pair.label()):
        if candidate.confidence not in {"high", "medium"}:
            continue
        pair = candidate.pair
        veto_reason = _role_veto_reason(pair, role_sets)
        if veto_reason is not None:
            rejected_by_role += 1
            logger.info(
                "join proposal rejected by role gate: %s (%s)",
                pair.label(),
                veto_reason,
            )
            continue
        validation = validation_map.get(pair.label())
        cardinality = (
            validation.cardinality
            if validation is not None
            else _inferred_cardinality(candidate)
        )
        both_sides_id_named = _columns_id_named(pair.left_columns) and _columns_id_named(
            pair.right_columns
        )
        quality: Literal["normal", "low"] = "normal"
        if cardinality == "many_to_many" and not (
            _columns_id_named(pair.left_columns) or _columns_id_named(pair.right_columns)
        ):
            quality = "low"
        # `sampled` means the overlap statistics behind `confidence` came from a
        # bounded sample, so "high" is an estimate. Such an edge may be proposed
        # but never machine-confirmed; a human confirms it or nobody does.
        auto_confirm = (
            validation is not None
            and validation.verified
            and not validation.sampled
            and not candidate.signals.sampled
            and candidate.confidence == "high"
            and both_sides_id_named
            and cardinality != "many_to_many"
            and quality == "normal"
        )
        proposals.append(
            JoinWhitelistEntry(
                left_dataset=pair.left_dataset_name,
                left_dataset_id=pair.left_dataset_id,
                left_columns=list(pair.left_columns),
                right_dataset=pair.right_dataset_name,
                right_dataset_id=pair.right_dataset_id,
                right_columns=list(pair.right_columns),
                cardinality=cardinality,
                join_row_multiplier=(
                    validation.join_row_multiplier if validation is not None else None
                ),
                validation_verified=bool(validation is not None and validation.verified),
                confidence_source=(
                    f"relationship_discovery: confidence={candidate.confidence}, "
                    f"ensemble={candidate.ensemble_score:.3f}, "
                    f"overlap={candidate.signals.overlap_left_in_right:.3f}, "
                    f"name_similarity={candidate.signals.name_similarity:.3f}"
                ),
                status="auto_confirmed" if auto_confirm else "proposed",
                quality=quality,
                confirmed_at=datetime.now(UTC) if auto_confirm else None,
                confirmed_by="auto" if auto_confirm else "",
            )
        )
    if rejected_by_role:
        logger.info(
            "join proposal role gate: %d candidate(s) rejected "
            "(timestamp/measure join keys)",
            rejected_by_role,
        )
    # Low-quality proposals sink to the end (stable within each group, so the
    # label ordering above is preserved for equal quality).
    proposals.sort(key=lambda entry: entry.quality == "low")
    return proposals


def _role_veto_reason(
    pair: RelationshipColumnPair,
    role_sets: Mapping[str, ColumnRoleSet] | None,
) -> str | None:
    """DI10-W5 hard filter: verified timestamp/measure columns never join."""
    if not role_sets:
        return None
    vetoed_roles = (ColumnRoleName.TIMESTAMP, ColumnRoleName.MEASURE)
    sides = (
        (pair.left_dataset_name, pair.left_columns),
        (pair.right_dataset_name, pair.right_columns),
    )
    for dataset_name, columns in sides:
        role_set = role_sets.get(dataset_name)
        if role_set is None:
            continue
        for column in columns:
            role = role_set.role_of(column)
            if (
                role is not None
                and role.role in vetoed_roles
                and role.provenance in _VERIFIED_PROVENANCES
            ):
                return (
                    f"{dataset_name}.{column} has verified role "
                    f"{role.role.value!r}; not a join key"
                )
    return None


def _columns_id_named(columns: Sequence[str]) -> bool:
    """Return whether every column has a strict identifier naming pattern."""
    if not columns:
        return False
    return all(_id_name_pattern(column) for column in columns)


def _id_name_pattern(column: str) -> bool:
    tokens = [token for token in re.split(r"[^0-9a-z]+", column.lower()) if token]
    if tokens and tokens[-1] in _ID_NAME_TOKENS:
        return True
    return _normalize_name(column).endswith(tuple(_ID_NAME_TOKENS))


def _inferred_cardinality(candidate: RelationshipCandidate) -> str:
    """Infer fallback cardinality when DuckDB validation has not run."""
    if candidate.signals.right_unique_rate >= 0.99:
        return "many_to_one"
    return "many_to_many"


def _eligible_columns(
    dataset: LoadedDataset,
    *,
    max_null_rate: float,
) -> list[_ColumnStats]:
    relation_name = _relation_name(dataset.record.dataset_id)
    stats: list[_ColumnStats] = []
    for column in dataset.frame.columns:
        series = cast(pd.Series, dataset.frame[column])
        row_count = int(len(series))
        non_null = series.dropna()
        non_null_count = int(len(non_null))
        null_rate = _safe_ratio(row_count - non_null_count, row_count)
        if null_rate > max_null_rate or non_null_count == 0:
            continue
        normalized = _normalized_values(non_null)
        distinct_count = int(normalized.nunique(dropna=True))
        if distinct_count <= 1:
            continue
        kind = _column_kind(series, normalized)
        if kind == "boolean":
            continue
        unique_rate = _safe_ratio(distinct_count, non_null_count)
        stats.append(
            _ColumnStats(
                dataset_id=dataset.record.dataset_id,
                dataset_name=dataset.record.name,
                relation_name=relation_name,
                column=str(column),
                kind=kind,
                format_fingerprint=_format_fingerprint(normalized, kind),
                row_count=row_count,
                non_null_count=non_null_count,
                distinct_count=distinct_count,
                unique_rate=unique_rate,
                null_rate=null_rate,
                key_like=_is_key_like(str(column)),
                semantic_key_like=_semantic_key_like(kind),
            )
        )
    return stats


def _candidate_pairs(
    left_columns: Sequence[_ColumnStats],
    right_columns: Sequence[_ColumnStats],
    engine: DuckDBQueryEngine,
    *,
    weights: Mapping[str, float],
    high_overlap_threshold: float,
    high_unique_threshold: float,
    medium_overlap_threshold: float,
    medium_unique_threshold: float,
    sample_threshold_rows: int,
    sample_size: int,
    max_overlap_checks: int,
) -> tuple[list[RelationshipCandidate], int, int]:
    """Retrieve cheap structural pairs, then run bounded overlap reranking."""
    structural_pairs: list[tuple[_ColumnStats, _ColumnStats]] = []
    for left in left_columns:
        for right in right_columns:
            name_similarity = _name_similarity(left.column, right.column)
            if _numeric_metric_pair(left, right):
                continue
            if not (_join_key_like(left) and _join_key_like(right)):
                continue
            if not _type_compatible(left, right, name_similarity=name_similarity):
                continue
            structural_pairs.append((left, right))

    structural_pairs.sort(key=_structural_pair_sort_key)
    selected_pairs = structural_pairs[: max(0, max_overlap_checks)]
    candidates: list[RelationshipCandidate] = []
    for left, right in selected_pairs:
        signal_values = _overlap_signals(
            left,
            right,
            engine,
            sample_threshold_rows=sample_threshold_rows,
            sample_size=sample_size,
        )
        candidates.extend(
            (
                _candidate_from_overlap(
                    left,
                    right,
                    signal_values,
                    weights=weights,
                    high_overlap_threshold=high_overlap_threshold,
                    high_unique_threshold=high_unique_threshold,
                    medium_overlap_threshold=medium_overlap_threshold,
                    medium_unique_threshold=medium_unique_threshold,
                ),
                _candidate_from_overlap(
                    right,
                    left,
                    {
                        **signal_values,
                        "overlap_left_in_right": signal_values[
                            "overlap_right_in_left"
                        ],
                        "overlap_right_in_left": signal_values[
                            "overlap_left_in_right"
                        ],
                        "right_unique_rate": signal_values["left_unique_rate"],
                        "left_unique_rate": signal_values["right_unique_rate"],
                    },
                    weights=weights,
                    high_overlap_threshold=high_overlap_threshold,
                    high_unique_threshold=high_unique_threshold,
                    medium_overlap_threshold=medium_overlap_threshold,
                    medium_unique_threshold=medium_unique_threshold,
                ),
            )
        )
    return candidates, len(selected_pairs), len(structural_pairs) - len(selected_pairs)


def _structural_pair_sort_key(
    pair: tuple[_ColumnStats, _ColumnStats],
) -> tuple[float, float, float, float, float, str]:
    left, right = pair
    name_similarity = _name_similarity(left.column, right.column)
    exact_name = float(_normalize_name(left.column) == _normalize_name(right.column))
    strict_key_names = float(left.key_like and right.key_like)
    format_match = float(_format_compatible(left, right))
    uniqueness = max(left.unique_rate, right.unique_rate)
    return (
        -exact_name,
        -strict_key_names,
        -name_similarity,
        -format_match,
        -uniqueness,
        f"{left.column}\0{right.column}",
    )


def _candidate_from_overlap(
    left: _ColumnStats,
    right: _ColumnStats,
    signal_values: Mapping[str, float],
    *,
    weights: Mapping[str, float],
    high_overlap_threshold: float,
    high_unique_threshold: float,
    medium_overlap_threshold: float,
    medium_unique_threshold: float,
) -> RelationshipCandidate:
    signals = RelationshipSignals(
        name_similarity=_name_similarity(left.column, right.column),
        type_compatible=True,
        overlap_left_in_right=signal_values["overlap_left_in_right"],
        overlap_right_in_left=signal_values["overlap_right_in_left"],
        right_unique_rate=signal_values["right_unique_rate"],
        left_null_rate=left.null_rate,
        right_null_rate=right.null_rate,
        format_fingerprint_match=_format_compatible(left, right),
        sampled=signal_values["sampled"] > 0,
    )
    return RelationshipCandidate(
        pair=RelationshipColumnPair(
            left_dataset_id=left.dataset_id,
            left_dataset_name=left.dataset_name,
            left_columns=[left.column],
            right_dataset_id=right.dataset_id,
            right_dataset_name=right.dataset_name,
            right_columns=[right.column],
        ),
        signals=signals,
        ensemble_score=_ensemble_score(signals, weights),
        confidence=_confidence(
            signals,
            high_overlap_threshold=high_overlap_threshold,
            high_unique_threshold=high_unique_threshold,
            medium_overlap_threshold=medium_overlap_threshold,
            medium_unique_threshold=medium_unique_threshold,
        ),
        # Discovery is reference-only. A join belongs to a selected question,
        # never to the default EDA analysis path.
        auto_adopted=False,
    )


def _demote_competing_left_columns(
    candidates: Sequence[RelationshipCandidate],
) -> list[RelationshipCandidate]:
    best_by_left: dict[tuple[str, tuple[str, ...]], str] = {}
    for candidate in sorted(candidates, key=_candidate_sort_key):
        key = (candidate.pair.left_dataset_id, tuple(candidate.pair.left_columns))
        best_by_left.setdefault(key, candidate.pair.label())

    pruned: list[RelationshipCandidate] = []
    for candidate in candidates:
        key = (candidate.pair.left_dataset_id, tuple(candidate.pair.left_columns))
        if candidate.pair.label() == best_by_left[key]:
            pruned.append(candidate)
        else:
            pruned.append(
                candidate.model_copy(update={"confidence": "low", "auto_adopted": False})
            )
    return pruned


def _overlap_signals(
    left: _ColumnStats,
    right: _ColumnStats,
    engine: DuckDBQueryEngine,
    *,
    sample_threshold_rows: int,
    sample_size: int,
) -> dict[str, float]:
    sample_left = _should_sample(
        left.row_count,
        right.row_count,
        sample_threshold_rows=sample_threshold_rows,
        sample_size=sample_size,
    )
    sample_right = _should_sample(
        right.row_count,
        left.row_count,
        sample_threshold_rows=sample_threshold_rows,
        sample_size=sample_size,
    )
    left_column = _quote_identifier(left.column)
    right_column = _quote_identifier(right.column)
    left_sample_predicate = _sample_predicate(left, sample_left, sample_size)
    right_sample_predicate = _sample_predicate(right, sample_right, sample_size)
    sql = f"""
with
left_base as (
    select cast({left_column} as varchar) as value
    from {_quote_identifier(left.relation_name)}
    where {left_column} is not null{left_sample_predicate}
),
right_base as (
    select cast({right_column} as varchar) as value
    from {_quote_identifier(right.relation_name)}
    where {right_column} is not null{right_sample_predicate}
),
left_values as (
    select distinct value from left_base
),
right_values as (
    select distinct value from right_base
),
intersection_values as (
    select l.value
    from left_values as l
    inner join right_values as r on l.value = r.value
)
select
    (select count(*) from left_base) as left_non_null_count,
    (select count(*) from right_base) as right_non_null_count,
    (select count(*) from left_values) as left_distinct_count,
    (select count(*) from right_values) as right_distinct_count,
    (select count(*) from intersection_values) as intersection_count
""".strip()
    row = engine.execute_select(sql).to_dict("records")[0]
    left_distinct = int(row["left_distinct_count"] or 0)
    right_distinct = int(row["right_distinct_count"] or 0)
    right_non_null = int(row["right_non_null_count"] or 0)
    intersection = int(row["intersection_count"] or 0)
    return {
        "overlap_left_in_right": _safe_ratio(intersection, left_distinct),
        "overlap_right_in_left": _safe_ratio(intersection, right_distinct),
        "right_unique_rate": _safe_ratio(right_distinct, right_non_null),
        "left_unique_rate": _safe_ratio(
            left_distinct, int(row["left_non_null_count"] or 0)
        ),
        "sampled": float(sample_left or sample_right),
    }


def _validation_sql(pair: RelationshipColumnPair) -> str:
    left_relation = _quote_identifier(_relation_name(pair.left_dataset_id))
    right_relation = _quote_identifier(_relation_name(pair.right_dataset_id))
    left_key = _key_expression("l", pair.left_columns)
    right_key = _key_expression("r", pair.right_columns)
    return f"""
with
left_source as (
    select {_key_expression(None, pair.left_columns)} as join_key
    from {left_relation}
),
right_source as (
    select {_key_expression(None, pair.right_columns)} as join_key
    from {right_relation}
),
left_non_null as (
    select join_key from left_source where join_key is not null
),
right_non_null as (
    select join_key from right_source where join_key is not null
),
left_rows as (
    select count(*) as row_count from {left_relation}
),
right_rows as (
    select count(*) as row_count from {right_relation}
),
joined_rows as (
    select count(*) as row_count
    from {left_relation} as l
    left join {right_relation} as r on {left_key} = {right_key}
),
left_orphans as (
    select count(*) as row_count
    from left_non_null as l
    where not exists (
        select 1 from right_non_null as r where r.join_key = l.join_key
    )
),
right_orphans as (
    select count(*) as row_count
    from right_non_null as r
    where not exists (
        select 1 from left_non_null as l where l.join_key = r.join_key
    )
),
left_key_counts as (
    select join_key, count(*) as rows_per_key
    from left_non_null
    group by join_key
),
right_key_counts as (
    select join_key, count(*) as rows_per_key
    from right_non_null
    group by join_key
)
select
    case
        when (select row_count from left_rows) = 0 then 0.0
        else (select row_count from joined_rows)::double / (select row_count from left_rows)
    end as join_row_multiplier,
    case
        when (select row_count from left_rows) = 0 then 0.0
        else (select row_count from left_orphans)::double / (select row_count from left_rows)
    end as orphan_rate_left,
    case
        when (select row_count from right_rows) = 0 then 0.0
        else (select row_count from right_orphans)::double / (select row_count from right_rows)
    end as orphan_rate_right,
    coalesce((select max(rows_per_key) from left_key_counts), 0) as left_max_group_size,
    coalesce((select max(rows_per_key) from right_key_counts), 0) as right_max_group_size
""".strip()


def _column_kind(series: pd.Series, normalized: pd.Series) -> str:
    if is_bool_dtype(series):
        return "boolean"
    if is_integer_dtype(series):
        return "integer"
    if is_float_dtype(series):
        return "float"
    if is_datetime64_any_dtype(series):
        return "date"
    if is_numeric_dtype(series):
        return "number"
    if _ratio_matching(normalized, r"^[+-]?\d+$") >= 0.95:
        return "integer"
    if _ratio_matching(normalized, r"^[+-]?(?:\d+\.?\d*|\.\d+)$") >= 0.95:
        return "float"
    if _looks_date_like(normalized):
        # format="mixed" silences the per-element inference UserWarning on
        # heterogeneous date strings; errors="coerce" keeps unparseable -> NaT.
        parsed_dates = pd.to_datetime(normalized, errors="coerce", format="mixed")
        if float(parsed_dates.notna().mean()) >= 0.95:
            return "date"
    return "string"


def _format_fingerprint(normalized: pd.Series, kind: str) -> str:
    if kind == "date":
        date_lengths = {len(value) for value in normalized.head(25)}
        if date_lengths == {4}:
            return "date_year"
        if date_lengths <= {7}:
            return "date_month"
        return "date_day"
    if _ratio_matching(
        normalized,
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    ) >= 0.95:
        return "uuid"
    if _ratio_matching(normalized, r"^[^@\s]+@[^@\s]+\.[^@\s]+$") >= 0.95:
        return "email"
    if kind == "integer":
        return "integer_like"
    if kind in {"float", "number"}:
        return "decimal_like"
    if _ratio_matching(normalized.str.upper(), r"^[A-Z]+[0-9]+$") >= 0.80:
        return "alnum_code"
    return "text"


def _type_compatible(
    left: _ColumnStats,
    right: _ColumnStats,
    *,
    name_similarity: float,
) -> bool:
    if left.kind == right.kind:
        compatible = True
    elif {left.kind, right.kind} <= {"integer", "float", "number"}:
        compatible = True
    else:
        compatible = False
    if not compatible:
        return False
    if left.kind in {"integer", "float", "number"} or right.kind in {
        "integer",
        "float",
        "number",
    }:
        return left.key_like or right.key_like or name_similarity >= 0.70
    if left.kind == "date" or right.kind == "date":
        return left.key_like or right.key_like or name_similarity >= 0.70
    return True


def _numeric_metric_pair(left: _ColumnStats, right: _ColumnStats) -> bool:
    return _numeric_metric_like(left) and _numeric_metric_like(right)


def _numeric_metric_like(stats: _ColumnStats) -> bool:
    if stats.kind in {"float", "number"}:
        return True
    return stats.kind == "integer" and not stats.key_like and stats.unique_rate < 0.90


def _join_key_like(stats: _ColumnStats) -> bool:
    return stats.semantic_key_like or stats.key_like or stats.unique_rate >= 0.90


def _semantic_key_like(kind: str) -> bool:
    return kind == "string"


def _confidence(
    signals: RelationshipSignals,
    *,
    high_overlap_threshold: float,
    high_unique_threshold: float,
    medium_overlap_threshold: float,
    medium_unique_threshold: float,
) -> Confidence:
    if (
        signals.overlap_left_in_right >= high_overlap_threshold
        and signals.right_unique_rate >= high_unique_threshold
    ):
        return "high"
    # Uniqueness alone is not relationship evidence.
    medium_overlap = (
        medium_overlap_threshold
        <= signals.overlap_left_in_right
        < high_overlap_threshold
    )
    medium_unique = (
        signals.overlap_left_in_right >= medium_overlap_threshold
        and medium_unique_threshold
        <= signals.right_unique_rate
        < high_unique_threshold
    )
    if medium_overlap or medium_unique:
        return "medium"
    return "low"


def _ensemble_score(signals: RelationshipSignals, weights: Mapping[str, float]) -> float:
    null_quality = 1.0 - max(signals.left_null_rate, signals.right_null_rate)
    weighted = {
        "name_similarity": signals.name_similarity,
        "type_compatible": 1.0 if signals.type_compatible else 0.0,
        "overlap_left_in_right": signals.overlap_left_in_right,
        "overlap_right_in_left": signals.overlap_right_in_left,
        "right_unique_rate": signals.right_unique_rate,
        "null_quality": null_quality,
        "format_fingerprint_match": 1.0 if signals.format_fingerprint_match else 0.0,
    }
    total_weight = sum(max(weight, 0.0) for weight in weights.values()) or 1.0
    score = sum(weighted.get(name, 0.0) * max(weight, 0.0) for name, weight in weights.items())
    return round(max(0.0, min(1.0, score / total_weight)), 6)


def _validation_warnings(
    cardinality: str,
    join_multiplier: float,
    *,
    threshold: float,
) -> list[str]:
    warnings: list[str] = []
    if cardinality == "many_to_many":
        warnings.append(
            "Relationship appears many-to-many; joining can duplicate rows on both sides."
        )
    if join_multiplier > threshold:
        warnings.append(
            f"LEFT JOIN row multiplier is {join_multiplier:.3f}; downstream aggregates may inflate."
        )
    return warnings


def _cardinality(left_max: int, right_max: int) -> Cardinality:
    if left_max <= 1 and right_max <= 1:
        return "one_to_one"
    if left_max > 1 and right_max <= 1:
        return "many_to_one"
    if left_max <= 1 and right_max > 1:
        return "one_to_many"
    return "many_to_many"


def _name_similarity(left: str, right: str) -> float:
    left_norm = _normalize_name(left)
    right_norm = _normalize_name(right)
    if not left_norm or not right_norm:
        return 0.0
    distance = _levenshtein(left_norm, right_norm)
    base = 1.0 - (distance / max(len(left_norm), len(right_norm)))
    bonus = 0.0
    if left_norm == right_norm:
        bonus += 0.15
    if _is_key_like(left) and _is_key_like(right):
        bonus += 0.10
    if _last_token(left) == _last_token(right) and _last_token(left) in _KEY_TOKENS:
        bonus += 0.10
    return round(max(0.0, min(1.0, base + bonus)), 6)


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insert = current[right_index - 1] + 1
            delete = previous[right_index] + 1
            replace = previous[right_index - 1] + int(left_char != right_char)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _format_compatible(left: _ColumnStats, right: _ColumnStats) -> bool:
    if left.format_fingerprint == right.format_fingerprint:
        return True
    return {left.format_fingerprint, right.format_fingerprint} <= {
        "integer_like",
        "decimal_like",
    }


def _should_sample(
    candidate_rows: int,
    other_rows: int,
    *,
    sample_threshold_rows: int,
    sample_size: int,
) -> bool:
    # Bound each large side independently. Comparing only the larger table left
    # equal-sized million-row pairs completely unsampled and also let the
    # slightly smaller side of two large tables build an unbounded DISTINCT set.
    _ = other_rows
    return candidate_rows > sample_threshold_rows and candidate_rows > sample_size


def _sample_predicate(stats: _ColumnStats, sampled: bool, sample_size: int) -> str:
    if not sampled:
        return ""
    cutoff = max(
        1,
        min(
            _SAMPLE_HASH_MODULUS,
            int((sample_size / max(stats.row_count, 1)) * _SAMPLE_HASH_MODULUS),
        ),
    )
    column = _quote_identifier(stats.column)
    return f" and hash(cast({column} as varchar)) % {_SAMPLE_HASH_MODULUS} < {cutoff}"


def _key_expression(alias: str | None, columns: Sequence[str]) -> str:
    quoted_columns = [
        f"{alias}.{_quote_identifier(column)}" if alias else _quote_identifier(column)
        for column in columns
    ]
    if len(quoted_columns) == 1:
        return f"cast({quoted_columns[0]} as varchar)"
    cast_columns = ", ".join(f"cast({column} as varchar)" for column in quoted_columns)
    return f"concat_ws('|', {cast_columns})"


def _normalized_values(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _ratio_matching(series: pd.Series, pattern: str) -> float:
    if series.empty:
        return 0.0
    return float(series.str.match(pattern, case=False, na=False).mean())


def _looks_date_like(series: pd.Series) -> bool:
    if series.empty:
        return False
    pattern = r"^(\d{4}[-/]\d{1,2}([-/]\d{1,2})?|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})$"
    return _ratio_matching(series, pattern) >= 0.80


def _normalize_name(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", value.lower())


def _last_token(value: str) -> str:
    tokens = [token for token in re.split(r"[^0-9a-z]+", value.lower()) if token]
    return tokens[-1] if tokens else ""


def _is_key_like(value: str) -> bool:
    tokens = {token for token in re.split(r"[^0-9a-z]+", value.lower()) if token}
    normalized = _normalize_name(value)
    return bool(tokens & _KEY_TOKENS) or normalized.endswith(
        ("id", "key", "code", "uuid", "name")
    )


def _relation_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", value.strip())
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned




def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _float(value: Any) -> float:
    return round(float(value or 0.0), 6)


def _candidate_sort_key(candidate: RelationshipCandidate) -> tuple[float, str, str, str]:
    return (
        -candidate.ensemble_score,
        candidate.pair.label(),
        candidate.pair.left_dataset_id,
        candidate.pair.right_dataset_id,
    )
