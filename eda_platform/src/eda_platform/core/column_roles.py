"""Column semantic role layer."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, cast

import pandas as pd
from pydantic import BaseModel, Field

from eda_platform.core.ids import make_artifact_id
from eda_platform.core.semantic import SemanticSeeds
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile


class ColumnRoleName(StrEnum):
    """Mutually exclusive column roles, with LLM-facing definitions below."""

    IDENTIFIER = "identifier"
    SEQUENCE = "sequence"
    MEASURE = "measure"
    DIMENSION = "dimension"
    TIMESTAMP = "timestamp"
    GEO = "geo"
    TEXT = "text"
    CODE = "code"


# One-sentence definitions written for an LLM audience: they are injected into
# the bootstrap prompt so hypotheses target exactly this vocabulary.
ROLE_DEFINITIONS: dict[ColumnRoleName, str] = {
    ColumnRoleName.IDENTIFIER: (
        "A key naming one entity or row (primary or foreign key); aggregating, "
        "averaging or correlating its values is meaningless."
    ),
    ColumnRoleName.SEQUENCE: (
        "A per-group ordinal counter (1, 2, 3 ... within a parent entity, e.g. a "
        "line-item number); it is a position, not a quantity — never average it."
    ),
    ColumnRoleName.MEASURE: (
        "A quantitative value where sums, means and distributions are meaningful "
        "(price, weight, a numeric rating scale)."
    ),
    ColumnRoleName.DIMENSION: (
        "A categorical attribute used to group, filter or segment rows (status, type, category)."
    ),
    ColumnRoleName.TIMESTAMP: (
        "A point in time (date or datetime) used for ordering and time-window analysis."
    ),
    ColumnRoleName.GEO: (
        "A geographic attribute: coordinates (latitude/longitude) or an "
        "administrative area name (city, state, country)."
    ),
    ColumnRoleName.TEXT: (
        "Free-form natural language (comments, descriptions); analyzed with text "
        "methods, never numeric aggregation."
    ),
    ColumnRoleName.CODE: (
        "A number-shaped nominal code (zip/postal prefix, phone number, SKU); its "
        "digits carry no magnitude, so it is never a measure."
    ),
}

RoleProvenance = Literal["inferred", "seeded", "unverified"]

# Roles whose deterministic verification is structural (keyed on hard data
# facts). When a data-verified structural role conflicts with a differently
# verified LLM hypothesis, the structural role wins (priors accelerate
# hypothesis generation; the data ruling is final).
STRUCTURAL_ROLES: frozenset[ColumnRoleName] = frozenset(
    {
        ColumnRoleName.IDENTIFIER,
        ColumnRoleName.SEQUENCE,
        ColumnRoleName.TIMESTAMP,
        ColumnRoleName.GEO,
        ColumnRoleName.CODE,
    }
)

# Confidence assigned when the named deterministic check verified the role.
VERIFIED_CONFIDENCE: dict[ColumnRoleName, float] = {
    ColumnRoleName.IDENTIFIER: 0.95,
    ColumnRoleName.SEQUENCE: 0.95,
    ColumnRoleName.TIMESTAMP: 0.9,
    ColumnRoleName.GEO: 0.9,
    ColumnRoleName.CODE: 0.85,
    ColumnRoleName.TEXT: 0.8,
    ColumnRoleName.MEASURE: 0.85,
    ColumnRoleName.DIMENSION: 0.8,
}

UNVERIFIED_CONFIDENCE = 0.4
SEEDED_CONFIDENCE = 1.0


class ColumnRole(BaseModel):
    """One column's role with its provenance and verification trail."""

    column: str
    role: ColumnRoleName
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: RoleProvenance
    verified_by: list[str] = Field(default_factory=list)
    rationale: str = ""
    # Tracks user adoption so an inferred role can later be promoted to seeded.
    adoption_count: int = 0


class ColumnRoleSet(BaseModel):
    """Store the inferred semantic roles for one dataset."""

    dataset: str
    entity: str = ""
    roles: list[ColumnRole] = Field(default_factory=list)
    model_version: str | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def role_of(self, column: str) -> ColumnRole | None:
        for role in self.roles:
            if role.column == column:
                return role
        return None

    def excluded_from_stats(self) -> set[str]:
        """Columns to drop from correlation/ANOVA/trend candidate pools."""
        return {
            role.column
            for role in self.roles
            if role.role in (ColumnRoleName.IDENTIFIER, ColumnRoleName.SEQUENCE)
            and role.provenance in ("inferred", "seeded")
        }

    def impact_weight(self, column: str) -> float:
        """Business-impact weight for insight scoring (QuickInsights-style)."""
        if column in self.excluded_from_stats():
            return 0.0
        return 1.0


class ColumnFacts(BaseModel):
    """Deterministic, LLM-free statistics for one column (validator input)."""

    name: str
    dtype: str = ""
    semantic_type: str = "unknown"
    row_count: int = 0
    missing_count: int = 0
    missing_percent: float = 0.0
    unique_count: int = 0
    unique_percent: float = 0.0
    sample_values: list[str] = Field(default_factory=list)

    @property
    def normalized_name(self) -> str:
        return self.name.strip().lower().replace(" ", "_")

    @property
    def name_tokens(self) -> list[str]:
        return [token for token in re.split(r"[^0-9a-z]+", self.normalized_name) if token]

    @property
    def avg_sample_length(self) -> float:
        if not self.sample_values:
            return 0.0
        return sum(len(value) for value in self.sample_values) / len(self.sample_values)


@dataclass
class TableFacts:
    """One table's columns plus (optionally) its frame for group-level checks."""

    dataset: str
    row_count: int
    columns: list[ColumnFacts]
    frame: pd.DataFrame | None = None

    def facts_of(self, column: str) -> ColumnFacts | None:
        for facts in self.columns:
            if facts.name == column:
                return facts
        return None


def column_facts_from_profile(profile: DatasetProfile) -> list[ColumnFacts]:
    """Project a deterministic profile artifact payload into validator inputs."""
    facts: list[ColumnFacts] = []
    for detail in profile.columns_detail:
        facts.append(
            ColumnFacts(
                name=detail.name,
                dtype=detail.dtype,
                semantic_type=detail.semantic_type,
                row_count=profile.rows,
                missing_count=detail.missing_count,
                missing_percent=detail.missing_percent,
                unique_count=detail.unique_count,
                unique_percent=detail.unique_percent,
                sample_values=list(detail.sample_values),
            )
        )
    return facts


def table_facts_from_profile(
    profile: DatasetProfile, *, frame: pd.DataFrame | None = None
) -> TableFacts:
    return TableFacts(
        dataset=profile.name,
        row_count=profile.rows,
        columns=column_facts_from_profile(profile),
        frame=frame,
    )


# Validators return passed check names or ``None`` and never depend on an LLM.

_ID_NAME_TOKENS = frozenset({"id", "uuid", "guid", "key"})
_CODE_NAME_TOKENS = frozenset(
    {
        "zip",
        "zipcode",
        "postal",
        "postcode",
        "cep",
        "phone",
        "tel",
        "telephone",
        "code",
        "prefix",
        "barcode",
        "sku",
        "ean",
        "isbn",
    }
)
_GEO_COORD_TOKENS = frozenset({"lat", "latitude", "lng", "lon", "longitude"})
_GEO_ADMIN_TOKENS = frozenset(
    {"city", "state", "country", "region", "province", "county", "municipality"}
)
_ORDINAL_SCALE_TOKENS = frozenset({"score", "rating", "stars", "grade", "level"})
_YEAR_TOKENS = frozenset({"year", "yr"})
# Names that suggest a per-group counter. Only the strict frame check can
# *verify* sequence; these tokens merely keep counter-named columns out of the
# measure/dimension pools when no frame is available (profile-only degrade
# path), so a line number is never endorsed as a quantity.
_SEQUENCE_NAME_TOKENS = frozenset({"sequential", "sequence", "seq", "ordinal"})

_DIGITS_RE = re.compile(r"\d+")

# Thresholds (identifier criteria follow the industry standard the plan cites:
# uniqueness ~= 1 plus a naming pattern; HoPF for the per-group sequence rule).
_ID_NAME_UNIQUE_PERCENT = 95.0
_ID_GENERIC_UNIQUE_PERCENT = 99.0
# Sequence verification tolerates a small violation ratio for incomplete groups.
_SEQUENCE_CONFORMITY_MIN = 0.995
_ID_GENERIC_MIN_ROWS = 20
_ID_GENERIC_MAX_AVG_LENGTH = 40.0
_DIMENSION_MAX_UNIQUE_COUNT = 50
_DIMENSION_MAX_UNIQUE_PERCENT = 5.0
_TEXT_MIN_AVG_LENGTH = 24.0
_MEASURE_MIN_DISPERSION_UNIQUE = 10


def _has_id_name(facts: ColumnFacts) -> bool:
    tokens = facts.name_tokens
    return bool(tokens) and tokens[-1] in _ID_NAME_TOKENS


def _is_integer_dtype(dtype: str) -> bool:
    lowered = dtype.strip().lower()
    return lowered.startswith("int") or lowered.startswith("uint")


def _is_float_dtype(dtype: str) -> bool:
    return dtype.strip().lower().startswith("float")


def _is_numeric(facts: ColumnFacts) -> bool:
    return (
        facts.semantic_type == "numeric"
        or _is_integer_dtype(facts.dtype)
        or _is_float_dtype(facts.dtype)
    )


def _is_bool(facts: ColumnFacts) -> bool:
    return facts.semantic_type == "boolean" or facts.dtype.strip().lower() == "bool"


def _samples_all_digits(facts: ColumnFacts) -> bool:
    values = [value.strip() for value in facts.sample_values]
    return bool(values) and all(_DIGITS_RE.fullmatch(value) is not None for value in values)


def _samples_as_floats(facts: ColumnFacts) -> list[float] | None:
    parsed: list[float] = []
    for value in facts.sample_values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            return None
    return parsed or None


def _samples_are_integer_valued_numbers(facts: ColumnFacts) -> bool:
    """Return whether every sample is finite and has no fractional component."""
    values = _samples_as_floats(facts)
    return bool(values) and all(math.isfinite(value) and value.is_integer() for value in values)


def check_identifier(facts: ColumnFacts, table: TableFacts) -> list[str] | None:
    """identifier = uniqueness ~= 1 + naming pattern (or near-unique short values)."""
    if facts.missing_count > 0:
        return None
    if _has_id_name(facts) and facts.unique_percent >= _ID_NAME_UNIQUE_PERCENT:
        return ["id_name_pattern", "id_unique_ratio"]
    if (
        facts.unique_percent >= _ID_GENERIC_UNIQUE_PERCENT
        and facts.row_count >= _ID_GENERIC_MIN_ROWS
        and facts.avg_sample_length <= _ID_GENERIC_MAX_AVG_LENGTH
        and not _is_numeric(facts)
        # A near-unique datetime is a timeline, not a key (Olist purchase
        # timestamps are ~99% unique); keep the generic key path off it.
        and facts.semantic_type != "datetime"
        and "datetime" not in facts.dtype.strip().lower()
    ):
        return ["id_unique_ratio", "id_short_values"]
    return None


def check_sequence(facts: ColumnFacts, table: TableFacts) -> list[str] | None:
    """Identify columns that form a 1-to-n sequence within each group."""
    if table.frame is None:
        return None
    if not _is_integer_dtype(facts.dtype):
        return None
    if facts.unique_count < 2:  # a constant 1-column is degenerate, not a sequence
        return None
    frame = table.frame
    if facts.name not in frame.columns:
        return None
    group_candidates = [
        column.name
        for column in table.columns
        if column.name != facts.name and _has_id_name(column) and column.name in frame.columns
    ]
    for group in group_candidates:
        sub = cast(pd.DataFrame, frame[[group, facts.name]].dropna())
        if sub.empty:
            continue
        ordered = cast(pd.DataFrame, sub.sort_values(by=[group, facts.name]))
        expected = ordered.groupby(group, sort=False).cumcount() + 1
        matches = ordered[facts.name].to_numpy() == expected.to_numpy()
        if matches.mean() >= _SEQUENCE_CONFORMITY_MIN:
            return [f"sequence_strict_1n_within_group:{group}"]
    return None


# Bare integer time fields are timestamps only when values fall in this epoch range.
_EPOCH_SECONDS_MIN = 946_684_800  # 2000-01-01T00:00:00Z
_EPOCH_SECONDS_MAX = 2_051_222_400  # 2035-01-01T00:00:00Z
_BARE_TIME_TOKENS = frozenset({"time", "timestamp"})


def _samples_in_epoch_seconds_range(facts: ColumnFacts) -> bool:
    values = _samples_as_floats(facts)
    if values is None:
        return False
    return all(_EPOCH_SECONDS_MIN <= value <= _EPOCH_SECONDS_MAX for value in values)


def timestamp_numeric_offset_rationale(facts: ColumnFacts) -> str | None:
    """Why the DI10-W5 veto refuses the timestamp role, or ``None`` if it doesn't."""
    if not (_is_integer_dtype(facts.dtype) or _is_float_dtype(facts.dtype)):
        return None
    if not set(facts.name_tokens) <= _BARE_TIME_TOKENS or not facts.name_tokens:
        return None
    if not _samples_are_integer_valued_numbers(facts):
        return None
    if _samples_in_epoch_seconds_range(facts):
        return None
    return (
        f"Column {facts.name!r} is numeric with integer-valued samples outside "
        "the epoch-seconds range; treated as a numeric offset, not a timestamp."
    )


def check_timestamp(facts: ColumnFacts, table: TableFacts) -> list[str] | None:
    """Identify datetime columns and columns with a high datetime parse rate."""
    if "datetime" in facts.dtype.strip().lower():
        return ["timestamp_dtype"]
    if facts.semantic_type == "datetime":
        if timestamp_numeric_offset_rationale(facts) is not None:
            return None
        if (
            _is_integer_dtype(facts.dtype)
            and _samples_all_digits(facts)
            and _samples_in_epoch_seconds_range(facts)
        ):
            return ["timestamp_profiled_datetime", "timestamp_epoch_seconds_range"]
        return ["timestamp_profiled_datetime"]
    return None


def check_geo(facts: ColumnFacts, table: TableFacts) -> list[str] | None:
    """geo = coordinate name + in-bounds values, or admin-area name + low cardinality."""
    tokens = set(facts.name_tokens)
    coord_tokens = tokens & _GEO_COORD_TOKENS
    if coord_tokens and _is_numeric(facts):
        values = _samples_as_floats(facts)
        if values is not None:
            bound = 90.0 if coord_tokens & {"lat", "latitude"} else 180.0
            if all(abs(value) <= bound for value in values):
                return ["geo_coordinate_bounds"]
    if tokens & _GEO_ADMIN_TOKENS and (
        facts.semantic_type == "categorical" or facts.unique_percent <= 20.0
    ):
        return ["geo_admin_name_low_cardinality"]
    return None


def check_code(facts: ColumnFacts, table: TableFacts) -> list[str] | None:
    """Identify number-shaped nominal values that should not be treated as measures."""
    tokens = set(facts.name_tokens)
    digit_like = _samples_all_digits(facts)
    if tokens & _CODE_NAME_TOKENS and digit_like:
        return ["code_name_pattern", "code_digit_values"]
    if digit_like:
        stripped = [value.strip() for value in facts.sample_values]
        if any(len(value) > 1 and value.startswith("0") for value in stripped):
            return ["code_leading_zero_digits"]
    return None


def check_text(facts: ColumnFacts, table: TableFacts) -> list[str] | None:
    """text = profiled free-form text, or long average values (non-numeric)."""
    if _has_id_name(facts):
        return None
    if facts.semantic_type == "text":
        return ["text_profiled_text"]
    if facts.avg_sample_length >= _TEXT_MIN_AVG_LENGTH and not _is_numeric(facts):
        return ["text_long_values"]
    return None


def check_measure(facts: ColumnFacts, table: TableFacts) -> list[str] | None:
    """Identify numeric columns with quantity-like dispersion or ordinal scales."""
    if not _is_numeric(facts) or _is_bool(facts):
        return None
    tokens = set(facts.name_tokens)
    if _has_id_name(facts) or tokens & _CODE_NAME_TOKENS or tokens & _YEAR_TOKENS:
        return None
    if tokens & _SEQUENCE_NAME_TOKENS:
        return None
    if check_code(facts, table) is not None:
        return None
    if tokens & _ORDINAL_SCALE_TOKENS and facts.unique_count <= 12:
        return ["measure_numeric_dtype", "measure_ordinal_scale"]
    if _is_float_dtype(facts.dtype):
        return ["measure_numeric_dtype", "measure_fractional_values"]
    if facts.unique_count > _MEASURE_MIN_DISPERSION_UNIQUE:
        return ["measure_numeric_dtype", "measure_value_dispersion"]
    return None


def check_dimension(facts: ColumnFacts, table: TableFacts) -> list[str] | None:
    """dimension = low-cardinality grouping attribute (or boolean / year label)."""
    if _has_id_name(facts) or set(facts.name_tokens) & _SEQUENCE_NAME_TOKENS:
        return None
    if _is_bool(facts):
        return ["dimension_boolean"]
    if set(facts.name_tokens) & _YEAR_TOKENS:
        values = _samples_as_floats(facts)
        if values is not None and all(1800 <= value <= 2200 for value in values):
            return ["dimension_year_values"]
    if facts.unique_count >= 1 and (
        facts.unique_count <= _DIMENSION_MAX_UNIQUE_COUNT
        or facts.unique_percent <= _DIMENSION_MAX_UNIQUE_PERCENT
    ):
        return ["dimension_low_cardinality"]
    return None


_VALIDATORS: dict[ColumnRoleName, Callable[[ColumnFacts, TableFacts], list[str] | None]] = {
    ColumnRoleName.IDENTIFIER: check_identifier,
    ColumnRoleName.SEQUENCE: check_sequence,
    ColumnRoleName.TIMESTAMP: check_timestamp,
    ColumnRoleName.GEO: check_geo,
    ColumnRoleName.CODE: check_code,
    ColumnRoleName.TEXT: check_text,
    ColumnRoleName.MEASURE: check_measure,
    ColumnRoleName.DIMENSION: check_dimension,
}

# Structural roles take precedence; measures precede dimensions when checks overlap.
_DETERMINISTIC_PRIORITY: tuple[ColumnRoleName, ...] = (
    ColumnRoleName.IDENTIFIER,
    ColumnRoleName.SEQUENCE,
    ColumnRoleName.TIMESTAMP,
    ColumnRoleName.GEO,
    ColumnRoleName.CODE,
    ColumnRoleName.TEXT,
    ColumnRoleName.MEASURE,
    ColumnRoleName.DIMENSION,
)


def verify_role(role: ColumnRoleName, facts: ColumnFacts, table: TableFacts) -> list[str] | None:
    """Run the deterministic validator for ``role``; the only verification path."""
    return _VALIDATORS[role](facts, table)


def infer_column_roles(
    profile: DatasetProfile,
    *,
    frame: pd.DataFrame | None = None,
    seeds: SemanticSeeds | None = None,
) -> ColumnRoleSet:
    """Infer column roles deterministically without an LLM."""
    table = table_facts_from_profile(profile, frame=frame)
    roles: list[ColumnRole] = []
    for facts in table.columns:
        for role_name in _DETERMINISTIC_PRIORITY:
            checks = verify_role(role_name, facts, table)
            if checks:
                roles.append(
                    ColumnRole(
                        column=facts.name,
                        role=role_name,
                        confidence=VERIFIED_CONFIDENCE[role_name],
                        provenance="inferred",
                        verified_by=checks,
                        rationale=f"Deterministic checks passed: {', '.join(checks)}.",
                    )
                )
                break
    role_set = ColumnRoleSet(dataset=profile.name, roles=roles, model_version="deterministic")
    if seeds is not None:
        apply_role_seeds(role_set, seeds)
    return role_set


def apply_role_seeds(role_set: ColumnRoleSet, seeds: SemanticSeeds) -> ColumnRoleSet:
    """Overlay human-pinned roles on inferred roles."""
    for seed in seeds.column_role_seeds:
        if seed.dataset != role_set.dataset:
            continue
        try:
            role_name = ColumnRoleName(seed.role.strip().lower())
        except ValueError:
            continue
        seeded = ColumnRole(
            column=seed.column,
            role=role_name,
            confidence=SEEDED_CONFIDENCE,
            provenance="seeded",
            verified_by=["human_seed"],
            rationale=seed.note or "Pinned by a human seed.",
        )
        existing = role_set.role_of(seed.column)
        if existing is None:
            role_set.roles.append(seeded)
        else:
            role_set.roles[role_set.roles.index(existing)] = seeded
    return role_set


def column_role_set_artifact(
    role_set: ColumnRoleSet,
    *,
    project_id: str,
    session_id: str,
    parents: list[str] | None = None,
) -> Artifact:
    """Convert a role set into a structured artifact for persistence."""
    payload = role_set.model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("roles", payload),
        type=ArtifactType.COLUMN_ROLE_SET,
        project_id=project_id,
        session_id=session_id,
        parents=list(parents) if parents else [],
        payload=payload,
    )
