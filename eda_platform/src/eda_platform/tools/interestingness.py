"""Score finding interestingness and cluster redundant findings."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from eda_platform.schemas.questions import QuestionFinding

# Tunable constants (weighted geometric mean over three observable components).
DEVIATION_WEIGHT = 1.0
COVERAGE_WEIGHT = 1.0
NONTRIVIALITY_WEIGHT = 1.0
# Neutral fallbacks when a component cannot be derived from structured fields.
DEFAULT_DEVIATION = 0.5
DEFAULT_COVERAGE = 0.5
# Degenerate/identity patterns (single group, column compared with itself, count == row
# count) keep a small positive score: penalised, never deleted.
DEGENERATE_NONTRIVIALITY = 0.1
# Exploratory findings without statistical validation (no stat backing) carry
# the multiple-comparison false-discovery burden (~60% baseline) — penalise.
EXPLORATORY_PENALTY = 0.6
# Generic normalisation anchor for evidence-extracted effect sizes when the
# test type (and therefore its conventional cutoff) is unknown.
EFFECT_SIZE_LARGE_ANCHOR = 0.5
# Flagged-share percent at which anomaly deviation saturates to 1.0.
ANOMALY_SHARE_SATURATION_PERCENT = 10.0

# Evidence locators that mark a finding as statistically validated.
_STAT_VALIDATION_LOCATORS = frozenset({"p_value"})
# Metric locators usable as a 0..1 deviation-from-naive-baseline signal.
_MODEL_DEVIATION_LOCATORS = (
    "metrics.r2",
    "metrics.f1_weighted",
    "metrics.accuracy",
    "metrics.auc",
)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


class InterestingnessScore(BaseModel):
    """Deterministic insightfulness triple plus the combined value."""

    deviation: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    nontriviality: float = Field(ge=0.0, le=1.0)
    value: float = Field(ge=0.0, le=1.0)


def interestingness(
    *,
    deviation: float | None = None,
    rows_involved: int | None = None,
    dataset_row_count: int | None = None,
    degenerate: bool = False,
    exploratory: bool = False,
    stat_validated: bool = True,
) -> InterestingnessScore:
    """Combine structured signals into an interestingness score."""
    deviation_component = (
        _clamp01(deviation) if deviation is not None else DEFAULT_DEVIATION
    )
    if rows_involved is not None and dataset_row_count is not None and dataset_row_count > 0:
        coverage_component = _clamp01(rows_involved / dataset_row_count)
    else:
        coverage_component = DEFAULT_COVERAGE
    nontriviality_component = DEGENERATE_NONTRIVIALITY if degenerate else 1.0

    weight_sum = DEVIATION_WEIGHT + COVERAGE_WEIGHT + NONTRIVIALITY_WEIGHT
    combined = (
        deviation_component**DEVIATION_WEIGHT
        * coverage_component**COVERAGE_WEIGHT
        * nontriviality_component**NONTRIVIALITY_WEIGHT
    ) ** (1.0 / weight_sum)
    if exploratory and not stat_validated:
        combined *= EXPLORATORY_PENALTY
    return InterestingnessScore(
        deviation=round(deviation_component, 6),
        coverage=round(coverage_component, 6),
        nontriviality=round(nontriviality_component, 6),
        value=round(_clamp01(combined), 6),
    )


def finding_interestingness(
    finding: QuestionFinding,
    *,
    dataset_row_count: int | None = None,
) -> InterestingnessScore:
    """Score an already-built finding from its structured evidence refs only."""
    values = _evidence_values(finding)
    return interestingness(
        deviation=_evidence_deviation(values),
        rows_involved=_evidence_rows_involved(values),
        dataset_row_count=dataset_row_count,
        degenerate=_evidence_degenerate(values),
        exploratory=finding.exploratory,
        stat_validated=any(
            locator in values for locator in _STAT_VALIDATION_LOCATORS
        ),
    )


def _evidence_values(finding: QuestionFinding) -> dict[str, float]:
    values: dict[str, float] = {}
    for ref in finding.evidence:
        if isinstance(ref.value, bool) or not isinstance(ref.value, (int, float)):
            continue
        values.setdefault(ref.locator, float(ref.value))
    return values


def _evidence_deviation(values: Mapping[str, float]) -> float | None:
    effect_size = values.get("effect_size")
    if effect_size is not None:
        return _clamp01(abs(effect_size) / EFFECT_SIZE_LARGE_ANCHOR)
    for locator in _MODEL_DEVIATION_LOCATORS:
        metric = values.get(locator)
        if metric is not None:
            return _clamp01(metric)
    outlier_percent = values.get("outlier_percent")
    if outlier_percent is not None:
        return _clamp01(outlier_percent / ANOMALY_SHARE_SATURATION_PERCENT)
    p_value = values.get("p_value")
    if p_value is not None:
        return _clamp01(1.0 - p_value)
    return None


def _evidence_rows_involved(values: Mapping[str, float]) -> int | None:
    for locator in ("sample_size", "non_null_rows"):
        rows = values.get(locator)
        if rows is not None and rows >= 0:
            return int(rows)
    return None


def _evidence_degenerate(values: Mapping[str, float]) -> bool:
    """Identity pattern detectable from evidence alone: count == row count."""
    outlier_count = values.get("outlier_count")
    non_null_rows = values.get("non_null_rows")
    return (
        outlier_count is not None
        and non_null_rows is not None
        and non_null_rows > 0
        and outlier_count >= non_null_rows
    )


# Finding-cluster deduplication.


@dataclass
class DedupFinding:
    """One clustering participant with its structured key inputs."""

    ref: str
    finding: QuestionFinding
    columns: Sequence[str] = ()
    direction: float | None = None
    magnitude: float | None = None
    dataset_row_count: int | None = None


@dataclass
class FindingCluster:
    """One near-duplicate group: a representative plus labelled supporters."""

    cluster_key: str
    representative: DedupFinding
    supporting: list[DedupFinding] = field(default_factory=list)

    @property
    def members(self) -> list[DedupFinding]:
        return [self.representative, *self.supporting]


def deduplicate_findings(items: Sequence[DedupFinding]) -> list[FindingCluster]:
    """Cluster near-duplicate findings and label roles in place."""
    grouped: dict[str, list[DedupFinding]] = {}
    for item in items:
        grouped.setdefault(_cluster_key(item), []).append(item)

    clusters: list[FindingCluster] = []
    for cluster_key, members in grouped.items():
        ranked = sorted(
            members,
            key=lambda member: (-_selection_score(member), member.ref),
        )
        representative, *supporting = ranked
        representative.finding.dedup_role = "representative"
        representative.finding.dedup_cluster_key = cluster_key
        for member in supporting:
            member.finding.dedup_role = "supporting"
            member.finding.dedup_cluster_key = cluster_key
        clusters.append(
            FindingCluster(
                cluster_key=cluster_key,
                representative=representative,
                supporting=supporting,
            )
        )
    clusters.sort(
        key=lambda cluster: (
            -_selection_score(cluster.representative),
            cluster.cluster_key,
        )
    )
    return clusters


def _selection_score(item: DedupFinding) -> float:
    """In-cluster ranking key: interestingness x final (both deterministic)."""
    score = item.finding.score
    final = score.final if score is not None else 0.0
    if score is not None and score.interestingness is not None:
        interest = score.interestingness
    else:
        interest = finding_interestingness(
            item.finding, dataset_row_count=item.dataset_row_count
        ).value
    return final * interest


def _cluster_key(item: DedupFinding) -> str:
    columns = "|".join(
        sorted({column.strip().lower() for column in item.columns if column.strip()})
    )
    direction = item.direction if item.direction is not None else _auto_direction(item.finding)
    magnitude = item.magnitude if item.magnitude is not None else _auto_magnitude(item.finding)
    return f"cols={columns};dir={_direction_sign(direction)};mag={_magnitude_bucket(magnitude)}"


def _auto_direction(finding: QuestionFinding) -> float | None:
    values = _evidence_values(finding)
    for locator in ("effect_size", "statistic"):
        signal = values.get(locator)
        if signal is not None:
            return signal
    for locator in ("outlier_percent", *_MODEL_DEVIATION_LOCATORS):
        signal = values.get(locator)
        if signal is not None:
            return signal
    return None


def _auto_magnitude(finding: QuestionFinding) -> float | None:
    values = _evidence_values(finding)
    for locator in ("effect_size", "statistic", "outlier_percent", *_MODEL_DEVIATION_LOCATORS):
        signal = values.get(locator)
        if signal is not None:
            return abs(signal)
    return None


def _direction_sign(direction: float | None) -> str:
    if direction is None:
        return "na"
    if direction > 0:
        return "+"
    if direction < 0:
        return "-"
    return "0"


def _magnitude_bucket(magnitude: float | None) -> str:
    if magnitude is None:
        return "na"
    if magnitude == 0:
        return "zero"
    return str(math.floor(math.log10(abs(magnitude))))
