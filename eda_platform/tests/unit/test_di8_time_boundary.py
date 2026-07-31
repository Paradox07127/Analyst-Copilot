"""DI8-D time boundary defense: partial edge periods must not fake a trend.

The evidence chain guarantees the numbers were computed correctly; these tests
guarantee the conclusion is not tricked by the data boundary — a partial first
or last period (like Olist's residual final month) is detected, and trend
claims leaning on it are degraded with a structured ``time_boundary_flag``,
while claims already truncated to complete periods pass untouched.
"""

from __future__ import annotations

from eda_platform.schemas.artifacts import EvidenceRef, SqlResult
from eda_platform.schemas.reports import ReportAudit, ReportBundle, ReportClaim, ReportStatus
from eda_platform.tools.report_validator import apply_semantic_gate
from eda_platform.tools.time_boundary import (
    TimeBucket,
    assess_buckets,
    assess_sql_result,
    buckets_from_sql_result,
    parse_period_label,
    split_edge_values,
)


def _bucket(label: str, count: float | None) -> TimeBucket:
    key = parse_period_label(label)
    assert key is not None
    return TimeBucket(label=label, sort_key=key, record_count=count)


def _trend_sql_result() -> SqlResult:
    """Monthly series whose last month is a residual partial period.

    The metric drops from ~1.75 to 1.1 only because 2018-09 has a handful of
    records — the "1.75 -> 1.10 fake trend" from the Olist manual comparison.
    """
    rows = [
        {"month": "2018-04", "order_count": 900, "avg_payment": 1.72},
        {"month": "2018-05", "order_count": 1000, "avg_payment": 1.75},
        {"month": "2018-06", "order_count": 950, "avg_payment": 1.74},
        {"month": "2018-07", "order_count": 1020, "avg_payment": 1.76},
        {"month": "2018-08", "order_count": 980, "avg_payment": 1.73},
        {"month": "2018-09", "order_count": 12, "avg_payment": 1.10},
    ]
    return SqlResult(
        sql="SELECT ...",
        columns=["month", "order_count", "avg_payment"],
        dtypes={"month": "VARCHAR", "order_count": "BIGINT", "avg_payment": "DOUBLE"},
        rows_preview=rows,
        row_count=len(rows),
    )


# --------------------------------------------------------------------------- #
# Label parsing + bucket assessment
# --------------------------------------------------------------------------- #
def test_parse_period_label_accepts_common_period_shapes() -> None:
    assert parse_period_label("2018") == (2018, 0, 0)
    assert parse_period_label("2018-09") == (2018, 9, 0)
    assert parse_period_label("2018-09-30") == (2018, 9, 30)
    assert parse_period_label("2018/09/30") == (2018, 9, 30)
    assert parse_period_label("East") is None
    assert parse_period_label("123") is None
    assert parse_period_label(42) is None


def test_partial_first_and_last_buckets_are_detected() -> None:
    buckets = [
        _bucket("2018-01", 5),
        _bucket("2018-02", 100),
        _bucket("2018-03", 110),
        _bucket("2018-04", 90),
        _bucket("2018-05", 105),
        _bucket("2018-06", 8),
    ]

    assessment = assess_buckets(buckets)

    assert assessment.partial_edge_labels == ["2018-01", "2018-06"]
    assert assessment.interior_median_count == 102.5
    assert assessment.complete_labels == ["2018-02", "2018-03", "2018-04", "2018-05"]


def test_complete_periods_are_not_falsely_flagged() -> None:
    buckets = [
        _bucket("2018-01", 100),
        _bucket("2018-02", 110),
        _bucket("2018-03", 90),
        _bucket("2018-04", 105),
    ]
    assert assess_buckets(buckets).partial_edge_labels == []


def test_short_series_below_min_buckets_is_never_flagged() -> None:
    buckets = [_bucket("2018-01", 1), _bucket("2018-02", 100), _bucket("2018-03", 1)]
    assert assess_buckets(buckets).partial_edge_labels == []


def test_missing_count_column_yields_no_flags() -> None:
    result = SqlResult(
        sql="SELECT ...",
        columns=["month", "avg_payment"],
        dtypes={"month": "VARCHAR", "avg_payment": "DOUBLE"},
        rows_preview=[
            {"month": "2018-04", "avg_payment": 1.72},
            {"month": "2018-05", "avg_payment": 1.75},
            {"month": "2018-06", "avg_payment": 1.74},
            {"month": "2018-07", "avg_payment": 1.10},
        ],
        row_count=4,
    )
    assessment = assess_sql_result(result)
    assert assessment is not None
    assert assessment.count_column is None
    assert not assessment.flagged


def test_non_time_series_result_is_not_assessed() -> None:
    result = SqlResult(
        sql="SELECT ...",
        columns=["region", "order_count"],
        dtypes={"region": "VARCHAR", "order_count": "BIGINT"},
        rows_preview=[
            {"region": "East", "order_count": 10},
            {"region": "West", "order_count": 20},
            {"region": "North", "order_count": 30},
        ],
        row_count=3,
    )
    assert assess_sql_result(result) is None


def test_buckets_and_edge_value_split_from_sql_result() -> None:
    result = _trend_sql_result()
    extracted = buckets_from_sql_result(result)
    assert extracted is not None
    time_column, buckets = extracted
    assert time_column == "month"
    assert len(buckets) == 6

    assessment = assess_sql_result(result)
    assert assessment is not None
    assert assessment.time_column == "month"
    assert assessment.count_column == "order_count"
    assert assessment.partial_edge_labels == ["2018-09"]

    partial_values, complete_values = split_edge_values(result, assessment)
    assert 1.10 in partial_values
    assert 1.75 in complete_values
    assert 1.10 not in complete_values


# --------------------------------------------------------------------------- #
# Validator enforcement on trend claims
# --------------------------------------------------------------------------- #
def _bundle_with(claim: ReportClaim) -> tuple[ReportBundle, ReportAudit]:
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    bundle.sections[0].claims.append(claim)
    return bundle, ReportAudit(status=ReportStatus.VALIDATED)


def _trend_claim(text: str) -> ReportClaim:
    return ReportClaim(
        id="trend_claim",
        text=text,
        evidence=[EvidenceRef(kind="table", artifact_id="sql_trend", locator="rows")],
    )


def test_claim_leaning_on_partial_tail_number_is_degraded_with_flag() -> None:
    claim = _trend_claim("Average payment declined to 1.10 by the end of the window.")
    bundle, audit = _bundle_with(claim)

    outcome = apply_semantic_gate(
        bundle, audit, sql_results={"sql_trend": _trend_sql_result()}
    )

    assert outcome.verdict == "degraded"
    assert outcome.time_boundary_truncations == 1
    assert claim.gate_verdict == "degraded"
    assert claim.time_boundary_flag == "partial_periods:2018-09"
    assert audit.time_boundary_truncations == 1
    assert any(f.code == "time_boundary_partial_period" for f in audit.findings)
    # Degraded, not rejected: the claim still publishes and status is untouched.
    assert bundle.sections[0].claims == [claim]
    assert audit.status is ReportStatus.VALIDATED


def test_claim_naming_partial_period_label_is_degraded() -> None:
    claim = _trend_claim("The 2018-09 bucket shows a sharp drop.")
    bundle, audit = _bundle_with(claim)

    apply_semantic_gate(bundle, audit, sql_results={"sql_trend": _trend_sql_result()})

    assert claim.time_boundary_flag == "partial_periods:2018-09"


def test_qualitative_trend_claim_over_partial_series_is_degraded() -> None:
    claim = _trend_claim("Average payment shows a clear declining trend.")
    bundle, audit = _bundle_with(claim)

    apply_semantic_gate(bundle, audit, sql_results={"sql_trend": _trend_sql_result()})

    assert claim.gate_verdict == "degraded"
    assert claim.time_boundary_flag == "partial_periods:2018-09"


def test_truncated_claim_on_complete_periods_passes_untouched() -> None:
    claim = _trend_claim(
        "Across complete months the average payment held near 1.75."
    )
    bundle, audit = _bundle_with(claim)

    outcome = apply_semantic_gate(
        bundle, audit, sql_results={"sql_trend": _trend_sql_result()}
    )

    # The time-boundary axis leaves the claim untouched; the bundle verdict is
    # now the F6 strong ratio, and sql-only evidence caps at indicative.
    assert claim.gate_verdict == "pass"
    assert claim.time_boundary_flag == ""
    assert audit.time_boundary_truncations == 0
    assert claim.confidence_label == "indicative"
    assert outcome.verdict == "degraded"


def test_complete_series_claim_is_not_flagged() -> None:
    rows = [
        {"month": "2018-04", "order_count": 900, "avg_payment": 1.72},
        {"month": "2018-05", "order_count": 1000, "avg_payment": 1.75},
        {"month": "2018-06", "order_count": 950, "avg_payment": 1.74},
        {"month": "2018-07", "order_count": 1020, "avg_payment": 1.76},
    ]
    result = SqlResult(
        sql="SELECT ...",
        columns=["month", "order_count", "avg_payment"],
        dtypes={"month": "VARCHAR", "order_count": "BIGINT", "avg_payment": "DOUBLE"},
        rows_preview=rows,
        row_count=len(rows),
    )
    claim = _trend_claim("Average payment held stable around 1.75 month over month.")
    bundle, audit = _bundle_with(claim)

    apply_semantic_gate(bundle, audit, sql_results={"sql_trend": result})

    # Not flagged on the time-boundary axis (the F6 ratio verdict is graded
    # separately and sql-only evidence is indicative, not strong).
    assert claim.gate_verdict == "pass"
    assert claim.time_boundary_flag == ""
