"""F2 cross-review fixes: scientific-notation tokens, threshold eligibility,
SQL count-unit exactness, and the relative rounded epsilon."""

from __future__ import annotations

from eda_platform.schemas.artifacts import EvidenceRef, SqlResult
from eda_platform.schemas.reports import ReportClaim
from eda_platform.tools import report_validator as rv
from eda_platform.tools.evidence import (
    EvidenceAnalysisTable,
    EvidenceArtifactSummary,
    EvidencePack,
    EvidenceStatTest,
)

# --- Fix 1: scientific-notation tokens ---


def test_sci_notation_token_parses_whole_number() -> None:
    tokens = rv._numeric_tokens_from_text("p < 1e-10")
    assert [(t.value, t.is_percent, t.threshold_op) for t in tokens] == [
        (1e-10, False, "<")
    ]
    assert tokens[0].decimals == 0


def test_sci_notation_decimals_come_from_mantissa() -> None:
    (token,) = rv._numeric_tokens_from_text("statistic is 1.38e-8")
    assert token.value == 1.38e-8
    assert token.decimals == 2


def test_plain_tokens_unchanged_next_to_sci() -> None:
    values = [t.value for t in rv._numeric_tokens_from_text("1,470 rows, p = 2.5e-3")]
    assert values == [1470.0, 2.5e-3]


def test_sci_bound_not_satisfied_by_larger_p_value() -> None:
    # Truth 1.38e-8 does NOT satisfy "p < 1e-10"; pre-fix the token was cut to
    # "1" and the bound trivially passed.
    claim = ReportClaim(
        text="The difference is significant (p < 1e-10).",
        evidence=[
            EvidenceRef(kind="stat", artifact_id="stat_1", locator="p_value")
        ],
    )
    statuses = rv._numeric_token_statuses(
        claim, evidence_pack=_pack(), numeric_tolerance=0.01, sql_results={}
    )
    assert [(s.number, s.status) for s in statuses] == [(1e-10, "failed")]


# --- Fix 2: threshold eligibility + own-unit pool ---


def test_threshold_without_eligible_values_falls_back_to_exact() -> None:
    # "< 9999" over a Table int pool {1470}: no stat-derived value is eligible
    # for inequalities, so the token is judged like a plain number and fails.
    claim = ReportClaim(
        text="Records processed: revenue < 9999.",
        evidence=[
            EvidenceRef(kind="table", artifact_id="table_1", locator="rows[0]")
        ],
    )
    statuses = rv._numeric_token_statuses(
        claim, evidence_pack=_pack(), numeric_tolerance=0.01, sql_results={}
    )
    assert [(s.number, s.status) for s in statuses] == [(9999.0, "failed")]


def test_threshold_fallback_still_accepts_true_value() -> None:
    # Fallback means plain-token rules, not unconditional failure.
    claim = ReportClaim(
        text="Records processed: < 1470.",
        evidence=[
            EvidenceRef(kind="table", artifact_id="table_1", locator="rows[0]")
        ],
    )
    statuses = rv._numeric_token_statuses(
        claim, evidence_pack=_pack(), numeric_tolerance=0.01, sql_results={}
    )
    assert [(s.number, s.status) for s in statuses] == [(1470.0, "number_verified")]


def test_threshold_percent_token_cannot_borrow_raw_values() -> None:
    # raw 7827/96476 + percent 8.1129: "share > 9%" must not be satisfied by
    # the raw 7827; the percent pool has no eligible value and 8.1129 is not 9.
    claim = ReportClaim(
        text="Late share > 9% of orders.",
        evidence=[EvidenceRef(kind="sql", artifact_id="sql_1", locator="rows[0]")],
    )
    statuses = rv._numeric_token_statuses(
        claim,
        evidence_pack=_pack(),
        numeric_tolerance=0.01,
        sql_results=_sql_results(),
    )
    assert [(s.number, s.is_percent, s.status) for s in statuses] == [
        (9.0, True, "failed")
    ]


def test_threshold_on_stat_p_value_stays_verified() -> None:
    # Control: stat p_value/statistic/effect_size locators are eligible, so a
    # true inequality keeps verifying.
    claim = ReportClaim(
        text="The difference is significant (p < 0.0001).",
        evidence=[
            EvidenceRef(kind="stat", artifact_id="stat_1", locator="p_value")
        ],
    )
    statuses = rv._numeric_token_statuses(
        claim, evidence_pack=_pack(), numeric_tolerance=0.01, sql_results={}
    )
    assert [(s.number, s.status) for s in statuses] == [(0.0001, "number_verified")]


def test_sample_size_locator_is_not_threshold_eligible() -> None:
    # sample_size is a cardinality, not a stat quantity: "< 9999" over it must
    # fall back to exact matching and fail.
    claim = ReportClaim(
        text="Groups compared: < 9999.",
        evidence=[
            EvidenceRef(kind="stat", artifact_id="stat_1", locator="sample_size")
        ],
    )
    statuses = rv._numeric_token_statuses(
        claim, evidence_pack=_pack(), numeric_tolerance=0.01, sql_results={}
    )
    assert [(s.number, s.status) for s in statuses] == [(9999.0, "failed")]


# --- Fix 3: SQL units=="count" derives exact even for float-typed cells ---


def test_sql_count_unit_overrides_float_type_to_exact() -> None:
    evidence = EvidenceRef(kind="sql", artifact_id="sql_1", locator="late_rows")
    values = rv._resolve_evidence_numbers(evidence, _pack(), _sql_results())
    assert [(v[0], v[1], v[2]) for v in values] == [(7827.0, "raw", "exact")]


def test_sql_count_unit_rejects_off_by_one() -> None:
    claim = ReportClaim(
        text="7828 orders arrived late.",
        evidence=[EvidenceRef(kind="sql", artifact_id="sql_1", locator="late_rows")],
    )
    statuses = rv._numeric_token_statuses(
        claim,
        evidence_pack=_pack(),
        numeric_tolerance=0.01,
        sql_results=_sql_results(),
    )
    assert [(s.number, s.status) for s in statuses] == [(7828.0, "failed")]


# --- Fix 4: relative rounded epsilon ---


def test_rounded_window_is_relative_at_deep_decimals() -> None:
    token = rv._NumericToken(value=0.1234567890123, is_percent=False, decimals=13)
    # Half-ulp at 13 decimals is 5e-14; a 9e-13 offset must be rejected (the
    # old absolute +1e-12 slack widened the window 18x and accepted it).
    assert not rv._value_supports_token(token, 0.1234567890123 + 9e-13, "rounded")
    assert rv._value_supports_token(token, 0.1234567890123 + 4e-14, "rounded")


def test_rounded_window_unchanged_at_display_decimals() -> None:
    token = rv._NumericToken(value=0.20, is_percent=False, decimals=2)
    assert rv._value_supports_token(token, 0.196566, "rounded")
    assert not rv._value_supports_token(token, 0.2051, "rounded")


# --- Fixtures ---


def _pack() -> EvidencePack:
    return EvidencePack(
        payload_policy="schema+aggregates",
        artifact_index={
            "table_1": EvidenceArtifactSummary(
                artifact_id="table_1",
                artifact_type="Table",
                title="Numeric summary",
                dataset_id="ds_1",
            ),
            "stat_1": EvidenceArtifactSummary(
                artifact_id="stat_1",
                artifact_type="StatTestResult",
                title="t-test",
                dataset_id="ds_1",
            ),
        },
        analysis_tables=[
            EvidenceAnalysisTable(
                artifact_id="table_1",
                dataset_id="ds_1",
                title="Numeric summary",
                kind="aggregation",
                description="Row summary",
                rows=[{"revenue": 1470}],
            )
        ],
        stat_tests=[
            EvidenceStatTest(
                artifact_id="stat_1",
                dataset_id="ds_1",
                test_type="t_test",
                statistic=5.83,
                p_value=1.38e-8,
                effect_size=0.42,
                sample_size=1470,
            )
        ],
    )


def _sql_results() -> dict[str, SqlResult]:
    return {
        "sql_1": SqlResult(
            sql="select 1",
            columns=["row_count", "late_rows", "late_delivery_rate_percent"],
            dtypes={
                "row_count": "int64",
                "late_rows": "float64",
                "late_delivery_rate_percent": "float64",
            },
            units={
                "row_count": "count",
                "late_rows": "count",
                "late_delivery_rate_percent": "percent",
            },
            rows_preview=[
                {
                    "row_count": 96476,
                    "late_rows": 7827.0,
                    "late_delivery_rate_percent": 8.112898544715785,
                }
            ],
            row_count=1,
        )
    }
