"""Finding text must answer the question it was asked.

2026-07-22 audit: every ranked result was phrased "The strongest is X, followed
by Y", so a count distribution reported "the strongest is No (total_employees
1233)" instead of the 16.12% attrition rate the question asked for.
"""

from __future__ import annotations

from eda_platform.drivers.investigation_orchestrator import _probe_finding_numbers_supported
from eda_platform.drivers.question_exec import _ranked_finding
from eda_platform.schemas.questions import QuestionCandidate, QuestionScore
from eda_platform.tools.report_validator import extract_numbers


def _candidate(question: str, dataset: str) -> QuestionCandidate:
    return QuestionCandidate(
        question_id="q_" + str(abs(hash(question)) % 10**8),
        question_en=question,
        origin="llm",
        target_datasets=[dataset],
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.775,
        ),
    )


def test_count_distribution_is_answered_as_shares_not_a_ranking() -> None:
    """A count over few labels is a distribution; "the strongest is No" answers nothing."""
    candidate = _candidate("Can we predict employee attrition?", "hr.csv")
    rows = [
        {"Attrition": "No", "total_employees": 1233},
        {"Attrition": "Yes", "total_employees": 237},
    ]

    finding = _ranked_finding(
        candidate,
        "sql_1",
        rows,
        label_column="Attrition",
        metric_column="total_employees",
        sql="SELECT Attrition, COUNT(*) AS total_employees FROM hr GROUP BY Attrition",
    )

    assert "16.12%" in finding.text
    assert "1,470" in finding.text
    assert "strongest" not in finding.text
    # Derived shares must be evidence-backed or the validator prunes the claim.
    assert _probe_finding_numbers_supported(finding)


def test_magnitude_rankings_keep_the_ranking_phrasing() -> None:
    # 2026-07-24: ranking phrasing now requires the SQL to declare the order.
    candidate = _candidate("How does payment value differ by payment type?", "payments.csv")
    rows = [
        {"payment_type": "credit_card", "avg_payment_value": 163.319},
        {"payment_type": "boleto", "avg_payment_value": 145.034},
    ]

    finding = _ranked_finding(
        candidate,
        "sql_2",
        rows,
        label_column="payment_type",
        metric_column="avg_payment_value",
        sql=(
            "SELECT payment_type, AVG(payment_value) AS avg_payment_value "
            "FROM p GROUP BY 1 ORDER BY avg_payment_value DESC"
        ),
    )

    assert "Ranked by avg_payment_value descending" in finding.text
    assert "top is credit_card" in finding.text


def test_number_extraction_understands_thousands_separators() -> None:
    assert extract_numbers("GMV 13,591,643.70 over 1,470 rows") == [
        (13_591_643.70, False),
        (1470.0, False),
    ]


def test_a_sum_column_named_total_is_never_turned_into_shares() -> None:
    """`total_revenue` is a SUM: summing it into a denominator would be nonsense."""
    candidate = _candidate("Which region earns most?", "sales.csv")
    rows = [{"region": "EU", "total_revenue": 900}, {"region": "US", "total_revenue": 100}]

    finding = _ranked_finding(
        candidate,
        "sql_3",
        rows,
        label_column="region",
        metric_column="total_revenue",
        sql=(
            "SELECT region, SUM(revenue) AS total_revenue FROM sales "
            "GROUP BY region ORDER BY total_revenue DESC"
        ),
    )

    assert "Ranked by total_revenue descending" in finding.text
    assert "%" not in finding.text
