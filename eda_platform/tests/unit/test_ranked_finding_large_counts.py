"""A ranked finding must cite large values exactly, not in scientific notation.

`{value:.4g}` renders 25455 as "2.546e+04"; the report's numeric gate then reads
25460, disagrees with the stored evidence value 25455, and prunes the claim as
numeric_mismatch. Every figure in a finding is cited evidence, so it has to be
rendered the way `gate_safe_number` renders one.
"""

from __future__ import annotations

import re

from eda_platform.drivers.question_exec import _ranked_finding
from eda_platform.schemas.questions import QuestionCandidate, QuestionScore

_SCIENTIFIC = re.compile(r"\d[eE][+-]?\d")

_SQL = (
    "select seller_id, sum(payment_value) as total_payment_value "
    "from payments group by seller_id order by total_payment_value desc"
)


def _candidate() -> QuestionCandidate:
    return QuestionCandidate(
        question_id="q_ranked_values",
        question_en="Which sellers carry the most payment value?",
        origin="llm",
        target_datasets=["payments"],
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.775,
        ),
    )


def _rows() -> list[dict[str, object]]:
    return [
        {"seller_id": "seller_a", "total_payment_value": 25455},
        {"seller_id": "seller_b", "total_payment_value": 19784},
        {"seller_id": "seller_c", "total_payment_value": 12413},
    ]


def test_a_ranked_finding_cites_large_values_exactly() -> None:
    finding = _ranked_finding(
        _candidate(),
        artifact_id="sql_ranked",
        rows=_rows(),
        label_column="seller_id",
        metric_column="total_payment_value",
        sql=_SQL,
        total_rows=3,
    )
    assert not _SCIENTIFIC.search(finding.text), finding.text
    cited = {reference.value for reference in finding.evidence}
    for value in (25455.0, 19784.0, 12413.0):
        assert value in cited
        assert str(int(value)) in finding.text.replace(",", ""), finding.text


def test_the_single_leader_sentence_also_cites_its_value_exactly() -> None:
    finding = _ranked_finding(
        _candidate(),
        artifact_id="sql_ranked",
        rows=_rows()[:1],
        label_column=None,
        metric_column="total_payment_value",
        sql=_SQL,
        total_rows=1,
    )
    assert not _SCIENTIFIC.search(finding.text), finding.text
    assert "25455" in finding.text.replace(",", ""), finding.text
