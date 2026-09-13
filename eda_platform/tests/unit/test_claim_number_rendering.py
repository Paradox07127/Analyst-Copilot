"""Every figure a claim cites must round-trip through the report's numeric gate.

`{value:g}` drops to scientific notation past six significant digits, so the
2026-08-26 Compare run offered the model "1.35916e+07" for a GMV of
13591643.70001419; the gate read the token as 13591600, disagreed with the
evidence, and pruned the claim as numeric_mismatch. Rendering, not tolerance, is
what has to change: the gate's half-ULP window is the contract these texts owe.
"""

from __future__ import annotations

import re

from eda_platform.agents.interpretation import _allowed_numbers, _build_payload
from eda_platform.drivers.question_exec import (
    _generic_finding,
    _ranked_finding,
    _single_value_finding,
    _trend_finding,
)
from eda_platform.schemas.questions import (
    EvidenceRef,
    QuestionCandidate,
    QuestionFinding,
    QuestionScore,
)
from eda_platform.tools.report_validator import (
    numeric_tokens_from_text,
    value_supports_token,
)

_SCIENTIFIC = re.compile(r"\d[eE][+-]?\d")

# Real values from Compare session sess_1787771303025_kojblh.
_GMV = 13591643.70001419
_ORDER_COUNT = 112650.0


def _candidate(question: str = "What is the total GMV?") -> QuestionCandidate:
    return QuestionCandidate(
        question_id="q_numbers",
        question_en=question,
        origin="llm",
        target_datasets=["order_items"],
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.775,
        ),
    )


def _unsupported(finding: QuestionFinding) -> list[float]:
    """Tokens in the text that no cited evidence value supports."""
    values = [
        float(reference.value)
        for reference in finding.evidence
        if isinstance(reference.value, int | float)
    ]
    return [
        token.value
        for token in numeric_tokens_from_text(finding.text)
        if not any(value_supports_token(token, value, "rounded") for value in values)
    ]


def test_a_single_value_finding_cites_a_large_sum_the_gate_accepts() -> None:
    finding = _single_value_finding(
        _candidate(),
        artifact_id="sql_gmv",
        row={"gmv_total": _GMV},
        column="gmv_total",
    )
    assert not _SCIENTIFIC.search(finding.text), finding.text
    assert _unsupported(finding) == [], finding.text


def test_a_generic_single_row_finding_cites_a_large_sum_the_gate_accepts() -> None:
    finding = _generic_finding(
        _candidate("Which category carries the most GMV?"),
        artifact_id="sql_gmv",
        rows=[{"product_category": "beleza_saude", "gmv_total": _GMV}],
    )
    assert not _SCIENTIFIC.search(finding.text), finding.text
    assert _unsupported(finding) == [], finding.text


def test_a_trend_finding_cites_its_endpoints_the_gate_accepts() -> None:
    finding = _trend_finding(
        _candidate("How did monthly GMV move?"),
        artifact_id="sql_gmv_monthly",
        rows=[
            {"order_month": "2017-01", "avg_gmv": _GMV},
            {"order_month": "2017-02", "avg_gmv": _GMV * 2},
        ],
    )
    assert not _SCIENTIFIC.search(finding.text), finding.text
    assert _unsupported(finding) == [], finding.text


def test_a_ranked_finding_cites_its_group_sizes_the_gate_accepts() -> None:
    """The group-size suffix is a cited figure too, not decoration."""
    sql = (
        "select payment_type, sum(payment_value) as total_payment_value, "
        "count(*) as payment_record_count from payments group by payment_type "
        "order by total_payment_value desc"
    )
    finding = _ranked_finding(
        _candidate("Which payment types carry the most value?"),
        artifact_id="sql_ranked",
        rows=[
            {
                "payment_type": "credit_card",
                "total_payment_value": _GMV,
                "payment_record_count": 1234567.0,
            },
            {
                "payment_type": "boleto",
                "total_payment_value": _GMV / 4,
                "payment_record_count": 2345678.0,
            },
        ],
        label_column="payment_type",
        metric_column="total_payment_value",
        sql=sql,
        total_rows=2,
    )
    assert not _SCIENTIFIC.search(finding.text), finding.text
    assert _unsupported(finding) == [], finding.text


def test_the_numbers_offered_to_the_interpreter_are_gate_admissible() -> None:
    """allowed_numbers is where the model copies its figures from."""
    finding = QuestionFinding(
        text=f"What is the total GMV? Total GMV over 112650 rows is {_GMV}.",
        evidence=[
            EvidenceRef(
                kind="sql",
                artifact_id="sql_gmv",
                locator="rows_preview[0].row_count",
                value=_ORDER_COUNT,
            ),
            EvidenceRef(
                kind="sql",
                artifact_id="sql_gmv",
                locator="rows_preview[0].gmv_total",
                value=_GMV,
            ),
        ],
    )
    payload = _build_payload(
        question="What is the total GMV?",
        findings=[finding],
        method_context="",
        limitations=[],
        allowed_numbers=_allowed_numbers([finding]),
    )
    offered = payload["allowed_numbers"]
    assert not any(_SCIENTIFIC.search(text) for text in offered), offered
    for text, (value, _) in zip(offered, _allowed_numbers([finding]), strict=True):
        (token,) = numeric_tokens_from_text(text)
        assert value_supports_token(token, value, "rounded"), (text, value)
