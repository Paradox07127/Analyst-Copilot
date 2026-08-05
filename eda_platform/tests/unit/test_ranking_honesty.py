"""Ranking claims in finding text must come from the SQL's ORDER BY, not row order.

2026-07-24 audit (4 live instances): multi-column group-by results were phrased
"The strongest is <first row>" regardless of any real ordering, and the L1
interpretation layer escalated that into fabricated causal conclusions
(e.g. "the strongest is Yes (Age 41)" -> "41-year-olds churn most").
"""

from __future__ import annotations

from eda_platform.agents.interpretation import _build_payload, interpret_findings
from eda_platform.core.ids import make_artifact_id
from eda_platform.drivers.question_exec import (
    _findings_for,
    _parse_order_by,
    _ranking_basis,
    _successful_qexec_artifact,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType, SqlResult
from eda_platform.schemas.questions import QuestionCandidate, QuestionScore

_RANKING_WORDS = ("strongest", "top", "leading", "highest", "first is")


def _sql_artifact(rows: list[dict[str, object]], *, sql: str = "select 1") -> Artifact:
    columns = list(rows[0]) if rows else []
    payload = SqlResult(
        sql=sql,
        columns=columns,
        dtypes=dict.fromkeys(columns, "DOUBLE"),
        rows_preview=rows,
        row_count=len(rows),
    ).model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("sql", payload),
        type=ArtifactType.SQL_RESULT,
        project_id="project_demo",
        session_id="run_demo",
        payload=payload,
    )


def _candidate(question: str, *, template_id: str | None = None) -> QuestionCandidate:
    return QuestionCandidate(
        question_id="q_demo",
        question_en=question,
        origin="template" if template_id is not None else "llm",
        template_id=template_id,
        target_datasets=["genai.csv"],
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.6,
        ),
    )


_HALL_ROWS: list[dict[str, object]] = [
    {"model_name": "GPT-4", "rag_enabled": 0, "hall_rate": 0.56},
    {"model_name": "Claude 3.7", "rag_enabled": 1, "hall_rate": 0.41},
    {"model_name": "Claude 3.7", "rag_enabled": 0, "hall_rate": 0.22},
]

_HALL_SQL_ORDERED = (
    "SELECT model_name, rag_enabled, AVG(is_hallucination) AS hall_rate "
    "FROM genai GROUP BY model_name, rag_enabled ORDER BY hall_rate DESC"
)
_HALL_SQL_UNORDERED = (
    "SELECT model_name, rag_enabled, AVG(is_hallucination) AS hall_rate "
    "FROM genai GROUP BY model_name, rag_enabled"
)


# --------------------------------------------------------------------------- #
# 1. With ORDER BY: rank by the ORDER BY column, and say so
# --------------------------------------------------------------------------- #
def test_order_by_column_is_the_ranking_basis_not_the_first_column() -> None:
    candidate = _candidate("Which configuration hallucinates most?")
    artifact = _sql_artifact(_HALL_ROWS, sql=_HALL_SQL_ORDERED)

    finding = _findings_for(candidate, artifact)[0]
    text = finding.text

    assert "ranked by hall_rate descending" in text.lower()
    assert "0.56" in text
    # The metric cited is the ORDER BY column, never the group-size heuristic.
    locators = [ref.locator for ref in finding.evidence]
    assert "rows_preview[0].hall_rate" in locators
    assert "The strongest is" not in text


def test_multi_group_keys_are_combined_into_distinct_labels() -> None:
    candidate = _candidate("Which configuration hallucinates most?")
    artifact = _sql_artifact(_HALL_ROWS, sql=_HALL_SQL_ORDERED)

    text = _findings_for(candidate, artifact)[0].text

    assert "model_name=Claude 3.7 / rag_enabled=1" in text
    assert "model_name=Claude 3.7 / rag_enabled=0" in text
    assert "model_name=GPT-4 / rag_enabled=0" in text


# --------------------------------------------------------------------------- #
# 2. Without ORDER BY: neutral description, no ranking vocabulary
# --------------------------------------------------------------------------- #
def test_no_order_by_yields_neutral_description() -> None:
    candidate = _candidate("Which configuration hallucinates most?")
    artifact = _sql_artifact(_HALL_ROWS, sql=_HALL_SQL_UNORDERED)

    text = _findings_for(candidate, artifact)[0].text

    lowered = text.lower()
    assert all(word not in lowered for word in _RANKING_WORDS)
    assert "3 rows" in text


def test_order_by_a_group_key_is_not_a_magnitude_ranking() -> None:
    """Sorting by the label (the live NumCompaniesWorked case) proves nothing."""
    candidate = _candidate("How does attrition vary by companies worked?")
    rows: list[dict[str, object]] = [
        {"NumCompaniesWorked": 0, "attrition_rate": 11.8},
        {"NumCompaniesWorked": 1, "attrition_rate": 18.8},
        {"NumCompaniesWorked": 2, "attrition_rate": 10.9},
    ]
    sql = (
        "SELECT NumCompaniesWorked, AVG(a) AS attrition_rate FROM hr "
        "GROUP BY NumCompaniesWorked ORDER BY NumCompaniesWorked"
    )

    text = _findings_for(candidate, _sql_artifact(rows, sql=sql))[0].text

    lowered = text.lower()
    assert all(word not in lowered for word in _RANKING_WORDS)


def test_rows_contradicting_the_parsed_order_disable_ranking() -> None:
    """If preview rows are not monotone in the parsed column, the parse is wrong."""
    candidate = _candidate("Which configuration hallucinates most?")
    rows: list[dict[str, object]] = [
        {"model_name": "A", "hall_rate": 0.2},
        {"model_name": "B", "hall_rate": 0.6},
    ]
    artifact = _sql_artifact(
        rows, sql="SELECT model_name, AVG(h) AS hall_rate FROM t GROUP BY 1 ORDER BY hall_rate DESC"
    )

    text = _findings_for(candidate, artifact)[0].text

    assert all(word not in text.lower() for word in _RANKING_WORDS)


# --------------------------------------------------------------------------- #
# 3. Conservative ORDER BY parser: bail out rather than guess
# --------------------------------------------------------------------------- #
def test_parse_order_by_edges() -> None:
    assert _parse_order_by("select a from t order by b desc") == ("b", "descending")
    assert _parse_order_by("select a from t order by b") == ("b", "ascending")
    assert _parse_order_by('select a from t order by "B col" desc limit 5;') == (
        "B col",
        "descending",
    )
    assert _parse_order_by("select a from t order by t.b desc") == ("b", "descending")
    # First key of a multi-key ORDER BY still fixes the primary order.
    assert _parse_order_by("select a from t order by b desc, c") == ("b", "descending")
    assert _parse_order_by("select a from t order by 2 desc") == ("2", "descending")
    # Only the clause outside every parenthesis orders the result, so a nested
    # one neither counts nor disqualifies. Widened 2026-08-05: counting the
    # ORDER BY inside each `RANK() OVER (...)` made a correctly sorted 48-team
    # result read as unordered, and the fallback published the range of
    # `team_id` as the report's opening finding. `_ranking_basis` still checks
    # the parsed key against the rows, so a wrong parse cannot become a claim.
    assert (
        _parse_order_by("select * from (select a from t order by b) order by c desc")
        == ("c", "descending")
    )
    assert _parse_order_by(
        "select rank() over (order by b desc) as r from t order by r desc"
    ) == ("r", "descending")
    # Bail-outs: no clause, a clause that only exists inside a window, expressions.
    assert _parse_order_by("select a from t") is None
    assert _parse_order_by("select rank() over (order by b desc) from t") is None
    assert _parse_order_by("select a from t order by count(*) desc") is None


def test_positional_order_by_resolves_against_result_columns() -> None:
    rows: list[dict[str, object]] = [
        {"model_name": "A", "hall_rate": 0.6},
        {"model_name": "B", "hall_rate": 0.2},
    ]
    sql = "SELECT model_name, AVG(h) AS hall_rate FROM t GROUP BY 1 ORDER BY 2 DESC"
    assert _ranking_basis(sql, rows) == ("hall_rate", "descending")


# --------------------------------------------------------------------------- #
# 4. Regression anchors: single-scalar and template paths are byte-identical
# --------------------------------------------------------------------------- #
def test_single_scalar_paths_are_byte_identical_to_pre_change_output() -> None:
    anchors = [
        (
            _candidate("What is the total revenue?"),
            [{"region": "EU", "total_revenue": 900.0}],
            "select 1",
            "What is the total revenue? The total_revenue is 900.",
        ),
        (
            _candidate("What share of rows in Orders were late?"),
            [{"row_count": 96476, "late_rows": 7827, "late_delivery_rate_percent": 8.1129}],
            "select 1",
            "What share of rows in Orders were late? 7827 of 96476 rows (8.1129%).",
        ),
        (
            _candidate(
                "What is the total GMV (sum of Amount) in Creditcard?",
                template_id="domain_metric",
            ),
            [{"row_count": 284807, "gmv_total": 25162590.01, "avg_line_value": 88.3496}],
            "select 1",
            (
                "What is the total GMV (sum of Amount) in Creditcard? "
                "Total GMV over 284807 rows is 25162590.01."
            ),
        ),
    ]
    for candidate, rows, sql, expected in anchors:
        finding = _findings_for(candidate, _sql_artifact(rows, sql=sql))[0]
        assert finding.text == expected


def test_count_split_share_phrasing_is_unchanged() -> None:
    candidate = _candidate("Can we predict employee attrition?")
    artifact = _sql_artifact(
        [
            {"Attrition": "No", "total_employees": 1233},
            {"Attrition": "Yes", "total_employees": 237},
        ],
        sql="SELECT Attrition, COUNT(*) AS total_employees FROM hr GROUP BY Attrition",
    )

    text = _findings_for(candidate, artifact)[0].text

    assert text == (
        "Can we predict employee attrition? Across 1,470 rows the split is "
        "No 83.88% (1,233), Yes 16.12% (237) by Attrition."
    )


# --------------------------------------------------------------------------- #
# 5. ranking_basis flows into the L1 interpretation payload
# --------------------------------------------------------------------------- #
class _PayloadSpyLLM:
    """Minimal LLMClient double that records the interpretation payload."""

    def __init__(self, interpretation: str) -> None:
        self._interpretation = interpretation
        self.payloads: list[dict] = []

    def structured(self, *, task: str, schema: type, payload: dict):
        self.payloads.append(payload)
        return schema(interpretation=self._interpretation)

    def text(self, *, task: str, payload: dict) -> str:
        return self._interpretation

    def last_usage(self) -> None:
        return None


def test_build_payload_carries_ranking_basis_and_instruction() -> None:
    payload = _build_payload(
        question="q",
        findings=[],
        method_context="",
        limitations=[],
        allowed_numbers=[],
        ranking_basis={"column": "hall_rate", "direction": "descending"},
    )
    assert payload["ranking_basis"] == {"column": "hall_rate", "direction": "descending"}
    assert "ranking_basis" in payload["instructions"]

    null_payload = _build_payload(
        question="q",
        findings=[],
        method_context="",
        limitations=[],
        allowed_numbers=[],
    )
    assert null_payload["ranking_basis"] is None


def test_qexec_passes_ranking_basis_to_interpretation() -> None:
    spy = _PayloadSpyLLM("The gap between configurations is small in practice.")
    candidate = _candidate("Which configuration hallucinates most?")
    artifact = _sql_artifact(_HALL_ROWS, sql=_HALL_SQL_ORDERED)

    _successful_qexec_artifact(
        candidate,
        sql_artifact=artifact,
        project_id="project_demo",
        session_id="run_demo",
        parent_ids=[artifact.id],
        plan_summary="grouped hallucination rate",
        llm=spy,  # type: ignore[arg-type]
    )

    assert spy.payloads, "interpretation was never invoked"
    assert spy.payloads[0]["ranking_basis"] == {
        "column": "hall_rate",
        "direction": "descending",
    }


def test_qexec_passes_null_ranking_basis_without_order_by() -> None:
    spy = _PayloadSpyLLM("Rates vary across configurations.")
    candidate = _candidate("Which configuration hallucinates most?")
    artifact = _sql_artifact(_HALL_ROWS, sql=_HALL_SQL_UNORDERED)

    _successful_qexec_artifact(
        candidate,
        sql_artifact=artifact,
        project_id="project_demo",
        session_id="run_demo",
        parent_ids=[artifact.id],
        plan_summary="grouped hallucination rate",
        llm=spy,  # type: ignore[arg-type]
    )

    assert spy.payloads and spy.payloads[0]["ranking_basis"] is None


def test_interpret_findings_defaults_keep_old_signature_working() -> None:
    result = interpret_findings(
        _PayloadSpyLLM("No numbers here, just words."),  # type: ignore[arg-type]
        question="q",
        findings=_findings_for(
            _candidate("What is the total revenue?"),
            _sql_artifact([{"region": "EU", "total_revenue": 900.0}]),
        ),
    )
    assert result.status == "validated"
