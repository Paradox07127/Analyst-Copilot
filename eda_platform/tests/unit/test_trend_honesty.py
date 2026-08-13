"""Trend claims must not read endpoints off a truncated period series.

2026-08-12 deepseek run: a daily-grain freight series hit the row limit
(preview 50 of 100, `truncated: True`), and the finding published "increased
from 6.01 to 19.8 across the returned periods" — the "endpoints" were a 4-row
day and an arbitrary mid-series day, while the real series ran on for months.
A truncated trend now states its observed window and refuses a full-series
direction claim.
"""

from __future__ import annotations

from eda_platform.core.ids import make_artifact_id
from eda_platform.drivers.question_exec import _findings_for
from eda_platform.schemas.artifacts import Artifact, ArtifactType, SqlResult
from eda_platform.schemas.questions import QuestionCandidate, QuestionScore


def _trend_candidate() -> QuestionCandidate:
    return QuestionCandidate(
        question_id="q_demo",
        question_en="How is freight_value trending over shipping_limit_date?",
        origin="template",
        template_id="trend",
        target_datasets=["items.csv"],
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.6,
        ),
    )


def _sql_artifact(
    rows: list[dict[str, object]], *, row_count: int | None = None, truncated: bool = False
) -> Artifact:
    columns = list(rows[0]) if rows else []
    payload = SqlResult(
        sql="select period, avg(freight_value) as avg_freight_value from items group by 1",
        columns=columns,
        dtypes=dict.fromkeys(columns, "DOUBLE"),
        rows_preview=rows,
        row_count=len(rows) if row_count is None else row_count,
        truncated=truncated,
    ).model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("sql", payload),
        type=ArtifactType.SQL_RESULT,
        project_id="project_demo",
        session_id="run_demo",
        payload=payload,
    )


def _rows(n: int) -> list[dict[str, object]]:
    return [
        {"period": f"2017-01-{index + 1:02d}", "avg_freight_value": 100.0 + index}
        for index in range(n)
    ]


def test_full_series_trend_phrasing_is_unchanged() -> None:
    artifact = _sql_artifact(_rows(3))
    finding = _findings_for(_trend_candidate(), artifact)[0]
    assert finding.text == (
        "How is freight_value trending over shipping_limit_date? "
        "The metric increased from 100 to 102 across the returned periods."
    )


def test_truncated_series_does_not_claim_a_full_series_trend() -> None:
    artifact = _sql_artifact(_rows(50), row_count=100, truncated=True)
    finding = _findings_for(_trend_candidate(), artifact)[0]
    # No direction verdict over a series whose tail was never seen.
    assert "increased" not in finding.text
    assert "decreased" not in finding.text
    # The window and the unseen remainder are both named.
    assert "first 50 of 100" in finding.text
    assert "50 later periods were not returned" in finding.text
    # The endpoint locators still bind the numbers that ARE claimed.
    locators = [ref.locator for ref in finding.evidence]
    assert "rows_preview[0].avg_freight_value" in locators
    assert "rows_preview[49].avg_freight_value" in locators


def test_truncated_series_still_reports_observed_window_values() -> None:
    artifact = _sql_artifact(_rows(50), row_count=100, truncated=True)
    finding = _findings_for(_trend_candidate(), artifact)[0]
    assert "100" in finding.text and "149" in finding.text
