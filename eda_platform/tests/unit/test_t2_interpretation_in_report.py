"""Per-question interpretations must reach the main report as gated claims.

The interpretation is LLM-written at question-execution time, so the report
treats it as a platform-transcribed claim: it cites the question's own
evidence, its numbers pass the same hard gate as every other claim, and a
fabricated number prunes the claim instead of publishing it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

import pytest
from pydantic import BaseModel

from eda_platform.agents.reporting import generate_agentic_report
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    EvidenceRef,
    SqlResult,
)
from eda_platform.schemas.questions import QuestionExecutionResult, QuestionFinding
from eda_platform.schemas.reports import ReportPlanClaim, ReportPlanDraft
from eda_platform.tools.exporter import report_bundle_to_markdown
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.report_validator import is_platform_authored_claim

T = TypeVar("T", bound=BaseModel)

_SQL_RESULT_ID = "sql_qexec_revenue"


class FakeReportPlanLLM:
    def __init__(self, plan: ReportPlanDraft) -> None:
        self.plan = plan

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        return cast(T, self.plan)

    def text(self, *, task: str, payload: dict) -> str:
        return "fake"

    def last_usage(self) -> None:
        return None


def _base_artifacts(tmp_path: Path) -> list[Artifact]:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "order_id,order_date,revenue,region\n"
        "1,2026-01-01,10,East\n"
        "2,2026-01-02,,West\n"
        "3,2026-01-03,30,East\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_sales")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    return [profile]


def _sql_result_artifact() -> Artifact:
    result = SqlResult(
        sql="select region, sum(revenue) as total from sales group by region",
        columns=["region", "total"],
        dtypes={"region": "varchar", "total": "double"},
        rows_preview=[
            {"region": "East", "total": 40.0},
            {"region": "West", "total": 0.0},
        ],
        row_count=2,
    )
    return Artifact(
        id=_SQL_RESULT_ID,
        type=ArtifactType.SQL_RESULT,
        project_id="project_demo",
        session_id="run_demo",
        payload=result.model_dump(mode="json"),
    )


def _qexec_artifact(
    *,
    interpretation: str = "",
    interpretation_status: str = "absent",
) -> Artifact:
    result = QuestionExecutionResult(
        question_id="q_revenue_by_region",
        question="Which region drives the most revenue across the sales file?",
        origin="template",
        plan_summary="Group revenue by region.",
        sql="select region, sum(revenue) as total from sales group by region",
        sql_result_artifact_id=_SQL_RESULT_ID,
        findings=[
            QuestionFinding(
                text="East region total revenue is 40.",
                evidence=[
                    EvidenceRef(
                        kind="table",
                        artifact_id=_SQL_RESULT_ID,
                        locator="rows[0].total",
                        value=40.0,
                    )
                ],
            ),
            QuestionFinding(
                text="West region total revenue is 0.",
                evidence=[
                    EvidenceRef(
                        kind="table",
                        artifact_id=_SQL_RESULT_ID,
                        locator="rows[1].total",
                        value=0,
                    )
                ],
            ),
        ],
        status="succeeded",
        interpretation=interpretation,
        interpretation_status=interpretation_status,  # type: ignore[arg-type]
    )
    return Artifact(
        id="qexec_q_revenue_by_region",
        type=ArtifactType.QUESTION_EXECUTION_RESULT,
        project_id="project_demo",
        session_id="run_demo",
        payload=result.model_dump(mode="json"),
    )


def _minimal_plan(profile_id: str) -> ReportPlanDraft:
    return ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Dataset Overview",
                id="rows",
                text="The dataset has 3 rows.",
                evidence=[
                    EvidenceRef(kind="stat", artifact_id=profile_id, locator="rows", value=3)
                ],
                referenced_datasets=["sales.csv"],
            )
        ],
    )


def _generate(tmp_path: Path, qexec: Artifact) -> Any:
    base = _base_artifacts(tmp_path)
    artifacts = [*base, _sql_result_artifact(), qexec]
    return (
        generate_agentic_report(
            artifacts,
            project_id="project_demo",
            session_id="run_demo",
            business_context="Revenue analysis",
            llm=FakeReportPlanLLM(_minimal_plan(base[0].id)),
        ),
        artifacts,
    )


def _analysis_claims(result: Any) -> list[Any]:
    section = next(
        section
        for section in result.bundle.sections
        if section.title == "Agent-Performed Analysis"
    )
    return section.claims


def test_validated_interpretation_becomes_a_cited_analysis_claim(tmp_path: Path) -> None:
    interpretation = (
        "East dominates this file: its total revenue is 40 while West records 0, "
        "so any regional push should start from what East is doing."
    )
    result, artifacts = _generate(
        tmp_path,
        _qexec_artifact(
            interpretation=interpretation, interpretation_status="validated"
        ),
    )

    claims = _analysis_claims(result)
    ids = [claim.id for claim in claims]
    assert "qintp_q_revenue_by_region" in ids
    claim = next(c for c in claims if c.id == "qintp_q_revenue_by_region")
    assert claim.text == interpretation
    # The interpretation cites the question's own evidence chain, so the hard
    # gate verified its figures.
    assert claim.evidence
    assert {ref.artifact_id for ref in claim.evidence} == {_SQL_RESULT_ID}
    # It sits with the question's finding claims, after them, not in a
    # parallel section.
    assert ids.index("qintp_q_revenue_by_region") > ids.index(
        "qfind_q_revenue_by_region_0"
    )
    markdown = report_bundle_to_markdown(result.bundle, artifacts=artifacts)
    assert "East dominates this file" in markdown


def test_interpretation_with_untraceable_number_is_pruned(tmp_path: Path) -> None:
    # Same shape as the positive control above, but 77 exists nowhere in the
    # evidence: the hard gate must prune the claim, not publish it. The prefix
    # is registered as platform-authored so a numeric mismatch never burns an
    # LLM rewrite round on text no rewrite can change.
    assert is_platform_authored_claim("qintp_q_revenue_by_region")
    result, _ = _generate(
        tmp_path,
        _qexec_artifact(
            interpretation="East region contributes 77 in total revenue.",
            interpretation_status="validated",
        ),
    )

    claims = _analysis_claims(result)
    ids = [claim.id for claim in claims]
    assert "qintp_q_revenue_by_region" not in ids
    # The deterministic finding claims are untouched by the prune.
    assert "qfind_q_revenue_by_region_0" in ids
    assert "qfind_q_revenue_by_region_1" in ids


@pytest.mark.parametrize(
    ("interpretation", "status"),
    [
        ("East dominates with a total revenue of 40.", "fallback"),
        ("", "absent"),
        ("   ", "validated"),
    ],
)
def test_unvalidated_or_empty_interpretation_is_not_injected(
    tmp_path: Path, interpretation: str, status: str
) -> None:
    result, _ = _generate(
        tmp_path,
        _qexec_artifact(interpretation=interpretation, interpretation_status=status),
    )
    all_ids = [
        claim.id for section in result.bundle.sections for claim in section.claims
    ]
    assert not any((claim_id or "").startswith("qintp_") for claim_id in all_ids)
