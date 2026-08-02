from __future__ import annotations

from eda_platform.core.ids import make_artifact_id
from eda_platform.drivers.question_exec import _agent_evidence_refs, _agent_qexec_artifact
from eda_platform.schemas.artifacts import Artifact, ArtifactType, SqlResult
from eda_platform.schemas.questions import (
    QuestionCandidate,
    QuestionExecutionResult,
    QuestionScore,
)
from eda_platform.schemas.reports import ReportBundle, ReportClaim
from eda_platform.tools.evidence import build_evidence_pack
from eda_platform.tools.report_validator import validate_report_bundle


def test_agent_sql_evidence_resolves_numbers_in_the_report_validator() -> None:
    """The agent path cites evidence the report validator has to read back.

    A locator the validator cannot resolve is not a soft failure: the SQL
    branch reports invalid_evidence_locator and the number pool comes back
    empty, so every figure in the answer stays unverified.
    """
    artifact = _sql_artifact()
    claim = ReportClaim(
        text="The north region placed 128 orders.",
        evidence=_agent_evidence_refs(artifact),
    )
    bundle = ReportBundle.empty(project_id="project_demo", session_id="question_run")
    bundle.sections[0].claims.append(claim)

    audit = validate_report_bundle(
        bundle,
        build_evidence_pack([artifact]),
        sql_results={artifact.id: SqlResult.model_validate(artifact.payload)},
    )

    assert [f for f in audit.findings if f.code == "invalid_evidence_locator"] == []
    assert claim.numeric_rollup == "number_verified"


def test_agent_profile_findings_carry_a_resolvable_locator() -> None:
    """Without a SQL artifact the persisted finding is the agent's own answer,
    and its evidence refs are the only trace the downstream number gate has."""
    profile = _profile_artifact()
    qexec = _agent_qexec_artifact(
        _candidate(),
        agent_result=_completed_run("The dataset holds 10 rows.", [profile]),
        project_id="project_demo",
        session_id="question_run",
        parent_ids=["candidate_set_1"],
    )
    finding = QuestionExecutionResult.model_validate(qexec.payload).findings[0]

    claim = ReportClaim(text=finding.text, evidence=list(finding.evidence))
    bundle = ReportBundle.empty(project_id="project_demo", session_id="question_run")
    bundle.sections[0].claims.append(claim)

    validate_report_bundle(bundle, build_evidence_pack([profile]))

    assert claim.numeric_rollup == "number_verified"


def _sql_artifact() -> Artifact:
    return Artifact(
        id=make_artifact_id("sql", {"test": "agent-evidence-locator"}),
        type=ArtifactType.SQL_RESULT,
        project_id="project_demo",
        session_id="question_run",
        payload=SqlResult(
            sql="SELECT region, COUNT(*) AS order_count FROM orders GROUP BY region",
            columns=["region", "order_count"],
            dtypes={"region": "VARCHAR", "order_count": "BIGINT"},
            units={"order_count": "count"},
            rows_preview=[{"region": "north", "order_count": 128}],
            row_count=1,
        ).model_dump(mode="json"),
    )


def _profile_artifact() -> Artifact:
    return Artifact(
        id=make_artifact_id("prof", {"test": "agent-evidence-locator"}),
        type=ArtifactType.DATASET_PROFILE,
        project_id="project_demo",
        session_id="question_run",
        payload={
            "dataset_id": "ds_orders",
            "name": "orders.csv",
            "rows": 10,
            "columns": 2,
            "column_names": ["region", "amount"],
            "dtypes": {"region": "object", "amount": "float64"},
            "missing_values": {"region": 0, "amount": 0},
            "missing_percent": {"region": 0.0, "amount": 0.0},
            "numeric_columns": ["amount"],
            "categorical_columns": ["region"],
        },
    )


def _candidate() -> QuestionCandidate:
    return QuestionCandidate(
        question_id="question_1",
        question_en="How large is the orders table?",
        origin="llm",
        target_datasets=["orders.csv"],
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.75,
        ),
    )


def _completed_run(answer: str, artifacts: list[Artifact]):
    from eda_platform.agents.runtime import AgentRunResult

    return AgentRunResult(
        status="completed",
        answer=answer,
        artifacts=artifacts,
        tool_calls=1,
        tool_names=["profile_dataset"],
    )
