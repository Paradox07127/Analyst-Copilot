"""Decision report endpoint (§10.3): project-scoped lookup, SCQA/candidate
decision/evidence shaping, gate verdict, and the empty-project 200."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.core.ids import INTERNAL_SESSION_MARKER
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.decision_report import create_decision_report
from eda_platform.schemas.artifacts import Artifact, ArtifactType, EvidenceRef
from eda_platform.schemas.investigations import ReliabilityRating, ValidatedFinding
from eda_platform.schemas.questions import QuestionFinding
from eda_platform.schemas.synthesis import SynthesisBrief, SynthesisStoryBeat

PROJECT = "demo"
VIEW_RUN = "run_view"
FINDING_RUN = "run_findings"
SYNTHESIS_RUN = "synthesis_run"


def _finding(
    *, finding_id: str, question: str, action: str, reliability: ReliabilityRating
) -> ValidatedFinding:
    return ValidatedFinding(
        finding_id=finding_id,
        investigation_id=f"inv_{finding_id}",
        question_id=f"q_{finding_id}",
        question=question,
        decision_action=action,
        claim_class="observed",
        findings=[
            QuestionFinding(
                text="The observed average order value is 125.5.",
                evidence=[
                    EvidenceRef(
                        kind="table",
                        artifact_id=f"table_{finding_id}",
                        locator="rows[0].average_order_value",
                        value=125.5,
                    )
                ],
            )
        ],
        evidence_support="high",
        analytical_reliability=reliability,
        decision_readiness="medium",
        limitations=["Channel labels require review."],
        report_eligible=True,
        report_readiness="eligible_with_limitations",
        report_readiness_reason="Validated with disclosed data conditions.",
    )


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    for session_id in (VIEW_RUN, FINDING_RUN, SYNTHESIS_RUN):
        store.start_session(PROJECT, session_id)
    return store


def _seed_report(store: ArtifactStore) -> str:
    """Build a real decision report through the driver, so lineage and the
    publication fingerprint are the ones the platform actually writes."""
    findings = [
        _finding(
            finding_id="finding_orders",
            question="How do order values vary by channel?",
            action="Rebalance channel spend once labels are reviewed.",
            reliability="high",
        ),
        _finding(
            finding_id="finding_returns",
            question="What share of orders are returned?",
            action="",
            reliability="medium",
        ),
    ]
    finding_ids: list[str] = []
    for index, finding in enumerate(findings, start=1):
        artifact = Artifact(
            id=f"vf_{index}",
            type=ArtifactType.VALIDATED_FINDING,
            project_id=PROJECT,
            session_id=FINDING_RUN,
            payload=finding.model_dump(mode="json"),
        )
        store.save_artifact(artifact)
        finding_ids.append(artifact.id)
    # The evidence the findings point at must resolve to a navigable run.
    store.save_artifact(
        Artifact(
            id="table_finding_orders",
            type=ArtifactType.TABLE,
            project_id=PROJECT,
            session_id=FINDING_RUN,
            payload={
                "dataset_id": "ds_1",
                "title": "Order values",
                "kind": "numeric_summary",
                "description": "Averages per channel.",
                "rows": [{"average_order_value": 125.5}],
            },
        )
    )
    brief = SynthesisBrief(
        brief_id="brief_1",
        project_id=PROJECT,
        selected_finding_artifact_ids=finding_ids,
        decision_context="Which evidence should guide the channel decision?",
        headline="The observed average order value is 125.5.",
        storyline=[
            SynthesisStoryBeat(
                title="Evidence",
                body="Validated findings are ready for review.",
                finding_artifact_ids=finding_ids,
            )
        ],
        limitations=["Channel labels require review."],
        investigation_gaps=["Validate return patterns in later periods."],
        report_eligible=True,
        report_readiness="eligible_with_limitations",
    )
    store.save_artifact(
        Artifact(
            id="sbrief_1",
            type=ArtifactType.SYNTHESIS_BRIEF,
            project_id=PROJECT,
            session_id=SYNTHESIS_RUN,
            parents=finding_ids,
            payload=brief.model_dump(mode="json"),
        )
    )
    return create_decision_report(store, project_id=PROJECT, brief_artifact_id="sbrief_1")


def test_decision_report_absent_is_a_normal_200(store: ArtifactStore) -> None:
    response = TestClient(create_app(store.root)).get(
        f"/api/v1/sessions/{VIEW_RUN}/decision-report"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "none"
    assert body["session_id"] == VIEW_RUN
    assert body["sections"] == []
    assert body["title"] is None


def test_decision_report_is_found_from_another_run_in_the_project(
    store: ArtifactStore,
) -> None:
    """The report lives under its synthesis run; the Report page of an analysis
    run in the same project must still surface it."""
    report_artifact_id = _seed_report(store)
    body = (
        TestClient(create_app(store.root))
        .get(f"/api/v1/sessions/{VIEW_RUN}/decision-report")
        .json()
    )
    assert body["status"] == "available"
    assert body["artifact_id"] == report_artifact_id
    assert body["report_session_id"] == SYNTHESIS_RUN
    assert body["brief_id"] == "brief_1"
    assert body["title"]
    assert set(body["scqa"]) == {"situation", "complication", "question", "answer"}
    assert all(body["scqa"][key].strip() for key in body["scqa"])
    assert [section["title"] for section in body["sections"]]
    assert body["limitations"] == ["Channel labels require review."]
    assert body["investigation_gaps"] == ["Validate return patterns in later periods."]
    assert body["source_finding_artifact_ids"] == ["vf_1", "vf_2"]
    assert body["publication_status"] == "published"
    assert body["report_readiness"] == "eligible_with_limitations"
    assert body["narrative_status"] == "deterministic"


def test_candidate_decisions_and_evidence_come_from_source_findings(
    store: ArtifactStore,
) -> None:
    _seed_report(store)
    body = (
        TestClient(create_app(store.root))
        .get(f"/api/v1/sessions/{VIEW_RUN}/decision-report")
        .json()
    )
    # Only findings that proposed an action become candidate decisions.
    assert body["candidate_decisions"] == [
        {
            "finding_artifact_id": "vf_1",
            "question": "How do order values vary by channel?",
            "decision_action": "Rebalance channel spend once labels are reviewed.",
            "decision_readiness": "medium",
            "analytical_reliability": "high",
            "report_readiness": "eligible_with_limitations",
        }
    ]
    evidence = {ref["artifact_id"]: ref for ref in body["evidence_refs"]}
    assert evidence["table_finding_orders"]["kind"] == "table"
    assert evidence["table_finding_orders"]["session_id"] == FINDING_RUN
    # An evidence id with no artifact behind it is listed but not navigable.
    assert evidence["table_finding_returns"]["session_id"] is None
    # Weakest analytical_reliability across the source findings wins.
    assert body["confidence_label"] == "medium"


def test_gate_verdict_takes_the_worst_audit_of_the_viewed_run(
    store: ArtifactStore,
) -> None:
    _seed_report(store)
    for index, verdict in enumerate(("pass", "degraded"), start=1):
        store.save_artifact(
            Artifact(
                id=f"audit_{index}",
                type=ArtifactType.REPORT_AUDIT,
                project_id=PROJECT,
                session_id=VIEW_RUN,
                payload={"status": "validated", "gate_verdict": verdict},
            )
        )
    body = (
        TestClient(create_app(store.root))
        .get(f"/api/v1/sessions/{VIEW_RUN}/decision-report")
        .json()
    )
    assert body["gate_verdict"] == "degraded"


def test_gate_verdict_falls_back_to_the_bundle_audit(store: ArtifactStore) -> None:
    _seed_report(store)
    store.save_artifact(
        Artifact(
            id="bundle_1",
            type=ArtifactType.REPORT_BUNDLE,
            project_id=PROJECT,
            session_id=VIEW_RUN,
            payload={"status": "validated", "audit": {"gate_verdict": "rejected"}},
        )
    )
    body = (
        TestClient(create_app(store.root))
        .get(f"/api/v1/sessions/{VIEW_RUN}/decision-report")
        .json()
    )
    assert body["gate_verdict"] == "rejected"


def test_export_is_gated_on_freshness(store: ArtifactStore) -> None:
    """Fail-closed, like the legacy UI: these findings have no InvestigationPlan
    to verify their source datasets against, so freshness is unverifiable and
    export stays off."""
    _seed_report(store)
    body = (
        TestClient(create_app(store.root))
        .get(f"/api/v1/sessions/{VIEW_RUN}/decision-report")
        .json()
    )
    assert body["freshness"]["status"] == "unverifiable"
    assert body["freshness"]["reasons"]
    assert body["export_available"] is False


def test_newest_decision_report_wins(store: ArtifactStore) -> None:
    first = _seed_report(store)
    newer = Artifact(
        id="dreport_newer",
        type=ArtifactType.DECISION_REPORT,
        project_id=PROJECT,
        session_id=SYNTHESIS_RUN,
        payload={
            **store.get_artifact(first).payload,
            "report_id": "dreport_newer",
            "title": "Newer decision story",
        },
    )
    store.save_artifact(newer)
    body = (
        TestClient(create_app(store.root))
        .get(f"/api/v1/sessions/{VIEW_RUN}/decision-report")
        .json()
    )
    assert body["artifact_id"] == "dreport_newer"
    assert body["title"] == "Newer decision story"


def test_internal_run_reports_are_ignored(store: ArtifactStore) -> None:
    internal_run = f"{SYNTHESIS_RUN}{INTERNAL_SESSION_MARKER}_probe"
    store.start_session(PROJECT, internal_run)
    payload = store.get_artifact(_seed_report(store)).payload
    store.save_artifact(
        Artifact(
            id="dreport_internal",
            type=ArtifactType.DECISION_REPORT,
            project_id=PROJECT,
            session_id=internal_run,
            payload={**payload, "title": "Internal decision story"},
        )
    )
    body = (
        TestClient(create_app(store.root))
        .get(f"/api/v1/sessions/{VIEW_RUN}/decision-report")
        .json()
    )
    assert body["title"] != "Internal decision story"


def test_decision_report_unknown_run_404(store: ArtifactStore) -> None:
    response = TestClient(create_app(store.root)).get("/api/v1/sessions/missing/decision-report")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_decision_report_response_carries_no_server_paths(store: ArtifactStore) -> None:
    _seed_report(store)
    response = TestClient(create_app(store.root)).get(
        f"/api/v1/sessions/{VIEW_RUN}/decision-report"
    )
    assert str(store.root) not in response.text
