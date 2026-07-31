"""Decision coverage endpoint: computed on read across all project sessions,
never persisted."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.investigations import ValidatedFinding
from eda_platform.schemas.questions import (
    OpportunityFeasibility,
    QuestionCandidate,
    QuestionCandidateSet,
    QuestionScore,
)

PROJECT = "proj_coverage"
RUN = "run_coverage"
EMPTY_PROJECT = "proj_empty"
EMPTY_RUN = "run_empty"


def _candidate(question_id: str, text: str, score: float) -> QuestionCandidate:
    return QuestionCandidate(
        question_id=question_id,
        question_en=text,
        origin="template",
        target_datasets=["orders"],
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=score,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=score,
        ),
        feasibility=OpportunityFeasibility(status="ready"),
    )


def _candidate_set_artifact(*candidates: QuestionCandidate) -> Artifact:
    candidate_set = QuestionCandidateSet(candidates=list(candidates))
    return Artifact(
        id="qcs_coverage",
        type=ArtifactType.QUESTION_CANDIDATE_SET,
        project_id=PROJECT,
        session_id=RUN,
        payload=candidate_set.model_dump(mode="json"),
    )


def _finding_artifact(question_id: str, question: str) -> Artifact:
    finding = ValidatedFinding(
        finding_id="finding_1",
        investigation_id="inv_1",
        question_id=question_id,
        question=question,
        claim_class="observed",
        evidence_support="high",
        analytical_reliability="high",
        decision_readiness="high",
        report_eligible=True,
        report_readiness="eligible",
        report_readiness_reason="Deterministic test verdict.",
    )
    return Artifact(
        id="vf_coverage",
        type=ArtifactType.VALIDATED_FINDING,
        project_id=PROJECT,
        session_id=RUN,
        payload=finding.model_dump(mode="json"),
    )


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Coverage")
    store.start_session(PROJECT, RUN)
    store.ensure_project(EMPTY_PROJECT, name="Empty")
    store.start_session(EMPTY_PROJECT, EMPTY_RUN)
    return store


@pytest.fixture
def client(store: ArtifactStore) -> TestClient:
    return TestClient(create_app(store.root))


def test_coverage_reports_open_questions_and_gaps(store: ArtifactStore) -> None:
    store.save_artifact(
        _candidate_set_artifact(
            _candidate("q1", "Which region grew fastest?", 0.9),
            _candidate("q2", "Why did churn spike?", 0.8),
        )
    )
    store.save_artifact(_finding_artifact("q1", "Which region grew fastest?"))
    client = TestClient(create_app(store.root))

    response = client.get(f"/api/v1/sessions/{RUN}/decision-coverage")
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["session_id"], body["project_id"]) == (RUN, PROJECT)
    assert body["top_cards_total"] == 2
    assert body["top_cards_terminal"] == 1
    assert body["validated_findings"] == 1
    assert body["findings_not_eligible"] == 0
    assert body["coverage_ready"] is False
    assert body["uninvestigated_high_value"] == ["Why did churn spike?"]
    assert any("not been investigated" in gap for gap in body["gaps"])


def test_coverage_ready_when_every_top_card_is_terminal(store: ArtifactStore) -> None:
    store.save_artifact(_candidate_set_artifact(_candidate("q1", "Which region grew?", 0.9)))
    store.save_artifact(_finding_artifact("q1", "Which region grew?"))
    client = TestClient(create_app(store.root))

    body = client.get(f"/api/v1/sessions/{RUN}/decision-coverage").json()
    assert body["coverage_ready"] is True
    assert body["uninvestigated_high_value"] == []
    assert body["gaps"] == []


def test_coverage_empty_project_is_200_with_zero_cards(client: TestClient) -> None:
    """No candidates yet is an empty state, not a 404: the client tells it from
    top_cards_total."""
    response = client.get(f"/api/v1/sessions/{EMPTY_RUN}/decision-coverage")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["top_cards_total"] == 0
    assert body["coverage_ready"] is False
    assert body["uninvestigated_high_value"] == []


def test_coverage_unknown_run_404(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/nope/decision-coverage")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_coverage_does_not_persist_an_artifact(store: ArtifactStore) -> None:
    store.save_artifact(_candidate_set_artifact(_candidate("q1", "Which region grew?", 0.9)))
    client = TestClient(create_app(store.root))
    client.get(f"/api/v1/sessions/{RUN}/decision-coverage")

    artifacts, _warnings = store.list_artifacts_safe(project_id=PROJECT, session_id=RUN)
    assert all(
        artifact.type is not ArtifactType.DECISION_COVERAGE for artifact in artifacts
    )
