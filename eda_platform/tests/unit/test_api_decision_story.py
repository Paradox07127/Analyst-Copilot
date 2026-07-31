"""Decision Story write slice: curate report-eligible
findings into a synthesis brief, then generate a decision report from a draft.

Both are queued as background jobs on their own derived runs (`sbsess_*`,
`drsess_*`) — the artifacts land where their drivers put them, never on the
lifecycle run. Neither path spends an LLM: the brief is deterministic and the
report is generated with `llm=None` (deterministic, no token spend).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.application.services.decision_report_service import (
    MAX_DECISION_REPORT_BYTES,
    generate_synthesis_session_id,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType, EvidenceRef
from eda_platform.schemas.investigations import (
    InvestigationRecord,
    ReliabilityRating,
    ValidatedFinding,
)
from eda_platform.schemas.questions import QuestionFinding
from eda_platform.schemas.synthesis import SynthesisBrief

_JOB_TIMEOUT_SECONDS = 180.0
PROJECT = "demo"
VIEW_RUN = "run_view"
FINDING_RUN = "run_findings"


def _finding(
    *,
    finding_id: str,
    question: str,
    reliability: ReliabilityRating = "high",
    eligible: bool = True,
) -> ValidatedFinding:
    return ValidatedFinding(
        finding_id=finding_id,
        investigation_id=f"inv_{finding_id}",
        question_id=f"q_{finding_id}",
        question=question,
        decision_action="Rebalance channel spend once labels are reviewed.",
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
        report_eligible=eligible,
        report_readiness="eligible_with_limitations" if eligible else "not_eligible",
        report_readiness_reason="Validated with disclosed data conditions.",
    )


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    for session_id in (VIEW_RUN, FINDING_RUN):
        store.start_session(PROJECT, session_id)
    for index, (finding_id, question, eligible) in enumerate(
        (
            ("finding_orders", "How do order values vary by channel?", True),
            ("finding_returns", "What share of orders are returned?", True),
            ("finding_draft", "Is the pipeline healthy?", False),
        ),
        start=1,
    ):
        store.save_artifact(
            Artifact(
                id=f"vf_{index}",
                type=ArtifactType.VALIDATED_FINDING,
                project_id=PROJECT,
                session_id=FINDING_RUN,
                payload=_finding(
                    finding_id=finding_id, question=question, eligible=eligible
                ).model_dump(mode="json"),
            )
        )
        # A ValidatedFinding only enters the library when a same-run validated
        # InvestigationRecord vouches for it with matching provenance.
        store.save_artifact(
            Artifact(
                id=f"record_{finding_id}",
                type=ArtifactType.INVESTIGATION_RECORD,
                project_id=PROJECT,
                session_id=FINDING_RUN,
                payload=InvestigationRecord(
                    record_id=f"record_{finding_id}",
                    investigation_id=f"inv_{finding_id}",
                    question_id=f"q_{finding_id}",
                    status="validated",
                    reason_code="validated_for_test",
                    reason="Terminal outcome for the test.",
                    next_action="Review the evidence.",
                    finding_artifact_id=f"vf_{index}",
                ).model_dump(mode="json"),
            )
        )
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
    store.mark_session_status(PROJECT, FINDING_RUN, "completed")
    store.mark_session_status(PROJECT, VIEW_RUN, "completed")
    return tmp_path


@pytest.fixture()
def client(workspace: Path) -> TestClient:
    return TestClient(create_app(workspace))


def _wait_terminal(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + _JOB_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/jobs/{job_id}").json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body
        time.sleep(0.2)
    raise AssertionError(f"Job {job_id} did not reach a terminal status in time.")


def _story(client: TestClient, session_id: str = VIEW_RUN):
    return client.get(f"/api/v1/sessions/{session_id}/decision-story")


def _create_draft(
    client: TestClient,
    finding_ids: list[str],
    *,
    business_context: str = "",
    finding_session_ids: dict[str, str] | None = None,
    idempotency_key: str | None = None,
    session_id: str = VIEW_RUN,
):
    headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
    return client.post(
        f"/api/v1/sessions/{session_id}/decision-story/drafts",
        json={
            "finding_artifact_ids": finding_ids,
            "finding_session_ids": finding_session_ids or {},
            "business_context": business_context,
        },
        headers=headers,
    )


def _generate_report(
    client: TestClient,
    brief_artifact_id: str,
    *,
    brief_session_id: str | None = None,
    idempotency_key: str | None = None,
    session_id: str = VIEW_RUN,
):
    headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
    return client.post(
        f"/api/v1/sessions/{session_id}/decision-report/generate",
        json={
            "brief_artifact_id": brief_artifact_id,
            "brief_session_id": brief_session_id,
        },
        headers=headers,
    )


def _drafted(client: TestClient, finding_ids: list[str]) -> str:
    """Run one brief-creation job to completion and return its artifact id."""
    response = _create_draft(client, finding_ids, idempotency_key=uuid.uuid4().hex)
    assert response.status_code == 201, response.text
    final = _wait_terminal(client, response.json()["job"]["job_id"])
    assert final["status"] == "completed", final
    drafts = _story(client).json()["drafts"]
    assert len(drafts) == 1, drafts
    return drafts[0]["artifact_id"]


def _seed_invalid_report(workspace: Path, artifact_id: str = "decision_bad") -> Path:
    store = ArtifactStore(workspace)
    session_id = "synthesis_bad"
    store.start_session(PROJECT, session_id)
    store.save_artifact(
        Artifact(
            id=artifact_id,
            type=ArtifactType.DECISION_REPORT,
            project_id=PROJECT,
            session_id=session_id,
            payload={},
        )
    )
    return store.artifact_path(PROJECT, session_id, artifact_id)


def test_story_lists_eligible_findings_and_no_drafts_yet(client: TestClient) -> None:
    body = _story(client).json()
    # The library orders findings newest-first; vf_3 is not report-eligible.
    assert [item["artifact_id"] for item in body["eligible_findings"]] == ["vf_2", "vf_1"]
    assert body["drafts"] == []
    assert body["project_id"] == PROJECT
    # Every offered finding carries the freshness the curation control gates on.
    assert all(item["freshness"] for item in body["eligible_findings"])


def test_story_unknown_run_is_404(client: TestClient) -> None:
    response = _story(client, session_id="run_missing")
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "session_not_found"


def test_create_draft_runs_a_job_and_persists_the_brief(client: TestClient) -> None:
    """End-to-end: the POST queues a job, the worker runs the real driver, and
    the brief becomes readable through the story endpoint."""
    response = _create_draft(client, ["vf_1", "vf_2"], business_context="Q3 review.")
    assert response.status_code == 201, response.text
    started = response.json()
    assert started["execution_session_id"].startswith("sbsess_")
    assert started["job"]["session_id"] == started["execution_session_id"]

    final = _wait_terminal(client, started["job"]["job_id"])
    assert final["status"] == "completed", final
    assert final["kind"] == "synthesis_brief_create"

    drafts = _story(client).json()["drafts"]
    assert len(drafts) == 1, drafts
    assert drafts[0]["selected_finding_artifact_ids"] == ["vf_1", "vf_2"]
    assert drafts[0]["headline"].strip()
    assert drafts[0]["storyline"], drafts[0]
    assert drafts[0]["business_context"] == "Q3 review."
    # The brief lands on the driver's own synthesis run, not the lifecycle run.
    assert not drafts[0]["session_id"].startswith("sbsess_")


def test_create_draft_rejects_an_ineligible_finding(client: TestClient) -> None:
    response = _create_draft(client, ["vf_1", "vf_3"])
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "decision_story_not_draftable"
    assert "vf_3" in response.json()["error"]["message"]


def test_create_draft_rejects_an_unknown_finding(client: TestClient) -> None:
    response = _create_draft(client, ["vf_absent"])
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "decision_story_not_draftable"


def test_create_draft_requires_exact_run_for_duplicate_finding_id(
    client: TestClient,
    workspace: Path,
) -> None:
    store = ArtifactStore(workspace)
    duplicate_run = "run_duplicate_finding"
    store.start_session(PROJECT, duplicate_run)
    duplicate = _finding(
        finding_id="finding_duplicate",
        question="This is the wrong duplicate run.",
    )
    store.save_artifact(
        Artifact(
            id="vf_1",
            type=ArtifactType.VALIDATED_FINDING,
            project_id=PROJECT,
            session_id=duplicate_run,
            payload=duplicate.model_dump(mode="json"),
        )
    )
    store.save_artifact(
        Artifact(
            id="record_duplicate",
            type=ArtifactType.INVESTIGATION_RECORD,
            project_id=PROJECT,
            session_id=duplicate_run,
            payload=InvestigationRecord(
                record_id="record_duplicate",
                investigation_id=duplicate.investigation_id,
                question_id=duplicate.question_id,
                status="validated",
                reason_code="validated_for_test",
                reason="Terminal duplicate outcome.",
                next_action="Review the evidence.",
                finding_artifact_id="vf_1",
            ).model_dump(mode="json"),
        )
    )

    ambiguous = _create_draft(client, ["vf_1"])
    assert ambiguous.status_code == 422, ambiguous.text

    exact = _create_draft(
        client,
        ["vf_1"],
        finding_session_ids={"vf_1": FINDING_RUN},
    )
    assert exact.status_code == 201, exact.text
    final = _wait_terminal(client, exact.json()["job"]["job_id"])
    assert final["status"] == "completed", final
    draft = _story(client).json()["drafts"][0]
    artifact = store.get_artifact(
        draft["artifact_id"],
        project_id=PROJECT,
        session_id=draft["session_id"],
    )
    brief = SynthesisBrief.model_validate(artifact.payload)
    assert brief.selected_finding_session_ids == {"vf_1": FINDING_RUN}


@pytest.mark.parametrize(
    "payload",
    [
        {"finding_artifact_ids": []},
        {"business_context": "no ids"},
        {"finding_artifact_ids": [""]},
    ],
)
def test_create_draft_invalid_body_is_422(client: TestClient, payload: dict) -> None:
    response = client.post(
        f"/api/v1/sessions/{VIEW_RUN}/decision-story/drafts", json=payload
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "validation_error"


def test_create_draft_unknown_run_is_404(client: TestClient) -> None:
    response = _create_draft(client, ["vf_1"], session_id="run_missing")
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "session_not_found"


def test_create_draft_idempotency_key_replays_the_same_job(client: TestClient) -> None:
    key = uuid.uuid4().hex
    first = _create_draft(client, ["vf_1"], idempotency_key=key)
    assert first.status_code == 201, first.text
    replay = _create_draft(client, ["vf_1"], idempotency_key=key)
    assert replay.status_code == 201, replay.text
    assert replay.json()["job"]["job_id"] == first.json()["job"]["job_id"]
    assert replay.json()["execution_session_id"] == first.json()["execution_session_id"]


def test_generate_report_runs_a_job_and_publishes_the_report(
    client: TestClient,
) -> None:
    """End-to-end: a draft becomes a real DecisionReport artifact the existing
    read endpoint serves."""
    brief_artifact_id = _drafted(client, ["vf_1", "vf_2"])
    assert client.get(f"/api/v1/sessions/{VIEW_RUN}/decision-report").json()["status"] == "none"

    response = _generate_report(client, brief_artifact_id)
    assert response.status_code == 201, response.text
    started = response.json()
    assert started["execution_session_id"].startswith("drsess_")
    assert started["brief_artifact_id"] == brief_artifact_id

    final = _wait_terminal(client, started["job"]["job_id"])
    assert final["status"] == "completed", final
    assert final["kind"] == "decision_report_generate"

    report = client.get(f"/api/v1/sessions/{VIEW_RUN}/decision-report").json()
    assert report["status"] == "available"
    assert report["brief_id"]
    assert report["title"].strip()
    assert report["scqa"]["situation"].strip()
    assert report["source_finding_artifact_ids"] == ["vf_1", "vf_2"]
    # llm=None on the driver call: the report must stay deterministic.
    assert report["narrative_status"] != "llm_refined"


def test_persisted_decision_report_read_failures_are_typed(
    client: TestClient,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _seed_invalid_report(workspace)
    endpoint = f"/api/v1/sessions/{VIEW_RUN}/decision-report"

    path.unlink()
    missing = client.get(endpoint)
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "decision_report_missing"

    _seed_invalid_report(workspace)
    path.write_text("{broken", encoding="utf-8")
    corrupt = client.get(endpoint)
    assert corrupt.status_code == 500
    assert corrupt.json()["error"]["code"] == "decision_report_corrupt"

    _seed_invalid_report(workspace)
    invalid_schema = client.get(endpoint)
    assert invalid_schema.status_code == 500
    assert invalid_schema.json()["error"]["code"] == "decision_report_corrupt"

    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["id"] = "decision_other"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    invalid = client.get(endpoint)
    assert invalid.status_code == 500
    assert invalid.json()["error"]["code"] == "decision_report_identity_invalid"

    path.write_text("x" * (MAX_DECISION_REPORT_BYTES + 1), encoding="utf-8")
    too_large = client.get(endpoint)
    assert too_large.status_code == 500
    assert too_large.json()["error"]["code"] == "decision_report_too_large"

    _seed_invalid_report(workspace)
    original_read_text = Path.read_text

    def unavailable(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if self == path:
            raise PermissionError("private filesystem detail")
        return original_read_text(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", unavailable)
    retryable = client.get(endpoint)
    assert retryable.status_code == 503
    assert retryable.headers["retry-after"] == "1"
    assert retryable.json()["error"]["code"] == "decision_report_unavailable"

    for response in (missing, corrupt, invalid_schema, invalid, too_large, retryable):
        assert str(workspace) not in response.text
        assert "private filesystem detail" not in response.text

    monkeypatch.setattr(Path, "read_text", original_read_text)
    _seed_invalid_report(workspace)
    with sqlite3.connect(ArtifactStore(workspace).db_path) as conn:
        conn.execute(
            """
            update artifacts set artifact_id = ?
            where project_id = ? and artifact_type = ?
            """,
            (str(workspace / "private-report-id"), PROJECT, "DecisionReport"),
        )
    tampered_identifier = client.get(endpoint)
    assert tampered_identifier.status_code == 500
    assert (
        tampered_identifier.json()["error"]["code"]
        == "decision_report_identity_invalid"
    )
    assert str(workspace) not in tampered_identifier.text


def test_openapi_declares_decision_report_read_failures(client: TestClient) -> None:
    operation = client.get("/openapi.json").json()["paths"][
        "/api/v1/sessions/{session_id}/decision-report"
    ]["get"]
    assert {"404", "500", "503"} <= operation["responses"].keys()


def test_generate_report_unknown_brief_is_404(client: TestClient) -> None:
    response = _generate_report(client, "brief_absent")
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "decision_story_draft_not_found"


def test_generate_report_requires_exact_run_for_duplicate_brief_id(
    client: TestClient,
    workspace: Path,
) -> None:
    brief_artifact_id = _drafted(client, ["vf_1"])
    draft = _story(client).json()["drafts"][0]
    store = ArtifactStore(workspace)
    original = store.get_artifact(
        brief_artifact_id,
        project_id=PROJECT,
        session_id=draft["session_id"],
    )
    duplicate_run = "synthesis_duplicate"
    store.start_session(PROJECT, duplicate_run)
    store.save_artifact(
        original.model_copy(update={"session_id": duplicate_run})
    )

    ambiguous = _generate_report(client, brief_artifact_id)
    assert ambiguous.status_code == 404, ambiguous.text

    exact = _generate_report(
        client,
        brief_artifact_id,
        brief_session_id=draft["session_id"],
    )
    assert exact.status_code == 201, exact.text


def test_generate_report_rejects_a_non_brief_artifact(client: TestClient) -> None:
    response = _generate_report(client, "vf_1")
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "decision_story_draft_not_found"


def test_generate_report_invalid_body_is_422(client: TestClient) -> None:
    response = client.post(
        f"/api/v1/sessions/{VIEW_RUN}/decision-report/generate", json={"brief_artifact_id": ""}
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "validation_error"


def test_generate_report_idempotency_key_replays_the_same_job(
    client: TestClient,
) -> None:
    brief_artifact_id = _drafted(client, ["vf_1"])
    key = uuid.uuid4().hex
    first = _generate_report(client, brief_artifact_id, idempotency_key=key)
    assert first.status_code == 201, first.text
    replay = _generate_report(client, brief_artifact_id, idempotency_key=key)
    assert replay.status_code == 201, replay.text
    assert replay.json()["job"]["job_id"] == first.json()["job"]["job_id"]


def test_a_second_draft_is_refused_while_one_is_active(
    client: TestClient, workspace: Path
) -> None:
    """Two decision-story jobs would race on the same project-level story, so a
    second submission without an idempotency key is refused, not queued."""
    store = ArtifactStore(workspace)
    job = store.create_job(
        job_id=f"job_{uuid.uuid4().hex[:12]}",
        session_id=generate_synthesis_session_id(VIEW_RUN),
        project_id=PROJECT,
        kind="synthesis_brief_create",
        idempotency_key=None,
    )
    try:
        response = _create_draft(client, ["vf_1"])
        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == "decision_story_busy"
    finally:
        store.mark_job_status(str(job["job_id"]), "cancelled")
    # Cleared once it settles.
    assert _create_draft(client, ["vf_1"]).status_code == 201


def test_derived_lifecycle_runs_stay_out_of_the_run_list(client: TestClient) -> None:
    """`sbsess_`/`drsess_` are registered derived prefixes, so the runs they mint
    never bury the analyses a user started."""
    brief_artifact_id = _drafted(client, ["vf_1", "vf_2"])
    response = _generate_report(client, brief_artifact_id)
    assert response.status_code == 201, response.text
    assert _wait_terminal(client, response.json()["job"]["job_id"])["status"] == "completed"

    listed = [
        item["session_id"]
        for item in client.get(f"/api/v1/projects/{PROJECT}/sessions").json()["items"]
    ]
    assert not [session_id for session_id in listed if session_id.startswith(("sbsess_", "drsess_"))]
    assert VIEW_RUN in listed

    # They are still real, deep-linkable runs when explicitly asked for.
    including = [
        item["session_id"]
        for item in client.get(
            f"/api/v1/projects/{PROJECT}/sessions", params={"include_derived": True}
        ).json()["items"]
    ]
    assert [session_id for session_id in including if session_id.startswith("sbsess_")]
    assert [session_id for session_id in including if session_id.startswith("drsess_")]
