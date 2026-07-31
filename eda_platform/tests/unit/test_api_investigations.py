"""Investigation governance vertical slice.

Covers the whole Questions-page loop over HTTP: edit a card, draft one
from free text, build plans, approve or reject, execute the approved set, and
run the Ultra macro loop. Jobs run in spawned workers exactly as in production
(same pattern as test_api_questions); the macro loop's multi-round behaviour is
driven through the worker entry point with a stubbed follow-up model, because an
offline client concludes at round 1 by design.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from job_test_helpers import run_claimed_job

import eda_platform.worker.runner as worker_runner
from eda_platform.api.main import create_app
from eda_platform.application.services.approval_service import ApprovalService
from eda_platform.application.services.investigation_service import (
    InvestigationService,
    execution_lane,
)
from eda_platform.core.ids import make_artifact_id
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.investigations import InvestigationApproval
from eda_platform.schemas.loop import LoopLedger

_JOB_TIMEOUT_SECONDS = 300.0


def _seed_csv() -> str:
    """Same deterministic date/category/numeric mix as the questions slice, so
    offline discovery yields a group-comparison candidate that can be planned."""
    rows = ["order_date,region,amount,quantity"]
    regions = ["north", "south", "east", "west"]
    for index in range(120):
        day = 1 + (index % 28)
        month = 1 + (index // 28) % 6
        amount = round(50 + (index % 4) * 40 + (index * 7 % 25), 2)
        rows.append(
            f"2024-{month:02d}-{day:02d},{regions[index % 4]},{amount},{1 + index % 5}"
        )
    return "\n".join(rows) + "\n"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("investigations_api")
    store = ArtifactStore(root)
    store.ensure_project("demo", name="Demo")
    seed = root / "seed" / "orders.csv"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text(_seed_csv(), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def app(workspace: Path) -> FastAPI:
    return create_app(workspace)


@pytest.fixture(scope="module")
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _wait_terminal(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + _JOB_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/jobs/{job_id}").json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body
        time.sleep(0.2)
    raise AssertionError(f"Job {job_id} did not reach a terminal status in time.")


def _analysed_run(client: TestClient, session_id: str) -> str:
    response = client.post(
        f"/api/v1/sessions/{session_id}/jobs",
        json={
            "kind": "auto_eda",
            "project_id": "demo",
            "datasets": ["seed/orders.csv"],
            "llm": "offline",
            "generate_report": False,
        },
    )
    assert response.status_code == 201, response.text
    final = _wait_terminal(client, response.json()["job_id"])
    assert final["status"] == "completed", final
    return session_id


@pytest.fixture(scope="module")
def source_run(client: TestClient) -> str:
    return _analysed_run(client, "run_inv_src")


def _questions(client: TestClient, session_id: str) -> list[dict]:
    response = client.get(f"/api/v1/sessions/{session_id}/questions")
    assert response.status_code == 200, response.text
    return response.json()["questions"]


def _investigations(client: TestClient, session_id: str) -> dict:
    response = client.get(f"/api/v1/sessions/{session_id}/investigations")
    assert response.status_code == 200, response.text
    return response.json()


def _build_plans(client: TestClient, session_id: str, question_ids: list[str]) -> dict:
    response = client.post(
        f"/api/v1/sessions/{session_id}/investigations/plan",
        json={"question_ids": question_ids},
        headers={"Idempotency-Key": uuid.uuid4().hex},
    )
    assert response.status_code == 201, response.text
    final = _wait_terminal(client, response.json()["job"]["job_id"])
    assert final["status"] == "completed", final
    return response.json()


def _planned_session_id(store: ArtifactStore, lifecycle_session_id: str) -> str:
    """The plan run the build job minted, read from its own trace event."""
    for event in store.list_trace_events(project_id="demo", session_id=lifecycle_session_id):
        if event.event_type == "investigation.planned":
            return str(event.summary["plan_session_id"])
    raise AssertionError(f"no investigation.planned event on {lifecycle_session_id}")


def _prepare_decision(
    client: TestClient, session_id: str, plan_id: str, decision: str, reason: str = "test"
) -> dict:
    response = client.post(
        f"/api/v1/sessions/{session_id}/investigations/{plan_id}/prepare-decision",
        json={"decision": decision, "reason": reason},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _decide(client: TestClient, session_id: str, plan_id: str, decision: str, prepared: dict):
    verb = "approve" if decision == "approved" else "reject"
    return client.post(
        f"/api/v1/sessions/{session_id}/investigations/{plan_id}/{verb}",
        json={
            "action_hash": prepared["action_hash"],
            "approval_token": prepared["approval_token"],
        },
    )


def _plan_for(client: TestClient, session_id: str, plan_id: str) -> dict:
    return next(
        item for item in _investigations(client, session_id)["plans"] if item["plan_id"] == plan_id
    )


# --------------------------------------------------------------- card editing


def test_edit_card_bumps_version_and_persists_text(
    client: TestClient, source_run: str
) -> None:
    target = next(item for item in _questions(client, source_run) if item["executable"])
    before = target["card_version"]
    response = client.patch(
        f"/api/v1/sessions/{source_run}/questions/{target['question_id']}",
        json={
            "expected_version": before,
            "business_decision": "Decide how to split the regional budget.",
            "risks": ["Small per-region sample", "Observational only"],
        },
    )
    assert response.status_code == 200, response.text
    edited = response.json()
    assert edited["card_version"] == before + 1
    assert edited["business_decision"] == "Decide how to split the regional budget."
    assert edited["risks"] == ["Small per-region sample", "Observational only"]
    # The bump survives a fresh read of the candidate set, not just the response.
    listed = next(
        item
        for item in _questions(client, source_run)
        if item["question_id"] == target["question_id"]
    )
    assert listed["card_version"] == before + 1
    assert listed["risks"] == ["Small per-region sample", "Observational only"]


def test_edit_card_rejects_a_stale_writer(
    client: TestClient, source_run: str
) -> None:
    target = next(item for item in _questions(client, source_run) if item["executable"])
    version = target["card_version"]
    path = f"/api/v1/sessions/{source_run}/questions/{target['question_id']}"
    first = client.patch(
        path,
        json={"expected_version": version, "business_decision": "First writer"},
    )
    stale = client.patch(
        path,
        json={"expected_version": version, "business_decision": "Stale writer"},
    )

    assert first.status_code == 200
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "question_version_conflict"
    current = next(
        item
        for item in _questions(client, source_run)
        if item["question_id"] == target["question_id"]
    )
    assert current["business_decision"] == "First writer"


def test_edit_card_refuses_execution_defining_fields(
    client: TestClient, source_run: str
) -> None:
    """sql_template is not an editable field, so a body carrying only it edits
    nothing and is refused rather than silently ignored."""
    target = _questions(client, source_run)[0]
    response = client.patch(
        f"/api/v1/sessions/{source_run}/questions/{target['question_id']}",
        json={"expected_version": target["card_version"], "sql_template": "select 1"},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "question_invalid"


def test_edit_unknown_question_is_404(client: TestClient, source_run: str) -> None:
    response = client.patch(
        f"/api/v1/sessions/{source_run}/questions/q_nope",
        json={"expected_version": 1, "business_decision": "x"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "question_not_found"


# ------------------------------------------------------------- card drafting


def test_draft_card_from_free_text_appends_a_reviewable_card(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    question = "Which region has the highest average order amount?"
    before = len(_questions(client, source_run))
    prepared = client.post(
        f"/api/v1/sessions/{source_run}/questions/prepare-draft",
        json={"question": question, "llm": "offline"},
    )
    assert prepared.status_code == 200, prepared.text
    body = prepared.json()
    assert body["question"] == question
    assert len(body["action_hash"]) == 64

    started = client.post(
        f"/api/v1/sessions/{source_run}/questions",
        json={
            "action_hash": body["action_hash"],
            "approval_token": body["approval_token"],
        },
        headers={"Idempotency-Key": uuid.uuid4().hex},
    )
    assert started.status_code == 201, started.text
    assert started.json()["execution_session_id"].startswith("qdsess_")
    final = _wait_terminal(client, started.json()["job"]["job_id"])
    assert final["status"] == "completed", final
    assert final["kind"] == "question_draft"

    questions = _questions(client, source_run)
    assert len(questions) == before + 1
    drafted = next(item for item in questions if item["question"] == question)
    # Offline drafting is honest about what it did NOT do: no invented score.
    assert drafted["priority"] == 0.0
    assert drafted["origin"] == "llm"
    assert "without a model" in drafted["priority_rationale"]


def test_draft_replayed_token_is_409_consumed(
    client: TestClient, source_run: str
) -> None:
    prepared = client.post(
        f"/api/v1/sessions/{source_run}/questions/prepare-draft",
        json={"question": "Does order quantity track amount?", "llm": "offline"},
    ).json()
    first = client.post(
        f"/api/v1/sessions/{source_run}/questions",
        json={
            "action_hash": prepared["action_hash"],
            "approval_token": prepared["approval_token"],
        },
        headers={"Idempotency-Key": uuid.uuid4().hex},
    )
    assert first.status_code == 201, first.text
    _wait_terminal(client, first.json()["job"]["job_id"])

    replay = client.post(
        f"/api/v1/sessions/{source_run}/questions",
        json={
            "action_hash": prepared["action_hash"],
            "approval_token": prepared["approval_token"],
        },
        headers={"Idempotency-Key": uuid.uuid4().hex},
    )
    assert replay.status_code == 409, replay.text
    assert replay.json()["error"]["code"] == "approval_consumed"


def test_draft_approval_rejects_changed_disclosure_settings(
    client: TestClient, workspace: Path
) -> None:
    source_run = "run_inv_draft_settings"
    store = ArtifactStore(workspace)
    store.start_session("demo", source_run)
    payload = {"candidates": []}
    store.save_artifact(
        Artifact(
            id=make_artifact_id("qcand", {"run": source_run}),
            type=ArtifactType.QUESTION_CANDIDATE_SET,
            project_id="demo",
            session_id=source_run,
            payload=payload,
        )
    )
    store.mark_session_status("demo", source_run, "completed")
    session = f"draft-settings-{uuid.uuid4().hex}"
    headers = {"X-EDA-Session": session}
    initial = client.put(
        "/api/v1/settings",
        json={"payload_policy": "schema_only"},
        headers=headers,
    )
    assert initial.status_code == 200, initial.text
    prepared = client.post(
        f"/api/v1/sessions/{source_run}/questions/prepare-draft",
        json={"question": "Which region should we inspect?", "llm": "env"},
        headers=headers,
    ).json()

    changed = client.put(
        "/api/v1/settings",
        json={
            "payload_policy": "schema+aggregates+sample",
                "provider": "openai",
                "model": "gpt-4.1",
            "base_url": "https://provider.invalid/v1",
            "api_key": "sk-validation-only",
        },
        headers=headers,
    )
    assert changed.status_code == 200, changed.text
    response = client.post(
        f"/api/v1/sessions/{source_run}/questions",
        json={
            "action_hash": prepared["action_hash"],
            "approval_token": prepared["approval_token"],
        },
        headers={**headers, "Idempotency-Key": uuid.uuid4().hex},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "question_invalid"
    assert "Disclosure settings changed" in response.json()["error"]["message"]


def test_question_execute_approval_does_not_work_for_drafting(
    client: TestClient, source_run: str
) -> None:
    """Cross-kind: an approval registered for question execution must read as
    not-found on the draft endpoint."""
    target = next(item for item in _questions(client, source_run) if item["executable"])
    prepared = client.post(
        f"/api/v1/sessions/{source_run}/questions/{target['question_id']}/prepare",
        json={"llm": "offline"},
    ).json()
    response = client.post(
        f"/api/v1/sessions/{source_run}/questions",
        json={
            "action_hash": prepared["action_hash"],
            "approval_token": prepared["approval_token"],
        },
    )
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "approval_not_found"


# ------------------------------------------------------- plan → approve → run


@pytest.fixture(scope="module")
def approved_plan(client: TestClient, source_run: str) -> dict:
    """Build one plan for an executable question and approve it."""
    target = next(item for item in _questions(client, source_run) if item["executable"])
    _build_plans(client, source_run, [target["question_id"]])
    view = _investigations(client, source_run)
    plan = next(item for item in view["plans"] if item["question_id"] == target["question_id"])
    assert plan["status"] == "pending"
    assert plan["can_approve"] is True
    prepared = _prepare_decision(client, source_run, plan["plan_id"], "approved")
    decided = _decide(client, source_run, plan["plan_id"], "approved", prepared)
    assert decided.status_code == 200, decided.text
    assert decided.json()["plan"]["status"] == "approved"
    return decided.json()["plan"]


def test_plan_build_lists_a_reviewable_plan(
    client: TestClient, source_run: str, approved_plan: dict
) -> None:
    assert approved_plan["plan_session_id"].startswith("investigation_")
    assert approved_plan["method_family"]
    assert approved_plan["candidate_fingerprint"]
    assert approved_plan["validation_gates"], "a plan must carry its gates for review"
    assert approved_plan["can_execute"] is True


def test_plan_idempotency_key_binds_question_selection_and_depth(
    client: TestClient, source_run: str
) -> None:
    target = next(item for item in _questions(client, source_run) if item["executable"])
    key = uuid.uuid4().hex
    first = client.post(
        f"/api/v1/sessions/{source_run}/investigations/plan",
        json={"question_ids": [target["question_id"]], "deep": False},
        headers={"Idempotency-Key": key},
    )
    assert first.status_code == 201, first.text

    replay = client.post(
        f"/api/v1/sessions/{source_run}/investigations/plan",
        json={"question_ids": [target["question_id"]], "deep": False},
        headers={"Idempotency-Key": key},
    )
    assert replay.status_code == 201, replay.text
    assert replay.json()["job"]["job_id"] == first.json()["job"]["job_id"]

    changed = client.post(
        f"/api/v1/sessions/{source_run}/investigations/plan",
        json={"question_ids": [target["question_id"]], "deep": True},
        headers={"Idempotency-Key": key},
    )
    assert changed.status_code == 422, changed.text
    assert changed.json()["error"]["code"] == "idempotency_key_reused"
    assert _wait_terminal(client, first.json()["job"]["job_id"])["status"] == "completed"


def test_plan_build_unknown_question_is_404(client: TestClient, source_run: str) -> None:
    response = client.post(
        f"/api/v1/sessions/{source_run}/investigations/plan",
        json={"question_ids": ["q_missing"]},
        headers={"Idempotency-Key": uuid.uuid4().hex},
    )
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "investigation_not_found"


def test_approving_twice_is_409_not_decidable(
    client: TestClient, source_run: str, approved_plan: dict
) -> None:
    response = client.post(
        f"/api/v1/sessions/{source_run}/investigations/{approved_plan['plan_id']}/prepare-decision",
        json={"decision": "approved"},
    )
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "investigation_not_decidable"


def test_execute_approved_plan_produces_a_validated_finding(
    client: TestClient, workspace: Path, source_run: str, approved_plan: dict
) -> None:
    prepared = client.post(
        f"/api/v1/sessions/{source_run}/investigations/prepare-execute",
        json={"plan_ids": [approved_plan["plan_id"]], "llm": "offline"},
    )
    assert prepared.status_code == 200, prepared.text
    body = prepared.json()
    assert body["plan_session_id"] == approved_plan["plan_session_id"]
    assert body["llm_mode"] == "offline"

    started = client.post(
        f"/api/v1/sessions/{source_run}/investigations/execute",
        json={
            "action_hash": body["action_hash"],
            "approval_token": body["approval_token"],
        },
        headers={"Idempotency-Key": uuid.uuid4().hex},
    )
    assert started.status_code == 201, started.text
    assert started.json()["execution_session_id"].startswith("ixsess_")
    final = _wait_terminal(client, started.json()["job"]["job_id"])
    assert final["status"] == "completed", final
    assert final["kind"] == "investigation_execute"

    plan = _plan_for(client, source_run, approved_plan["plan_id"])
    assert plan["status"] == "executed"
    assert plan["outcome_status"] == "validated"
    assert plan["finding_texts"], "execution must publish at least one finding"
    assert plan["report_readiness"]

    store = ArtifactStore(workspace)
    findings = store.list_indexed_artifacts(
        project_id="demo",
        session_id=approved_plan["plan_session_id"],
        artifact_types=(ArtifactType.VALIDATED_FINDING,),
    )
    assert findings, "the plan run must carry the ValidatedFinding artifact"


def test_execute_replayed_token_is_409_consumed(
    client: TestClient, source_run: str, approved_plan: dict
) -> None:
    """The execute approval is single-use even after the plan already ran."""
    response = client.post(
        f"/api/v1/sessions/{source_run}/investigations/prepare-execute",
        json={"plan_ids": [approved_plan["plan_id"]], "llm": "offline"},
    )
    # Prepare itself refuses once an outcome exists — the fail-closed gate sits
    # before the token is ever minted.
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "investigation_not_executable"


# ------------------------------------------------------------- reject path


def test_reject_path_records_a_rejection_and_blocks_execution(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    executable = [item for item in _questions(client, source_run) if item["executable"]]
    target = executable[-1]
    _build_plans(client, source_run, [target["question_id"]])
    plan = next(
        item
        for item in _investigations(client, source_run)["plans"]
        if item["question_id"] == target["question_id"] and item["status"] == "pending"
    )
    prepared = _prepare_decision(
        client, source_run, plan["plan_id"], "rejected", reason="Out of scope for this review."
    )
    rejected = _decide(client, source_run, plan["plan_id"], "rejected", prepared)
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()["plan"]["status"] == "rejected"

    refreshed = _plan_for(client, source_run, plan["plan_id"])
    assert refreshed["can_execute"] is False
    assert refreshed["outcome_status"] == "rejected"

    execute = client.post(
        f"/api/v1/sessions/{source_run}/investigations/prepare-execute",
        json={"plan_ids": [plan["plan_id"]], "llm": "offline"},
    )
    assert execute.status_code == 409, execute.text
    assert execute.json()["error"]["code"] == "investigation_not_executable"

    store = ArtifactStore(workspace)
    records = [
        artifact
        for artifact in store.list_indexed_artifacts(
            project_id="demo",
            session_id=plan["plan_session_id"],
            artifact_types=(ArtifactType.INVESTIGATION_RECORD,),
        )
        if artifact.payload.get("investigation_id") == plan["investigation_id"]
    ]
    assert any(item.payload["status"] == "rejected" for item in records)


# --------------------------------------------------- approval binding to content


def test_decision_after_plan_content_change_is_409_source_changed(
    client: TestClient, workspace: Path
) -> None:
    """The approval binds the plan's content fingerprint: rewriting the plan
    artifact between prepare and approve must fail closed."""
    session_id = _analysed_run(client, "run_inv_fingerprint")
    target = next(item for item in _questions(client, session_id) if item["executable"])
    _build_plans(client, session_id, [target["question_id"]])
    plan = _investigations(client, session_id)["plans"][0]
    prepared = _prepare_decision(client, session_id, plan["plan_id"], "approved")

    store = ArtifactStore(workspace)
    artifact = store.get_artifact(
        plan["plan_id"], project_id="demo", session_id=plan["plan_session_id"]
    )
    payload = dict(artifact.payload)
    payload["method_recipe"] = payload["method_recipe"] + " (rewritten)"
    store.save_artifact(artifact.model_copy(update={"payload": payload}))

    response = _decide(client, session_id, plan["plan_id"], "approved", prepared)
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "investigation_source_changed"


def test_question_approval_is_not_accepted_by_the_decision_endpoint(
    client: TestClient, workspace: Path, source_run: str, approved_plan: dict
) -> None:
    """Cross-kind: an approval of another kind on the same run reads 404."""
    store = ArtifactStore(workspace)
    digest, token, _expires = ApprovalService(store).register(
        kind="question_execute",
        session_id=source_run,
        project_id="demo",
        action={"type": "question_probe"},
        payload={"plan_id": approved_plan["plan_id"], "decision": "approved"},
    )
    response = client.post(
        f"/api/v1/sessions/{source_run}/investigations/{approved_plan['plan_id']}/approve",
        json={"action_hash": digest, "approval_token": token},
    )
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "approval_not_found"


def test_expired_decision_approval_is_410(
    app: FastAPI, client: TestClient, workspace: Path
) -> None:
    session_id = _analysed_run(client, "run_inv_expiry")
    target = next(item for item in _questions(client, session_id) if item["executable"])
    _build_plans(client, session_id, [target["question_id"]])
    plan = _investigations(client, session_id)["plans"][0]

    store = ArtifactStore(workspace)
    original = app.state.investigation_service
    app.state.investigation_service = InvestigationService(
        store, ApprovalService(store, ttl_seconds=-1), app.state.job_service
    )
    try:
        prepared = _prepare_decision(client, session_id, plan["plan_id"], "approved")
        response = _decide(client, session_id, plan["plan_id"], "approved", prepared)
    finally:
        app.state.investigation_service = original
    assert response.status_code == 410, response.text
    assert response.json()["error"]["code"] == "approval_expired"


def test_approve_token_cannot_be_spent_on_reject(
    client: TestClient,
    workspace: Path,
    source_run: str,
) -> None:
    """The approval binds the decision, so an approve token cannot reject."""
    session_id = source_run
    pending = [
        item for item in _investigations(client, session_id)["plans"] if item["status"] == "pending"
    ]
    if not pending:
        target = _questions(client, session_id)[0]
        _build_plans(client, session_id, [target["question_id"]])
        pending = [
            item
            for item in _investigations(client, session_id)["plans"]
            if item["status"] == "pending"
        ]
    plan = pending[0]
    prepared = _prepare_decision(client, session_id, plan["plan_id"], "approved")
    response = _decide(client, session_id, plan["plan_id"], "rejected", prepared)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "investigation_invalid"
    row = ArtifactStore(workspace).get_pending_action(
        prepared["action_hash"], session_id=session_id
    )
    assert row is not None
    assert row["status"] == "pending"
    assert row["generation"] == prepared["approval_token"]

    retried = _decide(client, session_id, plan["plan_id"], "approved", prepared)
    assert retried.status_code == 200, retried.text


def test_decision_persistence_fault_restores_same_approval(
    client: TestClient,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = _analysed_run(client, "run_inv_decision_fault")
    target = next(item for item in _questions(client, session_id) if item["executable"])
    _build_plans(client, session_id, [target["question_id"]])
    plan = _investigations(client, session_id)["plans"][0]
    prepared = _prepare_decision(client, session_id, plan["plan_id"], "approved")

    import eda_platform.drivers.investigation_orchestrator as orchestrator

    original = orchestrator.approve_plan
    monkeypatch.setattr(
        orchestrator,
        "approve_plan",
        lambda **_kwargs: (_ for _ in ()).throw(
            OSError("injected decision persistence fault")
        ),
    )
    with pytest.raises(OSError, match="injected decision persistence fault"):
        _decide(client, session_id, plan["plan_id"], "approved", prepared)

    row = ArtifactStore(workspace).get_pending_action(
        prepared["action_hash"], session_id=session_id
    )
    assert row is not None
    assert row["status"] == "pending"
    assert row["generation"] == prepared["approval_token"]

    monkeypatch.setattr(orchestrator, "approve_plan", original)
    retried = _decide(client, session_id, plan["plan_id"], "approved", prepared)
    assert retried.status_code == 200, retried.text


# ---------------------------------------------------------------- Ultra loop


def _set_depth(client: TestClient, depth: int) -> dict:
    response = client.put("/api/v1/settings", json={"analysis_depth": depth})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="module")
def ultra_run(client: TestClient) -> str:
    """A second analysed run, kept apart from the fixture chain above so the
    macro-loop tests own their plan runs (the loop writes into them)."""
    return _analysed_run(client, "run_inv_ultra")


@pytest.fixture(scope="module")
def descriptive_question_ids(
    client: TestClient, workspace: Path, ultra_run: str
) -> list[str]:
    """Questions the planner routes to the descriptive (read-only SQL) family.

    Only that route leaves QuestionExecutionResult artifacts, and only those
    are admitted by the macro loop's §8.2 validation bridge — a method-route
    plan run gives the loop nothing to build follow-ups from. The family is
    decided by the planner, so it can only be read off a built plan.
    """
    ids = [item["question_id"] for item in _questions(client, ultra_run) if item["executable"]]
    build = _build_plans(client, ultra_run, ids)
    plan_session_id = _planned_session_id(ArtifactStore(workspace), build["execution_session_id"])
    descriptive = [
        item["question_id"]
        for item in _investigations(client, ultra_run)["plans"]
        if item["plan_session_id"] == plan_session_id
        and item["method_family"] == "descriptive_analysis"
        and item["execution_ready"]
    ]
    assert len(descriptive) >= 2, descriptive
    return descriptive


def _executed_descriptive_plan_run(
    client: TestClient, workspace: Path, session_id: str, question_ids: list[str]
) -> str:
    """Plan, approve and execute descriptive questions, returning the plan run.

    Descriptive plans run the read-only SQL path and therefore leave
    QuestionExecutionResult artifacts — the only artifact type the macro loop's
    §8.2 validation bridge admits, so a method-route plan run would leave the
    loop with nothing to build follow-ups from.
    """
    build = _build_plans(client, session_id, question_ids)
    plan_session_id = _planned_session_id(ArtifactStore(workspace), build["execution_session_id"])
    plans = [
        item
        for item in _investigations(client, session_id)["plans"]
        if item["plan_session_id"] == plan_session_id
        and item["method_family"] == "descriptive_analysis"
        and item["can_approve"]
    ]
    assert plans, "expected descriptive plans to approve"
    for plan in plans:
        prepared = _prepare_decision(client, session_id, plan["plan_id"], "approved")
        decided = _decide(client, session_id, plan["plan_id"], "approved", prepared)
        assert decided.status_code == 200, decided.text
    prepared_exec = client.post(
        f"/api/v1/sessions/{session_id}/investigations/prepare-execute",
        json={"plan_ids": [plan["plan_id"] for plan in plans], "llm": "offline"},
    )
    assert prepared_exec.status_code == 200, prepared_exec.text
    body = prepared_exec.json()
    started = client.post(
        f"/api/v1/sessions/{session_id}/investigations/execute",
        json={
            "action_hash": body["action_hash"],
            "approval_token": body["approval_token"],
        },
        headers={"Idempotency-Key": uuid.uuid4().hex},
    )
    assert started.status_code == 201, started.text
    final = _wait_terminal(client, started.json()["job"]["job_id"])
    assert final["status"] == "completed", final
    return plan_session_id


def test_thinking_level_round_trips_through_settings(client: TestClient) -> None:
    try:
        assert _set_depth(client, 2)["analysis_depth"] == 2
        assert client.get("/api/v1/settings").json()["analysis_depth"] == 2
        rejected = client.put("/api/v1/settings", json={"analysis_depth": 9})
        assert rejected.status_code == 422
        assert rejected.json()["error"]["code"] == "settings_invalid"
    finally:
        _set_depth(client, 0)


def test_macro_loop_requires_ultra_and_an_executed_plan_run(
    client: TestClient, source_run: str, approved_plan: dict
) -> None:
    _set_depth(client, 0)
    refused = client.post(
        f"/api/v1/sessions/{source_run}/investigations/prepare-macro-loop",
        json={"plan_session_id": approved_plan["plan_session_id"], "llm": "offline"},
    )
    assert refused.status_code == 409, refused.text
    assert refused.json()["error"]["code"] == "macro_loop_not_authorized"


def test_macro_loop_writes_preauthorization_and_ledger(
    client: TestClient,
    workspace: Path,
    ultra_run: str,
    descriptive_question_ids: list[str],
) -> None:
    """The §8.3 pre-authorization artifact lands only after a human approved
    this exact plan run and depth; run_macro_loop refuses without it."""
    source_run = ultra_run
    plan_session_id = _executed_descriptive_plan_run(
        client, workspace, source_run, descriptive_question_ids[:1]
    )
    store = ArtifactStore(workspace)
    before = [
        artifact
        for artifact in store.list_indexed_artifacts(
            project_id="demo",
            session_id=plan_session_id,
            artifact_types=(ArtifactType.INVESTIGATION_APPROVAL,),
        )
        if InvestigationApproval.model_validate(artifact.payload).investigation_id.startswith(
            "macro_loop_"
        )
    ]
    assert not before, "no macro-loop pre-authorization may exist before approval"

    _set_depth(client, 2)
    try:
        prepared = client.post(
            f"/api/v1/sessions/{source_run}/investigations/prepare-macro-loop",
            json={"plan_session_id": plan_session_id, "llm": "offline"},
        )
        assert prepared.status_code == 200, prepared.text
        body = prepared.json()
        assert body["depth"] == 2
        assert body["rounds_cap"] >= 1
        started = client.post(
            f"/api/v1/sessions/{source_run}/investigations/macro-loop",
            json={
                "action_hash": body["action_hash"],
                "approval_token": body["approval_token"],
            },
            headers={"Idempotency-Key": uuid.uuid4().hex},
        )
        assert started.status_code == 201, started.text
        assert started.json()["execution_session_id"].startswith("mlsess_")
        final = _wait_terminal(client, started.json()["job"]["job_id"])
        assert final["status"] == "completed", final
        assert final["kind"] == "macro_loop"
    finally:
        _set_depth(client, 0)

    preauth = [
        artifact
        for artifact in store.list_indexed_artifacts(
            project_id="demo",
            session_id=plan_session_id,
            artifact_types=(ArtifactType.INVESTIGATION_APPROVAL,),
        )
        if InvestigationApproval.model_validate(artifact.payload).investigation_id
        == f"macro_loop_{plan_session_id}"
    ]
    assert len(preauth) == 1, "the pre-authorization must be written exactly once"

    ledgers = store.list_indexed_artifacts(
        project_id="demo",
        session_id=plan_session_id,
        artifact_types=(ArtifactType.LOOP_LEDGER,),
    )
    assert ledgers, "the loop must persist its ledger"
    view = _investigations(client, source_run)
    loops = [item for item in view["macro_loops"] if item["plan_session_id"] == plan_session_id]
    assert loops and loops[0]["rounds"], "the ledger must surface in the read model"


class _FollowUpStubLLM:
    """Serves one canned follow-up generation per round; refuses other tasks.

    An offline client concludes at round 1 by design (followup_agent fail-safe),
    so multi-round behaviour can only be exercised with a live-shaped client.
    """

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.tasks: list[str] = []

    def structured(self, *, task: str, schema: type, payload: dict) -> Any:
        self.tasks.append(task)
        if task != "l2_followup_generation":
            raise RuntimeError(f"task {task!r} is not served by this stub")
        response: dict[str, Any] = (
            self._responses.pop(0) if self._responses else {"concluded": True}
        )
        # Cite a parent the loop actually admitted. Only findings that clear the
        # §8.2 bridge reach ``validated_findings``, and the converter drops
        # proposals citing anything else; a real model can only cite what it was
        # shown. Hard-coding an id here made the test depend on plan ordering.
        offered = [item["finding_id"] for item in payload.get("validated_findings", ())]
        if offered and response.get("proposals"):
            response = {
                **response,
                "proposals": [
                    {**proposal, "parent_finding_id": offered[0]}
                    for proposal in response["proposals"]
                ],
            }
        return schema.model_validate(response)

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> None:
        return None


def test_macro_loop_job_runs_multiple_rounds(
    client: TestClient,
    workspace: Path,
    ultra_run: str,
    descriptive_question_ids: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Depth 3 through the real worker branch with a stubbed follow-up model:
    the ledger must show the seed round plus executed follow-up rounds."""
    source_run = ultra_run
    # A different question set mints a different plan run, so this loop never
    # shares internal round runs with the ledger test above.
    plan_session_id = _executed_descriptive_plan_run(
        client, workspace, source_run, descriptive_question_ids[1:]
    )
    store = ArtifactStore(workspace)
    _set_depth(client, 3)
    try:
        prepared = client.post(
            f"/api/v1/sessions/{source_run}/investigations/prepare-macro-loop",
            json={"plan_session_id": plan_session_id, "llm": "env"},
        )
        assert prepared.status_code == 200, prepared.text
        body = prepared.json()
        assert body["depth"] == 3
        assert body["rounds_cap"] >= 2
        # Create the job row without enqueueing, then run the worker in-process
        # so the stubbed model reaches run_macro_loop.
        backend = client.app.state.job_service._backend  # type: ignore[attr-defined]
        original = backend.enqueue
        captured: dict = {}

        def capture(command):
            captured["params"] = command.params_json
            captured["job_id"] = command.job_id
            from eda_platform.application.ports import JobRef

            return JobRef(job_id=command.job_id, pid=None)

        backend.enqueue = capture
        try:
            started = client.post(
                f"/api/v1/sessions/{source_run}/investigations/macro-loop",
                json={
                    "action_hash": body["action_hash"],
                    "approval_token": body["approval_token"],
                },
                headers={"Idempotency-Key": uuid.uuid4().hex},
            )
        finally:
            backend.enqueue = original
        assert started.status_code == 201, started.text
    finally:
        _set_depth(client, 0)

    # The stub fills each proposal's parent from the findings the loop offered it,
    # so this exercises the funnel rather than the plan ordering.
    stub = _FollowUpStubLLM(
        [
            {
                "concluded": False,
                "proposals": [{"question_text": "How does amount vary by quantity band?"}],
            },
            {
                "concluded": False,
                "proposals": [{"question_text": "How does amount vary across order months?"}],
            },
        ]
    )
    monkeypatch.setattr(worker_runner, "_build_llm", lambda params: stub)
    run_claimed_job(workspace, captured["job_id"], captured["params"])

    final = store.get_job(captured["job_id"])
    assert final is not None
    assert final["status"] == "completed", final

    ledger_artifacts = store.list_indexed_artifacts(
        project_id="demo",
        session_id=plan_session_id,
        artifact_types=(ArtifactType.LOOP_LEDGER,),
    )
    ledger = LoopLedger.model_validate(ledger_artifacts[-1].payload)
    assert ledger.depth == 3
    round_ids = [row.round_id for row in ledger.rounds]
    assert round_ids[0] == 0
    assert len(round_ids) >= 2, f"expected the seed round plus follow-up rounds, got {round_ids}"
    executed = [row for row in ledger.rounds if row.executed_questions > 0]
    assert executed, "at least one follow-up round must have executed questions"
    assert stub.tasks, "the follow-up generator must have been called"
    trace = store.list_trace_events(project_id="demo", session_id=plan_session_id)
    usage_events = [event for event in trace if event.event_type == "llm_usage"]
    reserved_events = [event for event in trace if event.event_type == "budget_reserved"]
    settled_events = [event for event in trace if event.event_type == "budget_settled"]
    # Every physical attempt, including a downstream task the stub rejects, goes
    # through exactly one ledger wrapper. A stacked wrapper would duplicate rows.
    assert [event.name for event in usage_events] == stub.tasks
    assert len(reserved_events) == len(stub.tasks)
    assert len(settled_events) == len(stub.tasks)
    # The funnel really ran: each executed round left its own internal plan run.
    internal = [
        row["session_id"]
        for row in store.query_session_index_rows("demo", limit=500)
        if row["session_id"].startswith(f"{plan_session_id}_macro_r")
    ]
    assert internal, "follow-up rounds must run through derived internal runs"


def test_macro_loop_job_refuses_without_preauthorization(workspace: Path) -> None:
    """Defence in depth: a hand-made macro_loop job row cannot start rounds."""
    store = ArtifactStore(workspace)
    job = store.create_job(
        job_id=f"job_{uuid.uuid4().hex[:12]}",
        session_id=f"mlsess_probe_{execution_lane('run_inv_unknown')}_abcdef",
        project_id="demo",
        kind="macro_loop",
        idempotency_key=None,
    )
    params = {
        "source_session_id": "run_inv_src",
        "plan_session_id": "investigation_does_not_exist",
        "depth": 2,
        "llm": "offline",
    }
    run_claimed_job(workspace, str(job["job_id"]), json.dumps(params))
    final = store.get_job(str(job["job_id"]))
    assert final is not None
    assert final["status"] == "failed"


def test_execute_job_recomputes_plan_fingerprint_and_fails_closed(
    client: TestClient, workspace: Path
) -> None:
    """The worker re-derives each plan's fingerprint right before executing, so
    a plan rewritten between enqueue and pickup never sessions."""
    session_id = _analysed_run(client, "run_inv_worker_guard")
    target = next(item for item in _questions(client, session_id) if item["executable"])
    _build_plans(client, session_id, [target["question_id"]])
    plan = _investigations(client, session_id)["plans"][0]

    store = ArtifactStore(workspace)
    job = store.create_job(
        job_id=f"job_{uuid.uuid4().hex[:12]}",
        session_id=f"ixsess_probe_{execution_lane(plan['plan_session_id'])}_abcdef",
        project_id="demo",
        kind="investigation_execute",
        idempotency_key=None,
    )
    params = {
        "source_session_id": session_id,
        "plan_session_id": plan["plan_session_id"],
        "plan_ids": [plan["plan_id"]],
        "plan_fingerprints": {plan["plan_id"]: "0" * 64},
        "llm": "offline",
    }
    run_claimed_job(workspace, str(job["job_id"]), json.dumps(params))
    final = store.get_job(str(job["job_id"]))
    assert final is not None
    assert final["status"] == "failed"
    assert final["error_code"] == "ValueError"
    assert "investigation plan changed since approval" in str(final["error_message"])


def test_generic_jobs_route_rejects_investigation_kinds(
    client: TestClient, source_run: str
) -> None:
    for kind in ("investigation_plan", "investigation_execute", "macro_loop", "question_draft"):
        response = client.post(
            f"/api/v1/sessions/{source_run}/jobs",
            json={"kind": kind, "datasets": ["seed/orders.csv"]},
        )
        assert response.status_code == 422, (kind, response.text)
