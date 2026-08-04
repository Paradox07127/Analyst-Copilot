"""Questions API vertical slice: list shapes candidates + latest execution,
prepare registers a server-side approval, execute consumes it once into a
`question_exec` job that runs the existing question batch driver offline on a
derived qsess_* run. Spawned-worker pattern mirrors test_api_cleaning."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from job_test_helpers import run_claimed_job

from eda_platform.api.main import create_app
from eda_platform.application.ports import JobRef
from eda_platform.application.services.approval_service import ApprovalService
from eda_platform.application.services.question_service import (
    QuestionService,
    _failure_headline,
    candidate_fingerprint,
    latest_candidate_set,
)
from eda_platform.core.ids import make_artifact_id
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.questions import QuestionCandidate

_JOB_TIMEOUT_SECONDS = 120.0


def _seed_csv() -> str:
    """Deterministic date/category/numeric mix that yields template questions
    (trend, group difference, domain metrics) from offline question discovery."""
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
    root = tmp_path_factory.mktemp("questions_api")
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


@pytest.fixture(scope="module")
def source_run(client: TestClient) -> str:
    """A completed offline run whose QuestionCandidateSet the slice serves."""
    session_id = "run_questions_src"
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


def _questions(client: TestClient, session_id: str) -> list[dict]:
    response = client.get(f"/api/v1/sessions/{session_id}/questions")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_id"] == session_id
    return body["questions"]


def _prepare(
    client: TestClient, session_id: str, question_id: str, llm: str = "offline"
) -> dict:
    # llm rides in the prepare body now: execute runs whatever was approved.
    response = client.post(
        f"/api/v1/sessions/{session_id}/questions/{question_id}/prepare",
        json={"llm": llm},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _execute(
    client: TestClient,
    session_id: str,
    question_id: str,
    prepared: dict,
    idempotency_key: str | None = None,
):
    headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
    return client.post(
        f"/api/v1/sessions/{session_id}/questions/{question_id}/execute",
        json={
            "action_hash": prepared["action_hash"],
            "approval_token": prepared["approval_token"],
        },
        headers=headers,
    )


def test_list_shapes_candidates_with_auto_exec_status(
    client: TestClient, source_run: str
) -> None:
    questions = _questions(client, source_run)
    assert questions, "offline auto_eda produced no question candidates"
    first = questions[0]
    for field in (
        "question_id",
        "question",
        "origin",
        "priority",
        "executable",
        "target_datasets",
    ):
        assert field in first, f"missing field {field}"
    priorities = [item["priority"] for item in questions]
    assert priorities == sorted(priorities, reverse=True)
    # auto_eda auto-executes top template questions inside the source run, so
    # at least one candidate already carries an execution summary bound there.
    executed = [item for item in questions if item["execution"] is not None]
    assert executed, "expected auto-executed questions in the source run"
    summary = executed[0]["execution"]
    assert summary["execution_session_id"] == source_run
    assert summary["qexec_artifact_id"].startswith("qexec_")
    assert summary["outcome"] in {"answered", "abstained", "failed"}


def test_list_unknown_run_is_404(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/run_missing/questions")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_prepare_registers_pending_approval(client: TestClient, source_run: str) -> None:
    question = _questions(client, source_run)[0]
    prepared = _prepare(client, source_run, question["question_id"])
    assert prepared["session_id"] == source_run
    assert prepared["question_id"] == question["question_id"]
    assert len(prepared["action_hash"]) == 64
    assert len(prepared["approval_token"]) == 32
    assert prepared["question"] == question["question"]
    assert prepared["uses_llm"] is False
    assert prepared["llm_mode"] == "offline"
    if question["origin"] == "template":
        assert prepared["sql_preview"]


def test_prepare_live_mode_approves_autonomous_agent_scope(
    client: TestClient, source_run: str
) -> None:
    question = _questions(client, source_run)[0]
    prepared = _prepare(client, source_run, question["question_id"], llm="env")

    assert prepared["uses_llm"] is True
    assert prepared["llm_mode"] == "env"
    assert prepared["sql_preview"] is None
    assert prepared["target_datasets"] == question["target_datasets"]


def test_prepare_unknown_question_is_404(client: TestClient, source_run: str) -> None:
    response = client.post(f"/api/v1/sessions/{source_run}/questions/q_nope/prepare")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "question_not_found"


def test_execute_full_chain_offline_updates_findings(
    client: TestClient, workspace: Path, source_run: str
) -> None:
    questions = _questions(client, source_run)
    target = next(
        item
        for item in questions
        if item["executable"] and item["origin"] == "template"
    )
    question_id = target["question_id"]
    prepared = _prepare(client, source_run, question_id)

    response = _execute(client, source_run, question_id, prepared, uuid.uuid4().hex)
    assert response.status_code == 201, response.text
    started = response.json()
    assert started["question_id"] == question_id
    assert started["execution_session_id"].startswith("qsess_")
    assert started["job"]["session_id"] == started["execution_session_id"]

    final = _wait_terminal(client, started["job"]["job_id"])
    assert final["status"] == "completed", final
    assert final["kind"] == "question_exec"

    refreshed = next(
        item
        for item in _questions(client, source_run)
        if item["question_id"] == question_id
    )
    summary = refreshed["execution"]
    assert summary is not None
    # The fresh batch execution supersedes the source run's auto-exec result.
    assert summary["execution_session_id"] == started["execution_session_id"]
    assert summary["outcome"] == "answered"
    assert summary["findings_count"] >= 1

    store = ArtifactStore(workspace)
    qexec = store.list_indexed_artifacts(
        project_id="demo",
        session_id=started["execution_session_id"],
        artifact_types=(ArtifactType.QUESTION_EXECUTION_RESULT,),
    )
    assert len(qexec) == 1
    assert len(qexec[0].payload["findings"]) == summary["findings_count"]


def test_execute_replayed_token_is_409_consumed(
    client: TestClient, source_run: str
) -> None:
    questions = _questions(client, source_run)
    target = next(item for item in questions if item["executable"])
    prepared = _prepare(client, source_run, target["question_id"])
    first = _execute(client, source_run, target["question_id"], prepared, uuid.uuid4().hex)
    assert first.status_code == 201, first.text
    _wait_terminal(client, first.json()["job"]["job_id"])

    # Same token, fresh idempotency key: the approval is single-use.
    replay = _execute(client, source_run, target["question_id"], prepared, uuid.uuid4().hex)
    assert replay.status_code == 409, replay.text
    assert replay.json()["error"]["code"] == "approval_consumed"


def test_execute_idempotency_key_replays_same_job(
    client: TestClient, source_run: str
) -> None:
    questions = _questions(client, source_run)
    target = next(item for item in questions if item["executable"])
    prepared = _prepare(client, source_run, target["question_id"])
    key = uuid.uuid4().hex
    first = _execute(client, source_run, target["question_id"], prepared, key)
    assert first.status_code == 201, first.text
    _wait_terminal(client, first.json()["job"]["job_id"])

    replay = _execute(client, source_run, target["question_id"], prepared, key)
    assert replay.status_code == 201, replay.text
    assert replay.json()["job"]["job_id"] == first.json()["job"]["job_id"]
    assert replay.json()["execution_session_id"] == first.json()["execution_session_id"]


def test_concurrent_same_approval_and_idempotency_key_replays_winner(
    client: TestClient,
    source_run: str,
    workspace: Path,
) -> None:
    target = next(item for item in _questions(client, source_run) if item["executable"])
    prepared = _prepare(client, source_run, target["question_id"])
    key = uuid.uuid4().hex
    def submit() -> tuple[int, dict]:
        response = _execute(
            client, source_run, target["question_id"], prepared, key
        )
        return response.status_code, response.json()

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _index: submit(), range(2)))

    # The request contract owns the key before the approval layer.  A request
    # that arrives while the winner is still executing gets the documented
    # 409 retry response; one arriving after completion gets its stored 201.
    assert {status for status, _body in responses} <= {201, 409}
    winners = [body for status, body in responses if status == 201]
    assert winners
    winner = winners[0]
    _wait_terminal(client, winner["job"]["job_id"])
    replay = _execute(client, source_run, target["question_id"], prepared, key)
    assert replay.status_code == 201, replay.text
    assert replay.json()["job"]["job_id"] == winner["job"]["job_id"]
    assert replay.json()["execution_session_id"] == winner["execution_session_id"]
    with sqlite3.connect(ArtifactStore(workspace).db_path) as conn:
        count = conn.execute(
            "select count(*) from jobs where idempotency_key = ?", (key,)
        ).fetchone()
    assert count is not None and count[0] == 1


def test_execute_unknown_token_is_404(client: TestClient, source_run: str) -> None:
    questions = _questions(client, source_run)
    target = next(item for item in questions if item["executable"])
    prepared = _prepare(client, source_run, target["question_id"])
    response = _execute(
        client,
        source_run,
        target["question_id"],
        {**prepared, "approval_token": "f" * 32},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "approval_not_found"


def test_wrong_question_path_does_not_consume_correct_approval(
    app: FastAPI,
    client: TestClient,
    workspace: Path,
    source_run: str,
) -> None:
    targets = [item for item in _questions(client, source_run) if item["executable"]]
    assert len(targets) >= 2
    approved = targets[0]["question_id"]
    wrong = targets[1]["question_id"]
    prepared = _prepare(client, source_run, approved)

    swapped = _execute(client, source_run, wrong, prepared)
    assert swapped.status_code == 422, swapped.text
    row = ArtifactStore(workspace).get_pending_action(
        prepared["action_hash"], session_id=source_run
    )
    assert row is not None
    assert row["status"] == "pending"
    assert row["generation"] == prepared["approval_token"]

    backend = app.state.job_service._backend
    original = backend.enqueue
    backend.enqueue = lambda command: JobRef(job_id=command.job_id, pid=None)
    try:
        retried = _execute(client, source_run, approved, prepared)
    finally:
        backend.enqueue = original
    assert retried.status_code == 201, retried.text
    ArtifactStore(workspace).mark_job_status(
        retried.json()["job"]["job_id"],
        "completed",
    )


def test_execute_expired_approval_is_410(
    app: FastAPI, client: TestClient, workspace: Path, source_run: str
) -> None:
    store = ArtifactStore(workspace)
    original = app.state.question_service
    app.state.question_service = QuestionService(
        store, ApprovalService(store, ttl_seconds=-1), app.state.job_service
    )
    try:
        questions = _questions(client, source_run)
        target = next(item for item in questions if item["executable"])
        prepared = _prepare(client, source_run, target["question_id"])
        response = _execute(client, source_run, target["question_id"], prepared)
        assert response.status_code == 410, response.text
        assert response.json()["error"]["code"] == "approval_expired"
    finally:
        app.state.question_service = original


def test_prepare_blocked_by_feasibility_is_409(client: TestClient, workspace: Path) -> None:
    """A needs_data candidate must be listed as non-executable and refuse prepare."""
    store = ArtifactStore(workspace)
    session_id = "run_questions_blocked"
    store.start_session("demo", session_id)
    payload = {
        "candidates": [
            {
                "question_id": "q_blocked",
                "question_en": "Can churn be predicted from labels we lack?",
                "origin": "llm",
                "target_datasets": ["orders.csv"],
                "score": {
                    "data_availability": 0.2,
                    "statistical_signal": 0.5,
                    "quality_risk": 0.1,
                    "join_risk": 0.0,
                    "deterministic_score": 0.3,
                },
                "feasibility": {
                    "status": "needs_data",
                    "reasons": ["No churn label column exists."],
                    "missing": ["churn label"],
                },
            }
        ]
    }
    store.save_artifact(
        Artifact(
            id=make_artifact_id("qcand", payload),
            type=ArtifactType.QUESTION_CANDIDATE_SET,
            project_id="demo",
            session_id=session_id,
            payload=payload,
        )
    )
    store.mark_session_status("demo", session_id, "completed")

    questions = _questions(client, session_id)
    assert questions[0]["executable"] is False
    response = client.post(f"/api/v1/sessions/{session_id}/questions/q_blocked/prepare")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "question_not_executable"


def test_generic_jobs_route_rejects_question_exec_kind(
    client: TestClient, source_run: str
) -> None:
    response = client.post(
        f"/api/v1/sessions/{source_run}/jobs",
        json={"kind": "question_exec", "datasets": ["seed/orders.csv"]},
    )
    assert response.status_code == 422


def _template_candidate(
    question_id: str,
    *,
    sql: str = "select region, sum(amount) from orders group by 1",
    score: float = 0.8,
) -> dict:
    return {
        "question_id": question_id,
        "question_en": "How does amount vary by region?",
        "origin": "template",
        "sql_template": sql,
        "target_datasets": ["orders.csv"],
        "score": {
            "data_availability": 0.9,
            "statistical_signal": 0.7,
            "quality_risk": 0.1,
            "join_risk": 0.0,
            "deterministic_score": score,
        },
        "feasibility": {"status": "ready", "reasons": [], "missing": []},
    }


def _append_candidate_set(
    store: ArtifactStore, project_id: str, session_id: str, candidates: list[dict]
) -> None:
    payload = {"candidates": candidates}
    store.save_artifact(
        Artifact(
            id=make_artifact_id("qcand", {"run": session_id, "salt": uuid.uuid4().hex}),
            type=ArtifactType.QUESTION_CANDIDATE_SET,
            project_id=project_id,
            session_id=session_id,
            payload=payload,
        )
    )


def _seed_candidate_run(
    store: ArtifactStore, project_id: str, session_id: str, candidates: list[dict]
) -> None:
    store.start_session(project_id, session_id)
    _append_candidate_set(store, project_id, session_id, candidates)
    store.mark_session_status(project_id, session_id, "completed")


def test_execute_after_candidate_sql_change_is_409_source_changed(
    client: TestClient, workspace: Path
) -> None:
    store = ArtifactStore(workspace)
    session_id = "run_q_sql_change"
    _seed_candidate_run(store, "demo", session_id, [_template_candidate("q_fp", sql="select 1")])
    prepared = _prepare(client, session_id, "q_fp")
    # Regenerating the set with different SQL invalidates the approval.
    _append_candidate_set(
        store, "demo", session_id, [_template_candidate("q_fp", sql="select 2")]
    )
    response = _execute(client, session_id, "q_fp", prepared, uuid.uuid4().hex)
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "question_source_changed"


def test_score_only_change_keeps_approval_valid(
    app: FastAPI, client: TestClient, workspace: Path
) -> None:
    """Control probe: the fingerprint covers execution-affecting fields only,
    so a display-only score change must NOT invalidate the approval."""
    store = ArtifactStore(workspace)
    session_id = "run_q_score_change"
    _seed_candidate_run(
        store, "demo", session_id, [_template_candidate("q_fp2", sql="select 1", score=0.8)]
    )
    prepared = _prepare(client, session_id, "q_fp2")
    _append_candidate_set(
        store, "demo", session_id, [_template_candidate("q_fp2", sql="select 1", score=0.2)]
    )
    backend = app.state.job_service._backend
    original = backend.enqueue
    backend.enqueue = lambda command: JobRef(job_id=command.job_id, pid=None)
    try:
        response = _execute(client, session_id, "q_fp2", prepared, uuid.uuid4().hex)
    finally:
        backend.enqueue = original
    assert response.status_code == 201, response.text


def test_runner_recomputes_fingerprint_and_fails_closed(workspace: Path) -> None:
    """Defence in depth: even a job whose params carry a stale fingerprint must
    fail before the driver sessions."""
    store = ArtifactStore(workspace)
    session_id = "run_q_runner_guard"
    _seed_candidate_run(store, "demo", session_id, [_template_candidate("q_guard")])
    job = store.create_job(
        job_id=f"job_{uuid.uuid4().hex[:12]}",
        session_id="qsess_guard_probe",
        project_id="demo",
        kind="question_exec",
        idempotency_key=None,
    )
    params = {
        "source_session_id": session_id,
        "question_id": "q_guard",
        "candidate_fingerprint": "0" * 64,
        "generate_report": False,
        "llm": "offline",
    }
    run_claimed_job(workspace, str(job["job_id"]), json.dumps(params))
    final = store.get_job(str(job["job_id"]))
    assert final is not None
    assert final["status"] == "failed"
    assert final["error_code"] == "ValueError"
    assert "question source changed since approval" in str(final["error_message"])


def test_execute_llm_mode_comes_from_approval_not_client(
    app: FastAPI, client: TestClient, workspace: Path
) -> None:
    """The worker params must carry the approved llm mode and fingerprint; a
    client-sent llm field on execute is ignored."""
    store = ArtifactStore(workspace)
    session_id = "run_q_llm_bind"
    _seed_candidate_run(store, "demo", session_id, [_template_candidate("q_llm")])
    prepared = _prepare(client, session_id, "q_llm", llm="offline")
    assert prepared["llm_mode"] == "offline"

    backend = app.state.job_service._backend
    captured: dict = {}
    original = backend.enqueue

    def capture(command):
        captured["params"] = json.loads(command.params_json)
        return JobRef(job_id=command.job_id, pid=None)

    backend.enqueue = capture
    try:
        response = client.post(
            f"/api/v1/sessions/{session_id}/questions/q_llm/execute",
            json={
                "action_hash": prepared["action_hash"],
                "approval_token": prepared["approval_token"],
                # Extra field: must be ignored, not honoured.
                "llm": "env",
            },
            headers={"Idempotency-Key": uuid.uuid4().hex},
        )
    finally:
        backend.enqueue = original
    assert response.status_code == 201, response.text
    params = captured["params"]
    assert params["llm"] == "offline"
    candidate_set = latest_candidate_set(store, "demo", session_id)
    assert candidate_set is not None
    expected = candidate_fingerprint(
        next(item for item in candidate_set.candidates if item.question_id == "q_llm")
    )
    assert params["candidate_fingerprint"] == expected


def test_candidate_fingerprint_covers_execution_fields_only() -> None:
    base = QuestionCandidate.model_validate(_template_candidate("q_x"))
    changed_sql = QuestionCandidate.model_validate(
        _template_candidate("q_x", sql="select 42")
    )
    changed_score = QuestionCandidate.model_validate(
        _template_candidate("q_x", score=0.1)
    )
    assert candidate_fingerprint(base) != candidate_fingerprint(changed_sql)
    assert candidate_fingerprint(base) == candidate_fingerprint(changed_score)


def test_idempotency_key_cross_project_replay_is_typed_422(
    app: FastAPI, client: TestClient, workspace: Path
) -> None:
    store = ArtifactStore(workspace)
    store.ensure_project("other", name="Other")
    _seed_candidate_run(store, "demo", "run_q_xp_a", [_template_candidate("q_xp")])
    _seed_candidate_run(store, "other", "run_q_xp_b", [_template_candidate("q_xp")])

    backend = app.state.job_service._backend
    original = backend.enqueue
    backend.enqueue = lambda command: JobRef(job_id=command.job_id, pid=None)
    key = uuid.uuid4().hex
    try:
        prepared_a = _prepare(client, "run_q_xp_a", "q_xp")
        first = _execute(client, "run_q_xp_a", "q_xp", prepared_a, key)
        assert first.status_code == 201, first.text

        prepared_b = _prepare(client, "run_q_xp_b", "q_xp")
        replay = _execute(client, "run_q_xp_b", "q_xp", prepared_b, key)
    finally:
        backend.enqueue = original
    assert replay.status_code == 422, replay.text
    assert replay.json()["error"]["code"] == "idempotency_key_reused"


def test_idempotency_key_cross_run_same_project_is_typed_422(
    app: FastAPI, client: TestClient, workspace: Path
) -> None:
    """Two runs in one project share a question_id; run2 must not replay run1's
    job via run1's idempotency key (review C)."""
    store = ArtifactStore(workspace)
    _seed_candidate_run(store, "demo", "run_q_xr_1", [_template_candidate("q_cross")])
    _seed_candidate_run(store, "demo", "run_q_xr_2", [_template_candidate("q_cross")])

    backend = app.state.job_service._backend
    original = backend.enqueue
    backend.enqueue = lambda command: JobRef(job_id=command.job_id, pid=None)
    key_run1 = uuid.uuid4().hex
    try:
        prepared_1 = _prepare(client, "run_q_xr_1", "q_cross")
        first = _execute(client, "run_q_xr_1", "q_cross", prepared_1, key_run1)
        assert first.status_code == 201, first.text

        # run2 legitimately consumes its own approval first...
        prepared_2 = _prepare(client, "run_q_xr_2", "q_cross")
        own = _execute(client, "run_q_xr_2", "q_cross", prepared_2, uuid.uuid4().hex)
        assert own.status_code == 201, own.text

        # ...then replays run1's key: same project, same question, wrong run.
        replay = _execute(client, "run_q_xr_2", "q_cross", prepared_2, key_run1)
    finally:
        backend.enqueue = original
    assert replay.status_code == 422, replay.text
    body = replay.json()["error"]
    assert body["code"] == "idempotency_key_reused"
    assert "different request content" in body["message"]


def test_cleaning_kind_approval_rejected_on_questions(
    client: TestClient, workspace: Path
) -> None:
    """A consumed-elsewhere kind must not cross over: a cleaning approval on
    the same run reads as not-found for question execution."""
    store = ArtifactStore(workspace)
    session_id = "run_q_kind_probe"
    _seed_candidate_run(store, "demo", session_id, [_template_candidate("q_kind")])
    digest, token, _expires = ApprovalService(store).register(
        kind="cleaning_apply",
        session_id=session_id,
        project_id="demo",
        action={"type": "cleaning_probe"},
        payload={"question_id": "q_kind"},
    )
    response = client.post(
        f"/api/v1/sessions/{session_id}/questions/q_kind/execute",
        json={"action_hash": digest, "approval_token": token},
    )
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "approval_not_found"


def test_prepare_and_execute_refused_while_source_run_busy(
    client: TestClient, workspace: Path
) -> None:
    store = ArtifactStore(workspace)
    session_id = "run_q_busy"
    _seed_candidate_run(store, "demo", session_id, [_template_candidate("q_busy")])
    prepared = _prepare(client, session_id, "q_busy")
    blocker = store.create_job(
        job_id=f"job_{uuid.uuid4().hex[:12]}",
        session_id=session_id,
        project_id="demo",
        kind="auto_eda",
        idempotency_key=None,
    )
    try:
        again = client.post(
            f"/api/v1/sessions/{session_id}/questions/q_busy/prepare", json={"llm": "offline"}
        )
        assert again.status_code == 409, again.text
        assert again.json()["error"]["code"] == "question_session_busy"

        executed = _execute(client, session_id, "q_busy", prepared, uuid.uuid4().hex)
        assert executed.status_code == 409, executed.text
        assert executed.json()["error"]["code"] == "question_session_busy"
    finally:
        store.mark_job_status(str(blocker["job_id"]), "failed")


def test_enqueue_failure_frees_idempotency_key_for_retry(
    app: FastAPI, client: TestClient, workspace: Path
) -> None:
    """Review D: after an enqueue failure the same key must retry as a fresh
    execution (201), not 409 against the dead row."""
    store = ArtifactStore(workspace)
    session_id = "run_q_enqueue_fail"
    _seed_candidate_run(store, "demo", session_id, [_template_candidate("q_retry")])
    prepared = _prepare(client, session_id, "q_retry")

    plain_client = TestClient(app, raise_server_exceptions=False)
    backend = app.state.job_service._backend
    original = backend.enqueue
    key = uuid.uuid4().hex

    def boom(command):
        raise RuntimeError("enqueue exploded")

    backend.enqueue = boom
    try:
        failed = plain_client.post(
            f"/api/v1/sessions/{session_id}/questions/q_retry/execute",
            json={
                "action_hash": prepared["action_hash"],
                "approval_token": prepared["approval_token"],
            },
            headers={"Idempotency-Key": key},
        )
    finally:
        backend.enqueue = original
    assert failed.status_code == 500, failed.text

    backend.enqueue = lambda command: JobRef(job_id=command.job_id, pid=None)
    try:
        retried = _execute(client, session_id, "q_retry", prepared, key)
    finally:
        backend.enqueue = original
    assert retried.status_code == 201, retried.text


def test_runner_fails_unknown_job_kind(workspace: Path) -> None:
    store = ArtifactStore(workspace)
    job = store.create_job(
        job_id=f"job_{uuid.uuid4().hex[:12]}",
        session_id="run_kind_probe",
        project_id="demo",
        kind="bogus_kind",
        idempotency_key=None,
    )
    run_claimed_job(workspace, str(job["job_id"]), "{}")
    final = store.get_job(str(job["job_id"]))
    assert final is not None
    assert final["status"] == "failed"
    assert final["error_code"] == "ValueError"


class TestFailureHeadline:
    """A failed card must say what blocked it, not just that it failed."""

    GUARD_ERROR = (
        "Tool guard rejected parameters for `execute_question_candidate`.\n"
        "What was wrong:\n"
        "- `plan.sql` got 'SQL containing a JOIN with no declared "
        "required_relations': SQL joins tables but the question declares no "
        "required_relations.\n"
        "Allowed:\n"
        "- `plan.sql`: JOIN SQL only for questions declaring confirmed relations.\n"
        "How to fix:\n"
        "- `plan.sql`: Declare required_relations."
    )

    def test_prefers_the_cause_over_the_generic_first_line(self) -> None:
        # The first line names the mechanism; the reader needs the cause.
        assert _failure_headline(self.GUARD_ERROR) == (
            "SQL joins tables but the question declares no required_relations."
        )

    def test_falls_back_to_the_first_line_without_a_cause_section(self) -> None:
        assert _failure_headline("Timed out after 180s.") == "Timed out after 180s."

    def test_reads_a_bare_bullet_when_it_has_no_guard_prefix(self) -> None:
        error = "head\nWhat was wrong:\n- the table was empty\nAllowed:\n- rows"
        assert _failure_headline(error) == "the table was empty"

    @pytest.mark.parametrize("error", [None, "", "   \n  \n", 42, {"a": 1}])
    def test_returns_none_when_there_is_nothing_to_say(self, error: object) -> None:
        assert _failure_headline(error) is None

    def test_truncates_a_runaway_cause(self) -> None:
        headline = _failure_headline("x" * 900)
        assert headline is not None
        assert len(headline) == 240
