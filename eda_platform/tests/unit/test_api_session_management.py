"""Session management: run search (§10.1) and run deletion (§7.1)."""

from __future__ import annotations

import base64
import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.application.services.session_service import SessionService
from eda_platform.core.session_deletion import (
    AUDIT_SESSION_ID,
    SessionDeletionBlockedError,
    SessionDeletionBusyError,
    SessionDeletionCoordinator,
    SessionDeletionRetryableError,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.sessions import SessionManifest

RUNS = {
    "run_orders": "Orders overview",
    "run_churn": "Churn deep dive",
    "run_pct": "Revenue 50% drop",
    "run_under": "a_b margins",
}


class InjectedCrash(RuntimeError):
    pass


class _CrashOnce:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.op_id: str | None = None

    def __call__(self, stage: str, op_id: str, _ordinal: int | None) -> None:
        if self.op_id is None and stage == self.stage:
            self.op_id = op_id
            raise InjectedCrash(stage)


class _FailingDeletion:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def delete(self, _session_id: str) -> None:
        raise self.error


def _seed_run(store: ArtifactStore, session_id: str, title: str, day: int) -> None:
    store.start_session("demo", session_id)
    store.write_manifest(
        SessionManifest(
            session_id=session_id,
            project_id="demo",
            input_hashes={f"{session_id}.csv": "abc"},
            code_version="v1",
            created_at=datetime(2026, 7, day, tzinfo=UTC),
            title=title,
        )
    )
    store.mark_session_status("demo", session_id, "completed")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    for day, (session_id, title) in enumerate(RUNS.items(), start=1):
        _seed_run(store, session_id, title, day)
    return tmp_path


@pytest.fixture
def client(workspace: Path) -> TestClient:
    return TestClient(create_app(workspace))


def _ids(client: TestClient, **params: str) -> list[str]:
    body = client.get("/api/v1/projects/demo/sessions", params=params).json()
    return [run["session_id"] for run in body["items"]]


def test_search_matches_title(client: TestClient) -> None:
    assert _ids(client, q="churn") == ["run_churn"]


def test_search_is_case_insensitive(client: TestClient) -> None:
    assert _ids(client, q="ChUrN") == ["run_churn"]


def test_search_matches_session_id(client: TestClient) -> None:
    assert _ids(client, q="run_orders") == ["run_orders"]


def test_search_matches_dataset_name(client: TestClient) -> None:
    assert _ids(client, q="run_churn.csv") == ["run_churn"]


def test_search_escapes_percent(client: TestClient) -> None:
    """A literal % must not behave as a LIKE wildcard matching every run."""
    assert _ids(client, q="50%") == ["run_pct"]
    # Unescaped, "%" is LIKE's match-anything; escaped it only finds the title
    # that literally contains a percent sign.
    assert _ids(client, q="%") == ["run_pct"]


def test_search_escapes_underscore(client: TestClient) -> None:
    """`_` is LIKE's single-char wildcard; unescaped, "a_b" would also match run ids."""
    assert _ids(client, q="a_b") == ["run_under"]
    assert _ids(client, q="a?b") == []


def test_blank_search_returns_everything(client: TestClient) -> None:
    assert sorted(_ids(client, q="   ")) == sorted(RUNS)


def test_search_paginates(client: TestClient) -> None:
    body = client.get("/api/v1/projects/demo/sessions", params={"q": "run", "limit": 2}).json()
    assert len(body["items"]) == 2
    assert body["next_cursor"]
    page2 = client.get(
        "/api/v1/projects/demo/sessions",
        params={"q": "run", "limit": 2, "cursor": body["next_cursor"]},
    ).json()
    assert len(page2["items"]) == 2
    assert page2["next_cursor"] is None


def test_run_cursor_is_bound_to_project_search_and_derived_toggle(
    client: TestClient, workspace: Path
) -> None:
    first = client.get("/api/v1/projects/demo/sessions", params={"q": "run", "limit": 1}).json()
    cursor = first["next_cursor"]
    assert cursor

    store = ArtifactStore(workspace)
    store.ensure_project("other", "Other")
    cross_project = client.get(
        "/api/v1/projects/other/sessions",
        params={"q": "run", "limit": 1, "cursor": cursor},
    )
    cross_search = client.get(
        "/api/v1/projects/demo/sessions",
        params={"q": "churn", "limit": 1, "cursor": cursor},
    )
    cross_toggle = client.get(
        "/api/v1/projects/demo/sessions",
        params={
            "q": "run",
            "include_derived": "true",
            "limit": 1,
            "cursor": cursor,
        },
    )
    for response in (cross_project, cross_search, cross_toggle):
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_cursor"

    # Canonical query normalization means harmless surrounding whitespace does
    # not create a different pagination identity.
    normalized = client.get(
        "/api/v1/projects/demo/sessions",
        params={"q": "  run  ", "limit": 1, "cursor": cursor},
    )
    assert normalized.status_code == 200


def test_run_cursor_rejects_legacy_and_mutated_filter_binding(
    client: TestClient,
) -> None:
    page = client.get("/api/v1/projects/demo/sessions", params={"limit": 1}).json()
    cursor = page["next_cursor"]
    decoded = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
    decoded["f"] = "0" * len(decoded["f"])
    mutated = base64.urlsafe_b64encode(json.dumps(decoded).encode("utf-8")).decode("ascii")
    legacy = base64.urlsafe_b64encode(
        json.dumps({"c": decoded["c"], "r": decoded["r"]}).encode("utf-8")
    ).decode("ascii")

    for candidate in (mutated, legacy):
        response = client.get(
            "/api/v1/projects/demo/sessions",
            params={"limit": 1, "cursor": candidate},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_cursor"


def test_delete_run_removes_it_everywhere(client: TestClient, workspace: Path) -> None:
    response = client.delete("/api/v1/sessions/run_churn")
    assert response.status_code == 200
    assert response.json() == {
        "session_id": "run_churn",
        "project_id": "demo",
        "deleted": True,
    }
    assert client.get("/api/v1/sessions/run_churn").status_code == 404
    assert "run_churn" not in _ids(client)
    assert not (workspace / "projects" / "demo" / "sessions" / "run_churn").exists()


def test_delete_project_removes_its_workspace_and_index(
    client: TestClient, workspace: Path
) -> None:
    response = client.delete("/api/v1/projects/demo/self")

    assert response.status_code == 200
    assert response.json() == {"project_id": "demo", "deleted": True}
    assert client.get("/api/v1/projects").json() == []
    assert not (workspace / "projects" / "demo").exists()
    assert client.get("/api/v1/sessions/run_churn").status_code == 404


def test_delete_writes_an_audit_event(client: TestClient, workspace: Path) -> None:
    client.delete("/api/v1/sessions/run_churn")
    events = ArtifactStore(workspace).list_trace_events(
        project_id="demo", session_id=AUDIT_SESSION_ID
    )
    deleted = [event for event in events if event.event_type == "session.deleted"]
    assert [event.name for event in deleted] == ["run_churn"]
    assert deleted[0].summary["project_id"] == "demo"
    trace_file = workspace / "projects" / "demo" / "sessions" / AUDIT_SESSION_ID / "trace.jsonl"
    assert "run_churn" in trace_file.read_text(encoding="utf-8")


def test_audit_stream_is_hidden_from_run_listings(client: TestClient) -> None:
    client.delete("/api/v1/sessions/run_churn")
    assert AUDIT_SESSION_ID not in _ids(client)
    assert AUDIT_SESSION_ID not in _ids(client, include_derived="true")
    assert client.get("/api/v1/projects").json()[0]["session_count"] == len(RUNS) - 1


def test_delete_unknown_run_is_404(client: TestClient) -> None:
    response = client.delete("/api/v1/sessions/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_delete_is_not_idempotent_but_stays_typed(client: TestClient) -> None:
    assert client.delete("/api/v1/sessions/run_churn").status_code == 200
    assert client.delete("/api/v1/sessions/run_churn").status_code == 404


def test_delete_request_resumes_operation_after_runs_row_is_gone(
    client: TestClient, workspace: Path
) -> None:
    crash = _CrashOnce("after_db_commit")
    cast(FastAPI, client.app).state.session_service = SessionService(
        ArtifactStore(workspace),
        SessionDeletionCoordinator(ArtifactStore(workspace), fault_hook=crash),
    )
    with pytest.raises(InjectedCrash):
        client.delete("/api/v1/sessions/run_churn")

    assert crash.op_id is not None
    response = client.delete("/api/v1/sessions/run_churn")
    assert response.status_code == 200
    assert response.json() == {
        "session_id": "run_churn",
        "project_id": "demo",
        "deleted": True,
    }


def test_app_restart_recovers_active_delete_operation(workspace: Path) -> None:
    crash = _CrashOnce("after_reserve")
    first_app = create_app(workspace)
    first_app.state.session_service = SessionService(
        ArtifactStore(workspace),
        SessionDeletionCoordinator(ArtifactStore(workspace), fault_hook=crash),
    )
    with TestClient(first_app) as first:
        with pytest.raises(InjectedCrash):
            first.delete("/api/v1/sessions/run_churn")

    assert crash.op_id is not None
    restarted_app = create_app(workspace)
    with TestClient(restarted_app) as restarted:
        assert restarted.get("/api/v1/sessions/run_churn").status_code == 404
    with sqlite3.connect(workspace / "state.sqlite") as conn:
        state = conn.execute(
            "select state from storage_operations where op_id = ?",
            (crash.op_id,),
        ).fetchone()
    assert state == ("done",)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (SessionDeletionBusyError("run_churn", "job_live"), "remains busy"),
        (
            SessionDeletionRetryableError("del_retry", "temporary IO failure"),
            "remains retryable",
        ),
        (
            SessionDeletionBlockedError("del_blocked", "unsafe filesystem state"),
            "remains blocked",
        ),
    ],
)
def test_startup_preserves_and_logs_nonterminal_delete_outcomes(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    message: str,
) -> None:
    crash = _CrashOnce("after_reserve")
    with pytest.raises(InjectedCrash):
        SessionDeletionCoordinator(ArtifactStore(workspace), fault_hook=crash).delete("run_churn")
    assert crash.op_id is not None

    def fail_recovery(_coordinator: SessionDeletionCoordinator, _op_id: str) -> None:
        raise error

    monkeypatch.setattr(SessionDeletionCoordinator, "recover", fail_recovery)
    with caplog.at_level(logging.WARNING):
        create_app(workspace)

    assert message in caplog.text
    with sqlite3.connect(workspace / "state.sqlite") as conn:
        state = conn.execute(
            "select state from storage_operations where op_id = ?",
            (crash.op_id,),
        ).fetchone()
    assert state == ("prepared",)


@pytest.mark.parametrize(
    ("error", "status_code", "code"),
    [
        (
            SessionDeletionRetryableError("del_retry", "temporary IO failure"),
            503,
            "session_delete_retryable",
        ),
        (
            SessionDeletionBlockedError("del_blocked", "unsafe filesystem state"),
            409,
            "session_delete_blocked",
        ),
    ],
)
def test_delete_maps_coordinator_failures_to_typed_api_errors(
    client: TestClient,
    workspace: Path,
    error: Exception,
    status_code: int,
    code: str,
) -> None:
    service = SessionService(ArtifactStore(workspace))
    service._deletion = _FailingDeletion(error)  # type: ignore[assignment]
    cast(FastAPI, client.app).state.session_service = service

    response = client.delete("/api/v1/sessions/run_churn")

    assert response.status_code == status_code
    assert response.json()["error"]["code"] == code
    if status_code == 503:
        assert response.headers["Retry-After"] == "1"


def test_openapi_declares_delete_conflict_and_retryable_responses(
    client: TestClient,
) -> None:
    operation = client.get("/openapi.json").json()["paths"]["/api/v1/sessions/{session_id}"][
        "delete"
    ]
    assert {"409", "503"} <= operation["responses"].keys()


def test_delete_refuses_while_a_job_is_active(client: TestClient, workspace: Path) -> None:
    ArtifactStore(workspace).create_job(
        job_id="job_live",
        session_id="run_churn",
        project_id="demo",
        kind="auto_eda",
        idempotency_key=None,
    )
    response = client.delete("/api/v1/sessions/run_churn")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "session_busy"
    # Refusal must leave the run intact.
    assert client.get("/api/v1/sessions/run_churn").status_code == 200
    assert (workspace / "projects" / "demo" / "sessions" / "run_churn").is_dir()


def test_delete_refuses_an_active_derived_execution_run(
    client: TestClient, workspace: Path
) -> None:
    store = ArtifactStore(workspace)
    _seed_run(store, "fksess_active", "Active fork lifecycle", 10)
    store.create_job(
        job_id="job_active_fork",
        session_id="fksess_active",
        project_id="demo",
        kind="session_fork",
        lane_key="run_churn",
    )

    response = client.delete("/api/v1/sessions/fksess_active")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "session_busy"
    assert client.get("/api/v1/sessions/fksess_active").status_code == 200


def test_delete_traversal_id_is_rejected_without_touching_disk(
    client: TestClient, workspace: Path
) -> None:
    before = sorted(p.name for p in (workspace / "projects" / "demo" / "sessions").iterdir())
    assert client.delete("/api/v1/sessions/..%2F..%2Fuploads").status_code == 404
    assert (
        sorted(p.name for p in (workspace / "projects" / "demo" / "sessions").iterdir()) == before
    )
