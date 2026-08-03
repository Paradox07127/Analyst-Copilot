"""F-006 generated mutation matrix plus shared replay/version primitives."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eda_platform.api import mutation_contract
from eda_platform.api.main import create_app
from eda_platform.api.mutation_contract import (
    IDEMPOTENT_OPERATIONS,
    INTRINSIC_OPERATIONS,
    MAX_REPLAY_BODY_BYTES,
    VERSIONED_OPERATIONS,
    MutationContractMiddleware,
    MutationRoute,
)

UNSAFE = {"post", "put", "patch", "delete"}


def _operations(client: TestClient) -> list[dict]:
    schema = client.get("/openapi.json").json()
    return [
        operation
        for path_item in schema["paths"].values()
        for method, operation in path_item.items()
        if method in UNSAFE
    ]


def _request_schema(schema: dict, operation: dict) -> dict:
    body = operation["requestBody"]["content"]["application/json"]["schema"]
    if "$ref" not in body:
        return body
    name = body["$ref"].rsplit("/", 1)[-1]
    return schema["components"]["schemas"][name]


def test_every_openapi_mutation_has_one_generated_policy(tmp_path: Path) -> None:
    operations = _operations(TestClient(create_app(tmp_path)))
    # The reviewed matrix contained 53 operations. Client-failure ingestion was
    # added afterwards (54), then project deletion (55), upload deletion (56),
    # the model-catalog refresh (57), then project and session rename (59),
    # then project reorder (60), then user skill templates (create/delete/
    # import, 63), then the six exploration lifecycle mutations
    # (prepare/start/pause/resume/cancel/extend-budget, 69).
    assert len(operations) == 69
    operation_ids = {operation["operationId"] for operation in operations}
    assert operation_ids == (
        IDEMPOTENT_OPERATIONS | VERSIONED_OPERATIONS | INTRINSIC_OPERATIONS
    )
    assert not (IDEMPOTENT_OPERATIONS & VERSIONED_OPERATIONS)
    assert not (IDEMPOTENT_OPERATIONS & INTRINSIC_OPERATIONS)
    assert not (VERSIONED_OPERATIONS & INTRINSIC_OPERATIONS)
    assert {
        operation["x-mutation-policy"] for operation in operations
    } == {"idempotency", "version", "intrinsic"}
    assert all(
        operation["x-mutation-policy"] in {"idempotency", "version", "intrinsic"}
        for operation in operations
    )
    for operation in operations:
        if operation["x-mutation-policy"] != "idempotency":
            continue
        assert any(
            parameter["in"] == "header"
            and parameter["name"].lower() == "idempotency-key"
            for parameter in operation.get("parameters", [])
        ), operation["operationId"]


def test_every_version_policy_exposes_a_client_concurrency_token(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(tmp_path))
    schema = client.get("/openapi.json").json()
    for path_item in schema["paths"].values():
        for method, operation in path_item.items():
            if (
                method not in UNSAFE
                or operation.get("x-mutation-policy") != "version"
            ):
                continue
            if any(
                parameter["in"] == "header"
                and parameter["name"].lower() == "if-match"
                for parameter in operation.get("parameters", [])
            ):
                continue
            request_schema = _request_schema(schema, operation)
            assert "expected_version" in request_schema.get("required", []), operation[
                "operationId"
            ]


def test_content_bound_project_create_replays_original_response(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    headers = {"Idempotency-Key": "fixed-project-create"}
    first = client.post(
        "/api/v1/projects",
        json={"project_id": "demo", "name": "Demo"},
        headers=headers,
    )
    replay = client.post(
        "/api/v1/projects",
        json={"project_id": "demo", "name": "Demo"},
        headers=headers,
    )
    conflict = client.post(
        "/api/v1/projects",
        json={"project_id": "other", "name": "Other"},
        headers=headers,
    )

    assert first.status_code == 201
    assert replay.status_code == 201
    assert replay.content == first.content
    assert conflict.status_code == 422
    assert conflict.json()["error"]["code"] == "idempotency_key_reused"
    assert client.get("/api/v1/projects").json() == [
        {"project_id": "demo", "name": "Demo", "session_count": 0}
    ]

    with sqlite3.connect(tmp_path / "state.sqlite") as conn:
        row = conn.execute(
            """
            select state, status_code from mutation_replays
            where operation_id = ? and idempotency_key = ?
            """,
            ("create_project_api_v1_projects_post", "fixed-project-create"),
        ).fetchone()
        audits = conn.execute(
            """
            select operation_id, request_digest, outcome_status
            from mutation_audit
            where operation_id = ?
            """,
            ("create_project_api_v1_projects_post",),
        ).fetchall()
    assert row == ("completed", 201)
    assert len(audits) == 1
    assert audits[0][1] and audits[0][2] == 201


def test_concurrent_same_key_is_not_executed_twice(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original = app.state.session_service.create_project

    def slow_create(project_id: str, name: str = ""):
        entered.set()
        assert release.wait(timeout=5)
        return original(project_id, name)

    app.state.session_service.create_project = slow_create
    first_client = TestClient(app)
    second_client = TestClient(app)
    headers = {"Idempotency-Key": "concurrent-project-create"}
    payload = {"project_id": "demo", "name": "Demo"}

    with ThreadPoolExecutor(max_workers=1) as executor:
        first_future = executor.submit(
            first_client.post,
            "/api/v1/projects",
            json=payload,
            headers=headers,
        )
        assert entered.wait(timeout=5)
        concurrent = second_client.post(
            "/api/v1/projects", json=payload, headers=headers
        )
        release.set()
        first = first_future.result(timeout=5)

    assert first.status_code == 201
    assert concurrent.status_code == 409
    assert concurrent.headers["Retry-After"] == "1"
    assert concurrent.json()["error"]["code"] == "mutation_in_progress"
    replay = second_client.post(
        "/api/v1/projects", json=payload, headers=headers
    )
    assert replay.status_code == 201
    assert replay.content == first.content


def test_replay_bookkeeping_closes_every_sqlite_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``with sqlite3.connect(...)`` commits but never closes, so the middleware
    must own the handle explicitly or leak one per mutation request.

    Probing a connection from the test thread cannot answer this: SQLite raises
    ProgrammingError for a foreign thread too, which reads as a false pass.
    Count close() calls instead.
    """
    opened = 0
    closed = 0

    class TrackedConnection(sqlite3.Connection):
        def close(self) -> None:
            nonlocal closed
            closed += 1
            super().close()

    real_connect = sqlite3.connect

    def tracking_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        # Attribute by call site, not by helper, so a site that bypasses
        # ``_connect`` is still counted.
        nonlocal opened
        caller = traceback.extract_stack()[-2]
        if caller.filename.endswith("mutation_contract.py"):
            kwargs["factory"] = TrackedConnection
            opened += 1
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(mutation_contract.sqlite3, "connect", tracking_connect)
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/v1/projects",
        json={"project_id": "demo", "name": "Demo"},
        headers={"Idempotency-Key": "connection-close-check"},
    )

    assert response.status_code == 201
    assert opened >= 2, "expected at least the reserve and finish transactions"
    assert closed == opened


def test_replay_bookkeeping_runs_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both bookkeeping calls take a ``begin immediate`` write lock; running them
    on the loop thread stalls every other in-flight request."""
    on_loop: list[bool] = []
    original_reserve = MutationContractMiddleware._reserve
    original_finish = MutationContractMiddleware._finish

    def has_running_loop() -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    def reserve(self: MutationContractMiddleware, *args: object, **kwargs: object):
        on_loop.append(has_running_loop())
        return original_reserve(self, *args, **kwargs)  # type: ignore[arg-type]

    def finish(self: MutationContractMiddleware, *args: object, **kwargs: object):
        on_loop.append(has_running_loop())
        return original_finish(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(MutationContractMiddleware, "_reserve", reserve)
    monkeypatch.setattr(MutationContractMiddleware, "_finish", finish)
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/v1/projects",
        json={"project_id": "demo", "name": "Demo"},
        headers={"Idempotency-Key": "off-loop-check"},
    )

    assert response.status_code == 201
    assert len(on_loop) == 2
    assert not any(on_loop)


def test_replayable_body_over_the_cap_is_refused_before_buffering(
    tmp_path: Path,
) -> None:
    """A keyed replay must buffer the body; the buffer needs its own ceiling
    instead of inheriting the upload-sized global limit."""
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/v1/projects",
        json={"project_id": "demo", "name": "x" * (MAX_REPLAY_BODY_BYTES + 1)},
        headers={"Idempotency-Key": "oversized-body"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_route_template_literals_are_not_regex_metacharacters() -> None:
    route = MutationRoute(
        operation_id="probe",
        method="POST",
        template="/api/v1/a.b/{item_id}",
        policy="intrinsic",
    )

    assert route.matches("POST", "/api/v1/a.b/1")
    assert not route.matches("POST", "/api/v1/axb/1")
    assert not route.matches("POST", "/api/v1/a.b/1/2")


def test_audit_failure_does_not_mask_the_application_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_finish` runs in a finally block, so an audit write that raises would
    replace the real error and send the operator chasing the wrong stack."""

    class AuditUnavailable(RuntimeError):
        pass

    def failing_finish(self: MutationContractMiddleware, **_kwargs: object) -> None:
        raise AuditUnavailable("audit table is locked")

    app = create_app(tmp_path)

    def exploding_create(project_id: str, name: str = ""):
        raise ValueError("the real application error")

    app.state.session_service.create_project = exploding_create
    monkeypatch.setattr(MutationContractMiddleware, "_finish", failing_finish)
    client = TestClient(app, raise_server_exceptions=True)

    with pytest.raises(ValueError, match="the real application error"):
        client.post(
            "/api/v1/projects",
            json={"project_id": "demo", "name": "Demo"},
            headers={"Idempotency-Key": "audit-failure"},
        )
