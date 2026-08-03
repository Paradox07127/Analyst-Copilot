"""HTTP and SSE contracts for the certified E4b exploration API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from exploration_test_helpers import (
    TEST_RUNTIME_IDENTITY,
    TEST_TRUSTED_RELEASE_PUBLIC_KEYS,
    release_certificate,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.application.ports import JobCommand, JobRef
from eda_platform.application.services.approval_service import ApprovalService
from eda_platform.application.services.exploration_service import (
    ExplorationService,
    ExplorationSourceSnapshot,
)
from eda_platform.application.services.job_service import JobService
from eda_platform.application.services.settings_service import SettingsService
from eda_platform.core.exploration_journal import JsonlExplorationJournal
from eda_platform.core.exploration_shadow_store import shadow_run_root
from eda_platform.core.llm import LLMSettings
from eda_platform.core.provider_registry import LLMProvider
from eda_platform.core.store import ArtifactStore

SOURCE = "run_api_source"
PROJECT = "demo"
DATASET = "ds_orders"
WITNESS = "dsw1_" + "c" * 32


class _RecordingBackend:
    def __init__(self, store: ArtifactStore) -> None:
        self.store = store
        self.commands: list[JobCommand] = []

    def enqueue(self, command: JobCommand) -> JobRef:
        self.commands.append(command)
        return JobRef(job_id=command.job_id)

    def cancel(self, job_id: str) -> None:
        self.store.request_cancel(job_id)

    def status(self, job_id: str) -> str:
        return "queued"


@dataclass(frozen=True)
class _ApiFixture:
    app: FastAPI
    client: TestClient
    store: ArtifactStore
    backend: _RecordingBackend


@pytest.fixture
def api(tmp_path: Path) -> _ApiFixture:
    app = create_app(tmp_path)
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    backend = _RecordingBackend(store)

    def source(
        session_id: str, dataset_ids: tuple[str, ...]
    ) -> ExplorationSourceSnapshot:
        assert session_id == SOURCE
        assert dataset_ids == (DATASET,)
        return ExplorationSourceSnapshot(PROJECT, dataset_ids, WITNESS)

    app.state.exploration_service = ExplorationService(
        store,
        ApprovalService(store),
        JobService(store, backend),
        release_certificate=release_certificate(),
        trusted_release_public_keys=TEST_TRUSTED_RELEASE_PUBLIC_KEYS,
        trusted_runtime_identity=TEST_RUNTIME_IDENTITY,
        source_snapshot_resolver=source,
    )
    app.state.settings_service = SettingsService(
        workspace=tmp_path.resolve(),
        defaults=LLMSettings(
            provider=LLMProvider.OPENAI,
            api_key="api-test-secret",
            model="gpt-5.6-terra",
        ),
    )
    return _ApiFixture(app, TestClient(app), store, backend)


def _prepare_start(api: _ApiFixture) -> tuple[dict, dict]:
    prepared_response = api.client.post(
        f"/api/v1/sessions/{SOURCE}/explorations/prepare",
        json={
            "mode": "open",
            "goal": None,
            "dataset_ids": [DATASET],
            "thinking_level": "quick",
        },
    )
    assert prepared_response.status_code == 200, prepared_response.text
    prepared = prepared_response.json()
    started_response = api.client.post(
        f"/api/v1/sessions/{SOURCE}/explorations",
        headers={"Idempotency-Key": "api-start-key"},
        json={
            "action_hash": prepared["action_hash"],
            "approval_token": prepared["approval_token"],
        },
    )
    assert started_response.status_code == 201, started_response.text
    return prepared, started_response.json()


@pytest.mark.parametrize(
    ("method", "path", "body", "headers"),
    [
        (
            "post",
            f"/api/v1/sessions/{SOURCE}/explorations/prepare",
            {
                "mode": "open",
                "dataset_ids": [DATASET],
                "thinking_level": "quick",
            },
            {},
        ),
        (
            "post",
            f"/api/v1/sessions/{SOURCE}/explorations",
            {"action_hash": "12345678", "approval_token": "12345678"},
            {},
        ),
        ("get", f"/api/v1/sessions/{SOURCE}/explorations/expl_missing", None, {}),
        (
            "post",
            f"/api/v1/sessions/{SOURCE}/explorations/expl_missing/pause",
            {},
            {},
        ),
        (
            "post",
            f"/api/v1/sessions/{SOURCE}/explorations/expl_missing/resume",
            {},
            {},
        ),
        (
            "post",
            f"/api/v1/sessions/{SOURCE}/explorations/expl_missing/cancel",
            {},
            {},
        ),
        (
            "post",
            f"/api/v1/sessions/{SOURCE}/explorations/expl_missing/extend-budget",
            {"increase": {"max_rounds": 1}, "reason": "more coverage"},
            {"Idempotency-Key": "amend-key"},
        ),
        (
            "get",
            f"/api/v1/sessions/{SOURCE}/explorations/expl_missing/events",
            None,
            {},
        ),
    ],
)
def test_every_entry_is_closed_without_a_release_certificate(
    tmp_path: Path,
    method: str,
    path: str,
    body: dict | None,
    headers: dict[str, str],
) -> None:
    client = TestClient(create_app(tmp_path))
    response = client.request(method, path, json=body, headers=headers)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "exploration_release_unavailable"


def test_prepare_start_get_and_control_routes_project_journal_state(
    api: _ApiFixture,
) -> None:
    prepared, started = _prepare_start(api)
    exploration_id = prepared["exploration_id"]
    assert started["exploration"]["status"] == "running"
    assert started["exploration"]["goal"] == "Explore freely"
    assert started["exploration"]["thinking_level"] == "quick"
    assert started["exploration"]["current_evidence"] == []
    assert started["exploration"]["insights"] == []
    assert started["exploration"]["coverage_targets"] == []
    assert started["exploration"]["report"]["available"] is False
    assert started["exploration"]["budget"]["cost_usd"] == "0"

    paused = api.client.post(
        f"/api/v1/sessions/{SOURCE}/explorations/{exploration_id}/pause",
        json={},
    )
    assert paused.status_code == 200
    assert paused.json()["status"] == "pause_requested"
    assert paused.json()["stop_reason"] is None

    extended = api.client.post(
        f"/api/v1/sessions/{SOURCE}/explorations/{exploration_id}/extend-budget",
        headers={"Idempotency-Key": "api-amend-key"},
        json={"increase": {"max_rounds": 2}, "reason": "cover more segments"},
    )
    assert extended.status_code == 200, extended.text
    assert extended.json()["amendment"]["approved_by"] == "system:e4b-api"
    assert extended.json()["exploration"]["status"] == "pause_requested"

    cancelled = api.client.post(
        f"/api/v1/sessions/{SOURCE}/explorations/{exploration_id}/cancel",
        json={},
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "stopped"
    assert cancelled.json()["stop_reason"] == "cancelled"

    view = api.client.get(
        f"/api/v1/sessions/{SOURCE}/explorations/{exploration_id}"
    )
    assert view.status_code == 200
    assert view.json()["last_seq"] == 3


def test_sse_reconnect_uses_composite_ids_without_duplicate_job_or_frame(
    api: _ApiFixture,
) -> None:
    prepared, _started = _prepare_start(api)
    exploration_id = prepared["exploration_id"]
    journal = JsonlExplorationJournal(
        shadow_run_root(api.store.root, exploration_id) / "journal.jsonl"
    )
    journal.append_new(
        "exploration_stopped", stop_reason="cancelled", final_report_ref=None
    )

    path = f"/api/v1/sessions/{SOURCE}/explorations/{exploration_id}/events"
    initial = api.client.get(path)
    assert initial.status_code == 200
    assert f"id: {exploration_id}:0\n" in initial.text
    assert f"id: {exploration_id}:1\n" in initial.text

    replay = api.client.get(path, headers={"Last-Event-ID": f"{exploration_id}:0"})
    assert replay.status_code == 200
    assert f"id: {exploration_id}:0\n" not in replay.text
    assert replay.text.count(f"id: {exploration_id}:1\n") == 1
    assert len(api.backend.commands) == 1

    exhausted = api.client.get(
        path, headers={"Last-Event-ID": f"{exploration_id}:1"}
    )
    assert exhausted.status_code == 204
    assert len(api.backend.commands) == 1


def test_sse_rejects_foreign_or_malformed_composite_cursor(api: _ApiFixture) -> None:
    prepared, _started = _prepare_start(api)
    exploration_id = prepared["exploration_id"]
    path = f"/api/v1/sessions/{SOURCE}/explorations/{exploration_id}/events"
    for cursor in ("expl_other:0", f"{exploration_id}:oops", "1"):
        response = api.client.get(path, headers={"Last-Event-ID": cursor})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "exploration_invalid"


def test_client_cannot_supply_resume_or_amendment_fingerprints(api: _ApiFixture) -> None:
    prepared, _started = _prepare_start(api)
    exploration_id = prepared["exploration_id"]
    resume = api.client.post(
        f"/api/v1/sessions/{SOURCE}/explorations/{exploration_id}/resume",
        json={"effective_policy_fingerprint": "client-forged"},
    )
    assert resume.status_code == 422

    amendment = api.client.post(
        f"/api/v1/sessions/{SOURCE}/explorations/{exploration_id}/extend-budget",
        headers={"Idempotency-Key": "forged-amendment"},
        json={
            "increase": {"max_rounds": 1},
            "reason": "more",
            "amendment_id": "client-choice",
            "effective_policy_fingerprint": "client-forged",
        },
    )
    assert amendment.status_code == 422


def test_openapi_registers_all_exploration_routes_and_projection_fields(
    api: _ApiFixture,
) -> None:
    schema = api.app.openapi()
    paths = schema["paths"]
    prefix = "/api/v1/sessions/{session_id}/explorations"
    assert {
        prefix + suffix
        for suffix in (
            "/prepare",
            "",
            "/{exploration_id}",
            "/{exploration_id}/pause",
            "/{exploration_id}/resume",
            "/{exploration_id}/cancel",
            "/{exploration_id}/extend-budget",
            "/{exploration_id}/events",
        )
    }.issubset(paths)
    properties = schema["components"]["schemas"]["ExplorationView"]["properties"]
    assert {
        "current_hypothesis",
        "current_evidence",
        "insights",
        "coverage_targets",
        "coverage_completed",
        "coverage_unexplored",
        "report",
        "budget",
    }.issubset(properties)
