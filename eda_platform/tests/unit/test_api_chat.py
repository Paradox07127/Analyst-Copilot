"""Chat API vertical slice: reverse transcript pagination, accept-then-stream
turns against the real driver in offline mode, and the plan-approval gate —
a plan that needs approval must not execute until an approval is consumed."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, TypeVar, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from eda_platform.api.main import create_app
from eda_platform.application.services import chat_service as chat_service_module
from eda_platform.application.services.approval_service import ApprovalService
from eda_platform.core.bounded_pagination import (
    MAX_JSONL_RECORD_BYTES,
    JsonlPageIndex,
)
from eda_platform.core.budget import SessionBudgetPolicy
from eda_platform.core.llm import LLMResultMetadata, LLMSettings, LLMUsage
from eda_platform.core.llm_ledger import LLM_USAGE_EVENT
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile
from eda_platform.schemas.chat import ChatMessage, ChatTurnResult
from eda_platform.schemas.plans import AnalysisPlan, Intent
from eda_platform.schemas.session_metrics import SessionMetrics

PROJECT = "proj_chat"
RUN = "run_chat"
DATASET_ID = "ds_orders"
DATASET_NAME = "orders.csv"

_STREAM_TIMEOUT_SECONDS = 60.0
T = TypeVar("T", bound=BaseModel)


def _csv() -> str:
    rows = ["region,amount"]
    for index in range(20):
        amount = "" if index % 5 == 0 else str(100 + index)
        rows.append(f"r{index % 4},{amount}")
    return "\n".join(rows) + "\n"


def _profile() -> DatasetProfile:
    return DatasetProfile(
        dataset_id=DATASET_ID,
        name=DATASET_NAME,
        rows=20,
        columns=2,
        column_names=["region", "amount"],
        dtypes={"region": "object", "amount": "float64"},
        missing_values={"region": 0, "amount": 4},
        missing_percent={"region": 0.0, "amount": 20.0},
        numeric_columns=["amount"],
        categorical_columns=["region"],
    )


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, "Chat project")
    store.start_session(PROJECT, RUN)
    upload = tmp_path / "projects" / PROJECT / "uploads" / DATASET_ID / "v1" / DATASET_NAME
    upload.parent.mkdir(parents=True, exist_ok=True)
    upload.write_text(_csv(), encoding="utf-8")
    store.save_artifact(
        Artifact(
            id="prof_chat_orders",
            type=ArtifactType.DATASET_PROFILE,
            project_id=PROJECT,
            session_id=RUN,
            payload=_profile().model_dump(),
        )
    )
    store.refresh_session_index(PROJECT, RUN)
    return tmp_path


@pytest.fixture()
def client(workspace: Path) -> TestClient:
    app: FastAPI = create_app(workspace)
    return TestClient(app)


def _seed_transcript(workspace: Path, count: int) -> None:
    path = workspace / "projects" / PROJECT / "chat" / f"{RUN}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        ChatMessage(
            role="user" if index % 2 == 0 else "assistant",
            content=f"message {index}",
        ).model_dump_json()
        for index in range(count)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_stream(
    client: TestClient,
    session_id: str,
    message_id: str,
    *,
    last_event_id: int | None = None,
) -> list[dict[str, Any]]:
    """Drain the SSE stream into (seq, type, data) frames until it closes."""
    frames: list[dict[str, Any]] = []
    deadline = time.monotonic() + _STREAM_TIMEOUT_SECONDS
    url = f"/api/v1/sessions/{session_id}/chat/stream?message_id={message_id}"
    headers = {} if last_event_id is None else {"Last-Event-ID": str(last_event_id)}
    with client.stream("GET", url, headers=headers) as response:
        assert response.status_code == 200
        event_type: str | None = None
        for line in response.iter_lines():
            if time.monotonic() > deadline:
                raise AssertionError("Chat stream did not terminate in time.")
            if line.startswith("event: "):
                event_type = line[len("event: ") :]
            elif line.startswith("data: "):
                payload = json.loads(line[len("data: ") :])
                frames.append(
                    {
                        "seq": payload["seq"],
                        "type": event_type,
                        "data": payload["data"],
                    }
                )
    return frames


def _accept_turn(client: TestClient, text: str = "hello") -> dict[str, Any]:
    accepted = client.post(
        f"/api/v1/sessions/{RUN}/chat/messages", json={"text": text, "llm": "offline"}
    )
    assert accepted.status_code == 202, accepted.text
    return accepted.json()


@pytest.fixture()
def echo_driver(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A `run_chat_turn` stand-in that answers immediately; the tests that use
    it care about session/stream bookkeeping, not about the agent."""
    seen: list[str] = []
    seen_lock = threading.Lock()

    def echo(message: str, **kwargs: Any) -> ChatTurnResult:
        with seen_lock:
            seen.append(message)
        return ChatTurnResult(
            intent=Intent(kind="meta_help", confidence=1.0, raw_message=message),
            status="answer",
            message=f"echo: {message}",
        )

    import eda_platform.drivers.chat as chat_driver

    monkeypatch.setattr(chat_driver, "run_chat_turn", echo)
    return seen


def test_messages_page_backwards_from_the_newest(client: TestClient, workspace: Path) -> None:
    _seed_transcript(workspace, 12)

    newest = client.get(f"/api/v1/sessions/{RUN}/chat/messages?limit=5").json()
    assert newest["total"] == 12
    assert [item["content"] for item in newest["messages"]] == [
        "message 7",
        "message 8",
        "message 9",
        "message 10",
        "message 11",
    ]
    assert newest["next_cursor"]

    older = client.get(
        f"/api/v1/sessions/{RUN}/chat/messages?limit=5&cursor={newest['next_cursor']}"
    ).json()
    assert [item["content"] for item in older["messages"]] == [
        "message 2",
        "message 3",
        "message 4",
        "message 5",
        "message 6",
    ]
    assert older["next_cursor"]

    oldest = client.get(
        f"/api/v1/sessions/{RUN}/chat/messages?limit=5&cursor={older['next_cursor']}"
    ).json()
    assert [item["content"] for item in oldest["messages"]] == ["message 0", "message 1"]
    assert oldest["next_cursor"] is None


def test_large_transcript_page_never_uses_read_text(
    client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_transcript(workspace, 5_000)
    transcript = (
        workspace / "projects" / PROJECT / "chat" / f"{RUN}.jsonl"
    ).resolve()
    original = Path.read_text

    def guarded(path: Path, *args: Any, **kwargs: Any) -> str:
        if path.resolve() == transcript:
            raise AssertionError("transcript must be streamed, not materialized")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    body = client.get(f"/api/v1/sessions/{RUN}/chat/messages?limit=3").json()
    assert body["total"] == 5_000
    assert [item["content"] for item in body["messages"]] == [
        "message 4997",
        "message 4998",
        "message 4999",
    ]
    cursor = body["next_cursor"]
    assert cursor

    def no_rescan(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("an unchanged indexed transcript must not be rescanned")

    monkeypatch.setattr(JsonlPageIndex, "_extend_index", no_rescan)
    older = client.get(
        f"/api/v1/sessions/{RUN}/chat/messages",
        params={"limit": 3, "cursor": cursor},
    )
    assert older.status_code == 200


def test_chat_cursor_rejects_changed_source(
    client: TestClient, workspace: Path
) -> None:
    _seed_transcript(workspace, 10)
    first = client.get(f"/api/v1/sessions/{RUN}/chat/messages", params={"limit": 2}).json()
    path = workspace / "projects" / PROJECT / "chat" / f"{RUN}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(ChatMessage(role="user", content="new").model_dump_json() + "\n")
    stale = client.get(
        f"/api/v1/sessions/{RUN}/chat/messages",
        params={"limit": 2, "cursor": first["next_cursor"]},
    )
    assert stale.status_code == 400
    assert stale.json()["error"]["code"] == "invalid_cursor"


def test_chat_cursor_is_bound_to_run(client: TestClient, workspace: Path) -> None:
    _seed_transcript(workspace, 10)
    cursor = client.get(
        f"/api/v1/sessions/{RUN}/chat/messages", params={"limit": 2}
    ).json()["next_cursor"]
    other = "run_chat_other"
    ArtifactStore(workspace).start_session(PROJECT, other)
    path = workspace / "projects" / PROJECT / "chat" / f"{other}.jsonl"
    path.write_text(
        ChatMessage(role="user", content="other").model_dump_json() + "\n",
        encoding="utf-8",
    )
    replay = client.get(
        f"/api/v1/sessions/{other}/chat/messages",
        params={"limit": 2, "cursor": cursor},
    )
    assert replay.status_code == 400
    assert replay.json()["error"]["code"] == "invalid_cursor"


def test_invalid_cursor_is_rejected(client: TestClient, workspace: Path) -> None:
    _seed_transcript(workspace, 3)
    response = client.get(f"/api/v1/sessions/{RUN}/chat/messages?cursor=not-a-number")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_cursor"


def test_offline_turn_streams_events_and_persists_the_answer(
    client: TestClient, workspace: Path
) -> None:
    accepted = client.post(
        f"/api/v1/sessions/{RUN}/chat/messages",
        json={"text": "Which column has the most missing values?", "llm": "offline"},
    )
    assert accepted.status_code == 202
    body = accepted.json()
    # The whole URL, not just its suffix: an endswith() check let the service
    # hand back /api/v1/runs/... after the route moved to /sessions/, so every
    # live turn streamed from a 404 with the suite still green.
    assert body["stream_url"] == (
        f"/api/v1/sessions/{RUN}/chat/stream?message_id={body['message_id']}"
    )
    assert client.get(body["stream_url"]).status_code != 404

    frames = _read_stream(client, RUN, body["message_id"])
    types = [frame["type"] for frame in frames]
    assert types[0] == "turn.started"
    assert "progress" in types
    assert types[-1] == "message.completed"
    # The deterministic artifact answer names the column with missing values.
    completed = frames[-1]["data"]
    assert completed["status"] == "answer"
    assert "amount" in completed["content"]

    transcript = client.get(f"/api/v1/sessions/{RUN}/chat/messages").json()["messages"]
    assert [item["role"] for item in transcript] == ["user", "assistant"]
    assert transcript[1]["content"] == completed["content"]


def test_stream_for_an_unknown_message_id_is_404(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{RUN}/chat/stream?message_id=deadbeef")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "chat_message_not_found"


def test_empty_message_is_rejected(client: TestClient) -> None:
    response = client.post(
        f"/api/v1/sessions/{RUN}/chat/messages", json={"text": "   ", "llm": "offline"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "chat_invalid"


def test_unknown_run_is_404(client: TestClient) -> None:
    response = client.post(
        "/api/v1/sessions/nope/chat/messages", json={"text": "hi", "llm": "offline"}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


class _ScriptedProvider:
    def __init__(
        self,
        settings: LLMSettings,
        responses: list[BaseModel],
    ) -> None:
        self.settings = settings
        self.responses = list(responses)
        self.calls: list[str] = []
        self._last: LLMResultMetadata | None = None

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append(task)
        response = self.responses.pop(0)
        self._last = LLMResultMetadata(
            provider=self.settings.provider.value,
            model=self.settings.model,
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            estimated_cost_usd=0.001,
        )
        return cast(T, response)

    def text(self, *, task: str, payload: dict) -> str:
        raise AssertionError("Chat planning should use structured calls.")

    def last_usage(self) -> LLMResultMetadata | None:
        return self._last


def _new_analysis_intent() -> Intent:
    return Intent(
        kind="new_analysis",
        confidence=0.99,
        raw_message="total amount by region",
    )


def _chat_plan(*, valid: bool) -> AnalysisPlan:
    return AnalysisPlan(
        question="Total amount by region",
        method="group_by",
        rationale="Aggregate amount by region.",
        dataset_names=[DATASET_NAME],
        columns=["region", "amount" if valid else "invented"],
        sql=(
            "SELECT region, sum(amount) FROM orders GROUP BY region"
            if valid
            else "SELECT invented FROM orders"
        ),
        estimated_scan="small",
    )


def test_chat_uses_frozen_settings_and_payload_policy_per_session(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, str]] = []

    def fake_driver(_message: str, **kwargs: Any) -> ChatTurnResult:
        seen.append((kwargs["llm"].settings.model, kwargs["payload_policy"]))
        return ChatTurnResult(
            intent=Intent(kind="meta_help", confidence=1, raw_message="help"),
            status="answer",
            message="ok",
        )

    import eda_platform.drivers.chat as chat_driver

    monkeypatch.setattr(chat_driver, "run_chat_turn", fake_driver)
    monkeypatch.setattr(
        chat_service_module,
        "create_llm_client",
        lambda settings: _ScriptedProvider(settings, []),
    )
    sessions = (
        ("session-a", "gpt-4.1-mini", "schema_only"),
        ("session-b", "gpt-4.1", "schema+aggregates+sample"),
    )
    for session_id, model, policy in sessions:
        headers = {"X-EDA-Session": session_id}
        updated = client.put(
            "/api/v1/settings",
            headers=headers,
            json={
                "provider": "openai",
                "model": model,
                "api_key": "test-key",
                "payload_policy": policy,
            },
        )
        assert updated.status_code == 200
        accepted = client.post(
            f"/api/v1/sessions/{RUN}/chat/messages",
            headers=headers,
            json={"text": "help", "llm": "env"},
        ).json()
        _read_stream(client, RUN, accepted["message_id"])

    assert seen == [
        ("gpt-4.1-mini", "schema_only"),
        ("gpt-4.1", "schema+aggregates+sample"),
    ]


def test_chat_retry_is_metered_and_refreshes_persisted_metrics(
    client: TestClient,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(workspace)
    store.save_artifact(
        Artifact(
            id="metrics_before_chat",
            type=ArtifactType.SESSION_METRICS,
            project_id=PROJECT,
            session_id=RUN,
            payload=SessionMetrics(
                session_id=RUN,
                budget_reserved_calls=2,
                budget_settled_calls=2,
                budget_total_tokens=100,
                budget_est_cost_usd=0.02,
                budget_reconciliation="verified",
            ).model_dump(mode="json"),
        )
    )
    provider: _ScriptedProvider | None = None

    def build(settings: LLMSettings) -> _ScriptedProvider:
        nonlocal provider
        provider = _ScriptedProvider(
            settings,
            [_new_analysis_intent(), _chat_plan(valid=False), _chat_plan(valid=True)],
        )
        return provider

    monkeypatch.setattr(chat_service_module, "create_llm_client", build)
    headers = {"X-EDA-Session": "metered-chat"}
    client.put(
        "/api/v1/settings",
        headers=headers,
        json={
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "api_key": "test-key",
        },
    )
    accepted = client.post(
        f"/api/v1/sessions/{RUN}/chat/messages",
        headers=headers,
        json={"text": "total amount by region", "llm": "env"},
    ).json()
    frames = _read_stream(client, RUN, accepted["message_id"])

    assert frames[-1]["type"] == "message.completed"
    assert provider is not None
    assert provider.calls == ["m3_route_intent", "m3_build_plan", "m3_build_plan"]
    usage = [
        event
        for event in store.list_trace_events(project_id=PROJECT, session_id=RUN)
        if event.event_type == LLM_USAGE_EVENT
    ]
    assert len(usage) == 3
    metrics = client.get(f"/api/v1/sessions/{RUN}/metrics").json()
    assert metrics["source"] == "artifact+incremental"
    assert metrics["llm_calls"] == 3
    assert metrics["total_tokens"] == 45
    assert metrics["est_cost_usd"] == 0.003
    assert metrics["budget_reserved_calls"] == 5
    assert metrics["budget_settled_calls"] == 5
    assert metrics["budget_total_tokens"] == 145
    assert metrics["budget_est_cost_usd"] == 0.023
    assert metrics["budget_reconciliation"] == "verified"


def test_chat_restores_and_enforces_run_request_budget(
    client: TestClient,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider: _ScriptedProvider | None = None

    def build(settings: LLMSettings) -> _ScriptedProvider:
        nonlocal provider
        provider = _ScriptedProvider(
            settings,
            [_new_analysis_intent(), _chat_plan(valid=True)],
        )
        return provider

    monkeypatch.setattr(chat_service_module, "create_llm_client", build)
    store = ArtifactStore(workspace)
    app = cast(FastAPI, client.app)
    app.state.chat_service = chat_service_module.ChatService(
        store,
        ApprovalService(store),
        budget_policy=SessionBudgetPolicy(max_requests=1),
    )
    headers = {"X-EDA-Session": "budget-chat"}
    client.put(
        "/api/v1/settings",
        headers=headers,
        json={
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "api_key": "test-key",
        },
    )
    accepted = client.post(
        f"/api/v1/sessions/{RUN}/chat/messages",
        headers=headers,
        json={"text": "total amount by region", "llm": "env"},
    ).json()
    frames = _read_stream(client, RUN, accepted["message_id"])

    assert frames[-1]["type"] == "turn.failed"
    assert provider is not None and provider.calls == ["m3_route_intent"]
    event_types = [
        event.event_type
        for event in store.list_trace_events(project_id=PROJECT, session_id=RUN)
    ]
    assert event_types.count(LLM_USAGE_EVENT) == 1
    assert "budget_rejected" in event_types


class _PlanningDriver:
    """Stands in for `run_chat_turn`: the first turn returns a plan that needs
    approval, and only an `approved_plan` re-entry is allowed to 'execute'."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def plan(self) -> AnalysisPlan:
        return AnalysisPlan(
            question="Total amount by region",
            method="group_by",
            rationale="Aggregate the amount column per region.",
            dataset_names=[DATASET_NAME],
            columns=["region", "amount"],
            sql="SELECT region, sum(amount) FROM orders GROUP BY region",
            needs_approval=True,
            estimated_scan="small",
        )

    def __call__(self, message: str, **kwargs: Any) -> ChatTurnResult:
        self.calls.append({"message": message, **kwargs})
        intent = Intent(kind="new_analysis", confidence=0.9, raw_message=message)
        plan_artifact = Artifact(
            id="chatplan_pending",
            type=ArtifactType.CHAT_TURN_PLAN,
            project_id=kwargs["project_id"],
            session_id=kwargs["session_id"],
            payload=self.plan.model_dump(mode="json"),
        )
        if kwargs.get("approved_plan") is None:
            return ChatTurnResult(
                intent=intent,
                status="awaiting_approval",
                plan=self.plan,
                artifacts=[plan_artifact],
                sql=self.plan.sql,
                message="This analysis plan requires approval before execution.",
            )
        return ChatTurnResult(
            intent=intent,
            status="answer",
            plan=kwargs["approved_plan"],
            artifacts=[plan_artifact],
            sql=self.plan.sql,
            message="Ran SQL analysis: 4 rows returned.",
        )

    @property
    def executions(self) -> list[dict[str, Any]]:
        return [call for call in self.calls if call.get("approved_plan") is not None]


@pytest.fixture()
def planning_driver(monkeypatch: pytest.MonkeyPatch) -> _PlanningDriver:
    driver = _PlanningDriver()
    # ChatService imports the driver lazily inside the turn, so patching the
    # module attribute is what the running turn actually resolves.
    import eda_platform.drivers.chat as chat_driver

    monkeypatch.setattr(chat_driver, "run_chat_turn", driver)
    return driver


def _pending_plan(client: TestClient, planning_driver: _PlanningDriver) -> dict[str, Any]:
    accepted = client.post(
        f"/api/v1/sessions/{RUN}/chat/messages",
        json={"text": "total amount by region", "llm": "offline"},
    ).json()
    frames = _read_stream(client, RUN, accepted["message_id"])
    assert frames[-1]["type"] == "plan.pending"
    assert planning_driver.executions == []
    return frames[-1]["data"]


def test_plan_awaiting_approval_does_not_execute(
    client: TestClient, planning_driver: _PlanningDriver
) -> None:
    pending = _pending_plan(client, planning_driver)
    assert pending["plan_id"] == "chatplan_pending"
    assert pending["action_hash"] and pending["approval_token"]
    assert pending["sql"].startswith("SELECT region")

    # The transcript records the pending turn, and nothing has run.
    transcript = client.get(f"/api/v1/sessions/{RUN}/chat/messages").json()["messages"]
    assert transcript[-1]["status"] == "awaiting_approval"
    assert planning_driver.executions == []


def test_approve_without_a_valid_token_does_not_execute(
    client: TestClient, planning_driver: _PlanningDriver
) -> None:
    pending = _pending_plan(client, planning_driver)

    response = client.post(
        f"/api/v1/sessions/{RUN}/chat/plans/{pending['plan_id']}/approve",
        json={"action_hash": pending["action_hash"], "approval_token": "forged"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "approval_not_found"
    assert planning_driver.executions == []


def test_approve_consumes_the_approval_and_executes_once(
    client: TestClient, planning_driver: _PlanningDriver
) -> None:
    pending = _pending_plan(client, planning_driver)
    decision = {
        "action_hash": pending["action_hash"],
        "approval_token": pending["approval_token"],
    }

    approved = client.post(
        f"/api/v1/sessions/{RUN}/chat/plans/{pending['plan_id']}/approve", json=decision
    )
    assert approved.status_code == 202
    frames = _read_stream(client, RUN, approved.json()["message_id"])
    assert frames[-1]["type"] == "message.completed"
    assert frames[-1]["data"]["status"] == "answer"

    executions = planning_driver.executions
    assert len(executions) == 1
    assert executions[0]["approved_action_hash"] == pending["action_hash"]
    assert executions[0]["approved_plan"].sql == pending["sql"]

    # Replaying the same approval must not execute a second time.
    replay = client.post(
        f"/api/v1/sessions/{RUN}/chat/plans/{pending['plan_id']}/approve", json=decision
    )
    assert replay.status_code == 409
    assert replay.json()["error"]["code"] == "approval_consumed"
    assert len(planning_driver.executions) == 1


def test_reject_burns_the_approval_and_records_a_refusal(
    client: TestClient, planning_driver: _PlanningDriver
) -> None:
    pending = _pending_plan(client, planning_driver)
    decision = {
        "action_hash": pending["action_hash"],
        "approval_token": pending["approval_token"],
    }

    rejected = client.post(
        f"/api/v1/sessions/{RUN}/chat/plans/{pending['plan_id']}/reject", json=decision
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"

    transcript = client.get(f"/api/v1/sessions/{RUN}/chat/messages").json()["messages"]
    assert transcript[-1]["status"] == "refused"

    after = client.post(
        f"/api/v1/sessions/{RUN}/chat/plans/{pending['plan_id']}/approve", json=decision
    )
    assert after.status_code == 409
    assert planning_driver.executions == []


def test_approve_with_a_mismatched_plan_id_is_rejected(
    client: TestClient, planning_driver: _PlanningDriver
) -> None:
    pending = _pending_plan(client, planning_driver)
    response = client.post(
        f"/api/v1/sessions/{RUN}/chat/plans/other_plan/approve",
        json={
            "action_hash": pending["action_hash"],
            "approval_token": pending["approval_token"],
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "chat_invalid"
    assert planning_driver.executions == []


def test_a_second_turn_while_one_is_in_flight_is_rejected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = chat_service_module.threading.Event()
    release = chat_service_module.threading.Event()

    def blocking(message: str, **kwargs: Any) -> ChatTurnResult:
        started.set()
        release.wait(timeout=10)
        return ChatTurnResult(
            intent=Intent(kind="meta_help", confidence=1.0, raw_message=message),
            status="answer",
            message="done",
        )

    import eda_platform.drivers.chat as chat_driver

    monkeypatch.setattr(chat_driver, "run_chat_turn", blocking)

    first = client.post(
        f"/api/v1/sessions/{RUN}/chat/messages", json={"text": "hello", "llm": "offline"}
    )
    assert first.status_code == 202
    assert started.wait(timeout=10)

    second = client.post(
        f"/api/v1/sessions/{RUN}/chat/messages", json={"text": "again", "llm": "offline"}
    )
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "chat_busy"

    release.set()
    _read_stream(client, RUN, first.json()["message_id"])


def test_two_simultaneous_turns_start_exactly_one(
    client: TestClient, echo_driver: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The busy check and the session registration must be one atomic step.

    The gate widens the window that used to sit between them: with the check
    and the registration separated, both requests get past the check while the
    first is still writing the transcript, and two turns run for one run.
    """
    gate = threading.Barrier(2, timeout=2.0)
    original_append = chat_service_module.ChatService._append_message

    def gated_append(
        self: Any, project_id: str, session_id: str, message: ChatMessage
    ) -> None:
        if message.role == "user":
            with suppress(threading.BrokenBarrierError):
                gate.wait()
        original_append(self, project_id, session_id, message)

    monkeypatch.setattr(
        chat_service_module.ChatService, "_append_message", gated_append
    )

    results: list[tuple[int, dict[str, Any]]] = []
    results_lock = threading.Lock()
    start = threading.Barrier(2)

    def post() -> None:
        start.wait(timeout=10)
        response = client.post(
            f"/api/v1/sessions/{RUN}/chat/messages",
            json={"text": "same instant", "llm": "offline"},
        )
        with results_lock:
            results.append((response.status_code, response.json()))

    threads = [threading.Thread(target=post) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()

    assert sorted(status for status, _ in results) == [202, 409]
    rejected = next(body for status, body in results if status == 409)
    assert rejected["error"]["code"] == "chat_busy"

    accepted = next(body for status, body in results if status == 202)
    frames = _read_stream(client, RUN, accepted["message_id"])
    assert frames[-1]["type"] == "message.completed"
    # The decisive assertion: the loser never reached the driver at all.
    assert echo_driver == ["same instant"]


def test_reconnect_resumes_after_last_event_id_and_then_answers_204(
    client: TestClient, echo_driver: list[str]
) -> None:
    accepted = _accept_turn(client, "resume me")
    message_id = accepted["message_id"]
    frames = _read_stream(client, RUN, message_id)
    seqs = [frame["seq"] for frame in frames]
    assert seqs == sorted(seqs) and len(seqs) >= 2

    replay = _read_stream(client, RUN, message_id, last_event_id=seqs[0])
    assert [frame["seq"] for frame in replay] == seqs[1:]

    # Nothing left to replay on a finished turn: 204, not an empty stream the
    # browser would reconnect to forever.
    exhausted = client.get(
        f"/api/v1/sessions/{RUN}/chat/stream?message_id={message_id}",
        headers={"Last-Event-ID": str(seqs[-1])},
    )
    assert exhausted.status_code == 204


def test_evicted_session_reads_as_404(
    client: TestClient, echo_driver: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chat_service_module, "MAX_RETAINED_SESSIONS", 2)

    message_ids: list[str] = []
    for index in range(3):
        accepted = _accept_turn(client, f"turn {index}")
        message_ids.append(accepted["message_id"])
        _read_stream(client, RUN, accepted["message_id"])

    evicted = client.get(f"/api/v1/sessions/{RUN}/chat/stream?message_id={message_ids[0]}")
    assert evicted.status_code == 404
    assert evicted.json()["error"]["code"] == "chat_message_not_found"
    # The sessions still inside the window keep streaming.
    assert (
        client.get(
            f"/api/v1/sessions/{RUN}/chat/stream?message_id={message_ids[-1]}",
            headers={"Last-Event-ID": "999"},
        ).status_code
        == 204
    )


def test_session_events_are_capped_and_flagged_as_truncated() -> None:
    session = chat_service_module._TurnSession(
        message_id="m1", session_id=RUN, project_id=PROJECT
    )
    for index in range(chat_service_module.MAX_SESSION_EVENTS + 10):
        session.append("tool.call", {"n": index})
    session.append("message.completed", {"content": "done"})

    page = session.after(0)
    assert len(page.events) == chat_service_module.MAX_SESSION_EVENTS
    assert page.truncated is True
    assert page.done is True
    # Seq numbers stay monotonic across the drop, and the terminal frame stays.
    seqs = [event.seq for event in page.events]
    assert seqs == sorted(seqs)
    assert page.events[-1].type == "message.completed"


def test_pending_plan_survives_a_service_restart(
    client: TestClient, workspace: Path, planning_driver: _PlanningDriver
) -> None:
    pending = _pending_plan(client, planning_driver)

    # The persisted line carries the plan's identity, but never its token.
    raw = (workspace / "projects" / PROJECT / "chat" / f"{RUN}.jsonl").read_text(
        encoding="utf-8"
    )
    awaiting = json.loads(raw.strip().splitlines()[-1])
    assert awaiting["status"] == "awaiting_approval"
    assert awaiting["plan_id"] == pending["plan_id"]
    assert awaiting["action_hash"] == pending["action_hash"]
    assert awaiting["expires_at"]
    assert pending["approval_token"] not in raw

    # Restart: a fresh service keeps the store but loses every session buffer.
    store = ArtifactStore(workspace)
    app = cast(FastAPI, client.app)
    app.state.chat_service = chat_service_module.ChatService(
        store, ApprovalService(store)
    )

    recovered = client.get(f"/api/v1/sessions/{RUN}/chat/pending-plans")
    assert recovered.status_code == 200
    plans = recovered.json()["plans"]
    assert len(plans) == 1
    assert plans[0]["plan_id"] == pending["plan_id"]
    assert plans[0]["action_hash"] == pending["action_hash"]
    assert plans[0]["sql"] == pending["sql"]
    # Recovery is a read: the durable token survives the service restart.
    assert plans[0]["approval_token"] == pending["approval_token"]

    approved = client.post(
        f"/api/v1/sessions/{RUN}/chat/plans/{plans[0]['plan_id']}/approve",
        json={
            "action_hash": plans[0]["action_hash"],
            "approval_token": plans[0]["approval_token"],
        },
    )
    assert approved.status_code == 202
    frames = _read_stream(client, RUN, approved.json()["message_id"])
    assert frames[-1]["type"] == "message.completed"
    assert len(planning_driver.executions) == 1


def test_recovery_get_is_read_only_and_preserves_the_displayed_token(
    client: TestClient,
    workspace: Path,
    planning_driver: _PlanningDriver,
) -> None:
    pending = _pending_plan(client, planning_driver)
    store = ArtifactStore(workspace)
    before = store.get_pending_action(pending["action_hash"], session_id=RUN)
    first = client.get(f"/api/v1/sessions/{RUN}/chat/pending-plans")
    second = client.get(f"/api/v1/sessions/{RUN}/chat/pending-plans")
    after = store.get_pending_action(pending["action_hash"], session_id=RUN)

    assert first.status_code == 200
    assert first.json() == second.json()
    assert before == after
    recovered = first.json()["plans"][0]
    assert recovered["approval_token"] == pending["approval_token"]
    approved = client.post(
        f"/api/v1/sessions/{RUN}/chat/plans/{pending['plan_id']}/approve",
        json={
            "action_hash": pending["action_hash"],
            "approval_token": pending["approval_token"],
        },
    )
    assert approved.status_code == 202


def test_concurrent_pending_plan_gets_return_identical_token_without_writes(
    client: TestClient,
    workspace: Path,
    planning_driver: _PlanningDriver,
) -> None:
    pending = _pending_plan(client, planning_driver)
    store = ArtifactStore(workspace)
    before = store.get_pending_action(pending["action_hash"], session_id=RUN)
    barrier = threading.Barrier(8)
    responses: list[dict[str, Any]] = []
    lock = threading.Lock()

    def read() -> None:
        barrier.wait()
        response = client.get(f"/api/v1/sessions/{RUN}/chat/pending-plans")
        assert response.status_code == 200
        with lock:
            responses.append(response.json())

    threads = [threading.Thread(target=read) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert len(responses) == 8
    assert all(response == responses[0] for response in responses)
    assert (
        responses[0]["plans"][0]["approval_token"]
        == pending["approval_token"]
    )
    assert store.get_pending_action(pending["action_hash"], session_id=RUN) == before


def test_pending_plan_without_durable_token_fails_expired_instead_of_rotating(
    client: TestClient,
    workspace: Path,
    planning_driver: _PlanningDriver,
) -> None:
    pending = _pending_plan(client, planning_driver)
    store = ArtifactStore(workspace)
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            update pending_actions set generation = ''
            where action_hash = ? and session_id = ?
            """,
            (pending["action_hash"], RUN),
        )

    response = client.get(f"/api/v1/sessions/{RUN}/chat/pending-plans")
    assert response.status_code == 410, response.text
    assert response.json()["error"]["code"] == "approval_expired"
    row = store.get_pending_action(pending["action_hash"], session_id=RUN)
    assert row is not None and row["generation"] == ""


def test_wrong_chat_plan_path_does_not_consume_correct_approval(
    client: TestClient,
    workspace: Path,
    planning_driver: _PlanningDriver,
) -> None:
    pending = _pending_plan(client, planning_driver)
    swapped = client.post(
        f"/api/v1/sessions/{RUN}/chat/plans/plan_wrong/approve",
        json={
            "action_hash": pending["action_hash"],
            "approval_token": pending["approval_token"],
        },
    )
    assert swapped.status_code == 422, swapped.text
    row = ArtifactStore(workspace).get_pending_action(
        pending["action_hash"], session_id=RUN
    )
    assert row is not None
    assert row["status"] == "pending"
    assert row["generation"] == pending["approval_token"]

    approved = client.post(
        f"/api/v1/sessions/{RUN}/chat/plans/{pending['plan_id']}/approve",
        json={
            "action_hash": pending["action_hash"],
            "approval_token": pending["approval_token"],
        },
    )
    assert approved.status_code == 202


def test_chat_spawn_failure_restores_same_approval_for_retry(
    client: TestClient,
    workspace: Path,
    planning_driver: _PlanningDriver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = _pending_plan(client, planning_driver)
    app = cast(FastAPI, client.app)
    service = app.state.chat_service
    original_spawn = service._spawn
    monkeypatch.setattr(
        service,
        "_spawn",
        lambda _session, _target: (_ for _ in ()).throw(
            RuntimeError("injected thread start failure")
        ),
    )
    with pytest.raises(RuntimeError, match="injected thread start failure"):
        client.post(
            f"/api/v1/sessions/{RUN}/chat/plans/{pending['plan_id']}/approve",
            json={
                "action_hash": pending["action_hash"],
                "approval_token": pending["approval_token"],
            },
        )
    row = ArtifactStore(workspace).get_pending_action(
        pending["action_hash"], session_id=RUN
    )
    assert row is not None
    assert row["status"] == "pending"
    assert row["generation"] == pending["approval_token"]

    monkeypatch.setattr(service, "_spawn", original_spawn)
    retried = client.post(
        f"/api/v1/sessions/{RUN}/chat/plans/{pending['plan_id']}/approve",
        json={
            "action_hash": pending["action_hash"],
            "approval_token": pending["approval_token"],
        },
    )
    assert retried.status_code == 202


def test_pending_plans_is_empty_once_the_plan_is_decided(
    client: TestClient, planning_driver: _PlanningDriver
) -> None:
    pending = _pending_plan(client, planning_driver)
    assert len(client.get(f"/api/v1/sessions/{RUN}/chat/pending-plans").json()["plans"]) == 1

    latest = client.get(f"/api/v1/sessions/{RUN}/chat/pending-plans").json()["plans"][0]
    rejected = client.post(
        f"/api/v1/sessions/{RUN}/chat/plans/{pending['plan_id']}/reject",
        json={
            "action_hash": latest["action_hash"],
            "approval_token": latest["approval_token"],
        },
    )
    assert rejected.status_code == 200
    assert client.get(f"/api/v1/sessions/{RUN}/chat/pending-plans").json()["plans"] == []


def test_pending_plans_for_an_unknown_run_is_404(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/nope/chat/pending-plans")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_oversized_transcript_line_is_counted_and_shown_as_a_placeholder(
    client: TestClient, workspace: Path
) -> None:
    """The index refuses to read a >1 MiB record, but dropping it silently makes
    `total` wrong and hides the gap from the reader."""
    path = workspace / "projects" / PROJECT / "chat" / f"{RUN}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    normal = ChatMessage(role="user", content="before").model_dump_json()
    huge = ChatMessage(
        role="assistant", content="x" * (MAX_JSONL_RECORD_BYTES + 16)
    ).model_dump_json()
    after = ChatMessage(role="user", content="after").model_dump_json()
    path.write_text("\n".join([normal, huge, after]) + "\n", encoding="utf-8")

    body = client.get(f"/api/v1/sessions/{RUN}/chat/messages", params={"limit": 50}).json()

    assert body["total"] == 3
    assert [message["seq"] for message in body["messages"]] == [0, 1, 2]
    assert body["messages"][0]["content"] == "before"
    assert body["messages"][2]["content"] == "after"
    placeholder = body["messages"][1]
    assert placeholder["status"] == "omitted"
    assert "too large" in placeholder["content"].lower()
