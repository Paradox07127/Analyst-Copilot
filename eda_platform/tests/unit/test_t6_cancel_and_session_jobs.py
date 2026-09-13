"""T6: per-session job listing and cooperative chat-turn cancellation."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import eda_platform.core.session_loader as session_loader_module
import eda_platform.drivers.chat as chat_driver_module
from eda_platform.application.ports import JobCommand, JobRef
from eda_platform.application.services.approval_service import ApprovalService
from eda_platform.application.services.chat_service import ChatService
from eda_platform.application.services.job_service import JobService
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.core.llm import LLMToolCall, LLMToolResponse
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.chat import run_chat_turn
from eda_platform.schemas.chat import ChatTurnResult
from eda_platform.schemas.plans import Intent
from eda_platform.tools.loader import load_csv

PROJECT = "demo"
SOURCE = "run_t6_source"
DEADLINE_SECONDS = 10.0


class _RecordingBackend:
    def enqueue(self, command: JobCommand) -> JobRef:
        return JobRef(job_id=command.job_id)

    def cancel(self, job_id: str) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def status(self, job_id: str) -> str:
        return "queued"


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    store.start_session(PROJECT, SOURCE)
    return store


def test_session_jobs_lists_derived_jobs_for_their_launching_session(
    store: ArtifactStore,
) -> None:
    service = JobService(store, _RecordingBackend())
    assert service.list_session_jobs(SOURCE).jobs == []

    report = service.create_report_generate_job(
        "rgsess_t6_report",
        project_id=PROJECT,
        source_session_id=SOURCE,
    )
    question = service.create_question_exec_job(
        "qsess_t6_question",
        project_id=PROJECT,
        source_session_id=SOURCE,
        question_id="q1",
        candidate_fingerprint="f" * 32,
    )

    listing = service.list_session_jobs(SOURCE)
    assert listing.session_id == SOURCE
    assert {job.job_id for job in listing.jobs} == {report.job_id, question.job_id}
    for job in listing.jobs:
        assert job.source_session_id == SOURCE
        assert job.status == "queued"
        assert job.events_url == f"/api/v1/jobs/{job.job_id}/events"
    kinds = {job.kind for job in listing.jobs}
    assert kinds == {"report_generate", "question_exec"}

    # Scoped to the launching session, not the workspace.
    store.start_session(PROJECT, "run_t6_other")
    assert JobService(store, _RecordingBackend()).list_session_jobs("run_t6_other").jobs == []


class _ScriptedToolLLM:
    def __init__(self, responses: list[LLMToolResponse]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolResponse:
        self.calls += 1
        return self.responses.pop(0)

    def structured(self, *, task: str, schema: type[Any], payload: dict[str, Any]) -> Any:
        raise AssertionError("The agentic path should use native tool calls.")

    def last_usage(self) -> None:
        return None


def test_run_chat_turn_maps_a_cancelled_loop_to_a_readable_stop(
    tmp_path: Path,
) -> None:
    source = tmp_path / "orders.csv"
    source.write_text("region,amount\nEast,10\nWest,20\n", encoding="utf-8")
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project(PROJECT, name="Demo")
    store.start_session(PROJECT, SOURCE)
    llm = _ScriptedToolLLM(
        [
            LLMToolResponse(
                tool_calls=[
                    LLMToolCall(call_id="c1", name="inspect_data_catalog", arguments={})
                ]
            ),
            LLMToolResponse(content="This second step must never run."),
        ]
    )
    result = run_chat_turn(
        "Which region has the highest total amount?",
        datasets=[load_csv(source, dataset_id="orders_1")],
        project_id=PROJECT,
        session_id=SOURCE,
        llm=llm,  # type: ignore[arg-type]
        store=store,
        cancel_check=lambda: True,
    )

    assert result.status == "cancelled"
    assert "Stopped at your request" in result.message
    assert llm.calls == 0  # the flag was already set, so no model call started


def test_cancel_turn_stops_the_in_flight_turn_and_records_a_stop(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_load_run(project_id: str, session_id: str, *, workspace: Path) -> Any:
        return SimpleNamespace(
            result=SimpleNamespace(loaded_datasets=[object()], artifacts=[])
        )

    def fake_run_chat_turn(message: str, **kwargs: Any) -> ChatTurnResult:
        cancel_check = kwargs["cancel_check"]
        deadline = time.monotonic() + DEADLINE_SECONDS
        while not cancel_check() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert cancel_check(), "cancel flag never reached the driver"
        return ChatTurnResult(
            intent=Intent(kind="new_analysis", confidence=1.0, raw_message=message),
            status="cancelled",
            message="Stopped at your request.",
        )

    monkeypatch.setattr(session_loader_module, "load_run", fake_load_run)
    monkeypatch.setattr(chat_driver_module, "run_chat_turn", fake_run_chat_turn)

    service = ChatService(store, ApprovalService(store))
    accepted = service.send_message(SOURCE, text="count everything", llm="offline")

    cancelled = service.cancel_turn(SOURCE)
    assert cancelled.cancel_requested is True
    assert cancelled.message_id == accepted.message_id

    deadline = time.monotonic() + DEADLINE_SECONDS
    page = service.events_after(SOURCE, accepted.message_id, 0)
    while not page.done and time.monotonic() < deadline:
        time.sleep(0.02)
        page = service.events_after(SOURCE, accepted.message_id, 0)
    assert page.done, "the cancelled turn never settled"

    completed = next(
        event for event in page.events if event.type == "message.completed"
    )
    assert completed.data["status"] == "cancelled"
    assert "Stopped" in str(completed.data["content"])

    # The stop is durable: the transcript ends with the readable stop message.
    messages = service.list_messages(SOURCE).messages
    assert messages[-1].role == "assistant"
    assert messages[-1].status == "cancelled"
    assert "Stopped" in messages[-1].content

    # With nothing in flight any more, cancel is a truthful no-op.
    assert service.cancel_turn(SOURCE).cancel_requested is False


def test_cancel_turn_on_an_unknown_run_is_not_found(store: ArtifactStore) -> None:
    service = ChatService(store, ApprovalService(store))
    with pytest.raises(SessionNotFoundError):
        service.cancel_turn("run_missing")
