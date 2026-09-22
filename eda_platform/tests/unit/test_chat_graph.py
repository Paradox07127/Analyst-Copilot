"""Restart coverage for structured-provider chat, including planner repair."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from eda_platform.core.graph_execution import (
    GraphEffectUncertain,
    GraphExecution,
    GraphIdentityError,
)
from eda_platform.core.llm import LLMResultMetadata, LLMSettings, LLMUsage
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers import chat
from eda_platform.schemas.plans import AnalysisPlan, Intent
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset


class WorkerLost(BaseException):
    pass


class Model:
    def __init__(self, replies: list[BaseModel]) -> None:
        self.replies = list(replies)
        self.calls: list[str] = []
        self.settings = LLMSettings(model="structured-test")
        self.usage: LLMResultMetadata | None = None

    def structured[T: BaseModel](self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append(task)
        self.usage = LLMResultMetadata(
            provider="test",
            model="structured-test",
            usage=LLMUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        return schema.model_validate(self.replies.pop(0).model_dump())

    def last_usage(self) -> LLMResultMetadata | None:
        return self.usage


def setup(tmp_path: Path):
    path = tmp_path / "orders.csv"
    path.write_text("region,amount\nEast,10\nWest,20\n")
    loaded = load_csv(path, dataset_id="orders")
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("p", "Project")
    store.start_session("p", "s")
    profile = profile_dataset(loaded, project_id="p", session_id="s")
    store.save_artifact(profile)
    return store, loaded, profile


def plans() -> list[BaseModel]:
    good = AnalysisPlan(
        question="Total amount",
        dataset_names=["orders"],
        columns=["amount"],
        filters=[],
        sql="select sum(amount) as total from orders",
        method="aggregate",
        rationale="Sum the amount",
        needs_approval=False,
        estimated_scan="small",
    )
    return [
        Intent(kind="new_analysis", confidence=1.0, raw_message="Total amount"),
        good.model_copy(update={"columns": ["imaginary"]}),
        good,
    ]


@pytest.mark.parametrize("boundary", ["model:1", "model:3", "chat-local:chat_execute_plan"])
def test_structured_chat_resumes_without_repeating_requests_or_committed_sql(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    store, loaded, _profile = setup(tmp_path)
    first = Model(plans())
    original_effect = GraphExecution.effect
    original_sql = chat.run_sql
    crashed = False
    query_calls = 0

    def effect(self: GraphExecution, key: str, request: dict, call: Any, **kwargs: Any) -> dict:
        nonlocal crashed
        result = original_effect(self, key, request, call, **kwargs)
        if key == boundary and not crashed:
            crashed = True
            raise WorkerLost()
        return result

    def query(*args: Any, **kwargs: Any):
        nonlocal query_calls
        query_calls += 1
        return original_sql(*args, **kwargs)

    monkeypatch.setattr(GraphExecution, "effect", effect)
    monkeypatch.setattr(chat, "run_sql", query)

    def run(client: Model, *, resume: bool = False):
        return chat.run_chat_turn(
            "Total amount",
            datasets=[loaded],
            project_id="p",
            session_id="s",
            llm=client,
            artifacts=store.list_artifacts(project_id="p", session_id="s"),
            store=store,
            turn_id="structured-turn",
            resume_mode="structured" if resume else None,
        )

    with pytest.raises(WorkerLost):
        run(first)
    # The resumed source collection now includes artifacts written by this very
    # turn; its original evidence snapshot must remain stable.
    resumed = Model(first.replies)
    monkeypatch.setattr(chat, "tool_calling_readiness", lambda *_: pytest.fail("reprobed"))
    result = run(resumed, resume=True)
    assert result.status == "answer"
    assert len(first.calls) + len(resumed.calls) == 3
    assert query_calls == 1
    assert result.intent.raw_message == "Total amount"
    assert len(result.artifacts) == 2
    assert run(Model([]), resume=True) == result
    assert query_calls == 1
    usage_events = store.list_trace_events(
        project_id="p",
        session_id="s",
        event_types=["llm_call"],
    )
    assert len(usage_events) == 3
    assert sum(event.summary["total_tokens"] for event in usage_events) == 24


def test_structured_chat_rejects_changed_frozen_evidence_before_paid_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, loaded, profile = setup(tmp_path)
    original = GraphExecution.effect
    crashed = False

    def effect(self: GraphExecution, key: str, request: dict, call: Any, **kwargs: Any) -> dict:
        nonlocal crashed
        result = original(self, key, request, call, **kwargs)
        if key == "model:1" and not crashed:
            crashed = True
            raise WorkerLost()
        return result

    monkeypatch.setattr(GraphExecution, "effect", effect)
    first = Model(plans())
    kwargs: dict[str, Any] = dict(
        datasets=[loaded],
        project_id="p",
        session_id="s",
        store=store,
        turn_id="turn",
        resume_mode="structured",
    )
    with pytest.raises(WorkerLost):
        chat.run_chat_turn("Total amount", llm=first, artifacts=[profile], **kwargs)
    changed = profile.model_copy(deep=True)
    changed.payload["rows"] = 99
    store.save_artifact(changed)
    resumed = Model([])
    with pytest.raises(GraphIdentityError):
        chat.run_chat_turn("Total amount", llm=resumed, artifacts=[changed], **kwargs)
    assert not resumed.calls


def test_structured_router_timeout_cannot_fall_through_to_paid_planning(tmp_path: Path) -> None:
    store, loaded, profile = setup(tmp_path)

    class TimeoutModel(Model):
        def structured[T: BaseModel](self, *, task: str, schema: type[T], payload: dict) -> T:
            self.calls.append(task)
            raise TimeoutError("Unknown remote outcome")

    client = TimeoutModel([])
    for _ in range(2):
        with pytest.raises(GraphEffectUncertain):
            chat.run_chat_turn(
                "Total amount",
                datasets=[loaded],
                project_id="p",
                session_id="s",
                llm=client,
                artifacts=[profile],
                store=store,
                turn_id="turn",
                resume_mode="structured",
            )
    assert client.calls == ["m3_route_intent"]


def test_approved_graph_preserves_plan_and_rejects_changed_results(tmp_path: Path) -> None:
    from eda_platform.core.permissions import action_hash, analysis_plan_action

    store, loaded, profile = setup(tmp_path)
    plan = AnalysisPlan.model_validate(plans()[-1].model_dump())
    pending = chat._plan_artifact(
        plan,
        message="Total amount",
        project_id="p",
        session_id="s",
        parents=[profile.id],
    )
    store.save_artifact(pending)
    kwargs: dict[str, Any] = dict(
        datasets=[loaded],
        project_id="p",
        session_id="s",
        llm=Model([]),
        store=store,
        approved_plan=plan,
        approved_action_hash=action_hash(analysis_plan_action(plan)),
        turn_id="approved",
    )
    first = chat.run_chat_turn(
        "Total amount", artifacts=store.list_artifacts(project_id="p", session_id="s"), **kwargs
    )
    assert first.status == "answer"
    saved = store.get_artifact(pending.id, project_id="p", session_id="s")
    assert saved.parents == [profile.id]
    assert chat.run_chat_turn("Total amount", artifacts=first.artifacts, **kwargs) == first
    sql_result = first.artifacts[-1]
    sql_result.payload["rows_preview"] = [{"total": 999}]
    store.save_artifact(sql_result)
    with pytest.raises(GraphIdentityError, match="Saved chat evidence changed"):
        chat.run_chat_turn("Total amount", artifacts=first.artifacts, **kwargs)
