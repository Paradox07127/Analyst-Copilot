"""Local graph lineage, event idempotency and transport-attempt boundaries."""

import io
from email.message import Message
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import pytest
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from eda_platform.core.graph_execution import (
    GraphEffectUncertain,
    GraphPersistence,
    graph_execution,
)
from eda_platform.core.llm import LLMSettings, OpenAICompatibleLLMClient
from eda_platform.core.store import ArtifactStore
from eda_platform.core.trace_correlation import trace_execution_scope
from eda_platform.schemas.sessions import TraceEvent


@pytest.mark.parametrize("failure", ["read_timeout", "gateway"])
def test_durable_effect_never_reposts_uncertain_provider_request(tmp_path: Path, failure):
    client = OpenAICompatibleLLMClient(
        LLMSettings.model_validate({"provider": "openai", "api_key": "test", "model": "test"})
    )
    count = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            raise TimeoutError("response lost after provider execution")

    def transport(*args, **kwargs):
        nonlocal count
        count += 1
        if failure == "gateway":
            raise HTTPError(
                "https://api.openai.com", 504, "gateway timeout", Message(), io.BytesIO(b"unknown")
            )
        return Response()

    persistence = GraphPersistence(tmp_path, "request")
    with patch("eda_platform.core.llm.credential_safe_urlopen", transport):
        with graph_execution(persistence, definition="test", inputs={}) as execution:
            with pytest.raises((RuntimeError, TimeoutError)):
                execution.model_effect("model:1", {}, lambda: client._send_json("/test", {}))
        with graph_execution(persistence, definition="test", inputs={}) as execution:
            with pytest.raises(GraphEffectUncertain):
                execution.model_effect("model:1", {}, lambda: client._send_json("/test", {}))
    assert count == 1


class State(TypedDict):
    status: str


def test_node_domain_trace_and_nested_graph_lineage_are_correlated(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    store.start_session("p", "s")
    directory = store.session_dir("p", "s")

    def emit(event):
        return store.append_trace("p", event)

    def work(state):
        for _ in range(2):
            emit(
                TraceEvent(
                    session_id="s",
                    event_type="code_agent_attempt",
                    name="code_agent",
                    summary={"attempt": 1},
                )
            )
        with graph_execution(GraphPersistence(directory, "child"), definition="child", inputs={}):
            emit(TraceEvent(session_id="s", event_type="domain", name="nested"))
        return {"status": "completed"}

    with trace_execution_scope(session_id="s", turn_id="turn", attempt_id="turn:1", emit=emit):
        with graph_execution(
            GraphPersistence(directory, "parent"), definition="parent", inputs={}
        ) as execution:
            builder = StateGraph(State)
            builder.add_node("work", work)
            builder.add_edge(START, "work")
            builder.add_edge("work", END)
            builder.compile(checkpointer=execution.saver).invoke(
                {"status": "running"},
                execution.config,
                durability="sync",
            )
    rows = store.list_trace_events(project_id="p", session_id="s")
    attempts = [r for r in rows if r.event_type == "code_agent_attempt"]
    assert len(attempts) == 1
    assert attempts[0].graph_node == "work"
    assert attempts[0].checkpoint_ns
    child = next(r for r in rows if r.name == "nested")
    assert (child.execution_id, child.parent_execution_id, child.turn_id) == (
        "child",
        "parent",
        "turn",
    )
    completed = next(r for r in rows if r.event_type == "graph.node_completed")
    assert completed.summary["status"] == "completed"
