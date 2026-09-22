"""Cross-layer chat admission, approval, recovery and audit contracts."""

from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
import test_api_chat as fixtures
from fastapi.testclient import TestClient

from eda_platform.agents.model_workflow import run_model_workflow
from eda_platform.api.main import create_app
from eda_platform.core.graph_execution import GraphPersistence
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.chat import ChatTurnResult


def setup(tmp_path):
    root = cast(Any, fixtures.workspace).__wrapped__(tmp_path)
    app = create_app(root)
    return root, app, TestClient(app)


def synchronous(session, target):
    target()


def test_admitted_chat_survives_restart_before_graph_exists(tmp_path: Path):
    root, app, client = setup(tmp_path)
    with patch.object(app.state.chat_service, "_spawn", lambda *_: None):
        accepted = fixtures._accept_turn(client, "help")
    restarted = create_app(root)
    client = TestClient(restarted)
    turns = client.get(f"/api/v1/sessions/{fixtures.RUN}/chat/recoverable-turns").json()["turns"]
    assert [turn["message_id"] for turn in turns] == [accepted["message_id"]]
    with patch.object(restarted.state.chat_service, "_spawn", synchronous):
        resumed = client.post(
            f"/api/v1/sessions/{fixtures.RUN}/chat/turns/{accepted['message_id']}/resume"
        )
    assert resumed.status_code == 202
    assert "message.completed" in client.get(resumed.json()["stream_url"]).text
    assert (
        client.get(f"/api/v1/sessions/{fixtures.RUN}/chat/recoverable-turns").json()["turns"] == []
    )


def test_consumed_approval_has_durable_execution_and_resumes_once(tmp_path: Path):
    root, app, client = setup(tmp_path)

    class ValidPlanning(fixtures._PlanningDriver):
        @property
        def plan(self):
            return super().plan.model_copy(update={"dataset_names": ["orders"]})

    planning = ValidPlanning()
    with patch("eda_platform.drivers.chat.run_chat_turn", planning):
        pending = fixtures._pending_plan(client, planning)
    decision = dict(action_hash=pending["action_hash"], approval_token=pending["approval_token"])
    approve_url = f"/api/v1/sessions/{fixtures.RUN}/chat/plans/{pending['plan_id']}/approve"
    with patch.object(app.state.chat_service, "_spawn", lambda *_: None):
        accepted = client.post(approve_url, json=decision)
    assert accepted.status_code == 202
    restarted = create_app(root)
    client = TestClient(restarted)
    turn_id = accepted.json()["message_id"]
    turns = client.get(f"/api/v1/sessions/{fixtures.RUN}/chat/recoverable-turns").json()["turns"]
    assert [t["message_id"] for t in turns] == [turn_id]
    assert client.post(approve_url, json=decision).status_code == 409
    with patch.object(restarted.state.chat_service, "_spawn", synchronous):
        resumed = client.post(f"/api/v1/sessions/{fixtures.RUN}/chat/turns/{turn_id}/resume")
    assert resumed.status_code == 202
    response = client.get(resumed.json()["stream_url"]).text
    assert "message.completed" in response and '"status": "answer"' in response
    store = ArtifactStore(root)
    traces = store.list_trace_events(project_id=fixtures.PROJECT, session_id=fixtures.RUN)
    assert any(t.execution_id == "chat-approved:" + turn_id for t in traces)
    assert sum(t.event_type == "tool_completed" and t.name == "run_sql" for t in traces) == 1
    assert (
        client.post(f"/api/v1/sessions/{fixtures.RUN}/chat/turns/{turn_id}/resume").status_code
        == 404
    )


def test_rejected_plan_cannot_be_reactivated_by_delivery_replay(tmp_path: Path):
    root, app, client = setup(tmp_path)
    planning = fixtures._PlanningDriver()

    def durable_driver(message, **kwargs):
        def calculate(_):
            result = planning(message, **kwargs)
            for artifact in result.artifacts:
                kwargs["store"].save_artifact(artifact)
            return result.model_dump(mode="json")

        persistence = GraphPersistence(
            kwargs["store"].session_dir(kwargs["project_id"], kwargs["session_id"]),
            "chat-structured:" + kwargs["turn_id"],
        )
        return ChatTurnResult.model_validate(
            run_model_workflow(
                calculate,
                persistence=persistence,
                inputs={},
                definition="chat-structured-functional-v1",
                checkpoint_input={"message": message, "turn_id": kwargs["turn_id"]},
            )
        )

    with patch("eda_platform.drivers.chat.run_chat_turn", durable_driver):
        original_append = app.state.chat_service._executions.append_event

        def fail_delivery(project, session, turn, event_type, data):
            if event_type == "plan.pending":
                raise OSError("terminal event commit lost")
            return original_append(project, session, turn, event_type, data)

        with patch.object(app.state.chat_service._executions, "append_event", fail_delivery):
            accepted = fixtures._accept_turn(client, "total amount by region")
            frames = fixtures._read_stream(client, fixtures.RUN, accepted["message_id"])
        assert frames[-1]["type"] == "turn.failed"
        previous_seq = frames[-1]["seq"]
        pending = client.get(f"/api/v1/sessions/{fixtures.RUN}/chat/pending-plans").json()["plans"][
            0
        ]
        decision = dict(
            action_hash=pending["action_hash"], approval_token=pending["approval_token"]
        )
        assert (
            client.post(
                f"/api/v1/sessions/{fixtures.RUN}/chat/plans/{pending['plan_id']}/reject",
                json=decision,
            ).status_code
            == 200
        )
        with patch.object(app.state.chat_service, "_spawn", synchronous):
            resumed = client.post(
                f"/api/v1/sessions/{fixtures.RUN}/chat/turns/{accepted['message_id']}/resume"
            )
        assert resumed.status_code == 202
        response = client.get(
            resumed.json()["stream_url"], headers={"Last-Event-ID": str(previous_seq)}
        )
        assert response.status_code == 200
        assert "message.completed" in response.text and '"status": "refused"' in response.text
        assert "plan.pending" not in response.text
        assert (
            client.get(f"/api/v1/sessions/{fixtures.RUN}/chat/pending-plans").json()["plans"] == []
        )
        row = ArtifactStore(root).get_pending_action(
            pending["action_hash"], session_id=fixtures.RUN
        )
        assert row is not None
        assert row["status"] == "consumed" and row["generation"] == pending["approval_token"]


def test_trace_api_exposes_turn_graph_and_attempt_without_inputs(tmp_path: Path):
    _, app, client = setup(tmp_path)
    with patch.object(app.state.chat_service, "_spawn", synchronous):
        accepted = fixtures._accept_turn(client, "help")
    response = client.get(f"/api/v1/sessions/{fixtures.RUN}/trace")
    assert response.status_code == 200
    events = response.json()["items"]
    assert events and all(e["turn_id"] == accepted["message_id"] for e in events)
    graph_events = [e for e in events if e["event_type"].startswith("graph.")]
    assert graph_events and all(e["execution_id"] for e in graph_events)
    assert all(e["attempt_id"] == f"chat:{accepted['message_id']}:1" for e in graph_events)
    assert any(e["graph_node"] for e in graph_events)
    assert all(
        "messages" not in e["summary"] and "payload" not in e["summary"] for e in graph_events
    )


@pytest.mark.parametrize("boundary", ["admit", "spawn"])
def test_failed_admission_releases_live_slot(tmp_path: Path, boundary):
    _, app, client = setup(tmp_path)
    service = app.state.chat_service
    target = service._executions if boundary == "admit" else service
    attribute = "admit" if boundary == "admit" else "_spawn"
    with patch.object(target, attribute, side_effect=RuntimeError("injected admission failure")):
        with pytest.raises(RuntimeError, match="injected admission failure"):
            fixtures._accept_turn(client, "help")
    with patch.object(service, "_spawn", synchronous):
        accepted = fixtures._accept_turn(client, "help")
    assert "message.completed" in client.get(accepted["stream_url"]).text


def test_delivery_event_and_registry_commit_before_graph_ack(tmp_path: Path):
    from eda_platform.agents import agent_recovery

    _, app, client = setup(tmp_path)
    with (
        patch.object(app.state.chat_service, "_spawn", synchronous),
        patch.object(
            agent_recovery,
            "mark_turn_delivered",
            side_effect=OSError("ack lost"),
        ),
    ):
        accepted = fixtures._accept_turn(client, "help")
    assert "message.completed" in client.get(accepted["stream_url"]).text
    assert (
        client.get(f"/api/v1/sessions/{fixtures.RUN}/chat/recoverable-turns").json()["turns"] == []
    )


def test_queued_turn_rejects_changed_data_sharing_before_first_model(tmp_path: Path):
    from types import SimpleNamespace

    from eda_platform.application.services.chat_service import ChatValidationError
    from eda_platform.core.llm import LLMSettings

    _, app, _ = setup(tmp_path)
    service = app.state.chat_service
    with patch.object(service, "_spawn", lambda *_: None):
        accepted = service.send_message(fixtures.RUN, text="help", llm="offline")
    with pytest.raises(ChatValidationError, match="settings changed"):
        service.resume_turn(
            fixtures.RUN,
            accepted.message_id,
            effective_settings=SimpleNamespace(
                llm=LLMSettings(),
                payload_policy="schema_only",
            ),
        )


def test_recovery_blocks_unknown_child_graph_effect(tmp_path: Path):
    from eda_platform.core.graph_execution import graph_execution
    from eda_platform.core.trace_correlation import trace_execution_scope

    root, _, client = setup(tmp_path)
    message_id = "a" * 32
    fixtures._seed_native_checkpoint(root, message_id)
    directory = ArtifactStore(root).session_dir(fixtures.PROJECT, fixtures.RUN)
    with trace_execution_scope(execution_id="chat:" + message_id):
        with graph_execution(
            GraphPersistence(directory, "child-code"), definition="code", inputs={}
        ) as execution:
            with pytest.raises(TimeoutError):
                execution.model_effect(
                    "generate:1", {}, lambda: (_ for _ in ()).throw(TimeoutError())
                )
    turns = client.get(f"/api/v1/sessions/{fixtures.RUN}/chat/recoverable-turns").json()["turns"]
    assert turns[0]["status"] == "blocked"


def test_chat_message_count_repairs_index_after_file_only_commit(tmp_path: Path):
    import sqlite3

    from eda_platform.schemas.chat import ChatMessage

    root, _, client = setup(tmp_path)
    store = ArtifactStore(root)
    first = ChatMessage(turn_id="count-turn", role="user", content="help").model_dump_json()
    second = ChatMessage(turn_id="count-turn", role="assistant", content="answer").model_dump_json()
    for line in (first, second):
        store.append_chat_line(fixtures.PROJECT, fixtures.RUN, line)
    detail = client.get(f"/api/v1/sessions/{fixtures.RUN}").json()
    assert detail["chat_message_count"] == 2
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE sessions SET chat_message_count=0,chat_indexed_bytes=0 WHERE session_id=?",
            (fixtures.RUN,),
        )
    store.append_chat_line(fixtures.PROJECT, fixtures.RUN, second, deduplicate=True)
    assert client.get(f"/api/v1/sessions/{fixtures.RUN}").json()["chat_message_count"] == 2
    assert len(client.get(f"/api/v1/sessions/{fixtures.RUN}/chat/messages").json()["messages"]) == 2
