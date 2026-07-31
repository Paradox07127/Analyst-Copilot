from __future__ import annotations

from pathlib import Path
from typing import Any

from eda_platform.core.llm import LLMToolCall, LLMToolResponse
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.chat import run_chat_turn
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.tools.loader import load_csv


class _ScriptedToolLLM:
    def __init__(self, responses: list[LLMToolResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolResponse:
        assert task == "chat_agent_tool_loop"
        assert {tool["name"] for tool in tools} >= {
            "inspect_data_catalog",
            "run_sql",
            "list_saved_skills",
        }
        self.calls.append(messages)
        return self.responses.pop(0)

    def structured(self, *, task: str, schema: type[Any], payload: dict[str, Any]) -> Any:
        raise AssertionError(
            "The agentic path should use native tool calls, not the legacy planner."
        )

    def text(self, *, task: str, payload: dict[str, Any]) -> str:
        raise AssertionError("The agentic path should use native tool calls, not text completion.")

    def last_usage(self) -> None:
        return None


def test_agentic_chat_can_inspect_then_query_loaded_data(tmp_path: Path) -> None:
    source = tmp_path / "orders.csv"
    source.write_text("region,amount\nEast,10\nWest,20\nEast,5\n", encoding="utf-8")
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "run_demo")
    llm = _ScriptedToolLLM(
        [
            LLMToolResponse(
                tool_calls=[
                    LLMToolCall(
                        call_id="catalog_1",
                        name="inspect_data_catalog",
                        arguments={},
                    )
                ]
            ),
            LLMToolResponse(
                tool_calls=[
                    LLMToolCall(
                        call_id="sql_1",
                        name="run_sql",
                        arguments={
                            "purpose": "Compare total amount by region.",
                            "sql": (
                                "SELECT region, SUM(amount) AS total "
                                "FROM orders GROUP BY region ORDER BY total DESC"
                            ),
                        },
                    )
                ]
            ),
            LLMToolResponse(content="West has the highest total amount. Sources: query result."),
        ]
    )

    result = run_chat_turn(
        "Which region has the highest total amount?",
        datasets=[load_csv(source, dataset_id="orders_1")],
        project_id="project_demo",
        session_id="run_demo",
        llm=llm,  # type: ignore[arg-type]
        store=store,
    )

    assert result.status == "answer"
    assert "West" in result.message
    sql_artifact = next(
        artifact for artifact in result.artifacts if artifact.type is ArtifactType.SQL_RESULT
    )
    assert sql_artifact.payload["rows_preview"][0]["region"] == "West"
    names = [
        event.name
        for event in store.list_trace_events(
            project_id="project_demo",
            session_id="run_demo",
        )
    ]
    assert names.count("inspect_data_catalog") == 2
    assert names.count("run_sql") == 2
