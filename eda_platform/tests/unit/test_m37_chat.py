from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from eda_platform.core.llm import LLMResultMetadata, LLMUsage
from eda_platform.core.permissions import action_hash, analysis_plan_action
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.chat import (
    append_chat_message,
    load_chat_messages,
    run_chat_turn,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType, SqlResult
from eda_platform.schemas.chat import ChatMessage
from eda_platform.schemas.plans import AnalysisPlan, Intent
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.quality import scan_quality

T = TypeVar("T", bound=BaseModel)


class ScriptedLLM:
    """Fake structured LLM: pops responses in order; Exception entries are raised."""

    def __init__(self, responses: list[BaseModel | Exception]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append({"task": task, "payload": payload})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return cast(T, response)

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> LLMResultMetadata:
        return LLMResultMetadata(
            provider="fake",
            model="fake-model",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            estimated_cost_usd=0.001,
        )


def _new_analysis_intent(message: str = "sales by region") -> Intent:
    return Intent(kind="new_analysis", params={}, confidence=0.91, raw_message=message)


def _plan(
    *,
    sql: str,
    columns: list[str],
    needs_approval: bool = False,
    dataset_names: list[str] | None = None,
) -> AnalysisPlan:
    return AnalysisPlan(
        question="sales by region",
        dataset_names=dataset_names or ["orders"],
        columns=columns,
        filters=[],
        sql=sql,
        method="grouped_aggregate",
        rationale="Aggregate sales by region.",
        needs_approval=needs_approval,
        estimated_scan="small",
    )


def _orders_csv(tmp_path: Path) -> Path:
    orders = tmp_path / "orders.csv"
    orders.write_text(
        "region,amount\nEast,10\nWest,20\nEast,5\n",
        encoding="utf-8",
    )
    return orders


def _store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "run_demo")
    return store


# --------------------------------------------------------------------------- T4


def test_planner_llm_runtime_error_returns_error_status_with_trace(tmp_path: Path) -> None:
    orders = _orders_csv(tmp_path)
    store = _store(tmp_path)
    # First structured call is routing (valid intent), second is build_plan -> raises.
    llm = ScriptedLLM(
        [
            _new_analysis_intent(),
            RuntimeError("LLM provider unreachable"),
        ]
    )

    result = run_chat_turn(
        "sales by region",
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        project_id="project_demo",
        session_id="run_demo",
        llm=llm,
        store=store,
    )

    assert result.status == "error"
    assert result.plan is None
    assert "unreachable" in result.message.lower() or "could not build" in result.message.lower()

    events = store.list_trace_events(project_id="project_demo", session_id="run_demo")
    failed = [event for event in events if event.event_type == "chat_turn_failed"]
    assert len(failed) == 1
    assert failed[0].summary["error_type"] == "RuntimeError"
    assert failed[0].summary["stage"] == "planning"


def test_planner_hallucinated_column_returns_error_status(tmp_path: Path) -> None:
    orders = _orders_csv(tmp_path)
    store = _store(tmp_path)
    # Routing intent, then build_plan returns a bad-SQL plan twice (dry-run rejects
    # both). build_plan raises ValueError after its internal retry.
    llm = ScriptedLLM(
        [
            _new_analysis_intent(),
            _plan(sql="select nonexistent_col from orders", columns=["region"]),
            _plan(sql="select also_missing from orders", columns=["region"]),
        ]
    )

    result = run_chat_turn(
        "sales by region",
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        project_id="project_demo",
        session_id="run_demo",
        llm=llm,
        store=store,
    )

    assert result.status == "error"
    events = store.list_trace_events(project_id="project_demo", session_id="run_demo")
    assert any(event.event_type == "chat_turn_failed" for event in events)


def test_duckdb_execution_error_returns_error_status(tmp_path: Path) -> None:
    orders = _orders_csv(tmp_path)
    store = _store(tmp_path)
    # Plan passes catalog/column validation and EXPLAIN dry-run, but fails at real
    # execution: casting the text 'region' to INTEGER binds fine yet raises a
    # duckdb.ConversionException at runtime.
    llm = ScriptedLLM(
        [
            _new_analysis_intent(),
            _plan(
                sql="select region, cast(region as integer) as boom from orders group by region",
                columns=["region"],
            ),
        ]
    )

    result = run_chat_turn(
        "sales by region",
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        project_id="project_demo",
        session_id="run_demo",
        llm=llm,
        store=store,
    )

    assert result.status == "error"
    assert result.plan is not None  # plan built successfully; execution is what failed
    events = store.list_trace_events(project_id="project_demo", session_id="run_demo")
    failed = [event for event in events if event.event_type == "chat_turn_failed"]
    assert len(failed) == 1
    assert failed[0].summary["stage"] == "execution"


def test_no_store_still_returns_error_without_raising(tmp_path: Path) -> None:
    orders = _orders_csv(tmp_path)
    llm = ScriptedLLM([_new_analysis_intent(), RuntimeError("boom")])

    result = run_chat_turn(
        "sales by region",
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        project_id="project_demo",
        session_id="run_demo",
        llm=llm,
    )

    assert result.status == "error"


def test_out_of_scope_status_is_refused(tmp_path: Path) -> None:
    orders = _orders_csv(tmp_path)
    llm = ScriptedLLM(
        [Intent(kind="out_of_scope", params={}, confidence=0.9, raw_message="weather?")]
    )

    result = run_chat_turn(
        "what's the weather?",
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        project_id="project_demo",
        session_id="run_demo",
        llm=llm,
    )

    assert result.status == "refused"


# --------------------------------------------------------------------------- T7 approval


def test_needs_approval_then_approved_plan_executes(tmp_path: Path) -> None:
    orders = _orders_csv(tmp_path)
    store = _store(tmp_path)
    approval_plan = _plan(
        sql="select region, sum(amount) as total_amount from orders group by region",
        columns=["region", "amount"],
        needs_approval=True,
    )
    llm = ScriptedLLM([_new_analysis_intent(), approval_plan])

    blocked = run_chat_turn(
        "sales by region",
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        project_id="project_demo",
        session_id="run_demo",
        llm=llm,
        store=store,
    )

    assert blocked.status == "awaiting_approval"
    assert blocked.plan is not None
    assert blocked.validation is None
    assert {artifact.type for artifact in blocked.artifacts} == {ArtifactType.CHAT_TURN_PLAN}

    # Re-entry without its matching approval hash cannot execute the plan.
    empty_llm = ScriptedLLM([])
    denied = run_chat_turn(
        "sales by region",
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        project_id="project_demo",
        session_id="run_demo",
        llm=empty_llm,
        store=store,
        approved_plan=blocked.plan,
    )

    assert denied.status == "refused"
    assert empty_llm.calls == []
    assert ArtifactType.SQL_RESULT not in {artifact.type for artifact in denied.artifacts}

    # Re-entry with the approved plan: no LLM should be consulted (responses empty).
    empty_llm = ScriptedLLM([])
    approved = run_chat_turn(
        "sales by region",
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        project_id="project_demo",
        session_id="run_demo",
        llm=empty_llm,
        store=store,
        approved_plan=blocked.plan,
        approved_action_hash=action_hash(analysis_plan_action(blocked.plan)),
    )

    assert approved.status == "answer"
    assert empty_llm.calls == []
    assert ArtifactType.SQL_RESULT in {artifact.type for artifact in approved.artifacts}
    sql_artifact = next(
        artifact
        for artifact in approved.artifacts
        if artifact.type is ArtifactType.SQL_RESULT
    )
    sql_result = SqlResult.model_validate(sql_artifact.payload)
    assert sql_result.row_count == 2


# --------------------------------------------------------------------------- T5


def _artifacts_for_missing(tmp_path: Path) -> list[Artifact]:
    customers = tmp_path / "customers.csv"
    customers.write_text(
        "region,note\nEast,ok\nWest,\n,\n",
        encoding="utf-8",
    )
    loaded = load_csv(customers, dataset_id="ds_customers")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    return [profile]


def _ask(message: str, artifacts: list[Artifact], tmp_path: Path):
    orders = _orders_csv(tmp_path)
    llm = ScriptedLLM(
        [Intent(kind="ask_from_artifacts", params={}, confidence=0.9, raw_message=message)]
    )
    return run_chat_turn(
        message,
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        project_id="project_demo",
        session_id="run_demo",
        llm=llm,
        artifacts=artifacts,
    )


def test_ask_missing_values_returns_real_content_with_ids(tmp_path: Path) -> None:
    artifacts = _artifacts_for_missing(tmp_path)
    result = _ask("which columns have the most missing values?", artifacts, tmp_path)

    assert result.status == "answer"
    assert "missing" in result.message.lower()
    assert "note" in result.message  # note column has missing values
    assert artifacts[0].id in result.message  # cited source id
    assert result.artifacts == [artifacts[0]]


def test_ask_missing_values_chinese_keyword(tmp_path: Path) -> None:
    artifacts = _artifacts_for_missing(tmp_path)
    result = _ask("哪一列缺失最多？", artifacts, tmp_path)

    assert result.status == "answer"
    assert artifacts[0].id in result.message
    assert result.artifacts == [artifacts[0]]


def test_ask_quality_groups_by_severity_with_ids(tmp_path: Path) -> None:
    empty_col = tmp_path / "empty.csv"
    # 'blank' column is 100% missing -> critical empty_column issue.
    empty_col.write_text(
        "region,blank\nEast,\nWest,\nEast,\n",
        encoding="utf-8",
    )
    loaded = load_csv(empty_col, dataset_id="ds_empty")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    quality = scan_quality(profile, project_id="project_demo", session_id="run_demo")

    result = _ask("what quality issues are there?", [profile, quality], tmp_path)

    assert result.status == "answer"
    assert "quality" in result.message.lower() or "severity" in result.message.lower()
    assert quality.id in result.message
    assert quality in result.artifacts


def test_ask_size_returns_rows_and_columns(tmp_path: Path) -> None:
    artifacts = _artifacts_for_missing(tmp_path)
    result = _ask("how many rows and columns?", artifacts, tmp_path)

    assert result.status == "answer"
    assert "rows" in result.message.lower()
    assert artifacts[0].id in result.message


def test_ask_primary_keys_returns_candidates(tmp_path: Path) -> None:
    people = tmp_path / "people.csv"
    people.write_text(
        "user_id,city\n1,East\n2,West\n3,East\n",
        encoding="utf-8",
    )
    loaded = load_csv(people, dataset_id="ds_people")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")

    result = _ask("what are the primary key candidates?", [profile], tmp_path)

    assert result.status == "answer"
    assert profile.id in result.message
    # user_id is an id-typed, unique, non-null column -> PK candidate.
    assert "user_id" in result.message


def test_ask_fallback_returns_inventory_not_type_counts(tmp_path: Path) -> None:
    artifacts = _artifacts_for_missing(tmp_path)
    result = _ask("tell me about the data", artifacts, tmp_path)

    assert result.status == "answer"
    assert "customers" in result.message  # dataset name, not "DatasetProfile: 1"
    assert artifacts[0].id in result.message


def test_ask_from_artifacts_empty_returns_placeholder(tmp_path: Path) -> None:
    result = _ask("what's missing?", [], tmp_path)
    assert result.status == "answer"
    assert "No report artifacts" in result.message
    assert result.artifacts == []


# --------------------------------------------------------------------------- T6


def test_chat_message_persistence_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    user_msg = ChatMessage(role="user", content="sales by region")
    assistant_msg = ChatMessage(
        role="assistant",
        content="Ran SQL analysis: 2 rows returned.",
        status="answer",
        artifact_refs=["sql_abc", "chatplan_def"],
    )

    append_chat_message(store, "project_demo", "run_demo", user_msg)
    append_chat_message(store, "project_demo", "run_demo", assistant_msg)

    loaded = load_chat_messages(store, "project_demo", "run_demo")

    assert len(loaded) == 2
    assert loaded[0].role == "user"
    assert loaded[0].content == "sales by region"
    assert loaded[1].role == "assistant"
    assert loaded[1].artifact_refs == ["sql_abc", "chatplan_def"]
    assert loaded[1].status == "answer"


def test_load_chat_messages_missing_session_is_empty(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert load_chat_messages(store, "project_demo", "never_written") == []


def test_chat_session_path_location(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.start_session("project_demo", "sess1")
    append_chat_message(
        store, "project_demo", "sess1", ChatMessage(role="user", content="hi")
    )
    expected = store.project_dir("project_demo") / "chat" / "sess1.jsonl"
    assert expected.exists()


# --------------------------------------------------------------------------- T7 charts


def _sql_result(
    columns: list[str], rows: list[dict[str, Any]], *, row_count: int | None = None
) -> SqlResult:
    return SqlResult(
        sql="select ...",
        columns=columns,
        dtypes={column: "object" for column in columns},
        rows_preview=rows,
        row_count=row_count if row_count is not None else len(rows),
    )


