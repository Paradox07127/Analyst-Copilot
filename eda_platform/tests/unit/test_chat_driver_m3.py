from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from eda_platform.core.llm import LLMResultMetadata, LLMUsage
from eda_platform.core.permissions import action_hash, analysis_plan_action
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.chat import run_chat_turn
from eda_platform.schemas.artifacts import Artifact, ArtifactType, SqlResult
from eda_platform.schemas.chat import SqlResultValidation
from eda_platform.schemas.plans import AnalysisPlan, Intent
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.sql_result_validator import validate_sql_result

T = TypeVar("T", bound=BaseModel)


class FakeStructuredLLM:
    def __init__(self, responses: list[BaseModel]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append({"task": task, "payload": payload})
        return cast(T, self.responses.pop(0))

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> LLMResultMetadata:
        return LLMResultMetadata(
            provider="fake",
            model="fake-model",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            estimated_cost_usd=0.001,
        )


def test_run_chat_turn_executes_new_analysis_and_returns_sql_artifact(
    tmp_path: Path,
) -> None:
    orders = tmp_path / "orders.csv"
    orders.write_text(
        "region,amount\n"
        "East,10\n"
        "West,20\n"
        "East,5\n",
        encoding="utf-8",
    )
    llm = FakeStructuredLLM(
        [
            Intent(
                kind="new_analysis",
                params={},
                confidence=0.91,
                raw_message="sales by region",
            ),
            AnalysisPlan(
                question="sales by region",
                dataset_names=["orders"],
                columns=["region", "amount"],
                filters=[],
                sql=(
                    "select region, sum(amount) as total_amount "
                    "from orders group by region order by region"
                ),
                method="grouped_aggregate",
                rationale="Aggregate sales by region.",
                needs_approval=False,
                estimated_scan="small",
            ),
        ]
    )

    result = run_chat_turn(
        "sales by region",
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        project_id="project_demo",
        session_id="run_demo",
        llm=llm,
        preview_rows=10,
    )

    assert result.intent.kind == "new_analysis"
    assert result.plan is not None
    assert result.validation == SqlResultValidation(status="pass", findings=[])
    assert {artifact.type for artifact in result.artifacts} == {
        ArtifactType.CHAT_TURN_PLAN,
        ArtifactType.SQL_RESULT,
    }
    artifact = next(
        artifact
        for artifact in result.artifacts
        if artifact.type is ArtifactType.SQL_RESULT
    )
    sql_result = SqlResult.model_validate(artifact.payload)
    assert artifact.type is ArtifactType.SQL_RESULT
    assert sql_result.rows_preview == [
        {"region": "East", "total_amount": 15.0},
        {"region": "West", "total_amount": 20.0},
    ]
    assert "2 rows" in result.message
    assert [call["task"] for call in llm.calls] == ["m3_route_intent", "m3_build_plan"]


def test_schema_only_payload_policy_removes_planner_value_context(
    tmp_path: Path,
) -> None:
    orders = tmp_path / "orders.csv"
    orders.write_text("region,amount\nEast,10\n", encoding="utf-8")
    llm = FakeStructuredLLM(
        [
            Intent(
                kind="new_analysis",
                confidence=0.99,
                raw_message="sales by region",
            ),
            AnalysisPlan(
                question="sales by region",
                dataset_names=["orders"],
                columns=["region", "amount"],
                sql="select region, sum(amount) from orders group by region",
                method="grouped_aggregate",
                rationale="Aggregate sales by region.",
                estimated_scan="small",
            ),
        ]
    )

    run_chat_turn(
        "sales by region",
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        project_id="project_demo",
        session_id="run_demo",
        llm=llm,
        value_context={"orders.region": ["East"]},
        payload_policy="schema_only",
    )

    planner_payload = next(
        call["payload"] for call in llm.calls if call["task"] == "m3_build_plan"
    )
    assert planner_payload["value_context"] == {}


def test_run_chat_turn_meta_help_does_not_execute_sql(tmp_path: Path) -> None:
    orders = tmp_path / "orders.csv"
    orders.write_text("region,amount\nEast,10\n", encoding="utf-8")
    llm = FakeStructuredLLM(
        [
            Intent(
                kind="meta_help",
                params={},
                confidence=0.95,
                raw_message="你能做什么？",
            )
        ]
    )

    result = run_chat_turn(
        "你能做什么？",
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        project_id="project_demo",
        session_id="run_demo",
        llm=llm,
    )

    assert result.intent.kind == "meta_help"
    assert result.plan is None
    assert result.artifacts == []
    assert "loaded datasets" in result.message


def test_run_chat_turn_records_debug_trace_events(tmp_path: Path) -> None:
    orders = tmp_path / "orders.csv"
    orders.write_text(
        "region,amount\n"
        "East,10\n"
        "West,20\n",
        encoding="utf-8",
    )
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "run_demo")
    llm = FakeStructuredLLM(
        [
            Intent(
                kind="new_analysis",
                params={},
                confidence=0.91,
                raw_message="sales by region",
            ),
            AnalysisPlan(
                question="sales by region",
                dataset_names=["orders"],
                columns=["region", "amount"],
                filters=[],
                sql=(
                    "select region, sum(amount) as total_amount "
                    "from orders group by region order by region"
                ),
                method="grouped_aggregate",
                rationale="Aggregate sales by region.",
                needs_approval=False,
                estimated_scan="small",
            ),
        ]
    )

    run_chat_turn(
        "sales by region",
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        project_id="project_demo",
        session_id="run_demo",
        llm=llm,
        store=store,
    )

    events = store.list_trace_events(project_id="project_demo", session_id="run_demo")

    assert [(event.event_type, event.name) for event in events] == [
        ("agent_intent", "m3_route_intent"),
        ("llm_call", "m3_route_intent"),
        ("agent_plan", "m3_build_plan"),
        ("llm_call", "m3_build_plan"),
        ("tool_completed", "run_sql"),
        ("validator_result", "validate_sql_result"),
    ]
    assert events[0].summary["intent"] == "new_analysis"
    assert events[2].summary["method"] == "grouped_aggregate"
    assert events[4].summary["row_count"] == 2
    assert events[5].summary["status"] == "pass"


def test_run_chat_turn_builds_masked_value_context_and_plan_artifact(
    tmp_path: Path,
) -> None:
    customers = tmp_path / "customers.csv"
    customers.write_text(
        "region,email,amount\n"
        "East,a@example.com,10\n"
        "East,b@example.com,20\n"
        "West,c@example.com,30\n",
        encoding="utf-8",
    )
    loaded = load_csv(customers, dataset_id="ds_customers")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "run_demo")
    llm = FakeStructuredLLM(
        [
            Intent(
                kind="new_analysis",
                params={},
                confidence=0.91,
                raw_message="sales by region",
            ),
            AnalysisPlan(
                question="sales by region",
                dataset_names=["customers"],
                columns=["region", "amount"],
                filters=[],
                sql=(
                    "select region, sum(amount) as total_amount "
                    "from customers group by region order by region"
                ),
                method="grouped_aggregate",
                rationale="Aggregate sales by region.",
                needs_approval=False,
                estimated_scan="small",
            ),
        ]
    )

    result = run_chat_turn(
        "sales by region",
        datasets=[loaded],
        project_id="project_demo",
        session_id="run_demo",
        llm=llm,
        artifacts=[profile],
        store=store,
    )

    planner_payload = llm.calls[1]["payload"]
    assert planner_payload["value_context"]["customers.region"] == ["East", "West"]
    assert planner_payload["value_context"]["customers.email"] == ["[PII:email]"]
    assert {artifact.type for artifact in result.artifacts} == {
        ArtifactType.CHAT_TURN_PLAN,
        ArtifactType.SQL_RESULT,
    }
    stored_types = {
        artifact.type
        for artifact in store.list_artifacts(project_id="project_demo", session_id="run_demo")
    }
    assert ArtifactType.CHAT_TURN_PLAN in stored_types


def test_run_chat_turn_requires_matching_approval_hash_before_expensive_plan_executes(
    tmp_path: Path,
) -> None:
    orders = tmp_path / "orders.csv"
    orders.write_text("region,amount\nEast,10\n", encoding="utf-8")

    def llm() -> FakeStructuredLLM:
        return FakeStructuredLLM(
            [
                Intent(
                    kind="new_analysis",
                    params={},
                    confidence=0.91,
                    raw_message="scan everything",
                ),
                AnalysisPlan(
                    question="scan everything",
                    dataset_names=["orders"],
                    columns=["region", "amount"],
                    filters=[],
                    sql="select * from orders",
                    method="full_scan",
                    rationale="Inspect all records.",
                    needs_approval=True,
                    estimated_scan="large",
                ),
            ]
        )

    blocked = run_chat_turn(
        "scan everything",
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        project_id="project_demo",
        session_id="run_demo",
        llm=llm(),
    )
    assert blocked.plan is not None
    mismatched_plan = blocked.plan.model_copy(update={"sql": "select amount from orders"})
    denied = run_chat_turn(
        "scan everything",
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        project_id="project_demo",
        session_id="run_demo",
        llm=FakeStructuredLLM([]),
        approved_plan=blocked.plan,
        approved_action_hash=action_hash(analysis_plan_action(mismatched_plan)),
    )

    assert blocked.validation is None
    assert {artifact.type for artifact in blocked.artifacts} == {ArtifactType.CHAT_TURN_PLAN}
    assert blocked.pending_action is not None
    assert blocked.pending_action["action_hash"] == action_hash(analysis_plan_action(blocked.plan))
    assert denied.status == "refused"
    assert ArtifactType.SQL_RESULT not in {artifact.type for artifact in denied.artifacts}


def test_validate_sql_result_warns_on_empty_rows() -> None:
    artifact = Artifact(
        id="sql_empty",
        type=ArtifactType.SQL_RESULT,
        project_id="project_demo",
        session_id="run_demo",
        payload=SqlResult(
            sql="select * from orders where amount > 1000",
            columns=["region", "amount"],
            dtypes={"region": "object", "amount": "int64"},
            rows_preview=[],
            row_count=0,
            truncated=False,
        ).model_dump(mode="json"),
    )
    plan = AnalysisPlan(
        question="large orders",
        dataset_names=["orders"],
        columns=["region", "amount"],
        filters=["amount > 1000"],
        sql="select * from orders where amount > 1000",
        method="filter",
        rationale="Find large orders.",
        needs_approval=False,
        estimated_scan="small",
    )

    validation = validate_sql_result(artifact, plan)

    assert validation.status == "warn"
    assert validation.findings == ["Query returned zero rows."]
