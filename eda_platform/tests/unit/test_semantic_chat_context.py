from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel
from semantic_test_helpers import save_seeds

from eda_platform.core.semantic import (
    FieldMeaning,
    MetricDefinition,
    SemanticSeeds,
    VerifiedAnswer,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.chat import _semantic_seed_context, run_chat_turn
from eda_platform.schemas.plans import AnalysisPlan, Intent
from eda_platform.tools.loader import load_csv

T = TypeVar("T", bound=BaseModel)


class ScriptedLLM:
    def __init__(self, responses: list[BaseModel]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append({"task": task, "payload": payload})
        return cast(T, self.responses.pop(0))

    def text(self, *, task: str, payload: dict) -> str:
        return ""


def _orders_csv(tmp_path: Path) -> Path:
    orders = tmp_path / "orders.csv"
    orders.write_text("region,amount\nEast,10\nWest,20\n", encoding="utf-8")
    return orders


def _store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "run_demo")
    return store


def _new_analysis_intent() -> Intent:
    return Intent(
        kind="new_analysis",
        params={},
        confidence=0.91,
        raw_message="sales by region",
    )


def _plan() -> AnalysisPlan:
    return AnalysisPlan(
        question="sales by region",
        dataset_names=["orders"],
        columns=["region", "amount"],
        filters=[],
        sql="select region, sum(amount) as total_amount from orders group by region",
        method="grouped_aggregate",
        rationale="Aggregate sales by region.",
        needs_approval=False,
        estimated_scan="small",
    )


def test_semantic_seed_context_includes_current_seed_models() -> None:
    seeds = SemanticSeeds(
        field_meanings=[
            FieldMeaning(
                dataset="orders",
                column="amount",
                meaning="Gross merchandise value before refunds.",
                unit="USD",
                aliases=["gmv", "gross_sales"],
            )
        ],
        metric_definitions=[
            MetricDefinition(
                name="Net revenue",
                definition="Revenue after refunds.",
                formula="sum(amount - refund_amount)",
                caveats="Excludes tax.",
            )
        ],
        verified_answers=[
            VerifiedAnswer(
                question="What was Q3 revenue?",
                answer="$4.2M.",
                evidence_note="Audited close.",
            )
        ],
    )

    context = _semantic_seed_context(seeds)

    assert context == [
        {
            "kind": "field_meaning",
            "identifier": "orders.amount",
            "meaning": "Gross merchandise value before refunds.",
            "unit": "USD",
            "aliases": "gmv, gross_sales",
        },
        {
            "kind": "metric_definition",
            "name": "Net revenue",
            "definition": "Revenue after refunds.",
            "formula": "sum(amount - refund_amount)",
            "caveats": "Excludes tax.",
        },
        {
            "kind": "verified_answer",
            "question": "What was Q3 revenue?",
            "answer": "$4.2M.",
            "evidence_note": "Audited close.",
        },
    ]


def test_run_chat_turn_passes_saved_semantic_seeds_to_planner(tmp_path: Path) -> None:
    orders = _orders_csv(tmp_path)
    store = _store(tmp_path)
    save_seeds(
        store.project_dir("project_demo"),
        SemanticSeeds(
            field_meanings=[
                FieldMeaning(
                    dataset="orders",
                    column="amount",
                    meaning="Sales amount.",
                    unit="USD",
                )
            ],
            metric_definitions=[
                MetricDefinition(name="Revenue", definition="Sum of order amount.")
            ],
            verified_answers=[
                VerifiedAnswer(
                    question="Which region led last period?",
                    answer="West led by revenue.",
                    evidence_note="Finance review.",
                )
            ],
        ),
    )
    llm = ScriptedLLM([_new_analysis_intent(), _plan()])

    result = run_chat_turn(
        "sales by region",
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        project_id="project_demo",
        session_id="run_demo",
        llm=llm,
        store=store,
    )

    planner_payload = llm.calls[1]["payload"]
    assert result.status == "answer"
    assert planner_payload["semantic_seeds"] == [
        {
            "kind": "field_meaning",
            "identifier": "orders.amount",
            "meaning": "Sales amount.",
            "unit": "USD",
        },
        {
            "kind": "metric_definition",
            "name": "Revenue",
            "definition": "Sum of order amount.",
        },
        {
            "kind": "verified_answer",
            "question": "Which region led last period?",
            "answer": "West led by revenue.",
            "evidence_note": "Finance review.",
        },
    ]


def test_run_chat_turn_sends_empty_semantic_seeds_when_file_missing(
    tmp_path: Path,
) -> None:
    orders = _orders_csv(tmp_path)
    store = _store(tmp_path)
    llm = ScriptedLLM([_new_analysis_intent(), _plan()])

    result = run_chat_turn(
        "sales by region",
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        project_id="project_demo",
        session_id="run_demo",
        llm=llm,
        store=store,
    )

    planner_payload = llm.calls[1]["payload"]
    assert result.status == "answer"
    assert planner_payload["semantic_seeds"] == []


def test_run_chat_turn_ignores_corrupt_semantic_seeds(tmp_path: Path) -> None:
    orders = _orders_csv(tmp_path)
    store = _store(tmp_path)
    seeds_path = store.project_dir("project_demo") / "semantic" / "seeds.json"
    seeds_path.parent.mkdir(parents=True, exist_ok=True)
    seeds_path.write_text("{not valid json", encoding="utf-8")
    llm = ScriptedLLM([_new_analysis_intent(), _plan()])

    result = run_chat_turn(
        "sales by region",
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        project_id="project_demo",
        session_id="run_demo",
        llm=llm,
        store=store,
    )

    planner_payload = llm.calls[1]["payload"]
    assert result.status == "answer"
    assert planner_payload["semantic_seeds"] == []
