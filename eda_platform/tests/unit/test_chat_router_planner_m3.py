from __future__ import annotations

from typing import Any, TypeVar, cast

import pytest
from pydantic import BaseModel

from eda_platform.agents.chat_router import route_intent
from eda_platform.agents.planner import build_plan, validate_plan_references
from eda_platform.schemas.plans import AnalysisPlan, Intent

T = TypeVar("T", bound=BaseModel)


class FakeStructuredLLM:
    def __init__(self, response: BaseModel) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append({"task": task, "payload": payload})
        return cast(T, self.response)

    def text(self, *, task: str, payload: dict) -> str:
        return ""


def test_route_intent_uses_structured_llm_when_confident() -> None:
    llm = FakeStructuredLLM(
        Intent(
            kind="new_analysis",
            params={"metric": "amount"},
            confidence=0.91,
            raw_message="Compare sales by region",
        )
    )

    intent = route_intent("Compare sales by region", llm=llm)

    assert intent.kind == "new_analysis"
    assert intent.params == {"metric": "amount"}
    assert llm.calls[0]["task"] == "m3_route_intent"


def test_route_intent_falls_back_to_rules_for_low_confidence() -> None:
    llm = FakeStructuredLLM(
        Intent(kind="new_analysis", params={}, confidence=0.1, raw_message="What can you do?")
    )

    intent = route_intent("What can you do?", llm=llm)

    assert intent.kind == "meta_help"
    assert intent.confidence >= 0.8


def test_build_plan_validates_llm_plan_references_real_tables_and_columns() -> None:
    plan = AnalysisPlan(
        question="各区域销售额是多少？",
        dataset_names=["orders"],
        columns=["region", "amount"],
        filters=[],
        sql="select region, sum(amount) as total_amount from orders group by region",
        method="grouped_aggregate",
        rationale="Compare sales by region.",
        needs_approval=False,
        estimated_scan="small",
    )
    llm = FakeStructuredLLM(plan)

    result = build_plan(
        "各区域销售额是多少？",
        llm=llm,
        catalog_columns={"orders": {"region", "amount"}},
        value_context={"orders.region": ["East", "West"]},
    )

    assert result.sql.startswith("select region")
    assert result.needs_approval is False
    assert llm.calls[0]["payload"]["value_context"] == {"orders.region": ["East", "West"]}


def test_validate_plan_references_blocks_unknown_columns() -> None:
    plan = AnalysisPlan(
        question="按利润看",
        dataset_names=["orders"],
        columns=["profit"],
        filters=[],
        sql="select profit from orders",
        method="select",
        rationale="Unknown column.",
        needs_approval=False,
        estimated_scan="small",
    )

    with pytest.raises(ValueError, match="Unknown column"):
        validate_plan_references(plan, {"orders": {"amount"}})
