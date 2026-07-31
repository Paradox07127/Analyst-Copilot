from __future__ import annotations

from pathlib import Path
from typing import TypeVar, cast

from pydantic import BaseModel

from eda_platform.schemas.plans import AnalysisPlan, Intent
from eda_platform.tools.loader import load_csv
from eda_platform.tools.nl2sql_eval import NL2SQLEvalCase, run_nl2sql_eval_case

T = TypeVar("T", bound=BaseModel)


class ScriptedEvalLLM:
    def __init__(self, responses: list[BaseModel]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append(task)
        return cast(T, self.responses.pop(0))


def test_nl2sql_eval_runs_chat_path_and_checks_expected_rows(tmp_path: Path) -> None:
    orders = tmp_path / "orders.csv"
    orders.write_text(
        "region,amount\n"
        "East,10\n"
        "West,20\n"
        "East,5\n",
        encoding="utf-8",
    )
    llm = ScriptedEvalLLM(
        [
            Intent(
                kind="new_analysis",
                params={},
                confidence=0.92,
                raw_message="total sales by region",
            ),
            AnalysisPlan(
                question="total sales by region",
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

    outcome = run_nl2sql_eval_case(
        NL2SQLEvalCase(
            name="sales_by_region",
            question="total sales by region",
            expected_rows_preview=[
                {"region": "East", "total_amount": 15.0},
                {"region": "West", "total_amount": 20.0},
            ],
        ),
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        llm=llm,
    )

    assert outcome.passed is True
    assert outcome.validation_status == "pass"
    assert outcome.actual_sql.startswith("select region")
    assert llm.calls == ["m3_route_intent", "m3_build_plan"]
