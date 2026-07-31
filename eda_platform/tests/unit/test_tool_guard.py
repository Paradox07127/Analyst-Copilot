from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

from eda_platform.core.store import ArtifactStore
from eda_platform.core.tool_guard import (
    ToolGuardError,
    check_column_exists,
    check_column_semantic_type,
    check_enum,
    check_non_empty,
    check_range,
    raise_for_violations,
)
from eda_platform.drivers.chat import run_chat_turn
from eda_platform.schemas.artifacts import AnalysisTable
from eda_platform.schemas.cleaning import CleaningRecipe, CleaningTransform
from eda_platform.schemas.model_card import ModelCard
from eda_platform.schemas.plans import AnalysisPlan, Intent
from eda_platform.schemas.stats import StatTestResult
from eda_platform.tools.cleaning import guard_cleaning_recipe_params
from eda_platform.tools.loader import load_csv
from eda_platform.tools.ml_baseline import guard_baseline_model_params
from eda_platform.tools.stat_tests import guard_stat_test_params

T = TypeVar("T", bound=BaseModel)


class ScriptedStructuredLLM:
    def __init__(self, responses: list[BaseModel]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append({"task": task, "schema": schema, "payload": payload})
        return cast(T, self.responses.pop(0))


def test_range_guard_accepts_and_rejects_with_teachable_feedback() -> None:
    assert check_range("score", 0.8, minimum=0.0, maximum=1.0) is None

    violation = check_range("score", 8, minimum=0.0, maximum=1.0)
    with pytest.raises(ToolGuardError) as exc_info:
        raise_for_violations("question_score", [violation])

    feedback = exc_info.value.to_model_feedback()
    assert "What was wrong" in feedback
    assert "Allowed" in feedback
    assert "How to fix" in feedback
    assert "score" in feedback and "8" in feedback
    assert "0.0" in feedback and "1.0" in feedback


def test_enum_guard_accepts_and_rejects() -> None:
    allowed = ("small", "medium", "large", "unknown")
    assert check_enum("estimated_scan", "small", allowed) is None

    violation = check_enum("estimated_scan", "huge", allowed)
    with pytest.raises(ToolGuardError) as exc_info:
        raise_for_violations("analysis_plan", [violation])

    assert "estimated_scan" in exc_info.value.to_model_feedback()
    assert "small, medium, large, unknown" in exc_info.value.to_model_feedback()


def test_non_empty_guard_accepts_and_rejects() -> None:
    assert check_non_empty("columns", ["amount"]) is None

    violation = check_non_empty("columns", [])
    with pytest.raises(ToolGuardError) as exc_info:
        raise_for_violations("analysis_plan", [violation])

    assert "columns" in exc_info.value.to_model_feedback()
    assert "at least one item" in exc_info.value.to_model_feedback()


def test_column_exists_guard_accepts_and_rejects() -> None:
    assert check_column_exists("target_column", "churned", ["churned", "spend"]) is None

    violation = check_column_exists("target_column", "profit", ["churned", "spend"])
    with pytest.raises(ToolGuardError) as exc_info:
        raise_for_violations("run_baseline_model", [violation])

    feedback = exc_info.value.to_model_feedback()
    assert "profit" in feedback
    assert "churned" in feedback and "spend" in feedback


def test_column_semantic_type_guard_accepts_and_rejects() -> None:
    frame = pd.DataFrame(
        {
            "segment": ["A", "B", "A"],
            "amount": [10.0, 20.0, 30.0],
        }
    )

    assert (
        check_column_semantic_type(
            "value_column",
            "amount",
            frame,
            allowed_semantic_types=("numeric",),
        )
        is None
    )
    violation = check_column_semantic_type(
        "value_column",
        "segment",
        frame,
        allowed_semantic_types=("numeric",),
    )

    with pytest.raises(ToolGuardError) as exc_info:
        raise_for_violations("run_stat_test", [violation])

    feedback = exc_info.value.to_model_feedback()
    assert "value_column" in feedback
    assert "numeric" in feedback
    assert "segment" in feedback


def test_stat_guard_accepts_valid_params_and_rejects_wrong_column_type() -> None:
    frame = pd.DataFrame({"segment": ["A", "B", "A", "B"], "amount": [1, 2, 3, 4]})

    guard_stat_test_params(
        frame,
        test_type="independent_t_test",
        group_column="segment",
        value_column="amount",
        category_column=None,
        comparison_count=1,
    )

    with pytest.raises(ToolGuardError) as exc_info:
        guard_stat_test_params(
            frame,
            test_type="independent_t_test",
            group_column="amount",
            value_column="segment",
            category_column=None,
            comparison_count=1,
        )

    feedback = exc_info.value.to_model_feedback()
    assert "group_column" in feedback
    assert "categorical" in feedback
    assert "value_column" in feedback
    assert "numeric" in feedback


def test_ml_guard_accepts_valid_params_and_rejects_bad_time_column() -> None:
    frame = pd.DataFrame(
        {
            "event_date": pd.date_range("2026-01-01", periods=4, freq="D"),
            "target": [1, 0, 1, 0],
            "amount": [10.0, 20.0, 30.0, 40.0],
        }
    )

    guard_baseline_model_params(
        frame,
        target_column="target",
        time_column="event_date",
        random_state=42,
    )

    with pytest.raises(ToolGuardError) as exc_info:
        guard_baseline_model_params(
            frame,
            target_column="target",
            time_column="amount",
            random_state=42,
        )

    feedback = exc_info.value.to_model_feedback()
    assert "time_column" in feedback
    assert "datetime" in feedback


def test_cleaning_guard_accepts_valid_recipe_and_rejects_missing_target_column() -> None:
    frame = pd.DataFrame({"name": [" Alice ", "Bob"]})
    valid = CleaningRecipe(
        dataset_id="ds_people",
        transforms=[CleaningTransform(type="trim_whitespace", target_column="name")],
    )
    guard_cleaning_recipe_params(frame, valid)

    invalid = CleaningRecipe(
        dataset_id="ds_people",
        transforms=[CleaningTransform(type="trim_whitespace", target_column="missing_name")],
    )
    with pytest.raises(ToolGuardError) as exc_info:
        guard_cleaning_recipe_params(frame, invalid)

    feedback = exc_info.value.to_model_feedback()
    assert "missing_name" in feedback
    assert "name" in feedback


def test_schema_invariants_reject_out_of_range_values_and_empty_llm_plan_fields() -> None:
    StatTestResult(
        dataset_id="ds",
        test_type="independent_t_test",
        statistic=1.2,
        p_value=0.4,
        sample_size=10,
    )
    ModelCard(
        dataset_id="ds",
        task_type="classification",
        target_column="target",
        feature_columns=["amount"],
        split_strategy="random",
        train_rows=8,
        test_rows=2,
        model_type="RandomForestClassifier",
        metrics={"accuracy": 0.75},
    )
    AnalysisTable(
        dataset_id="ds",
        title="Correlations",
        kind="correlation",
        description="Top correlations",
        rows=[{"column_a": "x", "column_b": "y", "pearson": 0.8, "abs_pearson": 0.8}],
    )

    with pytest.raises(ValidationError):
        StatTestResult(
            dataset_id="ds",
            test_type="independent_t_test",
            statistic=1.2,
            p_value=1.2,
            sample_size=10,
        )
    with pytest.raises(ValidationError):
        ModelCard(
            dataset_id="ds",
            task_type="classification",
            target_column="target",
            feature_columns=[],
            split_strategy="random",
            train_rows=8,
            test_rows=2,
            model_type="RandomForestClassifier",
            metrics={"accuracy": 1.2},
        )
    with pytest.raises(ValidationError):
        Intent(kind="new_analysis", confidence=1.4, raw_message="sales by region")
    with pytest.raises(ValidationError):
        AnalysisPlan(
            question="sales by region",
            dataset_names=["orders"],
            columns=[],
            filters=[],
            sql="select count(*) from orders",
            method="aggregate",
            rationale="Count rows.",
        )
    with pytest.raises(ValidationError):
        AnalysisTable(
            dataset_id="ds",
            title="Correlations",
            kind="correlation",
            description="Top correlations",
            rows=[{"column_a": "x", "column_b": "y", "pearson": 1.2}],
        )


def test_chat_plan_guard_feedback_retries_and_self_heals(tmp_path: Path) -> None:
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
    llm = ScriptedStructuredLLM(
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
                columns=["missing_amount"],
                filters=[],
                sql="select missing_amount from orders",
                method="grouped_aggregate",
                rationale="Aggregate a hallucinated column.",
                needs_approval=False,
                estimated_scan="small",
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
        store=store,
        preview_rows=10,
    )

    assert result.status == "answer"
    assert result.sql is not None and "sum(amount)" in result.sql
    retry_payload = llm.calls[2]["payload"]
    assert "previous_error" in retry_payload
    feedback = retry_payload["previous_error"]
    assert "Tool guard rejected" in feedback
    assert "missing_amount" in feedback
    assert "Allowed" in feedback
    assert "How to fix" in feedback

    events = store.list_trace_events(project_id="project_demo", session_id="run_demo")
    guard_events = [event for event in events if event.event_type == "tool_guard_rejected"]
    assert len(guard_events) == 1
    assert guard_events[0].name == "m3_build_plan"
    assert guard_events[0].summary["tool_name"] == "m3_build_plan"
    assert "missing_amount" in guard_events[0].summary["feedback"]
