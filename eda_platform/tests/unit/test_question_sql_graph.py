"""Restart contracts for the structured-only question path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from eda_platform.core.graph_execution import GraphEffectUncertain, GraphPersistence
from eda_platform.drivers import question_exec
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.questions import QuestionCandidate, QuestionScore
from eda_platform.tools.loader import load_csv


class WorkerExit(BaseException):
    pass


class Provider:
    def __init__(self, *, uncertain: bool = False) -> None:
        self.calls: list[str] = []
        self.uncertain = uncertain

    def structured[T: BaseModel](self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append(task)
        if task == "m3_build_plan":
            # Exercise the planner's internal repair: every physical request
            # must have its own cached task and effect record.
            column = "revenue" if payload.get("previous_error") else "missing_column"
            return schema.model_validate({
                "question": "What is total revenue?", "dataset_names": ["orders"],
                "columns": [column], "filters": [],
                "sql": f"select sum({column}) as total from orders",
                "method": "sum", "rationale": "Compute total revenue.",
                "needs_approval": False, "estimated_scan": "small",
            })
        if self.uncertain:
            raise TimeoutError("provider may have accepted the interpretation")
        return schema.model_validate({"interpretation": "The total revenue is 30."})

    def text(self, *, task: str, payload: dict) -> str:
        raise AssertionError("unused")

    def last_usage(self) -> None:
        return None


def _run(tmp_path: Path, provider: Provider) -> list[Any]:
    source = tmp_path / "orders.csv"
    if not source.exists():
        source.write_text("revenue\n10\n20\n")
    # Reopening the dataset changes non-semantic created_at metadata; it must
    # not invalidate the durable execution's data identity.
    dataset = load_csv(source, dataset_id="ds-orders")
    candidate = QuestionCandidate(
        question_id="q-total", question_en="What is total revenue?", origin="llm",
        target_datasets=["orders.csv"], exploratory=True,
        score=QuestionScore(data_availability=1, statistical_signal=0.5,
                            quality_risk=0, join_risk=0, deterministic_score=0.6),
    )
    return question_exec.execute_question_candidate(
        candidate, datasets=[dataset], project_id="project", session_id="session",
        parent_ids=[], llm=provider,
        persistence=GraphPersistence(tmp_path, "question-sql:q-total"),
    )


@pytest.mark.parametrize("crash_after", ["plan", "query"])
def test_restart_adopts_each_planner_request_and_completed_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_after: str,
) -> None:
    original_sql = question_exec.run_sql
    original_success = question_exec._successful_qexec_artifact
    queries: list[str] = []

    def sql(*args: Any, **kwargs: Any) -> Any:
        queries.append(args[1])
        return original_sql(*args, **kwargs)

    def crash(*args: Any, **kwargs: Any) -> Any:
        raise WorkerExit

    monkeypatch.setattr(question_exec, "run_sql", crash if crash_after == "plan" else sql)
    if crash_after == "query":
        monkeypatch.setattr(question_exec, "_successful_qexec_artifact", crash)
    first = Provider()
    with pytest.raises(WorkerExit):
        _run(tmp_path, first)
    assert first.calls == ["m3_build_plan", "m3_build_plan"]

    monkeypatch.setattr(question_exec, "run_sql", sql)
    monkeypatch.setattr(question_exec, "_successful_qexec_artifact", original_success)
    resumed = Provider()
    artifacts = _run(tmp_path, resumed)
    assert resumed.calls == ["di4_l1_interpretation"]
    assert len(queries) == 1
    result = next(a.payload for a in artifacts if a.type == ArtifactType.QUESTION_EXECUTION_RESULT)
    assert result["status"] == "succeeded"
    assert result["interpretation_status"] == "validated"


def test_unknown_interpretation_outcome_bypasses_domain_retry_and_fallback(tmp_path: Path) -> None:
    first = Provider(uncertain=True)
    with pytest.raises(GraphEffectUncertain):
        _run(tmp_path, first)
    assert first.calls == ["m3_build_plan", "m3_build_plan", "di4_l1_interpretation"]
    resumed = Provider()
    with pytest.raises(GraphEffectUncertain):
        _run(tmp_path, resumed)
    assert resumed.calls == []
