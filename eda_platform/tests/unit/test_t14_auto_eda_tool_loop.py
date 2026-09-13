"""T14: the auto_eda main pipeline reaches the typed tool loop, not only SQL.

The 2026-08-25 Olist run executed all eleven questions with
``execution_mode="pipeline"`` and ``tool_calls=0``, so the forecast question was
answered with SQL and failed. These tests pin the route, its output contract,
its fallback, and the context the agent is given.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from eda_platform.core.ids import make_artifact_id
from eda_platform.core.kernel import SessionContext, run_pipeline
from eda_platform.core.llm import (
    LLMToolCall,
    LLMToolResponse,
    OfflineLLMClient,
    ToolCallingUnsupportedError,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import ExecuteTopQuestionsStep
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.schemas.plans import AnalysisPlan
from eda_platform.schemas.questions import (
    QuestionCandidate,
    QuestionCandidateSet,
    QuestionScore,
)
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.profiler import profile_dataset

PROJECT = "project_t14"
SESSION = "run_t14"


class _ScriptedToolLLM:
    """Tool-calling double: no ``settings``, so readiness answers usable/client."""

    def __init__(self, responses: list[LLMToolResponse]) -> None:
        self.responses = list(responses)
        self.seen_context: list[dict[str, Any]] = []

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolResponse:
        assert task == "question_agent_tool_loop"
        self.seen_context.append({"messages": messages, "tools": tools})
        return self.responses.pop(0)

    def structured(self, *, task: str, schema: type[Any], payload: dict[str, Any]) -> Any:
        if task != "m3_build_plan":
            # Interpretation and the like: an empty draft keeps the SQL path
            # deterministic without scripting every downstream task.
            return schema.model_construct()
        return AnalysisPlan(
            question="amount over time",
            dataset_names=["series"],
            columns=["day", "amount"],
            filters=[],
            sql="select day, sum(amount) as total_amount from series group by day",
            method="grouped_aggregate",
            rationale="Aggregate the amount per day.",
            needs_approval=False,
            estimated_scan="small",
        )

    def text(self, *, task: str, payload: dict[str, Any]) -> str:
        return ""

    def last_usage(self) -> None:
        return None


class _RefusingToolLLM(_ScriptedToolLLM):
    """A provider that diagnoses the tools payload as unsupported, then plans SQL."""

    def tool_call(self, **_kwargs: Any) -> LLMToolResponse:
        raise ToolCallingUnsupportedError("this deployment rejects a tools payload")


def _series_dataset() -> LoadedDataset:
    frame = pd.DataFrame(
        {
            "day": pd.date_range("2024-01-01", periods=24, freq="D"),
            "amount": [100.0 + 3.0 * index for index in range(24)],
        }
    )
    return LoadedDataset(
        record=DatasetRecord(
            dataset_id="ds_series",
            name="series.csv",
            path=Path("/data/series.csv"),
            content_hash="hash_series",
        ),
        frame=frame,
    )


def _forecast_candidate() -> QuestionCandidate:
    return QuestionCandidate(
        question_id="q_forecast",
        question_en="If recent patterns continue, what does next week's amount look like?",
        origin="llm",
        analysis_mode="forecast",
        target_datasets=["series.csv"],
        exploratory=True,
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.6,
            quality_risk=0.1,
            join_risk=0.0,
            deterministic_score=0.9,
        ),
    )


def _forecast_script() -> list[LLMToolResponse]:
    return [
        LLMToolResponse(
            tool_calls=[
                LLMToolCall(call_id="c1", name="inspect_data_catalog", arguments={})
            ]
        ),
        LLMToolResponse(
            tool_calls=[
                LLMToolCall(
                    call_id="c2",
                    name="run_forecast",
                    arguments={
                        "dataset_id": "ds_series",
                        "time_column": "day",
                        "value_column": "amount",
                    },
                )
            ]
        ),
        LLMToolResponse(
            content=(
                "If recent patterns continue, the amount keeps drifting upward; "
                "see the forecast table for the projected band."
            )
        ),
    ]


def _session(tmp_path: Path, candidates: list[QuestionCandidate]) -> tuple[
    ArtifactStore, SessionContext, str, list[Artifact]
]:
    store = ArtifactStore(tmp_path / "workspace")
    ctx = SessionContext(project_id=PROJECT, session_id=SESSION, store=store)
    dataset = _series_dataset()
    # Stand in for the EDA front half: this run's profile artifact.
    profile = profile_dataset(dataset, project_id=PROJECT, session_id=SESSION)
    store.save_artifact(profile)
    payload = QuestionCandidateSet(candidates=candidates).model_dump(mode="json")
    qcand = Artifact(
        id=make_artifact_id("qcand", payload),
        type=ArtifactType.QUESTION_CANDIDATE_SET,
        project_id=PROJECT,
        session_id=SESSION,
        payload=payload,
    )
    store.save_artifact(qcand)
    return store, ctx, qcand.id, [profile]


def _step(qcand_id: str, llm: Any) -> ExecuteTopQuestionsStep:
    return ExecuteTopQuestionsStep(
        [_series_dataset()],
        question_candidate_artifact_id=qcand_id,
        relationship_artifact_ids=[],
        llm=llm,
    )


def _qexec(artifacts: list[Artifact]) -> dict[str, Any]:
    return next(
        artifact.payload
        for artifact in artifacts
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT
    )


def _events(store: ArtifactStore, event_type: str) -> list[Any]:
    return [
        event
        for event in store.list_trace_events(project_id=PROJECT, session_id=SESSION)
        if event.event_type == event_type
    ]


def test_forecast_question_runs_the_tool_loop_inside_auto_eda(tmp_path: Path) -> None:
    store, ctx, qcand_id, _ = _session(tmp_path, [_forecast_candidate()])
    llm = _ScriptedToolLLM(_forecast_script())

    result = run_pipeline([_step(qcand_id, llm)], ctx)

    payload = _qexec(result.artifacts)
    assert payload["execution_mode"] == "agent"
    assert "run_forecast" in payload["tool_names"]
    assert payload["tool_calls"] == 2
    # The method contract is satisfied, so the answer publishes instead of
    # abstaining with method_contract_failed.
    assert payload["status"] == "succeeded"
    assert payload["outcome"] == "answered"
    assert payload["contract_status"] == "passed"


def test_typed_method_artifacts_survive_the_step_output_contract(tmp_path: Path) -> None:
    store, ctx, qcand_id, _ = _session(tmp_path, [_forecast_candidate()])
    llm = _ScriptedToolLLM(_forecast_script())

    result = run_pipeline([_step(qcand_id, llm)], ctx)

    produced = {artifact.type for artifact in result.artifacts}
    # Neither type is declarable as SQL_RESULT/QUESTION_EXECUTION_RESULT; before
    # `produces` was widened the pipeline raised StepContractError here.
    assert ArtifactType.TABLE in produced
    assert ArtifactType.EVIDENCE_RECEIPT in produced
    assert produced <= set(ExecuteTopQuestionsStep.produces)
    assert not _events(store, "step_contract_violation")


def test_the_agent_sees_this_run_s_earlier_eda_artifacts(tmp_path: Path) -> None:
    store, ctx, qcand_id, eda_artifacts = _session(tmp_path, [_forecast_candidate()])
    llm = _ScriptedToolLLM(_forecast_script())

    run_pipeline([_step(qcand_id, llm)], ctx)

    # inspect_data_catalog reports the profile ids it was handed; an agent run
    # with an empty context artifact list would report none.
    catalog_message = next(
        message
        for message in llm.seen_context[1]["messages"]
        if message.get("role") == "tool"
    )
    assert eda_artifacts[0].id in str(catalog_message)


def test_the_profile_artifact_is_not_republished_as_step_output(tmp_path: Path) -> None:
    store, ctx, qcand_id, eda_artifacts = _session(tmp_path, [_forecast_candidate()])
    llm = _ScriptedToolLLM(_forecast_script())

    result = run_pipeline([_step(qcand_id, llm)], ctx)

    # inspect_data_catalog returns the profile as evidence. It shares this
    # session id, so only the id filter keeps it out of the step's output.
    assert eda_artifacts[0].id not in {artifact.id for artifact in result.artifacts}


@pytest.mark.parametrize(
    ("llm", "expected_reason"),
    [
        (OfflineLLMClient(), "deterministic"),
        (_RefusingToolLLM([]), "tools payload"),
    ],
)
def test_an_unusable_tool_route_falls_back_to_sql_and_says_so(
    tmp_path: Path, llm: Any, expected_reason: str
) -> None:
    store, ctx, qcand_id, _ = _session(tmp_path, [_forecast_candidate()])

    result = run_pipeline([_step(qcand_id, llm)], ctx)

    payload = _qexec(result.artifacts)
    assert payload["execution_mode"] != "agent"
    degraded = _events(store, "agent_route_degraded")
    assert len(degraded) == 1
    assert expected_reason in str(degraded[0].summary["reason"])


def test_a_plain_aggregate_question_stays_on_the_sql_path(tmp_path: Path) -> None:
    aggregate = _forecast_candidate().model_copy(
        update={
            "question_id": "q_total",
            "question_en": "What is the total amount?",
            "analysis_mode": "descriptive",
            "sql_template": None,
        }
    )
    store, ctx, qcand_id, _ = _session(tmp_path, [aggregate])
    llm = _ScriptedToolLLM(_forecast_script())

    run_pipeline([_step(qcand_id, llm)], ctx)

    # No tool loop was entered at all: a descriptive question carries no method
    # contract, so it must not pay for the agent.
    assert llm.seen_context == []
