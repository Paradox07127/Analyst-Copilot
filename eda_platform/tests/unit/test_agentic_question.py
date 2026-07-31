from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from eda_platform.agents.question_runtime import run_question_agent
from eda_platform.agents.runtime import AgentRunResult
from eda_platform.core.ids import make_artifact_id
from eda_platform.core.llm import (
    LLMToolCall,
    LLMToolResponse,
    ToolCallingUnsupportedError,
)
from eda_platform.core.sandbox import ExecArtifact, SandboxBackendInfo
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.question_exec import _agent_qexec_artifact
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.questions import (
    QuestionCandidate,
    QuestionExecutionResult,
    QuestionScore,
)
from eda_platform.tools.loader import load_csv


class _ScriptedQuestionLLM:
    def __init__(
        self,
        responses: list[LLMToolResponse],
        *,
        code: str = "",
    ) -> None:
        self.responses = list(responses)
        self.code = code
        self.exposed_tools: list[set[str]] = []

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolResponse:
        assert task == "question_agent_tool_loop"
        self.exposed_tools.append({str(tool["name"]) for tool in tools})
        return self.responses.pop(0)

    def structured(
        self,
        *,
        task: str,
        schema: type[Any],
        payload: dict[str, Any],
    ) -> Any:
        assert task == "m5_code_agent_generate"
        return schema.model_validate({"code": self.code, "notes": "test"})

    def text(self, *, task: str, payload: dict[str, Any]) -> str:
        raise AssertionError("The autonomous question path uses native tool calls.")

    def last_usage(self) -> None:
        return None


class _SafeFakeBackend:
    name = "safe_fake"

    @property
    def info(self) -> SandboxBackendInfo:
        return SandboxBackendInfo(
            name=self.name,
            safe_for_untrusted_code=True,
            available=True,
            detail="test backend",
        )

    def run_python(self, code: str, **_kwargs: Any) -> ExecArtifact:
        assert "print" in code
        return ExecArtifact(
            status="succeeded",
            backend=self.name,
            stdout='{"summary":"A non-linear comparison completed.","result_files":[]}\n',
            exit_code=0,
        )


def _context(tmp_path: Path) -> tuple[ArtifactStore, Any]:
    source = tmp_path / "orders.csv"
    source.write_text("region,amount\nEast,10\nWest,20\nEast,5\n", encoding="utf-8")
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "question_run")
    return store, load_csv(source, dataset_id="orders_1")


def test_question_agent_can_choose_multiple_tools_before_answering(tmp_path: Path) -> None:
    store, dataset = _context(tmp_path)
    llm = _ScriptedQuestionLLM(
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
            LLMToolResponse(
                content="West has the highest total amount. Sources: the SQL artifact."
            ),
        ]
    )

    result = run_question_agent(
        "Which region has the highest total amount?",
        candidate_context={"target_datasets": ["orders.csv"]},
        datasets=[dataset],
        project_id="project_demo",
        session_id="question_run",
        llm=llm,  # type: ignore[arg-type]
        artifacts=[],
        store=store,
    )

    assert result.status == "completed"
    assert result.tool_names == ["inspect_data_catalog", "run_sql"]
    sql = next(
        artifact for artifact in result.artifacts if artifact.type is ArtifactType.SQL_RESULT
    )
    assert sql.payload["rows_preview"][0]["region"] == "West"
    assert {"run_sql", "run_saved_skill", "list_artifacts"} <= llm.exposed_tools[0]
    assert "run_open_analysis" not in llm.exposed_tools[0]


def test_question_agent_can_choose_secured_python_when_available(tmp_path: Path) -> None:
    store, dataset = _context(tmp_path)
    llm = _ScriptedQuestionLLM(
        [
            LLMToolResponse(
                tool_calls=[
                    LLMToolCall(
                        call_id="code_1",
                        name="run_open_analysis",
                        arguments={
                            "task": "Compare the regional distributions with a robust test."
                        },
                    )
                ]
            ),
            LLMToolResponse(content="The robust comparison completed. Sources: the code artifact."),
        ],
        code=(
            "import json\n"
            'print(json.dumps({"summary":"A non-linear comparison completed.",'
            '"result_files":[]}))'
        ),
    )

    result = run_question_agent(
        "Do regional amount distributions differ?",
        candidate_context={"target_datasets": ["orders.csv"]},
        datasets=[dataset],
        project_id="project_demo",
        session_id="question_run",
        llm=llm,  # type: ignore[arg-type]
        artifacts=[],
        store=store,
        code_backend=_SafeFakeBackend(),  # type: ignore[arg-type]
    )

    assert result.status == "completed"
    assert result.tool_names == ["run_open_analysis"]
    assert "run_open_analysis" in llm.exposed_tools[0]
    assert any(artifact.type is ArtifactType.CODE_EXECUTION_RESULT for artifact in result.artifacts)


def test_agent_answer_is_persisted_with_tools_and_evidence() -> None:
    evidence = Artifact(
        id=make_artifact_id("code", {"test": "question-agent"}),
        type=ArtifactType.CODE_EXECUTION_RESULT,
        project_id="project_demo",
        session_id="question_run",
        payload={"status": "succeeded"},
    )
    candidate = QuestionCandidate(
        question_id="question_1",
        question_en="Do the regional distributions differ?",
        origin="llm",
        target_datasets=["orders.csv"],
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.75,
        ),
    )

    artifact = _agent_qexec_artifact(
        candidate,
        agent_result=AgentRunResult(
            status="completed",
            answer="The robust comparison found a material difference.",
            artifacts=[evidence],
            tool_calls=2,
            tool_names=["inspect_data_catalog", "run_open_analysis"],
        ),
        project_id="project_demo",
        session_id="question_run",
        parent_ids=["candidate_set_1"],
    )
    result = QuestionExecutionResult.model_validate(artifact.payload)

    assert result.execution_mode == "agent"
    assert result.tool_calls == 2
    assert result.tool_names == ["inspect_data_catalog", "run_open_analysis"]
    assert result.evidence_artifact_ids == [evidence.id]
    assert result.interpretation == "The robust comparison found a material difference."
    assert result.findings[0].evidence[0].kind == "code"


class _RefusesTools(_ScriptedQuestionLLM):
    def __init__(self) -> None:
        super().__init__([])

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolResponse:
        raise ToolCallingUnsupportedError("tools rejected by this endpoint")


def test_a_refused_tools_payload_is_not_recorded_as_one_failed_question(
    tmp_path: Path,
) -> None:
    """The batch driver has to see this so it can degrade the whole batch.
    Swallowed into a failed-question result, it would repeat the same doomed
    call for every remaining question instead."""
    store, dataset = _context(tmp_path)

    with pytest.raises(ToolCallingUnsupportedError):
        run_question_agent(
            "Do the regional distributions differ?",
            candidate_context={},
            datasets=[dataset],
            project_id="project_demo",
            session_id="question_run",
            llm=_RefusesTools(),
            artifacts=[],
            store=store,
        )
