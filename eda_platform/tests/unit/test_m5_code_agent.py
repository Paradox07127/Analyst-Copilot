from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from eda_platform.agents.code_agent import CodeAgent
from eda_platform.core.budget import Budget
from eda_platform.core.cancellation import CancellationContext, cancellation_scope
from eda_platform.core.sandbox import PortableSubprocessBackend, SandboxLimits

T = TypeVar("T", bound=BaseModel)


class FakeCodeLLM:
    def __init__(self, drafts: list[dict[str, Any]]) -> None:
        self.drafts = drafts
        self.calls: list[dict[str, Any]] = []

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append({"task": task, "payload": payload})
        index = min(len(self.calls) - 1, len(self.drafts) - 1)
        return schema.model_validate(self.drafts[index])

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> None:
        return None


def test_code_agent_repairs_failed_sandbox_execution(tmp_path: Path) -> None:
    llm = FakeCodeLLM(
        [
            {"code": "raise RuntimeError('bad draft')"},
            {"code": "print('ok')"},
        ]
    )
    events: list[dict[str, object]] = []
    agent = CodeAgent(
        llm=cast(Any, llm),
        backend=PortableSubprocessBackend(work_root=tmp_path),
        limits=SandboxLimits(timeout_seconds=2),
        on_event=events.append,
    )

    result = agent.run(
        task="Compute a tiny result.",
        evidence_manifest={"datasets": []},
    )

    assert result.status == "succeeded"
    assert result.final_artifact is not None
    assert result.final_artifact.stdout.strip() == "ok"
    assert len(result.attempts) == 2
    assert "previous_error" not in llm.calls[0]["payload"]
    assert "bad draft" in llm.calls[1]["payload"]["previous_error"]
    assert [event["status"] for event in events] == ["failed", "succeeded"]


def test_code_agent_attempt_trace_supports_repair_success_rate(tmp_path: Path) -> None:
    llm = FakeCodeLLM(
        [
            {"code": "raise RuntimeError('bad draft')"},
            {"code": ("import json\nprint(json.dumps({'summary': 'ok', 'result_files': []}))")},
        ]
    )
    events: list[dict[str, object]] = []
    agent = CodeAgent(
        llm=cast(Any, llm),
        backend=PortableSubprocessBackend(work_root=tmp_path),
        limits=SandboxLimits(timeout_seconds=2),
        on_event=events.append,
        require_stdout_json=True,
    )

    result = agent.run(
        task="Compute a tiny result.",
        evidence_manifest={"datasets": []},
    )

    assert result.status == "succeeded"
    attempts = [event for event in events if event["event"] == "code_agent_attempt"]
    assert [event["attempt"] for event in attempts] == [1, 2]
    success_rate = sum(event["status"] == "succeeded" for event in attempts) / len(attempts)
    assert success_rate == 0.5
    assert attempts[0]["error_category"] == "sandbox_failed"
    assert attempts[1]["error_category"] is None


def test_code_agent_stops_after_three_attempts_without_crashing(tmp_path: Path) -> None:
    llm = FakeCodeLLM(
        [
            {"code": "raise RuntimeError('one')"},
            {"code": "raise RuntimeError('two')"},
            {"code": "raise RuntimeError('three')"},
            {"code": "print('fourth must not run')"},
        ]
    )
    agent = CodeAgent(
        llm=cast(Any, llm),
        backend=PortableSubprocessBackend(work_root=tmp_path),
        limits=SandboxLimits(timeout_seconds=2),
        max_repairs=10,
    )

    result = agent.run(task="Keep failing.", evidence_manifest={"datasets": []})

    assert result.status == "failed"
    assert len(result.attempts) == 3
    assert len(llm.calls) == 3


def test_code_agent_budget_exhaustion_aborts_cleanly(tmp_path: Path) -> None:
    llm = FakeCodeLLM(
        [{"code": ("import json\nprint(json.dumps({'summary': 'ok', 'result_files': []}))")}]
    )
    agent = CodeAgent(
        llm=cast(Any, llm),
        backend=PortableSubprocessBackend(work_root=tmp_path),
        limits=SandboxLimits(timeout_seconds=2),
        require_stdout_json=True,
    )

    result = agent.run(
        task="Budget should be exhausted before execution.",
        evidence_manifest={"datasets": []},
        budget=Budget(max_seconds=0),
    )

    assert result.status == "failed"
    assert result.error_category == "budget_exhausted"
    assert result.attempts == []


def test_code_agent_cancellation_after_provider_call_stops_before_sandbox(
    tmp_path: Path,
) -> None:
    cancellation = CancellationContext()

    class CancellingLLM(FakeCodeLLM):
        def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
            result = super().structured(task=task, schema=schema, payload=payload)
            cancellation.request_cancel("stop after provider")
            return result

    llm = CancellingLLM([{"code": "print('must not execute')"}])
    backend = PortableSubprocessBackend(work_root=tmp_path)
    agent = CodeAgent(llm=cast(Any, llm), backend=backend)

    with cancellation_scope(cancellation):
        result = agent.run(
            task="Cancel before code.",
            evidence_manifest={"datasets": []},
        )

    assert result.status == "failed"
    assert result.error_category == "cancelled"
    assert result.attempts == []
    assert list(tmp_path.glob("exec_*")) == []
