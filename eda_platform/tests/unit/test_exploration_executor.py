from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

import pytest
from pydantic import BaseModel, ConfigDict, Field

from eda_platform.agents.exploration.executor import (
    InMemoryLlmResponseStore,
    InMemoryToolResultStore,
    JsonlProbeJournalHooks,
    PhaseToolMap,
    ProbeExecutor,
    ProviderCallRejectedError,
    canonical_probe_fingerprint,
)
from eda_platform.agents.runtime import AgentTool, AgentToolResult
from eda_platform.agents.tool_context import current_execution_context
from eda_platform.core.budget import BudgetExceeded
from eda_platform.core.cancellation import (
    CancellationCause,
    CancellationError,
    CancellationSnapshot,
    KillFenceState,
)
from eda_platform.core.event_journal import EventTransitionError
from eda_platform.core.exploration_budget import (
    ToolBudgetExceeded,
    ToolCallLedger,
    ToolCallProjection,
)
from eda_platform.core.exploration_journal import (
    JsonlExplorationJournal,
    sealed_policy,
)
from eda_platform.core.ids import stable_hash
from eda_platform.core.llm import (
    LLMToolCall,
    LLMToolResponse,
    MalformedProviderResponseError,
)
from eda_platform.schemas.exploration import ExplorationPolicy
from eda_platform.schemas.exploration_budget import (
    ExplorationBudgetPolicy,
    SessionBudgetPolicyModel,
)


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _SqlArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sql: str = Field(min_length=1)
    purpose: str = Field(min_length=1)


@dataclass(frozen=True)
class _Receipt:
    id: str


class _Provider:
    def __init__(self, responses: list[LLMToolResponse | BaseException]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolResponse:
        self.calls.append(
            {
                "task": task,
                "messages": [dict(message) for message in messages],
                "tools": tools,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _Hooks:
    attempt_epoch = 7

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def llm_started(self, **fields: Any) -> None:
        self.events.append(("llm_started", fields))

    def llm_terminal(self, **fields: Any) -> None:
        self.events.append(("llm_terminal", fields))

    def tool_started(self, **fields: Any) -> None:
        self.events.append(("tool_started", fields))

    def tool_terminal(self, **fields: Any) -> None:
        self.events.append(("tool_terminal", fields))


class _CrashOnSecondToolStart(_Hooks):
    def __init__(self) -> None:
        super().__init__()
        self.started_ids: list[str] = []

    def tool_started(self, **fields: Any) -> None:
        super().tool_started(**fields)
        self.started_ids.append(fields["logical_step_id"])
        if len(self.started_ids) == 2:
            raise RuntimeError("simulated crash before the second tool executes")


@dataclass(frozen=True)
class _Meter:
    projected_rows: int = 0
    successful_rows: int | None = None
    failed_rows: int | None = None

    def project(self, *, call: Any, tool: AgentTool, arguments: Any) -> ToolCallProjection:
        del call, arguments
        return ToolCallProjection(kind=tool.name, rows_scanned=self.projected_rows)

    def success(
        self,
        *,
        call: Any,
        tool: AgentTool,
        arguments: Any,
        result: AgentToolResult,
        projected: ToolCallProjection,
    ) -> ToolCallProjection:
        del call, tool, arguments, result
        return ToolCallProjection(
            kind=projected.kind,
            rows_scanned=(
                projected.rows_scanned
                if self.successful_rows is None
                else self.successful_rows
            ),
        )

    def failure(
        self,
        *,
        call: Any,
        tool: AgentTool,
        arguments: Any,
        error: BaseException,
        projected: ToolCallProjection,
    ) -> ToolCallProjection:
        del call, tool, arguments, error
        return ToolCallProjection(
            kind=projected.kind,
            rows_scanned=(projected.rows_scanned if self.failed_rows is None else self.failed_rows),
        )


@dataclass(frozen=True)
class _RaisingMeter(_Meter):
    """A ``_Meter`` whose project()/success() calls can raise on demand."""

    raise_on_project: BaseException | None = None
    raise_on_success: BaseException | None = None

    def project(self, *, call: Any, tool: AgentTool, arguments: Any) -> ToolCallProjection:
        if self.raise_on_project is not None:
            raise self.raise_on_project
        return super().project(call=call, tool=tool, arguments=arguments)

    def success(
        self,
        *,
        call: Any,
        tool: AgentTool,
        arguments: Any,
        result: AgentToolResult,
        projected: ToolCallProjection,
    ) -> ToolCallProjection:
        if self.raise_on_success is not None:
            raise self.raise_on_success
        return super().success(
            call=call, tool=tool, arguments=arguments, result=result, projected=projected
        )


def _policy(
    *kinds: str,
    max_calls: int = 12,
    max_rows: int | None = None,
) -> ExplorationBudgetPolicy:
    return ExplorationBudgetPolicy(
        llm=SessionBudgetPolicyModel(max_requests=20),
        max_successful_tool_calls=max_calls,
        max_tool_calls_by_kind={kind: max_calls for kind in kinds},
        max_rows_scanned=max_rows,
        max_result_cells=None,
        idle_timeout_seconds=60,
        max_rounds=4,
    )


def _tool(
    name: str,
    execute: Any,
    *,
    args_schema: type[BaseModel] = _NoArgs,
) -> AgentTool:
    return AgentTool(
        name=name,
        description=f"Execute {name}.",
        args_schema=args_schema,
        execute=execute,
    )


def _call(name: str, call_id: str = "call-1", **arguments: Any) -> LLMToolCall:
    return LLMToolCall(call_id=call_id, name=name, arguments=arguments)


def _cancel_error() -> CancellationError:
    return CancellationError(
        CancellationSnapshot(
            cause=CancellationCause.CANCEL_REQUESTED,
            reason="cancelled by user",
            deadline=None,
            shield_depth=0,
            kill_fence_state=KillFenceState.ELIGIBLE,
        )
    )


def test_native_loop_settles_success_and_pairs_started_terminal_hooks() -> None:
    contexts = []

    def execute(_args: BaseModel) -> AgentToolResult:
        contexts.append(current_execution_context())
        return AgentToolResult(
            content={"value": 3},
            receipt_artifact=_Receipt("rcpt-1"),
        )

    provider = _Provider(
        [
            LLMToolResponse(tool_calls=[_call("inspect")]),
            LLMToolResponse(content="Probe complete.", finish_reason="stop"),
        ]
    )
    hooks = _Hooks()
    ledger = ToolCallLedger(_policy("inspect"))
    executor = ProbeExecutor(
        provider,
        [_tool("inspect", execute)],
        ledger,
        journal=hooks,
        usage_meter=_Meter(projected_rows=4, successful_rows=3),
    )

    result = executor.run(
        phase="execute_probes",
        system_prompt="Use tools.",
        user_message="Inspect.",
        run_id="run-1",
    )

    assert result.status == "completed"
    assert result.answer == "Probe complete."
    assert result.tool_calls == 1
    assert ledger.snapshot() == {
        "successful_tool_calls": 1,
        "calls_by_kind": {"inspect": 1},
        "rows_scanned": 3,
        "result_cells": 0,
    }
    assert contexts[0] is not None
    assert contexts[0].attempt_epoch == 7
    assert [event for event, _fields in hooks.events] == [
        "llm_started",
        "llm_terminal",
        "tool_started",
        "tool_terminal",
        "llm_started",
        "llm_terminal",
    ]


def test_completed_llm_response_is_adopted_without_resending() -> None:
    run_id = "durable-run"
    step_id = "llmstep_" + stable_hash({"run_id": run_id, "step": 1}, length=24)
    store = InMemoryLlmResponseStore()
    store.remember(step_id, LLMToolResponse(content="Recovered answer."))
    provider = _Provider([])
    hooks = _Hooks()
    result = ProbeExecutor(
        provider,
        [_tool("inspect", lambda _args: AgentToolResult(content={}))],
        ToolCallLedger(_policy("inspect")),
        journal=hooks,
        response_store=store,
    ).run(
        phase="execute_probes",
        system_prompt="x",
        user_message="y",
        run_id=run_id,
        completed_step_ids={step_id},
    )

    assert result.status == "completed"
    assert result.answer == "Recovered answer."
    assert provider.calls == []
    assert hooks.events == []


def test_completed_response_without_its_body_fails_closed_instead_of_resending() -> None:
    run_id = "durable-run"
    step_id = "llmstep_" + stable_hash({"run_id": run_id, "step": 1}, length=24)
    provider = _Provider([])
    result = ProbeExecutor(
        provider,
        [_tool("inspect", lambda _args: AgentToolResult(content={}))],
        ToolCallLedger(_policy("inspect")),
    ).run(
        phase="execute_probes",
        system_prompt="x",
        user_message="y",
        run_id=run_id,
        completed_step_ids={step_id},
    )

    assert result.status == "failed"
    assert result.error_code == "completed_response_unavailable"
    assert provider.calls == []


def test_uncertain_llm_call_id_is_never_resent() -> None:
    run_id = "uncertain-run"
    call_id = "llm_" + stable_hash({"run_id": run_id, "step": 1}, length=24)
    provider = _Provider([])
    result = ProbeExecutor(
        provider,
        [_tool("inspect", lambda _args: AgentToolResult(content={}))],
        ToolCallLedger(_policy("inspect")),
    ).run(
        phase="execute_probes",
        system_prompt="x",
        user_message="y",
        run_id=run_id,
        blocked_llm_call_ids={call_id},
    )

    assert result.status == "failed"
    assert result.error_code == "provider_outcome_uncertain"
    assert provider.calls == []


def test_phase_subset_is_recomputed_and_unknown_tool_returns_complete_current_inventory() -> None:
    provider = _Provider(
        [
            LLMToolResponse(tool_calls=[_call("hidden")]),
            LLMToolResponse(content="I used the current inventory."),
        ]
    )
    selected_steps: list[int] = []

    def selector(*, phase: str, step: int, registered_tools: tuple[str, ...]) -> list[str]:
        assert phase == "execute_probes"
        assert registered_tools == ("inspect", "hidden")
        selected_steps.append(step)
        return ["inspect"]

    ledger = ToolCallLedger(_policy("inspect", "hidden"))
    executor = ProbeExecutor(
        provider,
        [
            _tool("inspect", lambda _args: AgentToolResult(content={})),
            _tool("hidden", lambda _args: pytest.fail("hidden tool must be unreachable")),
        ],
        ledger,
        phase_tools=selector,
    )

    result = executor.run(
        phase="execute_probes",
        system_prompt="Use the phase inventory.",
        user_message="Try.",
    )

    assert result.status == "completed"
    assert result.tool_calls == 0
    assert selected_steps == [1, 2]
    assert [[tool["name"] for tool in call["tools"]] for call in provider.calls] == [
        ["inspect"],
        ["inspect"],
    ]
    unknown = json.loads(provider.calls[1]["messages"][-2]["content"])
    assert unknown["available_tools"] == ["inspect"]
    assert "hidden" not in unknown["available_tools"]
    assert ledger.snapshot()["successful_tool_calls"] == 0


def test_phase_map_fails_closed_for_an_unconfigured_phase() -> None:
    executor = ProbeExecutor(
        _Provider([]),
        [_tool("inspect", lambda _args: AgentToolResult(content={}))],
        ToolCallLedger(_policy("inspect")),
        phase_tools=PhaseToolMap({"validate": ["inspect"]}),
    )

    with pytest.raises(ValueError, match="No tool subset"):
        executor.run(
            phase="execute_probes",
            system_prompt="x",
            user_message="y",
        )


def test_large_observation_has_one_bounded_truncation_envelope() -> None:
    provider = _Provider(
        [
            LLMToolResponse(tool_calls=[_call("inspect")]),
            LLMToolResponse(content="Done."),
        ]
    )
    executor = ProbeExecutor(
        provider,
        [_tool("inspect", lambda _args: AgentToolResult(content={"blob": "x" * 2_000}))],
        ToolCallLedger(_policy("inspect")),
        max_observation_chars=300,
    )

    assert executor.run(phase="execute_probes", system_prompt="x", user_message="y").status == (
        "completed"
    )

    encoded = provider.calls[1]["messages"][-1]["content"]
    observation = json.loads(encoded)
    assert len(encoded) <= 300
    assert set(observation) == {"ok", "truncated", "original_chars", "preview"}
    assert observation["truncated"] is True
    assert isinstance(observation["preview"], str)


@pytest.mark.parametrize(
    ("finish_reason", "error_code"),
    [("length", "finish_reason_length"), ("content_filter", "content_filtered")],
)
def test_finish_reason_failures_are_terminal_without_retry(
    finish_reason: str,
    error_code: str,
) -> None:
    provider = _Provider([LLMToolResponse(finish_reason=finish_reason)])
    result = ProbeExecutor(
        provider,
        [_tool("inspect", lambda _args: AgentToolResult(content={}))],
        ToolCallLedger(_policy("inspect")),
    ).run(phase="execute_probes", system_prompt="x", user_message="y")

    assert result.status == "failed"
    assert result.error_code == error_code
    assert len(provider.calls) == 1


def test_empty_response_gets_a_legal_exit_retry_within_a_bounded_allowance() -> None:
    provider = _Provider([LLMToolResponse() for _ in range(6)])
    result = ProbeExecutor(
        provider,
        [_tool("inspect", lambda _args: AgentToolResult(content={}))],
        ToolCallLedger(_policy("inspect")),
    ).run(phase="execute_probes", system_prompt="x", user_message="y")

    assert result.status == "failed"
    assert result.error_code == "empty_response"
    assert len(provider.calls) == 4
    assert "only valid exits" in provider.calls[1]["messages"][-1]["content"]


def test_canonical_sql_duplicate_is_rejected_before_execution_without_a_slot() -> None:
    executions = 0

    def execute(_args: BaseModel) -> AgentToolResult:
        nonlocal executions
        executions += 1
        return AgentToolResult(content={"rows": 1})

    provider = _Provider(
        [
            LLMToolResponse(
                tool_calls=[
                    _call("run_sql", "one", sql=" SELECT  * FROM T ; ", purpose="first"),
                    _call("run_sql", "two", sql="select *   from t", purpose="different prose"),
                ]
            ),
            LLMToolResponse(content="Concluded."),
        ]
    )
    ledger = ToolCallLedger(_policy("run_sql", max_calls=1))
    result = ProbeExecutor(
        provider,
        [_tool("run_sql", execute, args_schema=_SqlArgs)],
        ledger,
        max_tool_calls=1,
    ).run(phase="execute_probes", system_prompt="x", user_message="y")

    assert result.status == "completed"
    assert executions == 1
    assert result.tool_calls == 1
    assert ledger.snapshot()["successful_tool_calls"] == 1
    assert len(result.probe_fingerprints) == 1
    assert any("canonical fingerprint" in entry for entry in result.failure_history)
    assert canonical_probe_fingerprint(
        "run_sql", {"sql": " SELECT  * FROM T ; ", "purpose": "a"}
    ) == canonical_probe_fingerprint(
        "run_sql", {"sql": "select *   from t", "purpose": "b"}
    )


def test_whole_batch_projection_rejects_before_any_tool_started() -> None:
    executions = 0

    def execute(_args: BaseModel) -> AgentToolResult:
        nonlocal executions
        executions += 1
        return AgentToolResult(content={})

    # Distinct names produce distinct canonical fingerprints in one native batch.
    provider = _Provider(
        [LLMToolResponse(tool_calls=[_call("one", "1"), _call("two", "2")])]
    )
    hooks = _Hooks()
    ledger = ToolCallLedger(_policy("one", "two", max_rows=5))
    executor = ProbeExecutor(
        provider,
        [_tool("one", execute), _tool("two", execute)],
        ledger,
        usage_meter=_Meter(projected_rows=3),
        journal=hooks,
    )

    with pytest.raises(ToolBudgetExceeded):
        executor.run(phase="execute_probes", system_prompt="x", user_message="y")

    assert executions == 0
    assert not any(event == "tool_started" for event, _fields in hooks.events)
    assert ledger.snapshot()["rows_scanned"] == 0


def test_failed_execution_settles_resources_without_success_and_budget_stops_retry() -> None:
    provider = _Provider(
        [
            LLMToolResponse(tool_calls=[_call("inspect")]),
            LLMToolResponse(content="must not retry"),
        ]
    )

    def fail(_args: BaseModel) -> AgentToolResult:
        raise RuntimeError("query failed after scanning")

    ledger = ToolCallLedger(_policy("inspect", max_rows=5))
    hooks = _Hooks()
    executor = ProbeExecutor(
        provider,
        [_tool("inspect", fail)],
        ledger,
        usage_meter=_Meter(projected_rows=2, failed_rows=6),
        journal=hooks,
    )

    with pytest.raises(ToolBudgetExceeded):
        executor.run(phase="execute_probes", system_prompt="x", user_message="y")

    assert len(provider.calls) == 1
    assert ledger.snapshot()["successful_tool_calls"] == 0
    assert ledger.snapshot()["calls_by_kind"] == {}
    assert ledger.snapshot()["rows_scanned"] == 6
    assert [event for event, _fields in hooks.events][-2:] == [
        "tool_started",
        "tool_terminal",
    ]


def test_actual_success_over_cap_is_failed_before_receipt_commit() -> None:
    provider = _Provider([LLMToolResponse(tool_calls=[_call("inspect")])])
    hooks = _Hooks()
    ledger = ToolCallLedger(_policy("inspect", max_rows=5))
    executor = ProbeExecutor(
        provider,
        [
            _tool(
                "inspect",
                lambda _args: AgentToolResult(
                    content={"ok": True}, receipt_artifact=_Receipt("rcpt-over")
                ),
            )
        ],
        ledger,
        usage_meter=_Meter(projected_rows=2, successful_rows=6),
        journal=hooks,
    )

    with pytest.raises(ToolBudgetExceeded):
        executor.run(phase="execute_probes", system_prompt="x", user_message="y")

    terminal = [fields for event, fields in hooks.events if event == "tool_terminal"]
    assert terminal[-1]["outcome"] == "failed"
    assert terminal[-1].get("receipt_id") is None
    assert ledger.snapshot()["rows_scanned"] == 6


def test_batch_action_index_stays_stable_when_an_earlier_tool_is_adopted() -> None:
    run_id = "batch-recovery"
    response_store = InMemoryLlmResponseStore()
    completed_steps: set[str] = set()
    seen: set[str] = set()
    ledger = ToolCallLedger(_policy("one", "two"))
    first_hooks = _CrashOnSecondToolStart()
    tools = [
        _tool(
            "one",
            lambda _args: AgentToolResult(content={}, receipt_artifact=_Receipt("rcpt-1")),
        ),
        _tool(
            "two",
            lambda _args: AgentToolResult(content={}, receipt_artifact=_Receipt("rcpt-2")),
        ),
    ]
    first_provider = _Provider(
        [LLMToolResponse(tool_calls=[_call("one", "call-1"), _call("two", "call-2")])]
    )
    first = ProbeExecutor(
        first_provider,
        tools,
        ledger,
        journal=first_hooks,
        response_store=response_store,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        first.run(
            phase="execute_probes",
            system_prompt="x",
            user_message="y",
            run_id=run_id,
            seen_probe_fingerprints=seen,
            completed_step_ids=completed_steps,
        )
    assert len(seen) == 1
    assert len(completed_steps) == 2

    resumed_hooks = _Hooks()
    resumed_provider = _Provider([LLMToolResponse(content="Recovered and finished.")])
    resumed = ProbeExecutor(
        resumed_provider,
        tools,
        ledger,
        journal=resumed_hooks,
        response_store=response_store,
    ).run(
        phase="execute_probes",
        system_prompt="x",
        user_message="y",
        run_id=run_id,
        seen_probe_fingerprints=seen,
        completed_step_ids=completed_steps,
    )

    resumed_started = [
        fields["logical_step_id"]
        for event, fields in resumed_hooks.events
        if event == "tool_started"
    ]
    assert resumed.status == "completed"
    assert resumed_started[0] == first_hooks.started_ids[1]
    assert len(resumed_provider.calls) == 1  # only the next provider step was sent


def test_committed_tool_body_is_adopted_without_reexecution_after_crash() -> None:
    run_id = "tool-body-recovery"
    response_store = InMemoryLlmResponseStore()
    tool_store = InMemoryToolResultStore()
    completed_steps: set[str] = set()
    executions = {"one": 0, "two": 0}

    def execute_one(_args: BaseModel) -> AgentToolResult:
        executions["one"] += 1
        return AgentToolResult(content={"value": 1}, receipt_artifact=_Receipt("rcpt-1"))

    def execute_two(_args: BaseModel) -> AgentToolResult:
        executions["two"] += 1
        return AgentToolResult(content={"value": 2}, receipt_artifact=_Receipt("rcpt-2"))

    tools = [_tool("one", execute_one), _tool("two", execute_two)]
    ledger = ToolCallLedger(_policy("one", "two"))
    first_hooks = _CrashOnSecondToolStart()
    first = ProbeExecutor(
        _Provider(
            [LLMToolResponse(tool_calls=[_call("one", "c1"), _call("two", "c2")])]
        ),
        tools,
        ledger,
        journal=first_hooks,
        response_store=response_store,
        tool_result_store=tool_store,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        first.run(
            phase="execute_probes",
            system_prompt="x",
            user_message="y",
            run_id=run_id,
            completed_step_ids=completed_steps,
        )

    resumed = ProbeExecutor(
        _Provider([LLMToolResponse(content="done")]),
        tools,
        ledger,
        journal=_Hooks(),
        response_store=response_store,
        tool_result_store=tool_store,
    ).run(
        phase="execute_probes",
        system_prompt="x",
        user_message="y",
        run_id=run_id,
        completed_step_ids=completed_steps,
    )

    assert resumed.status == "completed"
    assert executions == {"one": 1, "two": 1}
    assert [artifact.id for artifact in resumed.artifacts] == ["rcpt-1", "rcpt-2"]


@pytest.mark.parametrize("terminal_error", [BudgetExceeded("spent"), _cancel_error()])
def test_budget_and_cancellation_errors_penetrate_even_if_failure_settlement_latches(
    terminal_error: BaseException,
) -> None:
    provider = _Provider([LLMToolResponse(tool_calls=[_call("inspect")])])

    def fail(_args: BaseModel) -> AgentToolResult:
        raise terminal_error

    ledger = ToolCallLedger(_policy("inspect", max_rows=5))
    executor = ProbeExecutor(
        provider,
        [_tool("inspect", fail)],
        ledger,
        usage_meter=_Meter(projected_rows=2, failed_rows=6),
    )

    with pytest.raises(type(terminal_error)) as captured:
        executor.run(phase="execute_probes", system_prompt="x", user_message="y")

    assert captured.value is terminal_error
    assert ledger.snapshot()["successful_tool_calls"] == 0
    assert ledger.snapshot()["rows_scanned"] == 6


def test_usage_meter_success_error_settles_as_failed_tool_and_run_continues() -> None:
    provider = _Provider(
        [
            LLMToolResponse(tool_calls=[_call("inspect")]),
            LLMToolResponse(content="Recovered after meter failure."),
        ]
    )

    def execute(_args: BaseModel) -> AgentToolResult:
        return AgentToolResult(content={"value": 1}, receipt_artifact=_Receipt("rcpt-1"))

    ledger = ToolCallLedger(_policy("inspect"))
    hooks = _Hooks()
    executor = ProbeExecutor(
        provider,
        [_tool("inspect", execute)],
        ledger,
        journal=hooks,
        usage_meter=_RaisingMeter(raise_on_success=ValueError("dataset_id not in scope")),
    )

    result = executor.run(phase="execute_probes", system_prompt="x", user_message="y")

    assert result.status == "completed"
    assert result.answer == "Recovered after meter failure."
    assert result.tool_calls == 1
    assert ledger.snapshot()["successful_tool_calls"] == 0
    terminal = [fields for event, fields in hooks.events if event == "tool_terminal"]
    assert terminal[-1]["outcome"] == "failed"
    assert "dataset_id not in scope" in terminal[-1]["error"]
    second_call_messages = provider.calls[1]["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert tool_messages
    assert json.loads(tool_messages[-1]["content"])["ok"] is False


def test_usage_meter_project_error_skips_the_call_without_a_journal_entry() -> None:
    executions = 0

    def execute(_args: BaseModel) -> AgentToolResult:
        nonlocal executions
        executions += 1
        return AgentToolResult(content={"value": 1})

    provider = _Provider(
        [
            LLMToolResponse(tool_calls=[_call("inspect")]),
            LLMToolResponse(content="Recovered after project failure."),
        ]
    )
    ledger = ToolCallLedger(_policy("inspect"))
    hooks = _Hooks()
    executor = ProbeExecutor(
        provider,
        [_tool("inspect", execute)],
        ledger,
        journal=hooks,
        usage_meter=_RaisingMeter(raise_on_project=ValueError("dataset_id not in scope")),
    )

    result = executor.run(phase="execute_probes", system_prompt="x", user_message="y")

    assert result.status == "completed"
    assert result.answer == "Recovered after project failure."
    assert executions == 0
    assert result.tool_calls == 0
    assert not any(event in {"tool_started", "tool_terminal"} for event, _fields in hooks.events)
    second_call_messages = provider.calls[1]["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert tool_messages
    assert json.loads(tool_messages[-1]["content"])["ok"] is False


def test_usage_meter_success_budget_error_still_terminates_the_run() -> None:
    provider = _Provider([LLMToolResponse(tool_calls=[_call("inspect")])])

    def execute(_args: BaseModel) -> AgentToolResult:
        return AgentToolResult(content={"value": 1}, receipt_artifact=_Receipt("rcpt-1"))

    ledger = ToolCallLedger(_policy("inspect"))
    hooks = _Hooks()
    budget_error = BudgetExceeded("meter enforced its own cap")
    executor = ProbeExecutor(
        provider,
        [_tool("inspect", execute)],
        ledger,
        journal=hooks,
        usage_meter=_RaisingMeter(raise_on_success=budget_error),
    )

    with pytest.raises(BudgetExceeded) as captured:
        executor.run(phase="execute_probes", system_prompt="x", user_message="y")

    assert captured.value is budget_error
    terminal = [fields for event, fields in hooks.events if event == "tool_terminal"]
    assert terminal[-1]["outcome"] == "failed"


def test_failure_history_is_capped_at_five_entries_of_160_chars() -> None:
    provider = _Provider(
        [
            LLMToolResponse(
                tool_calls=[_call(f"missing-{index}", str(index)) for index in range(7)]
            ),
            LLMToolResponse(content="Done."),
        ]
    )
    result = ProbeExecutor(
        provider,
        [_tool("inspect", lambda _args: AgentToolResult(content={}))],
        ToolCallLedger(_policy("inspect")),
    ).run(phase="execute_probes", system_prompt="x", user_message="y")

    assert result.status == "completed"
    assert len(result.failure_history) == 5
    assert all(len(entry) <= 160 for entry in result.failure_history)
    injected = provider.calls[1]["messages"][-1]["content"].splitlines()[1:]
    assert injected == list(result.failure_history)


def test_provider_rejected_is_counted_and_retried_but_unknown_error_is_uncertain() -> None:
    rejected_hooks = _Hooks()
    rejected_provider = _Provider(
        [ProviderCallRejectedError("HTTP 400"), LLMToolResponse(content="Recovered.")]
    )
    recovered = ProbeExecutor(
        rejected_provider,
        [_tool("inspect", lambda _args: AgentToolResult(content={}))],
        ToolCallLedger(_policy("inspect")),
        journal=rejected_hooks,
    ).run(phase="execute_probes", system_prompt="x", user_message="y")

    assert recovered.status == "completed"
    assert [
        fields["outcome"]
        for event, fields in rejected_hooks.events
        if event == "llm_terminal"
    ] == ["rejected", "completed"]

    uncertain_hooks = _Hooks()
    uncertain_provider = _Provider([RuntimeError("socket outcome unknown")])
    uncertain = ProbeExecutor(
        uncertain_provider,
        [_tool("inspect", lambda _args: AgentToolResult(content={}))],
        ToolCallLedger(_policy("inspect")),
        journal=uncertain_hooks,
    ).run(phase="execute_probes", system_prompt="x", user_message="y")

    assert uncertain.status == "failed"
    assert uncertain.error_code == "provider_outcome_uncertain"
    assert len(uncertain_provider.calls) == 1
    assert uncertain_hooks.events[-1][1]["outcome"] == "uncertain"


def test_jsonl_hooks_commit_receipt_under_claimed_attempt_fence(tmp_path: Any) -> None:
    budget = _policy("inspect")
    policy = sealed_policy(
        ExplorationPolicy(
            mode="open",
            dataset_scope=("dataset-1",),
            thinking_level="quick",
            coverage_targets=(),
            budget=budget,
            scoring_policy_version="score-v1",
            statistical_policy_version="stats-v1",
            tool_capability_digest="tools-v1",
        )
    )
    journal = JsonlExplorationJournal(tmp_path / "exploration.jsonl")
    journal.initialize(
        exploration_id="exploration-1",
        policy=policy,
        code_fingerprint="code-v1",
        data_state_witness="witness-v1",
    )
    journal.claim_attempt()
    journal.append_new("round_started", round_index=0)
    provider = _Provider(
        [
            LLMToolResponse(tool_calls=[_call("inspect")]),
            LLMToolResponse(content="Done."),
        ]
    )
    result = ProbeExecutor(
        provider,
        [
            _tool(
                "inspect",
                lambda _args: AgentToolResult(
                    content={"value": 1},
                    receipt_artifact=_Receipt("receipt-1"),
                ),
            )
        ],
        ToolCallLedger(budget),
        journal=JsonlProbeJournalHooks(journal),
    ).run(
        phase="execute_probes",
        system_prompt="x",
        user_message="y",
        run_id="run-jsonl",
    )

    state = journal.rebuild()
    assert result.status == "completed"
    assert state is not None
    assert state.pending_call_id is None
    assert state.pending_logical_step_id is None
    assert state.tool_calls_committed == 1
    assert list(state.step_receipt_refs.values()) == ["receipt-1"]
    assert [event.event_type for event in journal.events()][-7:] == [
        "llm_call_started",
        "llm_call_completed",
        "tool_call_started",
        "receipt_prepared",
        "receipt_committed",
        "llm_call_started",
        "llm_call_completed",
    ]


def test_jsonl_hooks_reject_an_unclaimed_executor_before_provider_call(tmp_path: Any) -> None:
    budget = _policy("inspect")
    policy = sealed_policy(
        ExplorationPolicy(
            mode="open",
            dataset_scope=("dataset-1",),
            thinking_level="quick",
            coverage_targets=(),
            budget=budget,
            scoring_policy_version="score-v1",
            statistical_policy_version="stats-v1",
            tool_capability_digest="tools-v1",
        )
    )
    journal = JsonlExplorationJournal(tmp_path / "unclaimed.jsonl")
    journal.initialize(
        exploration_id="exploration-1",
        policy=policy,
        code_fingerprint="code-v1",
        data_state_witness="witness-v1",
    )
    journal.append_new("round_started", round_index=0)
    provider = _Provider([LLMToolResponse(content="must not run")])

    with pytest.raises(EventTransitionError, match="claim an executor attempt"):
        ProbeExecutor(
            provider,
            [_tool("inspect", lambda _args: AgentToolResult(content={}))],
            ToolCallLedger(budget),
            journal=JsonlProbeJournalHooks(journal),
        ).run(phase="execute_probes", system_prompt="x", user_message="y")

    assert provider.calls == []


def test_a_malformed_tool_call_is_retried_not_fatal() -> None:
    """A provider that emits unparseable tool arguments has answered — the
    answer is just unusable. Killing the run discards every round already paid
    for; observed on deepseek-v4-flash at round 2 of the 2026-08-03 trial."""
    hooks = _Hooks()
    provider = _Provider(
        [
            MalformedProviderResponseError("LLM tool arguments are not valid JSON."),
            LLMToolResponse(content="Recovered."),
        ]
    )
    result = ProbeExecutor(
        provider,
        [_tool("inspect", lambda _args: AgentToolResult(content={}))],
        ToolCallLedger(_policy("inspect")),
        journal=hooks,
    ).run(phase="execute_probes", system_prompt="x", user_message="y")

    assert result.status == "completed"
    assert [
        fields["outcome"] for event, fields in hooks.events if event == "llm_terminal"
    ] == ["rejected", "completed"]
    assert "not valid JSON" in provider.calls[1]["messages"][-1]["content"]


class _IndexArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int


def test_a_later_empty_turn_is_retried_rather_than_ending_the_probe() -> None:
    """One empty turn early and one empty turn later are two independent model
    hiccups, not evidence the probe is stuck. Latching a single run-wide bool
    killed probes that had already committed receipts."""
    provider = _Provider(
        [
            LLMToolResponse(),
            LLMToolResponse(tool_calls=[_call("inspect")]),
            LLMToolResponse(),
            LLMToolResponse(content="Probe complete.", finish_reason="stop"),
        ]
    )
    result = ProbeExecutor(
        provider,
        [_tool("inspect", lambda _args: AgentToolResult(content={}))],
        ToolCallLedger(_policy("inspect")),
    ).run(phase="execute_probes", system_prompt="x", user_message="y")

    assert result.status == "completed"
    assert result.answer == "Probe complete."
    assert len(provider.calls) == 4


def test_an_oversized_native_batch_rejects_the_overflow_before_executing_any_tool() -> None:
    """A 14-call batch whose first two calls are filtered out used to execute 12
    tools and then raise on the 13th action index, leaving the journal holding
    work no caller could settle. The overflow must be refused up front."""
    executed: list[int] = []

    def execute(args: BaseModel) -> AgentToolResult:
        executed.append(cast(_IndexArgs, args).n)
        return AgentToolResult(content={})

    calls = [
        _call("ghost", "call-0", n=0),
        _call("ghost", "call-00", n=-1),
        *[
            _call("inspect", f"call-{index}", n=index)
            for index in range(1, 13)
        ],
    ]
    assert len(calls) == 14
    provider = _Provider(
        [
            LLMToolResponse(tool_calls=calls),
            LLMToolResponse(content="Probe complete.", finish_reason="stop"),
        ]
    )
    result = ProbeExecutor(
        provider,
        [_tool("inspect", execute, args_schema=_IndexArgs)],
        ToolCallLedger(_policy("inspect", max_calls=12)),
    ).run(phase="execute_probes", system_prompt="x", user_message="y")

    assert result.status == "completed"
    # Actions 13 and 14 are past the per-step ceiling and never ran.
    assert executed == list(range(1, 11))
    overflow = [
        message
        for message in provider.calls[1]["messages"]
        if message.get("role") == "tool"
        and "batch position" in str(message.get("content", ""))
    ]
    assert len(overflow) == 2
