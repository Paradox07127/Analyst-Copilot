"""W1/W3/W4 probe-efficiency guards (exploration speedup plan section 4)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from eda_platform.agents.exploration.executor import ProbeExecutor
from eda_platform.agents.runtime import AgentTool, AgentToolResult
from eda_platform.core.exploration_budget import ToolCallLedger
from eda_platform.core.llm import LLMToolCall, LLMToolResponse
from eda_platform.schemas.exploration_budget import (
    ExplorationBudgetPolicy,
    SessionBudgetPolicyModel,
)


class _ValueArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int = Field(default=0)


class _SliceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    where_sql: str | None = Field(default=None, min_length=1)


@dataclass(frozen=True)
class _Receipt:
    id: str
    payload: dict[str, Any] = field(default_factory=dict)


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


def _policy(*kinds: str, max_calls: int = 12) -> ExplorationBudgetPolicy:
    return ExplorationBudgetPolicy(
        llm=SessionBudgetPolicyModel(max_requests=50),
        max_successful_tool_calls=max_calls,
        max_tool_calls_by_kind={kind: max_calls for kind in kinds},
        max_rows_scanned=None,
        max_result_cells=None,
        idle_timeout_seconds=60,
        max_rounds=4,
    )


def _tool(
    name: str,
    execute: Any,
    *,
    args_schema: type[BaseModel] = _ValueArgs,
) -> AgentTool:
    return AgentTool(
        name=name,
        description=f"Execute {name}.",
        args_schema=args_schema,
        execute=execute,
    )


def _call(name: str, call_id: str, **arguments: Any) -> LLMToolCall:
    return LLMToolCall(call_id=call_id, name=name, arguments=arguments)


def _offered_tools(provider: _Provider, call_index: int) -> list[str]:
    return [tool["name"] for tool in provider.calls[call_index]["tools"]]


def _last_tool_observation(provider: _Provider, call_index: int) -> dict[str, Any]:
    tool_messages = [
        message
        for message in provider.calls[call_index]["messages"]
        if message.get("role") == "tool"
    ]
    return dict(json.loads(tool_messages[-1]["content"]))


def _message_texts(provider: _Provider, call_index: int) -> list[str]:
    return [
        str(message.get("content", ""))
        for message in provider.calls[call_index]["messages"]
    ]


def _run(executor: ProbeExecutor, run_id: str) -> Any:
    return executor.run(
        phase="execute_probes",
        system_prompt="Use tools.",
        user_message="Probe.",
        run_id=run_id,
    )


# --- W1: run-level error-fingerprint circuit breaker -------------------------


def _flaky_executor(
    responses: list[LLMToolResponse],
    *,
    prior_failure_events: Any = None,
    errors: list[str] | None = None,
) -> tuple[ProbeExecutor, _Provider, list[int]]:
    provider = _Provider(responses)
    executions: list[int] = []
    error_texts = list(errors or [])

    def flaky(args: BaseModel) -> AgentToolResult:
        executions.append(args.value)  # type: ignore[attr-defined]
        if error_texts:
            raise RuntimeError(error_texts.pop(0))
        raise RuntimeError(f"precondition failed for metric {args.value}")  # type: ignore[attr-defined]

    tools = [
        _tool("flaky", flaky),
        _tool("inspect", lambda _args: AgentToolResult(content={"ok": True})),
    ]
    kwargs: dict[str, Any] = {}
    if prior_failure_events is not None:
        kwargs["prior_failure_events"] = prior_failure_events
    executor = ProbeExecutor(
        provider,
        tools,
        ToolCallLedger(_policy("flaky", "inspect")),
        **kwargs,
    )
    return executor, provider, executions


def test_second_identical_error_fingerprint_disables_tool_for_next_session() -> None:
    executor, provider, executions = _flaky_executor(
        [
            LLMToolResponse(tool_calls=[_call("flaky", "c1", value=1)]),
            LLMToolResponse(tool_calls=[_call("flaky", "c2", value=2)]),
            LLMToolResponse(content="session one over."),
            LLMToolResponse(content="session two over."),
        ]
    )

    first = _run(executor, "run-s1")
    assert first.status == "completed"
    assert executions == [1, 2]

    second = _run(executor, "run-s2")
    assert second.status == "completed"
    # The disabled tool is out of the offered inventory for the whole session.
    assert _offered_tools(provider, 3) == ["inspect"]
    note = (
        "tool flaky is disabled for this run after repeated failure: "
        "RuntimeError: precondition failed for metric N"
    )
    assert any(note in text for text in _message_texts(provider, 3))


def test_distinct_error_fingerprints_do_not_trip_the_breaker() -> None:
    executor, provider, executions = _flaky_executor(
        [
            LLMToolResponse(tool_calls=[_call("flaky", "c1", value=1)]),
            LLMToolResponse(tool_calls=[_call("flaky", "c2", value=2)]),
            LLMToolResponse(content="session one over."),
            LLMToolResponse(content="session two over."),
        ],
        errors=["alpha exploded", "beta timed out"],
    )

    _run(executor, "run-s1")
    _run(executor, "run-s2")
    assert executions == [1, 2]
    assert _offered_tools(provider, 3) == ["flaky", "inspect"]
    assert not any("disabled for this run" in text for text in _message_texts(provider, 3))


def test_disabled_tool_call_is_rejected_without_execution() -> None:
    executor, provider, executions = _flaky_executor(
        [
            LLMToolResponse(tool_calls=[_call("flaky", "c1", value=1)]),
            LLMToolResponse(tool_calls=[_call("flaky", "c2", value=2)]),
            LLMToolResponse(content="session one over."),
            # Session two: a scripted provider stubbornly calls the disabled tool.
            LLMToolResponse(tool_calls=[_call("flaky", "c3", value=3)]),
            LLMToolResponse(content="session two over."),
        ]
    )

    _run(executor, "run-s1")
    result = _run(executor, "run-s2")
    assert result.status == "completed"
    assert executions == [1, 2]
    assert result.tool_calls == 0
    rejection = _last_tool_observation(provider, 4)
    assert rejection["ok"] is False
    assert "disabled for this run" in rejection["error"]


@dataclass(frozen=True)
class _JournalEvent:
    event_type: str
    logical_step_id: str
    tool_kind: str = "legacy_unknown"
    error: str = ""


def test_resume_rebuilds_breaker_counter_from_prior_journal_events() -> None:
    prior = [
        _JournalEvent("tool_call_started", "step-1", tool_kind="flaky"),
        _JournalEvent(
            "tool_call_failed",
            "step-1",
            error="RuntimeError: precondition failed for metric 42",
        ),
        _JournalEvent("tool_call_started", "step-2", tool_kind="flaky"),
        _JournalEvent(
            "tool_call_failed",
            "step-2",
            error="RuntimeError: precondition failed for metric 7",
        ),
    ]
    executor, provider, executions = _flaky_executor(
        [LLMToolResponse(content="post-resume session over.")],
        prior_failure_events=prior,
    )

    result = _run(executor, "run-resumed")
    assert result.status == "completed"
    assert executions == []
    assert _offered_tools(provider, 0) == ["inspect"]
    assert any(
        "tool flaky is disabled for this run after repeated failure" in text
        for text in _message_texts(provider, 0)
    )


# --- W3: sufficient-evidence wind-down ---------------------------------------


def _adjudicating_tool(outcomes: list[str | None]) -> AgentTool:
    remaining = list(outcomes)

    def execute(_args: BaseModel) -> AgentToolResult:
        outcome = remaining.pop(0)
        payload: dict[str, Any] = {
            "receipt_id": f"rcpt-{len(remaining)}",
            "statistics": None
            if outcome is None
            else {"hypothesis_id": "hyp-1", "hypothesis_outcome": outcome},
        }
        return AgentToolResult(
            content={"ok": True},
            receipt_artifact=_Receipt(id=f"rcpt-{len(remaining)}", payload=payload),
        )

    return _tool("stat", execute)


def test_first_adjudicating_receipt_adds_corroboration_guidance() -> None:
    provider = _Provider(
        [
            LLMToolResponse(tool_calls=[_call("stat", "c1", value=1)]),
            LLMToolResponse(content="Concluded."),
        ]
    )
    executor = ProbeExecutor(
        provider,
        [_adjudicating_tool(["supports"])],
        ToolCallLedger(_policy("stat")),
    )

    result = _run(executor, "run-w3a")
    assert result.status == "completed"
    guidance = (
        "You already hold adjudicating evidence for this hypothesis. "
        "Corroborate once at most, then conclude."
    )
    assert not any(guidance in text for text in _message_texts(provider, 0))
    assert any(guidance in text for text in _message_texts(provider, 1))
    assert _offered_tools(provider, 1) == ["stat"]


def test_second_same_direction_receipt_empties_the_tool_inventory() -> None:
    provider = _Provider(
        [
            LLMToolResponse(tool_calls=[_call("stat", "c1", value=1)]),
            LLMToolResponse(tool_calls=[_call("stat", "c2", value=2)]),
            LLMToolResponse(content="Concluded."),
        ]
    )
    executor = ProbeExecutor(
        provider,
        [_adjudicating_tool(["supports", "supports"])],
        ToolCallLedger(_policy("stat")),
    )

    result = _run(executor, "run-w3b")
    assert result.status == "completed"
    assert result.tool_calls == 2
    assert _offered_tools(provider, 2) == []
    assert any("conclu" in text.lower() for text in _message_texts(provider, 2))


def test_opposite_direction_receipts_do_not_wind_down() -> None:
    provider = _Provider(
        [
            LLMToolResponse(tool_calls=[_call("stat", "c1", value=1)]),
            LLMToolResponse(tool_calls=[_call("stat", "c2", value=2)]),
            LLMToolResponse(content="Concluded."),
        ]
    )
    executor = ProbeExecutor(
        provider,
        [_adjudicating_tool(["supports", "contradicts"])],
        ToolCallLedger(_policy("stat")),
    )

    result = _run(executor, "run-w3c")
    assert result.status == "completed"
    assert _offered_tools(provider, 2) == ["stat"]


def test_untyped_receipts_never_trigger_wind_down() -> None:
    provider = _Provider(
        [
            LLMToolResponse(tool_calls=[_call("stat", "c1", value=1)]),
            LLMToolResponse(tool_calls=[_call("stat", "c2", value=2)]),
            LLMToolResponse(content="Concluded."),
        ]
    )
    executor = ProbeExecutor(
        provider,
        [_adjudicating_tool([None, None])],
        ToolCallLedger(_policy("stat")),
    )

    result = _run(executor, "run-w3d")
    assert result.status == "completed"
    assert _offered_tools(provider, 2) == ["stat"]
    assert not any(
        "adjudicating evidence" in text for text in _message_texts(provider, 2)
    )


# --- W4: zero-row filter family fingerprint ----------------------------------


def _slice_executor(
    responses: list[LLMToolResponse],
    *,
    rows_by_call: list[int],
) -> tuple[ProbeExecutor, _Provider, list[str | None]]:
    provider = _Provider(responses)
    rows = list(rows_by_call)
    executed_filters: list[str | None] = []

    def profile_slice(args: BaseModel) -> AgentToolResult:
        executed_filters.append(args.where_sql)  # type: ignore[attr-defined]
        return AgentToolResult(
            content={"rows_in_slice": rows.pop(0), "receipt_id": "r"},
            receipt_artifact=_Receipt(id="r"),
        )

    executor = ProbeExecutor(
        provider,
        [_tool("profile_slice", profile_slice, args_schema=_SliceArgs)],
        ToolCallLedger(_policy("profile_slice")),
    )
    return executor, provider, executed_filters


def test_third_zero_row_family_variant_is_rejected_without_execution() -> None:
    executor, provider, executed = _slice_executor(
        [
            LLMToolResponse(
                tool_calls=[_call("profile_slice", "c1", dataset_id="d",
                                  where_sql="region = 'north'")]
            ),
            LLMToolResponse(
                tool_calls=[_call("profile_slice", "c2", dataset_id="d",
                                  where_sql="region = 'south'")]
            ),
            LLMToolResponse(
                tool_calls=[_call("profile_slice", "c3", dataset_id="d",
                                  where_sql="region = 'east'")]
            ),
            LLMToolResponse(content="Concluded."),
        ],
        rows_by_call=[0, 0, 999],
    )

    result = _run(executor, "run-w4a")
    assert result.status == "completed"
    assert executed == ["region = 'north'", "region = 'south'"]
    assert result.tool_calls == 2
    rejection = _last_tool_observation(provider, 3)
    assert rejection["ok"] is False
    assert (
        "this filter family has matched zero rows twice; "
        "revise the plan instead of the value"
    ) in rejection["error"]


def test_zero_row_family_with_a_non_zero_hit_is_not_rejected() -> None:
    executor, _provider, executed = _slice_executor(
        [
            LLMToolResponse(
                tool_calls=[_call("profile_slice", "c1", dataset_id="d",
                                  where_sql="region = 'north'")]
            ),
            LLMToolResponse(
                tool_calls=[_call("profile_slice", "c2", dataset_id="d",
                                  where_sql="region = 'south'")]
            ),
            LLMToolResponse(
                tool_calls=[_call("profile_slice", "c3", dataset_id="d",
                                  where_sql="region = 'east'")]
            ),
            LLMToolResponse(content="Concluded."),
        ],
        rows_by_call=[0, 5, 0],
    )

    result = _run(executor, "run-w4b")
    assert result.status == "completed"
    assert executed == ["region = 'north'", "region = 'south'", "region = 'east'"]


def test_different_columns_or_operators_are_unaffected() -> None:
    executor, _provider, executed = _slice_executor(
        [
            LLMToolResponse(
                tool_calls=[_call("profile_slice", "c1", dataset_id="d",
                                  where_sql="region = 'north'")]
            ),
            LLMToolResponse(
                tool_calls=[_call("profile_slice", "c2", dataset_id="d",
                                  where_sql="region = 'south'")]
            ),
            LLMToolResponse(
                tool_calls=[_call("profile_slice", "c3", dataset_id="d",
                                  where_sql="city = 'lyon'")]
            ),
            LLMToolResponse(
                tool_calls=[_call("profile_slice", "c4", dataset_id="d",
                                  where_sql="region > 'a'")]
            ),
            LLMToolResponse(content="Concluded."),
        ],
        rows_by_call=[0, 0, 3, 3],
    )

    result = _run(executor, "run-w4c")
    assert result.status == "completed"
    assert executed == [
        "region = 'north'",
        "region = 'south'",
        "city = 'lyon'",
        "region > 'a'",
    ]


def test_filter_family_fingerprint_excludes_values() -> None:
    from eda_platform.agents.exploration.executor import zero_row_filter_family

    same_a = zero_row_filter_family("region = 'north' AND year > 2020")
    same_b = zero_row_filter_family("region = 'SOUTH-42'  and  year > 1999")
    assert same_a is not None
    assert same_a == same_b
    assert zero_row_filter_family("city = 'north' AND year > 2020") != same_a
    assert zero_row_filter_family("region LIKE 'north%' AND year > 2020") != same_a
