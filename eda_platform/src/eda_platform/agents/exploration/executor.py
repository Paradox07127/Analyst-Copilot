"""Bounded, provider-neutral executor for one exploration probe loop.

The executor owns deterministic control-flow only.  Providers return the
existing native ``LLMToolResponse`` shape; local ``AgentTool`` objects retain
argument validation and execution authority.  Journal, phase selection and
tool-usage metering are narrow injectable seams so the loop is testable without
a live provider and recoverable when backed by ``JsonlExplorationJournal``.

Budget timing follows exploration plan sections 5.2/5.4: a whole native tool
batch is projected before any tool starts, successful calls settle actual
usage, and failed calls settle resource usage without incrementing the
successful-call counters.  Budget and cancellation exceptions are never
converted into model observations.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ValidationError

from eda_platform.agents.runtime import (
    AgentTool,
    AgentToolResult,
    canonical_tool_arguments,
)
from eda_platform.agents.tool_context import (
    HypothesisExecutionBinding,
    ToolExecutionContext,
    make_logical_step_id,
    tool_execution_scope,
)
from eda_platform.core.budget import BudgetExceeded
from eda_platform.core.cancellation import CancellationError
from eda_platform.core.exploration_budget import (
    ToolCallLedger,
    ToolCallProjection,
)
from eda_platform.core.exploration_journal import JsonlExplorationJournal
from eda_platform.core.ids import stable_hash
from eda_platform.core.kernel import SessionCancelled
from eda_platform.core.llm import LLMToolCall, LLMToolResponse
from eda_platform.core.llm_ledger import logical_llm_call

type ProbeExecutionStatus = Literal[
    "completed",
    "failed",
    "limit_reached",
    "budget_exhausted",
    "cancelled",
]
type LlmTerminalOutcome = Literal["completed", "rejected", "uncertain"]
type ToolTerminalOutcome = Literal["completed", "failed"]

_FAILURE_HISTORY_LIMIT = 5
_FAILURE_ENTRY_MAX_CHARS = 160
_FAILURE_HISTORY_NOTE = (
    "Recent failed probes (do not repeat the same path; repair it, choose a different "
    "direction, or conclude):"
)
_EMPTY_RESPONSE_RETRY = (
    "The previous response was empty. The only valid exits are to return non-empty text "
    "or call one or more tools from the current inventory. Try once more."
)
_CONTENT_FILTER_FINISH_REASONS = frozenset(
    {"content_filter", "content_filtered", "safety", "safety_filter"}
)
_TERMINAL_TOOL_ERRORS = (BudgetExceeded, SessionCancelled, CancellationError)
DEFAULT_MAX_TOOL_CALLS = 12


class NativeToolProvider(Protocol):
    """The only provider capability the executor needs."""

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolResponse: ...


class LlmResponseStore(Protocol):
    """Durable response body keyed by the journal's completed logical step."""

    def load_required(self, logical_step_id: str) -> LLMToolResponse: ...

    def remember(self, logical_step_id: str, response: LLMToolResponse) -> None: ...


@dataclass(frozen=True, slots=True)
class DurableToolResult:
    result: AgentToolResult
    usage: ToolCallProjection


def durable_tool_result_payload(value: DurableToolResult) -> dict[str, object]:
    """Canonical durable body covered by the authoritative journal digest."""
    return {
        "result": {
            "content": value.result.content,
            "artifacts": [_serializable_artifact(item) for item in value.result.artifacts],
            "receipt_artifact": (
                None
                if value.result.receipt_artifact is None
                else _serializable_artifact(value.result.receipt_artifact)
            ),
        },
        "usage": {
            "kind": value.usage.kind,
            "rows_scanned": value.usage.rows_scanned,
            "result_cells": value.usage.result_cells,
        },
    }


def durable_tool_result_digest(value: DurableToolResult) -> str:
    return stable_hash(durable_tool_result_payload(value), length=32)


class ToolResultStore(Protocol):
    """Durable local-tool body used to adopt a committed logical step."""

    def load_required(self, logical_step_id: str) -> DurableToolResult: ...

    def remember(self, logical_step_id: str, result: DurableToolResult) -> None: ...


@dataclass(slots=True)
class InMemoryLlmResponseStore:
    """Process-local default; production drivers inject a durable JSON store."""

    responses: dict[str, LLMToolResponse] = field(default_factory=dict)

    def load_required(self, logical_step_id: str) -> LLMToolResponse:
        try:
            return self.responses[logical_step_id]
        except KeyError as exc:
            raise KeyError(
                f"completed LLM response {logical_step_id!r} is unavailable; "
                "refusing to resend it."
            ) from exc

    def remember(self, logical_step_id: str, response: LLMToolResponse) -> None:
        prior = self.responses.get(logical_step_id)
        if prior is not None and prior != response:
            raise ValueError(f"LLM response {logical_step_id!r} cannot be replaced.")
        self.responses[logical_step_id] = response


@dataclass(slots=True)
class InMemoryToolResultStore:
    results: dict[str, DurableToolResult] = field(default_factory=dict)

    def load_required(self, logical_step_id: str) -> DurableToolResult:
        try:
            return self.results[logical_step_id]
        except KeyError as exc:
            raise KeyError(
                f"committed tool result {logical_step_id!r} is unavailable; "
                "refusing to re-run it."
            ) from exc

    def remember(self, logical_step_id: str, result: DurableToolResult) -> None:
        prior = self.results.get(logical_step_id)
        if prior is not None and prior != result:
            raise ValueError(f"tool result {logical_step_id!r} cannot be replaced.")
        self.results[logical_step_id] = result


class PhaseToolSelector(Protocol):
    """Re-evaluate the reachable tool subset before every model step."""

    def __call__(
        self,
        *,
        phase: str,
        step: int,
        registered_tools: tuple[str, ...],
    ) -> Sequence[str]: ...


@dataclass(frozen=True, slots=True)
class PhaseToolMap:
    """Fail-closed phase-to-tool selector convenient for production policies."""

    tools_by_phase: Mapping[str, Sequence[str]]

    def __call__(
        self,
        *,
        phase: str,
        step: int,
        registered_tools: tuple[str, ...],
    ) -> Sequence[str]:
        del step, registered_tools
        try:
            return self.tools_by_phase[phase]
        except KeyError as exc:
            raise ValueError(f"No tool subset is configured for phase {phase!r}.") from exc


class ToolUsageMeter(Protocol):
    """Project a batch and settle successful or failed tool resource usage."""

    def project(
        self,
        *,
        call: LLMToolCall,
        tool: AgentTool,
        arguments: BaseModel,
    ) -> ToolCallProjection: ...

    def success(
        self,
        *,
        call: LLMToolCall,
        tool: AgentTool,
        arguments: BaseModel,
        result: AgentToolResult,
        projected: ToolCallProjection,
    ) -> ToolCallProjection: ...

    def failure(
        self,
        *,
        call: LLMToolCall,
        tool: AgentTool,
        arguments: BaseModel,
        error: BaseException,
        projected: ToolCallProjection,
    ) -> ToolCallProjection | None: ...


@dataclass(frozen=True, slots=True)
class DefaultToolUsageMeter:
    """Count by tool name; resource estimates default to zero.

    A failed execution returns its projection rather than zero.  That is the
    fail-closed fallback when the adapter cannot observe actual scan/materialize
    counts, and prevents a repeatedly failing expensive probe from bypassing a
    non-zero projected rows/cells cap.
    """

    def project(
        self,
        *,
        call: LLMToolCall,
        tool: AgentTool,
        arguments: BaseModel,
    ) -> ToolCallProjection:
        del call, arguments
        return ToolCallProjection(kind=tool.name)

    def success(
        self,
        *,
        call: LLMToolCall,
        tool: AgentTool,
        arguments: BaseModel,
        result: AgentToolResult,
        projected: ToolCallProjection,
    ) -> ToolCallProjection:
        del call, tool, arguments, result
        return projected

    def failure(
        self,
        *,
        call: LLMToolCall,
        tool: AgentTool,
        arguments: BaseModel,
        error: BaseException,
        projected: ToolCallProjection,
    ) -> ToolCallProjection:
        del call, tool, arguments, error
        return projected


class ProbeJournalHooks(Protocol):
    """Invocation-level started/terminal journal contract.

    A terminal hook is required for every operation whose started hook
    succeeded.  Unknown, invalid and duplicate requests never started a local
    tool operation, so they deliberately produce no tool journal pair.
    """

    @property
    def attempt_epoch(self) -> int: ...

    def llm_started(self, *, call_id: str, step_id: str) -> None: ...

    def llm_terminal(
        self,
        *,
        call_id: str,
        step_id: str,
        outcome: LlmTerminalOutcome,
        response_digest: str | None = None,
        error: str | None = None,
    ) -> None: ...

    def tool_started(
        self,
        *,
        logical_step_id: str,
        input_fingerprint: str,
        tool_kind: str,
        projected_rows_scanned: int,
        projected_result_cells: int,
    ) -> None: ...

    def tool_terminal(
        self,
        *,
        logical_step_id: str,
        outcome: ToolTerminalOutcome,
        receipt_id: str | None = None,
        error: str | None = None,
        rows_scanned: int = 0,
        result_cells: int = 0,
        result_digest: str | None = None,
    ) -> None: ...


@dataclass(slots=True)
class NullProbeJournalHooks:
    """No-op hooks for isolated tests and non-durable callers."""

    attempt_epoch: int = 0

    def llm_started(self, *, call_id: str, step_id: str) -> None:
        del call_id, step_id

    def llm_terminal(
        self,
        *,
        call_id: str,
        step_id: str,
        outcome: LlmTerminalOutcome,
        response_digest: str | None = None,
        error: str | None = None,
    ) -> None:
        del call_id, step_id, outcome, response_digest, error

    def tool_started(
        self,
        *,
        logical_step_id: str,
        input_fingerprint: str,
        tool_kind: str,
        projected_rows_scanned: int,
        projected_result_cells: int,
    ) -> None:
        del (
            logical_step_id,
            input_fingerprint,
            tool_kind,
            projected_rows_scanned,
            projected_result_cells,
        )

    def tool_terminal(
        self,
        *,
        logical_step_id: str,
        outcome: ToolTerminalOutcome,
        receipt_id: str | None = None,
        error: str | None = None,
        rows_scanned: int = 0,
        result_cells: int = 0,
        result_digest: str | None = None,
    ) -> None:
        del (
            logical_step_id,
            outcome,
            receipt_id,
            error,
            rows_scanned,
            result_cells,
            result_digest,
        )


@dataclass(frozen=True, slots=True)
class JsonlProbeJournalHooks:
    """Map executor hooks onto the authoritative exploration journal.

    The supervisor must initialize the journal, claim an attempt and open a
    round before invoking the executor.  Every append is wrapped by
    ``fenced_side_effect``: an unclaimed or stale writer therefore fails before
    it can record execution.  Receipt preparation and commit share one fence.
    """

    journal: JsonlExplorationJournal

    @property
    def attempt_epoch(self) -> int:
        state = self.journal.rebuild()
        if state is None:
            raise RuntimeError("Initialize the exploration journal before executing probes.")
        return state.attempt_epoch

    def llm_started(self, *, call_id: str, step_id: str) -> None:
        with self.journal.fenced_side_effect():
            state = self.journal.rebuild()
            if state is not None and state.pending_call_id == call_id:
                if state.pending_call_step_id not in {None, step_id}:
                    raise ValueError("pending LLM call is bound to another logical step.")
                return
            self.journal.append_new(
                "llm_call_started", call_id=call_id, step_id=step_id
            )

    def llm_terminal(
        self,
        *,
        call_id: str,
        step_id: str,
        outcome: LlmTerminalOutcome,
        response_digest: str | None = None,
        error: str | None = None,
    ) -> None:
        with self.journal.fenced_side_effect():
            if outcome == "completed":
                if not response_digest:
                    raise ValueError("A completed LLM call requires response_digest.")
                self.journal.append_new(
                    "llm_call_completed",
                    call_id=call_id,
                    step_id=step_id,
                    response_digest=response_digest,
                )
            elif outcome == "rejected":
                self.journal.append_new(
                    "llm_call_rejected",
                    call_id=call_id,
                    error=error or "provider rejected the call",
                )
            else:
                self.journal.append_new(
                    "llm_call_uncertain",
                    call_id=call_id,
                    error=error or "provider call outcome is uncertain",
                )

    def tool_started(
        self,
        *,
        logical_step_id: str,
        input_fingerprint: str,
        tool_kind: str,
        projected_rows_scanned: int,
        projected_result_cells: int,
    ) -> None:
        with self.journal.fenced_side_effect():
            state = self.journal.rebuild()
            if state is not None and state.pending_logical_step_id == logical_step_id:
                expected = (
                    state.pending_tool_kind,
                    state.pending_tool_input_fingerprint,
                    state.pending_projected_rows_scanned,
                    state.pending_projected_result_cells,
                )
                actual = (
                    tool_kind,
                    input_fingerprint,
                    projected_rows_scanned,
                    projected_result_cells,
                )
                if expected != actual:
                    raise ValueError("pending tool step metadata cannot be replaced.")
                return
            self.journal.append_new(
                "tool_call_started",
                logical_step_id=logical_step_id,
                input_fingerprint=input_fingerprint,
                tool_kind=tool_kind,
                projected_rows_scanned=projected_rows_scanned,
                projected_result_cells=projected_result_cells,
            )

    def tool_terminal(
        self,
        *,
        logical_step_id: str,
        outcome: ToolTerminalOutcome,
        receipt_id: str | None = None,
        error: str | None = None,
        rows_scanned: int = 0,
        result_cells: int = 0,
        result_digest: str | None = None,
    ) -> None:
        with self.journal.fenced_side_effect():
            if outcome == "failed":
                self.journal.append_new(
                    "tool_call_failed",
                    logical_step_id=logical_step_id,
                    error=error or "tool execution failed",
                    rows_scanned=rows_scanned,
                    result_cells=result_cells,
                )
                return
            if not receipt_id:
                raise ValueError(
                    "A journaled successful probe requires an EvidenceReceipt id."
                )
            if not result_digest:
                raise ValueError(
                    "A journaled successful probe requires a durable result digest."
                )
            self.journal.append_new(
                "receipt_prepared",
                logical_step_id=logical_step_id,
                receipt_id=receipt_id,
                result_digest=result_digest,
            )
            self.journal.append_new(
                "receipt_committed",
                logical_step_id=logical_step_id,
                receipt_id=receipt_id,
                rows_scanned=rows_scanned,
                result_cells=result_cells,
                result_digest=result_digest,
            )


class ProviderCallRejectedError(RuntimeError):
    """A provider definitively rejected a request; a bounded retry is safe."""


class ProviderOutcomeUncertainError(RuntimeError):
    """The provider may have accepted a request; never resend the logical step."""


@dataclass(slots=True)
class ProbeExecutionResult:
    status: ProbeExecutionStatus
    answer: str = ""
    artifacts: list[Any] = field(default_factory=list)
    tool_calls: int = 0
    tool_names: list[str] = field(default_factory=list)
    error: str | None = None
    error_code: str | None = None
    failure_history: tuple[str, ...] = ()
    probe_fingerprints: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _PreparedCall:
    call: LLMToolCall
    tool: AgentTool
    arguments: BaseModel
    fingerprint: str
    projection: ToolCallProjection
    action_index: int


class ProbeExecutor:
    """Run one bounded native-tool exploration loop."""

    def __init__(
        self,
        provider: NativeToolProvider,
        tools: Sequence[AgentTool],
        ledger: ToolCallLedger,
        *,
        phase_tools: PhaseToolSelector | None = None,
        usage_meter: ToolUsageMeter | None = None,
        journal: ProbeJournalHooks | None = None,
        response_store: LlmResponseStore | None = None,
        tool_result_store: ToolResultStore | None = None,
        max_steps: int = 8,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        max_observation_chars: int = 12_000,
        cancel_check: Callable[[], object] | None = None,
        task: str = "exploration_probe_loop",
    ) -> None:
        if max_steps < 1 or max_tool_calls < 1:
            raise ValueError("Probe executor limits must be positive.")
        if max_observation_chars < 256:
            raise ValueError("max_observation_chars must be at least 256.")
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("Probe tool names must be unique.")
        if not task.strip():
            raise ValueError("task must be non-empty.")
        self._provider = provider
        self._tools = {tool.name: tool for tool in tools}
        self._registered_names = tuple(names)
        self._ledger = ledger
        self._phase_tools = phase_tools
        self._usage_meter = usage_meter or DefaultToolUsageMeter()
        self._journal = journal or NullProbeJournalHooks()
        self._response_store = response_store or InMemoryLlmResponseStore()
        self._tool_result_store = tool_result_store or InMemoryToolResultStore()
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        self._max_observation_chars = max_observation_chars
        self._cancel_check = cancel_check
        self._task = task

    def run(
        self,
        *,
        phase: str,
        system_prompt: str,
        user_message: str,
        run_id: str | None = None,
        seen_probe_fingerprints: set[str] | None = None,
        failure_history: Sequence[str] = (),
        completed_step_ids: set[str] | None = None,
        completed_response_digests: Mapping[str, str] | None = None,
        blocked_llm_call_ids: set[str] | None = None,
        hypothesis: HypothesisExecutionBinding | None = None,
    ) -> ProbeExecutionResult:
        if not phase.strip():
            raise ValueError("phase must be non-empty.")
        execution_run_id = run_id or "proberun_" + uuid.uuid4().hex
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        artifacts: list[Any] = []
        tool_calls = 0
        tool_names: list[str] = []
        seen = seen_probe_fingerprints if seen_probe_fingerprints is not None else set()
        completed = completed_step_ids if completed_step_ids is not None else set()
        response_digests = completed_response_digests
        blocked_calls = blocked_llm_call_ids or set()
        failures = [
            " ".join(str(item).split())[:_FAILURE_ENTRY_MAX_CHARS]
            for item in failure_history
        ]
        failures = failures[-_FAILURE_HISTORY_LIMIT:]
        empty_response_retried = False

        for step in range(1, self._max_steps + 1):
            self._checkpoint()
            phase_names = self._select_phase_tools(phase=phase, step=step)
            phase_registry = {name: self._tools[name] for name in phase_names}
            call_id = "llm_" + stable_hash(
                {"run_id": execution_run_id, "step": step}, length=24
            )
            llm_step_id = make_executor_llm_step_id(execution_run_id, step)
            request_messages = _with_failure_history(messages, failures)
            if call_id in blocked_calls:
                return self._result(
                    status="failed",
                    artifacts=artifacts,
                    tool_calls=tool_calls,
                    tool_names=tool_names,
                    failures=failures,
                    seen=seen,
                    error=(
                        f"Provider outcome for {call_id} is uncertain; refusing to "
                        "resend the logical request."
                    ),
                    error_code="provider_outcome_uncertain",
                )
            if llm_step_id in completed:
                try:
                    response = self._response_store.load_required(llm_step_id)
                except (KeyError, ValueError) as exc:
                    return self._result(
                        status="failed",
                        artifacts=artifacts,
                        tool_calls=tool_calls,
                        tool_names=tool_names,
                        failures=failures,
                        seen=seen,
                        error=_safe_error(exc),
                        error_code="completed_response_unavailable",
                    )
                expected_digest = (
                    None if response_digests is None else response_digests.get(llm_step_id)
                )
                actual_digest = stable_hash(response.model_dump(mode="json"), length=24)
                if response_digests is not None and (
                    expected_digest is None or actual_digest != expected_digest
                ):
                    return self._result(
                        status="failed",
                        artifacts=artifacts,
                        tool_calls=tool_calls,
                        tool_names=tool_names,
                        failures=failures,
                        seen=seen,
                        error="completed LLM response digest does not match the journal.",
                        error_code="completed_response_digest_mismatch",
                    )
            else:
                preflight = getattr(self._provider, "preflight_tool_call", None)
                if callable(preflight):
                    preflight(
                        task=self._task,
                        messages=request_messages,
                        tools=[
                            phase_registry[name].provider_schema()
                            for name in phase_names
                        ],
                    )
                self._journal.llm_started(call_id=call_id, step_id=llm_step_id)
                try:
                    with logical_llm_call(call_id):
                        response = self._provider.tool_call(
                            task=self._task,
                            messages=request_messages,
                            tools=[
                                phase_registry[name].provider_schema()
                                for name in phase_names
                            ],
                        )
                except BudgetExceeded as exc:
                    self._journal.llm_terminal(
                        call_id=call_id,
                        step_id=llm_step_id,
                        outcome="rejected",
                        error=_safe_error(exc),
                    )
                    raise
                except (SessionCancelled, CancellationError) as exc:
                    self._journal.llm_terminal(
                        call_id=call_id,
                        step_id=llm_step_id,
                        outcome="uncertain",
                        error=_safe_error(exc),
                    )
                    raise
                except ProviderCallRejectedError as exc:
                    error = _safe_error(exc)
                    self._journal.llm_terminal(
                        call_id=call_id,
                        step_id=llm_step_id,
                        outcome="rejected",
                        error=error,
                    )
                    failures = _add_failure(failures, call_id, error)
                    messages.append(
                        {
                            "role": "user",
                            "content": f"The provider rejected the prior request: {error}",
                        }
                    )
                    continue
                except Exception as exc:
                    error = _safe_error(exc)
                    self._journal.llm_terminal(
                        call_id=call_id,
                        step_id=llm_step_id,
                        outcome="uncertain",
                        error=error,
                    )
                    return self._result(
                        status="failed",
                        artifacts=artifacts,
                        tool_calls=tool_calls,
                        tool_names=tool_names,
                        failures=failures,
                        seen=seen,
                        error=error,
                        error_code="provider_outcome_uncertain",
                    )

                response_digest = stable_hash(response.model_dump(mode="json"), length=24)
                try:
                    self._response_store.remember(llm_step_id, response)
                except Exception as exc:
                    error = _safe_error(exc)
                    self._journal.llm_terminal(
                        call_id=call_id,
                        step_id=llm_step_id,
                        outcome="uncertain",
                        error=error,
                    )
                    return self._result(
                        status="failed",
                        artifacts=artifacts,
                        tool_calls=tool_calls,
                        tool_names=tool_names,
                        failures=failures,
                        seen=seen,
                        error=error,
                        error_code="response_persistence_failed",
                    )
                self._journal.llm_terminal(
                    call_id=call_id,
                    step_id=llm_step_id,
                    outcome="completed",
                    response_digest=response_digest,
                )
                completed.add(llm_step_id)

            finish_reason = response.finish_reason.strip().lower()
            if finish_reason == "length":
                return self._result(
                    status="failed",
                    artifacts=artifacts,
                    tool_calls=tool_calls,
                    tool_names=tool_names,
                    failures=failures,
                    seen=seen,
                    error="The provider stopped because the response length limit was reached.",
                    error_code="finish_reason_length",
                )
            if finish_reason in _CONTENT_FILTER_FINISH_REASONS:
                return self._result(
                    status="failed",
                    artifacts=artifacts,
                    tool_calls=tool_calls,
                    tool_names=tool_names,
                    failures=failures,
                    seen=seen,
                    error="The provider filtered the response content.",
                    error_code="content_filtered",
                )
            if not response.tool_calls:
                answer = response.content.strip()
                if answer:
                    return self._result(
                        status="completed",
                        artifacts=artifacts,
                        tool_calls=tool_calls,
                        tool_names=tool_names,
                        failures=failures,
                        seen=seen,
                        answer=answer,
                    )
                if empty_response_retried:
                    return self._result(
                        status="failed",
                        artifacts=artifacts,
                        tool_calls=tool_calls,
                        tool_names=tool_names,
                        failures=failures,
                        seen=seen,
                        error=(
                            "The provider returned no text and no tool calls after the "
                            "single empty-response retry."
                        ),
                        error_code="empty_response",
                    )
                empty_response_retried = True
                messages.append({"role": "user", "content": _EMPTY_RESPONSE_RETRY})
                continue

            messages.append(_assistant_message(response))
            prepared, immediate_messages, failures = self._prepare_batch(
                calls=response.tool_calls,
                phase_registry=phase_registry,
                phase_names=phase_names,
                seen=seen,
                failures=failures,
            )
            messages.extend(immediate_messages)

            if tool_calls + len(prepared) > self._max_tool_calls:
                return self._result(
                    status="limit_reached",
                    artifacts=artifacts,
                    tool_calls=tool_calls,
                    tool_names=tool_names,
                    failures=failures,
                    seen=seen,
                    error=(
                        "The probe reached its tool-call safety limit before the native "
                        "batch could start."
                    ),
                    error_code="tool_call_limit",
                )

            # One projection check for the entire executable native batch.  It
            # mutates no ledger state and occurs before any tool_started event.
            self._ledger.check_batch([item.projection for item in prepared])

            for item in prepared:
                self._checkpoint()
                tool_calls += 1
                tool_names.append(item.tool.name)
                sequence_index = make_tool_sequence_index(
                    step,
                    item.action_index,
                    max_tool_calls=self._max_tool_calls,
                )
                logical_step_id = make_logical_step_id(
                    execution_run_id,
                    item.call.call_id,
                    sequence_index,
                )
                if logical_step_id in completed:
                    try:
                        durable = self._tool_result_store.load_required(logical_step_id)
                    except (KeyError, ValueError) as exc:
                        return self._result(
                            status="failed",
                            artifacts=artifacts,
                            tool_calls=tool_calls,
                            tool_names=tool_names,
                            failures=failures,
                            seen=seen,
                            error=_safe_error(exc),
                            error_code="committed_tool_result_unavailable",
                        )
                    _require_matching_kind(item.projection, durable.usage)
                    seen.add(item.fingerprint)
                    self._append_tool_observation(
                        item=item,
                        result=durable.result,
                        artifacts=artifacts,
                        messages=messages,
                    )
                    continue
                self._journal.tool_started(
                    logical_step_id=logical_step_id,
                    input_fingerprint=item.fingerprint,
                    tool_kind=item.projection.kind,
                    projected_rows_scanned=item.projection.rows_scanned,
                    projected_result_cells=item.projection.result_cells,
                )
                seen.add(item.fingerprint)
                execution = ToolExecutionContext(
                    run_id=execution_run_id,
                    provider_call_id=item.call.call_id,
                    logical_step_id=logical_step_id,
                    attempt_epoch=self._journal.attempt_epoch,
                    sequence_index=sequence_index,
                    hypothesis=hypothesis,
                )
                try:
                    with tool_execution_scope(execution):
                        result = item.tool.execute(item.arguments)
                    if not isinstance(result, AgentToolResult):
                        raise TypeError("AgentTool.execute must return AgentToolResult.")
                except _TERMINAL_TOOL_ERRORS as exc:
                    try:
                        self._settle_failed_tool(
                            item,
                            logical_step_id=logical_step_id,
                            error=exc,
                        )
                    except BudgetExceeded:
                        # record_failure_usage commits counts before enforcing.
                        # A secondary rows/cells latch must not hide the budget
                        # or cancellation exception that actually stopped the
                        # in-flight tool.
                        pass
                    raise
                except Exception as exc:
                    self._settle_failed_tool(item, logical_step_id=logical_step_id, error=exc)
                    error = _safe_error(exc)
                    failures = _add_failure(failures, item.fingerprint, error)
                    messages.append(
                        _tool_message(
                            item.call,
                            {"ok": False, "error": error},
                            limit=self._max_observation_chars,
                        )
                    )
                    continue

                actual = self._usage_meter.success(
                    call=item.call,
                    tool=item.tool,
                    arguments=item.arguments,
                    result=result,
                    projected=item.projection,
                )
                _require_matching_kind(item.projection, actual)
                receipt_id = _receipt_id(result.receipt_artifact)
                durable = DurableToolResult(result=result, usage=actual)
                try:
                    self._tool_result_store.remember(logical_step_id, durable)
                except Exception as exc:
                    self._journal.tool_terminal(
                        logical_step_id=logical_step_id,
                        outcome="failed",
                        error=_safe_error(exc),
                        rows_scanned=actual.rows_scanned,
                        result_cells=actual.result_cells,
                    )
                    self._ledger.record_failure_usage(
                        actual.kind,
                        rows_scanned=actual.rows_scanned,
                        result_cells=actual.result_cells,
                    )
                    raise
                try:
                    self._ledger.record_success(
                        actual.kind,
                        rows_scanned=actual.rows_scanned,
                        result_cells=actual.result_cells,
                    )
                except BudgetExceeded as exc:
                    self._journal.tool_terminal(
                        logical_step_id=logical_step_id,
                        outcome="failed",
                        error=_safe_error(exc),
                        rows_scanned=actual.rows_scanned,
                        result_cells=actual.result_cells,
                    )
                    raise
                self._journal.tool_terminal(
                    logical_step_id=logical_step_id,
                    outcome="completed",
                    receipt_id=receipt_id,
                    rows_scanned=actual.rows_scanned,
                    result_cells=actual.result_cells,
                    result_digest=durable_tool_result_digest(durable),
                )
                completed.add(logical_step_id)
                self._append_tool_observation(
                    item=item,
                    result=result,
                    artifacts=artifacts,
                    messages=messages,
                )

        return self._result(
            status="limit_reached",
            artifacts=artifacts,
            tool_calls=tool_calls,
            tool_names=tool_names,
            failures=failures,
            seen=seen,
            error="The probe reached its reasoning-step safety limit.",
            error_code="step_limit",
        )

    def _prepare_batch(
        self,
        *,
        calls: Sequence[LLMToolCall],
        phase_registry: Mapping[str, AgentTool],
        phase_names: tuple[str, ...],
        seen: set[str],
        failures: list[str],
    ) -> tuple[list[_PreparedCall], list[dict[str, Any]], list[str]]:
        prepared_parts: list[tuple[LLMToolCall, AgentTool, BaseModel, str, int]] = []
        observations: list[dict[str, Any]] = []
        tentative = set(seen)

        for action_index, call in enumerate(calls, start=1):
            tool = phase_registry.get(call.name)
            raw_fingerprint = canonical_probe_fingerprint(call.name, call.arguments)
            if tool is None:
                error = f"Unknown tool {call.name!r} for the current phase."
                failures = _add_failure(failures, raw_fingerprint, error)
                observations.append(
                    _tool_message(
                        call,
                        {
                            "ok": False,
                            "error": error,
                            # Deliberately complete and untruncated: this is the
                            # exact current-step inventory, not the global registry.
                            "available_tools": list(phase_names),
                        },
                        limit=None,
                    )
                )
                continue
            try:
                canonical_arguments = canonical_tool_arguments(
                    tool.args_schema, call.arguments
                )
                arguments = tool.args_schema.model_validate(canonical_arguments)
            except ValidationError as exc:
                error = "Tool arguments did not match the declared schema."
                failures = _add_failure(failures, raw_fingerprint, error)
                observations.append(
                    _tool_message(
                        call,
                        {
                            "ok": False,
                            "error": error,
                            "details": _validation_feedback(exc),
                        },
                        limit=self._max_observation_chars,
                    )
                )
                continue

            fingerprint = canonical_probe_fingerprint(call.name, canonical_arguments)
            if fingerprint in tentative:
                error = (
                    "Probe rejected without execution: its canonical fingerprint was "
                    "already attempted. Choose a different direction or conclude."
                )
                failures = _add_failure(failures, fingerprint, error)
                observations.append(
                    _tool_message(
                        call,
                        {
                            "ok": False,
                            "error": error,
                            "probe_fingerprint": fingerprint,
                            "duplicate": True,
                        },
                        limit=self._max_observation_chars,
                    )
                )
                continue
            tentative.add(fingerprint)
            prepared_parts.append((call, tool, arguments, fingerprint, action_index))

        prepared: list[_PreparedCall] = []
        for call, tool, arguments, fingerprint, action_index in prepared_parts:
            projection = self._usage_meter.project(
                call=call,
                tool=tool,
                arguments=arguments,
            )
            prepared.append(
                _PreparedCall(
                    call=call,
                    tool=tool,
                    arguments=arguments,
                    fingerprint=fingerprint,
                    projection=projection,
                    action_index=action_index,
                )
            )
        return prepared, observations, failures

    def _settle_failed_tool(
        self,
        item: _PreparedCall,
        *,
        logical_step_id: str,
        error: BaseException,
    ) -> None:
        usage = self._usage_meter.failure(
            call=item.call,
            tool=item.tool,
            arguments=item.arguments,
            error=error,
            projected=item.projection,
        )
        if usage is None:
            # ``None`` is reserved for adapters that can prove no rows/cells
            # were touched (for example, a schema rejection before execution).
            self._journal.tool_terminal(
                logical_step_id=logical_step_id,
                outcome="failed",
                error=_safe_error(error),
            )
            return
        _require_matching_kind(item.projection, usage)
        self._journal.tool_terminal(
            logical_step_id=logical_step_id,
            outcome="failed",
            error=_safe_error(error),
            rows_scanned=usage.rows_scanned,
            result_cells=usage.result_cells,
        )
        self._ledger.record_failure_usage(
            usage.kind,
            rows_scanned=usage.rows_scanned,
            result_cells=usage.result_cells,
        )

    def _append_tool_observation(
        self,
        *,
        item: _PreparedCall,
        result: AgentToolResult,
        artifacts: list[Any],
        messages: list[dict[str, Any]],
    ) -> None:
        call_artifacts = list(result.artifacts)
        if result.receipt_artifact is not None:
            call_artifacts.append(result.receipt_artifact)
        artifacts.extend(call_artifacts)
        content = (
            result.content
            if isinstance(result.content, dict)
            else {"result": result.content}
        )
        messages.append(
            _tool_message(
                item.call,
                {"ok": True, **content},
                limit=self._max_observation_chars,
            )
        )

    def _select_phase_tools(self, *, phase: str, step: int) -> tuple[str, ...]:
        if self._phase_tools is None:
            return self._registered_names
        selected = tuple(
            self._phase_tools(
                phase=phase,
                step=step,
                registered_tools=self._registered_names,
            )
        )
        if len(selected) != len(set(selected)):
            raise ValueError(f"Tool selector returned duplicate names for phase {phase!r}.")
        unknown = [name for name in selected if name not in self._tools]
        if unknown:
            raise ValueError(
                "Tool selector cannot add unregistered capabilities: " + ", ".join(unknown)
            )
        return selected

    def _checkpoint(self) -> None:
        if self._cancel_check is not None:
            self._cancel_check()

    @staticmethod
    def _result(
        *,
        status: ProbeExecutionStatus,
        artifacts: Sequence[Any],
        tool_calls: int,
        tool_names: Sequence[str],
        failures: Sequence[str],
        seen: set[str],
        answer: str = "",
        error: str | None = None,
        error_code: str | None = None,
    ) -> ProbeExecutionResult:
        return ProbeExecutionResult(
            status=status,
            answer=answer,
            artifacts=_unique_artifacts(artifacts),
            tool_calls=tool_calls,
            tool_names=list(tool_names),
            error=error,
            error_code=error_code,
            failure_history=tuple(failures[-_FAILURE_HISTORY_LIMIT:]),
            probe_fingerprints=tuple(sorted(seen)),
        )


def canonical_probe_fingerprint(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Canonical, wording-insensitive identity for a probe request.

    SQL probes follow the established investigation-loop rule (trim a trailing
    semicolon, collapse whitespace, lowercase) and deliberately exclude
    ``purpose`` prose.  Other probes hash canonical JSON arguments after
    recursively normalizing mapping order and cosmetic string whitespace.
    """
    normalized_tool = tool_name.strip().lower()
    sql = arguments.get("sql")
    if isinstance(sql, str):
        normalized_sql = re.sub(r"\s+", " ", sql.strip().rstrip(";").strip()).lower()
        payload: Any = {"tool": normalized_tool, "sql": normalized_sql}
    else:
        payload = {
            "tool": normalized_tool,
            "arguments": _canonical_value(arguments),
        }
    return stable_hash(payload, length=16)


def make_executor_llm_step_id(run_id: str, step: int) -> str:
    if step < 1:
        raise ValueError("executor LLM step must be positive")
    return "llmstep_" + stable_hash({"run_id": run_id, "step": step}, length=24)


def make_tool_sequence_index(
    step: int,
    action_index: int,
    *,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> int:
    if step < 1 or action_index < 1 or action_index > max_tool_calls:
        raise ValueError("invalid executor tool sequence position")
    return (step - 1) * max_tool_calls + action_index


def split_tool_sequence_index(
    sequence_index: int,
    *,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> tuple[int, int]:
    if sequence_index < 1:
        raise ValueError("executor tool sequence index must be positive")
    zero_based = sequence_index - 1
    return zero_based // max_tool_calls + 1, zero_based % max_tool_calls + 1


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.split())
    return value


def _with_failure_history(
    messages: Sequence[dict[str, Any]], failures: Sequence[str]
) -> list[dict[str, Any]]:
    copied = [dict(message) for message in messages]
    if failures:
        copied.append(
            {
                "role": "user",
                "content": _FAILURE_HISTORY_NOTE
                + "\n"
                + "\n".join(failures[-_FAILURE_HISTORY_LIMIT:]),
            }
        )
    return copied


def _add_failure(failures: Sequence[str], fingerprint: str, reason: str) -> list[str]:
    one_line = " ".join(reason.split())
    entry = f"[fp:{fingerprint}] {one_line}"[:_FAILURE_ENTRY_MAX_CHARS]
    return [*failures, entry][-_FAILURE_HISTORY_LIMIT:]


def _assistant_message(response: LLMToolResponse) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": response.content,
        "tool_calls": [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in response.tool_calls
        ],
    }
    reasoning_content = response.provider_state.get("reasoning_content")
    if isinstance(reasoning_content, str):
        message["reasoning_content"] = reasoning_content
    return message


def _tool_message(
    call: LLMToolCall,
    observation: Mapping[str, Any],
    *,
    limit: int | None,
) -> dict[str, Any]:
    content = _observation_text(observation, limit=limit) if limit is not None else json.dumps(
        observation,
        ensure_ascii=False,
        default=str,
    )
    return {
        "role": "tool",
        "tool_call_id": call.call_id,
        "name": call.name,
        "content": content,
    }


def _observation_text(observation: Mapping[str, Any], *, limit: int) -> str:
    """Serialize an observation with at most one executor truncation envelope."""
    text = json.dumps(observation, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    envelope: dict[str, Any] = {
        "ok": bool(observation.get("ok", False)),
        "truncated": True,
        "original_chars": len(text),
        "preview": "",
    }
    # JSON escaping can expand the preview, so shrink against the serialized
    # envelope rather than assuming one input character equals one output char.
    preview_chars = max(0, limit - len(json.dumps(envelope, ensure_ascii=False)))
    envelope["preview"] = text[:preview_chars]
    encoded = json.dumps(envelope, ensure_ascii=False)
    while len(encoded) > limit and envelope["preview"]:
        overflow = len(encoded) - limit
        envelope["preview"] = envelope["preview"][: -max(1, overflow)]
        encoded = json.dumps(envelope, ensure_ascii=False)
    return encoded


def _validation_feedback(exc: ValidationError) -> list[dict[str, str]]:
    return [
        {
            "field": ".".join(str(part) for part in error.get("loc", ())),
            "message": str(error.get("msg", "invalid value")),
        }
        for error in exc.errors(include_url=False)
    ]


def _safe_error(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {message}"[:800]


def _artifact_id(artifact: Any | None) -> str | None:
    value = getattr(artifact, "id", None)
    return value if isinstance(value, str) and value else None


def _receipt_id(artifact: Any | None) -> str | None:
    payload = getattr(artifact, "payload", None)
    if isinstance(payload, Mapping):
        value = payload.get("receipt_id")
        if isinstance(value, str) and value:
            return value
    # Lightweight executor tests and legacy adapters expose only ``id``.
    return _artifact_id(artifact)


def _serializable_artifact(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    artifact_id = getattr(value, "id", None)
    if isinstance(artifact_id, str) and artifact_id:
        return {"id": artifact_id}
    raise TypeError(f"durable tool artifact {type(value).__name__} is not serializable.")


def _unique_artifacts(artifacts: Sequence[Any]) -> list[Any]:
    unique: list[Any] = []
    seen: set[str] = set()
    for artifact in artifacts:
        key = _artifact_id(artifact) or f"object:{id(artifact)}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(artifact)
    return unique


def _require_matching_kind(
    projected: ToolCallProjection,
    actual: ToolCallProjection,
) -> None:
    if projected.kind != actual.kind:
        raise ValueError(
            "Tool usage kind changed between projection and settlement: "
            f"{projected.kind!r} -> {actual.kind!r}."
        )


__all__ = [
    "DefaultToolUsageMeter",
    "JsonlProbeJournalHooks",
    "NativeToolProvider",
    "NullProbeJournalHooks",
    "PhaseToolMap",
    "PhaseToolSelector",
    "ProbeExecutionResult",
    "ProbeExecutionStatus",
    "ProbeExecutor",
    "ProbeJournalHooks",
    "ProviderCallRejectedError",
    "ProviderOutcomeUncertainError",
    "ToolUsageMeter",
    "canonical_probe_fingerprint",
]
