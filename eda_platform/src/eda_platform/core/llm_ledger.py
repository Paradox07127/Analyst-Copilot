"""Single accounting choke point for LLM spend.

Every call that reaches a provider passes through :class:`LedgerLLMClient`, which
emits one ``llm_usage`` trace event per call. Domain drivers still emit their own
``llm_call`` events for readability, but those are narrative, not accounting:
a new call site that forgets one stays visible in the ledger.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel

from eda_platform.core.budget import (
    BudgetReservation,
    BudgetUsageUncertain,
    SessionBudgetPolicy,
    SessionBudgetState,
)
from eda_platform.core.ids import stable_hash
from eda_platform.core.llm import ProviderUnavailableError, is_offline_client
from eda_platform.core.provider_registry import pricing_per_1m
from eda_platform.schemas.sessions import TraceEvent

if TYPE_CHECKING:
    from collections.abc import Callable

    from eda_platform.core.llm import LLMClient, LLMResultMetadata

T = TypeVar("T", bound=BaseModel)

LLM_USAGE_EVENT = "llm_usage"
BUDGET_REJECTED_EVENT = "budget_rejected"
BUDGET_RESERVED_EVENT = "budget_reserved"
BUDGET_SETTLED_EVENT = "budget_settled"
# The only event types budget restore and incremental metrics consume; reading
# a run's whole trace to find them is what makes those callers unbounded.
BUDGET_EVENT_TYPES = (
    LLM_USAGE_EVENT,
    BUDGET_REJECTED_EVENT,
    BUDGET_RESERVED_EVENT,
    BUDGET_SETTLED_EVENT,
)
_PROTECTED_TASKS = frozenset({"m2_report_claim_plan"})
_LOGICAL_CALL_ID: ContextVar[str | None] = ContextVar("logical_llm_call_id", default=None)


@contextmanager
def logical_llm_call(call_id: str) -> Iterator[None]:
    """Correlate a durable logical operation with its physical provider attempt."""
    token = _LOGICAL_CALL_ID.set(call_id)
    try:
        yield
    finally:
        _LOGICAL_CALL_ID.reset(token)


def _correlated_summary(summary: dict[str, Any]) -> dict[str, Any]:
    logical_call_id = _LOGICAL_CALL_ID.get()
    if logical_call_id is not None:
        summary = {**summary, "logical_call_id": logical_call_id}
    return summary


class LedgerLLMClient:
    """Wrap any LLMClient so every call lands in the run's spend ledger.

    Sibling of :class:`~eda_platform.core.dev_log.InstrumentedLLMClient`, which
    captures the same calls for the developer log; this one is the accounting
    record the run's metrics are built from.
    """

    def __init__(
        self,
        inner: LLMClient,
        *,
        session_id: str,
        emit: Callable[[TraceEvent], None],
        budget: SessionBudgetState | None = None,
    ) -> None:
        self._inner = inner
        self._session_id = session_id
        self._emit = emit
        self._budget = None if is_offline_client(inner) else budget

    @property
    def inner(self) -> LLMClient:
        """Expose the wrapped client so capability checks can unwrap decorators."""
        return self._inner

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def settings(self) -> Any:  # reporting reads .settings for completion caps
        return getattr(self._inner, "settings", None)

    def last_usage(self) -> LLMResultMetadata | None:
        return self._inner.last_usage()

    def preflight_structured(
        self, *, task: str, schema: type[BaseModel], payload: dict[str, Any]
    ) -> None:
        self._preflight(task=task, payload=payload, schema=schema)

    def preflight_tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> None:
        self._preflight(
            task=task,
            payload={"messages": messages, "tools": tools},
            schema=None,
        )

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        return self._call("structured", task=task, payload=payload, schema=schema)

    def text(self, *, task: str, payload: dict) -> str:
        return self._call("text", task=task, payload=payload, schema=None)

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        # Ledger payload intentionally records only the bounded agent transcript
        # shape and tool names; provider debug capture retains its own clipped
        # preview without turning every call into an unbounded trace row.
        return self._call(
            "tool_call",
            task=task,
            payload={"messages": messages, "tools": tools},
            schema=None,
        )

    def _preflight(self, *, task: str, payload: dict, schema: type | None) -> None:
        """Check exact reservation capacity without consuming a request slot."""
        if self._budget is None:
            return
        input_tokens = _conservative_input_token_estimate(task, payload, schema)
        output_tokens = _configured_output_token_cap(self.settings)
        output_bound_required = any(
            limit is not None
            for limit in (
                self._budget.policy.max_output_tokens,
                self._budget.policy.max_total_tokens,
                self._budget.policy.max_cost_usd,
            )
        )
        preflight_id = f"preflight:{self._session_id}:{uuid4().hex}:{task}"
        if output_bound_required and output_tokens <= 0:
            raise BudgetUsageUncertain(
                preflight_id,
                stage="reservation",
                missing=("output_tokens",),
            )
        cost = _worst_case_cost(self.settings, input_tokens, output_tokens)
        reservation = self._budget.reserve(
            preflight_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            protected=task in _PROTECTED_TASKS,
        )
        self._budget.release(reservation.call_id)

    def _call(self, kind: str, *, task: str, payload: dict, schema: type | None) -> Any:
        # Concurrent calls are safe: reservations/settlements are lock-guarded
        # in SessionBudgetState, the ledger store append is atomic, and adapter
        # last_usage() is thread-local — so the network wait runs unlocked.
        return self._call_serialized(
            kind,
            task=task,
            payload=payload,
            schema=schema,
        )

    def _call_serialized(
        self,
        kind: str,
        *,
        task: str,
        payload: dict,
        schema: type | None,
    ) -> Any:
        started_at = datetime.now(UTC)
        call_id = self._next_call_id(task)
        reservation = self._reserve(
            call_id=call_id,
            task=task,
            payload=payload,
            schema=schema,
            started_at=started_at,
        )
        # Providers only refresh last_usage() on success, so a failed call would
        # otherwise be billed the previous call's tokens.
        usage_before = self._safe_usage()
        status = "success"
        unserved = False
        try:
            if kind == "structured":
                assert schema is not None
                return self._inner.structured(task=task, schema=schema, payload=payload)
            if kind == "tool_call":
                return cast(Any, self._inner).tool_call(
                    task=task,
                    messages=list(payload.get("messages", [])),
                    tools=list(payload.get("tools", [])),
                )
            return self._inner.text(task=task, payload=payload)
        except ProviderUnavailableError as exc:
            status = type(exc).__name__
            unserved = True
            raise
        except Exception as exc:
            status = type(exc).__name__
            raise
        finally:
            usage = self._fresh_usage(status=status, usage_before=usage_before)
            settlement_error: Exception | None = None
            settled_reservation: BudgetReservation | None = None
            try:
                settled_reservation = self._settle(
                    reservation, usage=usage, unserved=unserved
                )
            except Exception as exc:  # budget terminal state must remain visible
                settlement_error = exc
                if self._budget is not None and reservation is not None:
                    settled_reservation = self._budget.reservation(reservation.call_id)
            finally:
                self._record(
                    kind=kind,
                    task=task,
                    call_id=call_id,
                    status=status if settlement_error is None else type(settlement_error).__name__,
                    started_at=started_at,
                    usage=usage,
                    reservation=reservation,
                    unserved=unserved,
                )
                self._record_budget_terminal(task, settled_reservation)
            if settlement_error is not None:
                raise settlement_error

    def _next_call_id(self, task: str) -> str:
        # A random physical-attempt component cannot collide after process
        # restart. Durable loops keep their separate stable logical call ID.
        return f"{self._session_id}:{uuid4().hex}:{task}"

    def _reserve(
        self,
        *,
        call_id: str,
        task: str,
        payload: dict,
        schema: type | None,
        started_at: datetime,
    ) -> BudgetReservation | None:
        if self._budget is None:
            return None
        reservation: BudgetReservation | None = None
        try:
            input_tokens = _conservative_input_token_estimate(task, payload, schema)
            output_tokens = _configured_output_token_cap(self.settings)
            output_bound_required = any(
                limit is not None
                for limit in (
                    self._budget.policy.max_output_tokens,
                    self._budget.policy.max_total_tokens,
                    self._budget.policy.max_cost_usd,
                )
            )
            if output_bound_required and output_tokens <= 0:
                raise BudgetUsageUncertain(
                    call_id,
                    stage="reservation",
                    missing=("output_tokens",),
                )
            cost = _worst_case_cost(self.settings, input_tokens, output_tokens)
            reservation = self._budget.reserve(
                call_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                protected=task in _PROTECTED_TASKS,
            )
            summary = _reservation_summary(reservation)
            summary["policy_fingerprint"] = _budget_policy_fingerprint(
                self._budget.policy
            )
            self._emit_required(
                TraceEvent(
                    session_id=self._session_id,
                    event_type=BUDGET_RESERVED_EVENT,
                    name=task,
                    call_id=call_id,
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    summary=_correlated_summary(summary),
                ),
                stage="reservation",
                missing=("budget_reserved",),
            )
            return reservation
        except Exception as exc:
            if reservation is not None:
                current = self._budget.reservation(reservation.call_id)
                if current is not None and current.status == "reserved":
                    self._budget.release(reservation.call_id)
            self._emit_safely(
                TraceEvent(
                    session_id=self._session_id,
                    event_type=BUDGET_REJECTED_EVENT,
                    name=task,
                    call_id=call_id,
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    summary=_correlated_summary(
                        {
                            "status": type(exc).__name__,
                            "reason": str(exc)[:500],
                        }
                    ),
                )
            )
            raise

    def _safe_usage(self) -> LLMResultMetadata | None:
        try:
            return self._inner.last_usage()
        except Exception:  # noqa: BLE001 — metering must never break the call
            return None

    def _fresh_usage(
        self,
        *,
        status: str,
        usage_before: LLMResultMetadata | None,
    ) -> LLMResultMetadata | None:
        usage = self._safe_usage()
        if status != "success" and usage == usage_before:
            return None
        return usage

    def _settle(
        self,
        reservation: BudgetReservation | None,
        *,
        usage: LLMResultMetadata | None,
        unserved: bool = False,
    ) -> BudgetReservation | None:
        if self._budget is None or reservation is None:
            return None
        if unserved:
            # The provider stated it never processed the request, so there is
            # nothing to bill and the request slot goes back (seed-8: six 503s
            # each consumed a full 12k-token reservation for zero output).
            self._budget.release(reservation.call_id)
            return None
        if usage is None or not usage.usage_reported:
            return self._budget.mark_uncertain(reservation.call_id)
        actual_cost: Decimal | float | None = usage.estimated_cost_usd
        if actual_cost is None:
            actual_cost = _worst_case_cost(
                self.settings,
                usage.usage.prompt_tokens,
                usage.usage.completion_tokens,
            )
        try:
            return self._budget.settle(
                reservation.call_id,
                input_tokens=usage.usage.prompt_tokens,
                output_tokens=usage.usage.completion_tokens,
                total_tokens=usage.usage.total_tokens,
                cost_usd=actual_cost,
            )
        except BudgetUsageUncertain:
            # The provider may already have charged the call. Convert the open
            # reservation to an explicit terminal uncertainty before surfacing.
            self._budget.mark_uncertain(reservation.call_id)
            raise

    def _record_budget_terminal(
        self,
        task: str,
        reservation: BudgetReservation | None,
    ) -> None:
        if reservation is None:
            return
        summary = _reservation_summary(reservation)
        if self._budget is not None:
            summary["policy_fingerprint"] = _budget_policy_fingerprint(
                self._budget.policy
            )
        self._emit_required(
            TraceEvent(
                session_id=self._session_id,
                event_type=BUDGET_SETTLED_EVENT,
                name=task,
                call_id=reservation.call_id,
                finished_at=datetime.now(UTC),
                summary=_correlated_summary(summary),
            ),
            stage="settlement",
            missing=("budget_settled",),
        )

    def _record(
        self,
        *,
        kind: str,
        task: str,
        call_id: str,
        status: str,
        started_at: datetime,
        usage: LLMResultMetadata | None,
        reservation: BudgetReservation | None,
        unserved: bool = False,
    ) -> None:
        summary: dict[str, Any] = {
            "task": task,
            "kind": kind,
            "transport_kind": _transport_kind(self.settings, kind),
            "status": status,
            "usage_known": usage is not None and usage.usage_reported,
        }
        if usage is not None:
            estimated_cost = usage.estimated_cost_usd
            # Only worth guessing when the provider actually reported tokens;
            # against a silent provider the inputs are zeros, so a worst case
            # here would publish $0 as if it were measured.
            if estimated_cost is None and usage.usage_reported:
                estimated_cost = _worst_case_cost(
                    self.settings,
                    usage.usage.prompt_tokens,
                    usage.usage.completion_tokens,
                )
            summary.update(
                {
                    "provider": usage.provider,
                    "model": usage.model,
                    "prompt_tokens": usage.usage.prompt_tokens,
                    "completion_tokens": usage.usage.completion_tokens,
                    "total_tokens": usage.usage.total_tokens,
                    "cached_tokens": usage.usage.cached_tokens,
                    "cache_creation_tokens": usage.usage.cache_creation_tokens,
                    "reasoning_tokens": usage.usage.reasoning_tokens,
                    "estimated_cost_usd": (
                        None if estimated_cost is None else float(estimated_cost)
                    ),
                    "cost_basis": usage.cost_basis,
                    "pricing_version": usage.pricing_version,
                    "provider_usage_reported": usage.usage_reported,
                    "request_id": usage.request_id,
                    "response_id": usage.response_id,
                    "finish_reason": usage.finish_reason,
                    "endpoint_host": usage.endpoint_host,
                    "request_bytes": usage.request_bytes,
                    "response_bytes": usage.response_bytes,
                }
            )
        elif unserved:
            # The provider refused to process the request, so the reservation
            # was released rather than consumed; recording it would inflate the
            # run's spend with tokens that were never generated.
            summary.update(
                {
                    "provider": "",
                    "model": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                }
            )
        else:
            # A call that produced no fresh usage marker still costs a call slot;
            # when a hard budget is active its reservation is consumed
            # conservatively and exposed so accounting can reconcile.
            summary.update(
                {
                    "provider": "",
                    "model": "",
                    "prompt_tokens": reservation.input_tokens if reservation else 0,
                    "completion_tokens": reservation.output_tokens if reservation else 0,
                    "total_tokens": reservation.total_tokens if reservation else 0,
                    "estimated_cost_usd": (
                        float(reservation.cost_usd) if reservation is not None else None
                    ),
                }
            )
        self._emit_required(
            TraceEvent(
                session_id=self._session_id,
                event_type=LLM_USAGE_EVENT,
                name=task,
                call_id=call_id,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                summary=_correlated_summary(summary),
            ),
            stage="settlement",
            missing=("llm_usage",),
        )

    def _emit_required(
        self,
        event: TraceEvent,
        *,
        stage: Literal["reservation", "settlement"],
        missing: tuple[str, ...],
    ) -> None:
        try:
            self._emit(event)
        except Exception as exc:
            raise BudgetUsageUncertain(
                event.call_id or "unidentified_call",
                stage=stage,
                missing=missing,
            ) from exc

    def _emit_safely(self, event: TraceEvent) -> None:
        try:
            self._emit(event)
        except Exception:  # noqa: BLE001 — observability must not break provider calls
            pass


def meter_llm_client(
    llm: LLMClient,
    *,
    session_id: str,
    emit: Callable[[TraceEvent], None],
    budget: SessionBudgetState | None = None,
    session_dir: Path | None = None,
) -> LedgerLLMClient:
    """Attach one ledger to a run without stacking duplicate ledger decorators.

    ``session_dir`` also installs the developer-log wrapper. Nothing else in
    production constructed one, so ``llm_debug.jsonl`` — which
    ``GET /sessions/{id}/llm-debug`` reads — was always empty outside tests.
    """
    if isinstance(llm, LedgerLLMClient):
        if llm.session_id == session_id and (budget is None or llm._budget is budget):
            return llm
        llm = llm.inner
    if session_dir is not None and not _has_instrumented_wrapper(llm):
        from eda_platform.core.dev_log import InstrumentedLLMClient

        llm = InstrumentedLLMClient(llm, session_dir=session_dir)
    return LedgerLLMClient(llm, session_id=session_id, emit=emit, budget=budget)


def _has_instrumented_wrapper(llm: LLMClient) -> bool:
    from eda_platform.core.dev_log import InstrumentedLLMClient

    current: object | None = llm
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, InstrumentedLLMClient):
            return True
        seen.add(id(current))
        current = getattr(current, "inner", None)
    return False


def restore_run_budget_state(
    policy: SessionBudgetPolicy,
    events: list[TraceEvent],
    *,
    run_started_at: datetime | None = None,
    accepted_policy_fingerprints: frozenset[str] | None = None,
) -> SessionBudgetState:
    """Rebuild consumed budget and completed call IDs from persisted events.

    Pass ``run_started_at`` when ``events`` is a budget-only slice: the wall
    clock must still run from the run's first event, otherwise a filtered
    caller under-reports elapsed time and loosens the time budget.
    """
    started_candidates = [event.started_at for event in events]
    if run_started_at is not None:
        started_candidates.append(run_started_at)
    elapsed = 0.0
    if started_candidates:
        elapsed = max(
            0.0,
            (datetime.now(UTC) - min(started_candidates)).total_seconds(),
        )
    state = SessionBudgetState(policy, started_at=monotonic() - elapsed)
    if _policy_has_hard_limits(policy) and any(
        event.event_type == LLM_USAGE_EVENT and not event.call_id for event in events
    ):
        raise BudgetUsageUncertain(
            "legacy_call",
            stage="restore",
            missing=("call_id", "budget_reserved"),
        )
    reserved_events = _unique_call_events(
        events,
        event_type=BUDGET_RESERVED_EVENT,
    )
    terminal_events = [
        event
        for event in events
        if event.event_type == BUDGET_SETTLED_EVENT and event.call_id
    ]
    terminal_by_call = _unique_call_events(
        terminal_events,
        event_type=BUDGET_SETTLED_EVENT,
    )
    ledger_events = [
        event
        for event in events
        if event.event_type == LLM_USAGE_EVENT and event.call_id
    ]
    _unique_call_events(ledger_events, event_type=LLM_USAGE_EVENT)
    ledger_call_ids: set[str] = {
        event.call_id for event in ledger_events if event.call_id is not None
    }
    unbudgeted = ledger_call_ids - set(reserved_events)
    if unbudgeted:
        raise BudgetUsageUncertain(
            sorted(unbudgeted)[0],
            stage="restore",
            missing=("budget_reserved",),
        )
    expected_policy_fingerprint = _budget_policy_fingerprint(policy)
    accepted_fingerprints = accepted_policy_fingerprints or frozenset(
        {expected_policy_fingerprint}
    )
    if expected_policy_fingerprint not in accepted_fingerprints:
        raise ValueError("accepted policy fingerprints must include the effective policy.")
    for call_id, reserved in reserved_events.items():
        assert call_id is not None
        persisted_policy_fingerprint = reserved.summary.get("policy_fingerprint")
        if (
            persisted_policy_fingerprint is not None
            and persisted_policy_fingerprint not in accepted_fingerprints
        ):
            raise BudgetUsageUncertain(
                call_id,
                stage="restore",
                missing=("matching_budget_policy",),
            )
        terminal = terminal_by_call.get(call_id)
        if (
            terminal is not None
            and terminal.summary.get("policy_fingerprint") is not None
            and terminal.summary.get("policy_fingerprint")
            not in accepted_fingerprints
        ):
            raise BudgetUsageUncertain(
                call_id,
                stage="restore",
                missing=("matching_budget_policy",),
            )
        if terminal is None and call_id not in ledger_call_ids:
            # Crash after durable reservation but before a durable provider
            # outcome: whether the request crossed the network is unknowable.
            terminal = None
        if reserved is None:
            raise BudgetUsageUncertain(
                call_id,
                stage="restore",
                missing=("budget_reserved",),
            )
        state.reserve(
            call_id,
            input_tokens=_summary_int(reserved.summary, "input_tokens"),
            output_tokens=_summary_int(reserved.summary, "output_tokens"),
            total_tokens=_summary_int(reserved.summary, "total_tokens"),
            cost_usd=_summary_cost(reserved.summary),
            protected=bool(reserved.summary.get("protected")),
        )
        if terminal is None:
            state.mark_uncertain(call_id)
            continue
        if (
            terminal.summary.get("status") == "uncertain"
            or terminal.summary.get("usage_known") is not True
        ):
            state.mark_uncertain(call_id)
            continue
        state.settle(
            call_id,
            input_tokens=_summary_int(terminal.summary, "input_tokens"),
            output_tokens=_summary_int(terminal.summary, "output_tokens"),
            total_tokens=_summary_int(terminal.summary, "total_tokens"),
            cost_usd=_summary_cost(terminal.summary),
        )
    return state


def budget_policy_fingerprint(policy: SessionBudgetPolicy) -> str:
    """Public stable identity used to restore a trusted amendment history."""
    return _budget_policy_fingerprint(policy)


def _unique_call_events(
    events: list[TraceEvent],
    *,
    event_type: str,
) -> dict[str, TraceEvent]:
    indexed: dict[str, TraceEvent] = {}
    for event in events:
        if event.event_type != event_type or event.call_id is None:
            continue
        if event.call_id in indexed:
            raise BudgetUsageUncertain(
                event.call_id,
                stage="restore",
                missing=(f"unique_{event_type}",),
            )
        indexed[event.call_id] = event
    return indexed


def _conservative_input_token_estimate(
    task: str,
    payload: dict,
    schema: type | None,
) -> int:
    """Return a tokenizer-free upper estimate based on UTF-8 bytes."""
    envelope: dict[str, Any] = {"task": task, "payload": payload}
    if schema is not None and issubclass(schema, BaseModel):
        envelope["schema"] = schema.model_json_schema()
    encoded = json.dumps(envelope, ensure_ascii=False, default=str).encode("utf-8")
    return max(1, len(encoded))


def _configured_output_token_cap(settings: Any) -> int:
    value = getattr(settings, "max_tokens", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _transport_kind(settings: Any, requested_kind: str) -> str:
    """What actually went over the wire. Anthropic's structured() rides a forced
    `emit_result` tool call, so it is not the same transport as an OpenAI
    response_format request even though both are logically "structured"."""
    provider = getattr(settings, "provider", None)
    provider_value = getattr(provider, "value", provider)
    if requested_kind == "structured" and provider_value == "anthropic":
        return "provider_tool"
    return requested_kind


def _worst_case_cost(
    settings: Any,
    input_tokens: int,
    output_tokens: int,
) -> Decimal | None:
    if settings is None:
        return None
    prompt_rate = float(getattr(settings, "usd_per_1k_prompt", 0.0) or 0.0)
    completion_rate = float(getattr(settings, "usd_per_1k_completion", 0.0) or 0.0)
    if prompt_rate <= 0 and completion_rate <= 0:
        provider = getattr(settings, "provider", None)
        model = str(getattr(settings, "model", ""))
        rates = pricing_per_1m(provider, model) if provider is not None else None
        if rates is None:
            return None
        prompt_rate, completion_rate = rates[0] / 1000, rates[1] / 1000
    value = input_tokens / 1000 * prompt_rate + output_tokens / 1000 * completion_rate
    return Decimal(str(round(value, 9)))


def _reservation_summary(reservation: BudgetReservation) -> dict[str, Any]:
    return {
        "status": reservation.status,
        "input_tokens": reservation.input_tokens,
        "output_tokens": reservation.output_tokens,
        "total_tokens": reservation.total_tokens,
        "estimated_cost_usd": float(reservation.cost_usd),
        "protected": reservation.protected,
        "usage_known": reservation.usage_known,
    }


def _budget_policy_fingerprint(policy: SessionBudgetPolicy) -> str:
    return stable_hash(
        {
            "schema_version": 1,
            "max_requests": policy.max_requests,
            "max_input_tokens": policy.max_input_tokens,
            "max_output_tokens": policy.max_output_tokens,
            "max_total_tokens": policy.max_total_tokens,
            "max_cost_usd": (
                None if policy.max_cost_usd is None else str(policy.max_cost_usd)
            ),
            "max_wall_seconds": policy.max_wall_seconds,
            "protected_requests": policy.protected_requests,
            "protected_input_tokens": policy.protected_input_tokens,
            "protected_output_tokens": policy.protected_output_tokens,
            "protected_total_tokens": policy.protected_total_tokens,
            "protected_cost_usd": str(policy.protected_cost_usd),
            "unknown_usage_policy": policy.unknown_usage_policy,
        },
        length=64,
    )


def _summary_int(summary: dict[str, Any], name: str) -> int:
    value = summary.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BudgetUsageUncertain(
            "persisted_budget",
            stage="restore",
            missing=(name,),
        )
    return value


def _summary_cost(summary: dict[str, Any]) -> float:
    value = summary.get("estimated_cost_usd")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise BudgetUsageUncertain(
            "persisted_budget",
            stage="restore",
            missing=("estimated_cost_usd",),
        )
    return float(value)


def _policy_has_hard_limits(policy: SessionBudgetPolicy) -> bool:
    return any(
        value is not None
        for value in (
            policy.max_requests,
            policy.max_input_tokens,
            policy.max_output_tokens,
            policy.max_total_tokens,
            policy.max_cost_usd,
            policy.max_wall_seconds,
        )
    )
