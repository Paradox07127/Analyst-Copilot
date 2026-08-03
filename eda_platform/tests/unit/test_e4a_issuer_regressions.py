"""Regression cover for I4: a run must not become permanently unauditable
just because one provider call reported no usage.

``_llm_usage`` used to hard-require ``usage_known`` on every ledger event,
raising if even one call was uncertain. But the budget ledger already has a
tolerant path for this (``llm_ledger._settle`` -> ``mark_uncertain`` ->
``restore_run_budget_state``): an uncertain call is billed the conservative
reservation the ledger consumed, not blocked. ``_llm_usage`` must use that
same reservation-derived number, because a call that succeeds without a
provider usage block still carries its own (zero-valued) ``LLMUsage`` in the
``llm_usage`` event -- summing that instead of the settled reservation would
silently disagree with ``restore_run_budget_state``'s totals.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pytest

from eda_platform.core.budget import SessionBudgetPolicy, SessionBudgetState
from eda_platform.core.llm import LLMResultMetadata, LLMUsage
from eda_platform.core.llm_ledger import (
    LLM_USAGE_EVENT,
    LedgerLLMClient,
    restore_run_budget_state,
)
from eda_platform.drivers.exploration_evidence_issuer import _llm_usage
from eda_platform.schemas.sessions import TraceEvent


class _MixedUsageLLM:
    """First call reports full usage; second succeeds without one."""

    def __init__(self) -> None:
        self._last: LLMResultMetadata | None = None

    def text(self, *, task: str, payload: dict) -> str:
        if task == "known":
            self._last = LLMResultMetadata(
                provider="fake",
                model="fake-1",
                usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                estimated_cost_usd=0.002,
            )
        else:
            self._last = LLMResultMetadata(
                provider="fake", model="fake-1", usage_reported=False
            )
        return "text"

    def last_usage(self) -> LLMResultMetadata | None:
        return self._last


def _run_two_calls() -> tuple[list[TraceEvent], SessionBudgetState]:
    events: list[TraceEvent] = []
    budget = SessionBudgetState(SessionBudgetPolicy())
    client = LedgerLLMClient(
        cast(Any, _MixedUsageLLM()),
        session_id="run_x",
        emit=events.append,
        budget=budget,
    )
    client.text(task="known", payload={"p": 1})
    client.text(task="unknown_usage", payload={"p": 2})
    return events, budget


def test_uncertain_llm_call_does_not_block_issuance_and_matches_restored_ledger() -> None:
    events, budget = _run_two_calls()

    # The uncertain call really is uncertain: prove the discriminating case
    # exists rather than assuming it (the LLM_USAGE_EVENT's own total_tokens
    # for that call is 0 -- a bare LLMUsage() -- while the budget ledger
    # billed it the non-zero conservative reservation).
    uncertain_usage_event = next(
        event
        for event in events
        if event.event_type == LLM_USAGE_EVENT and event.name == "unknown_usage"
    )
    assert uncertain_usage_event.summary["usage_known"] is False
    assert uncertain_usage_event.summary["total_tokens"] == 0
    assert budget.total_tokens_used > 15  # known call's 15 plus a real reservation

    provider, model, llm_requests, total_tokens, cost = _llm_usage(events)

    assert provider == "fake"
    assert model == "fake-1"
    assert llm_requests == 2
    assert total_tokens == budget.total_tokens_used
    assert cost >= 0.002

    restored = restore_run_budget_state(
        SessionBudgetPolicy(), events, run_started_at=datetime.now(UTC)
    )
    assert restored.requests_used == llm_requests
    assert restored.total_tokens_used == total_tokens


def test_usage_known_true_but_provider_usage_not_reported_is_still_rejected() -> None:
    """The uncertain-call carve-out must not weaken the certain-call contract:
    a call claiming usage_known must still carry provider_usage_reported."""
    event = TraceEvent(
        session_id="run_x",
        event_type=LLM_USAGE_EVENT,
        name="t",
        call_id="run_x:c1:t",
        summary={
            "usage_known": True,
            "provider_usage_reported": False,
            "provider": "fake",
            "model": "fake-1",
            "total_tokens": 10,
            "estimated_cost_usd": 0.01,
        },
    )

    with pytest.raises(ValueError, match="measured usage"):
        _llm_usage([event])
