"""A busy provider must not bill a run or end it (seed-8 regression).

DeepSeek answered six concurrent probes with HTTP 503 "Service is too busy".
Each was classified as an unknown outcome, so each consumed its full 12k-token
reservation ($0.037 for zero output), and the first one ended the whole
exploration as `failed` with no report. A 503/429/502/504 is the provider
stating it did NOT serve the request: retrying carries no duplicate-generation
risk, and an exhausted retry is a rejection, not an unknown.
"""

from __future__ import annotations

from typing import Any, cast
from urllib import error

import pytest

from eda_platform.core.budget import SessionBudgetPolicy, SessionBudgetState
from eda_platform.core.llm import (
    LLMProvider,
    LLMSettings,
    OpenAICompatibleLLMClient,
    ProviderUnavailableError,
)
from eda_platform.core.llm_ledger import LLM_USAGE_EVENT, LedgerLLMClient
from eda_platform.schemas.sessions import TraceEvent

_BUSY_BODY = (
    b'{"error":{"message":"Service is too busy.","type":"service_unavailable_error"}}'
)


def _settings() -> LLMSettings:
    return LLMSettings(
        provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key="k", timeout_seconds=1
    )


class _Http:
    """Raises the given statuses in order, then returns a canned success."""

    def __init__(self, statuses: list[int]) -> None:
        self.statuses = list(statuses)
        self.attempts = 0

    def __call__(self, req: Any, timeout: float | None = None) -> Any:
        self.attempts += 1
        if self.statuses:
            code = self.statuses.pop(0)
            raise error.HTTPError(
                "http://x", code, "busy", {}, cast(Any, _BodyStream(_BUSY_BODY))
            )
        return _Response(b'{"choices":[{"message":{"content":"ok"}}]}')


class _BodyStream:
    """HTTPError hands this to a temporary-file closer, which calls close()."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        return None


class _Response:
    def __init__(self, raw: bytes) -> None:
        self._raw = raw
        self.headers: dict[str, str] = {}

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def test_a_busy_provider_is_retried_and_then_succeeds(monkeypatch: Any) -> None:
    http = _Http([503, 429])
    monkeypatch.setattr("eda_platform.core.llm.request.urlopen", http)
    monkeypatch.setattr("eda_platform.core.llm.time.sleep", lambda _s: None)
    client = OpenAICompatibleLLMClient(_settings())

    assert client.text(task="probe", payload={}) == "ok"
    assert http.attempts == 3, "503/429 must be retried, not surfaced immediately"


def test_a_persistently_busy_provider_raises_a_typed_unavailable_error(
    monkeypatch: Any,
) -> None:
    http = _Http([503] * 10)
    monkeypatch.setattr("eda_platform.core.llm.request.urlopen", http)
    monkeypatch.setattr("eda_platform.core.llm.time.sleep", lambda _s: None)
    client = OpenAICompatibleLLMClient(_settings())

    with pytest.raises(ProviderUnavailableError) as raised:
        client.text(task="probe", payload={})
    assert "503" in str(raised.value)


def test_a_served_client_error_is_not_retried(monkeypatch: Any) -> None:
    """Control: a 400 is a served answer about the request itself."""
    http = _Http([400])
    monkeypatch.setattr("eda_platform.core.llm.request.urlopen", http)
    monkeypatch.setattr("eda_platform.core.llm.time.sleep", lambda _s: None)
    client = OpenAICompatibleLLMClient(_settings())

    with pytest.raises(RuntimeError) as raised:
        client.text(task="probe", payload={})
    assert not isinstance(raised.value, ProviderUnavailableError)
    assert http.attempts == 1


class _UnavailableLLM:
    settings = _settings()

    def last_usage(self) -> None:
        return None

    def text(self, *, task: str, payload: dict) -> str:
        raise ProviderUnavailableError("LLM provider is unavailable: HTTP 503")


def test_an_unavailable_provider_bills_nothing() -> None:
    """The seed-8 cost bug: an unknown outcome consumes its whole reservation,
    but a provider that says it did not serve the request owes no spend."""
    events: list[TraceEvent] = []
    budget = SessionBudgetState(SessionBudgetPolicy(max_requests=4))
    client = LedgerLLMClient(
        cast(Any, _UnavailableLLM()),
        session_id="run_busy",
        emit=events.append,
        budget=budget,
    )

    with pytest.raises(ProviderUnavailableError):
        client.text(task="probe", payload={})

    assert budget.total_tokens_used == 0
    assert budget.cost_usd_used == 0
    usage = [event for event in events if event.event_type == LLM_USAGE_EVENT]
    assert [event.summary["total_tokens"] for event in usage] == [0]
    # The request slot is released too: an unserved call must not eat the cap.
    assert budget.requests_used == 0


# --- the run must survive it ------------------------------------------------


def test_provider_unavailable_is_a_probe_local_outcome() -> None:
    """Seed-8: the first 503 raised SupervisorInvariantError and the whole run
    ended `failed` with no report, discarding two committed receipts. A busy
    provider is an external condition, like a truncated response."""
    from eda_platform.agents.exploration.workflow import (
        PROBE_LOCAL_ERROR_CODES,
        _probe_outcome_is_usable,
    )
    from eda_platform.agents.exploration.executor import ProbeExecutionResult

    assert "provider_unavailable" in PROBE_LOCAL_ERROR_CODES
    assert _probe_outcome_is_usable(
        ProbeExecutionResult(
            status="failed",
            error="HTTP 503",
            error_code="provider_unavailable",
        )
    )
    # Control: an integrity fault still ends the run.
    assert not _probe_outcome_is_usable(
        ProbeExecutionResult(
            status="failed", error="digest mismatch", error_code="digest_mismatch"
        )
    )
