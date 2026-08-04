"""Concurrent provider calls through the ledger client (speedup plan P4-prep).

The ledger's call lock existed only because adapter ``last_usage()`` was
client-wide mutable state; with thread-local usage the network call must run
unlocked, and each thread must settle its own numbers.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from eda_platform.core.budget import SessionBudgetPolicy, SessionBudgetState
from eda_platform.core.llm import LLMResultMetadata, LLMUsage
from eda_platform.core.llm_ledger import LedgerLLMClient


class _SlowThreadLocalLLM:
    """Fake provider: each call sleeps, then records its own thread's usage."""

    settings = None

    def __init__(self) -> None:
        self._local = threading.local()

    def last_usage(self) -> LLMResultMetadata | None:
        return getattr(self._local, "value", None)

    def text(self, *, task: str, payload: dict) -> str:
        tokens = int(payload["tokens"])
        time.sleep(0.25)
        self._local.value = LLMResultMetadata(
            provider="fake",
            model="fake-1",
            usage=LLMUsage(
                prompt_tokens=tokens, completion_tokens=tokens, total_tokens=2 * tokens
            ),
            estimated_cost_usd=0.0,
            usage_reported=True,
        )
        return f"done-{tokens}"


def test_two_provider_calls_overlap_and_settle_their_own_usage() -> None:
    events: list[Any] = []
    budget = SessionBudgetState(SessionBudgetPolicy(max_requests=4))
    client = LedgerLLMClient(
        cast(Any, _SlowThreadLocalLLM()),
        session_id="s-conc",
        emit=events.append,
        budget=budget,
    )

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda tokens: client.text(task="probe", payload={"tokens": tokens}),
                (100, 200),
            )
        )
    elapsed = time.perf_counter() - started

    assert sorted(results) == ["done-100", "done-200"]
    # Two 0.25s calls must overlap; the old client-wide call lock serialized
    # them to ~0.5s.
    assert elapsed < 0.45, f"provider calls did not overlap: {elapsed:.3f}s"

    usages = [
        event.summary for event in events if event.event_type == "llm_usage"
    ]
    assert sorted(usage["prompt_tokens"] for usage in usages) == [100, 200]
    # Each settlement carries its own thread's numbers, not the other call's.
    assert sorted(usage["total_tokens"] for usage in usages) == [200, 400]
