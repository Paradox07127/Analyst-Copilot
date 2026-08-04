"""Shadow budget runtime under concurrency: one-winner reservation races and
atomic ledger-file appends (workstream P3)."""

from __future__ import annotations

import os as real_os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from eda_platform.core.budget import SessionBudgetExceeded
from eda_platform.core.exploration_journal import JsonlExplorationJournal, sealed_policy
from eda_platform.core.llm_ledger import (
    BUDGET_REJECTED_EVENT,
    BUDGET_RESERVED_EVENT,
    restore_run_budget_state,
)
from eda_platform.drivers import exploration as exploration_driver
from eda_platform.drivers.exploration import (
    JsonlShadowBudgetStore,
    build_shadow_budget_runtime,
)
from eda_platform.schemas.exploration import ExplorationPolicy, InsightFamily
from eda_platform.schemas.exploration_budget import (
    ExplorationBudgetPolicy,
    SessionBudgetPolicyModel,
)
from eda_platform.schemas.sessions import TraceEvent


def _policy(max_requests: int) -> ExplorationPolicy:
    return sealed_policy(
        ExplorationPolicy.model_validate(
            {
                "mode": "open",
                "goal": "budget race goal",
                "dataset_scope": ("ds_a",),
                "thinking_level": "standard",
                "coverage_targets": (InsightFamily.DESCRIPTIVE,),
                "budget": ExplorationBudgetPolicy.model_validate(
                    {
                        "llm": SessionBudgetPolicyModel(max_requests=max_requests),
                        "max_successful_tool_calls": 4,
                        "max_tool_calls_by_kind": {"run_open_analysis": 4},
                        "idle_timeout_seconds": 30.0,
                        "max_rounds": 3,
                    }
                ),
                "scoring_policy_version": "score-v1",
                "statistical_policy_version": "stats-v1",
                "tool_capability_digest": "tools-v1",
            }
        )
    )


class _BlockingFakeProvider:
    """Online-looking provider; usage stays unreported so settlement is
    conservative (the reservation is fully consumed)."""

    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def structured(self, *, task: str, schema: type, payload: dict) -> Any:
        raise NotImplementedError

    def text(self, *, task: str, payload: dict) -> str:
        with self._lock:
            self.calls += 1
        return "ok"

    def last_usage(self) -> None:
        return None


def _runtime(tmp_path: Path, *, max_requests: int):
    policy = _policy(max_requests)
    journal = JsonlExplorationJournal(tmp_path / "run" / "journal.jsonl")
    journal.initialize(
        exploration_id="xpl_budget_race",
        policy=policy,
        code_fingerprint="code-v1",
        data_state_witness="dsw1_test",
    )
    store = JsonlShadowBudgetStore(tmp_path / "run" / "llm-budget.jsonl")
    return build_shadow_budget_runtime(
        exploration_id="xpl_budget_race",
        provider=_BlockingFakeProvider(),
        policy=policy,
        journal=journal,
        event_store=store,
    ), store, policy


def test_two_threads_racing_one_remaining_request_admit_exactly_one(
    tmp_path: Path,
) -> None:
    runtime, store, policy = _runtime(tmp_path, max_requests=1)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    outcome_lock = threading.Lock()

    def attempt() -> None:
        barrier.wait()
        try:
            runtime.provider.text(task="probe", payload={"q": "x"})
        except SessionBudgetExceeded:
            with outcome_lock:
                outcomes.append("rejected")
        else:
            with outcome_lock:
                outcomes.append("success")

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["rejected", "success"]

    events = store.events()
    reserved = [e for e in events if e.event_type == BUDGET_RESERVED_EVENT]
    rejected = [e for e in events if e.event_type == BUDGET_REJECTED_EVENT]
    assert len(reserved) == 1
    assert len(rejected) == 1

    # Ledger replay agrees: exactly one request consumed, cap reached.
    restored = restore_run_budget_state(policy.budget.llm.to_policy(), events)
    assert restored.requests_used == 1
    with pytest.raises(SessionBudgetExceeded):
        restored.reserve("post-restore-call")


class _ByteAtATimeOs:
    """Proxy that forces os.write to land one byte per syscall with a thread
    yield in between, so unsynchronized concurrent appends interleave."""

    def __init__(self) -> None:
        self.write_calls = 0

    def write(self, descriptor: int, payload: bytes) -> int:
        self.write_calls += 1
        view = bytes(payload[:1])
        written = real_os.write(descriptor, view)
        # Force a context switch mid-record.
        time.sleep(0.0002)
        return written

    def __getattr__(self, name: str) -> Any:
        return getattr(real_os, name)


def test_concurrent_ledger_appends_do_not_interleave_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlShadowBudgetStore(tmp_path / "llm-budget.jsonl")
    monkeypatch.setattr(exploration_driver, "os", _ByteAtATimeOs())

    per_thread = 8
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def append_events(worker: str) -> None:
        barrier.wait()
        for index in range(per_thread):
            try:
                store.append(
                    TraceEvent(
                        session_id="xpl_budget_race",
                        event_type=BUDGET_RESERVED_EVENT,
                        name="probe",
                        call_id=f"{worker}-{index}",
                        started_at=datetime.now(UTC),
                        summary={"worker": worker, "index": index},
                    )
                )
            except Exception as exc:  # noqa: BLE001 — collected for assertion
                errors.append(exc)

    threads = [
        threading.Thread(target=append_events, args=(worker,))
        for worker in ("alpha", "beta")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    events = store.events()  # raises ValueError on an interleaved/corrupt record
    assert len(events) == 2 * per_thread
    assert sorted(e.call_id for e in events) == sorted(
        f"{worker}-{index}"
        for worker in ("alpha", "beta")
        for index in range(per_thread)
    )
