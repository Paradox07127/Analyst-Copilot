from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import TypeVar

import pytest
from pydantic import BaseModel

from eda_platform.core.cancellation import (
    CancellationContext,
    CancellationRequested,
    DurableCancellationRecord,
    StorageBackedCancellationToken,
    cancellation_scope,
)
from eda_platform.core.llm import (
    CancellableLLMClient,
    OfflineLLMClient,
    is_offline_client,
)

T = TypeVar("T", bound=BaseModel)


class Result(BaseModel):
    value: int


class BlockingClient:
    def __init__(self) -> None:
        self.entered = Event()
        self.release = Event()
        self.calls = 0

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls += 1
        self.entered.set()
        assert self.release.wait(timeout=3)
        return schema.model_validate({"value": 1})

    def text(self, *, task: str, payload: dict) -> str:
        self.calls += 1
        return "ok"

    def last_usage(self) -> None:
        return None


def test_cancellable_wrapper_preserves_offline_capability() -> None:
    client = CancellableLLMClient(OfflineLLMClient(), CancellationContext())

    assert is_offline_client(client)


def test_adapter_capable_provider_aborts_active_call_and_suppresses_result() -> None:
    cancellation = CancellationContext()
    provider = BlockingClient()
    client = CancellableLLMClient(
        provider,  # type: ignore[arg-type]
        cancellation,
        abort_active_call=provider.release.set,
    )

    def call_in_scope() -> Result:
        with cancellation_scope(cancellation):
            return client.structured(task="probe", schema=Result, payload={})

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(call_in_scope)
        assert provider.entered.wait(timeout=2)
        cancellation.request_cancel("stop provider")
        with pytest.raises(CancellationRequested, match="stop provider"):
            future.result(timeout=3)

    assert provider.calls == 1


def test_provider_without_abort_adapter_discards_result_after_call_returns() -> None:
    cancellation = CancellationContext()
    provider = BlockingClient()
    client = CancellableLLMClient(
        provider,  # type: ignore[arg-type]
        cancellation,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            client.structured,
            task="probe",
            schema=Result,
            payload={},
        )
        assert provider.entered.wait(timeout=2)
        cancellation.request_cancel("stop provider without adapter")
        assert not future.done()
        provider.release.set()
        with pytest.raises(
            CancellationRequested,
            match="stop provider without adapter",
        ):
            future.result(timeout=3)

    assert provider.calls == 1


def test_provider_call_never_starts_when_already_cancelled() -> None:
    cancellation = CancellationContext()
    cancellation.request_cancel("already stopped")
    provider = BlockingClient()
    client = CancellableLLMClient(provider, cancellation)  # type: ignore[arg-type]

    with pytest.raises(CancellationRequested, match="already stopped"):
        client.structured(task="probe", schema=Result, payload={})

    assert provider.calls == 0


def test_durable_cancel_flag_aborts_provider_mid_flight() -> None:
    durable_cancelled = False

    def read(_job_id: str) -> DurableCancellationRecord:
        return DurableCancellationRecord(
            job_id="job_llm",
            generation=3,
            owner="worker-a",
            cancel_requested=durable_cancelled,
            reason="durable provider stop",
        )

    cancellation = StorageBackedCancellationToken(
        job_id="job_llm",
        generation=3,
        owner="worker-a",
        reader=read,
    )
    provider = BlockingClient()
    client = CancellableLLMClient(
        provider,  # type: ignore[arg-type]
        cancellation,
        abort_active_call=provider.release.set,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            client.structured,
            task="probe",
            schema=Result,
            payload={},
        )
        assert provider.entered.wait(timeout=2)
        durable_cancelled = True
        with pytest.raises(CancellationRequested, match="durable provider stop"):
            future.result(timeout=3)
