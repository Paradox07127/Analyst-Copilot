"""Capability is settled before spend, and a refusal degrades rather than fails."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from eda_platform.core.llm import (
    LLMProvider,
    LLMSettings,
    LLMToolResponse,
    OfflineLLMClient,
    ToolCallingUnsupportedError,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.core.tool_calling_probe import (
    ToolCallingVerdict,
    forget_probe_results,
    tool_calling_readiness,
)
from eda_platform.drivers.chat import run_chat_turn

UNVERIFIED = LLMSettings(
    provider=LLMProvider.OPENAI_COMPATIBLE,
    base_url="http://localhost:8000/v1",
    model="my-finetune:latest",
)
VERIFIED = LLMSettings(
    provider=LLMProvider.DEEPSEEK,
    api_key="k",
    model="deepseek-v4-pro",
)


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    forget_probe_results()
    yield
    forget_probe_results()


class _Client:
    """Counts tool_call invocations so a probe cannot hide."""

    def __init__(self, settings: LLMSettings, *, raises: Exception | None = None) -> None:
        self.settings = settings
        self._raises = raises
        self.tool_calls = 0

    def tool_call(self, *, task: str, messages: list, tools: list) -> LLMToolResponse:
        self.tool_calls += 1
        if self._raises is not None:
            raise self._raises
        return LLMToolResponse(content="ok")

    def structured(self, *, task: str, schema: type, payload: dict) -> Any:
        raise RuntimeError("no structured route in this double")

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> None:
        return None


def test_a_verified_model_is_never_probed() -> None:
    client = _Client(VERIFIED)

    verdict = tool_calling_readiness(client)

    assert verdict == ToolCallingVerdict(
        True, "catalog", "deepseek-v4-pro is in the verified catalog."
    )
    assert client.tool_calls == 0, "the catalog already answered; probing would be pure spend"


def test_an_unverified_model_is_probed_once_and_accepted() -> None:
    client = _Client(UNVERIFIED)

    verdict = tool_calling_readiness(client)

    assert verdict.usable is True
    assert verdict.source == "probe"
    assert client.tool_calls == 1


def test_a_refused_tools_payload_is_a_verdict_not_a_crash() -> None:
    client = _Client(UNVERIFIED, raises=ToolCallingUnsupportedError("no tools here"))

    verdict = tool_calling_readiness(client)

    assert verdict.usable is False
    assert verdict.source == "probe"
    assert "no tools here" in verdict.detail


def test_the_verdict_is_reused_for_the_rest_of_the_process() -> None:
    first = _Client(UNVERIFIED)
    tool_calling_readiness(first)

    second = _Client(UNVERIFIED)
    verdict = tool_calling_readiness(second)

    assert verdict.usable is True
    assert verdict.source == "cached"
    assert second.tool_calls == 0, "one probe per model, not one per run"


def test_an_unrelated_failure_is_not_read_as_a_capability_verdict() -> None:
    """Treating any error as 'no tool calling' would turn a bad key into a
    permanently degraded analysis."""
    client = _Client(UNVERIFIED, raises=RuntimeError("HTTP 401: invalid api key"))

    with pytest.raises(RuntimeError, match="401"):
        tool_calling_readiness(client)


def test_offline_is_answered_without_touching_the_client() -> None:
    verdict = tool_calling_readiness(OfflineLLMClient())

    assert verdict.usable is False
    assert verdict.source == "offline"


def test_probing_can_be_declined_when_the_caller_will_not_pay_for_it() -> None:
    client = _Client(UNVERIFIED)

    verdict = tool_calling_readiness(client, allow_probe=False)

    assert verdict.source == "unprobed"
    assert client.tool_calls == 0


# --- driver-level degradation (scheme B) --------------------------------------


class _RefusesTools(_Client):
    """Refuses from the very first call, so the probe is what catches it."""

    def __init__(self) -> None:
        super().__init__(
            UNVERIFIED,
            raises=ToolCallingUnsupportedError("tools is not supported by this endpoint"),
        )


class _RefusesAfterTheProbe(_Client):
    """Accepts the probe, then refuses inside the loop.

    This is the case the probe cannot catch: a gateway that advertises tools
    and rejects them once a real schema is attached.
    """

    def __init__(self) -> None:
        super().__init__(UNVERIFIED)

    def tool_call(self, *, task: str, messages: list, tools: list) -> LLMToolResponse:
        self.tool_calls += 1
        if task == "tool_calling_probe":
            return LLMToolResponse(content="ok")
        raise ToolCallingUnsupportedError("tools rejected once a schema is attached")


def _run_help_turn(store: ArtifactStore, llm: object) -> Any:
    return run_chat_turn(
        "help",  # routes to meta_help deterministically on the legacy path
        datasets=[],
        project_id="project_demo",
        session_id="run_demo",
        llm=llm,
        store=store,
    )


def _store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "run_demo")
    return store


def _events(store: ArtifactStore) -> list:
    return store.list_trace_events(project_id="project_demo", session_id="run_demo")


def test_a_refusal_at_probe_time_keeps_the_turn_off_the_agent_route(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    result = _run_help_turn(store, _RefusesTools())

    assert result.intent.kind == "meta_help"
    probes = [event for event in _events(store) if event.event_type == "tool_calling_probe"]
    assert probes and probes[0].summary["usable"] is False


def test_a_refusal_inside_the_loop_degrades_the_turn_and_says_so(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    llm = _RefusesAfterTheProbe()

    result = _run_help_turn(store, llm)

    assert result.intent.kind == "meta_help"
    assert llm.tool_calls >= 2, "the probe passed, so the loop must have been entered"
    degraded = [
        event for event in _events(store) if event.event_type == "agent_route_degraded"
    ]
    assert degraded, "a silent fallback would hide a permanently worse analysis"
    assert "schema is attached" in degraded[0].summary["reason"]
