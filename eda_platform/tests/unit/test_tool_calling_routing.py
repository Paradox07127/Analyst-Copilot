"""Routing never spends; the graph handles actual provider refusals durably."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from eda_platform.core.llm import (
    LLMProvider,
    LLMSettings,
    LLMToolResponse,
    OfflineLLMClient,
    StructuredLLM,
    ToolCallingUnsupportedError,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.core.tool_calling_routing import (
    ToolCallingVerdict,
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


@pytest.mark.parametrize(
    "failure", [None, ToolCallingUnsupportedError("no tools"), RuntimeError("401")]
)
def test_unknown_model_routing_never_sends_a_request(failure: Exception | None) -> None:
    client = _Client(UNVERIFIED, raises=failure)
    for _ in range(2):
        verdict = tool_calling_readiness(client)
        assert verdict == ToolCallingVerdict(True, "unprobed")
    assert client.tool_calls == 0


def test_distinct_endpoints_are_routed_without_shared_learned_state() -> None:
    first = _Client(UNVERIFIED, raises=ToolCallingUnsupportedError("no tools"))
    second = _Client(UNVERIFIED.model_copy(update={"base_url": "http://localhost:9000/v1"}))
    assert tool_calling_readiness(first) == tool_calling_readiness(second)
    assert first.tool_calls == second.tool_calls == 0


def test_offline_is_answered_without_touching_the_client() -> None:
    verdict = tool_calling_readiness(OfflineLLMClient())

    assert verdict.usable is False
    assert verdict.source == "offline"


class _RefusesTools(_Client):
    """Refuses the first real request inside the graph."""

    def __init__(self) -> None:
        super().__init__(
            UNVERIFIED,
            raises=ToolCallingUnsupportedError("tools is not supported by this endpoint"),
        )


def _run_help_turn(store: ArtifactStore, llm: StructuredLLM) -> Any:
    return run_chat_turn(
        "help",  # routes to meta_help through the deterministic fallback graph
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


def test_chat_uses_its_durable_request_instead_of_a_separate_paid_probe(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    llm = _RefusesTools()
    result = _run_help_turn(store, llm)

    assert result.intent.kind == "meta_help"
    probes = [event for event in _events(store) if event.event_type == "tool_calling_probe"]
    assert not probes
    assert llm.tool_calls == 1
    assert any(event.event_type == "agent_route_degraded" for event in _events(store))


def test_a_refusal_inside_the_loop_degrades_the_turn_and_says_so(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    llm = _Client(
        UNVERIFIED, raises=ToolCallingUnsupportedError("tools rejected once a schema is attached")
    )

    result = _run_help_turn(store, llm)

    assert result.intent.kind == "meta_help"
    assert llm.tool_calls == 1, "the graph request also establishes tool capability"
    degraded = [event for event in _events(store) if event.event_type == "agent_route_degraded"]
    assert degraded, "a silent fallback would hide a permanently worse analysis"
    assert "schema is attached" in degraded[0].summary["reason"]
