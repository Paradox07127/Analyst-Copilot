"""Settle whether a model can drive the tool loop, before a run spends on it.

The catalog answers for models this repo has checked. For everything else —
which now includes every self-hosted deployment — the endpoint is asked once,
with the cheapest call that can produce an answer, and the verdict is reused
for the rest of the process.

Probing beats waiting for the loop to fail because the failure would otherwise
land after the run has already paid for planning and question discovery.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from eda_platform.core.llm import (
    LLMClient,
    ToolCallingUnsupportedError,
    is_offline_client,
    supports_tool_calling,
)
from eda_platform.core.model_capabilities import is_verified_agent_model
from eda_platform.core.provider_registry import LLMProvider

PROBE_TOOL: dict[str, Any] = {
    "name": "acknowledge",
    "description": "Call this tool with no arguments to acknowledge.",
    "parameters": {"type": "object", "properties": {}},
}
PROBE_MESSAGES: list[dict[str, Any]] = [
    {"role": "user", "content": "Call the acknowledge tool."}
]


@dataclass(frozen=True)
class ToolCallingVerdict:
    usable: bool
    source: str  # offline | client | catalog | probe | cached | unprobed
    detail: str = ""


_verdicts: dict[tuple[LLMProvider, str], ToolCallingVerdict] = {}
_lock = threading.RLock()


def forget_probe_results() -> None:
    with _lock:
        _verdicts.clear()


def tool_calling_readiness(
    llm: LLMClient | None,
    *,
    allow_probe: bool = True,
) -> ToolCallingVerdict:
    if is_offline_client(llm):
        return ToolCallingVerdict(False, "offline", "Offline runs the deterministic path.")
    if not supports_tool_calling(llm):
        return ToolCallingVerdict(False, "client", "This client has no tool-calling transport.")

    settings = getattr(llm, "settings", None)
    provider = getattr(settings, "provider", None)
    model = str(getattr(settings, "model", "") or "")
    if not isinstance(provider, LLMProvider) or not model:
        # A client without provider settings is a test double or an adapter
        # that already asserted its own capability; there is nothing to probe.
        return ToolCallingVerdict(True, "client")

    if is_verified_agent_model(provider, model):
        return ToolCallingVerdict(True, "catalog", f"{model} is in the verified catalog.")

    key = (provider, model)
    with _lock:
        cached = _verdicts.get(key)
    if cached is not None:
        return ToolCallingVerdict(cached.usable, "cached", cached.detail)
    if not allow_probe:
        return ToolCallingVerdict(True, "unprobed")

    verdict = _probe(llm, model)
    with _lock:
        _verdicts[key] = verdict
    return verdict


def _probe(llm: Any, model: str) -> ToolCallingVerdict:
    """One tools-enabled call. Accepting the payload is the whole question —
    whether the model then chooses to call the tool is a quality matter, not a
    capability one, so a plain text reply still counts as support."""
    try:
        llm.tool_call(task="tool_calling_probe", messages=PROBE_MESSAGES, tools=[PROBE_TOOL])
    except ToolCallingUnsupportedError as exc:
        return ToolCallingVerdict(False, "probe", str(exc))
    return ToolCallingVerdict(True, "probe", f"{model} accepted a tools payload.")
