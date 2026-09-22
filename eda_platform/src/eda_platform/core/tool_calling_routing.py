"""Side-effect-free routing for the graph's first durable model request.

Unknown models are allowed to try tool calling inside the graph. Provider
refusals are handled by the durable driver; routing never sends a paid probe.
"""

from __future__ import annotations

from dataclasses import dataclass

from eda_platform.core.llm import (
    LLMClient,
    is_offline_client,
    supports_tool_calling,
)
from eda_platform.core.model_capabilities import is_verified_agent_model
from eda_platform.core.provider_registry import LLMProvider


@dataclass(frozen=True)
class ToolCallingVerdict:
    usable: bool
    source: str  # offline | client | catalog | unprobed
    detail: str = ""


def tool_calling_readiness(llm: LLMClient | None) -> ToolCallingVerdict:
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

    return ToolCallingVerdict(True, "unprobed")
