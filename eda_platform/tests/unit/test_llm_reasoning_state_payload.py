"""gpt-5.6-luna rejects function tools unless reasoning_effort is 'none' on
/v1/chat/completions (provider 400, observed 2026-08-01); the catalog marks the
pin and the OpenAI-compatible tool path must honour it without touching any
other model or request shape."""

from __future__ import annotations

from pydantic import BaseModel

from eda_platform.core.llm import (
    LLMProvider,
    LLMSettings,
    OpenAICompatibleLLMClient,
    build_structured_chat_payload,
)
from eda_platform.core.model_capabilities import (
    agent_model_profile,
    is_verified_agent_model,
)

_TOOL_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "inspect", "arguments": "{}"},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ]
}


class CapturingClient(OpenAICompatibleLLMClient):
    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(settings)
        self.captured: dict = {}

    def _post_json(self, path: str, body: dict) -> dict:
        self.captured = body
        return _TOOL_RESPONSE


def _tool_turn_body(provider: LLMProvider, model: str) -> dict:
    client = CapturingClient(
        LLMSettings(provider=provider, api_key="test", model=model)
    )
    client.tool_call(
        task="agent",
        messages=[{"role": "user", "content": "inspect"}],
        tools=[{"name": "inspect", "parameters": {"type": "object"}}],
    )
    return client.captured


class ToyOutput(BaseModel):
    answer: str


def test_luna_tool_turn_pins_reasoning_effort_none() -> None:
    body = _tool_turn_body(LLMProvider.OPENAI, "gpt-5.6-luna")

    assert body["reasoning_effort"] == "none"
    assert body["tool_choice"] == "auto"  # the rest of the dialect is unchanged


def test_reasoning_capable_models_keep_their_default_effort_with_tools() -> None:
    for provider, model in (
        (LLMProvider.OPENAI, "gpt-5.6-sol"),
        (LLMProvider.OPENAI, "gpt-5.6-terra"),
        (LLMProvider.DEEPSEEK, "deepseek-v4-pro"),
    ):
        body = _tool_turn_body(provider, model)
        assert "reasoning_effort" not in body, (provider, model)


def test_luna_requests_without_tools_do_not_carry_the_pin() -> None:
    settings = LLMSettings(
        provider=LLMProvider.OPENAI, api_key="test", model="gpt-5.6-luna"
    )

    structured = build_structured_chat_payload(
        settings, task="toy", schema=ToyOutput, payload={}
    )
    assert "reasoning_effort" not in structured

    client = CapturingClient(settings)
    client.text(task="toy", payload={})
    assert "reasoning_effort" not in client.captured


def test_catalog_pins_luna_only_and_keeps_it_verified() -> None:
    luna = agent_model_profile(LLMProvider.OPENAI, "gpt-5.6-luna")
    assert luna is not None and luna.tools_reasoning_effort == "none"

    for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-4.1-mini", "gpt-4.1"):
        profile = agent_model_profile(LLMProvider.OPENAI, model)
        assert profile is not None and profile.tools_reasoning_effort == "", model

    # tool_calling_probe's catalog fast path must keep answering from here.
    assert is_verified_agent_model(LLMProvider.OPENAI, "gpt-5.6-luna")
