from __future__ import annotations

import pytest
from pydantic import BaseModel

from eda_platform.core.llm import (
    AnthropicLLMClient,
    LLMProvider,
    LLMSettings,
    OfflineLLMClient,
    OpenAICompatibleLLMClient,
    build_generation_controls,
    build_structured_chat_payload,
    create_llm_client,
    llm_configuration_status,
    to_strict_json_schema,
)
from eda_platform.core.provider_registry import (
    provider_request_profile,
    requires_api_key,
    requires_base_url,
)


class ToyOutput(BaseModel):
    answer: str
    score: int = 0


class FakeUsageClient(OpenAICompatibleLLMClient):
    def _post_json(self, path: str, body: dict) -> dict:
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
            },
        }


class FakeToolClient(OpenAICompatibleLLMClient):
    def __init__(self, settings: LLMSettings, response: dict) -> None:
        super().__init__(settings)
        self.response = response
        self.captured: dict = {}

    def _post_json(self, path: str, body: dict) -> dict:
        self.captured = body
        return self.response


def test_llm_settings_resolve_provider_base_urls() -> None:
    assert LLMSettings(provider=LLMProvider.OPENAI, api_key="sk-test").resolved_base_url == (
        "https://api.openai.com/v1"
    )
    assert LLMSettings(provider=LLMProvider.DEEPSEEK, api_key="sk-test").resolved_base_url == (
        "https://api.deepseek.com"
    )
    assert (
        LLMSettings(
            provider=LLMProvider.OPENAI_COMPATIBLE,
            base_url="http://localhost:11434/v1/",
            api_key="local",
        ).resolved_base_url
        == "http://localhost:11434/v1"
    )


def test_live_llm_client_requires_api_key() -> None:
    settings = LLMSettings(provider=LLMProvider.OPENAI, model="gpt-test")

    with pytest.raises(ValueError, match="API key"):
        create_llm_client(settings)


def test_offline_llm_structured_remains_unavailable() -> None:
    with pytest.raises(RuntimeError, match="unavailable"):
        OfflineLLMClient().structured(task="toy", schema=ToyOutput, payload={})


def test_openai_uses_strict_json_schema_response_format() -> None:
    settings = LLMSettings(
        provider=LLMProvider.OPENAI,
        api_key="sk-test",
        model="gpt-test",
        temperature=0.1,
        max_tokens=500,
    )

    payload = build_structured_chat_payload(
        settings,
        task="toy_task",
        schema=ToyOutput,
        payload={"evidence": {"rows": 3}},
    )

    assert payload["model"] == "gpt-test"
    assert payload["max_completion_tokens"] == 500
    assert "max_tokens" not in payload
    schema = payload["response_format"]["json_schema"]["schema"]
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    # strict mode requires additionalProperties:false and every field required
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"answer", "score"}
    # defaults must be stripped for strict mode
    assert "default" not in schema["properties"]["score"]


@pytest.mark.parametrize(
    ("provider", "expected_param"),
    [
        (LLMProvider.OPENAI, "max_completion_tokens"),
        (LLMProvider.GROQ, "max_completion_tokens"),
        (LLMProvider.DEEPSEEK, "max_tokens"),
        (LLMProvider.OPENAI_COMPATIBLE, "max_tokens"),
    ],
)
def test_generation_controls_follow_provider_request_profile(
    provider: LLMProvider,
    expected_param: str,
) -> None:
    settings = LLMSettings(
        provider=provider,
        base_url="http://localhost:8080/v1",
        api_key="test",
        model="test-model",
        temperature=0.25,
        max_tokens=321,
    )

    controls = build_generation_controls(settings)

    assert controls[expected_param] == 321
    assert controls["temperature"] == 0.25
    other_param = (
        "max_tokens" if expected_param == "max_completion_tokens" else "max_completion_tokens"
    )
    assert other_param not in controls


def test_request_profiles_keep_protocol_metadata_in_registry() -> None:
    openai = provider_request_profile(LLMProvider.OPENAI)
    deepseek = provider_request_profile(LLMProvider.DEEPSEEK)

    assert openai.transport == "openai_chat_completions"
    assert openai.output_token_param == "max_completion_tokens"
    assert "developers.openai.com" in openai.docs_url
    assert deepseek.output_token_param == "max_tokens"
    assert "deepseek.com" in deepseek.docs_url


def test_deepseek_tool_turn_omits_tool_choice_and_preserves_reasoning_state() -> None:
    client = FakeToolClient(
        LLMSettings(
            provider=LLMProvider.DEEPSEEK,
            api_key="test",
            model="deepseek-v4-pro",
            max_tokens=2_000,
        ),
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "opaque-reasoning-state",
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
        },
    )

    response = client.tool_call(
        task="agent",
        messages=[{"role": "user", "content": "inspect"}],
        tools=[{"name": "inspect", "parameters": {"type": "object"}}],
    )

    assert "tool_choice" not in client.captured
    assert client.captured["max_tokens"] == 2_000
    assert response.provider_state["reasoning_content"] == "opaque-reasoning-state"


def test_openai_compatible_defaults_to_json_object_mode() -> None:
    settings = LLMSettings(
        provider=LLMProvider.OPENAI_COMPATIBLE,
        base_url="http://localhost:8080/v1",
        api_key="local",
        model="local-model",
    )

    payload = build_structured_chat_payload(
        settings,
        task="toy_task",
        schema=ToyOutput,
        payload={"evidence": {"rows": 3}},
    )

    assert payload["response_format"]["type"] == "json_object"
    # schema is embedded in the system prompt so the model still targets the shape
    assert "JSON Schema" in payload["messages"][0]["content"]


def test_deepseek_default_pricing_estimates_cost_when_rates_are_not_overridden() -> None:
    settings = LLMSettings(
        provider=LLMProvider.DEEPSEEK,
        api_key="sk-test",
        model="deepseek-v4-flash",
    )
    client = FakeUsageClient(settings)

    assert client.text(task="toy_task", payload={"question": "hello"}) == "ok"

    usage = client.last_usage()
    assert usage is not None
    assert usage.estimated_cost_usd == 0.00028


class FakeCachedUsageClient(OpenAICompatibleLLMClient):
    """Returns a caller-supplied ``usage`` block so cache-field parsing is testable."""

    def __init__(self, settings: LLMSettings, usage_block: dict) -> None:
        super().__init__(settings)
        self._usage_block = usage_block

    def _post_json(self, path: str, body: dict) -> dict:
        return {"choices": [{"message": {"content": "ok"}}], "usage": self._usage_block}


def test_cached_tokens_parsed_from_openai_prompt_tokens_details() -> None:
    settings = LLMSettings(provider=LLMProvider.OPENAI, api_key="sk-test", model="gpt-4.1-mini")
    client = FakeCachedUsageClient(
        settings,
        {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "total_tokens": 1200,
            "prompt_tokens_details": {"cached_tokens": 400},
        },
    )
    client.text(task="toy_task", payload={"q": "hi"})

    usage = client.last_usage()
    assert usage is not None
    assert usage.usage.cached_tokens == 400
    assert usage.usage.cache_hit_rate == 0.4


def test_cached_tokens_fall_back_to_deepseek_hit_field() -> None:
    settings = LLMSettings(
        provider=LLMProvider.DEEPSEEK, api_key="sk-test", model="deepseek-v4-flash"
    )
    client = FakeCachedUsageClient(
        settings,
        {
            "prompt_tokens": 1000,
            "completion_tokens": 0,
            "total_tokens": 1000,
            # DeepSeek mirrors the cache-read count at the top level.
            "prompt_cache_hit_tokens": 320,
            "prompt_cache_miss_tokens": 680,
        },
    )
    client.text(task="toy_task", payload={"q": "hi"})

    assert client.last_usage().usage.cached_tokens == 320  # type: ignore[union-attr]


def test_cached_tokens_clamped_to_prompt_tokens() -> None:
    # A malformed provider response must not push cached tokens above the prompt.
    settings = LLMSettings(provider=LLMProvider.OPENAI, api_key="sk-test", model="gpt-4.1-mini")
    client = FakeCachedUsageClient(
        settings,
        {
            "prompt_tokens": 100,
            "completion_tokens": 0,
            "total_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 999},
        },
    )
    client.text(task="toy_task", payload={"q": "hi"})

    usage = client.last_usage()
    assert usage is not None
    assert usage.usage.cached_tokens == 100
    assert usage.usage.cache_hit_rate == 1.0


def test_cost_discounts_cached_prompt_tokens() -> None:
    settings = LLMSettings(
        provider=LLMProvider.DEEPSEEK, api_key="sk-test", model="deepseek-v4-flash"
    )
    client = FakeCachedUsageClient(
        settings,
        {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
            "prompt_tokens_details": {"cached_tokens": 400},
        },
    )
    client.text(task="toy_task", payload={"q": "hi"})

    usage = client.last_usage()
    assert usage is not None
    # 600 fresh prompt at $0.14/1M + 400 cached at DeepSeek's published
    # cache-hit rate of $0.0028/1M + 500 output at $0.28/1M. That hit rate is
    # 0.02x the miss price; the 0.1x cross-provider guess this used to apply
    # overcharged a cache read fivefold.
    assert usage.estimated_cost_usd is not None
    assert usage.estimated_cost_usd == 0.000225
    assert usage.estimated_cost_usd < 0.00028


def test_to_strict_json_schema_marks_nested_objects() -> None:
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"inner": {"type": "string", "default": "x"}},
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                },
            },
        },
    }
    strict = to_strict_json_schema(schema)
    assert strict["additionalProperties"] is False
    assert set(strict["required"]) == {"outer", "items"}
    assert strict["properties"]["outer"]["additionalProperties"] is False
    assert strict["properties"]["outer"]["required"] == ["inner"]
    assert "default" not in strict["properties"]["outer"]["properties"]["inner"]
    assert strict["properties"]["items"]["items"]["additionalProperties"] is False


def test_create_llm_client_returns_openai_compatible_client() -> None:
    settings = LLMSettings(
        provider=LLMProvider.DEEPSEEK,
        api_key="sk-test",
        model="deepseek-v4-flash",
    )

    client = create_llm_client(settings)

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.base_url == "https://api.deepseek.com"


def test_llm_configuration_status_explains_offline_and_ready_states() -> None:
    offline = llm_configuration_status(LLMSettings(provider=LLMProvider.OFFLINE))
    ready = llm_configuration_status(
        LLMSettings(provider=LLMProvider.OPENAI, api_key="sk-test", model="gpt-4.1-mini")
    )

    assert offline.state == "offline"
    assert offline.is_ready_for_live_calls is False
    assert "deterministic fallback" in offline.message
    assert ready.state == "ready"
    assert ready.is_ready_for_live_calls is True
    assert ready.resolved_base_url == "https://api.openai.com/v1"


def test_llm_configuration_status_lists_missing_live_fields() -> None:
    deepseek = llm_configuration_status(LLMSettings(provider=LLMProvider.DEEPSEEK))
    compatible = llm_configuration_status(
        LLMSettings(provider=LLMProvider.OPENAI_COMPATIBLE, api_key="local")
    )

    assert deepseek.state == "incomplete"
    assert deepseek.missing_fields == ["api_key"]
    assert compatible.state == "incomplete"
    assert compatible.missing_fields == ["base_url"]


@pytest.mark.parametrize("provider", list(LLMProvider))
def test_llm_configuration_status_matches_provider_registry(
    provider: LLMProvider,
) -> None:
    if provider is LLMProvider.OFFLINE:
        status = llm_configuration_status(LLMSettings(provider=provider))
        assert status.state == "offline"
        assert status.is_ready_for_live_calls is False
        return

    incomplete = llm_configuration_status(LLMSettings(provider=provider))
    expected_missing = [
        field
        for field, required in (
            ("api_key", requires_api_key(provider)),
            ("base_url", requires_base_url(provider)),
        )
        if required
    ]
    assert incomplete.missing_fields == expected_missing
    assert incomplete.state == ("incomplete" if expected_missing else "ready")

    complete = llm_configuration_status(
        LLMSettings(
            provider=provider,
            api_key="test-key" if requires_api_key(provider) else "",
            base_url="https://provider.invalid/v1" if requires_base_url(provider) else "",
        )
    )
    assert complete.state == "ready"
    assert complete.is_ready_for_live_calls is True
    assert complete.missing_fields == []


def test_registry_resolves_base_urls_and_structured_modes() -> None:
    def _s(provider: LLMProvider, model: str = "m") -> LLMSettings:
        return LLMSettings(provider=provider, api_key="k", model=model)

    assert _s(LLMProvider.ANTHROPIC).resolved_base_url == "https://api.anthropic.com"
    assert _s(LLMProvider.QWEN).resolved_base_url.endswith("/compatible-mode/v1")
    assert _s(LLMProvider.OLLAMA).resolved_base_url == "http://localhost:11434/v1"
    assert (
        _s(LLMProvider.GEMINI).resolved_base_url
        == "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    # "auto" structured mode resolves per provider: strict schema where supported,
    # json_object for labs that don't guarantee schema enforcement.
    assert _s(LLMProvider.OPENAI).resolved_structured_mode == "json_schema"
    assert _s(LLMProvider.XAI).resolved_structured_mode == "json_schema"
    assert _s(LLMProvider.DEEPSEEK).resolved_structured_mode == "json_object"
    assert _s(LLMProvider.QWEN).resolved_structured_mode == "json_object"


def test_local_providers_do_not_require_api_key() -> None:
    settings = LLMSettings(provider=LLMProvider.OLLAMA, model="llama3.1:8b")
    client = create_llm_client(settings)  # no api_key — must not raise
    assert isinstance(client, OpenAICompatibleLLMClient)


def test_azure_uses_api_key_header_not_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}'

    def _fake_urlopen(req, timeout: float = 0):  # noqa: ANN001
        captured.update(req.headers)
        return _FakeResp()

    monkeypatch.setattr("eda_platform.core.llm.request.urlopen", _fake_urlopen)
    settings = LLMSettings(
        provider=LLMProvider.AZURE_OPENAI,
        api_key="azkey",
        base_url="https://res.openai.azure.com/openai/v1",
        model="my-deployment",
    )
    OpenAICompatibleLLMClient(settings).text(task="t", payload={})
    # urllib capitalizes header keys; Azure auth is the `api-key` header, not Bearer.
    assert captured.get("Api-key") == "azkey"
    assert "Authorization" not in captured


class _FakeAnthropic(AnthropicLLMClient):
    def __init__(self, settings: LLMSettings, response: dict) -> None:
        super().__init__(settings)
        self._response = response
        self.captured: dict = {}

    def _post_json(self, path: str, body: dict) -> dict:
        self.captured = body
        return self._response


def test_anthropic_structured_forces_tool_and_maps_usage() -> None:
    settings = LLMSettings(provider=LLMProvider.ANTHROPIC, api_key="sk", model="claude-opus-4-8")
    response = {
        "content": [
            {"type": "tool_use", "name": "emit_result", "input": {"answer": "ok", "score": 3}}
        ],
        "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 30},
    }
    client = _FakeAnthropic(settings, response)
    out = client.structured(task="t", schema=ToyOutput, payload={"q": "hi"})

    assert out.answer == "ok" and out.score == 3
    assert client.captured["tool_choice"] == {"type": "tool", "name": "emit_result"}
    assert client.captured["max_tokens"] == settings.max_tokens
    assert "temperature" not in client.captured
    usage = client.last_usage()
    assert usage is not None
    # input_tokens excludes cached, so prompt = input + cache_read.
    assert usage.usage.prompt_tokens == 130
    assert usage.usage.cached_tokens == 30
    assert usage.usage.completion_tokens == 20


def test_anthropic_text_extracts_text_blocks() -> None:
    settings = LLMSettings(
        provider=LLMProvider.ANTHROPIC,
        api_key="sk",
        model="claude-haiku-4-5",
        temperature=0.4,
    )
    response = {
        "content": [{"type": "text", "text": "hello"}],
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    client = _FakeAnthropic(settings, response)
    assert client.text(task="t", payload={}) == "hello"
    assert client.captured["temperature"] == 0.4


def test_create_llm_client_routes_anthropic_to_native() -> None:
    settings = LLMSettings(provider=LLMProvider.ANTHROPIC, api_key="sk", model="claude-opus-4-8")
    assert isinstance(create_llm_client(settings), AnthropicLLMClient)


def test_socket_timeout_becomes_actionable_runtime_error(monkeypatch) -> None:
    # A per-call socket read timeout must surface as a clear, actionable error
    # naming EDA_LLM_TIMEOUT_SECONDS — not leak "The read operation timed out".
    # Backoff is zeroed so the bounded transport retry does not add real sleeps.
    monkeypatch.setattr(
        "eda_platform.core.llm._TRANSPORT_RETRY_BACKOFF_SECONDS", (0.0, 0.0)
    )
    settings = LLMSettings(
        provider=LLMProvider.DEEPSEEK,
        api_key="sk-test",
        model="deepseek-v4-flash",
        timeout_seconds=42.0,
    )
    client = OpenAICompatibleLLMClient(settings)

    def _raise_timeout(*args, **kwargs):
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr("eda_platform.core.llm.request.urlopen", _raise_timeout)

    with pytest.raises(RuntimeError) as excinfo:
        client._post_json("/chat/completions", {})
    message = str(excinfo.value)
    assert "42s" in message
    assert "EDA_LLM_TIMEOUT_SECONDS" in message


def test_default_timeout_is_generous_out_of_box() -> None:
    assert LLMSettings(provider=LLMProvider.OFFLINE).timeout_seconds == 180.0


def test_an_unlisted_anthropic_model_still_reaches_the_endpoint() -> None:
    """A capability catalog that predates a model must not make it unreachable.

    Refusing locally also raised the wrong type: the batch-level fallback only
    catches ToolCallingUnsupportedError, so each question failed on its own
    instead of the batch degrading once to the deterministic pipeline.
    """
    settings = LLMSettings(
        provider=LLMProvider.ANTHROPIC, api_key="sk", model="claude-not-yet-catalogued"
    )
    client = _FakeAnthropic(
        settings,
        {
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        },
    )

    response = client.tool_call(
        task="tool_calling_probe",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
    )

    assert response.content == "done"
    assert client.captured["model"] == "claude-not-yet-catalogued"
