from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Protocol, TypeVar
from urllib import error, request
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from eda_platform.core.cancellation import CancellationToken
from eda_platform.core.ids import stable_hash
from eda_platform.core.model_capabilities import agent_model_profile
from eda_platform.core.provider_registry import (
    PRICING_CATALOG_VERSION,
    LLMProvider,
    auth_style,
    cache_read_price_per_1m,
    cache_write_price_per_1m,
    default_base_url,
    default_structured_mode,
    pricing_per_1m,
    provider_request_profile,
    requires_api_key,
    requires_base_url,
)
from eda_platform.core.request_dialect import (
    MAX_PARAM_REPAIRS,
    apply_learned_repairs,
    apply_repair,
    plan_repair,
    remember_repair,
)

T = TypeVar("T", bound=BaseModel)


class ToolCallingUnsupportedError(RuntimeError):
    """The endpoint answered that it will not accept a tools payload.

    Narrow on purpose. Degrading to the deterministic path on any error would
    turn a transient outage or a bad key into a silently worse analysis, so
    only a provider that diagnosed the tools payload itself raises this.
    """


# Substrings a provider uses when it is the tools payload it objects to. An
# error merely containing the word "function" (a Python traceback, say) is not
# enough — the rejection must also name a tool field this request actually sent.
_TOOL_REJECTION_MARKERS = (
    "tools",
    "tool_choice",
    "tool use",
    "tool calling",
    "function calling",
    "functions",
)


class _RejectedRequest(Exception):
    """An HTTP error still holding its body, so the caller can decide whether
    the rejection describes a repairable request parameter."""

    def __init__(self, *, code: int, detail: str, cause: BaseException) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code
        self.detail = detail
        self.cause = cause

    def as_error(self) -> RuntimeError:
        return RuntimeError(f"LLM provider returned HTTP {self.code}: {self.detail}")


class LLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Reused prompt tokens, a subset of prompt_tokens billed at the cache-read rate.
    cached_tokens: int = 0
    # Anthropic meters cache writes separately because they carry a premium;
    # also a subset of prompt_tokens.
    cache_creation_tokens: int = 0
    # A subset of completion_tokens, not tokens on top of them. Adding it to a
    # total would double-count.
    reasoning_tokens: int = 0

    @property
    def cache_hit_rate(self) -> float:
        if self.prompt_tokens <= 0:
            return 0.0
        return round(min(self.cached_tokens, self.prompt_tokens) / self.prompt_tokens, 6)


class LLMResultMetadata(BaseModel):
    provider: str
    model: str
    usage: LLMUsage = Field(default_factory=LLMUsage)
    estimated_cost_usd: float | None = None
    # Inference APIs report usage, not the invoice. cost_basis says which rates
    # produced estimated_cost_usd so an estimate is never read as billed cost.
    cost_basis: str = "unavailable"
    pricing_version: str = ""
    # False when the provider returned no usage block. Without it a silent
    # provider is indistinguishable from a genuine 0-token call, and the ledger
    # reports an unknown call as a free one.
    usage_reported: bool = True
    request_id: str = ""
    response_id: str = ""
    finish_reason: str = ""
    endpoint_host: str = ""
    request_bytes: int = 0
    response_bytes: int = 0
    # Request parameters this call had to rename or drop because the provider
    # rejected them. Self-healing that left no trace would hide a stale catalog
    # entry indefinitely.
    param_repairs: list[str] = Field(default_factory=list)


class LLMConfigurationStatus(BaseModel):
    state: str
    message: str
    is_ready_for_live_calls: bool
    missing_fields: list[str] = Field(default_factory=list)
    resolved_base_url: str = ""


class LLMToolCall(BaseModel):
    """One provider-normalised function invocation requested by a model.

    The rest of the application must never need to know whether a provider
    called this a ``tool_call``, ``function`` or ``tool_use`` block.
    """

    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMToolResponse(BaseModel):
    """The text and/or tool calls returned for one agent-loop model step."""

    content: str = ""
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    finish_reason: str = ""
    provider_state: dict[str, Any] = Field(default_factory=dict)


class StructuredLLM(Protocol):
    """Minimal surface for agents that only need schema-constrained calls."""

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T: ...


class LLMClient(Protocol):
    def structured(self, *, task: str, schema: type[T], payload: dict) -> T: ...

    def text(self, *, task: str, payload: dict) -> str: ...

    def last_usage(self) -> LLMResultMetadata | None: ...


class ToolCallingLLM(LLMClient, Protocol):
    """Provider-neutral surface used by the bounded agent runtime."""

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolResponse: ...


class CancellableLLMClient:
    """Execution seam adding durable checkpoints around an existing provider.

    HTTP clients based on blocking stdlib sockets cannot guarantee mid-flight
    cancellation. ``abort_active_call`` is the hard-termination adapter for a
    provider SDK/transport that can close its active request. Without it,
    cancellation is still observed immediately after the call and prevents
    every downstream/repair step.
    """

    def __init__(
        self,
        client: LLMClient,
        cancellation: CancellationToken,
        *,
        abort_active_call: Callable[[], None] | None = None,
    ) -> None:
        self._client = client
        self._cancellation = cancellation
        self._abort_active_call = abort_active_call or (lambda: None)

    @property
    def inner(self) -> LLMClient:
        """Expose the decorated client for capability checks and metering."""

        return self._client

    @property
    def settings(self) -> Any:
        """Carry provider configuration through the seam.

        Without this the worker's ledger sees ``settings is None`` — every job
        goes through ``_build_llm``, which wraps in this class — so
        ``_worst_case_cost`` returns None and a priced call is recorded as
        cost-unknown.
        """

        return getattr(self._client, "settings", None)

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self._cancellation.checkpoint()
        with self._cancellation.interrupt_on_cancel(self._abort_active_call):
            result = self._client.structured(task=task, schema=schema, payload=payload)
        self._cancellation.checkpoint()
        return result

    def text(self, *, task: str, payload: dict) -> str:
        self._cancellation.checkpoint()
        with self._cancellation.interrupt_on_cancel(self._abort_active_call):
            result = self._client.text(task=task, payload=payload)
        self._cancellation.checkpoint()
        return result

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolResponse:
        self._cancellation.checkpoint()
        with self._cancellation.interrupt_on_cancel(self._abort_active_call):
            result = _require_tool_calling(self._client).tool_call(
                task=task,
                messages=messages,
                tools=tools,
            )
        self._cancellation.checkpoint()
        return result

    def last_usage(self) -> LLMResultMetadata | None:
        return self._client.last_usage()


class OfflineLLMClient:
    def __init__(self) -> None:
        self._last: LLMResultMetadata | None = None

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        raise RuntimeError(f"LLM task {task!r} is unavailable in offline mode.")

    def text(self, *, task: str, payload: dict) -> str:
        return "LLM is not configured. Deterministic profiling only."

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolResponse:
        raise RuntimeError(f"LLM task {task!r} is unavailable in offline mode.")

    def last_usage(self) -> LLMResultMetadata | None:
        return self._last


StructuredMode = str  # "auto" | "json_schema" | "json_object"


class LLMSettings(BaseModel):
    provider: LLMProvider = LLMProvider.OFFLINE
    api_key: str = ""
    base_url: str = ""
    model: str = "gpt-4.1-mini"
    temperature: float = 0.2
    max_tokens: int = 6000
    # Per-request socket read timeout. Slow reasoning models can take well over
    # a minute on a large structured-output call (report/synthesis), so the
    # out-of-box default is generous; raise EDA_LLM_TIMEOUT_SECONDS for models
    # that are slower still.
    timeout_seconds: float = 180.0
    organization: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    structured_output_mode: StructuredMode = "auto"
    usd_per_1k_prompt: float = 0.0
    usd_per_1k_completion: float = 0.0

    @property
    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        return default_base_url(self.provider).rstrip("/")

    @property
    def is_live_provider(self) -> bool:
        return self.provider is not LLMProvider.OFFLINE

    @property
    def resolved_structured_mode(self) -> str:
        if self.structured_output_mode != "auto":
            return self.structured_output_mode
        return default_structured_mode(self.provider)


class OpenAICompatibleLLMClient:
    # urllib exposes no supported primitive for aborting an in-flight blocking
    # request from another thread. Worker jobs therefore rely on cooperative
    # post-call checkpoints followed by the durable grace-timeout process fence.
    active_abort_supported = False

    def __init__(self, settings: LLMSettings) -> None:
        if requires_api_key(settings.provider) and not settings.api_key:
            raise ValueError("API key is required for this provider.")
        if requires_base_url(settings.provider) and not settings.resolved_base_url:
            raise ValueError("Base URL is required for this provider.")
        if not settings.resolved_base_url:
            raise ValueError("Base URL could not be resolved for this provider.")
        self.settings = settings
        self.base_url = settings.resolved_base_url
        self._last: LLMResultMetadata | None = None
        self._last_request_id = ""
        self._last_request_bytes = 0
        self._last_response_bytes = 0
        self._last_param_repairs: list[str] = []

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        body = build_structured_chat_payload(
            self.settings,
            task=task,
            schema=schema,
            payload=payload,
        )
        response = self._post_json("/chat/completions", body)
        self._record_usage(response)
        content = _message_content(response)
        return schema.model_validate_json(content)

    def text(self, *, task: str, payload: dict) -> str:
        body = {
            "model": self.settings.model,
            **build_generation_controls(self.settings),
            "messages": [
                {"role": "system", "content": "You are a concise data analysis assistant."},
                {"role": "user", "content": json.dumps({"task": task, "payload": payload})},
            ],
        }
        response = self._post_json("/chat/completions", body)
        self._record_usage(response)
        return _message_content(response)

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolResponse:
        """Run one native OpenAI-compatible tool-calling turn.

        DeepSeek and the other OpenAI-compatible providers use this same wire
        format. Tool argument validation deliberately happens in the local
        runtime, not here or in the model prompt.
        """
        # No profile means unverified, not unusable: fall through on the
        # provider's own defaults and let the endpoint answer. Refusing here is
        # what made every self-hosted model unreachable.
        capability = agent_model_profile(self.settings.provider, self.settings.model)
        body: dict[str, Any] = {
            "model": self.settings.model,
            **build_generation_controls(self.settings),
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": str(tool["name"]),
                        "description": str(tool.get("description", "")),
                        "parameters": tool.get("parameters", {"type": "object"}),
                    },
                }
                for tool in tools
            ],
        }
        # Unverified models get the OpenAI-compatible default. If the endpoint
        # objects, `tool_choice` is repairable and the next attempt drops it.
        if capability is None or capability.tool_choice_policy != "omit":
            body["tool_choice"] = "auto"
        response = self._post_json("/chat/completions", body)
        self._record_usage(response)
        return _openai_tool_response(response)

    def last_usage(self) -> LLMResultMetadata | None:
        return self._last

    def _post_json(self, path: str, body: dict) -> dict:
        """Send the request, repairing a rejected generation parameter in place.

        The retry is deliberately not a generic one: it fires only when the
        provider names a parameter this module is willing to rewrite, so an
        auth failure or a rate limit still surfaces on the first response.
        """
        provider, model = self.settings.provider, self.settings.model
        body, repairs = apply_learned_repairs(provider, model, body)
        self._last_param_repairs = list(repairs)
        for _ in range(MAX_PARAM_REPAIRS + 1):
            try:
                return self._send_json(path, body)
            except _RejectedRequest as rejection:
                repair = plan_repair(body, rejection.detail)
                if repair is None:
                    if "tools" in body and _rejects_tools(rejection.detail):
                        raise ToolCallingUnsupportedError(
                            f"Model '{model}' will not accept a tools payload: "
                            f"{rejection.detail}"
                        ) from rejection.cause
                    raise rejection.as_error() from rejection.cause
                body = apply_repair(body, repair)
                remember_repair(provider, model, repair)
                repairs.append(repair.describe())
                self._last_param_repairs = list(repairs)
        raise RuntimeError(
            f"LLM provider kept rejecting request parameters after "
            f"{MAX_PARAM_REPAIRS} repairs ({', '.join(repairs)})."
        )

    def _send_json(self, path: str, body: dict) -> dict:
        url = f"{self.base_url}{path}"
        headers: dict[str, str] = {"Content-Type": "application/json", **self.settings.headers}
        style = auth_style(self.settings.provider)
        key = self.settings.api_key
        if style == "api_key_header" and key:  # Azure: `api-key` header, not Bearer
            headers["api-key"] = key
        elif key:  # bearer (also local servers when a key is provided)
            headers["Authorization"] = f"Bearer {key}"
        if self.settings.organization:
            headers["OpenAI-Organization"] = self.settings.organization
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self._last_request_bytes = len(data)
        req = request.Request(url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.settings.timeout_seconds) as response:
                self._last_request_id = _response_request_id(getattr(response, "headers", None))
                raw = response.read()
                self._last_response_bytes = len(raw)
                return json.loads(raw.decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise _RejectedRequest(code=exc.code, detail=detail, cause=exc) from exc
        except TimeoutError as exc:
            # socket.timeout is TimeoutError (not a URLError subclass), so it
            # would otherwise leak its bare "read operation timed out" message.
            raise RuntimeError(
                f"LLM provider did not respond within "
                f"{self.settings.timeout_seconds:.0f}s "
                f"(a single call to model '{self.settings.model}' timed out). "
                f"Raise EDA_LLM_TIMEOUT_SECONDS for a slower model."
            ) from exc
        except error.URLError as exc:
            # A wrapped socket timeout can also arrive here (reason is the
            # underlying OSError); surface the tuning knob in that case too.
            if isinstance(exc.reason, TimeoutError):
                raise RuntimeError(
                    f"LLM provider did not respond within "
                    f"{self.settings.timeout_seconds:.0f}s "
                    f"(a single call to model '{self.settings.model}' timed out). "
                    f"Raise EDA_LLM_TIMEOUT_SECONDS for a slower model."
                ) from exc
            raise RuntimeError(f"LLM provider request failed: {exc.reason}") from exc

    def _record_usage(self, response: dict) -> None:
        usage_raw = response.get("usage") if isinstance(response, dict) else None
        usage = LLMUsage()
        if isinstance(usage_raw, dict):
            prompt_tokens = _coerce_int(usage_raw.get("prompt_tokens"))
            completion_tokens = _coerce_int(usage_raw.get("completion_tokens"))
            total_tokens = _coerce_int(usage_raw.get("total_tokens"))
            usage = LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens or prompt_tokens + completion_tokens,
                cached_tokens=_extract_cached_tokens(usage_raw, prompt_tokens),
                reasoning_tokens=_extract_reasoning_tokens(usage_raw, completion_tokens),
            )
        cost, cost_basis = (
            _estimate_cost(self.settings, usage)
            if isinstance(usage_raw, dict)
            else (None, "unavailable")
        )
        self._last = LLMResultMetadata(
            provider=self.settings.provider.value,
            model=_response_model(response) or self.settings.model,
            usage=usage,
            estimated_cost_usd=cost,
            cost_basis=cost_basis,
            pricing_version=_pricing_version(cost_basis),
            usage_reported=isinstance(usage_raw, dict),
            request_id=self._last_request_id,
            response_id=_response_id(response),
            finish_reason=_openai_finish_reason(response),
            endpoint_host=urlsplit(self.base_url).netloc,
            request_bytes=self._last_request_bytes,
            response_bytes=self._last_response_bytes,
            param_repairs=list(self._last_param_repairs),
        )


_ANTHROPIC_VERSION = "2023-06-01"
_ANTHROPIC_SYSTEM = (
    "You are an evidence-grounded EDA reporting agent. Return only the requested "
    "structured result via the provided tool. Do not add commentary."
)


class AnthropicLLMClient:
    """Native Anthropic Messages client. Structured output is forced via a single
    tool call (the compat endpoint ignores response_format), so json_schema holds."""

    active_abort_supported = False

    def __init__(self, settings: LLMSettings) -> None:
        if not settings.api_key:
            raise ValueError("API key is required for Anthropic.")
        self.settings = settings
        self.base_url = settings.resolved_base_url
        self._last: LLMResultMetadata | None = None
        self._last_request_id = ""
        self._last_request_bytes = 0
        self._last_response_bytes = 0

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        strict_schema = to_strict_json_schema(schema.model_json_schema())
        body = {
            "model": self.settings.model,
            **build_generation_controls(self.settings),
            "system": _ANTHROPIC_SYSTEM,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps({"task": task, "payload": payload}, ensure_ascii=False),
                }
            ],
            "tools": [
                {
                    "name": "emit_result",
                    "description": "Return the structured analysis result.",
                    "input_schema": strict_schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": "emit_result"},
        }
        response = self._post_json("/v1/messages", body)
        self._record_usage(response)
        return schema.model_validate(_anthropic_tool_input(response))

    def text(self, *, task: str, payload: dict) -> str:
        body = {
            "model": self.settings.model,
            **build_generation_controls(self.settings),
            "system": "You are a concise data analysis assistant.",
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps({"task": task, "payload": payload}, ensure_ascii=False),
                }
            ],
        }
        response = self._post_json("/v1/messages", body)
        self._record_usage(response)
        return _anthropic_text(response)

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolResponse:
        if agent_model_profile(self.settings.provider, self.settings.model) is None:
            raise ValueError("Model is not verified for the Agent tool loop.")
        system, provider_messages = _anthropic_agent_messages(messages)
        body: dict[str, Any] = {
            "model": self.settings.model,
            **build_generation_controls(self.settings),
            "system": system,
            "messages": provider_messages,
            "tools": [
                {
                    "name": str(tool["name"]),
                    "description": str(tool.get("description", "")),
                    "input_schema": tool.get("parameters", {"type": "object"}),
                }
                for tool in tools
            ],
        }
        response = self._post_json("/v1/messages", body)
        self._record_usage(response)
        return _anthropic_tool_response(response)

    def last_usage(self) -> LLMResultMetadata | None:
        return self._last

    def _post_json(self, path: str, body: dict) -> dict:
        url = f"{self.base_url}{path}"
        headers = {
            "x-api-key": self.settings.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
            **self.settings.headers,
        }
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self._last_request_bytes = len(data)
        req = request.Request(url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.settings.timeout_seconds) as response:
                self._last_request_id = _response_request_id(getattr(response, "headers", None))
                raw = response.read()
                self._last_response_bytes = len(raw)
                return json.loads(raw.decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic returned HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, error.URLError) as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):
                raise RuntimeError(
                    f"Anthropic did not respond within {self.settings.timeout_seconds:.0f}s "
                    f"(model '{self.settings.model}'). Raise EDA_LLM_TIMEOUT_SECONDS."
                ) from exc
            raise RuntimeError(f"Anthropic request failed: {reason}") from exc

    def _record_usage(self, response: dict) -> None:
        usage_raw = response.get("usage") if isinstance(response, dict) else None
        usage = LLMUsage()
        if isinstance(usage_raw, dict):
            # Native input_tokens EXCLUDES cached tokens, so recombine for a
            # prompt total comparable to the OpenAI shape.
            cache_read = _coerce_int(usage_raw.get("cache_read_input_tokens"))
            cache_write = _coerce_int(usage_raw.get("cache_creation_input_tokens"))
            input_tokens = _coerce_int(usage_raw.get("input_tokens"))
            completion = _coerce_int(usage_raw.get("output_tokens"))
            prompt = input_tokens + cache_read + cache_write
            usage = LLMUsage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
                cached_tokens=cache_read,
                cache_creation_tokens=cache_write,
            )
        cost, cost_basis = (
            _estimate_cost(self.settings, usage)
            if isinstance(usage_raw, dict)
            else (None, "unavailable")
        )
        self._last = LLMResultMetadata(
            provider=self.settings.provider.value,
            model=_response_model(response) or self.settings.model,
            usage=usage,
            estimated_cost_usd=cost,
            cost_basis=cost_basis,
            pricing_version=_pricing_version(cost_basis),
            usage_reported=isinstance(usage_raw, dict),
            request_id=self._last_request_id,
            response_id=_response_id(response),
            finish_reason=str(response.get("stop_reason") or ""),
            endpoint_host=urlsplit(self.base_url).netloc,
            request_bytes=self._last_request_bytes,
            response_bytes=self._last_response_bytes,
        )


def _anthropic_tool_input(response: dict) -> dict:
    for block in response.get("content", []) if isinstance(response, dict) else []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            result = block.get("input")
            if isinstance(result, dict):
                return result
    raise RuntimeError("Anthropic response did not contain a tool_use result.")


def _anthropic_text(response: dict) -> str:
    parts = [
        block.get("text", "")
        for block in (response.get("content", []) if isinstance(response, dict) else [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    joined = "".join(parts)
    if not joined:
        raise RuntimeError("Anthropic response did not contain text content.")
    return joined


def _openai_tool_response(response: dict) -> LLMToolResponse:
    """Parse a Chat Completions response without trusting tool arguments."""
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("LLM response did not contain choices.")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("LLM response message is malformed.")
    content = _content_text(message.get("content"))
    calls: list[LLMToolCall] = []
    raw_calls = message.get("tool_calls")
    if raw_calls is not None and not isinstance(raw_calls, list):
        raise RuntimeError("LLM response tool_calls is malformed.")
    for index, raw in enumerate(raw_calls or []):
        if not isinstance(raw, dict):
            raise RuntimeError("LLM response tool call is malformed.")
        function = raw.get("function")
        if not isinstance(function, dict):
            raise RuntimeError("LLM response tool call function is malformed.")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError("LLM response tool call has no function name.")
        arguments = _tool_arguments(function.get("arguments"))
        call_id = raw.get("id")
        calls.append(
            LLMToolCall(
                call_id=str(call_id or f"tool_{index + 1}"),
                name=name,
                arguments=arguments,
            )
        )
    return LLMToolResponse(
        content=content,
        tool_calls=calls,
        finish_reason=str(choice.get("finish_reason") or ""),
        provider_state=(
            {"reasoning_content": message["reasoning_content"]}
            if isinstance(message.get("reasoning_content"), str)
            else {}
        ),
    )


def _anthropic_tool_response(response: dict) -> LLMToolResponse:
    blocks = response.get("content") if isinstance(response, dict) else None
    if not isinstance(blocks, list):
        raise RuntimeError("Anthropic response did not contain content blocks.")
    texts: list[str] = []
    calls: list[LLMToolCall] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif kind == "tool_use":
            name = block.get("name")
            arguments = block.get("input")
            if not isinstance(name, str) or not name or not isinstance(arguments, dict):
                raise RuntimeError("Anthropic response tool_use block is malformed.")
            calls.append(
                LLMToolCall(
                    call_id=str(block.get("id") or f"tool_{index + 1}"),
                    name=name,
                    arguments=arguments,
                )
            )
    if not texts and not calls:
        raise RuntimeError("Anthropic response contained neither text nor tool calls.")
    return LLMToolResponse(
        content="".join(texts),
        tool_calls=calls,
        finish_reason=str(response.get("stop_reason") or ""),
    )


def _anthropic_agent_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Translate the runtime's provider-neutral transcript to Messages API.

    The runtime keeps OpenAI-style ``tool`` messages because they make the
    loop and test fixtures compact. Translation stays at the provider seam.
    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content", "")
        if role == "system":
            text = _content_text(content)
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if not call_id:
                raise RuntimeError("Agent tool result is missing tool_call_id.")
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": _content_text(content),
                        }
                    ],
                }
            )
            continue
        blocks: list[dict[str, Any]] = []
        text = _content_text(content)
        if text:
            blocks.append({"type": "text", "text": text})
        if role == "assistant":
            raw_calls = message.get("tool_calls")
            if isinstance(raw_calls, list):
                for raw in raw_calls:
                    if not isinstance(raw, dict):
                        continue
                    function = raw.get("function")
                    if not isinstance(function, dict):
                        continue
                    name = function.get("name")
                    if not isinstance(name, str) or not name:
                        continue
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": str(raw.get("id") or "tool"),
                            "name": name,
                            "input": _tool_arguments(function.get("arguments")),
                        }
                    )
        converted.append(
            {
                "role": "assistant" if role == "assistant" else "user",
                "content": blocks or [{"type": "text", "text": "Continue."}],
            }
        )
    if not converted:
        converted.append({"role": "user", "content": "Continue."})
    return "\n\n".join(system_parts), converted


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [item.get("text", "") for item in value if isinstance(item, dict)]
        return "".join(part for part in parts if isinstance(part, str))
    return ""


def _tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise RuntimeError("LLM tool arguments must be a JSON object.")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM tool arguments are not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("LLM tool arguments must decode to a JSON object.")
    return parsed


def _rejects_tools(detail: str) -> bool:
    return any(marker in detail.casefold() for marker in _TOOL_REJECTION_MARKERS)


def supports_tool_calling(llm: LLMClient | None) -> bool:
    """Whether a decorated client can actually run the native agent loop.

    Offline is excluded here rather than at each call site. ``OfflineLLMClient``
    defines ``tool_call`` only to raise, so a pure "is the method there?" check
    answered True for it and every agent gate had to remember to write
    ``not is_offline_client(x) and supports_tool_calling(x)``. Two did; a third
    that forgot would have entered the loop and failed on the first tool call.
    """
    if is_offline_client(llm):
        return False
    seen: set[int] = set()
    current: object | None = llm
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        inner = getattr(current, "inner", None)
        if inner is None:
            return callable(getattr(current, "tool_call", None))
        current = inner
    return False


def _require_tool_calling(llm: LLMClient) -> ToolCallingLLM:
    if not supports_tool_calling(llm):
        raise RuntimeError("The configured LLM provider does not support tool calling.")
    return llm  # type: ignore[return-value]


def is_offline_client(llm: LLMClient | None) -> bool:
    """Capability check that survives decoration.

    Clients get wrapped (spend ledger, dev-log instrumentation), so asking
    ``isinstance(llm, OfflineLLMClient)`` silently answers False for a wrapped
    offline client and lets offline runs make "live" calls. Unwrap first.
    """
    if llm is None:
        return True
    seen: set[int] = set()
    current: object | None = llm
    while current is not None and id(current) not in seen:
        if isinstance(current, OfflineLLMClient):
            return True
        seen.add(id(current))  # a self-referencing wrapper must not spin forever
        current = getattr(current, "inner", None)
    return False


def create_llm_client(settings: LLMSettings | None = None) -> LLMClient:
    if settings is None or settings.provider is LLMProvider.OFFLINE:
        return OfflineLLMClient()
    if settings.provider is LLMProvider.ANTHROPIC:
        return AnthropicLLMClient(settings)
    return OpenAICompatibleLLMClient(settings)


def manifest_model_versions(llm: LLMClient | None) -> dict[str, str]:
    """Return the provider-to-model mapping used in run manifests."""
    settings = getattr(llm, "settings", None)
    if isinstance(settings, LLMSettings) and settings.is_live_provider:
        return {settings.provider.value: settings.model}
    return {"offline": "deterministic"}


def llm_execution_fingerprint(llm: LLMClient | None) -> str:
    """Digest every effective setting that can change provider behaviour.

    Secrets and custom header values are only inputs to the digest; checkpoint
    files persist the resulting hash, never the configuration itself.
    """
    settings = getattr(llm, "settings", None)
    if not isinstance(settings, LLMSettings):
        return stable_hash({"provider": "offline", "mode": "deterministic"})
    return stable_hash(
        {
            "provider": settings.provider.value,
            "model": settings.model,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
            "timeout_seconds": settings.timeout_seconds,
            "base_url": settings.resolved_base_url,
            "organization": settings.organization,
            "headers_digest": stable_hash(settings.headers, length=24),
            "structured_output_mode": settings.resolved_structured_mode,
            "api_key_digest": stable_hash(settings.api_key, length=24),
            "usd_per_1k_prompt": settings.usd_per_1k_prompt,
            "usd_per_1k_completion": settings.usd_per_1k_completion,
        }
    )


def llm_configuration_status(settings: LLMSettings) -> LLMConfigurationStatus:
    if settings.provider is LLMProvider.OFFLINE:
        return LLMConfigurationStatus(
            state="offline",
            message="Offline mode: deterministic fallback is used; live Chat is disabled.",
            is_ready_for_live_calls=False,
            resolved_base_url="",
        )

    missing: list[str] = []
    if requires_api_key(settings.provider) and not settings.api_key:
        missing.append("api_key")
    if requires_base_url(settings.provider) and not settings.base_url:
        missing.append("base_url")

    if missing:
        return LLMConfigurationStatus(
            state="incomplete",
            message=f"Missing required LLM setting: {', '.join(missing)}.",
            is_ready_for_live_calls=False,
            missing_fields=missing,
            resolved_base_url=settings.resolved_base_url,
        )

    return LLMConfigurationStatus(
        state="ready",
        message="Configuration is ready for live LLM calls.",
        is_ready_for_live_calls=True,
        resolved_base_url=settings.resolved_base_url,
    )


def build_generation_controls(settings: LLMSettings) -> dict[str, float | int]:
    profile = provider_request_profile(settings.provider)
    capability = agent_model_profile(settings.provider, settings.model)
    controls: dict[str, float | int] = {
        profile.output_token_param: settings.max_tokens,
    }
    if profile.send_temperature and (
        capability is None or capability.temperature_policy != "omit"
    ):
        controls["temperature"] = settings.temperature
    return controls


def build_structured_chat_payload(
    settings: LLMSettings,
    *,
    task: str,
    schema: type[BaseModel],
    payload: dict,
) -> dict:
    mode = settings.resolved_structured_mode
    raw_schema = schema.model_json_schema()
    strict_schema = to_strict_json_schema(raw_schema)
    system = (
        "You are an evidence-grounded EDA reporting agent. "
        "Return only JSON that matches the provided schema. Do not add commentary."
    )
    body: dict[str, Any] = {
        "model": settings.model,
        **build_generation_controls(settings),
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps({"task": task, "payload": payload}, ensure_ascii=False),
            },
        ],
    }
    if mode == "json_schema":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": strict_schema,
            },
        }
    else:
        # json_object mode: broadly compatible; embed the schema in the prompt so
        # the model still targets the exact shape, then Pydantic validates.
        body["response_format"] = {"type": "json_object"}
        body["messages"][0]["content"] = (
            f"{system} The JSON MUST validate against this JSON Schema:\n"
            f"{json.dumps(strict_schema, ensure_ascii=False)}"
        )
    return body


def to_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a Pydantic JSON Schema satisfy OpenAI strict structured-output rules."""
    return _strict_node(deepcopy(schema))


_DROP_KEYS = {
    "default",
    "format",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "pattern",
    "multipleOf",
}


def _strict_node(node: Any) -> Any:
    if isinstance(node, dict):
        for key in list(node.keys()):
            if key in _DROP_KEYS:
                del node[key]
        for defs_key in ("$defs", "definitions"):
            if defs_key in node and isinstance(node[defs_key], dict):
                node[defs_key] = {name: _strict_node(sub) for name, sub in node[defs_key].items()}
        if "properties" in node and isinstance(node["properties"], dict):
            node["properties"] = {
                name: _strict_node(sub) for name, sub in node["properties"].items()
            }
            node["required"] = list(node["properties"].keys())
            node["additionalProperties"] = False
        for combinator in ("anyOf", "oneOf", "allOf"):
            if combinator in node and isinstance(node[combinator], list):
                node[combinator] = [_strict_node(sub) for sub in node[combinator]]
        if "items" in node:
            node["items"] = _strict_node(node["items"])
        return node
    if isinstance(node, list):
        return [_strict_node(item) for item in node]
    return node


def _coerce_int(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _extract_cached_tokens(usage_raw: dict, prompt_tokens: int) -> int:
    """Cache-read tokens from an OpenAI-compatible usage block, clamped to prompt_tokens."""
    cached = 0
    details = usage_raw.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = _coerce_int(details.get("cached_tokens"))
    if cached <= 0:  # DeepSeek mirrors it at the top level
        cached = _coerce_int(usage_raw.get("prompt_cache_hit_tokens"))
    return max(0, min(cached, prompt_tokens))


def _estimate_cost(settings: LLMSettings, usage: LLMUsage) -> tuple[float | None, str]:
    """Cost plus the basis it came from — a user override and a shipped table
    are not equally trustworthy, and neither is a bill."""
    prompt_rate = settings.usd_per_1k_prompt
    completion_rate = settings.usd_per_1k_completion
    cost_basis = "configured_rates"
    if prompt_rate <= 0 and completion_rate <= 0:
        prompt_rate, completion_rate = _default_rates_per_1k(settings)
        cost_basis = "registry_estimate"
    if prompt_rate <= 0 and completion_rate <= 0:
        return None, "unavailable"
    cached = max(0, min(usage.cached_tokens, usage.prompt_tokens))
    # Anthropic reports cache writes outside cached_tokens but inside the prompt
    # total, and they carry a premium rather than a discount.
    cache_creation = max(0, min(usage.cache_creation_tokens, usage.prompt_tokens - cached))
    fresh_prompt = usage.prompt_tokens - cached - cache_creation
    # A user override supplies ordinary input/output rates only, so cache
    # components cannot be claimed exact under it — fall back to the plain input
    # rate rather than applying a published rate to a private contract.
    catalog = cost_basis == "registry_estimate"
    listed_read = cache_read_price_per_1m(settings.provider, settings.model)
    listed_write = cache_write_price_per_1m(settings.provider, settings.model)
    cache_rate = listed_read / 1000 if catalog and listed_read is not None else prompt_rate
    write_rate = listed_write / 1000 if catalog and listed_write is not None else prompt_rate
    return round(
        fresh_prompt / 1000 * prompt_rate
        + cached / 1000 * cache_rate
        + cache_creation / 1000 * write_rate
        + usage.completion_tokens / 1000 * completion_rate,
        6,
    ), cost_basis


def _pricing_version(cost_basis: str) -> str:
    if cost_basis == "registry_estimate":
        return PRICING_CATALOG_VERSION
    if cost_basis == "configured_rates":
        return "runtime_override"
    return ""


def _extract_reasoning_tokens(usage_raw: dict, completion_tokens: int) -> int:
    details = usage_raw.get("completion_tokens_details")
    reasoning = _coerce_int(details.get("reasoning_tokens")) if isinstance(details, dict) else 0
    return max(0, min(reasoning, completion_tokens))


def _response_request_id(headers: Any) -> str:
    for key in ("x-request-id", "request-id", "x-ms-request-id"):
        value = headers.get(key) if headers is not None else None
        if value:
            return str(value)
    return ""


def _response_id(response: dict) -> str:
    value = response.get("id") if isinstance(response, dict) else None
    return value if isinstance(value, str) else ""


def _response_model(response: dict) -> str:
    """The model the provider actually served, which an alias or a routing
    layer can make different from the one configured."""
    value = response.get("model") if isinstance(response, dict) else None
    return value if isinstance(value, str) else ""


def _openai_finish_reason(response: dict) -> str:
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    value = choices[0].get("finish_reason")
    return value if isinstance(value, str) else ""


def _default_rates_per_1k(settings: LLMSettings) -> tuple[float, float]:
    rates_per_1m = pricing_per_1m(settings.provider, settings.model)
    if rates_per_1m is not None:
        return (rates_per_1m[0] / 1000, rates_per_1m[1] / 1000)
    return (0.0, 0.0)


def _message_content(response: dict) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM response did not contain choices.")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError("LLM response choice is malformed.")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("LLM response message is malformed.")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Some providers return content as a list of parts.
        texts = [part.get("text", "") for part in content if isinstance(part, dict)]
        joined = "".join(texts)
        if joined:
            return joined
    raise RuntimeError("LLM response message did not contain text content.")
