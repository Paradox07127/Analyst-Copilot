"""The provider's rejection is the source of truth for its request dialect."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, Protocol
from urllib import error

import pytest

from eda_platform.core.llm import (
    LLMProvider,
    LLMSettings,
    OpenAICompatibleLLMClient,
    ToolCallingUnsupportedError,
)
from eda_platform.core.request_dialect import (
    MAX_PARAM_REPAIRS,
    ParamRepair,
    forget_learned_repairs,
    learned_repairs,
    plan_repair,
)

# Verbatim from an OpenAI 400 on a gpt-5.x model.
UNSUPPORTED_MAX_TOKENS = json.dumps(
    {
        "error": {
            "message": (
                "Unsupported parameter: 'max_tokens' is not supported with this "
                "model. Use 'max_completion_tokens' instead."
            ),
            "type": "invalid_request_error",
            "param": "max_tokens",
            "code": "unsupported_parameter",
        }
    }
)
UNSUPPORTED_TEMPERATURE = json.dumps(
    {
        "error": {
            "message": (
                "Unsupported value: 'temperature' does not support 0.2 with this "
                "model. Only the default (1) value is supported."
            ),
            "type": "invalid_request_error",
            "param": "temperature",
            "code": "unsupported_value",
        }
    }
)


@pytest.fixture(autouse=True)
def _clean_memo() -> Iterator[None]:
    forget_learned_repairs()
    yield
    forget_learned_repairs()


class _Resp:
    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}'


def _reject(detail: str) -> error.HTTPError:
    return error.HTTPError(
        url="https://api.openai.com/v1/chat/completions",
        code=400,
        msg="Bad Request",
        hdrs=None,  # type: ignore[arg-type]
        fp=None,
    )


class _Endpoint:
    """Rejects each listed detail once, in order, then accepts."""

    def __init__(self, *rejections: str, always: str | None = None) -> None:
        self.pending = list(rejections)
        self.always = always
        self.bodies: list[dict] = []

    def __call__(self, req, timeout: float = 0):  # noqa: ANN001
        self.bodies.append(json.loads(req.data.decode("utf-8")))
        detail = self.always or (self.pending.pop(0) if self.pending else None)
        if detail is None:
            return _Resp()
        exc = _reject(detail)
        exc.read = lambda: detail.encode("utf-8")  # type: ignore[method-assign]
        raise exc


def _settings(**overrides: object) -> LLMSettings:
    base = {
        "provider": LLMProvider.OPENAI_COMPATIBLE,
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-test",
        "model": "gpt-5.6-terra",
        "temperature": 0.2,
        "max_tokens": 321,
    }
    base.update(overrides)
    return LLMSettings(**base)  # type: ignore[arg-type]


class _EndpointLike(Protocol):
    bodies: list[dict]

    def __call__(self, req: Any, timeout: float = 0) -> Any: ...


def _install(monkeypatch: pytest.MonkeyPatch, endpoint: _EndpointLike) -> None:
    monkeypatch.setattr("eda_platform.core.llm.request.urlopen", endpoint)


# --- plan_repair: pure reading of a rejection ---------------------------------


def test_rename_is_read_from_the_use_x_instead_clause() -> None:
    repair = plan_repair({"max_tokens": 321}, UNSUPPORTED_MAX_TOKENS)
    assert repair == ParamRepair("rename", "max_tokens", "max_completion_tokens")


def test_unsupported_value_drops_the_parameter_rather_than_renaming_it() -> None:
    repair = plan_repair({"temperature": 0.2}, UNSUPPORTED_TEMPERATURE)
    assert repair == ParamRepair("drop", "temperature")


def test_a_rejection_naming_tools_is_not_repairable() -> None:
    detail = json.dumps(
        {"error": {"message": "Unsupported parameter: 'tools'", "param": "tools"}}
    )
    assert plan_repair({"tools": [], "max_tokens": 1}, detail) is None


def test_a_replacement_outside_the_allowlist_is_refused() -> None:
    """A provider must not be able to steer a rewrite into an arbitrary key."""
    detail = json.dumps(
        {
            "error": {
                "message": (
                    "Unsupported parameter: 'max_tokens' is not supported. "
                    "Use 'api_key' instead."
                ),
                "param": "max_tokens",
            }
        }
    )
    assert plan_repair({"max_tokens": 1}, detail) == ParamRepair("drop", "max_tokens")


def test_an_ordinary_error_is_not_mistaken_for_a_dialect_problem() -> None:
    detail = json.dumps({"error": {"message": "Incorrect API key provided."}})
    assert plan_repair({"max_tokens": 1}, detail) is None


def test_a_parameter_absent_from_the_body_is_never_repaired() -> None:
    assert plan_repair({"max_completion_tokens": 1}, UNSUPPORTED_MAX_TOKENS) is None


# --- wired into the client ----------------------------------------------------


def test_client_renames_the_token_param_and_retries_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _Endpoint(UNSUPPORTED_MAX_TOKENS)
    _install(monkeypatch, endpoint)

    OpenAICompatibleLLMClient(_settings()).text(task="t", payload={})

    assert len(endpoint.bodies) == 2
    assert endpoint.bodies[0]["max_tokens"] == 321
    assert "max_tokens" not in endpoint.bodies[1]
    assert endpoint.bodies[1]["max_completion_tokens"] == 321


def test_two_rejections_in_a_row_are_both_repaired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _Endpoint(UNSUPPORTED_MAX_TOKENS, UNSUPPORTED_TEMPERATURE)
    _install(monkeypatch, endpoint)

    OpenAICompatibleLLMClient(_settings()).text(task="t", payload={})

    assert len(endpoint.bodies) == 3
    assert endpoint.bodies[2]["max_completion_tokens"] == 321
    assert "temperature" not in endpoint.bodies[2]


def test_an_unrepairable_rejection_surfaces_as_the_provider_wrote_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _Endpoint(always=UNSUPPORTED_TEMPERATURE)
    _install(monkeypatch, endpoint)

    with pytest.raises(RuntimeError, match="HTTP 400"):
        OpenAICompatibleLLMClient(_settings()).text(task="t", payload={})

    # One repairable attempt, then the repair, then it gives up: temperature is
    # gone from the body, so the same complaint is no longer attributable.
    assert len(endpoint.bodies) == 2


class _PingPong:
    """A provider that always demands the token param it was just given.

    Without a repair ceiling this rewrites forever, one paid round trip each.
    """

    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def __call__(self, req, timeout: float = 0):  # noqa: ANN001
        body = json.loads(req.data.decode("utf-8"))
        self.bodies.append(body)
        held = "max_tokens" if "max_tokens" in body else "max_completion_tokens"
        wanted = "max_completion_tokens" if held == "max_tokens" else "max_tokens"
        detail = json.dumps(
            {
                "error": {
                    "message": (
                        f"Unsupported parameter: '{held}' is not supported with "
                        f"this model. Use '{wanted}' instead."
                    ),
                    "param": held,
                }
            }
        )
        exc = _reject(detail)
        exc.read = lambda: detail.encode("utf-8")  # type: ignore[method-assign]
        raise exc


def test_repair_cannot_loop_forever(monkeypatch: pytest.MonkeyPatch) -> None:
    endpoint = _PingPong()
    _install(monkeypatch, endpoint)

    with pytest.raises(RuntimeError, match="kept rejecting"):
        OpenAICompatibleLLMClient(_settings()).text(task="t", payload={})

    # The ceiling itself is the guard, so its size is asserted rather than
    # read from the constant: every repair is another paid round trip.
    assert MAX_PARAM_REPAIRS <= 3
    assert len(endpoint.bodies) == MAX_PARAM_REPAIRS + 1


def test_the_learned_rename_is_applied_before_the_next_call_is_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    first = _Endpoint(UNSUPPORTED_MAX_TOKENS)
    _install(monkeypatch, first)
    OpenAICompatibleLLMClient(settings).text(task="t", payload={})

    second = _Endpoint()
    _install(monkeypatch, second)
    OpenAICompatibleLLMClient(settings).text(task="t", payload={})

    assert len(second.bodies) == 1, "a learned dialect must not be re-discovered"
    assert second.bodies[0]["max_completion_tokens"] == 321
    assert learned_repairs(settings.provider, settings.model) == (
        ParamRepair("rename", "max_tokens", "max_completion_tokens"),
    )


def test_a_repair_is_recorded_on_the_call_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silent self-healing would hide a stale catalog entry forever."""
    _install(monkeypatch, _Endpoint(UNSUPPORTED_MAX_TOKENS))
    client = OpenAICompatibleLLMClient(_settings())
    client.text(task="t", payload={})

    usage = client.last_usage()
    assert usage is not None
    assert usage.param_repairs == ["max_tokens->max_completion_tokens"]


def test_a_learned_repair_keeps_reporting_itself_on_later_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    _install(monkeypatch, _Endpoint(UNSUPPORTED_MAX_TOKENS))
    OpenAICompatibleLLMClient(settings).text(task="t", payload={})

    _install(monkeypatch, _Endpoint())
    later = OpenAICompatibleLLMClient(settings)
    later.text(task="t", payload={})

    usage = later.last_usage()
    assert usage is not None
    assert usage.param_repairs == ["max_tokens->max_completion_tokens"]


def test_a_rejection_naming_tools_becomes_the_capability_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detail = json.dumps(
        {"error": {"message": "This model does not support tools.", "param": "tools"}}
    )
    _install(monkeypatch, _Endpoint(always=detail))

    with pytest.raises(ToolCallingUnsupportedError):
        OpenAICompatibleLLMClient(_settings()).tool_call(
            task="t",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "probe", "parameters": {"type": "object"}}],
        )


def test_an_unrelated_rejection_is_not_a_capability_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Degrading on any error would turn a bad key into a worse analysis."""
    detail = json.dumps({"error": {"message": "Incorrect API key provided."}})
    _install(monkeypatch, _Endpoint(always=detail))

    with pytest.raises(RuntimeError) as caught:
        OpenAICompatibleLLMClient(_settings()).tool_call(
            task="t",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "probe", "parameters": {"type": "object"}}],
        )
    assert not isinstance(caught.value, ToolCallingUnsupportedError)


def test_a_clean_call_reports_no_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _Endpoint())
    client = OpenAICompatibleLLMClient(_settings())
    client.text(task="t", payload={})

    usage = client.last_usage()
    assert usage is not None
    assert usage.param_repairs == []
