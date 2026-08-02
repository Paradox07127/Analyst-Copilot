"""Single adapter seam mapping platform TraceEvents onto OTel GenAI spans.

Journal/trace files stay the source of truth; this mirror is optional,
off by default, and redacts content by default.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from eda_platform.schemas.sessions import TraceEvent

MirrorResult = Literal["disabled", "mirrored", "error"]

# --- platform event families -> GenAI operations --------------------------- #
# "llm_usage" mirrors core.llm_ledger.LLM_USAGE_EVENT; kept as a literal so
# this optional-dependency module stays import-light.
_CHAT_EVENT_TYPES = {"llm_call", "llm_error"}
_USAGE_EVENT_TYPES = {"llm_usage"}
_TOOL_EVENT_TYPES = {"tool_completed", "tool_failed", "run_sql", "code_agent_attempt"}
_EVALUATION_EVENT_TYPES = {
    "evaluation",
    "evaluation_result",
    "validator_result",
    "report_validation",
}

# Summary keys allowed into span attributes verbatim. Everything else is
# treated as content (prompts, tool results, cell values) and reduced to
# chars + digest unless EDA_LLM_DEBUG_FULL opts in.
_METADATA_KEYS = frozenset(
    {
        "provider",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "cache_creation_tokens",
        "reasoning_tokens",
        "cache_hit_rate",
        "estimated_cost_usd",
        "cost_basis",
        "pricing_version",
        "provider_usage_reported",
        "usage_reported",
        "request_id",
        "response_id",
        "finish_reason",
        "endpoint_host",
        "request_bytes",
        "response_bytes",
        "duration_s",
        "dataset",
        "question_id",
        "tool",
        "tool_call_id",
        "source",
        "score",
        "passed",
    }
)
_METRIC_SUFFIXES = ("_count", "_tokens", "_bytes")

# Platform summary key -> semconv attribute. gen_ai.* limited to the stable
# subset; cached/reasoning tokens, cost, and evaluation attrs stay under
# eda.* until the GenAI conventions for them stabilise.
_ATTR_NAMES = {
    "model": "gen_ai.request.model",
    "provider": "gen_ai.provider.name",
    "prompt_tokens": "gen_ai.usage.input_tokens",
    "completion_tokens": "gen_ai.usage.output_tokens",
    "response_id": "gen_ai.response.id",
    "tool": "gen_ai.tool.name",
    "tool_call_id": "gen_ai.tool.call.id",
    "score": "eda.evaluation.score",
    "passed": "eda.evaluation.passed",
}

_FULL_CAPTURE_ENV = "EDA_LLM_DEBUG_FULL"  # same opt-in as core.dev_log
_CONTENT_EXCERPT_CHARS = 2_000


@dataclass(frozen=True)
class ObservabilityConfig:
    enabled: bool = False
    endpoint: str | None = None
    service_name: str = "eda-agent-platform"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> ObservabilityConfig:
        source = env if env is not None else os.environ
        enabled = source.get("EDA_OBSERVABILITY", "").strip().lower() in {"1", "true", "yes", "on"}
        endpoint = (
            source.get("PHOENIX_COLLECTOR_ENDPOINT")
            or source.get("OTEL_EXPORTER_OTLP_ENDPOINT")
            or None
        )
        return cls(enabled=enabled, endpoint=endpoint)


@dataclass
class _Exporter:
    """Holds the process-wide tracer provider once observability is turned on."""

    provider: Any
    tracer: Any
    run_roots: dict[str, Any] = field(default_factory=dict)


# Process-wide, lazily built. ``None`` means "not yet resolved this process".
_STATE: _Exporter | None = None
_FORCED: _Exporter | None = None  # test injection wins over env resolution


@lru_cache(maxsize=1)
def _env_config() -> ObservabilityConfig:
    return ObservabilityConfig.from_env()


def _active_exporter() -> _Exporter | None:
    """Return the live exporter, building it from env on first use, or ``None``."""
    global _STATE
    if _FORCED is not None:
        return _FORCED
    if not _env_config().enabled:
        return None
    if _STATE is None:
        _STATE = _build_exporter(_env_config())
    return _STATE


def _build_exporter(config: ObservabilityConfig) -> _Exporter | None:
    try:
        from opentelemetry.sdk.resources import Resource  # pyright: ignore[reportMissingImports]
        from opentelemetry.sdk.trace import TracerProvider  # pyright: ignore[reportMissingImports]
        from opentelemetry.sdk.trace.export import (  # pyright: ignore[reportMissingImports]
            BatchSpanProcessor,
        )

        provider = TracerProvider(resource=Resource.create({"service.name": config.service_name}))
        if config.endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # pyright: ignore[reportMissingImports]
                OTLPSpanExporter,
            )

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=config.endpoint))
            )
        tracer = provider.get_tracer("eda_platform.observability")
        return _Exporter(provider=provider, tracer=tracer)
    except Exception:
        # Missing extra or a broken collector must never break the run.
        return None


def mirror_trace_event(event: TraceEvent | dict[str, Any]) -> MirrorResult:
    """The adapter seam: mirror one platform event into a GenAI span. No-op when off."""
    exporter = _active_exporter()
    if exporter is None:
        return "disabled"
    try:
        _emit_span(exporter, _as_fields(event))
    except Exception:
        return "error"
    return "mirrored"


def _as_fields(event: TraceEvent | dict[str, Any]) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    return {
        "session_id": event.session_id,
        "event_type": event.event_type,
        "name": event.name,
        "call_id": event.call_id,
        "trial_id": event.trial_id,
        "investigation_id": event.investigation_id,
        "attempt_id": event.attempt_id,
        "started_at": event.started_at,
        "finished_at": event.finished_at,
        "summary": dict(event.summary),
    }


def _emit_span(exporter: _Exporter, fields: dict[str, Any]) -> None:
    from opentelemetry import trace as otel_trace  # pyright: ignore[reportMissingImports]

    session_id = str(fields.get("session_id", "unknown"))
    root = _run_root(exporter, session_id)
    ctx = otel_trace.set_span_in_context(root)

    start = _epoch_nanos(fields.get("started_at"))
    span = exporter.tracer.start_span(_span_name(fields), context=ctx, start_time=start)
    try:
        _set_event_attributes(span, fields)
    finally:
        span.end(end_time=_epoch_nanos(fields.get("finished_at")) or start)


def _run_root(exporter: _Exporter, session_id: str) -> Any:
    """One invoke_agent root span per run; children nest under it."""
    root = exporter.run_roots.get(session_id)
    if root is None:
        root = exporter.tracer.start_span(f"invoke_agent {session_id}")
        _safe_set(root, "openinference.span.kind", "AGENT")
        _safe_set(root, "gen_ai.operation.name", "invoke_agent")
        _safe_set(root, "gen_ai.conversation.id", session_id)
        _safe_set(root, "eda.session_id", session_id)
        exporter.run_roots[session_id] = root
    return root


def _span_name(fields: dict[str, Any]) -> str:
    event_type = str(fields.get("event_type", ""))
    name = str(fields.get("name", "") or event_type or "event")
    summary = fields.get("summary") or {}
    if event_type in _CHAT_EVENT_TYPES:
        model = str(summary.get("model") or "").strip()
        return f"chat {model}" if model else "chat"
    if event_type in _TOOL_EVENT_TYPES:
        return f"execute_tool {str(summary.get('tool') or name).strip()}"
    if event_type in _EVALUATION_EVENT_TYPES:
        return f"evaluation {name}"
    return name


def _set_event_attributes(span: Any, fields: dict[str, Any]) -> None:
    event_type = str(fields.get("event_type", ""))
    summary = fields.get("summary") or {}
    session_id = str(fields.get("session_id", ""))
    _safe_set(span, "eda.event_type", event_type)
    _safe_set(span, "eda.session_id", session_id)
    _safe_set(span, "gen_ai.conversation.id", session_id)
    name = str(fields.get("name", ""))
    if name:
        _safe_set(span, "eda.task", name)
    for key in ("call_id", "trial_id", "investigation_id", "attempt_id"):
        value = fields.get(key)
        if value:
            _safe_set(span, f"eda.{key}", str(value))

    if event_type in _CHAT_EVENT_TYPES:
        _safe_set(span, "openinference.span.kind", "LLM")
        _safe_set(span, "gen_ai.operation.name", "chat")
        if event_type == "llm_error":
            _safe_set(span, "error.type", "llm_error")
    elif event_type in _USAGE_EVENT_TYPES:
        # Ledger settlement is a billing fact, not a second chat call, so it
        # carries usage attributes without a gen_ai.operation.name.
        _safe_set(span, "openinference.span.kind", "LLM")
    elif event_type in _TOOL_EVENT_TYPES:
        _safe_set(span, "openinference.span.kind", "TOOL")
        _safe_set(span, "gen_ai.operation.name", "execute_tool")
        _safe_set(span, "gen_ai.tool.name", str(summary.get("tool") or name))
        if event_type == "tool_failed":
            _safe_set(span, "error.type", "tool_failed")
    elif event_type in _EVALUATION_EVENT_TYPES:
        _safe_set(span, "openinference.span.kind", "EVALUATOR")
        if name:
            # Pending alignment with gen_ai.evaluation.* once those stabilise.
            _safe_set(span, "eda.evaluation.name", name)
    else:
        _safe_set(span, "openinference.span.kind", "CHAIN")

    _apply_summary(span, summary)


# --- default-deny summary mapping ------------------------------------------ #
def _apply_summary(span: Any, summary: dict[str, Any]) -> None:
    full_capture = _full_capture_enabled()
    for key, value in summary.items():
        if value is None:
            continue
        if key == "status":
            _apply_status(span, value)
        elif _is_metadata(key, value):
            _apply_metadata(span, key, value)
        else:
            _apply_redacted(span, key, value, full_capture)


def _is_metadata(key: str, value: Any) -> bool:
    if not isinstance(value, bool | int | float | str):
        return False
    if key in _METADATA_KEYS:
        return True
    return key.endswith(_METRIC_SUFFIXES) and isinstance(value, int | float)


def _apply_metadata(span: Any, key: str, value: Any) -> None:
    if isinstance(value, str) and not value.strip():
        return
    if key == "finish_reason":
        span.set_attribute("gen_ai.response.finish_reasons", [str(value)])
        return
    _safe_set(span, _ATTR_NAMES.get(key, f"eda.{key}"), value)


def _apply_status(span: Any, value: Any) -> None:
    # "error: ValueError: <message>" may embed payload text; keep only the
    # outcome token and the exception class.
    head, sep, rest = str(value).partition(":")
    _safe_set(span, "eda.status", head.strip()[:64])
    if sep:
        error_type = rest.strip().partition(":")[0].strip()
        if error_type:
            _safe_set(span, "error.type", error_type[:128])


def _apply_redacted(span: Any, key: str, value: Any, full_capture: bool) -> None:
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    )
    _safe_set(span, f"eda.redacted.{key}.chars", len(text))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    _safe_set(span, f"eda.redacted.{key}.sha256", digest)
    if full_capture:
        _safe_set(span, f"eda.content.{key}", text[:_CONTENT_EXCERPT_CHARS])


def _full_capture_enabled() -> bool:
    return os.environ.get(_FULL_CAPTURE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_set(span: Any, key: str, value: Any) -> None:
    if isinstance(value, bool):
        span.set_attribute(key, value)
    elif isinstance(value, int | float | str):
        span.set_attribute(key, value)
    else:
        span.set_attribute(key, str(value))


def _epoch_nanos(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value.timestamp() * 1_000_000_000)
    except (AttributeError, ValueError, OverflowError):
        return None


def flush_run(session_id: str) -> None:
    """Close a run's invoke_agent root span so the exported tree is complete."""
    exporter = _active_exporter()
    if exporter is None:
        return
    root = exporter.run_roots.pop(session_id, None)
    if root is not None:
        root.end()


# --------------------------------------------------------------------------- #
# Test / embedding hooks
# --------------------------------------------------------------------------- #
def configure_for_test(exporter_impl: Any) -> None:
    """Force observability on with a caller-supplied span exporter (tests)."""
    global _FORCED
    from opentelemetry.sdk.resources import Resource  # pyright: ignore[reportMissingImports]
    from opentelemetry.sdk.trace import TracerProvider  # pyright: ignore[reportMissingImports]
    from opentelemetry.sdk.trace.export import (  # pyright: ignore[reportMissingImports]
        SimpleSpanProcessor,
    )

    provider = TracerProvider(resource=Resource.create({"service.name": "eda-test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter_impl))
    _FORCED = _Exporter(provider=provider, tracer=provider.get_tracer("eda_platform.test"))


def reset() -> None:
    """Drop any forced/built exporter and cached env config (tests)."""
    global _STATE, _FORCED
    _STATE = None
    _FORCED = None
    # _env_config may be a stand-in during a test; a teardown helper must not
    # depend on it still being the lru_cache-wrapped original.
    cache_clear = getattr(_env_config, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
