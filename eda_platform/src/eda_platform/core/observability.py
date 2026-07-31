"""Optional OpenInference/OTel span export mirroring TraceEvents."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from eda_platform.schemas.sessions import TraceEvent

MirrorResult = Literal["disabled", "mirrored", "error"]

# Mirrors the spend-ledger event name (core.llm_ledger.LLM_USAGE_EVENT); kept as
# a literal so this optional-dependency module stays import-light.
_LLM_EVENT_TYPES = {"llm_call", "llm_usage"}
_TOOL_EVENT_TYPES = {"tool_completed", "tool_failed", "run_sql", "code_agent_attempt"}


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
    """Mirror one persisted TraceEvent into an OpenInference span. No-op when off."""
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
    span = exporter.tracer.start_span(
        str(fields.get("name", fields.get("event_type", "event"))),
        context=ctx,
        start_time=start,
    )
    try:
        _set_event_attributes(span, fields)
    finally:
        span.end(end_time=_epoch_nanos(fields.get("finished_at")) or start)


def _run_root(exporter: _Exporter, session_id: str) -> Any:
    root = exporter.run_roots.get(session_id)
    if root is None:
        root = exporter.tracer.start_span(f"run:{session_id}")
        _safe_set(root, "openinference.span.kind", "CHAIN")
        _safe_set(root, "eda.session_id", session_id)
        exporter.run_roots[session_id] = root
    return root


def _set_event_attributes(span: Any, fields: dict[str, Any]) -> None:
    event_type = str(fields.get("event_type", ""))
    summary = fields.get("summary") or {}
    _safe_set(span, "eda.event_type", event_type)
    _safe_set(span, "eda.session_id", str(fields.get("session_id", "")))

    if event_type in _LLM_EVENT_TYPES:
        _safe_set(span, "openinference.span.kind", "LLM")
        _copy_attr(span, summary, "model", "llm.model_name")
        _copy_attr(span, summary, "total_tokens", "llm.token_count.total")
        _copy_attr(span, summary, "prompt_tokens", "llm.token_count.prompt")
        _copy_attr(span, summary, "completion_tokens", "llm.token_count.completion")
        _copy_attr(span, summary, "estimated_cost_usd", "eda.estimated_cost_usd")
    elif event_type in _TOOL_EVENT_TYPES:
        _safe_set(span, "openinference.span.kind", "TOOL")
    else:
        _safe_set(span, "openinference.span.kind", "CHAIN")


def _copy_attr(span: Any, summary: dict[str, Any], src_key: str, attr: str) -> None:
    value = summary.get(src_key)
    if value is not None:
        _safe_set(span, attr, value)


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
    """Close a run's root span so the exported tree is complete for that run."""
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
