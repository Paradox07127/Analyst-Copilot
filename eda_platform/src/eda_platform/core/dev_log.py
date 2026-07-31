"""Developer log: live event formatting + LLM call capture."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from eda_platform.core.llm import LLMClient, LLMResultMetadata
    from eda_platform.schemas.sessions import TraceEvent

T = TypeVar("T")

LLM_DEBUG_FILENAME = "llm_debug.jsonl"
PREVIEW_CHARS = 2_000
_FULL_CAPTURE_ENV = "EDA_LLM_DEBUG_FULL"

# Summary keys that carry the metrics developers scan for; anything else in a
# summary is elided from the one-line format (full payloads stay in trace.jsonl).
_METRIC_KEYS = (
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "estimated_cost_usd",
    "candidate_count",
    "selected_count",
    "proposed_count",
    "resolved_count",
    "skipped_count",
    "artifact_count",
    "status",
    "dataset",
    "question_id",
    "model",
)


def _clock(event_time: datetime) -> str:
    return event_time.strftime("%H:%M:%S") + f".{event_time.microsecond // 1000:03d}"


def format_event_line(event: TraceEvent) -> str:
    """One fixed-column log line: ``HH:MM:SS.mmm  event_type        name  k=v ...  [dur]``."""
    duration = ""
    if event.finished_at is not None:
        elapsed = (event.finished_at - event.started_at).total_seconds()
        duration = f"{elapsed:.2f}s" if elapsed >= 0.005 else ""
    details: list[str] = []
    for key in _METRIC_KEYS:
        value = event.summary.get(key)
        if value is None or value == "" or value == 0:
            continue
        if key == "estimated_cost_usd":
            details.append(f"cost=${value}")
        else:
            details.append(f"{key.removeprefix('estimated_')}={value}")
    line = (
        f"{_clock(event.started_at)}  {event.event_type:<24.24}  "
        f"{event.name:<22.22}  {' '.join(details)}"
    ).rstrip()
    if duration:
        line = f"{line}  [{duration}]"
    return line


@dataclass
class DevLogBuffer:
    """Ring buffer of formatted log lines with a flush throttle."""

    max_lines: int = 200
    flush_every: int = 8
    flush_interval: float = 0.3
    lines: list[str] = field(default_factory=list)
    _pending: int = 0
    # ``None`` is the only unambiguous sentinel for an untouched buffer: a
    # monotonic clock starts at an arbitrary process/host-relative value, so
    # using ``0.0`` can suppress the first render on a freshly started runner.
    _last_flush: float | None = None

    def add(self, event: TraceEvent) -> bool:
        self.lines.append(format_event_line(event))
        if len(self.lines) > self.max_lines:
            del self.lines[: len(self.lines) - self.max_lines]
        self._pending += 1
        now = time.monotonic()
        if (
            self._last_flush is None
            or self._pending >= self.flush_every
            or (now - self._last_flush) >= self.flush_interval
        ):
            self._pending = 0
            self._last_flush = now
            return True
        return False

    def render(self) -> str:
        return "\n".join(self.lines)


def _preview(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if os.environ.get(_FULL_CAPTURE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        return text
    if len(text) > PREVIEW_CHARS:
        return text[:PREVIEW_CHARS] + f"… [truncated {len(text) - PREVIEW_CHARS} chars]"
    return text


def _serialized_bytes(value: Any) -> int:
    if value is None:
        return 0
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    try:
        text = (
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        )
    except Exception:  # noqa: BLE001 — diagnostics must not break a call
        text = str(value)
    return len(text.encode("utf-8"))


def _transport_kind(settings: Any, requested_kind: str) -> str:
    provider = getattr(settings, "provider", None)
    provider_value = getattr(provider, "value", provider)
    if requested_kind == "structured" and provider_value == "anthropic":
        return "provider_tool"
    return requested_kind


class InstrumentedLLMClient:
    """Wraps any LLMClient; records every call to memory + ``llm_debug.jsonl``."""

    def __init__(self, inner: LLMClient, *, session_dir: Path | None = None) -> None:
        self._inner = inner
        self._run_dir = session_dir
        self.records: list[dict[str, Any]] = []

    # -- passthroughs -------------------------------------------------------
    @property
    def inner(self) -> LLMClient:
        """Expose the wrapped client so capability checks can unwrap decorators."""
        return self._inner

    @property
    def settings(self) -> Any:  # reporting reads .settings for completion caps
        return getattr(self._inner, "settings", None)

    def last_usage(self) -> LLMResultMetadata | None:
        return self._inner.last_usage()

    # -- instrumented calls -------------------------------------------------
    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        return self._call("structured", task=task, payload=payload, schema=schema)

    def text(self, *, task: str, payload: dict) -> str:
        return self._call("text", task=task, payload=payload, schema=None)

    def tool_call(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        return self._call(
            "tool_call",
            task=task,
            payload={"messages": messages, "tools": tools},
            schema=None,
        )

    def _call(self, kind: str, *, task: str, payload: dict, schema: type | None) -> Any:
        started = time.monotonic()
        status = "success"
        response: Any = None
        # Snapshot the inner client's usage marker up front: providers only
        # refresh it on success, so after a FAILED call last_usage() still
        # holds the PREVIOUS call's tokens/cost — attaching that to an error
        # record would mislead the exact panel built for debugging.
        try:
            usage_before = self._inner.last_usage()
        except Exception:  # noqa: BLE001 — metering must never break the call
            usage_before = None
        try:
            if kind == "structured":
                assert schema is not None
                response = self._inner.structured(task=task, schema=schema, payload=payload)
            elif kind == "tool_call":
                response = cast(Any, self._inner).tool_call(
                    task=task,
                    messages=list(payload.get("messages", [])),
                    tools=list(payload.get("tools", [])),
                )
            else:
                response = self._inner.text(task=task, payload=payload)
            return response
        except Exception as exc:
            status = f"error: {type(exc).__name__}: {exc}"
            raise
        finally:
            self._record(
                kind=kind,
                task=task,
                payload=payload,
                response=response,
                status=status,
                duration_s=round(time.monotonic() - started, 3),
                usage_before=usage_before,
            )

    def _record(
        self,
        *,
        kind: str,
        task: str,
        payload: dict,
        response: Any,
        status: str,
        duration_s: float,
        usage_before: LLMResultMetadata | None,
    ) -> None:
        usage = None
        try:
            usage = self._inner.last_usage()
        except Exception:  # noqa: BLE001 — metering must never break the call
            usage = None
        # Only attach usage the call itself produced (see _call): if the
        # marker did not move, this call never got far enough to be metered.
        if usage is not None and usage is usage_before:
            usage = None
        record: dict[str, Any] = {
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "kind": kind,
            "transport_kind": _transport_kind(self.settings, kind),
            "task": task,
            "status": status,
            "duration_s": duration_s,
            "request_bytes": _serialized_bytes({"task": task, "payload": payload}),
            "response_bytes": _serialized_bytes(response),
            "payload_preview": _preview(payload),
            "response_preview": _preview(
                response.model_dump(mode="json") if hasattr(response, "model_dump") else response
            )
            if response is not None
            else "",
        }
        if usage is not None:
            record.update(
                {
                    "provider": usage.provider,
                    "model": usage.model,
                    "prompt_tokens": usage.usage.prompt_tokens,
                    "completion_tokens": usage.usage.completion_tokens,
                    "total_tokens": usage.usage.total_tokens,
                    "cached_tokens": usage.usage.cached_tokens,
                    "cache_creation_tokens": usage.usage.cache_creation_tokens,
                    "reasoning_tokens": usage.usage.reasoning_tokens,
                    "cache_hit_rate": usage.usage.cache_hit_rate,
                    "estimated_cost_usd": usage.estimated_cost_usd,
                    "cost_basis": usage.cost_basis,
                    "pricing_version": usage.pricing_version,
                    "usage_reported": usage.usage_reported,
                    "request_id": usage.request_id,
                    "response_id": usage.response_id,
                    "finish_reason": usage.finish_reason,
                    "endpoint_host": usage.endpoint_host,
                    "request_bytes": usage.request_bytes or record["request_bytes"],
                    "response_bytes": usage.response_bytes or record["response_bytes"],
                }
            )
        self.records.append(record)
        if self._run_dir is None:
            return
        try:
            from eda_platform.core.store import ArtifactStore

            runs_dir = self._run_dir.parent
            project_dir = runs_dir.parent
            workspace = project_dir.parent.parent
            ArtifactStore(workspace, init_db=False).append_session_line(
                project_dir.name,
                self._run_dir.name,
                LLM_DEBUG_FILENAME,
                json.dumps(record, ensure_ascii=False),
            )
        except (OSError, RuntimeError, ValueError):
            pass  # debug capture must never fail a run


def read_llm_debug(session_dir: Path) -> list[dict[str, Any]]:
    """Replay-path reader; tolerant of a missing or partially written file."""
    path = Path(session_dir) / LLM_DEBUG_FILENAME
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return records


