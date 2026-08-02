"""Trace & cost read use cases (§10.1 Trace): the metrics rollup comes from the
run's persisted SessionMetrics artifact when it exists and is otherwise recomputed
from trace events; the event feed is a pure SQL page over the ``trace_events``
table (never trace.jsonl).

The developer-inspector reads (debug rollup, debug.jsonl download, captured LLM
payloads) reuse the pure row builders in ``tools.debug``."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from eda_platform.application.dto import (
    ClientFailureRecorded,
    ClientFailureRequest,
    DebugArtifactRow,
    DebugErrorRow,
    DebugLlmCallRow,
    DebugTimelineRow,
    DebugToolCallRow,
    LlmDebugRecord,
    Page,
    ReportQualitySummary,
    SessionDebugSummary,
    SessionDebugView,
    SessionMetricsView,
    TraceEventPage,
    TraceEventRow,
    UsageDay,
    UsageRecentSession,
    WorkspaceUsageView,
)
from eda_platform.application.services.session_service import (
    InvalidCursorError,
    SessionNotFoundError,
)
from eda_platform.application.workspace_paths import relativize_workspace_paths
from eda_platform.core.bounded_pagination import (
    JsonlPageIndex,
    decode_bound_cursor,
    encode_bound_cursor,
)
from eda_platform.core.debug_log import DEBUG_LOG_FILENAME
from eda_platform.core.dev_log import LLM_DEBUG_FILENAME
from eda_platform.core.env import API_KEY_ENV_VARS, DEFAULT_ENV_PATH, parse_env_file
from eda_platform.core.ids import (
    DERIVED_SESSION_PREFIXES,
    INTERNAL_SESSION_MARKER,
    is_internal_project_id,
)
from eda_platform.core.llm_ledger import (
    BUDGET_EVENT_TYPES,
    BUDGET_REJECTED_EVENT,
    BUDGET_RESERVED_EVENT,
    BUDGET_SETTLED_EVENT,
    LLM_USAGE_EVENT,
)
from eda_platform.core.session_metrics import spend_events, summarize_session
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.session_metrics import SessionMetrics
from eda_platform.schemas.sessions import TraceEvent
from eda_platform.tools.debug import (
    artifact_rows,
    error_rows,
    llm_call_rows,
    timeline_rows,
    tool_call_rows,
)

logger = logging.getLogger(__name__)

DEFAULT_TRACE_LIMIT = 100
MAX_TRACE_LIMIT = 500

# The usage rollup counts every session, so it reads a project's runs in one
# unpaged sweep. A local-first workspace does not reach this; the cap only
# stops one runaway project from turning the home page into a full-table read.
_USAGE_SESSION_SCAN_LIMIT = 2000
DEFAULT_USAGE_WINDOW_DAYS = 180
MAX_USAGE_WINDOW_DAYS = 366
_USAGE_RECENT_LIMIT = 8
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _utc_day(timestamp: object) -> str:
    """Session index timestamps are RFC 3339 strings; bucket by UTC date."""
    if not isinstance(timestamp, str) or not timestamp:
        return ""
    try:
        return datetime.fromisoformat(timestamp).astimezone(UTC).date().isoformat()
    except ValueError:
        return ""


DEFAULT_LLM_DEBUG_LIMIT = 50
MAX_LLM_DEBUG_LIMIT = 200
DEFAULT_SESSION_DEBUG_LIMIT = 100
MAX_SESSION_DEBUG_LIMIT = 250
CLIENT_FAILURE_EVENT_TYPE = "failure_recorded"
CLIENT_FAILURE_RATE_LIMIT = 20
CLIENT_FAILURE_RATE_WINDOW_SECONDS = 60
_DEBUG_ARTIFACT_READ_BYTES = 5 * 1024 * 1024
_MIN_REDACTABLE_KEY_CHARS = 8


class DebugLogNotFoundError(Exception):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"No debug log was recorded for run {session_id}.")
        self.session_id = session_id


class ClientFailureRateLimitError(Exception):
    pass


class ClientFailureTooLargeError(Exception):
    pass


@dataclass(frozen=True)
class DebugLogDownload:
    """A debug.jsonl proven to live inside its own run directory."""

    path: Path
    filename: str
    media_type: str
    byte_size: int
    replacements: tuple[tuple[bytes, bytes], ...]

    def chunks(self) -> Iterator[bytes]:
        """Stream and redact a fixed-size snapshot one JSONL record at a time."""
        yield from _stream_redacted(
            self.path,
            byte_limit=self.byte_size,
            replacements=self.replacements,
        )


class TraceService:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def workspace_usage(self, *, days: int) -> WorkspaceUsageView:
        """Roll sessions active in ``days`` into figures for the home page.

        Deliberately reads only *persisted* metrics artifacts: falling back to
        ``summarize_session`` per session would make opening the home page a full
        trace scan of the whole workspace. Sessions with no artifact are
        reported as ``unpriced_sessions`` instead of being recomputed or, worse,
        silently counted as zero cost. Project count, recent work, and current
        upload storage remain workspace-wide because they are not period totals.
        """
        now = datetime.now(UTC)
        window = [
            (now - timedelta(days=offset)).date().isoformat() for offset in range(days - 1, -1, -1)
        ]
        per_day = dict.fromkeys(window, 0)
        status_counts: dict[str, int] = {}
        sessions = llm_calls = tokens = priced = unpriced = artifacts = 0
        profiled_datasets = profiled_rows = 0
        projects = truncated = 0
        cost = 0.0
        recent: list[UsageRecentSession] = []

        # One enumeration, not two: counting projects in a second pass let
        # project_count disagree with the session totals computed in the first.
        for project_row in self._store.project_index_rows(
            exclude_session_id_containing=INTERNAL_SESSION_MARKER,
            exclude_session_id_prefixes=DERIVED_SESSION_PREFIXES,
        ):
            project_id = project_row["project_id"]
            # A standalone session lives in an internal bucket so the
            # project-scoped filesystem and quotas still apply, but it is not a
            # project: it must not raise project_count, and its spend is real
            # money that has to reach these totals.
            if not is_internal_project_id(project_id):
                projects += 1
            rows = self._store.query_session_index_rows(
                project_id,
                limit=_USAGE_SESSION_SCAN_LIMIT + 1,
                exclude_session_id_containing=INTERNAL_SESSION_MARKER,
                exclude_session_id_prefixes=DERIVED_SESSION_PREFIXES,
            )
            if len(rows) > _USAGE_SESSION_SCAN_LIMIT:
                # Reported rather than swallowed: these figures are presented as
                # workspace totals, and a silently truncated total is worse than
                # one that says it is partial.
                truncated += len(rows) - _USAGE_SESSION_SCAN_LIMIT
                rows = rows[:_USAGE_SESSION_SCAN_LIMIT]
            for row in rows:
                status = row["status"] or "unknown"
                last_activity = row["updated_at"] or row["created_at"]
                day = _utc_day(last_activity)
                recent.append(
                    UsageRecentSession(
                        session_id=row["session_id"],
                        project_id=project_id,
                        title=row["title"],
                        status=status,
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                )
                if day not in per_day:
                    continue

                per_day[day] += 1
                sessions += 1
                status_counts[status] = status_counts.get(status, 0) + 1
                artifacts += int(row["artifact_count"] or 0)
                profiles = self._store.list_indexed_artifacts(
                    project_id=project_id,
                    session_id=row["session_id"],
                    artifact_types=(ArtifactType.DATASET_PROFILE,),
                )
                profiled_datasets += len(profiles)
                for profile in profiles:
                    rows = profile.payload.get("rows")
                    if isinstance(rows, int) and rows >= 0:
                        profiled_rows += rows
                persisted = self._persisted_metrics(project_id, row["session_id"])
                if persisted is None:
                    unpriced += 1
                    continue
                metrics, generated_at = persisted
                # Same merge get_metrics applies. Without it the dashboard would
                # report a smaller number than the run's own Trace page for any
                # run that had a chat turn after it persisted its metrics — two
                # screens disagreeing about the same run.
                if generated_at is not None:
                    metrics = self._merge_incremental_usage(
                        project_id, row["session_id"], metrics, generated_at
                    )
                llm_calls += metrics.llm_calls
                tokens += metrics.total_tokens
                # An artifact whose cost could not be priced is not a priced
                # session: counting it would add $0 to the total while claiming
                # the total covers it.
                if metrics.est_cost_usd is None:
                    unpriced += 1
                    continue
                priced += 1
                cost += metrics.est_cost_usd

        _, data_bytes = self._store.workspace_upload_totals()
        return WorkspaceUsageView(
            generated_at=now,
            window_days=days,
            project_count=projects,
            session_count=sessions,
            truncated_sessions=truncated,
            status_counts=status_counts,
            daily=[UsageDay(date=date, sessions=count) for date, count in per_day.items()],
            llm_calls=llm_calls,
            total_tokens=tokens,
            est_cost_usd=round(cost, 6),
            priced_sessions=priced,
            unpriced_sessions=unpriced,
            artifact_count=artifacts,
            dataset_count=profiled_datasets,
            profiled_rows=profiled_rows,
            data_bytes=data_bytes,
            recent=sorted(
                recent,
                key=lambda item: (
                    (item.updated_at or item.created_at) is not None,
                    item.updated_at or item.created_at or _EPOCH,
                ),
                reverse=True,
            )[:_USAGE_RECENT_LIMIT],
        )

    def get_metrics(self, session_id: str) -> SessionMetricsView:
        project_id = self._project_for_run(session_id)
        persisted = self._persisted_metrics(project_id, session_id)
        if persisted is None:
            metrics, source, generated_at = (
                summarize_session(self._store, project_id, session_id),
                "aggregated",
                None,
            )
        else:
            metrics, source, generated_at = persisted[0], "artifact", persisted[1]
            if generated_at is not None:
                metrics = self._merge_incremental_usage(
                    project_id,
                    session_id,
                    metrics,
                    generated_at,
                )
                if metrics is not persisted[0]:
                    source = "artifact+incremental"
        event_count = sum(
            self._store.trace_event_type_counts(
                project_id=project_id, session_id=session_id
            ).values()
        )
        return SessionMetricsView.model_validate(
            {
                **metrics.model_dump(mode="python"),
                "source": source,
                "event_count": event_count,
                "generated_at": generated_at,
            }
        )

    def record_client_failure(
        self,
        session_id: str,
        failure: ClientFailureRequest,
    ) -> ClientFailureRecorded:
        """Persist only typed, allowlisted UI failure metadata.

        The client-provided dedupe key is used exclusively as an event key; it
        is not copied into the human-visible summary. Duplicate delivery is a
        successful no-op and does not consume the per-run rate budget.
        """
        project_id = self._project_for_run(session_id)
        # The browser UUID is only a replay token. Persist a one-way binding so
        # durable dedupe does not turn the raw client identifier into telemetry.
        dedupe_digest = hashlib.sha256(failure.dedupe_key.encode()).hexdigest()
        event_key = f"client-failure:{session_id}:{dedupe_digest}"
        since = datetime.now(UTC) - timedelta(seconds=CLIENT_FAILURE_RATE_WINDOW_SECONDS)
        now = datetime.now(UTC)
        outcome = self._store.append_trace_bounded(
            project_id,
            TraceEvent(
                session_id=session_id,
                event_type=CLIENT_FAILURE_EVENT_TYPE,
                name=failure.operation,
                event_key=event_key,
                started_at=now,
                finished_at=now,
                summary={
                    "source": "react",
                    "error_code": failure.error_code,
                    "operation": failure.operation,
                },
            ),
            max_events=CLIENT_FAILURE_RATE_LIMIT,
            since_iso=since.isoformat(),
            summary_source="react",
        )
        if outcome == "duplicate":
            return ClientFailureRecorded(recorded=False)
        if outcome == "rate_limited":
            raise ClientFailureRateLimitError(
                "Handled client failure rate limit exceeded for this run."
            )
        return ClientFailureRecorded(recorded=True)

    def _merge_incremental_usage(
        self,
        project_id: str,
        session_id: str,
        persisted: SessionMetrics,
        generated_at: datetime,
    ) -> SessionMetrics:
        """Merge provider calls made after the immutable metrics artifact."""
        events = self._store.list_trace_events(
            project_id=project_id, session_id=session_id, event_types=BUDGET_EVENT_TYPES
        )
        incremental = [
            event for event in events if (event.finished_at or event.started_at) > generated_at
        ]
        billed = spend_events(
            [event for event in incremental if event.event_type == LLM_USAGE_EVENT]
        )
        reserved = [event for event in incremental if event.event_type == BUDGET_RESERVED_EVENT]
        settled = [event for event in incremental if event.event_type == BUDGET_SETTLED_EVENT]
        rejected = [event for event in incremental if event.event_type == BUDGET_REJECTED_EVENT]
        if not (billed or reserved or settled or rejected):
            return persisted
        prompt_tokens = sum(_metric_int(event.summary.get("prompt_tokens")) for event in billed)
        cached_tokens = sum(_metric_int(event.summary.get("cached_tokens")) for event in billed)
        total_tokens = sum(_metric_int(event.summary.get("total_tokens")) for event in billed)
        costs = [
            float(event.summary["estimated_cost_usd"])
            for event in billed
            if event.summary.get("estimated_cost_usd") is not None
        ]
        merged_prompt = persisted.prompt_tokens + prompt_tokens
        merged_cached = persisted.cached_tokens + cached_tokens
        current = summarize_session(self._store, project_id, session_id)
        budget_costs = [
            float(event.summary["estimated_cost_usd"])
            for event in settled
            if event.summary.get("estimated_cost_usd") is not None
        ]
        return persisted.model_copy(
            update={
                "llm_calls": persisted.llm_calls + len(billed),
                "total_tokens": persisted.total_tokens + total_tokens,
                "prompt_tokens": merged_prompt,
                "cached_tokens": merged_cached,
                "cache_hit_rate": (
                    0.0
                    if merged_prompt <= 0
                    else round(min(merged_cached, merged_prompt) / merged_prompt, 6)
                ),
                "est_cost_usd": _merge_cost(persisted.est_cost_usd, costs),
                "budget_reserved_calls": persisted.budget_reserved_calls + len(reserved),
                "budget_settled_calls": persisted.budget_settled_calls + len(settled),
                "budget_rejected_calls": persisted.budget_rejected_calls + len(rejected),
                "budget_uncertain_calls": persisted.budget_uncertain_calls
                + sum(event.summary.get("status") == "uncertain" for event in settled),
                "budget_total_tokens": persisted.budget_total_tokens
                + sum(_metric_int(event.summary.get("total_tokens")) for event in settled),
                "budget_est_cost_usd": _merge_cost(persisted.budget_est_cost_usd, budget_costs),
                "budget_reconciliation": _merge_reconciliation(
                    persisted.budget_reconciliation,
                    current.budget_reconciliation,
                ),
                "duration_seconds": max(persisted.duration_seconds, current.duration_seconds),
                "artifact_counts": current.artifact_counts,
            }
        )

    def list_events(
        self,
        session_id: str,
        *,
        limit: int = DEFAULT_TRACE_LIMIT,
        cursor: str | None = None,
        event_type: str | None = None,
    ) -> TraceEventPage:
        limit = max(1, min(limit, MAX_TRACE_LIMIT))
        project_id = self._project_for_run(session_id)
        event_types = self._store.trace_event_type_counts(
            project_id=project_id, session_id=session_id
        )
        after_id = _decode_cursor(cursor, event_type, session_id) if cursor else None
        rows = self._store.query_trace_rows(
            project_id=project_id,
            session_id=session_id,
            event_type=event_type,
            after_id=after_id,
            limit=limit + 1,
        )
        has_more = len(rows) > limit
        rows = rows[:limit]
        items = [row for row in (self._to_row(*raw) for raw in rows) if row is not None]
        # Cursor follows the last row read, not the last row rendered: a
        # malformed payload must not make the page repeat forever.
        next_cursor = (
            _encode_cursor(rows[-1][0], event_type, session_id) if has_more and rows else None
        )
        total = sum(event_types.values()) if event_type is None else event_types.get(event_type, 0)
        return TraceEventPage(
            session_id=session_id,
            items=items,
            next_cursor=next_cursor,
            event_types=event_types,
            total=total,
        )

    def get_debug(
        self,
        session_id: str,
        *,
        limit: int = DEFAULT_SESSION_DEBUG_LIMIT,
        cursor: str | None = None,
    ) -> SessionDebugView:
        """Developer-inspector rollup: run totals, report-quality gauges, and the
        timeline / LLM / tool / error / artifact tables, one bounded page."""
        limit = max(1, min(limit, MAX_SESSION_DEBUG_LIMIT))
        project_id = self._project_for_run(session_id)
        after_trace_id, after_artifact_rowid = _decode_debug_cursor(cursor, session_id)
        trace_rows = self._store.query_trace_rows(
            project_id=project_id,
            session_id=session_id,
            after_id=after_trace_id or None,
            limit=limit + 1,
        )
        trace_has_more = len(trace_rows) > limit
        trace_page = trace_rows[:limit]
        events = [
            event
            for _row_id, payload in trace_page
            if (event := self._parse_trace_payload(payload)) is not None
        ]
        artifact_rows_page = self._store.query_artifact_debug_rows(
            project_id=project_id,
            session_id=session_id,
            after_rowid=after_artifact_rowid,
            limit=limit + 1,
        )
        artifact_has_more = len(artifact_rows_page) > limit
        artifact_page = artifact_rows_page[:limit]
        artifacts = [
            artifact
            for _row_id, _artifact_id, _artifact_type, path in artifact_page
            if (artifact := _read_debug_artifact(path, self._store.root)) is not None
        ]
        summary, quality = self._debug_rollups(project_id, session_id)
        next_cursor = None
        if trace_has_more or artifact_has_more:
            next_cursor = _encode_debug_cursor(
                trace_page[-1][0] if trace_page else after_trace_id,
                artifact_page[-1][0] if artifact_page else after_artifact_rowid,
                session_id,
            )
        return SessionDebugView(
            session_id=session_id,
            code_version=self._code_version(project_id, session_id),
            summary=SessionDebugSummary(
                events=int(summary["events"]),
                artifacts=int(summary["artifacts"]),
                llm_calls=int(summary["llm_calls"]),
                tool_calls=int(summary["tool_calls"]),
                errors=int(summary["errors"]),
                total_tokens=int(summary["total_tokens"]),
                estimated_cost_usd=float(summary["estimated_cost_usd"]),
            ),
            report_quality=ReportQualitySummary(
                section_coverage=float(quality["section_coverage"]),
                claim_section_coverage=float(quality["claim_section_coverage"]),
                claim_survival_rate=float(quality["claim_survival_rate"]),
                deterministic_repair_count=int(quality["deterministic_repair_count"]),
                prompt_tokens_by_attempt=str(quality["prompt_tokens_by_attempt"]),
            ),
            timeline=[
                DebugTimelineRow(
                    event_type=_text(row.get("event_type")),
                    name=_text(row.get("name")),
                    started_at=_text(row.get("started_at")),
                    duration_ms=_optional_int(row.get("duration_ms")),
                    summary=self._safe_text(row.get("summary")),
                )
                for row in timeline_rows(events)
            ],
            llm_calls=[
                DebugLlmCallRow(
                    task=_text(row.get("task")),
                    provider=_text(row.get("provider")),
                    model=_text(row.get("model")),
                    prompt_tokens=_optional_int(row.get("prompt_tokens")) or 0,
                    completion_tokens=_optional_int(row.get("completion_tokens")) or 0,
                    total_tokens=_optional_int(row.get("total_tokens")) or 0,
                    estimated_cost_usd=_optional_float(row.get("estimated_cost_usd")),
                    schema=_text(row.get("schema")),
                    duration_ms=_optional_int(row.get("duration_ms")),
                    status=self._safe_text(row.get("status")),
                    attempt=_text(row.get("attempt")),
                    error_type=_text(row.get("error_type")),
                    error=self._safe_text(row.get("error")),
                )
                for row in llm_call_rows(events)
            ],
            tool_calls=[
                DebugToolCallRow(
                    event_type=_text(row.get("event_type")),
                    tool=_text(row.get("tool")),
                    duration_ms=_optional_int(row.get("duration_ms")),
                    row_count=_optional_int(row.get("row_count")),
                    truncated=_optional_bool(row.get("truncated")),
                    artifact_id=_text(row.get("artifact_id")),
                    summary=self._safe_text(row.get("summary")),
                )
                for row in tool_call_rows(events)
            ],
            errors=[
                DebugErrorRow(
                    event_type=_text(row.get("event_type")),
                    name=_text(row.get("name")),
                    error_type=_text(row.get("error_type")),
                    error=self._safe_text(row.get("error")),
                )
                for row in error_rows(events)
            ],
            artifacts=[
                DebugArtifactRow(
                    artifact_id=_text(row.get("artifact_id")),
                    type=_text(row.get("type")),
                    parents=_optional_int(row.get("parents")) or 0,
                    warnings=_optional_int(row.get("warnings")) or 0,
                )
                for row in artifact_rows(artifacts)
            ],
            next_cursor=next_cursor,
        )

    def _parse_trace_payload(self, payload: str) -> TraceEvent | None:
        try:
            return TraceEvent.model_validate_json(payload)
        except (ValueError, ValidationError):
            return None

    def _debug_rollups(
        self, project_id: str, session_id: str
    ) -> tuple[dict[str, int | float], dict[str, int | float | str]]:
        """Aggregate the debug header in keyset batches with bounded memory."""
        event_types = self._store.trace_event_type_counts(
            project_id=project_id, session_id=session_id
        )
        use_ledger = event_types.get(LLM_USAGE_EVENT, 0) > 0
        summary: dict[str, int | float] = {
            "events": sum(event_types.values()),
            "artifacts": self._store.count_session_artifacts(
                project_id=project_id, session_id=session_id
            ),
            "llm_calls": 0,
            "tool_calls": sum(
                event_types.get(kind, 0)
                for kind in ("tool_started", "tool_completed", "tool_failed")
            ),
            "errors": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
        latest_validation: TraceEvent | None = None
        prompt_attempts: deque[str] = deque(maxlen=100)
        after_id = 0
        while True:
            rows = self._store.query_trace_rows(
                project_id=project_id,
                session_id=session_id,
                after_id=after_id or None,
                limit=500,
            )
            if not rows:
                break
            after_id = rows[-1][0]
            for _row_id, payload in rows:
                event = self._parse_trace_payload(payload)
                if event is None:
                    continue
                billed = (
                    event.event_type == LLM_USAGE_EVENT
                    if use_ledger
                    else event.event_type in {"llm_call", "llm_error"}
                )
                if billed:
                    summary["llm_calls"] = int(summary["llm_calls"]) + 1
                    summary["total_tokens"] = int(summary["total_tokens"]) + _int(
                        event.summary.get("total_tokens")
                    )
                    summary["estimated_cost_usd"] = float(summary["estimated_cost_usd"]) + _float(
                        event.summary.get("estimated_cost_usd")
                    )
                if error_rows([event]):
                    summary["errors"] = int(summary["errors"]) + 1
                if event.event_type == "report_validation":
                    latest_validation = event
                if (
                    event.event_type in {"llm_call", "llm_error"}
                    and event.name == "m2_report_claim_plan"
                ):
                    prompt_attempts.append(
                        f"{event.summary.get('attempt', '')}: "
                        f"{_int(event.summary.get('prompt_tokens'))}"
                    )
            if len(rows) < 500:
                break
        summary["estimated_cost_usd"] = round(float(summary["estimated_cost_usd"]), 6)
        validation = latest_validation.summary if latest_validation else {}
        quality: dict[str, int | float | str] = {
            "section_coverage": _float(validation.get("section_coverage")),
            "claim_section_coverage": _float(validation.get("claim_section_coverage")),
            "claim_survival_rate": _float(validation.get("claim_survival_rate")),
            "deterministic_repair_count": _int(validation.get("deterministic_repair_count")),
            "prompt_tokens_by_attempt": ", ".join(prompt_attempts),
        }
        return summary, quality

    def open_debug_log(self, session_id: str) -> DebugLogDownload:
        """Locate the run's debug.jsonl, refusing anything that resolves outside
        the run directory (a symlinked log would otherwise stream any file)."""
        project_id = self._project_for_run(session_id)
        session_dir = self._store.session_dir(project_id, session_id)
        resolved = _contained_file(session_dir, DEBUG_LOG_FILENAME)
        if resolved is None:
            raise DebugLogNotFoundError(session_id)
        replacements = _debug_log_replacements(
            self._store.root, _configured_api_keys(self._store.root)
        )
        return DebugLogDownload(
            path=resolved,
            filename=DEBUG_LOG_FILENAME,
            media_type="application/x-ndjson",
            byte_size=resolved.stat().st_size,
            replacements=replacements,
        )

    def list_llm_calls(
        self,
        session_id: str,
        *,
        limit: int = DEFAULT_LLM_DEBUG_LIMIT,
        cursor: str | None = None,
    ) -> Page[LlmDebugRecord]:
        """Captured LLM prompt/response previews for a run. A run with no
        capture file is an empty page, not a 404."""
        limit = max(1, min(limit, MAX_LLM_DEBUG_LIMIT))
        project_id = self._project_for_run(session_id)
        session_dir = self._store.session_dir(project_id, session_id)
        path = _contained_file(session_dir, LLM_DEBUG_FILENAME)
        index = JsonlPageIndex(self._store.db_path, self._store.root)
        indexed_path = session_dir / LLM_DEBUG_FILENAME if path is None else path
        scope = f"llm-debug:{project_id}:{session_id}"
        current_version = index.file_source_version(indexed_path)
        offset = (
            decode_bound_cursor(
                cursor,
                scope=scope,
                source_version=current_version,
            )
            if cursor
            else 0
        )
        state = index.ensure(
            indexed_path,
            accept=_is_json_object,
        )
        if cursor and state.source_version != current_version:
            raise InvalidCursorError
        current_version = state.source_version
        indexed = index.page(state, start=offset, limit=limit + 1)
        if index.file_source_version(indexed_path) != current_version:
            raise InvalidCursorError
        has_more = len(indexed) > limit
        window = indexed[:limit]
        # An oversized capture was indexed but never read, so there is no JSON to
        # redact or render; the developer inspector drops it. The cursor still
        # advances over the whole window, or a page of only oversized records
        # would end pagination and hide the tail.
        page = [record for record in window if not record.oversized]
        try:
            records = [json.loads(record.payload) for record in page]
        except ValueError as exc:
            # The stat re-check narrows but cannot close the window between
            # reading the bytes at an indexed offset and observing the file
            # identity; a torn record is a stale cursor, not a server fault.
            raise InvalidCursorError from exc
        api_keys = _configured_api_keys(self._store.root)
        if records and not api_keys:
            logger.warning(
                "No provider API key could be resolved; captured LLM payloads for "
                "run %s are served without redaction.",
                session_id,
            )
        items = [
            self._llm_record(indexed_record.ordinal + 1, record, api_keys)
            for indexed_record, record in zip(page, records, strict=True)
        ]
        return Page[LlmDebugRecord](
            items=items,
            next_cursor=(
                encode_bound_cursor(
                    window[-1].ordinal + 1,
                    scope=scope,
                    source_version=current_version,
                )
                if has_more and window
                else None
            ),
        )

    def _llm_record(
        self, index: int, record: dict[str, object], api_keys: Sequence[str]
    ) -> LlmDebugRecord:
        def text(key: str) -> str:
            return _redact(self._safe_text(record.get(key)), api_keys)

        return LlmDebugRecord(
            index=index,
            ts=text("ts"),
            kind=text("kind"),
            transport_kind=text("transport_kind"),
            task=text("task"),
            provider=text("provider"),
            model=text("model"),
            endpoint_host=text("endpoint_host"),
            status=text("status"),
            finish_reason=text("finish_reason"),
            duration_s=_optional_float(record.get("duration_s")),
            prompt_tokens=_optional_int(record.get("prompt_tokens")),
            completion_tokens=_optional_int(record.get("completion_tokens")),
            cached_tokens=_optional_int(record.get("cached_tokens")),
            cache_creation_tokens=_optional_int(record.get("cache_creation_tokens")),
            reasoning_tokens=_optional_int(record.get("reasoning_tokens")),
            estimated_cost_usd=_optional_float(record.get("estimated_cost_usd")),
            cost_basis=text("cost_basis"),
            pricing_version=text("pricing_version"),
            usage_reported=record.get("usage_reported") is not False,
            request_id=text("request_id"),
            response_id=text("response_id"),
            request_bytes=_optional_int(record.get("request_bytes")),
            response_bytes=_optional_int(record.get("response_bytes")),
            payload_preview=text("payload_preview"),
            response_preview=text("response_preview"),
        )

    def _safe_text(self, value: object) -> str:
        return _strip_workspace(_text(value), self._store.root)

    def _code_version(self, project_id: str, session_id: str) -> str | None:
        try:
            manifest = self._store.read_manifest(project_id, session_id)
        except (OSError, ValueError):
            return None
        return None if manifest is None else manifest.code_version

    def _to_row(self, event_id: int, payload: str) -> TraceEventRow | None:
        try:
            event = TraceEvent.model_validate_json(payload)
        except ValidationError:
            logger.warning("Skipped malformed trace row %s", event_id)
            return None
        duration = (
            (event.finished_at - event.started_at).total_seconds()
            if event.finished_at is not None
            else None
        )
        return TraceEventRow(
            event_id=event_id,
            event_type=event.event_type,
            name=event.name,
            started_at=event.started_at,
            finished_at=event.finished_at,
            duration_seconds=None if duration is None else round(max(duration, 0.0), 6),
            # Summaries are producer-controlled and have carried file paths
            # (loader/export events), so they are relativized like any payload.
            summary=relativize_workspace_paths(event.summary, self._store.root),
        )

    def _persisted_metrics(
        self, project_id: str, session_id: str
    ) -> tuple[SessionMetrics, datetime | None] | None:
        """Newest SessionMetrics artifact for the run, or None when absent/invalid."""
        for row in self._store.latest_artifact_index_rows(
            project_id, session_id, ArtifactType.SESSION_METRICS.value
        ):
            artifact = self._read_contained(
                str(row["artifact_id"]), project_id=project_id, session_id=session_id
            )
            if artifact is None:
                continue
            try:
                return SessionMetrics.model_validate(artifact.payload), artifact.created_at
            except ValidationError:
                continue
        return None

    def _read_contained(
        self, artifact_id: str, *, project_id: str, session_id: str
    ) -> Artifact | None:
        row = self._store.artifact_index_row(
            artifact_id, project_id=project_id, session_id=session_id
        )
        if row is None:
            return None
        path = row["path"]
        try:
            if not path.resolve().is_relative_to(self._store.root.resolve()):
                return None
            return Artifact.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _project_for_run(self, session_id: str) -> str:
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        row = self._store.get_session_index_row(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        return str(row["project_id"])


def _contained_file(session_dir: Path, filename: str) -> Path | None:
    """The named file, only when it really sits inside ``session_dir``."""
    try:
        resolved = (session_dir / filename).resolve(strict=True)
        if not resolved.is_relative_to(session_dir.resolve()) or not resolved.is_file():
            return None
    except OSError:
        return None
    return resolved


def _is_json_object(payload: bytes) -> bool:
    try:
        return isinstance(json.loads(payload), dict)
    except (UnicodeError, json.JSONDecodeError):
        return False


def _encode_debug_cursor(trace_id: int, artifact_rowid: int, session_id: str) -> str:
    raw = json.dumps(
        {"t": trace_id, "a": artifact_rowid, "r": session_id},
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_debug_cursor(cursor: str | None, session_id: str) -> tuple[int, int]:
    if cursor is None:
        return 0, 0
    try:
        decoded = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
    except ValueError:
        raise InvalidCursorError() from None
    if (
        not isinstance(decoded, dict)
        or decoded.get("r") != session_id
        # ``isinstance(True, int)`` is True; match decode_bound_cursor and
        # reject a bool that would silently index as 1.
        or type(decoded.get("t")) is not int
        or type(decoded.get("a")) is not int
        or decoded["t"] < 0
        or decoded["a"] < 0
    ):
        raise InvalidCursorError()
    return decoded["t"], decoded["a"]


def _read_debug_artifact(path: Path, workspace_root: Path) -> Artifact | None:
    """``workspace_root`` must already be resolved; ArtifactStore.root is, and
    re-resolving it per artifact costs a syscall on every page row."""
    try:
        resolved = path.resolve()
        if not resolved.is_relative_to(workspace_root):
            return None
        with resolved.open("rb") as handle:
            payload = handle.read(_DEBUG_ARTIFACT_READ_BYTES + 1)
        if len(payload) > _DEBUG_ARTIFACT_READ_BYTES:
            return None
        return Artifact.model_validate_json(payload.decode("utf-8"))
    except (OSError, ValueError):
        return None


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _env_file_candidates(root: Path) -> list[Path]:
    """Find workspace-adjacent and repository-anchored secret sources."""
    candidates: list[Path] = []
    try:
        base = root.resolve()
    except OSError:
        base = root
    for directory in (base, *base.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            candidates.append(candidate)
            break
    if DEFAULT_ENV_PATH.is_file() and DEFAULT_ENV_PATH not in candidates:
        candidates.append(DEFAULT_ENV_PATH)
    return candidates


def _configured_api_keys(root: Path) -> list[str]:
    """Every provider key slot core.env knows about, not just the active
    provider's: a capture predates key rotations and provider switches, and the
    stale key in it is exactly as sensitive. Providers echo request bodies into
    error text, so a capture can carry one (settings_service._redact)."""
    sources: list[Mapping[str, str]] = [os.environ]
    for path in _env_file_candidates(root):
        try:
            sources.append(parse_env_file(path))
        except (OSError, ValueError):
            continue
    keys: set[str] = set()
    for source in sources:
        for name in API_KEY_ENV_VARS:
            value = source.get(name, "").strip()
            # Short keys would match too much ordinary text to blank out safely.
            if len(value) >= _MIN_REDACTABLE_KEY_CHARS:
                keys.add(value)
    # Longest first, so a key that contains a shorter one is masked whole.
    return sorted(keys, key=len, reverse=True)


def _redact(text: str, api_keys: Sequence[str]) -> str:
    for api_key in api_keys:
        text = text.replace(api_key, "***")
    return text


def _strip_workspace(text: str, root: Path) -> str:
    """Debug text is free-form (summary previews, exception messages, captured
    payloads), so a workspace path can sit mid-string where
    relativize_workspace_paths — whole-value only — will not catch it."""
    for prefix in {str(root.resolve()), str(root)}:
        if prefix and prefix != os.sep:
            text = text.replace(prefix + os.sep, "").replace(prefix, ".")
    return text


def _debug_log_replacements(root: Path, api_keys: Sequence[str]) -> tuple[tuple[bytes, bytes], ...]:
    replacements: set[tuple[bytes, bytes]] = set()
    for prefix in {str(root.resolve()), str(root)}:
        if prefix and prefix != os.sep:
            encoded = prefix.encode("utf-8")
            replacements.add((encoded + os.sep.encode("utf-8"), b""))
            replacements.add((encoded, b"."))
    for api_key in api_keys:
        replacements.add((api_key.encode("utf-8"), b"***"))
    return tuple(sorted(replacements, key=lambda item: len(item[0]), reverse=True))


def _stream_redacted(
    path: Path,
    *,
    byte_limit: int,
    replacements: Sequence[tuple[bytes, bytes]],
) -> Iterator[bytes]:
    """Redact each JSONL record while stopping at the open-time file size."""
    remaining = byte_limit
    with path.open("rb") as handle:
        while remaining > 0:
            line = handle.readline(remaining)
            if not line:
                break
            remaining -= len(line)
            for pattern, replacement in replacements:
                line = line.replace(pattern, replacement)
            yield line


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) else None


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _metric_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    return int(value) if isinstance(value, int | float) else 0


def _merge_cost(existing: float | None, incremental: list[float]) -> float | None:
    if existing is None and not incremental:
        return None
    return round((existing or 0.0) + sum(incremental), 9)


def _merge_reconciliation(existing: str, current: str) -> str:
    if "unverifiable" in {existing, current}:
        return "unverifiable"
    if current == "verified":
        return "verified"
    return existing


def _encode_cursor(event_id: int, event_type: str | None, session_id: str) -> str:
    raw = json.dumps({"i": event_id, "t": event_type or "", "r": session_id}).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_cursor(cursor: str, event_type: str | None, session_id: str) -> int:
    try:
        decoded = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
    except (ValueError, UnicodeDecodeError):
        raise InvalidCursorError() from None
    if not isinstance(decoded, dict) or not isinstance(decoded.get("i"), int):
        raise InvalidCursorError()
    # Bound to the filter and run it was minted under, or replaying it
    # elsewhere would silently skip or repeat rows.
    if decoded.get("t") != (event_type or "") or decoded.get("r") != session_id:
        raise InvalidCursorError()
    return decoded["i"]
