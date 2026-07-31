"""Structured debug JSONL mirror for trace events."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from eda_platform.schemas.sessions import TraceEvent

DEBUG_LOG_ENV = "EDA_DEBUG_LOG"
DEBUG_LOG_FILENAME = "debug.jsonl"

_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})


def debug_log_enabled() -> bool:
    """True only when ``EDA_DEBUG_LOG`` is set to an enabling value."""
    value = os.environ.get(DEBUG_LOG_ENV)
    if not value:
        return False
    return value.strip().lower() in _ENABLED_VALUES


def debug_log_path(session_dir: Path) -> Path:
    return session_dir / DEBUG_LOG_FILENAME


def mirror_event_to_debug_log(session_dir: Path, event: TraceEvent) -> None:
    """Append one trace event to the run's debug.jsonl. No-op when disabled."""
    if not debug_log_enabled():
        return
    try:
        record = {
            "session_id": event.session_id,
            "event_type": event.event_type,
            "name": event.name,
            "started_at": event.started_at.isoformat(),
            "finished_at": (event.finished_at.isoformat() if event.finished_at else None),
            "summary": event.summary,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        session_dir.mkdir(parents=True, exist_ok=True)
        with debug_log_path(session_dir).open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")
    except Exception:  # noqa: BLE001 - debug mirroring must never break a run
        return
