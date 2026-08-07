from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import PurePath
from typing import Any

from pydantic import BaseModel, Field

# Display cap for human-facing run titles.
SESSION_TITLE_MAX_CHARS = 40


class SessionManifest(BaseModel):
    session_id: str
    project_id: str
    input_hashes: dict[str, str]
    # Names the pipeline that produced the run -- every writer is a constant, and
    # none of them moves with the build. Auto-EDA used to report "local" here,
    # which reads as a build identifier and never was one.
    code_version: str
    model_versions: dict[str, str] = Field(default_factory=dict)
    seed: int = 42
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Optional human-facing title, absent on older sessions.
    title: str | None = None
    source_session_id: str | None = None


class TraceEvent(BaseModel):
    schema_version: int = Field(default=2, ge=1)
    session_id: str
    event_type: str
    name: str
    # Optional correlation fields keep legacy trace rows readable while making
    # new evaluation, budget, and recovery events joinable without parsing
    # task-specific summary dictionaries.
    trial_id: str | None = None
    investigation_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    call_id: str | None = None
    attempt_id: str | None = None
    # Durable worker correlation. These fields are also indexed as dedicated
    # trace_events columns so job SSE never infers ownership from cursor ranges.
    job_id: str | None = None
    job_generation: int | None = Field(default=None, ge=0)
    event_key: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class SessionInfo(BaseModel):
    """Best-effort persisted-run summary with safe defaults."""

    session_id: str
    created_at: datetime | None = None
    status: str = "unknown"
    dataset_names: list[str] = Field(default_factory=list)
    artifact_count: int = 0
    report_status: str | None = None
    chat_message_count: int = 0
    code_version: str | None = None
    seed: int | None = None
    source_session_id: str | None = None
    input_hashes: dict[str, str] | None = None
    model_versions: dict[str, str] | None = None
    title: str | None = None
    manifest_read: bool = False
    """False when the manifest was absent or unparseable. Without it a failed
    read is indistinguishable from a manifest that legitimately carries no
    lineage, and the index can neither trust nor clear the stored value."""


_STEM_SEPARATORS = re.compile(r"[_\-\s]+")


def clip_run_title(text: str) -> str:
    """Collapse whitespace and enforce the display cap with an ellipsis."""
    cleaned = " ".join(text.split())
    if len(cleaned) > SESSION_TITLE_MAX_CHARS:
        cleaned = cleaned[: SESSION_TITLE_MAX_CHARS - 1].rstrip() + "…"
    return cleaned


def _clean_stem(name: str) -> str:
    """Convert a file name stem into a human-facing title."""
    stem = PurePath(name.strip()).stem
    words = [word for word in _STEM_SEPARATORS.split(stem) if word]
    return " ".join(word[:1].upper() + word[1:] for word in words)


def build_run_title(dataset_names: Sequence[str]) -> str:
    """Build a capped human-facing title from dataset file names."""
    stems = [stem for stem in (_clean_stem(name) for name in dataset_names) if stem]
    if not stems:
        return ""
    if len(stems) == 1:
        return clip_run_title(stems[0])
    counts = Counter(stem.split(" ", 1)[0] for stem in stems)
    prefix, covered = counts.most_common(1)[0]
    # Require a strict majority before naming the collection after one family.
    if covered >= 2 and covered * 2 > len(stems):
        return clip_run_title(f"{prefix} ({len(stems)} tables)")
    return clip_run_title(f"{stems[0]} +{len(stems) - 1} more")
