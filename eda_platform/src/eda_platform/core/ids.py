from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

_FILE_CHUNK_SIZE = 1 << 20  # 1 MiB
MAX_SESSION_ID_LENGTH = 200
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def stable_hash(value: Any, length: int = 12) -> str:
    """Return a deterministic short hash for JSON-serializable content."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def hash_file(
    path: Path | str,
    length: int = 12,
    *,
    cancel_check: Callable[[], object] | None = None,
) -> str:
    """Hash file contents by streaming, so large files do not inflate memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_FILE_CHUNK_SIZE), b""):
            if cancel_check is not None:
                cancel_check()
            digest.update(chunk)
    return digest.hexdigest()[:length]


def make_artifact_id(prefix: str, value: Any) -> str:
    return f"{prefix}_{stable_hash(value)}"


def make_dataset_id(name: str, content_hash: str) -> str:
    """Deterministic dataset id from a file name + content hash."""
    return f"ds_{stable_hash({'name': name, 'hash': content_hash})}"


def is_safe_session_id(session_id: str) -> bool:
    """Return whether ``session_id`` is one safe, portable filesystem segment."""
    return (
        0 < len(session_id) <= MAX_SESSION_ID_LENGTH
        and session_id not in {".", ".."}
        and _SESSION_ID_RE.fullmatch(session_id) is not None
    )


def validate_session_id(session_id: str) -> str:
    """Return a safe run id or raise before it can reach a filesystem join."""
    if not is_safe_session_id(session_id):
        raise ValueError(
            "Session ID must be 1-200 characters using only letters, numbers, "
            "dots, hyphens and underscores."
        )
    return session_id


# Marker suffix for machinery-owned derived runs (macro-loop follow-up funnels);
# they must not surface in session history or project-level coverage.
INTERNAL_SESSION_MARKER = "__internal"
AUDIT_SESSION_ID = f"audit{INTERNAL_SESSION_MARKER}"


def is_internal_session_id(session_id: str) -> bool:
    return INTERNAL_SESSION_MARKER in session_id


# Run-id prefixes minted for a user action that derives from another run:
# question batches (`qsess_`), skill replays (`ssess_`), relationship
# validations (`rvsess_`), relationship discovery (`rdsess_`), on-demand report
# generation (`rpsess_`) and what-if forks (`fksess_`, the job's lifecycle run —
# the forked analysis itself gets an ordinary `run_` id). The investigation
# governance loop adds card drafting (`qdsess_`), plan building (`ipsess_`),
# plan execution (`ixsess_`), the macro loop (`mlsess_`) and the plan runs
# themselves (`investigation_`, minted by the orchestrator). The Decision Story
# slice adds brief drafting (`sbsess_`) and decision-report generation
# (`drsess_`); both are lifecycle-only, because their drivers write onto runs
# they mint themselves. They are real, deep-linkable runs, but listing them
# alongside top-level runs buries the analyses a user started (review I2).
DERIVED_SESSION_PREFIXES = (
    "qsess_",
    "ssess_",
    "rvsess_",
    "rdsess_",
    "rpsess_",
    "fksess_",
    "qdsess_",
    "ipsess_",
    "ixsess_",
    "mlsess_",
    "sbsess_",
    "drsess_",
    "dop_",
    "investigation_",
)


# Storage buckets that exist so a project-scoped filesystem, quota and API can
# back a feature that has no project. They are never user projects. Named here
# rather than inline so the project list and the usage rollup cannot drift into
# disagreeing about what the workspace contains.
INTERNAL_PROJECT_IDS = frozenset({"unfiled-sessions"})


def is_internal_project_id(project_id: str) -> bool:
    return project_id in INTERNAL_PROJECT_IDS
