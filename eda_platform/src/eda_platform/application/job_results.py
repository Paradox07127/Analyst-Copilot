"""Durable, bounded result documents for asynchronous data operations."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from eda_platform.core.config import require_absolute_workspace
from eda_platform.core.store import session_results_relative_path

MAX_JOB_RESULT_BYTES = 8 * 1024 * 1024
_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")


class JobResultError(Exception):
    pass


class JobResultNotReadyError(JobResultError):
    error_code = "job_result_not_ready"


class JobResultTooLargeError(JobResultError):
    error_code = "job_result_too_large"


def _require_safe_job_id(job_id: str) -> None:
    if _SAFE_JOB_ID.fullmatch(job_id) is None:
        raise JobResultError("Invalid job result identity.")


def _directory_path(workspace: Path | str, project_id: str, session_id: str) -> Path:
    root = require_absolute_workspace(workspace)
    try:
        relative = session_results_relative_path(project_id, session_id)
    except ValueError as exc:
        raise JobResultError("Invalid job result identity.") from exc
    return root / relative


def _path(workspace: Path | str, project_id: str, session_id: str, job_id: str) -> Path:
    _require_safe_job_id(job_id)
    return _directory_path(workspace, project_id, session_id) / f"{job_id}.json"


def write_job_result(
    workspace: Path | str,
    project_id: str,
    session_id: str,
    job_id: str,
    payload_json: str,
) -> None:
    encoded = payload_json.encode("utf-8")
    if len(encoded) > MAX_JOB_RESULT_BYTES:
        raise JobResultTooLargeError("Job result exceeds the 8 MiB result limit.")
    target = _path(workspace, project_id, session_id, job_id)
    directory = _result_directory(workspace, project_id, session_id, create=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".job-result-", dir=directory)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # Replacing a hostile leaf symlink replaces the link itself; the
        # temporary file was created inside the verified result directory.
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def read_job_result(
    workspace: Path | str, project_id: str, session_id: str, job_id: str
) -> str:
    target = _path(workspace, project_id, session_id, job_id)
    try:
        _result_directory(workspace, project_id, session_id, create=False)
        if target.is_symlink() or not target.is_file():
            raise JobResultNotReadyError(job_id)
        resolved = target.resolve()
        root = require_absolute_workspace(workspace).resolve()
        if not resolved.is_relative_to(root):
            raise JobResultNotReadyError(job_id)
        with resolved.open("rb") as handle:
            payload = handle.read(MAX_JOB_RESULT_BYTES + 1)
        if len(payload) > MAX_JOB_RESULT_BYTES:
            raise JobResultNotReadyError(job_id)
        return payload.decode("utf-8")
    except JobResultNotReadyError:
        raise
    except (JobResultError, OSError, UnicodeDecodeError) as exc:
        raise JobResultNotReadyError(job_id) from exc


def _result_directory(
    workspace: Path | str, project_id: str, session_id: str, *, create: bool
) -> Path:
    root = require_absolute_workspace(workspace).resolve()
    directory = _directory_path(root, project_id, session_id)
    # Every level, not just the leaf: a symlinked project or run directory would
    # otherwise redirect the whole tree past the containment check below.
    for level in (directory.parent.parent, directory.parent, directory):
        if level.is_symlink():
            raise JobResultError("Job result directory cannot be a symbolic link.")
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise JobResultError("Job result directory is unavailable.")
    resolved = directory.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise JobResultError("Job result directory escaped the workspace.")
    return directory
