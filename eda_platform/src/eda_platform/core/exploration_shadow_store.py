"""Strict evaluation store for E4a shadow exploration projections."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from eda_platform.core.config import require_absolute_workspace
from eda_platform.core.file_lock import lock_exclusive, unlock
from eda_platform.core.fs import BINARY_FLAG
from eda_platform.schemas.exploration_shadow import (
    ShadowExplorationProjection,
    validate_exploration_id,
)

SHADOW_DIRECTORY = "exploration-eval"


class ShadowPathViolationError(ValueError):
    """Raised when a shadow path is outside its run root or traverses a symlink."""


class ShadowProjectionConflictError(RuntimeError):
    """Raised when a stale or regressing journal projection is written."""


def shadow_run_root(workspace: Path | str, exploration_id: str) -> Path:
    """Return the canonical, validated root for one shadow exploration run."""
    workspace_path = require_absolute_workspace(Path(workspace))
    safe_id = validate_exploration_id(exploration_id)
    run_root = workspace_path / SHADOW_DIRECTORY / safe_id
    return validate_shadow_run_path(workspace_path, safe_id, run_root)


def validate_shadow_run_path(
    workspace: Path | str,
    exploration_id: str,
    path: Path | str,
) -> Path:
    """Validate and resolve a path contained by one shadow exploration run.

    This is the shared boundary for the projection store and driver-owned
    journal/recovery paths. Prospective paths may not exist, but every existing
    component from the workspace down must be a real directory rather than a
    symlink. The candidate leaf may be a regular file.
    """
    workspace_path = require_absolute_workspace(Path(workspace))
    safe_id = validate_exploration_id(exploration_id)
    run_root = workspace_path / SHADOW_DIRECTORY / safe_id
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise ShadowPathViolationError("shadow paths must be absolute.")
    if ".." in candidate.parts:
        raise ShadowPathViolationError("shadow paths cannot contain '..'.")
    try:
        candidate.relative_to(run_root)
    except ValueError as exc:
        raise ShadowPathViolationError(
            "shadow path must be lexically contained by its exploration run root."
        ) from exc

    _assert_safe_components(workspace_path, run_root, leaf_may_be_file=False)
    _assert_safe_components(workspace_path, candidate, leaf_may_be_file=True)
    try:
        resolved_workspace = workspace_path.resolve(strict=False)
        resolved_run_root = run_root.resolve(strict=False)
        resolved_candidate = candidate.resolve(strict=False)
    except OSError as exc:
        raise ShadowPathViolationError("cannot resolve shadow path containment.") from exc
    shadow_root = resolved_workspace / SHADOW_DIRECTORY
    if not resolved_run_root.is_relative_to(shadow_root):
        raise ShadowPathViolationError("shadow run root escapes the evaluation directory.")
    if not resolved_candidate.is_relative_to(resolved_run_root):
        raise ShadowPathViolationError("shadow path escapes its exploration run root.")
    return resolved_candidate


class ShadowExplorationStore:
    """Persist whole-file projections without exposing an ArtifactStore surface."""

    def __init__(self, workspace: Path | str) -> None:
        self.workspace = require_absolute_workspace(Path(workspace))
        self.root = self.workspace / SHADOW_DIRECTORY

    def path_for(self, exploration_id: str) -> Path:
        run_root = shadow_run_root(self.workspace, exploration_id)
        return validate_shadow_run_path(
            self.workspace,
            exploration_id,
            run_root / "projection.json",
        )

    def read(self, exploration_id: str) -> ShadowExplorationProjection | None:
        path = self.path_for(exploration_id)
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise ShadowPathViolationError("shadow projection must be a regular file.")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | BINARY_FLAG
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, encoding="utf-8") as handle:
            return ShadowExplorationProjection.model_validate_json(handle.read())

    def project(self, projection: ShadowExplorationProjection) -> Path:
        """Atomically replace a projection only when its journal seq advances."""
        path = self.path_for(projection.exploration_id)
        _create_shadow_run_directories(self.workspace, path.parent)
        path = validate_shadow_run_path(
            self.workspace, projection.exploration_id, path
        )
        lock_path = validate_shadow_run_path(
            self.workspace,
            projection.exploration_id,
            path.with_suffix(path.suffix + ".lock"),
        )
        with _projection_lock(lock_path):
            path = validate_shadow_run_path(
                self.workspace, projection.exploration_id, path
            )
            current = self.read(projection.exploration_id)
            if current is not None and projection.last_seq <= current.last_seq:
                raise ShadowProjectionConflictError(
                    f"projection seq must advance beyond {current.last_seq}, "
                    f"got {projection.last_seq}."
                )
            payload = projection.model_dump_json(indent=2).encode("utf-8")
            descriptor, temporary_name = tempfile.mkstemp(
                dir=path.parent, prefix=".projection-", suffix=".tmp"
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                validate_shadow_run_path(
                    self.workspace, projection.exploration_id, path
                )
                os.replace(temporary, path)
                _fsync_directory(path.parent)
            finally:
                temporary.unlink(missing_ok=True)
        return path


def _assert_safe_components(
    workspace: Path,
    candidate: Path,
    *,
    leaf_may_be_file: bool,
) -> None:
    try:
        relative = candidate.relative_to(workspace)
    except ValueError as exc:
        raise ShadowPathViolationError("shadow path escapes its workspace.") from exc
    current = workspace
    for index, part in enumerate(relative.parts):
        current /= part
        if current.is_symlink():
            raise ShadowPathViolationError(
                f"shadow path component cannot be a symlink: {current.name}."
            )
        is_leaf = index == len(relative.parts) - 1
        if current.exists() and not current.is_dir() and not (is_leaf and leaf_may_be_file):
            raise ShadowPathViolationError(
                f"shadow path parent must be a directory: {current.name}."
            )


def _create_shadow_run_directories(workspace: Path, run_root: Path) -> None:
    relative = run_root.relative_to(workspace)
    current = workspace
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ShadowPathViolationError(
                f"shadow path component cannot be a symlink: {current.name}."
            )
        if current.exists():
            if not current.is_dir():
                raise ShadowPathViolationError(
                    f"shadow path parent must be a directory: {current.name}."
                )
            continue
        current.mkdir()
        if current.is_symlink() or not current.is_dir():
            raise ShadowPathViolationError(
                f"shadow path component is not a safe directory: {current.name}."
            )
        _fsync_directory(current.parent)


@contextmanager
def _projection_lock(lock_path: Path) -> Iterator[None]:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_APPEND
        | getattr(os, "O_NOFOLLOW", 0)
        | BINARY_FLAG
    )
    descriptor = os.open(lock_path, flags, 0o600)
    with os.fdopen(descriptor, "a+b") as handle:
        lock_exclusive(handle.fileno())
        try:
            yield
        finally:
            unlock(handle.fileno())


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
