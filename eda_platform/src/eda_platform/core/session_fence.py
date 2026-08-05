"""Cross-process serialization for one run's filesystem and SQLite writes.

Two implementations that agree on the lock file layout and the lock identity.

POSIX resolves the control tree with directory descriptors, so a directory
swapped underneath an in-flight open is caught instead of followed. Windows has
no ``dir_fd`` (``os.supports_dir_fd`` is empty there), so it resolves by path
and verifies the same recorded ``(st_dev, st_ino)`` identity files afterwards.
Windows also refuses to rename or delete a directory while a handle is open,
which closes most of the swap window the descriptor dance defends against.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from eda_platform.core.config import require_absolute_workspace
from eda_platform.core.file_lock import exclusive_lock
from eda_platform.core.ids import validate_session_id

SUPPORTS_DIR_FD = os.open in os.supports_dir_fd

_OPEN_RETRIES = 32
_DIRECTORY_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
)
# O_BINARY matters only on Windows, where os.open defaults to text mode and
# would rewrite the identity file's trailing newline as CRLF — the readback
# comparison is byte-exact and would then never match.
_FILE_FLAGS = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)


class SessionFencePathError(RuntimeError):
    """The workspace lock path is unsafe or unusable."""


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _write_identity(identity_fd: int, identity: bytes) -> None:
    os.ftruncate(identity_fd, 0)
    os.lseek(identity_fd, 0, os.SEEK_SET)
    os.write(identity_fd, identity)
    os.fsync(identity_fd)


def _identity_bytes(info: os.stat_result) -> bytes:
    return f"{info.st_dev}:{info.st_ino}\n".encode()


def _identity_names_directory(recorded: bytes, info: os.stat_result) -> bool:
    """Whether a persisted identity still names this directory.

    The inode identifies the object; ``st_dev`` identifies the *mount*, and the
    kernel reassigns it every time the volume is mounted. Persisting both meant
    a reboot renumbered a workspace and the fence read it as a replacement:
    `16777232:98226571` recorded against `16777234:98226571` on disk, same
    inode (2026-08-05). The in-process comparisons above keep both halves —
    there the mount cannot change between the two stat calls.

    What this gives up: a path that now resolves onto a different volume whose
    directory carries the same inode number would pass. Replacing a directory
    in place — the case the fence exists for — always changes the inode.

    Identity files written by older Windows builds were opened in text mode, so
    their trailing LF persisted as CRLF; stripping covers that without relaxing
    anything else.
    """
    text = recorded.decode("utf-8", errors="replace").strip()
    _, separator, inode = text.rpartition(":")
    return bool(separator) and inode == str(info.st_ino)


def _open_directory_at(parent_fd: int, name: str) -> int:
    """Create/open a directory and prove the descriptor still names the path."""
    for _ in range(_OPEN_RETRIES):
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                continue
            raise SessionFencePathError(f"Cannot create run fence directory {name}.") from exc
        try:
            descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                continue
            raise SessionFencePathError(f"Cannot safely open run fence directory {name}.") from exc
        try:
            opened = os.fstat(descriptor)
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISDIR(opened.st_mode) and _same_inode(opened, named):
                return descriptor
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                os.close(descriptor)
                raise SessionFencePathError(
                    f"Cannot verify run fence directory {name}."
                ) from exc
        os.close(descriptor)
    raise SessionFencePathError(f"Session fence directory {name} changed during open.")


def _open_regular_at(parent_fd: int, name: str) -> tuple[int, bool]:
    """Safely create or reopen one file despite concurrent first creators.

    macOS may report ``ENOENT`` for one contender using ``O_CREAT|O_NOFOLLOW``.
    An exclusive creator followed by a bounded loser reopen avoids that kernel
    edge while descriptor/path identity checks reject replacement races.
    """
    for _ in range(_OPEN_RETRIES):
        created = False
        try:
            descriptor = os.open(name, _FILE_FLAGS, dir_fd=parent_fd)
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                raise SessionFencePathError(f"Cannot safely open run fence {name}.") from exc
            try:
                descriptor = os.open(
                    name,
                    _FILE_FLAGS | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=parent_fd,
                )
                created = True
            except OSError as create_exc:
                if create_exc.errno in (errno.EEXIST, errno.ENOENT):
                    continue
                raise SessionFencePathError(
                    f"Cannot safely create run fence {name}."
                ) from create_exc
        try:
            opened = os.fstat(descriptor)
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISREG(opened.st_mode) and _same_inode(opened, named):
                return descriptor, created
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                os.close(descriptor)
                raise SessionFencePathError(f"Cannot verify run fence {name}.") from exc
        os.close(descriptor)
    raise SessionFencePathError(f"Session fence {name} changed during open.")


def _verify_directory_at(parent_fd: int, name: str, descriptor: int) -> None:
    try:
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise SessionFencePathError(f"Session fence directory {name} was replaced.") from exc
    if not stat.S_ISDIR(named.st_mode) or not _same_inode(opened, named):
        raise SessionFencePathError(f"Session fence directory {name} was replaced.")


def _verify_recorded_identity(
    parent_fd: int,
    identity_name: str,
    target_fd: int,
    *,
    label: str,
) -> None:
    identity_fd, created = _open_regular_at(parent_fd, identity_name)
    try:
        target = os.fstat(target_fd)
        identity = _identity_bytes(target)
        if created:
            _write_identity(identity_fd, identity)
            return
        os.lseek(identity_fd, 0, os.SEEK_SET)
        recorded = os.read(identity_fd, 128)
        if recorded == identity:
            return
        if not _identity_names_directory(recorded, target):
            raise SessionFencePathError(f"Session fence {label} identity changed.")
        # Same directory, new mount: record this one so the next open compares
        # against it rather than re-deriving the tolerance every time.
        _write_identity(identity_fd, identity)
    finally:
        os.close(identity_fd)


def _open_workspace_root(root: Path) -> tuple[int, int]:
    if not root.name:
        raise SessionFencePathError("Workspace root must not be the filesystem root.")
    try:
        parent_fd = os.open(root.parent, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise SessionFencePathError("Cannot safely open the workspace parent.") from exc
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
    setup_name = f".eda-workspace-{digest}.lock"
    identity_name = f".eda-workspace-{digest}.identity"
    try:
        setup_fd, _ = _open_regular_at(parent_fd, setup_name)
    except Exception:
        os.close(parent_fd)
        raise
    try:
        with exclusive_lock(setup_fd):
            root_fd = os.open(root.name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
            try:
                _verify_recorded_identity(
                    parent_fd,
                    identity_name,
                    root_fd,
                    label="workspace root",
                )
                _verify_directory_at(parent_fd, root.name, root_fd)
            except Exception:
                os.close(root_fd)
                raise
    except Exception:
        os.close(parent_fd)
        raise
    finally:
        os.close(setup_fd)
    return parent_fd, root_fd


def _open_control_tree(root: Path) -> tuple[int, int, int, int]:
    parent_fd, root_fd = _open_workspace_root(root)
    try:
        root_setup_fd, _ = _open_regular_at(
            root_fd, ".storage-operations-setup.lock"
        )
    except Exception:
        os.close(root_fd)
        os.close(parent_fd)
        raise
    try:
        with exclusive_lock(root_setup_fd):
            control_fd = _open_directory_at(root_fd, ".storage-operations")
            try:
                _verify_recorded_identity(
                    root_fd,
                    ".storage-operations.identity",
                    control_fd,
                    label="control directory",
                )
                _verify_directory_at(
                    root_fd, ".storage-operations", control_fd
                )
            except Exception:
                os.close(control_fd)
                raise
    except Exception:
        os.close(root_fd)
        os.close(parent_fd)
        raise
    finally:
        os.close(root_setup_fd)
    try:
        setup_fd, _ = _open_regular_at(control_fd, ".locks-setup.lock")
    except Exception:
        os.close(control_fd)
        os.close(root_fd)
        os.close(parent_fd)
        raise
    try:
        with exclusive_lock(setup_fd):
            locks_fd = _open_directory_at(control_fd, "locks")
            try:
                _verify_recorded_identity(
                    control_fd,
                    ".locks.identity",
                    locks_fd,
                    label="locks directory",
                )
                _verify_directory_at(control_fd, "locks", locks_fd)
            except Exception:
                os.close(locks_fd)
                raise
    except Exception:
        os.close(control_fd)
        os.close(root_fd)
        os.close(parent_fd)
        raise
    finally:
        os.close(setup_fd)
    return parent_fd, root_fd, control_fd, locks_fd


def run_lock_relative_name(session_id: str) -> str:
    """The lock file name both implementations agree on for ``session_id``."""
    return f"run-{hashlib.sha256(session_id.encode('utf-8')).hexdigest()}.lock"


@contextmanager
def session_key_lock(workspace: Path | str, session_id: str) -> Iterator[None]:
    """Hold the stable lock for ``session_id`` across processes.

    Callers must acquire this lock before beginning a SQLite write transaction.
    The persistent locks-directory identity makes directory replacement fail
    closed instead of silently splitting contenders across distinct inodes.
    """

    validate_session_id(session_id)
    root = require_absolute_workspace(workspace)
    opener = _descriptor_run_key_lock if SUPPORTS_DIR_FD else _path_run_key_lock
    with opener(root, session_id):
        yield


@contextmanager
def _descriptor_run_key_lock(root: Path, session_id: str) -> Iterator[None]:
    parent_fd, root_fd, control_fd, locks_fd = _open_control_tree(root)
    lock_name = run_lock_relative_name(session_id)
    try:
        descriptor, _ = _open_regular_at(locks_fd, lock_name)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise SessionFencePathError(f"Session fence is not a regular file: {session_id}")
            with exclusive_lock(descriptor):
                _verify_directory_at(parent_fd, root.name, root_fd)
                _verify_directory_at(root_fd, ".storage-operations", control_fd)
                _verify_directory_at(control_fd, "locks", locks_fd)
                opened = os.fstat(descriptor)
                named = os.stat(lock_name, dir_fd=locks_fd, follow_symlinks=False)
                if not _same_inode(opened, named):
                    raise SessionFencePathError(f"Session fence was replaced: {session_id}")
                yield
        finally:
            os.close(descriptor)
    finally:
        os.close(locks_fd)
        os.close(control_fd)
        os.close(root_fd)
        os.close(parent_fd)


def _open_regular_path(path: Path) -> tuple[int, bool]:
    """Path-based counterpart of :func:`_open_regular_at` for platforms
    without ``dir_fd``. The exclusive-create-then-reopen shape is kept so two
    first creators still resolve to one file rather than one of them failing."""
    for _ in range(_OPEN_RETRIES):
        created = False
        try:
            descriptor = os.open(path, _FILE_FLAGS)
        except OSError as exc:
            if exc.errno not in (errno.ENOENT, errno.EACCES):
                raise SessionFencePathError(f"Cannot safely open run fence {path.name}.") from exc
            try:
                descriptor = os.open(path, _FILE_FLAGS | os.O_CREAT | os.O_EXCL, 0o600)
                created = True
            except OSError as create_exc:
                if create_exc.errno in (errno.EEXIST, errno.ENOENT, errno.EACCES):
                    continue
                raise SessionFencePathError(
                    f"Cannot safely create run fence {path.name}."
                ) from create_exc
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            return descriptor, created
        os.close(descriptor)
        raise SessionFencePathError(f"Session fence is not a regular file: {path.name}")
    raise SessionFencePathError(f"Session fence {path.name} changed during open.")


def _verify_recorded_path_identity(
    identity_path: Path, target: Path, *, label: str
) -> None:
    """Pin a directory to the identity first recorded for it.

    Without ``dir_fd`` the open cannot be proven atomic, so the check runs
    against the resolved directory afterwards: a replaced directory has a new
    ``(st_dev, st_ino)`` and fails closed here.
    """
    identity_fd, created = _open_regular_path(identity_path)
    try:
        info = target.stat()
        identity = _identity_bytes(info)
        if created:
            _write_identity(identity_fd, identity)
            return
        os.lseek(identity_fd, 0, os.SEEK_SET)
        recorded = os.read(identity_fd, 128)
        if recorded == identity:
            return
        if not _identity_names_directory(recorded, info):
            raise SessionFencePathError(f"Session fence {label} identity changed.")
        _write_identity(identity_fd, identity)
    finally:
        os.close(identity_fd)


def _open_directory_path(path: Path, identity_path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise SessionFencePathError(f"Session fence {label} is a symbolic link.")
    path.mkdir(mode=0o700, parents=False, exist_ok=True)
    if not path.is_dir():
        raise SessionFencePathError(f"Session fence {label} is not a directory.")
    _verify_recorded_path_identity(identity_path, path, label=label)


@contextmanager
def _path_run_key_lock(root: Path, session_id: str) -> Iterator[None]:
    """Windows fence: same files and same lock, resolved by path.

    Setup is serialized by the same workspace lock file the descriptor
    implementation uses, so two processes creating the control tree at once
    still agree on one directory identity.
    """
    if not root.name:
        raise SessionFencePathError("Workspace root must not be the filesystem root.")
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
    setup_fd, _ = _open_regular_path(root.parent / f".eda-workspace-{digest}.lock")
    try:
        with exclusive_lock(setup_fd):
            if root.is_symlink() or not root.is_dir():
                raise SessionFencePathError("Workspace root is not a directory.")
            _verify_recorded_path_identity(
                root.parent / f".eda-workspace-{digest}.identity",
                root,
                label="workspace root",
            )
            control = root / ".storage-operations"
            _open_directory_path(
                control, root / ".storage-operations.identity", label="control directory"
            )
            locks = control / "locks"
            _open_directory_path(
                locks, control / ".locks.identity", label="locks directory"
            )
    finally:
        os.close(setup_fd)
    lock_path = root / ".storage-operations" / "locks" / run_lock_relative_name(session_id)
    if lock_path.is_symlink():
        raise SessionFencePathError(f"Session fence is a symbolic link: {session_id}")
    descriptor, _ = _open_regular_path(lock_path)
    try:
        with exclusive_lock(descriptor):
            opened = os.fstat(descriptor)
            named = os.stat(lock_path, follow_symlinks=False)
            if not _same_inode(opened, named):
                raise SessionFencePathError(f"Session fence was replaced: {session_id}")
            yield
    finally:
        os.close(descriptor)
