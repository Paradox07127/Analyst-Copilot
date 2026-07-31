"""Filesystem helpers whose POSIX behaviour does not carry over to Windows."""

from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path

# Windows opens descriptors in text mode by default, which rewrites \n as \r\n
# on the way out. Every binary payload and every byte-exact readback in this
# codebase needs this flag; it does not exist elsewhere and reads as 0.
BINARY_FLAG = getattr(os, "O_BINARY", 0)


def fsync_directory(path: Path | str) -> None:
    """Flush a directory entry, where the platform has such a thing.

    A rename is only durable on POSIX once the containing directory is synced.
    Windows has no directory descriptor to sync — ``os.open`` on a directory
    fails there outright — and orders metadata itself, so this is a no-op.
    """
    if sys.platform == "win32":  # pragma: no cover - platform branch
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def remove_tree(path: Path | str, *, ignore_errors: bool = False) -> None:
    """Delete a directory tree, including entries that deny deletion.

    The two platforms deny it for different reasons. On POSIX a file's own mode
    is irrelevant and the parent directory's write/execute bits decide; on
    Windows the read-only attribute on the file itself blocks the unlink. The
    sandbox stages inputs at ``0o400`` under directories it also tightens, and
    a quarantined run may hold read-only user files, so a plain ``rmtree``
    leaves those trees behind on every cleanup.
    """

    def clear_and_retry(_function: object, target: str, exception: BaseException) -> None:
        if not isinstance(exception, PermissionError | OSError):
            raise exception
        try:
            if os.path.isdir(target) and not os.path.islink(target):
                # Needs write to unlink children and execute to traverse; the
                # subtree is then retried from the top rather than resumed,
                # because rmtree cannot restart the directory scan it aborted.
                os.chmod(target, stat.S_IRWXU)
                shutil.rmtree(target, onexc=clear_and_retry)
            else:
                os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
                os.unlink(target)
        except OSError:
            if not ignore_errors:
                raise

    try:
        shutil.rmtree(path, onexc=clear_and_retry)
    except OSError:
        if not ignore_errors:
            raise
