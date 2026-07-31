"""Cross-platform exclusive advisory file locking.

POSIX uses ``flock``: whole-file, advisory, released on close. Windows has no
equivalent, so it locks a single byte at a fixed high offset via
``msvcrt.locking``. That offset is far past any real content, which matters
because Windows locks are *mandatory*: locking byte 0 would make ordinary
reads and writes to the same region fail for other processes.

Locks are exclusive and blocking. Every caller of one lock file must go through
this module, or the two platforms disagree about what is being held.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

# 2**62: past the end of any file this application creates, and well inside the
# signed 64-bit range Windows uses for lock offsets.
_WINDOWS_LOCK_OFFSET = 1 << 62
_WINDOWS_RETRY_SECONDS = 0.05


class FileLockError(RuntimeError):
    """The lock could not be acquired or released."""


if sys.platform == "win32":  # pragma: no cover - platform branch
    import msvcrt
    import time

    def _acquire(descriptor: int) -> None:
        position = os.lseek(descriptor, 0, os.SEEK_CUR)
        os.lseek(descriptor, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        try:
            while True:
                try:
                    # LK_NBLCK rather than LK_LOCK: LK_LOCK gives up after ten
                    # tries and raises, which would turn contention into an
                    # error instead of a wait.
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    return
                except OSError as exc:
                    if exc.errno not in (36, 13):  # EDEADLOCK, EACCES
                        raise FileLockError("Cannot acquire the file lock.") from exc
                    time.sleep(_WINDOWS_RETRY_SECONDS)
        finally:
            os.lseek(descriptor, position, os.SEEK_SET)

    def _release(descriptor: int) -> None:
        position = os.lseek(descriptor, 0, os.SEEK_CUR)
        os.lseek(descriptor, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        except OSError as exc:
            raise FileLockError("Cannot release the file lock.") from exc
        finally:
            os.lseek(descriptor, position, os.SEEK_SET)

else:
    import fcntl

    def _acquire(descriptor: int) -> None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            raise FileLockError("Cannot acquire the file lock.") from exc

    def _release(descriptor: int) -> None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError as exc:
            raise FileLockError("Cannot release the file lock.") from exc


def lock_exclusive(descriptor: int) -> None:
    """Block until this process holds the exclusive lock on *descriptor*."""
    _acquire(descriptor)


def unlock(descriptor: int) -> None:
    """Release a lock taken with :func:`lock_exclusive`."""
    _release(descriptor)


@contextmanager
def exclusive_lock(descriptor: int) -> Iterator[None]:
    """Hold the exclusive lock on *descriptor* for the duration of the block."""
    lock_exclusive(descriptor)
    try:
        yield
    finally:
        unlock(descriptor)
