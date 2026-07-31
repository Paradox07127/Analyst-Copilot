"""Platform-correct process liveness and termination.

``os.kill`` does not mean the same thing on both platforms. On Windows it maps
every signal except the two console events onto ``TerminateProcess``, so the
POSIX idiom ``os.kill(pid, 0)`` would kill the process it claims to be probing,
and ``signal.SIGKILL`` does not exist at all.

Graceful stop is therefore asymmetric. POSIX sends ``SIGTERM`` and the worker's
handler unwinds. Windows has no equivalent signal for a process without a
shared console, so :func:`request_stop` does nothing there and the grace period
is served by the cooperative cancellation flag the worker already polls;
:func:`force_kill` is the only real termination primitive.
"""

from __future__ import annotations

import os
import signal
import sys
from contextlib import suppress

from eda_platform.core.process_identity import read_process_identity

_IS_WINDOWS = sys.platform == "win32"


def pid_is_alive(pid: int) -> bool:
    """Report whether *pid* currently names a live process.

    This is liveness only, never identity: a recycled PID looks alive here.
    Callers deciding whether to signal must compare a persisted
    :class:`~eda_platform.core.process_identity.ProcessIdentity` first.
    """
    if pid <= 0:
        return False
    if _IS_WINDOWS:  # pragma: no cover - platform branch
        # Reading the birth identity opens the process read-only; on Windows
        # that is the probe, because os.kill(pid, 0) would terminate it.
        return read_process_identity(pid) is not None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def request_stop(pid: int) -> None:
    """Ask *pid* to exit gracefully; a no-op where no such signal exists."""
    if _IS_WINDOWS:  # pragma: no cover - platform branch
        return
    with suppress(ProcessLookupError, PermissionError):
        os.kill(pid, signal.SIGTERM)


def force_kill(pid: int) -> None:
    """Terminate *pid* immediately, without giving it a chance to clean up."""
    with suppress(ProcessLookupError, PermissionError):
        if _IS_WINDOWS:  # pragma: no cover - platform branch
            # Any non-console signal reaches TerminateProcess on Windows.
            os.kill(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGKILL)
