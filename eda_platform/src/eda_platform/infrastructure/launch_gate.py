"""The parent/child handshake that gates a worker's first unit of work.

The ordering the gate exists to enforce is the same on both platforms:

1. the child starts and reports that it is alive,
2. the parent reads the child's birth identity and persists it,
3. only then is the child released to claim its job.

Without step 2 completing first, a cancel arriving in the gap has no recorded
identity to authorize against, and the job would run uncancellable.

Two transports, because POSIX and Windows do not share one. POSIX inherits an
anonymous pipe pair, which no other process can reach. Windows cannot inherit
arbitrary descriptors (``pass_fds`` is rejected) and cannot ``select`` on a
pipe, so it rendezvous over a loopback socket; the child proves it is the
intended process by presenting the launch token, which is the same secret the
job row already requires.
"""

from __future__ import annotations

import os
import socket
import sys
from abc import ABC, abstractmethod

START_ACK_TIMEOUT_SECONDS = 10.0
GATE_RELEASE = b"G"
GATE_ACK = b"R"


class LaunchGateError(RuntimeError):
    """The child never acknowledged, or acknowledged with the wrong identity."""


class ParentLaunchGate(ABC):
    """Parent half: hand `child_argument()` to the child, then drive the gate."""

    @abstractmethod
    def child_argument(self) -> str:
        """The single argv token the child needs to find this gate."""

    @abstractmethod
    def inheritable_descriptors(self) -> tuple[int, ...]:
        """Descriptors the child must inherit, for ``pass_fds``."""

    @abstractmethod
    def wait_for_acknowledgement(self, timeout: float) -> None:
        """Block until the child reports it is alive, or raise."""

    @abstractmethod
    def release(self) -> None:
        """Let the acknowledged child proceed."""

    @abstractmethod
    def close(self) -> None:
        """Release every resource; safe to call more than once."""

    def __enter__(self) -> ParentLaunchGate:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class _PipeParentGate(ParentLaunchGate):
    """POSIX: an inherited anonymous pipe pair, unreachable by other processes."""

    def __init__(self, token: str) -> None:
        self._token = token
        self._gate_read, self._gate_write = os.pipe()
        self._ready_read, self._ready_write = os.pipe()
        self._child_side_closed = False

    def child_argument(self) -> str:
        return f"fd:{self._gate_read}:{self._ready_write}"

    def inheritable_descriptors(self) -> tuple[int, ...]:
        return (self._gate_read, self._ready_write)

    def wait_for_acknowledgement(self, timeout: float) -> None:
        import select

        self._close_child_side()
        readable, _, _ = select.select([self._ready_read], [], [], timeout)
        if not readable or os.read(self._ready_read, 1) != GATE_ACK:
            raise LaunchGateError("Worker did not acknowledge the launch gate.")

    def release(self) -> None:
        os.write(self._gate_write, GATE_RELEASE)

    def close(self) -> None:
        self._close_child_side()
        for descriptor in ("_gate_write", "_ready_read"):
            value = getattr(self, descriptor, -1)
            if value >= 0:
                os.close(value)
                setattr(self, descriptor, -1)

    def _close_child_side(self) -> None:
        """Drop the parent's copies so a dead child yields EOF, not a hang."""
        if self._child_side_closed:
            return
        self._child_side_closed = True
        os.close(self._gate_read)
        os.close(self._ready_write)


class _SocketParentGate(ParentLaunchGate):
    """Windows: a loopback rendezvous, authenticated by the launch token.

    The listener is bound to 127.0.0.1 on an ephemeral port and accepts exactly
    one connection. Another local process could connect first, so the token is
    checked before the gate is considered acknowledged.
    """

    def __init__(self, token: str) -> None:
        self._token = token
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self._connection: socket.socket | None = None

    def child_argument(self) -> str:
        return f"tcp:{self._listener.getsockname()[1]}"

    def inheritable_descriptors(self) -> tuple[int, ...]:
        return ()

    def wait_for_acknowledgement(self, timeout: float) -> None:
        self._listener.settimeout(timeout)
        try:
            connection, _ = self._listener.accept()
        except (TimeoutError, OSError) as exc:
            raise LaunchGateError("Worker did not acknowledge the launch gate.") from exc
        connection.settimeout(timeout)
        self._connection = connection
        expected = GATE_ACK + self._token.encode("utf-8")
        try:
            received = _recv_exactly(connection, len(expected))
        except OSError as exc:
            raise LaunchGateError("Worker did not acknowledge the launch gate.") from exc
        if received != expected:
            raise LaunchGateError("Launch gate acknowledgement had the wrong token.")

    def release(self) -> None:
        if self._connection is None:
            raise LaunchGateError("Launch gate was released before acknowledgement.")
        self._connection.sendall(GATE_RELEASE)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._listener.close()


def open_parent_gate(token: str) -> ParentLaunchGate:
    """Create the gate half that belongs to the spawning process."""
    if sys.platform == "win32":  # pragma: no cover - platform branch
        return _SocketParentGate(token)
    return _PipeParentGate(token)


def acknowledge_and_wait(argument: str, token: str, timeout: float) -> bool:
    """Child half: report alive, then block until the parent releases.

    Returns false when the parent went away without releasing, which the worker
    must treat as "do not start", never as a release.
    """
    kind, _, rest = argument.partition(":")
    if kind == "fd":
        gate_fd, _, ready_fd = rest.partition(":")
        return _acknowledge_over_pipe(int(gate_fd), int(ready_fd))
    if kind == "tcp":
        return _acknowledge_over_socket(int(rest), token, timeout)
    raise SystemExit("unrecognized launch gate argument")


def _acknowledge_over_pipe(gate_fd: int, ready_fd: int) -> bool:
    try:
        os.write(ready_fd, GATE_ACK)
    finally:
        os.close(ready_fd)
    try:
        return os.read(gate_fd, 1) == GATE_RELEASE
    finally:
        os.close(gate_fd)


def _acknowledge_over_socket(
    port: int, token: str, timeout: float
) -> bool:  # pragma: no cover - platform branch
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        try:
            connection.connect(("127.0.0.1", port))
            connection.sendall(GATE_ACK + token.encode("utf-8"))
            return _recv_exactly(connection, 1) == GATE_RELEASE
        except OSError:
            return False


def _recv_exactly(connection: socket.socket, size: int) -> bytes:
    received = bytearray()
    while len(received) < size:
        chunk = connection.recv(size - len(received))
        if not chunk:
            break
        received.extend(chunk)
    return bytes(received)
