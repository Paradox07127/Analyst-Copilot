"""Application-layer ports (§4.3): services depend on these Protocols, never on
a concrete queue/process implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "cancelled"})


@dataclass(frozen=True)
class JobCommand:
    """Coarse-grained work order handed to a backend. Primitive fields only:
    the local backend crosses a spawn/pickle boundary with them."""

    job_id: str
    session_id: str
    project_id: str
    kind: str
    params_json: str
    env: dict[str, str] | None = None
    """Extra environment for the worker process. Secrets travel here, never in
    ``params_json``. Durable params are read from SQLite after the launch gate;
    the environment overlay is applied only to the child process."""


@dataclass(frozen=True)
class JobRef:
    job_id: str
    pid: int | None = None


class JobBackend(Protocol):
    def enqueue(self, command: JobCommand) -> JobRef: ...

    def cancel(self, job_id: str) -> None: ...

    def status(self, job_id: str) -> str: ...
