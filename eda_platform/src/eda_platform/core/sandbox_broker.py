from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from eda_platform.core.sandbox import (
    ExecutionBackend,
    SandboxBackendInfo,
    SandboxUnavailableError,
)
from eda_platform.core.sandbox_docker import DockerSandboxBackend

SandboxKind = Literal["auto", "docker"]

_ALLOWED_KINDS: set[str] = {"auto", "docker"}
_TRUTHY = {"1", "true", "yes", "on"}


def sandbox_required_at_startup() -> bool:
    return os.environ.get("EDA_SANDBOX_REQUIRED", "").strip().lower() in _TRUTHY


@dataclass(frozen=True)
class SandboxSettings:
    kind: SandboxKind = "auto"
    work_root: Path | None = None
    docker_image: str | None = None


class SandboxBroker:
    def __init__(self, settings: SandboxSettings) -> None:
        self.settings = settings

    @classmethod
    def from_env(cls, work_root: Path | None = None) -> SandboxBroker:
        raw_kind = os.environ.get("EDA_SANDBOX_BACKEND", "auto")
        kind_value = raw_kind.strip().lower() or "auto"
        if kind_value not in _ALLOWED_KINDS:
            raise SandboxUnavailableError(
                "Unknown EDA_SANDBOX_BACKEND. Use auto or docker."
            )
        raw_image = os.environ.get("EDA_SANDBOX_DOCKER_IMAGE")
        docker_image = raw_image.strip() if raw_image and raw_image.strip() else None
        return cls(
            SandboxSettings(
                kind=cast(SandboxKind, kind_value),
                work_root=work_root,
                docker_image=docker_image,
            )
        )

    def resolve_backend(self) -> ExecutionBackend:
        work_root = self._work_root()
        docker = self._docker_backend(work_root)
        docker_info = docker.info
        if docker_info.available:
            return docker
        raise SandboxUnavailableError(
            docker_info.detail
            or "Docker sandbox unavailable. Install/start Docker and build the sandbox image."
        )

    def require_safe_backend(self) -> ExecutionBackend:
        backend = self.resolve_backend()
        verify_runtime = getattr(backend, "verify_runtime", None)
        info = cast(
            SandboxBackendInfo,
            verify_runtime() if callable(verify_runtime) else backend.info,
        )
        if not info.safe_for_untrusted_code:
            raise SandboxUnavailableError(
                f"Resolved backend {info.name!r} is not safe for untrusted code."
            )
        if not info.available:
            raise SandboxUnavailableError(
                info.detail or f"Resolved backend {info.name!r} is unavailable."
            )
        return backend

    def _work_root(self) -> Path:
        return self.settings.work_root or Path(tempfile.mkdtemp(prefix="eda_sandbox_"))

    def _docker_backend(self, work_root: Path) -> ExecutionBackend:
        if self.settings.docker_image:
            return DockerSandboxBackend(work_root=work_root, image=self.settings.docker_image)
        return DockerSandboxBackend(work_root=work_root)
