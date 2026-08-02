"""Central config entry points shared by CLI, UI, and future FastAPI workers."""

from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from eda_platform.core.env import parse_env_file

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WORKSPACE_RELATIVE = Path("eda_platform") / "workspace"
_DEFAULT_PORTS = {"http": 80, "https": 443}
_REMOTE_HOST_RE = re.compile(
    r"^(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)(?:\.(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?))*$"
)


class WorkspaceConfigError(ValueError):
    """Raised when an explicitly configured workspace is unsafe or ambiguous."""


class DeploymentConfigError(ValueError):
    """Raised when a remote deployment is not explicitly fenced."""


@dataclass(frozen=True)
class DeploymentConfig:
    mode: str
    allowed_hosts: tuple[str, ...]
    allowed_origins: tuple[str, ...]
    remote_auth_token: str | None
    trusted_proxy_ips: tuple[str, ...]
    project_file_quota: int
    project_byte_quota: int
    concurrent_upload_quota: int
    upload_rate_limit: int | None
    upload_rate_window_seconds: int

    @property
    def remote(self) -> bool:
        return self.mode == "remote"


def deployment_config(*, repo_root: Path | None = None) -> DeploymentConfig:
    """Load the local/remote security boundary.

    Local remains the zero-configuration desktop default. Remote is fail-closed:
    public hosts and exact browser origins must both be supplied.
    """
    root = (repo_root if repo_root is not None else REPO_ROOT).resolve()
    values = {**parse_env_file(root / ".env"), **os.environ}
    mode = values.get("EDA_DEPLOYMENT_MODE", "local").strip().lower()
    if mode not in {"local", "remote"}:
        raise DeploymentConfigError("EDA_DEPLOYMENT_MODE must be 'local' or 'remote'.")
    if mode == "local":
        hosts = ("localhost", "127.0.0.1", "testserver")
        origins: tuple[str, ...] = ()
        remote_auth_token = None
        default_files, default_bytes, default_concurrent = 10_000, 1 << 40, 8
        default_rate: int | None = None
    else:
        hosts = tuple(item.lower() for item in _csv_env("EDA_ALLOWED_HOSTS", values))
        raw_origins = _csv_env("EDA_ALLOWED_ORIGINS", values)
        origins_list: list[str] = []
        if not hosts or not raw_origins:
            raise DeploymentConfigError(
                "Remote mode requires EDA_ALLOWED_HOSTS and EDA_ALLOWED_ORIGINS."
            )
        remote_auth_token = _remote_auth_token(values)
        for host in hosts:
            if (
                "://" in host
                or "/" in host
                or host == "*"
                or not _REMOTE_HOST_RE.fullmatch(host)
            ):
                raise DeploymentConfigError(
                    "EDA_ALLOWED_HOSTS entries must be exact host names, not URLs or wildcards."
                )
        for origin in raw_origins:
            origin = origin.rstrip("/")
            parsed = urlsplit(origin)
            try:
                port = parsed.port
            except ValueError as exc:
                raise DeploymentConfigError(
                    "EDA_ALLOWED_ORIGINS entries must be exact http(s) origins."
                ) from exc
            if (
                parsed.scheme.lower() != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or "*" in parsed.hostname
                or parsed.path
                or parsed.query
                or parsed.fragment
                or (port is not None and not 1 <= port <= 65535)
            ):
                raise DeploymentConfigError(
                    "EDA_ALLOWED_ORIGINS entries must be exact HTTPS origins."
                )
            # Browsers omit the scheme's default port when serializing Origin,
            # so keeping ':443' here would reject every request from that host.
            # urlsplit also strips an IPv6 host's brackets; they must go back on
            # or the rebuilt origin can never match what the browser sends.
            scheme = parsed.scheme.lower()
            host = parsed.hostname.lower()
            if ":" in host:
                host = f"[{host}]"
            explicit_port = (
                port
                if port is not None and port != _DEFAULT_PORTS[scheme]
                else None
            )
            origins_list.append(
                f"{scheme}://{host}:{explicit_port}"
                if explicit_port is not None
                else f"{scheme}://{host}"
            )
        origins = tuple(dict.fromkeys(origins_list))
        default_files, default_bytes, default_concurrent = 100, 20 << 30, 2
        default_rate = 30
    trusted_proxy_ips = _csv_env("EDA_TRUSTED_PROXY_IPS", values)
    for proxy_ip in trusted_proxy_ips:
        try:
            ipaddress.ip_address(proxy_ip)
        except ValueError as exc:
            raise DeploymentConfigError(
                "EDA_TRUSTED_PROXY_IPS entries must be exact IP addresses."
            ) from exc
    return DeploymentConfig(
        mode=mode,
        allowed_hosts=hosts,
        allowed_origins=origins,
        remote_auth_token=remote_auth_token,
        trusted_proxy_ips=trusted_proxy_ips,
        project_file_quota=_positive_env(
            "EDA_PROJECT_UPLOAD_FILE_QUOTA", default_files, values
        ),
        project_byte_quota=_positive_env(
            "EDA_PROJECT_UPLOAD_BYTE_QUOTA", default_bytes, values
        ),
        concurrent_upload_quota=_positive_env(
            "EDA_PROJECT_CONCURRENT_UPLOAD_QUOTA", default_concurrent, values
        ),
        upload_rate_limit=(
            _positive_env("EDA_UPLOAD_RATE_LIMIT", default_rate, values)
            if default_rate is not None
            else None
        ),
        upload_rate_window_seconds=_positive_env(
            "EDA_UPLOAD_RATE_WINDOW_SECONDS", 60, values
        ),
    )


def _remote_auth_token(values: Mapping[str, str]) -> str:
    token = values.get("EDA_REMOTE_AUTH_TOKEN", "").strip()
    if len(token) < 32 or any(
        ord(character) < 33 or ord(character) == 127 for character in token
    ):
        raise DeploymentConfigError(
            "Remote mode requires EDA_REMOTE_AUTH_TOKEN with at least 32 visible characters."
        )
    return token


def _csv_env(name: str, values: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(part.strip() for part in values.get(name, "").split(",") if part.strip())
    )


def _positive_env(name: str, default: int, values: Mapping[str, str]) -> int:
    raw = values.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise DeploymentConfigError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise DeploymentConfigError(f"{name} must be a positive integer.")
    return value


def resolve_workspace_path(
    explicit: Path | str | None = None,
    *,
    repo_root: Path | None = None,
) -> Path:
    """Return one absolute workspace path independent of the process CWD.

    Resolution order is an explicit caller value, ``EDA_WORKSPACE`` from the
    process environment, ``.env`` anchored at the repository root, then the
    repository's canonical ``eda_platform/workspace`` directory.
    """
    root = (repo_root if repo_root is not None else REPO_ROOT).resolve()
    if explicit is not None:
        return require_absolute_workspace(explicit)

    env_value = os.environ.get("EDA_WORKSPACE", "").strip()
    if not env_value:
        env_value = parse_env_file(root / ".env").get("EDA_WORKSPACE", "").strip()
    if env_value:
        return require_absolute_workspace(env_value, source="EDA_WORKSPACE")
    return (root / DEFAULT_WORKSPACE_RELATIVE).resolve()


def default_workspace() -> Path:
    """Return the canonical absolute workspace for CLI, API, UI, and workers."""
    return resolve_workspace_path()


def require_absolute_workspace(
    value: Path | str,
    *,
    source: str = "workspace",
) -> Path:
    """Validate an explicit workspace at every public process/write boundary.

    Relative values are ambiguous because ``Path.resolve()`` interprets them
    against the current process directory.  Callers must choose and pass an
    absolute workspace instead.
    """
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise WorkspaceConfigError(f"{source} must be an absolute path, got: {str(value)!r}")
    return candidate.resolve()
