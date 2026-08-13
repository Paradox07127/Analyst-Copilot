"""Fail-closed policy for credential-bearing LLM HTTP endpoints."""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from typing import Any
from urllib import request
from urllib.parse import urlsplit

from eda_platform.core.provider_registry import LLMProvider, default_base_url

ENDPOINT_ALLOWLIST_ENV = "EDA_LLM_ENDPOINT_ALLOWLIST"
_CUSTOM_PROVIDERS = frozenset(
    {
        LLMProvider.AZURE_OPENAI,
        LLMProvider.OPENAI_COMPATIBLE,
        LLMProvider.OLLAMA,
        LLMProvider.LM_STUDIO,
    }
)


class EndpointPolicyError(ValueError):
    """An endpoint exceeds the operator-granted network capability."""


class NoCredentialRedirect(request.HTTPRedirectHandler):
    """Never replay an authenticated request to a redirect target."""

    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


def validate_llm_base_url(
    provider: LLMProvider,
    base_url: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Validate and normalize a base URL without performing network I/O."""
    if provider is LLMProvider.OFFLINE:
        if base_url:
            raise EndpointPolicyError("Offline mode cannot configure an HTTP endpoint.")
        return ""

    candidate = (base_url or default_base_url(provider)).rstrip("/")
    if not candidate:
        return ""
    origin, hostname = _parse_endpoint(candidate)

    if provider not in _CUSTOM_PROVIDERS:
        expected, _ = _parse_endpoint(default_base_url(provider))
        if origin != expected:
            raise EndpointPolicyError(
                f"{provider.value} is pinned to {expected}; select an operator-owned "
                "provider for a custom endpoint."
            )
        return candidate

    if provider is LLMProvider.AZURE_OPENAI:
        if origin.startswith("https://") and hostname.endswith(".openai.azure.com"):
            return candidate
        if origin.startswith("https://") and origin in _allowed_origins(environ):
            return candidate
        raise EndpointPolicyError(
            "Azure OpenAI requires HTTPS on *.openai.azure.com or an exact origin in "
            f"{ENDPOINT_ALLOWLIST_ENV}."
        )

    if _is_loopback_hostname(hostname):
        return candidate
    if not origin.startswith("https://"):
        raise EndpointPolicyError("Operator-approved remote LLM endpoints require HTTPS.")
    if origin in _allowed_origins(environ):
        return candidate
    raise EndpointPolicyError(
        "Custom/local LLM endpoints are loopback-only by default; add the exact origin "
        f"to {ENDPOINT_ALLOWLIST_ENV} for an operator-approved remote endpoint."
    )


def build_credential_safe_opener() -> request.OpenerDirector:
    return request.build_opener(NoCredentialRedirect)


def credential_safe_urlopen(
    req: request.Request, *, timeout: float
) -> Any:
    """Open one request with redirects disabled (kept injectable for tests)."""
    return build_credential_safe_opener().open(req, timeout=timeout)


def _parse_endpoint(value: str) -> tuple[str, str]:
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise EndpointPolicyError("LLM endpoint contains an invalid port.") from exc
    hostname = (parsed.hostname or "").lower()
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise EndpointPolicyError(
            "LLM endpoint must be an http(s) URL without credentials, query, or fragment."
        )
    host = f"[{hostname}]" if ":" in hostname else hostname
    default_port = 443 if parsed.scheme.lower() == "https" else 80
    origin = f"{parsed.scheme.lower()}://{host}"
    if port is not None and port != default_port:
        origin += f":{port}"
    return origin, hostname


def _allowed_origins(environ: Mapping[str, str] | None) -> frozenset[str]:
    source = os.environ if environ is None else environ
    origins: set[str] = set()
    for raw in source.get(ENDPOINT_ALLOWLIST_ENV, "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        origin, _ = _parse_endpoint(raw)
        if urlsplit(raw).path not in {"", "/"}:
            raise EndpointPolicyError(
                f"{ENDPOINT_ALLOWLIST_ENV} entries must be exact origins, not URL paths."
            )
        origins.add(origin)
    return frozenset(origins)


def _is_loopback_hostname(hostname: str) -> bool:
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False
