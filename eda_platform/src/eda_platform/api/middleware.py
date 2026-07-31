"""ASGI body-size limit.

Starlette buffers an entire multipart body (spooling to disk) before the route
handler ever runs, so a handler-level size check cannot stop an oversized POST
from filling the temp disk. This middleware rejects on Content-Length up front
and, for chunked/lying clients, aborts while counting the actual stream.
"""

from __future__ import annotations

import ipaddress
import json
import re
import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    def __init__(self, app: Callable[..., Awaitable[None]], *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _content_length(scope)
        if declared is not None and declared > self.max_bytes:
            await _send_413(send, self.max_bytes)
            return

        received = 0
        tripped = False
        response_started = False
        override_sent = False

        async def counting_receive() -> Message:
            nonlocal received, tripped
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    tripped = True
                    raise _BodyTooLarge()
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal response_started, override_sent
            # FastAPI wraps any exception from form parsing into a generic 400,
            # so _BodyTooLarge may come back as the app's own error response.
            # Once tripped, replace whatever the app sends with our 413.
            if tripped and not response_started:
                if not override_sent:
                    override_sent = True
                    await _send_413(send, self.max_bytes)
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except _BodyTooLarge:
            if response_started:
                raise
            if not override_sent:
                await _send_413(send, self.max_bytes)


_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_UPLOAD_PATH = re.compile(r"^/api/v1/projects/[^/]+/uploads$")


class DeploymentSecurityMiddleware:
    """Remote-only CSRF and trusted-client upload-rate boundary."""

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        allowed_origins: tuple[str, ...],
        trusted_proxy_ips: tuple[str, ...],
        rate_limit: int | None,
        rate_window_seconds: int,
        rate_checker: Callable[..., tuple[bool, int]],
    ) -> None:
        self.app = app
        self.allowed_origins = frozenset(origin.rstrip("/") for origin in allowed_origins)
        self.trusted_proxy_ips = frozenset(trusted_proxy_ips)
        self.rate_limit = rate_limit
        self.rate_window_seconds = rate_window_seconds
        self.rate_checker = rate_checker

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method = str(scope.get("method", "GET")).upper()
        headers = _headers(scope)
        if method in _UNSAFE_METHODS:
            source = headers.get("origin")
            if source is None:
                source = _referer_origin(headers.get("referer"))
            if (
                source is None
                or source.rstrip("/") not in self.allowed_origins
                or headers.get("x-eda-csrf") != "1"
            ):
                await _send_error(
                    send,
                    403,
                    "csrf_rejected",
                    "Unsafe remote requests require an allowed Origin and X-EDA-CSRF: 1.",
                )
                return
        if (
            method == "POST"
            and self.rate_limit is not None
            and _UPLOAD_PATH.fullmatch(str(scope.get("path", "")))
        ):
            allowed, retry_after = self.rate_checker(
                _trusted_client_id(scope, headers, self.trusted_proxy_ips),
                now=time.time(),
                window_seconds=self.rate_window_seconds,
                limit=self.rate_limit,
            )
            if not allowed:
                await _send_error(
                    send,
                    429,
                    "upload_rate_limited",
                    "Upload rate limit exceeded. Retry after the indicated delay.",
                    headers=[(b"retry-after", str(retry_after).encode("ascii"))],
                )
                return
        await self.app(scope, receive, send)


def _content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers") or []:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


async def _send_413(send: Send, max_bytes: int) -> None:
    await _send_error(
        send,
        413,
        "request_too_large",
        f"Request body exceeds the {max_bytes} byte limit.",
    )


async def _send_error(
    send: Send,
    status: int,
    code: str,
    message: str,
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    body = json.dumps({"error": {"code": code, "message": message}}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                *(headers or []),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _headers(scope: Scope) -> dict[str, str]:
    return {
        name.decode("latin-1").lower(): value.decode("latin-1")
        for name, value in scope.get("headers") or []
    }


def _referer_origin(referer: str | None) -> str | None:
    if not referer:
        return None
    from urllib.parse import urlsplit

    parsed = urlsplit(referer)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _trusted_client_id(
    scope: Scope, headers: dict[str, str], trusted_proxy_ips: frozenset[str]
) -> str:
    client = scope.get("client")
    direct = str(client[0]) if isinstance(client, (tuple, list)) and client else "unknown"
    if direct not in trusted_proxy_ips:
        return direct
    forwarded = [
        item.strip()
        for item in headers.get("x-forwarded-for", "").split(",")
        if item.strip()
    ]
    # Walk from the socket peer toward the browser. A client can prepend a
    # forged XFF value, so trusting the leftmost value would let
    # ``spoofed, real-client`` evade the rate bucket. Only hops to the right of
    # the first untrusted address are allowed to be configured proxies.
    for item in reversed(forwarded):
        try:
            address = str(ipaddress.ip_address(item))
        except ValueError:
            # One unparsable hop must not collapse every remote client into the
            # proxy's shared bucket; stop at it and bill the nearest known peer.
            return direct
        if address not in trusted_proxy_ips:
            return address
    return direct
