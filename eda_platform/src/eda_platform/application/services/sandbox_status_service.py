"""Sandbox health read use case — status shown in Settings / Chat.

Resolution is cached for ``CACHE_TTL_SECONDS``: probing Docker shells out twice
with a 2s timeout each, so an unguarded read would let a polling client stall
the worker threadpool.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from eda_platform.application.dto import SandboxStatusView
from eda_platform.core.config import require_absolute_workspace
from eda_platform.core.sandbox import SandboxBackendInfo, SandboxUnavailableError
from eda_platform.core.sandbox_broker import SandboxBroker

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 30.0

# Backend detail strings are useful but can carry a server path. §14: no
# response returns a server file path.
_ABSOLUTE_PATH = re.compile(r"(?<![\w/])/[\w.\-~@+]+(?:/[\w.\-~@+]*)*")


def scrub_paths(text: str) -> str:
    return _ABSOLUTE_PATH.sub("<path>", text)

# The API-facing names (§M6 backends), independent of class attribute names.
_BACKEND_LABELS = {
    "docker": "docker",
}

_UNAVAILABLE_MESSAGE = (
    "Open-ended Python analysis is unavailable: no safe sandbox backend "
    "resolved. Everything else (SQL analysis, reports) still sessions."
)


class SandboxStatusService:
    def __init__(self, workspace: Path) -> None:
        # Never None: SandboxBroker mkdtemp()s a throwaway root when work_root
        # is omitted, which would leak a temp dir per status request. The live
        # canary may create this one stable root, then removes its execution dir.
        self._work_root = require_absolute_workspace(workspace) / "_sandbox"
        self._cached: tuple[float, SandboxStatusView] | None = None

    def get_status(self) -> SandboxStatusView:
        now = time.monotonic()
        if self._cached is not None and now - self._cached[0] < CACHE_TTL_SECONDS:
            return self._cached[1]
        view = self._resolve()
        self._cached = (now, view)
        return view

    def _resolve(self) -> SandboxStatusView:
        try:
            info = SandboxBroker.from_env(work_root=self._work_root).require_safe_backend().info
        except SandboxUnavailableError as exc:
            logger.info("No sandbox backend resolved: %s", exc)
            return SandboxStatusView(
                backend="none",
                available=False,
                safe_for_untrusted_code=False,
                open_python_analysis_available=False,
                detail=scrub_paths(str(exc)),
                message=_UNAVAILABLE_MESSAGE,
            )
        except Exception:  # noqa: BLE001 — a status read must never 500 the page
            return SandboxStatusView(
                backend="none",
                available=False,
                safe_for_untrusted_code=False,
                open_python_analysis_available=False,
                detail="Sandbox backend status could not be determined.",
                message=_UNAVAILABLE_MESSAGE,
            )
        return _to_view(info)


def _to_view(info: SandboxBackendInfo) -> SandboxStatusView:
    backend = _BACKEND_LABELS.get(info.name, info.name)
    usable = info.available and info.safe_for_untrusted_code
    if usable:
        message = f"{backend} sandbox active; open-ended Python analysis is available."
    elif not info.available:
        message = f"{backend} sandbox is unavailable. " + _UNAVAILABLE_MESSAGE
    else:
        message = (
            f"{backend} sandbox is not a security boundary and will not be used "
            "for untrusted code. " + _UNAVAILABLE_MESSAGE
        )
    return SandboxStatusView(
        backend=backend,
        available=info.available,
        safe_for_untrusted_code=info.safe_for_untrusted_code,
        open_python_analysis_available=usable,
        detail=scrub_paths(info.detail),
        message=message,
    )
