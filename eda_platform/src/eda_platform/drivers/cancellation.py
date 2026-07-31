"""Shared cooperative-cancellation checkpoint for long-running drivers."""

from __future__ import annotations

from collections.abc import Callable

from eda_platform.core.kernel import SessionCancelled

CancelCheck = Callable[[], bool]


def raise_if_cancelled(
    cancel_check: CancelCheck | None,
    *,
    operation: str,
) -> None:
    """Raise the worker's typed cancellation signal at a safe boundary."""
    if cancel_check is not None and cancel_check():
        raise SessionCancelled(f"{operation} was cancelled.")
