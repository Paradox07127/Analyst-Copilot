"""Translate worker exceptions into user-readable failure text.

The job row keeps a machine-readable ``error_code`` and a human ``message``;
the raw exception text survives in ``detail`` (persisted on the job.failed
trace event) so the Trace page can still show what actually happened.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class LLMNotConfiguredError(RuntimeError):
    """A job queued for a live LLM reached a worker with no provider config."""

    error_code = "llm_not_configured"


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    error_code: str
    message: str
    detail: str | None


def durable_error_code(exc: Exception) -> str:
    """The stable API code a worker persists across process boundaries."""
    code = getattr(exc, "error_code", None)
    if isinstance(code, str) and code:
        return code
    return type(exc).__name__

# Message fragments from provider SDKs and HTTP clients that mean "the model
# call was refused for credentials", regardless of the exception class used.
_LLM_AUTH_PATTERN = re.compile(
    r"api[ _-]?key|unauthorized|authentication|invalid[ _-]?credential", re.IGNORECASE
)

_GENERIC_MESSAGE = (
    "Analysis failed unexpectedly. Run it again; if it keeps failing, "
    "the Trace page has the technical details."
)


def describe_worker_failure(exc: Exception) -> WorkerFailure:
    """One user-readable sentence (with a suggested action) per failure class."""
    code = durable_error_code(exc)
    raw = f"{type(exc).__name__}: {exc}"

    if isinstance(exc, LLMNotConfiguredError):
        return WorkerFailure(
            code,
            "This analysis needed a live language model, but the worker had no "
            "provider configuration. Set up a provider in Settings (or choose "
            "offline mode), then run the analysis again.",
            raw,
        )
    if isinstance(exc, KeyError):
        missing = str(exc).strip("'\"")
        named = f" ({missing})" if missing else ""
        return WorkerFailure(
            code,
            f"The analysis referenced a column or field{named} that the data "
            "does not contain. Check that the selected files have the expected "
            "columns, then run the analysis again.",
            raw,
        )
    if isinstance(exc, FileNotFoundError):
        return WorkerFailure(
            code,
            "A data file this analysis needed could not be found. Re-upload "
            "the file or start a new session with the current data.",
            raw,
        )
    if isinstance(exc, UnicodeDecodeError):
        return WorkerFailure(
            code,
            "A data file could not be read as text. Save it as UTF-8 CSV and "
            "upload it again.",
            raw,
        )
    # Ordering: TimeoutError and ConnectionError are OSError subclasses, so
    # they must be named before the generic file-read branch.
    if isinstance(exc, TimeoutError):
        return WorkerFailure(
            code,
            "Part of the analysis timed out. Run it again; a busy provider or "
            "a very large dataset can cause this.",
            raw,
        )
    if isinstance(exc, ConnectionError):
        return WorkerFailure(
            code,
            "A network connection failed during the analysis. Check the "
            "connection to the model provider, then run the analysis again.",
            raw,
        )
    if isinstance(exc, OSError):
        return WorkerFailure(
            code,
            "A data file could not be read from disk. Re-upload the file, "
            "then run the analysis again.",
            raw,
        )
    if isinstance(exc, MemoryError):
        return WorkerFailure(
            code,
            "The analysis ran out of memory. Try a smaller dataset, fewer "
            "files at once, or a single dataset worker.",
            raw,
        )
    if _LLM_AUTH_PATTERN.search(str(exc)):
        return WorkerFailure(
            code,
            "The language-model provider rejected this run's credentials. "
            "Check the API key in Settings, then run the analysis again.",
            raw,
        )
    # Typed platform errors and ValueErrors already carry sentences written
    # for users (e.g. "question source changed since approval"); passing them
    # through beats burying them under the generic line.
    if isinstance(code, str) and code != type(exc).__name__:
        return WorkerFailure(code, str(exc) or _GENERIC_MESSAGE, raw)
    if isinstance(exc, ValueError) and str(exc).strip():
        return WorkerFailure(code, str(exc), raw)
    return WorkerFailure(code, _GENERIC_MESSAGE, raw)
