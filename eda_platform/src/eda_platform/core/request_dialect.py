"""Learn a model's accepted request dialect from the provider's own rejection.

No public catalog encodes whether a model takes ``max_tokens`` or
``max_completion_tokens``: LiteLLM's price/capability table and models.dev both
omit it, and OpenRouter publishes ``supported_parameters`` only for its own
routes. Predicting it per provider is what broke Azure, OpenRouter and every
self-hosted endpoint — the parameter is a property of the model, not the vendor.

The rejection, however, names the offending parameter and usually its
replacement, so the answer is read off the 400 instead of guessed. That also
covers models this repo has never heard of, which is the whole point for local
deployments.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Any

from eda_platform.core.provider_registry import LLMProvider

# Generation knobs only. `model`, `messages`, `tools` and `response_format`
# carry the request's meaning, so a provider that rejects one of those has to
# surface as an error rather than be silently rewritten into something that
# succeeds while asking a different question.
REPAIRABLE_PARAMS = frozenset(
    {
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "tool_choice",
        "parallel_tool_calls",
    }
)

# Two is enough for the observed pairs (token param + temperature) and keeps a
# provider that rejects every retry from becoming an unbounded spend loop.
MAX_PARAM_REPAIRS = 2

_UNSUPPORTED_MARKERS = (
    "unsupported parameter",
    "unsupported value",
    "unrecognized request argument",
    "unknown parameter",
    "extra inputs are not permitted",
    "is not supported with this model",
    "does not support",
)
_REPLACEMENT_RE = re.compile(r"use '([A-Za-z0-9_]+)' instead", re.IGNORECASE)
_QUOTED_RE = re.compile(r"'([A-Za-z0-9_]+)'")


@dataclass(frozen=True)
class ParamRepair:
    action: str  # "rename" | "drop"
    param: str
    replacement: str = ""

    def describe(self) -> str:
        if self.action == "rename":
            return f"{self.param}->{self.replacement}"
        return f"-{self.param}"


_learned: dict[tuple[LLMProvider, str], tuple[ParamRepair, ...]] = {}
_lock = threading.RLock()


def learned_repairs(provider: LLMProvider, model: str) -> tuple[ParamRepair, ...]:
    with _lock:
        return _learned.get((provider, model.strip()), ())


def remember_repair(provider: LLMProvider, model: str, repair: ParamRepair) -> None:
    key = (provider, model.strip())
    with _lock:
        existing = _learned.get(key, ())
        if repair not in existing:
            _learned[key] = (*existing, repair)


def forget_learned_repairs() -> None:
    """Drop the process-local memo. Settings changes and tests both need it:
    the same model id behind a new base URL can be a different server."""
    with _lock:
        _learned.clear()


def apply_learned_repairs(
    provider: LLMProvider,
    model: str,
    body: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Returns the rewritten body and what was rewritten. The second value is
    reported on every call, not just the one that discovered the mismatch — a
    model that permanently needs a different dialect is a catalog gap, and it
    stays visible only if each call keeps saying so."""
    repaired = body
    applied: list[str] = []
    for repair in learned_repairs(provider, model):
        if repair.param in repaired:
            repaired = apply_repair(repaired, repair)
            applied.append(repair.describe())
    return repaired, applied


def apply_repair(body: dict[str, Any], repair: ParamRepair) -> dict[str, Any]:
    updated = dict(body)
    value = updated.pop(repair.param, None)
    if repair.action == "rename" and repair.replacement not in updated:
        updated[repair.replacement] = value
    return updated


def plan_repair(body: dict[str, Any], error_detail: str) -> ParamRepair | None:
    """Read a provider rejection into one safe rewrite, or None to give up.

    Returning None is the default: a rejection this cannot confidently attribute
    to a repairable generation knob has to reach the caller as a failure.
    """
    payload = _error_payload(error_detail)
    message = str(payload.get("message") or error_detail)
    if not any(marker in message.casefold() for marker in _UNSUPPORTED_MARKERS):
        return None

    match = _REPLACEMENT_RE.search(message)
    suggested = match.group(1) if match is not None else ""

    param = _offending_param(payload, message, body, suggested=suggested)
    if param is None:
        return None

    if suggested:
        replacement = suggested
        # An out-of-allowlist "replacement" is a provider steering the request
        # somewhere this code never intended, so it is refused rather than
        # trusted; dropping the parameter is still a safe answer.
        if replacement in REPAIRABLE_PARAMS and replacement != param:
            if replacement in body:
                return ParamRepair(action="drop", param=param)
            return ParamRepair(action="rename", param=param, replacement=replacement)
    return ParamRepair(action="drop", param=param)


def _offending_param(
    payload: dict[str, Any],
    message: str,
    body: dict[str, Any],
    *,
    suggested: str,
) -> str | None:
    named = payload.get("param")
    if isinstance(named, str) and named in REPAIRABLE_PARAMS and named in body:
        return named
    # Providers that omit `param` still quote the parameter in the message.
    # The suggested replacement is quoted there too, and skipping it matters:
    # a request already carrying `max_completion_tokens` would otherwise read
    # "Use 'max_completion_tokens' instead" as an instruction to drop it.
    for candidate in _QUOTED_RE.findall(message):
        if candidate != suggested and candidate in REPAIRABLE_PARAMS and candidate in body:
            return candidate
    return None


def _error_payload(error_detail: str) -> dict[str, Any]:
    try:
        parsed = json.loads(error_detail)
    except (TypeError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    inner = parsed.get("error")
    if isinstance(inner, dict):
        return inner
    return parsed
