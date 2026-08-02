"""Deterministic ContextAssembler: the journal is the fact, context is a bounded
projection (plan R3.5/§5.4). Same state + ledger must render byte-identical output;
governance facts (fingerprints, witness, amendments) are never compacted away."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from eda_platform.core.exploration_budget import ToolCallLedger, finalization_view
from eda_platform.schemas.exploration import ExplorationLoopState

SECTION_ORDER: tuple[str, ...] = (
    "governance",
    "budget",
    "frontier",
    "recent_failures",
    "receipts",
    "insight_ledger",
)

PROVIDER_SECTIONS: frozenset[str] = frozenset({"frontier", "insight_ledger"})

# Per-section character caps (deterministic; deliberately no tokenizer dependency).
# governance is absent by construction: it is never truncated.
DEFAULT_SECTION_CHAR_LIMITS: Mapping[str, int] = {
    "budget": 1_200,
    "frontier": 2_000,
    "recent_failures": 1_000,
    "receipts": 1_200,
    "insight_ledger": 2_000,
}

# Failure compaction mirrors agents/investigation_loop.py (5 entries x 160 chars).
FAILURE_HISTORY_LIMIT = 5
FAILURE_ENTRY_MAX_CHARS = 160

_PLACEHOLDER = "(not yet available)"


class SectionProvider(Protocol):
    """E4a injects frontier / insight_ledger content through this callable.

    Must be deterministic for a given state and must raise (not default) when it
    reads state that is not restored, e.g. via the require_* accessors."""

    def __call__(self, state: ExplorationLoopState) -> str: ...


@dataclass(frozen=True, slots=True)
class Section:
    name: str
    text: str
    truncated: bool
    omitted_count: int


@dataclass(frozen=True, slots=True)
class ContextView:
    sections: tuple[Section, ...]

    def render(self) -> str:
        return "\n".join(f"[{section.name}]\n{section.text}" for section in self.sections)


def assemble(
    state: ExplorationLoopState,
    ledger: ToolCallLedger,
    providers: Mapping[str, SectionProvider] | None = None,
    *,
    char_limits: Mapping[str, int] | None = None,
) -> ContextView:
    """Build the bounded ContextView for one supervisor turn; render() is the
    fixed-format system reminder text (§5.4 soft countdown lives in `budget`)."""
    providers = providers or {}
    unknown = sorted(set(providers) - PROVIDER_SECTIONS)
    if unknown:
        raise ValueError(f"unknown provider sections: {', '.join(unknown)}.")
    limits = _resolve_limits(char_limits)

    sections: list[Section] = []
    for name in SECTION_ORDER:
        if name in PROVIDER_SECTIONS:
            lines = _provider_lines(name, state, providers)
            pre_omitted = 0
        elif name == "governance":
            lines, pre_omitted = _governance_lines(state), 0
        elif name == "budget":
            lines, pre_omitted = _budget_lines(state, ledger), 0
        elif name == "recent_failures":
            lines, pre_omitted = _failure_lines(state)
        else:  # receipts
            lines, pre_omitted = _receipt_lines(state), 0
        sections.append(_fit(name, lines, limits.get(name), pre_omitted))
    return ContextView(sections=tuple(sections))


def _resolve_limits(overrides: Mapping[str, int] | None) -> Mapping[str, int]:
    if overrides is None:
        return DEFAULT_SECTION_CHAR_LIMITS
    unknown = sorted(set(overrides) - set(DEFAULT_SECTION_CHAR_LIMITS))
    if unknown:
        raise ValueError(
            "char_limits allows only truncatable sections "
            f"(governance is never truncated): {', '.join(unknown)}."
        )
    for name, limit in overrides.items():
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError(f"char_limits[{name!r}] must be a positive integer.")
    return {**DEFAULT_SECTION_CHAR_LIMITS, **overrides}


def _governance_lines(state: ExplorationLoopState) -> list[str]:
    amendments = ", ".join(state.amendment_ids) if state.amendment_ids else "(none)"
    return [
        f"policy_fingerprint: {state.policy_fingerprint}",
        f"effective_policy_fingerprint: {state.effective_policy_fingerprint}",
        f"data_state_witness: {state.data_state_witness}",
        f"amendment_ids: {amendments}",
    ]


def _fmt(value: object) -> str:
    return "unlimited" if value is None else str(value)


def _budget_lines(state: ExplorationLoopState, ledger: ToolCallLedger) -> list[str]:
    lines = [
        f"remaining_llm_call_budget: {_fmt(state.remaining_llm_call_budget)}",
        f"remaining_tool_call_budget: {state.remaining_tool_call_budget}",
        f"remaining_round_budget: {state.remaining_round_budget}",
    ]
    remaining = ledger.remaining()
    lines.append(f"tool_ledger successful_tool_calls: {remaining['successful_tool_calls']}")
    by_kind = remaining["tool_calls_by_kind"]
    for kind in sorted(by_kind):
        lines.append(f"tool_ledger by_kind {kind}: {by_kind[kind]}")
    lines.append(f"tool_ledger rows_scanned: {_fmt(remaining['rows_scanned'])}")
    lines.append(f"tool_ledger result_cells: {_fmt(remaining['result_cells'])}")
    for dimension, entry in finalization_view(ledger.policy).items():
        left = "n/a" if entry["remaining"] is None else str(entry["remaining"])
        lines.append(
            f"finalization_reserve {dimension}: reserved={entry['reserved']}, remaining={left}"
        )
    return lines


def _failure_lines(state: ExplorationLoopState) -> tuple[list[str], int]:
    history = state.failure_history
    if not history:
        return ["(none)"], 0
    recent = history[-FAILURE_HISTORY_LIMIT:]
    lines = [
        "- " + " ".join(entry.split())[:FAILURE_ENTRY_MAX_CHARS] for entry in recent
    ]
    omitted = len(history) - len(recent)
    if omitted:
        lines.insert(0, f"[showing last {len(recent)} of {len(history)} failures]")
    return lines, omitted


def _receipt_lines(state: ExplorationLoopState) -> list[str]:
    refs = state.step_receipt_refs
    if not refs:
        return ["(none)"]
    lines = [f"receipts_committed: {len(refs)}"]
    lines.extend(f"- {step} -> {refs[step]}" for step in sorted(refs))
    return lines


def _provider_lines(
    name: str,
    state: ExplorationLoopState,
    providers: Mapping[str, SectionProvider],
) -> list[str]:
    provider = providers.get(name)
    if provider is None:
        return [f"{name}: {_PLACEHOLDER}"]
    text = provider(state)
    if not isinstance(text, str):
        raise TypeError(f"provider for {name!r} must return str, got {type(text).__name__}.")
    return text.splitlines() or ["(empty)"]


def _fit(name: str, lines: list[str], limit: int | None, pre_omitted: int) -> Section:
    body = "\n".join(lines)
    if limit is None or len(body) <= limit:
        return Section(
            name=name,
            text=body,
            truncated=pre_omitted > 0,
            omitted_count=pre_omitted,
        )
    kept: list[str] = []
    used = 0
    for line in lines:
        cost = len(line) + (1 if kept else 0)
        if used + cost > limit:
            break
        kept.append(line)
        used += cost
    if not kept:
        kept = [lines[0][:limit]]
    kept_text = "\n".join(kept)
    omitted_lines = len(lines) - len(kept)
    omitted_chars = len(body) - len(kept_text)
    marker = f"[truncated: {omitted_lines} entries, {omitted_chars} chars omitted]"
    return Section(
        name=name,
        text=f"{kept_text}\n{marker}",
        truncated=True,
        omitted_count=pre_omitted + omitted_lines,
    )
