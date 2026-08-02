"""E2 ContextAssembler: bounded, deterministic projection of journal state (R3.5)."""

from __future__ import annotations

from typing import Any

import pytest

from eda_platform.core.context_assembler import (
    DEFAULT_SECTION_CHAR_LIMITS,
    SECTION_ORDER,
    ContextView,
    Section,
    assemble,
)
from eda_platform.core.exploration_budget import ToolCallLedger
from eda_platform.schemas.exploration import (
    ExplorationLoopState,
    ExplorationStateUnavailableError,
)
from eda_platform.schemas.exploration_budget import (
    ExplorationBudgetPolicy,
    SessionBudgetPolicyModel,
)


def _policy(**overrides: Any) -> ExplorationBudgetPolicy:
    fields: dict[str, Any] = {
        "llm": SessionBudgetPolicyModel(),
        "max_successful_tool_calls": 10,
        "max_tool_calls_by_kind": {"run_open_analysis": 5, "profile_slice": 5},
        "max_rows_scanned": None,
        "max_result_cells": None,
        "idle_timeout_seconds": 60.0,
        "max_rounds": 8,
    }
    fields.update(overrides)
    return ExplorationBudgetPolicy(**fields)


def _state(**overrides: Any) -> ExplorationLoopState:
    fields: dict[str, Any] = {
        "exploration_id": "expl_1",
        "policy_fingerprint": "xplcy_base_fingerprint_000000001",
        "effective_policy_fingerprint": "xplcy_effective_fingerprint_002",
        "code_fingerprint": "code_1",
        "data_state_witness": "witness_sha256_abcdef0123456789",
        "attempt_epoch": 0,
        "max_llm_requests": 10,
        "max_successful_tool_calls": 10,
        "max_rounds": 8,
        "remaining_llm_call_budget": 10,
        "remaining_tool_call_budget": 10,
        "remaining_round_budget": 8,
        "last_seq": 0,
    }
    fields.update(overrides)
    return ExplorationLoopState(**fields)


def _section(view: ContextView, name: str) -> Section:
    matches = [section for section in view.sections if section.name == name]
    assert len(matches) == 1
    return matches[0]


# --- shape, order, placeholders ---


def test_sections_follow_the_fixed_order_with_explicit_placeholders() -> None:
    view = assemble(_state(), ToolCallLedger(_policy()))
    assert tuple(section.name for section in view.sections) == SECTION_ORDER
    assert "not yet available" in _section(view, "frontier").text
    assert "not yet available" in _section(view, "insight_ledger").text
    assert not _section(view, "frontier").truncated
    rendered = view.render()
    for name in SECTION_ORDER:
        assert f"[{name}]" in rendered


def test_assemble_is_deterministic_byte_for_byte() -> None:
    def build() -> str:
        ledger = ToolCallLedger(_policy())
        ledger.record_success("profile_slice", rows_scanned=100, result_cells=40)
        state = _state(
            failure_history=["boom one", "boom two"],
            step_receipt_refs={"step_b": "rcpt_2", "step_a": "rcpt_1"},
            completed_step_ids=["step_b", "step_a"],
            tool_calls_committed=2,
            remaining_tool_call_budget=8,
        )
        return assemble(state, ledger).render()

    assert build() == build()


def test_receipt_dict_insertion_order_does_not_leak_into_the_render() -> None:
    def build(refs: dict[str, str], steps: list[str]) -> str:
        return assemble(
            _state(
                step_receipt_refs=refs,
                completed_step_ids=steps,
                tool_calls_committed=2,
                remaining_tool_call_budget=8,
            ),
            ToolCallLedger(_policy()),
        ).render()

    forward = build({"step_a": "rcpt_1", "step_b": "rcpt_2"}, ["step_a", "step_b"])
    reversed_ = build({"step_b": "rcpt_2", "step_a": "rcpt_1"}, ["step_b", "step_a"])
    assert forward == reversed_


# --- governance: never truncated ---


def test_governance_facts_survive_the_smallest_possible_budgets() -> None:
    state = _state(amendment_ids=["amend_alpha", "amend_beta"])
    tiny = {name: 1 for name in DEFAULT_SECTION_CHAR_LIMITS}
    view = assemble(state, ToolCallLedger(_policy()), char_limits=tiny)
    governance = _section(view, "governance")
    assert not governance.truncated
    assert governance.omitted_count == 0
    rendered = view.render()
    assert state.policy_fingerprint in rendered
    assert state.effective_policy_fingerprint in rendered
    assert state.data_state_witness in rendered
    assert "amend_alpha" in rendered
    assert "amend_beta" in rendered
    for name in DEFAULT_SECTION_CHAR_LIMITS:
        assert _section(view, name).truncated


def test_a_char_limit_for_governance_is_rejected() -> None:
    with pytest.raises(ValueError, match="governance"):
        assemble(
            _state(),
            ToolCallLedger(_policy()),
            char_limits={"governance": 10},
        )


# --- budget section ---


def test_budget_section_reflects_state_ledger_and_finalization_reserve() -> None:
    policy = _policy(
        llm=SessionBudgetPolicyModel(max_requests=20, protected_requests=2),
    )
    ledger = ToolCallLedger(policy)
    ledger.record_success("profile_slice", rows_scanned=100, result_cells=40)
    state = _state(
        llm_calls_settled=3,
        remaining_llm_call_budget=7,
        tool_calls_committed=1,
        remaining_tool_call_budget=9,
        rounds_started=2,
        rounds_settled=2,
        remaining_round_budget=6,
        completed_step_ids=["s1"],
        last_seq=9,
    )
    text = _section(assemble(state, ledger), "budget").text
    assert "remaining_llm_call_budget: 7" in text
    assert "remaining_tool_call_budget: 9" in text
    assert "remaining_round_budget: 6" in text
    assert "successful_tool_calls: 9" in text
    assert "profile_slice: 4" in text
    assert "run_open_analysis: 5" in text
    assert "rows_scanned: unlimited" in text
    assert "finalization_reserve requests: reserved=2" in text


def test_unlimited_llm_budget_renders_as_unlimited() -> None:
    state = _state(max_llm_requests=None, remaining_llm_call_budget=None)
    text = _section(assemble(state, ToolCallLedger(_policy())), "budget").text
    assert "remaining_llm_call_budget: unlimited" in text


# --- recent failures: investigation-loop compaction semantics (5 x 160) ---


def test_failure_history_keeps_the_last_five_compressed_entries() -> None:
    noisy = "line one\n\t line two   " + "x" * 400
    history = [f"failure {i}: {noisy}" for i in range(7)]
    view = assemble(_state(failure_history=history), ToolCallLedger(_policy()))
    section = _section(view, "recent_failures")
    assert section.truncated
    assert section.omitted_count == 2
    assert "showing last 5 of 7 failures" in section.text
    assert "failure 0" not in section.text
    assert "failure 1" not in section.text
    assert "failure 2" in section.text
    for line in section.text.splitlines():
        if line.startswith("- "):
            assert len(line) <= 2 + 160
            assert "\t" not in line


def test_empty_failure_history_is_explicit() -> None:
    section = _section(
        assemble(_state(), ToolCallLedger(_policy())), "recent_failures"
    )
    assert section.text == "(none)"
    assert not section.truncated


# --- char budgets: tail truncation with explicit markers ---


def test_receipts_over_budget_are_tail_truncated_with_a_marker() -> None:
    refs = {f"step_{i:02d}": f"rcpt_{i:02d}" for i in range(8)}
    state = _state(
        step_receipt_refs=refs,
        completed_step_ids=list(refs),
        tool_calls_committed=8,
        remaining_tool_call_budget=2,
    )
    view = assemble(
        state, ToolCallLedger(_policy()), char_limits={"receipts": 80}
    )
    section = _section(view, "receipts")
    assert section.truncated
    assert section.omitted_count > 0
    assert "step_00" in section.text
    assert "step_07" not in section.text
    marker = section.text.splitlines()[-1]
    assert "truncated" in marker
    assert str(section.omitted_count) in marker
    assert "chars" in marker


# --- provider protocol (frontier / insight_ledger, E4a) ---


def test_provider_output_is_injected_and_bounded() -> None:
    providers = {"frontier": lambda state: "hypothesis A\nhypothesis B"}
    view = assemble(_state(), ToolCallLedger(_policy()), providers=providers)
    assert "hypothesis A" in _section(view, "frontier").text
    bounded = assemble(
        _state(),
        ToolCallLedger(_policy()),
        providers={"frontier": lambda state: "\n".join(f"h{i}" * 30 for i in range(9))},
        char_limits={"frontier": 100},
    )
    assert _section(bounded, "frontier").truncated


def test_unknown_provider_keys_fail_closed() -> None:
    with pytest.raises(ValueError, match="catalog"):
        assemble(
            _state(),
            ToolCallLedger(_policy()),
            providers={"catalog": lambda state: "nope"},
        )


def test_unrestored_field_errors_from_providers_are_not_swallowed() -> None:
    providers = {"frontier": lambda state: state.require_frontier_digest()}
    with pytest.raises(ExplorationStateUnavailableError):
        assemble(_state(), ToolCallLedger(_policy()), providers=providers)
