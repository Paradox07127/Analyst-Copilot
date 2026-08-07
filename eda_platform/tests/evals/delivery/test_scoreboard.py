"""The failure classifier is the scoreboard's only judgement; pin it on real text.

Every string below is the verbatim `error` of a stored QuestionExecutionResult.
A label that silently drifts to "unclassified" would report the loss without its
cause, which is the one thing SessionMetrics already fails to record.
"""

from __future__ import annotations

from .scoreboard import DeliveryScore, classify_failure

_GUARD_ERROR = (
    "Tool guard rejected parameters for `execute_question_candidate`.\n"
    "What was wrong:\n- `plan.sql` got 'SQL containing a JOIN with no declared "
    "required_relations': SQL joins tables but the question declares no "
    "required_relations."
)
_PLANNER_ERROR = (
    "ValueError: Planner produced invalid SQL after retry: SQL binding failed: "
    "Binder Error: No function matches the given name and argument types "
    "'/(VARCHAR, INTEGER_LITERAL)'."
)
_APPROVAL_ERROR = (
    "Automatic execution stopped: the generated plan requires explicit user "
    "approval. Run it from an approval-aware surface."
)


def test_a_guard_rejection_names_the_guard() -> None:
    assert classify_failure({"error": _GUARD_ERROR}) == "guard:join_scope"


def test_an_exhausted_repair_loop_is_not_a_guard_rejection() -> None:
    """Both mention SQL; only one means the model was given a second chance."""
    assert classify_failure({"error": _PLANNER_ERROR}) == "planner:unrepaired"


def test_an_abstention_code_outranks_the_error_prose() -> None:
    payload = {"abstention_code": "approval_required", "error": _APPROVAL_ERROR}
    assert classify_failure(payload) == "abstain:approval_required"


def test_an_unrecognised_exception_keeps_its_type() -> None:
    assert classify_failure({"error": "QueryTimeout: exceeded 10.0s"}) == "runtime:querytimeout"


def test_prose_with_no_marker_is_admitted_as_unclassified() -> None:
    """A wrong label is worse than an honest gap: never guess from free text."""
    assert classify_failure({"error": "something went sideways"}) == "unclassified"


def test_answer_rate_of_a_run_that_selected_nothing_is_zero() -> None:
    assert DeliveryScore(project="p", session="s").answer_rate == 0.0
