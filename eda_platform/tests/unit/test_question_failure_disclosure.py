"""A question that did not answer must say why, in the report and the payload.

The World Cup run executed 7 questions, 2 failed, and nothing anywhere said
why: the trace carried `abstention_code: null`, the report wrote
`(outcome: failed)`, and the reason sat unread in the artifact's `error`
field -- a tool-guard message addressed to the model, ending in "Return
corrected parameters".
"""

from __future__ import annotations

from eda_platform.schemas.questions import (
    QuestionExecutionResult,
    question_failure_reason,
)
from eda_platform.schemas.reports import ReportBundle
from eda_platform.tools.exporter import report_bundle_to_markdown

_GUARD_ERROR = (
    "Tool guard rejected parameters for `execute_question_candidate`.\n"
    "What was wrong:\n"
    "- `plan.sql` got 'SQL containing a JOIN with no declared "
    "required_relations': SQL joins tables but the question declares no "
    "required_relations.\n"
    "Allowed:\n"
    "- `plan.sql`: JOIN SQL only for questions declaring confirmed whitelist "
    "relations. Confirmed joins: (no confirmed joins in the whitelist)\n"
    "How to fix:\n"
    "- `plan.sql`: Declare the join in required_relations using a confirmed "
    "whitelist label, or rewrite the SQL without a JOIN.\n"
    "Return corrected parameters that satisfy these constraints. Do not "
    "compute tool results yourself."
)


def _result(**overrides: object) -> QuestionExecutionResult:
    values: dict[str, object] = {
        "question_id": "q_1",
        "question": "Which team behaviours distinguish attacking pressure?",
        "origin": "llm",
        "status": "failed",
        "outcome": "failed",
    }
    values.update(overrides)
    return QuestionExecutionResult.model_validate(values)


def test_a_guard_rejection_becomes_one_plain_sentence() -> None:
    reason = _result(error=_GUARD_ERROR).failure_reason

    assert "SQL joins tables but the question declares no required_relations." in reason
    # The model-facing repair protocol is not the user's business.
    assert "Return corrected parameters" not in reason
    assert "How to fix" not in reason
    assert "`" not in reason


def test_an_abstention_code_becomes_a_plain_sentence() -> None:
    reason = _result(
        outcome="abstained", abstention_code="agent_no_evidence"
    ).failure_reason

    assert "agent_no_evidence" not in reason
    assert "evidence" in reason.lower()


def test_an_answered_question_carries_no_failure_reason() -> None:
    assert _result(status="succeeded", outcome="answered").failure_reason == ""


def test_an_unmapped_error_still_names_the_specific_problem() -> None:
    reason = _result(error="ValueError: column `spend` is not numeric.").failure_reason

    assert "spend" in reason
    assert "`" not in reason


def test_the_report_says_why_a_focus_question_failed() -> None:
    from eda_platform.agents.reporting import _inject_question_claims

    bundle = ReportBundle.empty(project_id="p", session_id="s")
    _inject_question_claims(bundle, [_result(error=_GUARD_ERROR)])
    markdown = report_bundle_to_markdown(bundle)

    assert "outcome: failed" in markdown
    assert "SQL joins tables but the question declares no required_relations." in markdown


def test_a_database_binder_error_does_not_reach_the_reader() -> None:
    """2026-08-05 live run: three failures pasted DuckDB internals into the report.

    "ValueError: Planner produced invalid SQL after retry: SQL binding failed:
    Binder Error: No function matches the given name and argument types
    'date_diff(STRING_LITERAL, VARCHAR, DATE)'..." is written for whoever debugs
    the planner. The reader needs to know the query did not fit the data; the
    binder text stays in `error` and on the Trace page.
    """
    reason = question_failure_reason(
        error=(
            "ValueError: Planner produced invalid SQL after retry: SQL binding "
            "failed: Binder Error: No function matches the given name and argument "
            "types 'date_diff(STRING_LITERAL, VARCHAR, DATE)'. You might need to "
            "add explicit type casts.\n\tCandidate functions:\n\tdate_diff(VARCHAR, "
            "DATE, DATE) -> BIGINT"
        ),
        abstention_code=None,
    )
    assert "Binder Error" not in reason
    assert "date_diff" not in reason
    assert "ValueError" not in reason
    assert len(reason) < 200
    assert reason.endswith(".")
