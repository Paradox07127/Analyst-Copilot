from __future__ import annotations

from typing import Any

from eda_platform.application.services.compare_matchers import (
    ArtifactComparable,
    Change,
    ExecutionComparable,
    MatchResult,
    MatchStatus,
    QuestionComparable,
    ReportClaimComparable,
    ReportSectionComparable,
    match_artifacts,
    match_execution_records,
    match_questions,
    match_report_claims,
    match_report_sections,
)


def test_question_swap_reverses_added_and_removed() -> None:
    shared_left = QuestionComparable(
        record_id="left-shared",
        lineage_identity="question-1",
        question_text="Which segment grew?",
    )
    shared_right = QuestionComparable(
        record_id="right-shared",
        lineage_identity="question-1",
        question_text="Which segment grew the most?",
    )
    left_only = QuestionComparable(record_id="left-only", question_text="Left only")
    right_only = QuestionComparable(record_id="right-only", question_text="Right only")

    forward = match_questions([left_only, shared_left], [shared_right, right_only])
    reverse = match_questions([right_only, shared_right], [shared_left, left_only])

    assert _changes_by_records(forward) == {
        ("left-only", None): Change.REMOVED,
        ("left-shared", "right-shared"): Change.CHANGED,
        (None, "right-only"): Change.ADDED,
    }
    assert _changes_by_records(reverse) == {
        ("right-only", None): Change.REMOVED,
        ("right-shared", "left-shared"): Change.CHANGED,
        (None, "left-only"): Change.ADDED,
    }


def test_ambiguous_tie_is_not_forced_and_is_input_order_stable() -> None:
    left = QuestionComparable(record_id="left", question_text="Same question")
    right_a = QuestionComparable(record_id="right-a", question_text=" same  question ")
    right_b = QuestionComparable(record_id="right-b", question_text="SAME QUESTION")

    first = match_questions([left], [right_a, right_b])
    reordered = match_questions([left], [right_b, right_a])

    assert _result_signature(first) == _result_signature(reordered)
    assert all(result.match_status is MatchStatus.UNMATCHED for result in first)
    assert _changes_by_records(first) == {
        ("left", None): Change.REMOVED,
        (None, "right-a"): Change.ADDED,
        (None, "right-b"): Change.ADDED,
    }
    assert "ambiguous" in next(result.reason for result in first if result.left is left)


def test_question_rules_use_explicit_priority_before_fingerprint() -> None:
    left = QuestionComparable(
        record_id="left",
        question_text="Revenue?",
        template_id="trend",
        metric_id="revenue",
        target_datasets=("sales",),
        candidate_fingerprint="fingerprint",
    )
    template_match = QuestionComparable(
        record_id="template",
        question_text="Different wording",
        template_id="trend",
        metric_id="revenue",
        target_datasets=("sales",),
        candidate_fingerprint="other",
    )
    fingerprint_match = QuestionComparable(
        record_id="fingerprint",
        question_text="Another wording",
        candidate_fingerprint="fingerprint",
    )

    results = match_questions([left], [fingerprint_match, template_match])

    matched = next(result for result in results if result.left and result.right)
    assert matched.right is template_match
    assert matched.reason == "same template, metric, and target datasets"


def test_report_section_and_claim_matchers_use_semantic_keys() -> None:
    section_results = match_report_sections(
        [ReportSectionComparable(record_id="left", title="Executive   Summary")],
        [ReportSectionComparable(record_id="right", title="executive summary")],
    )
    claim_results = match_report_claims(
        [
            ReportClaimComparable(
                record_id="left-claim",
                section_title="Executive Summary",
                text="Revenue grew 12%.",
                evidence_signature=(("metric", "sales.revenue"),),
            )
        ],
        [
            ReportClaimComparable(
                record_id="right-claim",
                section_title="executive summary",
                text=" revenue grew 12%. ",
                evidence_signature=(("METRIC", "sales.revenue"),),
            )
        ],
    )

    assert section_results[0].change is Change.SAME
    assert section_results[0].version == "report-section-v1"
    assert claim_results[0].change is Change.CHANGED
    assert claim_results[0].match_status is MatchStatus.STRONG


def test_execution_identity_excludes_model_and_configuration() -> None:
    left = ExecutionComparable(
        record_id="left-call",
        parent_match_key="question:q1",
        span_kind="generation",
        operation_name="answer-question",
        occurrence_index=0,
        model="gpt-a",
        model_config={"temperature": 0},
    )
    right = ExecutionComparable(
        record_id="right-call",
        parent_match_key="question:q1",
        span_kind="generation",
        operation_name="answer-question",
        occurrence_index=0,
        model="gpt-b",
        model_config={"temperature": 0.5},
    )

    results = match_execution_records([left], [right])

    assert left.logical_span_key == right.logical_span_key
    assert len(results) == 1
    assert results[0].change is Change.CHANGED
    assert results[0].left is left
    assert results[0].right is right


def test_artifact_matching_uses_logical_parents_not_raw_parent_ids() -> None:
    left = ArtifactComparable(
        record_id="artifact-left",
        artifact_type="analysis_table",
        stable_identity="revenue-by-segment",
        raw_parent_ids=("raw-left-parent",),
        parent_logical_keys=("question:q1",),
    )
    right = ArtifactComparable(
        record_id="artifact-right",
        artifact_type="analysis_table",
        stable_identity="revenue-by-segment",
        raw_parent_ids=("raw-right-parent",),
        parent_logical_keys=("question:q1",),
    )

    results = match_artifacts([left], [right])

    assert left.logical_key == right.logical_key
    assert len(results) == 1
    assert results[0].change is Change.SAME
    assert results[0].right is right


def _changes_by_records(
    results: list[MatchResult[Any]],
) -> dict[tuple[str | None, str | None], Change]:
    changes: dict[tuple[str | None, str | None], Change] = {}
    for result in results:
        left = result.left
        right = result.right
        changes[
            (
                left.record_id if left is not None else None,
                right.record_id if right is not None else None,
            )
        ] = result.change
    return changes


def _result_signature(
    results: list[MatchResult[Any]],
) -> list[tuple[str, str | None, str | None]]:
    signature = []
    for result in results:
        left = result.left
        right = result.right
        signature.append(
            (
                str(result.change),
                left.record_id if left is not None else None,
                right.record_id if right is not None else None,
            )
        )
    return sorted(signature, key=repr)


def test_a_required_section_survives_a_title_rewrite() -> None:
    """Matching on title alone reported one section removed and another added
    whenever the wording changed, even though both fill the same required slot
    (plan section 6.4, tier 1 before tier 2)."""
    left = ReportSectionComparable(
        record_id="l1", title="Executive Summary", required_key="executive_summary"
    )
    right = ReportSectionComparable(
        record_id="r1", title="Summary", required_key="executive_summary"
    )

    [result] = match_report_sections([left], [right])

    assert result.match_status is MatchStatus.STRONG
    assert result.reason == "same required report section"
    assert (result.left, result.right) == (left, right)


def test_free_form_sections_still_fall_back_to_the_normalized_title() -> None:
    left = ReportSectionComparable(record_id="l1", title="Extra notes")
    right = ReportSectionComparable(record_id="r1", title="extra   notes")

    [result] = match_report_sections([left], [right])

    assert result.reason == "same normalized report section title"


def test_two_different_required_slots_are_never_paired_by_similar_titles() -> None:
    left = ReportSectionComparable(
        record_id="l1", title="Summary", required_key="executive_summary"
    )
    right = ReportSectionComparable(
        record_id="r1", title="Summary", required_key="method_summary"
    )

    results = match_report_sections([left], [right])

    assert {result.match_status for result in results} == {MatchStatus.UNMATCHED}
