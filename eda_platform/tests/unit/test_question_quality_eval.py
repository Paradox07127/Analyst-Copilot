from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from eda_platform.tools.question_quality_eval import (
    QuestionReview,
    ReviewSubmission,
    evaluate_question_reviews,
    prepare_question_reviews,
)

ASSETS = Path(__file__).parents[1] / "evals/question_quality"


def _inputs() -> tuple[dict, str, ReviewSubmission]:
    data = json.loads((ASSETS / "judge_calibration.json").read_text())
    rubric = (ASSETS / "judge_rubric.md").read_text()
    request = prepare_question_reviews(data, rubric)
    submission = ReviewSubmission(
        calibration_digest=request["calibration_digest"],
        prompt_digest=request["prompt_digest"],
        reviewer="oracle-fixture", reviewer_kind="fixture",
        reviews=[QuestionReview(
            item_id=item["id"], status="scored", scores=item["expected_scores"],
            reject=item["expected_reject"], rationale="Fixed anchor used only to test the grader.",
        ) for item in data["items"]],
    )
    return data, rubric, submission


def test_blind_request_excludes_anchor_answers_and_rationales() -> None:
    data, rubric, _ = _inputs()
    request = prepare_question_reviews(data, rubric)
    serialized = json.dumps(request)
    assert "expected_scores" not in serialized
    assert "handcrafted_negative" not in serialized
    for item in data["items"]:
        assert item["rationale"] not in serialized
    assert len(request["items"]) == len(data["items"])


def test_oracle_only_proves_grader_agreement_not_live_quality() -> None:
    data, rubric, submission = _inputs()
    result = evaluate_question_reviews(data, rubric, submission)
    assert result["anchor_agreement_passed"]
    assert result["within_half_point_rate"] == 1
    assert result["human_review_complete"] == all(
        item["human_scores"] is not None for item in data["items"]
    )
    assert not result["release_eligible"]


def test_missed_veto_fails_even_when_mean_agreement_is_high() -> None:
    data, rubric, submission = _inputs()
    target = next(review for review in submission.reviews if review.reject)
    target.reject = False
    result = evaluate_question_reviews(data, rubric, submission)
    assert result["within_half_point_rate"] == 1
    assert not result["anchor_agreement_passed"]
    assert f"{target.item_id}:missed_hard_veto" in result["failures"]


def test_unknown_context_cannot_be_counted_as_success() -> None:
    data, rubric, submission = _inputs()
    target = submission.reviews[0]
    submission.reviews[0] = QuestionReview(
        item_id=target.item_id, status="insufficient_context", rationale="Missing schema.",
    )
    result = evaluate_question_reviews(data, rubric, submission)
    assert not result["anchor_agreement_passed"]
    assert f"{target.item_id}:insufficient_context" in result["failures"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "data", "prompt"])
def test_incomplete_or_mismatched_submissions_are_rejected(mutation: str) -> None:
    data, rubric, submission = _inputs()
    if mutation == "missing":
        submission.reviews.pop()
    elif mutation == "duplicate":
        submission.reviews.append(copy.deepcopy(submission.reviews[0]))
    elif mutation == "unknown":
        submission.reviews[0].item_id = "invented"
    elif mutation == "data":
        submission.calibration_digest = "wrong"
    else:
        submission.prompt_digest = "wrong"
    with pytest.raises(ValueError):
        evaluate_question_reviews(data, rubric, submission)


@pytest.mark.parametrize("bad_score", [True, 0, 6, "5"])
def test_invalid_score_types_are_rejected(bad_score: object) -> None:
    with pytest.raises(ValidationError):
        QuestionReview.model_validate({
            "item_id": "q01", "status": "scored",
            "scores": {"answerability": bad_score, "business_value": 4,
                       "specificity": 4, "data_support": 5},
            "reject": False, "rationale": "Test invalid value.",
        })
