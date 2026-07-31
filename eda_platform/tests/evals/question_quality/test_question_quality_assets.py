"""Structure guards for the question-quality review foundation.

The scoring itself (LLM judge calibration + human rating) is deferred until a
live key / reviewer time exists; these tests keep the rubric document and the
calibration set well-formed and internally consistent so the deferred run can
start without repair work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ASSET_DIR = Path(__file__).parent
RUBRIC_PATH = ASSET_DIR / "rubric.md"
CALIBRATION_PATH = ASSET_DIR / "judge_calibration.json"

DIMENSIONS = ("answerability", "business_value", "specificity", "data_support")


def _calibration() -> dict:
    return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))


def test_rubric_document_covers_all_dimensions_and_protocol() -> None:
    text = RUBRIC_PATH.read_text(encoding="utf-8")
    for heading in (
        "可回答性",
        "业务价值",
        "具体性",
        "数据支撑",
        "Judge 校准协议",
        "人工评分流程",
        "待用户执行",
    ):
        assert heading in text, f"rubric.md is missing section/marker: {heading}"
    # 5-point anchors present for every dimension table.
    assert text.count("| 5 |") >= 4
    assert text.count("| 1 |") >= 4


def test_calibration_set_has_at_least_10_wellformed_items() -> None:
    data = _calibration()
    items = data["items"]
    assert len(items) >= 10
    ids = [item["id"] for item in items]
    assert len(set(ids)) == len(ids), "calibration ids must be unique"
    for item in items:
        assert item["question_en"], item["id"]
        assert len(item["rationale"]) >= 15, f"{item['id']}: rationale too thin"
        scores = item["expected_scores"]
        assert set(scores) == set(DIMENSIONS), item["id"]
        for dimension, value in scores.items():
            assert isinstance(value, int) and 1 <= value <= 5, (
                f"{item['id']}.{dimension} must be an int in 1..5, got {value!r}"
            )


def test_expected_overall_is_the_dimension_mean() -> None:
    for item in _calibration()["items"]:
        mean = sum(item["expected_scores"].values()) / len(DIMENSIONS)
        assert item["expected_overall"] == pytest.approx(mean, abs=1e-9), item["id"]


def test_reject_flag_matches_rubric_hard_veto_rule() -> None:
    """rubric.md §2: reject iff answerability <= 2 or data_support <= 2."""
    for item in _calibration()["items"]:
        scores = item["expected_scores"]
        should_reject = scores["answerability"] <= 2 or scores["data_support"] <= 2
        assert item["expected_reject"] is should_reject, item["id"]


def test_calibration_set_spans_high_and_low_anchors() -> None:
    overalls = [item["expected_overall"] for item in _calibration()["items"]]
    assert sum(1 for value in overalls if value >= 4.0) >= 3, "need >=3 high anchors"
    assert sum(1 for value in overalls if value <= 2.5) >= 3, "need >=3 low anchors"


def test_calibration_includes_real_template_route_outputs() -> None:
    origins = [item["origin"] for item in _calibration()["items"]]
    assert origins.count("template_route") >= 4


def test_human_review_is_explicitly_deferred() -> None:
    data = _calibration()
    assert "pending_human_review" in data["_meta"]["human_review_status"]
    for item in data["items"]:
        assert item["human_scores"] is None, (
            f"{item['id']}: human_scores pre-filled; human pass is supposed to be deferred"
        )
