"""A metric whose answer is a date must be allowed to say the date.

2026-08-04 FIFA runs 1 and 2: `time_coverage` asked for "first/last timestamp,
covered months" and its SQL returned all five fields, but the binder required
every template slot to parse as a number. The timestamps did not, so the whole
domain finding was abandoned and a generic fallback published "The returned
span_days is 38 for 2026-06-11T00:00:00" -- three of five answers dropped and
the remaining sentence ungrammatical.

Both runs also logged `numeric_mismatch: 2026 [raw pool: 1], 0 [raw pool: 1]`:
the validator reads an ISO timestamp as the magnitudes 2026, 0 and 0.
"""

from __future__ import annotations

from eda_platform.core.ids import make_artifact_id
from eda_platform.drivers.question_exec import _findings_for
from eda_platform.schemas.artifacts import Artifact, ArtifactType, SqlResult
from eda_platform.schemas.questions import QuestionCandidate, QuestionScore
from eda_platform.tools.report_validator import _numeric_tokens_from_text

_ROW: dict[str, object] = {
    "row_count": 104,
    "first_timestamp": "2026-06-11T00:00:00",
    "last_timestamp": "2026-07-19T00:00:00",
    "span_days": 38,
    "covered_months": 2,
}

_UNITS = {
    "row_count": "count",
    "first_timestamp": "timestamp",
    "last_timestamp": "timestamp",
    "span_days": "days",
    "covered_months": "months",
}


def _artifact() -> Artifact:
    payload = SqlResult(
        sql="select 1",
        columns=list(_ROW),
        dtypes={"span_days": "int64", "first_timestamp": "datetime64[us]"},
        units=_UNITS,
        rows_preview=[_ROW],
        row_count=1,
    ).model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("sql", payload),
        type=ArtifactType.SQL_RESULT,
        project_id="project_demo",
        session_id="run_demo",
        payload=payload,
    )


def _candidate() -> QuestionCandidate:
    return QuestionCandidate(
        question_id="q_time",
        question_en=(
            "What time span does date cover in Match Prediction Features "
            "(first/last timestamp, covered months)?"
        ),
        origin="template",
        template_id="domain_metric",
        metric_id="time_coverage",
        target_datasets=["match_prediction_features.csv"],
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.84,
        ),
    )


def test_the_answer_carries_every_field_the_question_asked_for() -> None:
    findings = _findings_for(_candidate(), _artifact())
    assert findings
    text = findings[0].text
    for expected in ("2026-06-11", "2026-07-19", "38", "2"):
        assert expected in text, text
    assert "The returned span_days is 38 for" not in text, text


def test_an_iso_timestamp_is_not_read_as_the_magnitude_2026() -> None:
    """The live `numeric_mismatch: 2026 [raw pool: 1]` finding, in isolation."""
    tokens = _numeric_tokens_from_text("Data spans 2026-06-11 to 2026-07-19 (38 days).")
    assert [token.value for token in tokens] == [38.0]


def test_a_bare_year_is_still_a_number() -> None:
    """The exemption covers date literals, not any four-digit run."""
    tokens = _numeric_tokens_from_text("The 2026 tournament had 104 matches.")
    assert [token.value for token in tokens] == [2026.0, 104.0]


def test_missing_hotspots_states_its_own_numbers() -> None:
    """The other half of the same defect (2026-08-05 offline FIFA run).

    `missing_hotspots` computed 103 of 208 rows missing (49.52%) and published
    "The highest-missing columns and their null shares." -- a registry sentence
    with no placeholders, so nothing bound, evidence came out empty and the
    validator dropped the claim. The question ran, answered, and reached the
    reader as nothing at all.
    """
    from eda_platform.tools.domain_metrics import missing_hotspot_interpretation

    row = {
        "row_count": 208,
        "missing_player_of_the_match": 103.0,
        "missing_player_of_the_match_percent": 49.51923076923077,
    }
    template = missing_hotspot_interpretation(["player_of_the_match"])
    assert "{missing_player_of_the_match_percent}" in template
    assert "{missing_player_of_the_match}" in template
    assert "{row_count}" in template
    filled = template
    for key, value in row.items():
        filled = filled.replace(f"{{{key}}}", str(value))
    assert "{" not in filled, filled
