from pathlib import Path

from eda_platform.schemas.artifacts import DatasetProfile, QualityIssueSet
from eda_platform.schemas.quality_context import QualityContext, QualityContextSet
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.quality import scan_quality
from eda_platform.tools.quality_context import build_quality_context


def _contexts(csv_path: Path) -> tuple[DatasetProfile, QualityIssueSet, QualityContextSet]:
    loaded = load_csv(csv_path, dataset_id="ds_orders")
    profile = profile_dataset(loaded, project_id="project", session_id="run")
    quality = scan_quality(profile, project_id="project", session_id="run")
    context = build_quality_context(
        loaded, profile, quality, project_id="project", session_id="run"
    )
    return (
        DatasetProfile.model_validate(profile.payload),
        QualityIssueSet.model_validate(quality.payload),
        QualityContextSet.model_validate(context.payload),
    )


def _facts(contexts: QualityContextSet, code: str) -> list[str]:
    return next(item.pattern_facts for item in contexts.contexts if item.issue_code == code)


def test_quality_context_contains_observations_without_business_cause_fields(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "orders.csv"
    csv_path.write_text(
        "region,amount\nNorth,10\nNorth,\nSouth,30\nSouth,\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_orders")
    profile = profile_dataset(loaded, project_id="project", session_id="run")
    quality = scan_quality(profile, project_id="project", session_id="run")
    issue_set = QualityIssueSet.model_validate(quality.payload)
    assert issue_set.issues

    artifact = build_quality_context(
        loaded,
        profile,
        quality,
        project_id="project",
        session_id="run",
    )
    context_set = QualityContextSet.model_validate(artifact.payload)

    assert context_set.contexts
    assert artifact.parents == [profile.id, quality.id]
    assert set(context_set.contexts[0].model_dump()) == set(QualityContext.model_fields)
    forbidden = {"business_cause", "root_cause", "cause", "caused_by"}
    assert forbidden.isdisjoint(context_set.contexts[0].model_dump())
    assert all(
        "business cause remains unconfirmed" in context.report_limitation
        or context.requires_data
        for context in context_set.contexts
    )


def test_quality_context_rejects_mismatched_profile_and_issue_set(tmp_path: Path) -> None:
    csv_path = tmp_path / "orders.csv"
    csv_path.write_text("amount\n1\n2\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds_orders")
    profile = profile_dataset(loaded, project_id="project", session_id="run")
    quality = scan_quality(profile, project_id="project", session_id="run")
    quality.payload["dataset_id"] = "ds_other"

    try:
        build_quality_context(
            loaded,
            profile,
            quality,
            project_id="project",
            session_id="run",
        )
    except ValueError as exc:
        assert "same dataset" in str(exc)
    else:
        raise AssertionError("mismatched artifacts must be rejected")


def test_duplicate_facts_use_the_profiler_scope_not_whole_row_equality(
    tmp_path: Path,
) -> None:
    """A surrogate key makes every row unique, so `frame.duplicated()` reported 0
    while the quality issue reported a payload duplicate."""
    csv_path = tmp_path / "orders.csv"
    rows = ["order_id,region,amount"]
    rows += [f"O{index:02d},North,{100 + index}" for index in range(6)]
    rows.append("O99,North,100")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    profile, issues, contexts = _contexts(csv_path)

    assert profile.duplicate_rows == 1
    assert profile.exact_duplicate_rows == 0
    facts = _facts(contexts, "duplicate_rows")
    assert any(str(profile.duplicate_rows) in fact for fact in facts)
    assert not any("0 rows are exact duplicates" in fact for fact in facts)
    assert issues.issues


def test_outlier_facts_parse_numeric_strings_like_the_profiler(tmp_path: Path) -> None:
    """`pd.to_numeric` turns a thousands-separated column into all-NaN, so the
    context claimed 0 outliers under an issue that reported one."""
    csv_path = tmp_path / "orders.csv"
    amounts = [f'"1,0{index:02d}"' for index in range(10)] + ['"80,824"']
    rows = ["order_id,amount"] + [
        f"O{index:02d},{amount}" for index, amount in enumerate(amounts)
    ]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    profile, _, contexts = _contexts(csv_path)

    amount = next(column for column in profile.columns_detail if column.name == "amount")
    assert amount.outlier_count == 1
    facts = _facts(contexts, "outlier_detected")
    assert facts[0].startswith(f"{amount.outlier_count} of ")


def test_date_parse_facts_use_the_profiler_datetime_parser(tmp_path: Path) -> None:
    """Bare `to_datetime` reads yyyymmdd integers as an epoch offset and finds no
    failures where the profiler flagged the column."""
    csv_path = tmp_path / "events.csv"
    valid = [f"2024010{index}" for index in range(1, 9)]
    invalid = ["20241350", "20240230", "20249999"]
    rows = ["event_id,event_date"] + [
        f"E{index:02d},{value}" for index, value in enumerate([*valid, *invalid])
    ]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    profile, _, contexts = _contexts(csv_path)

    event_date = next(
        column for column in profile.columns_detail if column.name == "event_date"
    )
    assert "date_parse_failure" in event_date.warnings
    facts = _facts(contexts, "date_parse_failure")
    assert facts[0].startswith(f"{event_date.parse_failure_count} ")
    assert not facts[0].startswith("0 ")
