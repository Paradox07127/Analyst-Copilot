"""Category levels offered for confirmation, and quality-driven cleaning advice."""

from pathlib import Path

from eda_platform.core.semantic import (
    CategoryLevelMeaning,
    SemanticSeeds,
    pinned_context_block,
)
from eda_platform.schemas.artifacts import DatasetProfile, QualityIssueSet
from eda_platform.tools.cleaning_advice import recommended_cleaning_operations
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.quality import scan_quality


def test_categorical_levels_are_extracted_for_confirmation(tmp_path: Path) -> None:
    # "status = C" is unreadable until a human says whether C is Cancelled or
    # Completed; the platform's job is to hand over the exact level set.
    path = tmp_path / "orders.csv"
    rows = ["status,amount"]
    for index in range(90):
        rows.append(f"{['C', 'P', 'X'][index % 3]},{float(index)}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_csv(path)
    profile = DatasetProfile.model_validate(
        profile_dataset(loaded, project_id="p", session_id="s").payload
    )

    status = next(column for column in profile.columns_detail if column.name == "status")
    levels = {level["value"]: level["count"] for level in status.category_levels}

    assert levels == {"C": 30, "P": 30, "X": 30}
    amount = next(column for column in profile.columns_detail if column.name == "amount")
    assert amount.category_levels == []


def test_high_cardinality_levels_are_bounded(tmp_path: Path) -> None:
    path = tmp_path / "wide.csv"
    rows = ["code"]
    for index in range(400):
        rows.append(f"code_{index}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_csv(path)
    profile = DatasetProfile.model_validate(
        profile_dataset(loaded, project_id="p", session_id="s").payload
    )

    code = next(column for column in profile.columns_detail if column.name == "code")
    assert len(code.category_levels) <= 30


def test_confirmed_level_meanings_reach_the_pinned_context() -> None:
    seeds = SemanticSeeds(
        category_level_meanings=[
            CategoryLevelMeaning(
                dataset="orders.csv",
                column="status",
                value="C",
                meaning="Cancelled by the customer",
            )
        ]
    )

    block = pinned_context_block(seeds)

    assert "orders.csv.status = C" in block
    assert "Cancelled by the customer" in block


def test_cleaning_advice_targets_the_issues_actually_found(tmp_path: Path) -> None:
    # A column whose numbers failed to parse needs parse_numeric, which the
    # fixed four-option form never offered.
    path = tmp_path / "messy.csv"
    rows = ["age,label"]
    for index in range(30):
        rows.append(f"{20 + index}-{index:03d}, spaced ")
    rows.append("27-003, spaced ")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_csv(path)
    profile_artifact = profile_dataset(loaded, project_id="p", session_id="s")
    issues = QualityIssueSet.model_validate(
        scan_quality(profile_artifact, project_id="p", session_id="s").payload
    )
    profile = DatasetProfile.model_validate(profile_artifact.payload)

    advice = recommended_cleaning_operations(profile, issues)
    by_operation = {item["operation"]: item for item in advice}

    assert "trim_whitespace" in by_operation
    assert by_operation["trim_whitespace"]["column"] == "label"
    assert by_operation["trim_whitespace"]["lossy"] is False
    assert all(item["reason"] for item in advice)


def test_cleaning_advice_is_empty_for_a_clean_dataset(tmp_path: Path) -> None:
    path = tmp_path / "clean.csv"
    rows = ["id,amount"]
    for index in range(40):
        rows.append(f"{index},{float(index) * 1.5}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_csv(path)
    profile_artifact = profile_dataset(loaded, project_id="p", session_id="s")
    issues = QualityIssueSet.model_validate(
        scan_quality(profile_artifact, project_id="p", session_id="s").payload
    )
    profile = DatasetProfile.model_validate(profile_artifact.payload)

    assert recommended_cleaning_operations(profile, issues) == []
