from pathlib import Path

from eda_platform.schemas.artifacts import QualityIssueSet
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.quality import scan_quality


def test_quality_rules_flag_dirty_dataset(tmp_path: Path) -> None:
    csv_path = tmp_path / "dirty.csv"
    csv_path.write_text(
        "customer_id,signup_date,status,amount,mostly_missing,constant,mixed\n"
        "C001,2026-01-01,active,10,,same,100\n"
        "C002,not-a-date,pending,20,,same,unknown\n"
        "C003,2026-01-03,closed,30,,same,300\n"
        "C003,2026-01-03,closed,30,,same,300\n",
        encoding="utf-8",
    )
    profile = profile_dataset(
        load_csv(csv_path, dataset_id="ds_dirty"),
        project_id="project_demo",
        session_id="run_demo",
    )

    artifact = scan_quality(profile, project_id="project_demo", session_id="run_demo")
    issue_set = QualityIssueSet.model_validate(artifact.payload)
    codes = {issue.code for issue in issue_set.issues}

    assert "high_missing" in codes
    assert "duplicate_rows" in codes
    assert "constant_column" in codes
    assert "likely_id_column" in codes
    assert "date_parse_failure" in codes
    assert "mixed_type_string" in codes


def test_quality_rules_warn_on_high_cardinality_category(tmp_path: Path) -> None:
    csv_path = tmp_path / "categories.csv"
    csv_path.write_text(
        "tag\n"
        "tag_a\n"
        "tag_b\n"
        "tag_c\n"
        "tag_d\n"
        "tag_e\n",
        encoding="utf-8",
    )
    profile = profile_dataset(
        load_csv(csv_path, dataset_id="ds_categories"),
        project_id="project_demo",
        session_id="run_demo",
    )

    artifact = scan_quality(profile, project_id="project_demo", session_id="run_demo")
    issue_set = QualityIssueSet.model_validate(artifact.payload)

    assert any(issue.code == "high_cardinality_category" for issue in issue_set.issues)


def test_quality_rules_do_not_warn_on_entity_identifier_cardinality(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "players.csv"
    csv_path.write_text(
        "player,age,attendance\n"
        'A Player,27-003,"80,824"\n'
        'B Player,23-047,"44,985"\n'
        'C Player,26-122,"43,002"\n'
        'D Player,24-127,"70,492"\n',
        encoding="utf-8",
    )
    profile = profile_dataset(
        load_csv(csv_path, dataset_id="ds_players"),
        project_id="project_demo",
        session_id="run_demo",
        entity_identifier_names=frozenset({"player", "team"}),
    )

    artifact = scan_quality(profile, project_id="project_demo", session_id="run_demo")
    issue_set = QualityIssueSet.model_validate(artifact.payload)

    warning_pairs = {
        (issue.code, issue.column)
        for issue in issue_set.issues
        if issue.severity == "warn"
    }
    info_pairs = {
        (issue.code, issue.column)
        for issue in issue_set.issues
        if issue.severity == "info"
    }

    assert ("high_cardinality_category", "player") not in warning_pairs
    assert ("high_cardinality_category", "age") not in warning_pairs
    assert ("high_cardinality_category", "attendance") not in warning_pairs
    assert ("likely_id_column", "player") in info_pairs
