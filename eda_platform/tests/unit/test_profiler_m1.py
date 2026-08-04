from pathlib import Path

import pandas as pd

import eda_platform.tools.profiler as profiler_module
from eda_platform.schemas.artifacts import DatasetProfile
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import _exact_duplicate_count, profile_dataset


def test_profiler_infers_semantic_types_and_column_details(tmp_path: Path) -> None:
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "customer_id,signup_date,is_active,total_spend,segment,notes\n"
        "C001,2026-01-01,true,10.5,SMB,Short note\n"
        "C002,2026-01-02,false,20.0,Enterprise,"
        "This customer has a long free-form note with extra detail\n"
        "C003,not-a-date,true,30.5,SMB,Another long free-form text value\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_customers")

    artifact = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    profile = DatasetProfile.model_validate(artifact.payload)

    columns = {column.name: column for column in profile.columns_detail}

    assert profile.duplicate_rows == 0
    assert columns["customer_id"].semantic_type == "id"
    assert columns["signup_date"].semantic_type == "datetime"
    assert columns["signup_date"].parse_success_percent == 66.67
    assert columns["is_active"].semantic_type == "boolean"
    assert columns["total_spend"].semantic_type == "numeric"
    assert columns["segment"].semantic_type == "categorical"
    assert columns["notes"].semantic_type == "text"
    assert columns["customer_id"].unique_percent == 100.0
    assert profile.semantic_type_counts["id"] == 1
    assert "C001" in columns["customer_id"].sample_values


def test_profiler_counts_duplicate_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "dupes.csv"
    csv_path.write_text(
        "order_id,amount\n"
        "1,10\n"
        "1,10\n"
        "2,20\n",
        encoding="utf-8",
    )

    artifact = profile_dataset(
        load_csv(csv_path, dataset_id="ds_orders"),
        project_id="project_demo",
        session_id="run_demo",
    )
    profile = DatasetProfile.model_validate(artifact.payload)

    assert profile.duplicate_rows == 1


def test_a_collapsed_duplicate_scope_reports_exact_rows_instead(tmp_path: Path) -> None:
    # Stripping id columns left tournament_stages.csv with one boolean column,
    # so 7 rows were reported as "5 duplicate rows" -- the definition of a
    # boolean, published as a data-quality limitation.
    csv_path = tmp_path / "stages.csv"
    csv_path.write_text(
        "stage_id,stage_name,is_knockout\n"
        "1,Group A,false\n2,Group B,false\n3,Group C,false\n"
        "4,Round of 32,true\n5,Round of 16,true\n6,Quarter,true\n7,Final,true\n",
        encoding="utf-8",
    )

    profile = DatasetProfile.model_validate(
        profile_dataset(
            load_csv(csv_path, dataset_id="ds_stages"),
            project_id="project_demo",
            session_id="run_demo",
        ).payload
    )

    assert profile.exact_duplicate_rows == 0
    assert profile.duplicate_rows == 0
    assert profile.duplicate_scope_columns == list(profile.column_names)


def test_a_wide_payload_scope_still_ignores_surrogate_keys(tmp_path: Path) -> None:
    # The floor must not disable the original behaviour: two rows that differ
    # only by their surrogate key are still payload duplicates.
    csv_path = tmp_path / "wide.csv"
    csv_path.write_text(
        "row_id,region,channel,amount,units\n"
        "1,North,web,10.5,3\n"
        "2,North,web,10.5,3\n"
        "3,South,store,20.0,4\n",
        encoding="utf-8",
    )

    profile = DatasetProfile.model_validate(
        profile_dataset(
            load_csv(csv_path, dataset_id="ds_wide"),
            project_id="project_demo",
            session_id="run_demo",
        ).payload
    )

    assert profile.exact_duplicate_rows == 0
    assert profile.duplicate_rows == 1
    assert "row_id" not in profile.duplicate_scope_columns


def test_exact_duplicate_count_resolves_hash_collisions(monkeypatch) -> None:
    frame = pd.DataFrame({"value": [10, 20, 20, 30], "label": ["a", "b", "b", "c"]})

    # Force every row into the same candidate bucket. The exact second pass
    # must still report only the genuinely duplicated row.
    monkeypatch.setattr(
        profiler_module,
        "hash_pandas_object",
        lambda selected, **_: pd.Series(0, index=selected.index, dtype="uint64"),
    )

    assert _exact_duplicate_count(frame) == 1


def test_profiler_types_shuffled_integer_key_as_id(tmp_path: Path) -> None:
    # A near-unique, near-contiguous integer column is a surrogate key (e.g. a
    # randomized respondent id): averaging it into a report is meaningless.
    # PCA-style floats and small-range integer measures must stay numeric.
    rows = 40
    respondent = list(range(1, rows + 1))
    respondent = respondent[1::2] + respondent[0::2]  # shuffled permutation
    lines = ["respondent,flag,component,training_count"]
    for index in range(rows):
        lines.append(
            f"{respondent[index]},{index % 2},{0.1 + index * 0.037:.6f},{index % 7}"
        )
    csv_path = tmp_path / "survey.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact = profile_dataset(
        load_csv(csv_path, dataset_id="ds_survey"),
        project_id="project_demo",
        session_id="run_demo",
    )
    profile = DatasetProfile.model_validate(artifact.payload)
    columns = {column.name: column for column in profile.columns_detail}

    assert columns["respondent"].semantic_type == "id"
    assert columns["flag"].semantic_type == "boolean"
    assert columns["component"].semantic_type == "numeric"
    assert columns["training_count"].semantic_type == "numeric"


def test_profiler_keeps_sparse_unique_integers_numeric(tmp_path: Path) -> None:
    # Unique but non-contiguous integers (e.g. amounts in cents) are measures,
    # not keys: the range check must reject them.
    lines = ["amount_cents"]
    lines.extend(str(1000 + index * 137 + (index % 5) * 9001) for index in range(30))
    csv_path = tmp_path / "amounts.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact = profile_dataset(
        load_csv(csv_path, dataset_id="ds_amounts"),
        project_id="project_demo",
        session_id="run_demo",
    )
    profile = DatasetProfile.model_validate(artifact.payload)
    columns = {column.name: column for column in profile.columns_detail}

    assert columns["amount_cents"].semantic_type == "numeric"


def test_profiler_duplicate_rows_ignore_id_columns(tmp_path: Path) -> None:
    # A surrogate key differs on every row, so counting it hides true payload
    # duplicates (creditcard_2023 pair 323127/510531 was missed this way).
    csv_path = tmp_path / "keyed.csv"
    csv_path.write_text(
        "record_id,amount,category\n"
        "1,10,a\n"
        "2,10,a\n"
        "3,20,b\n",
        encoding="utf-8",
    )

    artifact = profile_dataset(
        load_csv(csv_path, dataset_id="ds_keyed"),
        project_id="project_demo",
        session_id="run_demo",
    )
    profile = DatasetProfile.model_validate(artifact.payload)

    assert profile.duplicate_rows == 1


def test_profiler_duplicate_rows_fall_back_when_all_columns_are_ids(tmp_path: Path) -> None:
    # Excluding id columns must never leave an empty subset (which would count
    # every row after the first as a duplicate).
    lines = ["user_id"]
    lines.extend(str(index) for index in range(25))
    csv_path = tmp_path / "ids_only.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact = profile_dataset(
        load_csv(csv_path, dataset_id="ds_ids"),
        project_id="project_demo",
        session_id="run_demo",
    )
    profile = DatasetProfile.model_validate(artifact.payload)

    assert profile.duplicate_rows == 0


def test_profiler_handles_empty_frame_missing_percent(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty_after_cleaning.csv"
    csv_path.write_text("round,gameweek,home_team,away_team,score\n", encoding="utf-8")

    artifact = profile_dataset(
        load_csv(csv_path, dataset_id="ds_empty"),
        project_id="project_demo",
        session_id="run_demo",
    )
    profile = DatasetProfile.model_validate(artifact.payload)

    assert profile.rows == 0
    assert profile.missing_percent == {
        "round": 0.0,
        "gameweek": 0.0,
        "home_team": 0.0,
        "away_team": 0.0,
        "score": 0.0,
    }
    assert {column.missing_percent for column in profile.columns_detail} == {0.0}


def test_profiler_recognizes_numeric_strings_entities_and_pk_candidates(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "world_cup_like.csv"
    csv_path.write_text(
        "player,age,attendance,shot_rate,team\n"
        'A Player,27-003,"80,824",55.5%,Algeria\n'
        'B Player,23-047,"44,985",40.0%,Argentina\n'
        'C Player,26-122,"43,002",,Australia\n',
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_world_cup")

    # Entity names are configurable (no hardcoded domain vocabulary in the profiler).
    artifact = profile_dataset(
        loaded,
        project_id="project_demo",
        session_id="run_demo",
        entity_identifier_names=frozenset({"player", "team"}),
    )
    profile = DatasetProfile.model_validate(artifact.payload)

    columns = {column.name: column for column in profile.columns_detail}

    assert columns["player"].semantic_type == "id"
    assert columns["age"].semantic_type == "numeric"
    assert columns["attendance"].semantic_type == "numeric"
    assert columns["shot_rate"].semantic_type == "numeric"
    assert columns["team"].semantic_type == "id"
    assert set(profile.primary_key_candidates) == {"player", "team"}
