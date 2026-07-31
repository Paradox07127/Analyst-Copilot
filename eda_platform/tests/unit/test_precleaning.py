import pandas as pd

from eda_platform.schemas.cleaning import transform_is_lossy
from eda_platform.tools.cleaning import _apply_recipe_to_frame
from eda_platform.tools.loader import load_csv
from eda_platform.tools.precleaning import preclean_csv_files, preclean_frame


def test_missing_cleaning_drops_high_missing_columns_and_lower_missing_rows() -> None:
    frame = pd.DataFrame(
        {
            "mostly_missing": [None, None, None, "kept"],
            "sometimes_missing": [1, None, 3, 4],
            "label": ["a", "b", None, "d"],
        }
    )

    result = preclean_frame(
        frame,
        clean_missing_values=True,
        missing_threshold_percent=70.0,
        min_rows_keep_percent=50.0,
        drop_iqr_outliers=False,
    )

    assert result.dropped_missing_columns == ["mostly_missing"]
    assert result.dropped_missing_rows == 2
    assert list(result.frame.columns) == ["sometimes_missing", "label"]
    assert len(result.frame) == 2
    assert not result.skipped_missing_column_drop
    assert not result.skipped_missing_row_drop


def test_missing_cleaning_skips_row_drop_when_minimum_rows_guard_would_fail() -> None:
    frame = pd.DataFrame({"metric": [1, None, None, 4], "label": ["a", "b", "c", "d"]})

    result = preclean_frame(
        frame,
        clean_missing_values=True,
        missing_threshold_percent=70.0,
        min_rows_keep_percent=75.0,
        drop_iqr_outliers=False,
    )

    assert result.dropped_missing_rows == 0
    assert len(result.frame) == 4
    assert result.skipped_missing_row_drop


def test_missing_cleaning_skips_column_drop_when_every_column_would_be_removed() -> None:
    frame = pd.DataFrame({"a": [None, None], "b": [None, None]})

    result = preclean_frame(
        frame,
        clean_missing_values=True,
        missing_threshold_percent=70.0,
        min_rows_keep_percent=50.0,
        drop_iqr_outliers=False,
    )

    assert result.dropped_missing_columns == []
    assert list(result.frame.columns) == ["a", "b"]
    assert result.skipped_missing_column_drop


def test_preclean_csv_files_records_guardrail_only_recipe(tmp_path) -> None:
    csv_path = tmp_path / "emptyish.csv"
    csv_path.write_text("a,b\n,\n,\n", encoding="utf-8")

    batch = preclean_csv_files(
        [csv_path],
        clean_missing_values=True,
        missing_threshold_percent=70.0,
        min_rows_keep_percent=50.0,
        drop_iqr_outliers=False,
    )

    assert batch.dataset_paths == [csv_path]
    assert batch.created_paths == []
    recipe = batch.recipes[0]
    assert recipe is not None
    assert recipe.transforms == []
    assert recipe.requires_approval is False
    assert {guardrail.code for guardrail in recipe.guardrails} == {
        "missing_column_drop_would_remove_all_columns",
        "missing_row_drop_below_min_rows",
    }
    assert recipe.lineage is not None
    assert recipe.lineage.rows_before == recipe.lineage.rows_after == 2
    assert recipe.lineage.columns_before == recipe.lineage.columns_after == 2


def test_missing_cleaning_drops_rows_of_surviving_high_missing_column() -> None:
    # Single column above threshold: the column can't be dropped (that would
    # remove every column), so its missing rows must still be dropped instead
    # of silently surviving into the pipeline.
    frame = pd.DataFrame({"x": [None, None, 1, 2, 3, 4, 5, 6, 7, 8]})

    result = preclean_frame(
        frame,
        clean_missing_values=True,
        missing_threshold_percent=10.0,
        min_rows_keep_percent=50.0,
        drop_iqr_outliers=False,
    )

    assert result.skipped_missing_column_drop
    assert result.dropped_missing_columns == []
    assert result.dropped_missing_rows == 2
    assert result.frame["x"].isna().sum() == 0
    assert len(result.frame) == 8


def test_outlier_cleaning_is_independent_of_column_order() -> None:
    # Each column's IQR fence is computed on the original frame, so reordering
    # columns must not change which rows are dropped.
    data = {
        "a": [1, 2, 3, 4, 5, 500],
        "b": [10, 11, 12, 13, 900, 15],
    }
    frame = pd.DataFrame(data)
    reordered = pd.DataFrame({"b": data["b"], "a": data["a"]})

    forward = preclean_frame(
        frame,
        clean_missing_values=False,
        missing_threshold_percent=70.0,
        min_rows_keep_percent=10.0,
        drop_iqr_outliers=True,
    )
    backward = preclean_frame(
        reordered,
        clean_missing_values=False,
        missing_threshold_percent=70.0,
        min_rows_keep_percent=10.0,
        drop_iqr_outliers=True,
    )

    assert forward.dropped_outlier_rows == backward.dropped_outlier_rows
    assert sorted(forward.frame["a"].tolist()) == sorted(backward.frame["a"].tolist())


def test_outlier_cleaning_uses_same_minimum_rows_guard() -> None:
    frame = pd.DataFrame({"amount": [1, 1, 1, 1000], "label": ["a", "b", "c", "d"]})

    result = preclean_frame(
        frame,
        clean_missing_values=False,
        missing_threshold_percent=70.0,
        min_rows_keep_percent=50.0,
        drop_iqr_outliers=True,
    )

    assert result.dropped_outlier_rows == 1
    assert result.frame["amount"].tolist() == [1, 1, 1]


def test_preclean_csv_files_writes_new_version_without_overwriting_original(tmp_path) -> None:
    csv_path = tmp_path / "sales.csv"
    original_text = (
        "mostly_missing,sometimes_missing,label\n"
        ",1,a\n"
        ",,b\n"
        ",3,\n"
        "kept,4,d\n"
    )
    csv_path.write_text(original_text, encoding="utf-8")

    batch = preclean_csv_files(
        [csv_path],
        clean_missing_values=True,
        missing_threshold_percent=70.0,
        min_rows_keep_percent=50.0,
        drop_iqr_outliers=False,
    )

    # The original upload must never be overwritten in place (FR-3).
    assert csv_path.read_text(encoding="utf-8") == original_text

    report = batch.reports[0]
    assert report.dataset == "sales.csv"
    assert report.dropped_missing_columns == ["mostly_missing"]
    assert report.dropped_missing_rows == 2

    # A cleaned new version is produced and pointed at for ingestion.
    cleaned_path = batch.dataset_paths[0]
    assert cleaned_path != csv_path
    assert cleaned_path in batch.created_paths
    cleaned = pd.read_csv(cleaned_path)
    assert list(cleaned.columns) == ["sometimes_missing", "label"]
    assert len(cleaned) == 2


def test_preclean_csv_files_keeps_original_path_when_unchanged(tmp_path) -> None:
    csv_path = tmp_path / "clean.csv"
    csv_path.write_text("a,b\n1,x\n2,y\n3,z\n4,w\n", encoding="utf-8")

    batch = preclean_csv_files(
        [csv_path],
        clean_missing_values=True,
        missing_threshold_percent=70.0,
        min_rows_keep_percent=50.0,
        drop_iqr_outliers=False,
    )

    # Nothing to clean -> ingest the original, write no new file (no needless
    # CSV round-trip / dtype coercion).
    assert batch.dataset_paths == [csv_path]
    assert batch.created_paths == []
    assert not batch.reports[0].changed
    # An unchanged input carries no recipe (nothing to record in the chain).
    assert batch.recipes == [None]


def test_preclean_emits_cleaning_recipe_recording_drops_and_lineage(tmp_path) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "mostly_missing,sometimes_missing,label\n"
        ",1,a\n"
        ",,b\n"
        ",3,\n"
        "kept,4,d\n",
        encoding="utf-8",
    )

    batch = preclean_csv_files(
        [csv_path],
        clean_missing_values=True,
        missing_threshold_percent=70.0,
        min_rows_keep_percent=50.0,
        drop_iqr_outliers=False,
    )

    recipe = batch.recipes[0]
    assert recipe is not None
    assert recipe.created_by == "precleaning"
    # Every drop is lossy -> the recorded recipe would require HITL approval.
    assert recipe.requires_approval is True
    assert all(transform_is_lossy(t) for t in recipe.transforms)

    by_type = {t.type: t for t in recipe.transforms}
    assert by_type["drop_column"].target_column == "mostly_missing"
    assert by_type["drop_column"].params["missing_threshold_percent"] == 70.0
    assert by_type["drop_missing_rows"].expected_impact_rows == 2

    lineage = recipe.lineage
    assert lineage is not None
    assert lineage.source_name == "sales.csv"
    assert lineage.source_dataset_id.startswith("ds_")
    # The recipe's dataset id is the raw source; the raw upload is the lineage
    # parent that every downstream conclusion can be traced back to.
    assert recipe.dataset_id == lineage.source_dataset_id
    assert lineage.rows_before == 4
    assert lineage.rows_after == 2
    assert lineage.columns_before == 3
    assert lineage.columns_after == 2


def test_preclean_recipe_replays_onto_raw_frame(tmp_path) -> None:
    # The recorded recipe must not lie: replaying its transforms on the raw
    # frame reproduces exactly the cleaned frame the pre-cleaner wrote.
    csv_path = tmp_path / "mixed.csv"
    csv_path.write_text(
        "mostly_missing,value,label\n"
        ",1,a\n"
        ",2,b\n"
        ",3,c\n"
        ",4,d\n"
        ",5,e\n"
        ",6,f\n"
        ",7,g\n"  # mostly_missing empty in 7/10 rows -> 70% > 60% threshold
        "x,8,h\n"
        "y,,i\n"  # missing value -> drop_missing_rows
        "z,9000,j\n",  # extreme value -> drop_outlier_rows
        encoding="utf-8",
    )
    raw = load_csv(csv_path).frame

    batch = preclean_csv_files(
        [csv_path],
        clean_missing_values=True,
        missing_threshold_percent=60.0,
        min_rows_keep_percent=10.0,
        drop_iqr_outliers=True,
    )
    recipe = batch.recipes[0]
    assert recipe is not None
    # Exercises all three drop kinds together.
    assert {t.type for t in recipe.transforms} == {
        "drop_column",
        "drop_missing_rows",
        "drop_outlier_rows",
    }

    cleaned = pd.read_csv(batch.dataset_paths[0])
    replayed, _, _, _ = _apply_recipe_to_frame(raw, recipe)

    pd.testing.assert_frame_equal(
        replayed.reset_index(drop=True),
        cleaned.reset_index(drop=True),
        check_dtype=False,
    )
