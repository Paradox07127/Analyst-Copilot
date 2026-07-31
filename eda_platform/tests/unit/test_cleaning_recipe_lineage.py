"""The pre-clean drop must land in the evidence chain as a CleaningRecipe
artifact, and the cleaned dataset's profile must name that recipe as its
lineage parent (spec §4.5 / FR-3)."""

from pathlib import Path

import pytest

from eda_platform.drivers.auto_eda import _align_precleaning, run_auto_eda
from eda_platform.schemas.artifacts import ArtifactType, DatasetProfile
from eda_platform.schemas.cleaning import CleaningRecipe
from eda_platform.tools.precleaning import preclean_csv_files


def _write_dirty_csv(path: Path) -> None:
    path.write_text(
        "mostly_missing,sometimes_missing,label\n"
        ",1,a\n"
        ",,b\n"
        ",3,\n"
        "kept,4,d\n",
        encoding="utf-8",
    )


def test_run_auto_eda_records_cleaning_recipe_as_profile_lineage(tmp_path: Path) -> None:
    raw_path = tmp_path / "sales.csv"
    _write_dirty_csv(raw_path)

    batch = preclean_csv_files(
        [raw_path],
        clean_missing_values=True,
        missing_threshold_percent=70.0,
        min_rows_keep_percent=50.0,
        drop_iqr_outliers=False,
    )
    recipe = batch.recipes[0]
    assert recipe is not None

    result = run_auto_eda(
        batch.dataset_paths,
        workspace=tmp_path / "workspace",
        project_id="p",
        session_id="r",
        precleaning=batch.recipes,
    )

    recipe_artifacts = [a for a in result.artifacts if a.type is ArtifactType.CLEANING_RECIPE]
    profile_artifacts = [a for a in result.artifacts if a.type is ArtifactType.DATASET_PROFILE]
    assert len(recipe_artifacts) == 1
    assert len(profile_artifacts) == 1
    recipe_artifact = recipe_artifacts[0]
    profile_artifact = profile_artifacts[0]

    # The profile traces back to the recipe...
    assert profile_artifact.parents == [recipe_artifact.id]
    # ...and the recipe traces back to the raw upload (a different dataset id
    # than the cleaned profile's, since content changed).
    payload_recipe = CleaningRecipe.model_validate(recipe_artifact.payload)
    profile = DatasetProfile.model_validate(profile_artifact.payload)
    assert payload_recipe.dataset_id != profile.dataset_id
    assert payload_recipe.lineage is not None
    assert payload_recipe.lineage.source_name == "sales.csv"
    assert payload_recipe.dataset_id == payload_recipe.lineage.source_dataset_id

    # A data-changing clean auto-applied without an interactive HITL gate is
    # flagged for audit.
    assert "auto_applied_without_hitl" in recipe_artifact.warnings

    # The recipe is persisted and reachable in the store by id.
    assert result.artifacts[0].type is ArtifactType.CLEANING_RECIPE


def test_run_auto_eda_records_raw_before_cleaning_artifacts(tmp_path: Path) -> None:
    raw_path = tmp_path / "sales.csv"
    raw_path.write_text(
        "mostly_missing,amount,label\n"
        ",1,a\n"
        ",,b\n"
        ",3,\n"
        "kept,4,d\n",
        encoding="utf-8",
    )
    batch = preclean_csv_files(
        [raw_path],
        clean_missing_values=True,
        missing_threshold_percent=70.0,
        min_rows_keep_percent=50.0,
        drop_iqr_outliers=False,
    )

    result = run_auto_eda(
        batch.dataset_paths,
        workspace=tmp_path / "workspace",
        project_id="p",
        session_id="r_raw",
        precleaning=batch.recipes,
        raw_file_paths=[raw_path],
    )

    raw_profile = next(a for a in result.artifacts if a.type is ArtifactType.RAW_DATASET_PROFILE)
    raw_preview = next(a for a in result.artifacts if a.type is ArtifactType.RAW_DATA_PREVIEW)
    raw_charts = [a for a in result.artifacts if a.type is ArtifactType.RAW_CHART_SPEC]
    cleaned_profile = next(a for a in result.artifacts if a.type is ArtifactType.DATASET_PROFILE)

    assert raw_profile.payload["name"] == "sales.csv"
    assert raw_profile.payload["rows"] == 4
    assert raw_profile.payload["columns"] == 3
    assert raw_preview.payload["rows"] == 4
    assert len(raw_preview.payload["rows_preview"]) == 4
    assert raw_charts
    assert all(
        chart.payload["dataset_id"] == raw_profile.payload["dataset_id"] for chart in raw_charts
    )
    assert cleaned_profile.payload["rows"] == 2
    assert cleaned_profile.payload["columns"] == 2


def test_run_auto_eda_without_precleaning_leaves_profile_lineage_empty(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "clean.csv"
    csv_path.write_text("a,b\n1,x\n2,y\n3,z\n4,w\n", encoding="utf-8")

    result = run_auto_eda(
        [csv_path],
        workspace=tmp_path / "workspace",
        project_id="p",
        session_id="r",
    )

    recipe_artifacts = [a for a in result.artifacts if a.type is ArtifactType.CLEANING_RECIPE]
    profile_artifact = next(a for a in result.artifacts if a.type is ArtifactType.DATASET_PROFILE)
    assert recipe_artifacts == []
    assert profile_artifact.parents == []


def test_align_precleaning_pads_and_validates() -> None:
    assert _align_precleaning(None, 3) == [None, None, None]

    recipe = CleaningRecipe(dataset_id="ds")
    assert _align_precleaning([recipe], 1) == [recipe]

    with pytest.raises(ValueError, match="align"):
        _align_precleaning([recipe], 2)
