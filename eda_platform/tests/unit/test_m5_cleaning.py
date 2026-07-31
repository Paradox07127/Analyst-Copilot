from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from eda_platform.schemas.cleaning import (
    LOSSY_TYPES,
    CleaningRecipe,
    CleaningTransform,
    CleaningTransformType,
    transform_is_lossy,
)
from eda_platform.tools.cleaning import (
    AppliedCleaning,
    _apply_recipe_to_frame,
    apply_cleaning_recipe,
    preview_cleaning_recipe,
)
from eda_platform.tools.loader import load_csv
from eda_platform.tools.value_parsing import parse_numeric_like


def _col(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def test_preview_and_apply_cleaning_recipe_versions_dataset(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "people.csv"
    csv_path.write_text(
        "name,age,segment\n"
        " Alice ,27-003,A\n"
        "Bob,31-000,B\n"
        " Alice ,27-003,A\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_people")

    recipe = CleaningRecipe(
        dataset_id="ds_people",
        source_version=1,
        transforms=[
            CleaningTransform(type="trim_whitespace", target_column="name"),
            CleaningTransform(
                type="parse_numeric",
                target_column="age",
                params={"parser": "numeric_like"},
            ),
            CleaningTransform(type="drop_duplicate_rows"),
        ],
    )
    assert recipe.requires_approval is True

    preview = preview_cleaning_recipe(loaded, recipe)
    age_diff = next(diff for diff in preview.column_diffs if diff.column == "age")

    assert preview.row_count_before == 3
    assert preview.row_count_after == 2
    assert preview.target_version == 2
    assert age_diff.changed_rows == 3
    assert age_diff.after_dtype.startswith("float")

    applied = apply_cleaning_recipe(
        loaded,
        recipe,
        output_dir=tmp_path / "cleaned",
        approved_lossy_transform_ids={
            transform.transform_id
            for transform in recipe.transforms
            if transform_is_lossy(transform)
        },
    )

    assert applied.loaded.record.dataset_id == "ds_people"
    assert applied.loaded.record.version == 2
    assert applied.loaded.record.parent_version == 1
    assert applied.loaded.record.lineage_recipe_id == recipe.recipe_id
    assert applied.loaded.frame.shape == (2, 3)
    assert applied.loaded.frame["name"].tolist() == ["Alice", "Bob"]
    assert applied.loaded.frame["age"].tolist() == [pytest.approx(27.01), 31.0]
    assert applied.result.output_path.exists()


def test_lossy_cleaning_transform_requires_explicit_approval(tmp_path: Path) -> None:
    loaded = load_csv(_write_csv(tmp_path / "missing.csv"), dataset_id="ds_missing")
    recipe = CleaningRecipe(
        dataset_id="ds_missing",
        source_version=1,
        transforms=[
            CleaningTransform(
                type="drop_rows",
                target_column="amount",
                safety="lossy",
                params={"where": "missing"},
                reversible=False,
                description="Drop rows with missing amount.",
            )
        ],
    )

    with pytest.raises(ValueError, match="requires approval"):
        apply_cleaning_recipe(loaded, recipe, output_dir=tmp_path / "cleaned")

    applied = apply_cleaning_recipe(
        loaded,
        recipe,
        output_dir=tmp_path / "cleaned",
        approved_lossy_transform_ids={recipe.transforms[0].transform_id},
    )

    assert applied.loaded.frame["amount"].isna().sum() == 0
    assert applied.preview.row_count_before == 3
    assert applied.preview.row_count_after == 2


def _write_csv(path: Path) -> Path:
    pd.DataFrame(
        {
            "customer": ["A", "B", "C"],
            "amount": [10.0, None, 30.0],
        }
    ).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# CL-1: tier gate derives lossiness from operation type, ignoring payload label
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "transform",
    [
        CleaningTransform(
            type="drop_rows",
            target_column="amount",
            safety="safe",  # mislabeled
            params={"where": "missing"},
        ),
        CleaningTransform(
            type="drop_column",
            target_column="amount",
            # safety omitted -> defaults to "safe"
        ),
        CleaningTransform(
            type="fill_missing",
            target_column="amount",
            safety="safe",  # mislabeled
            params={"value": 0},
        ),
    ],
)
def test_lossy_operation_type_is_gated_even_when_labeled_safe(
    tmp_path: Path,
    transform: CleaningTransform,
) -> None:
    loaded = load_csv(_write_csv(tmp_path / "d.csv"), dataset_id="ds_d")
    original_amount = loaded.frame["amount"].copy()
    recipe = CleaningRecipe(
        dataset_id="ds_d", source_version=1, transforms=[transform]
    )

    # Tier is derived server-side: label "safe" cannot bypass approval.
    assert recipe.requires_approval is True
    with pytest.raises(ValueError, match="requires approval"):
        apply_cleaning_recipe(loaded, recipe, output_dir=tmp_path / "cleaned")

    # Nothing written, caller frame untouched.
    assert not (tmp_path / "cleaned").exists()
    pd.testing.assert_series_equal(loaded.frame["amount"], original_amount)


def test_transform_is_lossy_covers_all_lossy_types() -> None:
    lossy = [
        CleaningTransform(type="drop_rows", target_column="c", safety="safe"),
        CleaningTransform(type="drop_missing_rows", safety="safe"),
        CleaningTransform(type="drop_outlier_rows", safety="safe"),
        CleaningTransform(type="fill_missing", target_column="c", safety="safe"),
        CleaningTransform(type="drop_column", target_column="c", safety="safe"),
        CleaningTransform(type="clip_outliers", target_column="c", safety="safe"),
        CleaningTransform(type="parse_numeric", target_column="c", safety="safe"),
        CleaningTransform(type="drop_duplicate_rows", safety="safe"),
    ]
    assert {t.type for t in lossy} == set(LOSSY_TYPES)
    assert all(transform_is_lossy(t) for t in lossy)
    assert (
        transform_is_lossy(
            CleaningTransform(type="trim_whitespace", target_column="c")
        )
        is False
    )


# --------------------------------------------------------------------------- #
# CL-2: version collision does not silently overwrite a prior output
# --------------------------------------------------------------------------- #


def test_two_cleanings_of_same_source_do_not_overwrite(tmp_path: Path) -> None:
    csv_path = tmp_path / "d.csv"
    csv_path.write_text("name,age,city\n A ,10,NYC\n B ,20,LA\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds")
    store = tmp_path / "store"

    recipe_a = CleaningRecipe(
        dataset_id="ds",
        source_version=1,
        transforms=[CleaningTransform(type="trim_whitespace", target_column="name")],
    )
    applied_a = apply_cleaning_recipe(loaded, recipe_a, output_dir=store)

    recipe_b = CleaningRecipe(
        dataset_id="ds",
        source_version=1,
        transforms=[CleaningTransform(type="drop_column", target_column="city")],
    )
    applied_b = apply_cleaning_recipe(
        loaded,
        recipe_b,
        output_dir=store,
        approved_lossy_transform_ids={recipe_b.transforms[0].transform_id},
    )

    assert applied_a.result.output_path != applied_b.result.output_path
    assert applied_a.result.output_path.exists()
    assert applied_b.result.output_path.exists()
    # A's output survives intact; B allocated the next free version.
    assert "city" in pd.read_csv(applied_a.result.output_path).columns
    assert "city" not in pd.read_csv(applied_b.result.output_path).columns
    assert applied_b.result.target_version == 3
    assert applied_b.preview.target_version == 3


def test_concurrent_applies_reserve_distinct_versions(tmp_path: Path) -> None:
    """C5: two barrier-synchronized applies of different recipes on the same
    source must land in two version directories with both outputs intact —
    mkdir(exist_ok=False) is the reservation lock."""
    import threading

    csv_path = tmp_path / "d.csv"
    csv_path.write_text("name,age,city\n A ,10,NYC\n B ,20,LA\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds")
    store = tmp_path / "store"

    recipe_a = CleaningRecipe(
        dataset_id="ds",
        source_version=1,
        transforms=[CleaningTransform(type="trim_whitespace", target_column="name")],
    )
    recipe_b = CleaningRecipe(
        dataset_id="ds",
        source_version=1,
        transforms=[CleaningTransform(type="drop_column", target_column="city")],
    )
    barrier = threading.Barrier(2)
    results: dict[str, object] = {}
    errors: list[BaseException] = []

    def worker(label: str, recipe: CleaningRecipe, approved: set[str]) -> None:
        try:
            barrier.wait()
            results[label] = apply_cleaning_recipe(
                loaded,
                recipe,
                output_dir=store,
                approved_lossy_transform_ids=approved,
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced in the assertion
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=("a", recipe_a, set())),
        threading.Thread(
            target=worker,
            args=("b", recipe_b, {recipe_b.transforms[0].transform_id}),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    applied_a = results["a"]
    applied_b = results["b"]
    assert isinstance(applied_a, AppliedCleaning)
    assert isinstance(applied_b, AppliedCleaning)
    # Two distinct version dirs, no clobbering.
    assert applied_a.result.output_path.parent != applied_b.result.output_path.parent
    assert {applied_a.result.target_version, applied_b.result.target_version} == {2, 3}
    # Both files landed complete and reflect their own recipe.
    frame_a = pd.read_csv(applied_a.result.output_path)
    frame_b = pd.read_csv(applied_b.result.output_path)
    assert list(frame_a["name"]) == ["A", "B"]
    assert "city" in frame_a.columns
    assert "city" not in frame_b.columns
    assert len(frame_b) == 2


# --------------------------------------------------------------------------- #
# CL-4: recipe.source_version must match the loaded record version
# --------------------------------------------------------------------------- #


def test_mismatched_source_version_raises_without_writing(tmp_path: Path) -> None:
    csv_path = tmp_path / "d.csv"
    csv_path.write_text("name\nAlice\nBob\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds")  # version 1
    recipe = CleaningRecipe(
        dataset_id="ds",
        source_version=5,  # lies about the source
        transforms=[CleaningTransform(type="trim_whitespace", target_column="name")],
    )
    out = tmp_path / "mm"
    with pytest.raises(ValueError, match="source_version"):
        apply_cleaning_recipe(loaded, recipe, output_dir=out)
    assert not (out / "ds" / "v6").exists()


# --------------------------------------------------------------------------- #
# CL-5: deletes and edits reported separately
# --------------------------------------------------------------------------- #


def test_preview_distinguishes_drops_from_edits(tmp_path: Path) -> None:
    # 100 dropped + 1 edited
    lines = ["name,val", " x ,5"] + [f"y{i}," for i in range(100)]
    csv_path = tmp_path / "big.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="db")
    recipe = CleaningRecipe(
        dataset_id="db",
        source_version=1,
        transforms=[
            CleaningTransform(type="trim_whitespace", target_column="name"),
            CleaningTransform(
                type="drop_rows",
                target_column="val",
                safety="lossy",
                params={"where": "missing"},
            ),
        ],
    )
    preview = preview_cleaning_recipe(loaded, recipe)
    assert preview.rows_dropped == 100
    assert preview.rows_edited == 1
    assert preview.cells_changed == 1

    # Reverse case: 1 dropped + many edited must be distinguishable.
    lines2 = ["name,val"] + [f" p{i} ,{i}" for i in range(50)] + ["q,"]
    csv_path2 = tmp_path / "big2.csv"
    csv_path2.write_text("\n".join(lines2) + "\n", encoding="utf-8")
    loaded2 = load_csv(csv_path2, dataset_id="dc")
    recipe2 = CleaningRecipe(
        dataset_id="dc",
        source_version=1,
        transforms=[
            CleaningTransform(type="trim_whitespace", target_column="name"),
            CleaningTransform(
                type="drop_rows",
                target_column="val",
                safety="lossy",
                params={"where": "missing"},
            ),
        ],
    )
    preview2 = preview_cleaning_recipe(loaded2, recipe2)
    assert preview2.rows_dropped == 1
    assert preview2.rows_edited == 50
    # The two scenarios are no longer indistinguishable single numbers.
    assert (preview.rows_dropped, preview.rows_edited) != (
        preview2.rows_dropped,
        preview2.rows_edited,
    )


# --------------------------------------------------------------------------- #
# CL-6: scientific notation parses; absurd ages rejected (value_parsing add-only)
# --------------------------------------------------------------------------- #


def test_scientific_notation_parses() -> None:
    assert parse_numeric_like("1.5e3", column_name="amount") == 1500.0
    assert parse_numeric_like("1e10", column_name="amount") == 1e10
    assert parse_numeric_like("1.5E3", column_name="amount") == 1500.0
    assert parse_numeric_like("-2e2", column_name="amount") == -200.0
    # Not accidentally matching non-numeric "e" tokens.
    assert parse_numeric_like("e", column_name="amount") is None
    assert parse_numeric_like("1e", column_name="amount") is None


def test_age_duration_behavior_preserved_and_absurd_capped() -> None:
    # Preserved: analysis.py + golden depend on this exact value.
    assert parse_numeric_like("27-003", column_name="age") == pytest.approx(27.01)
    assert parse_numeric_like("27-158", column_name="age") == pytest.approx(27.43)
    # Absurd ages rejected instead of 1000.0.
    assert parse_numeric_like("999-366", column_name="age") is None
    assert parse_numeric_like("160-034", column_name="age") is None
    # Boundary just below the cap still parses.
    assert parse_numeric_like("150-034", column_name="age") == pytest.approx(150.09)


# --------------------------------------------------------------------------- #
# CL-7: duplicate column names raise a clear error
# --------------------------------------------------------------------------- #


def test_duplicate_column_names_raise_clear_error() -> None:
    frame = pd.DataFrame([[" a ", 1], [" a ", 1]], columns=["v", "v"])
    recipe = CleaningRecipe(
        dataset_id="d",
        transforms=[CleaningTransform(type="trim_whitespace", target_column="v")],
    )
    with pytest.raises(ValueError, match="duplicate column name"):
        _apply_recipe_to_frame(frame, recipe)


# --------------------------------------------------------------------------- #
# CL-8: flag_constant_column and clip_outliers implemented
# --------------------------------------------------------------------------- #


def test_flag_constant_column_emits_warning() -> None:
    frame = pd.DataFrame({"k": [1, 1, 1], "v": [1, 2, 3]})
    recipe = CleaningRecipe(
        dataset_id="d",
        transforms=[
            CleaningTransform(type="flag_constant_column", target_column="k"),
            CleaningTransform(type="flag_constant_column", target_column="v"),
        ],
    )
    _, _, _, warnings = _apply_recipe_to_frame(frame, recipe)
    assert "constant_column:k" in warnings
    assert "constant_column:v" not in warnings


def test_clip_outliers_clips_to_explicit_and_iqr_bounds() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 1000.0]})
    recipe = CleaningRecipe(
        dataset_id="d",
        transforms=[
            CleaningTransform(
                type="clip_outliers",
                target_column="x",
                safety="lossy",
                params={"lower": 0, "upper": 10},
            )
        ],
    )
    cleaned, _, _, _ = _apply_recipe_to_frame(frame, recipe)
    assert cleaned["x"].tolist() == [1.0, 2.0, 3.0, 4.0, 10.0]

    frame2 = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
    recipe2 = CleaningRecipe(
        dataset_id="d",
        transforms=[
            CleaningTransform(type="clip_outliers", target_column="x", safety="lossy")
        ],
    )
    cleaned2, _, _, _ = _apply_recipe_to_frame(frame2, recipe2)
    assert float(_col(cleaned2, "x").max()) < 1000.0  # IQR fence clipped the outlier
    assert _col(cleaned2, "x").tolist()[:5] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_clip_outliers_is_lossy_and_gated(tmp_path: Path) -> None:
    csv_path = tmp_path / "d.csv"
    csv_path.write_text("x\n1\n2\n3\n4\n1000\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds")
    recipe = CleaningRecipe(
        dataset_id="ds",
        source_version=1,
        transforms=[
            CleaningTransform(
                type="clip_outliers",
                target_column="x",
                safety="safe",  # mislabeled; type is lossy
                params={"lower": 0, "upper": 10},
            )
        ],
    )
    assert recipe.requires_approval is True
    with pytest.raises(ValueError, match="requires approval"):
        apply_cleaning_recipe(loaded, recipe, output_dir=tmp_path / "out")


# --------------------------------------------------------------------------- #
# Pre-cleaning drop transforms: replayable + gated as lossy
# --------------------------------------------------------------------------- #


def test_drop_missing_rows_removes_rows_with_any_missing() -> None:
    frame = pd.DataFrame({"a": [1, None, 3, 4], "b": ["x", "y", None, "w"]})
    recipe = CleaningRecipe(
        dataset_id="d",
        transforms=[CleaningTransform(type="drop_missing_rows", safety="lossy")],
    )
    cleaned, _, tally, _ = _apply_recipe_to_frame(frame, recipe)
    assert tally.rows_dropped == 2
    assert cleaned["a"].tolist() == [1.0, 4.0]
    assert cleaned["b"].tolist() == ["x", "w"]


def test_drop_missing_rows_honours_subset_param() -> None:
    frame = pd.DataFrame({"a": [1, None, 3], "b": [None, "y", "z"]})
    recipe = CleaningRecipe(
        dataset_id="d",
        transforms=[
            CleaningTransform(
                type="drop_missing_rows", safety="lossy", params={"subset": ["a"]}
            )
        ],
    )
    cleaned, _, tally, _ = _apply_recipe_to_frame(frame, recipe)
    # Only the row missing in "a" is dropped; the row missing only in "b" stays.
    assert tally.rows_dropped == 1
    assert cleaned["a"].tolist() == [1.0, 3.0]


def test_drop_outlier_rows_removes_iqr_outliers() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 900]})
    recipe = CleaningRecipe(
        dataset_id="d",
        transforms=[CleaningTransform(type="drop_outlier_rows", safety="lossy")],
    )
    cleaned, _, tally, _ = _apply_recipe_to_frame(frame, recipe)
    assert tally.rows_dropped == 1
    assert 900 not in cleaned["x"].tolist()


@pytest.mark.parametrize("transform_type", ["drop_missing_rows", "drop_outlier_rows"])
def test_precleaning_drop_transforms_are_lossy_and_gated(
    tmp_path: Path, transform_type: str
) -> None:
    csv_path = tmp_path / "d.csv"
    csv_path.write_text("x\n1\n2\n3\n4\n1000\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds")
    recipe = CleaningRecipe(
        dataset_id="ds",
        source_version=1,
        transforms=[
            CleaningTransform(
                type=cast(CleaningTransformType, transform_type),
                safety="safe",  # mislabeled; the op deletes rows -> lossy
            )
        ],
    )
    assert recipe.requires_approval is True
    with pytest.raises(ValueError, match="requires approval"):
        apply_cleaning_recipe(loaded, recipe, output_dir=tmp_path / "out")


# --------------------------------------------------------------------------- #
# Preserved invariants (regression guards for verified-correct behavior)
# --------------------------------------------------------------------------- #


def test_apply_does_not_mutate_caller_frame_or_source_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "people.csv"
    csv_path.write_text(
        "name,age\n Alice ,27-003\nBob,31\n Alice ,27-003\n", encoding="utf-8"
    )
    original_bytes = csv_path.read_bytes()
    loaded = load_csv(csv_path, dataset_id="ds_people")
    before_frame = loaded.frame.copy(deep=True)

    recipe = CleaningRecipe(
        dataset_id="ds_people",
        source_version=1,
        transforms=[
            CleaningTransform(type="trim_whitespace", target_column="name"),
            CleaningTransform(type="parse_numeric", target_column="age"),
            CleaningTransform(type="drop_duplicate_rows"),
        ],
    )
    applied = apply_cleaning_recipe(
        loaded,
        recipe,
        output_dir=tmp_path / "cleaned",
        approved_lossy_transform_ids={
            transform.transform_id
            for transform in recipe.transforms
            if transform_is_lossy(transform)
        },
    )

    # Caller frame + original file are byte/value identical after apply.
    pd.testing.assert_frame_equal(loaded.frame, before_frame)
    assert csv_path.read_bytes() == original_bytes
    assert loaded.frame is not applied.loaded.frame


def test_mixed_safe_and_unapproved_lossy_is_all_or_nothing(tmp_path: Path) -> None:
    csv_path = tmp_path / "mix.csv"
    pd.DataFrame({"name": [" A ", " B "], "a": [1.0, None]}).to_csv(csv_path, index=False)
    loaded = load_csv(csv_path, dataset_id="dmix")
    recipe = CleaningRecipe(
        dataset_id="dmix",
        source_version=1,
        transforms=[
            CleaningTransform(type="trim_whitespace", target_column="name"),
            CleaningTransform(
                type="drop_rows",
                target_column="a",
                safety="lossy",
                params={"where": "missing"},
            ),
        ],
    )
    out = tmp_path / "mixout"
    with pytest.raises(ValueError, match="requires approval"):
        apply_cleaning_recipe(loaded, recipe, output_dir=out)
    # No partial output; caller frame untouched.
    assert not (out / "dmix").exists()
    assert loaded.frame["name"].tolist() == [" A ", " B "]


def test_empty_and_all_nan_frames_do_not_crash(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty.csv"
    pd.DataFrame(
        {"name": pd.Series([], dtype="object"), "age": pd.Series([], dtype="object")}
    ).to_csv(empty_path, index=False)
    loaded = load_csv(empty_path, dataset_id="ds_empty")
    recipe = CleaningRecipe(
        dataset_id="ds_empty",
        source_version=1,
        transforms=[
            CleaningTransform(type="trim_whitespace", target_column="name"),
            CleaningTransform(type="parse_numeric", target_column="age"),
            CleaningTransform(type="drop_duplicate_rows"),
        ],
    )
    preview = preview_cleaning_recipe(loaded, recipe)
    assert preview.row_count_before == 0
    assert preview.rows_dropped == 0
    assert preview.rows_edited == 0

    all_nan = pd.DataFrame({"x": [None, None, None]})
    recipe2 = CleaningRecipe(
        dataset_id="d",
        transforms=[CleaningTransform(type="parse_numeric", target_column="x")],
    )
    cleaned, _, tally, _ = _apply_recipe_to_frame(all_nan, recipe2)
    assert bool(_col(cleaned, "x").isna().all())
    assert tally.rows_edited == 0
