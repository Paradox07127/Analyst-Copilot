from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pandas as pd

from eda_platform.core.ids import hash_file
from eda_platform.core.tool_guard import (
    GuardViolation,
    check_column_exists,
    check_column_semantic_type,
    check_enum,
    check_non_empty,
    check_range,
    raise_for_violations,
)
from eda_platform.schemas.cleaning import (
    CleaningApplyResult,
    CleaningColumnDiff,
    CleaningPreview,
    CleaningRecipe,
    CleaningTransform,
    CleaningValueExample,
    transform_is_lossy,
)
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.tools.frame_stats import drop_iqr_outlier_rows
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.value_parsing import parse_numeric_like

_CLEANING_TRANSFORM_TYPES: tuple[str, ...] = (
    "trim_whitespace",
    "parse_numeric",
    "drop_duplicate_rows",
    "drop_rows",
    "drop_missing_rows",
    "drop_outlier_rows",
    "fill_missing",
    "drop_column",
    "clip_outliers",
    "flag_constant_column",
)
_COLUMN_TARGET_TRANSFORMS: frozenset[str] = frozenset(
    {
        "trim_whitespace",
        "parse_numeric",
        "drop_rows",
        "fill_missing",
        "drop_column",
        "clip_outliers",
        "flag_constant_column",
    }
)


@dataclass(frozen=True)
class AppliedCleaning:
    loaded: LoadedDataset
    preview: CleaningPreview
    result: CleaningApplyResult


@dataclass
class _ColumnChange:
    before_dtype: str
    after_dtype: str
    before_missing: int
    after_missing: int
    changed_rows: int = 0
    examples: list[CleaningValueExample] | None = None


@dataclass
class _RecipeTally:
    """Separates deletes from edits so a preview shows true blast radius (CL-5)."""

    rows_dropped: int = 0
    dropped_row_indexes: set[Any] = field(default_factory=set)
    edited_rows: set[Any] = field(default_factory=set)
    cells_changed: int = 0

    @property
    def rows_edited(self) -> int:
        return len(self.edited_rows)


def preview_cleaning_recipe(
    loaded: LoadedDataset,
    recipe: CleaningRecipe,
) -> CleaningPreview:
    _require_recipe_dataset(loaded, recipe)
    guard_cleaning_recipe_params(loaded.frame, recipe)
    cleaned, changes, tally, warnings = _apply_recipe_to_frame(loaded.frame, recipe)
    diffs = [
        CleaningColumnDiff(
            column=column,
            before_dtype=change.before_dtype,
            after_dtype=change.after_dtype,
            before_missing=change.before_missing,
            after_missing=change.after_missing,
            changed_rows=change.changed_rows,
            examples=change.examples or [],
        )
        for column, change in sorted(changes.items())
        if change.changed_rows > 0 or change.before_dtype != change.after_dtype
    ]
    rows_dropped = int(tally.rows_dropped)
    rows_edited = int(tally.rows_edited)
    affected_rows = len(tally.dropped_row_indexes | tally.edited_rows)
    return CleaningPreview(
        dataset_id=recipe.dataset_id,
        recipe_id=recipe.recipe_id,
        source_version=recipe.source_version,
        target_version=recipe.source_version + 1,
        row_count_before=int(len(loaded.frame)),
        row_count_after=int(len(cleaned)),
        affected_rows=int(affected_rows),
        rows_dropped=rows_dropped,
        rows_edited=rows_edited,
        cells_changed=int(tally.cells_changed),
        column_diffs=diffs,
        warnings=warnings,
    )


def apply_cleaning_recipe(
    loaded: LoadedDataset,
    recipe: CleaningRecipe,
    *,
    output_dir: Path | str,
    approved_lossy_transform_ids: set[str] | None = None,
) -> AppliedCleaning:
    # CL-4: lineage must be derived from the loaded record, not a caller-claimed
    # source_version, or we mint bogus versions (record v1 + recipe v5 -> v6/parent 1).
    _require_recipe_dataset(loaded, recipe)
    if recipe.source_version != loaded.record.version:
        raise ValueError(
            "Recipe source_version does not match loaded dataset version: "
            f"recipe.source_version={recipe.source_version} != "
            f"loaded.record.version={loaded.record.version}."
        )
    guard_cleaning_recipe_params(loaded.frame, recipe)

    approved = approved_lossy_transform_ids or set()
    # derive the tier from the operation type server-side.
    unapproved = [
        transform
        for transform in recipe.transforms
        if transform_is_lossy(transform) and transform.transform_id not in approved
    ]
    if unapproved:
        ids = ", ".join(transform.transform_id for transform in unapproved)
        raise ValueError(f"Cleaning transform requires approval: {ids}")

    cleaned, _, _, _ = _apply_recipe_to_frame(loaded.frame, recipe)
    preview = preview_cleaning_recipe(loaded, recipe)
    # CL-2/C5: never silently clobber an existing version. mkdir(exist_ok=False)
    # is the reservation lock, so two concurrent applies of the same source can
    # never allocate the same version directory.
    target_version, _base_dir, output_path = _reserve_version_dir(
        Path(output_dir), recipe.dataset_id, loaded.record.version + 1, loaded.record.name
    )
    if target_version != preview.target_version:
        preview = preview.model_copy(update={"target_version": target_version})
    cleaned.to_csv(output_path, index=False)
    record = DatasetRecord(
        dataset_id=loaded.record.dataset_id,
        name=loaded.record.name,
        path=output_path,
        content_hash=hash_file(output_path),
        version=target_version,
        parent_dataset_id=loaded.record.dataset_id,
        parent_version=loaded.record.version,
        lineage_recipe_id=recipe.recipe_id,
        encoding=loaded.record.encoding,
        delimiter=loaded.record.delimiter,
    )
    result = CleaningApplyResult(
        dataset_id=recipe.dataset_id,
        recipe_id=recipe.recipe_id,
        source_version=recipe.source_version,
        target_version=target_version,
        output_path=output_path,
        row_count_before=preview.row_count_before,
        row_count_after=preview.row_count_after,
        applied_transform_ids=[transform.transform_id for transform in recipe.transforms],
    )
    return AppliedCleaning(
        loaded=LoadedDataset(record=record, frame=cleaned),
        preview=preview,
        result=result,
    )


def guard_cleaning_recipe_params(frame: pd.DataFrame, recipe: CleaningRecipe) -> None:
    violations: list[GuardViolation | None] = [
        check_non_empty("dataset_id", recipe.dataset_id),
        check_range(
            "source_version",
            recipe.source_version,
            minimum=1.0,
            fix_hint="Set `source_version` to the loaded dataset version, at least 1.",
        ),
    ]
    available_columns = [str(column) for column in frame.columns]
    for index, transform in enumerate(recipe.transforms):
        prefix = f"transforms[{index}]"
        transform_type = str(transform.type)
        violations.append(
            check_enum(f"{prefix}.type", transform_type, _CLEANING_TRANSFORM_TYPES)
        )
        if transform_type in _COLUMN_TARGET_TRANSFORMS:
            violations.append(
                check_non_empty(
                    f"{prefix}.target_column",
                    transform.target_column,
                    fix_hint=f"Set `{prefix}.target_column` to an existing column.",
                )
            )
            if _is_non_empty_string(transform.target_column):
                if transform_type == "clip_outliers":
                    violations.append(
                        check_column_semantic_type(
                            f"{prefix}.target_column",
                            transform.target_column,
                            frame,
                            allowed_semantic_types=("numeric",),
                            fix_hint=(
                                f"Choose a numeric column for `{prefix}.target_column` "
                                "or use a non-numeric cleaning transform."
                            ),
                        )
                    )
                else:
                    violations.append(
                        check_column_exists(
                            f"{prefix}.target_column",
                            transform.target_column,
                            available_columns,
                        )
                    )
        if transform_type == "drop_rows":
            violations.append(
                check_enum(
                    f"{prefix}.params.where",
                    transform.params.get("where"),
                    ("missing",),
                    fix_hint="Set `params.where` to `missing` for drop_rows.",
                )
            )
        elif transform_type == "drop_missing_rows":
            subset = transform.params.get("subset")
            if subset is not None:
                if not isinstance(subset, list):
                    violations.append(
                        GuardViolation(
                            field=f"{prefix}.params.subset",
                            got=subset,
                            allowed="a list of existing column names",
                            fix_hint="Provide `params.subset` as a JSON list of column names.",
                            problem="expected a list, not a scalar string or object.",
                        )
                    )
                violations.append(
                    check_non_empty(
                        f"{prefix}.params.subset",
                        subset,
                        fix_hint=(
                            "Omit `params.subset` to scan all columns, or provide "
                            "one or more existing columns."
                        ),
                    )
                )
                if isinstance(subset, list):
                    for subset_index, column in enumerate(subset):
                        violations.append(
                            check_column_exists(
                                f"{prefix}.params.subset[{subset_index}]",
                                column,
                                available_columns,
                            )
                        )
        elif transform_type == "fill_missing" and "value" not in transform.params:
            violations.append(
                GuardViolation(
                    field=f"{prefix}.params.value",
                    got=None,
                    allowed="a replacement value for missing cells",
                    fix_hint="Set `params.value` to the value that should replace missing cells.",
                    problem="required parameter is missing.",
                )
            )
        elif transform_type == "clip_outliers":
            lower = transform.params.get("lower")
            upper = transform.params.get("upper")
            if lower is not None:
                violations.append(check_range(f"{prefix}.params.lower", lower))
            if upper is not None:
                violations.append(check_range(f"{prefix}.params.upper", upper))
            if isinstance(lower, int | float) and isinstance(upper, int | float) and lower > upper:
                violations.append(
                    GuardViolation(
                        field=f"{prefix}.params.lower",
                        got=lower,
                        allowed="lower must be less than or equal to upper",
                        fix_hint=(
                            "Use a lower bound that is <= `params.upper`, or omit "
                            "one bound to let the tool infer it."
                        ),
                        problem=f"lower is greater than upper ({upper}).",
                    )
                )
    raise_for_violations("cleaning_recipe", violations)


def _apply_recipe_to_frame(
    frame: pd.DataFrame,
    recipe: CleaningRecipe,
) -> tuple[pd.DataFrame, dict[str, _ColumnChange], _RecipeTally, list[str]]:
    _require_unique_columns(frame)  # CL-7: fail clearly instead of a cryptic pandas error
    working = frame.copy()
    changes: dict[str, _ColumnChange] = {}
    tally = _RecipeTally()
    warnings: list[str] = []
    for transform in recipe.transforms:
        if transform.type == "trim_whitespace":
            _apply_trim(working, transform, changes, tally)
        elif transform.type == "parse_numeric":
            _apply_parse_numeric(working, transform, changes, tally)
        elif transform.type == "drop_duplicate_rows":
            before = working
            working = working.drop_duplicates()
            _record_dropped_rows(before, working, tally)
        elif transform.type == "drop_rows":
            before = working
            working = _apply_drop_rows(working, transform)
            _record_dropped_rows(before, working, tally)
        elif transform.type == "drop_missing_rows":
            before = working
            working = _apply_drop_missing_rows(working, transform)
            _record_dropped_rows(before, working, tally)
        elif transform.type == "drop_outlier_rows":
            before = working
            working = drop_iqr_outlier_rows(working)
            _record_dropped_rows(before, working, tally)
        elif transform.type == "fill_missing":
            _apply_fill_missing(working, transform, changes, tally)
        elif transform.type == "drop_column":
            if transform.target_column in working.columns:
                working = working.drop(columns=[transform.target_column])
        elif transform.type == "clip_outliers":
            _apply_clip_outliers(working, transform, changes, tally)
        elif transform.type == "flag_constant_column":
            _apply_flag_constant_column(working, transform, warnings)
        else:
            raise ValueError(f"Unsupported cleaning transform: {transform.type}")
    return working.reset_index(drop=True), changes, tally, warnings


def _require_recipe_dataset(loaded: LoadedDataset, recipe: CleaningRecipe) -> None:
    if recipe.dataset_id != loaded.record.dataset_id:
        raise ValueError(
            "Recipe dataset_id does not match loaded dataset: "
            f"{recipe.dataset_id} != {loaded.record.dataset_id}."
        )


def _record_dropped_rows(
    before: pd.DataFrame,
    after: pd.DataFrame,
    tally: _RecipeTally,
) -> None:
    removed = set(before.index) - set(after.index)
    tally.dropped_row_indexes.update(removed)
    tally.rows_dropped += len(removed)


def _apply_trim(
    frame: pd.DataFrame,
    transform: CleaningTransform,
    changes: dict[str, _ColumnChange],
    tally: _RecipeTally,
) -> None:
    column = _require_column(transform)
    before = _series(frame, column)
    after = before.map(lambda value: value.strip() if isinstance(value, str) else value)
    _record_change(changes, column, before, after, tally)
    frame[column] = after


def _apply_parse_numeric(
    frame: pd.DataFrame,
    transform: CleaningTransform,
    changes: dict[str, _ColumnChange],
    tally: _RecipeTally,
) -> None:
    column = _require_column(transform)
    before = _series(frame, column)
    after = pd.Series(
        [parse_numeric_like(value, column_name=column) for value in before],
        index=before.index,
        dtype="float64",
    )
    _record_change(changes, column, before, after, tally)
    frame[column] = after


def _apply_drop_rows(frame: pd.DataFrame, transform: CleaningTransform) -> pd.DataFrame:
    column = _require_column(transform)
    if transform.params.get("where") == "missing":
        return frame.dropna(subset=[column])
    raise ValueError(f"Unsupported drop_rows params: {transform.params}")


def _apply_drop_missing_rows(
    frame: pd.DataFrame,
    transform: CleaningTransform,
) -> pd.DataFrame:
    # Drop every row with a missing value.
    subset = transform.params.get("subset")
    columns = [column for column in subset if column in frame.columns] if subset else None
    if subset and not columns:
        return frame
    return frame.dropna(subset=columns)


def _apply_fill_missing(
    frame: pd.DataFrame,
    transform: CleaningTransform,
    changes: dict[str, _ColumnChange],
    tally: _RecipeTally,
) -> None:
    column = _require_column(transform)
    before = _series(frame, column)
    after = before.fillna(transform.params.get("value"))
    _record_change(changes, column, before, after, tally)
    frame[column] = after


def _apply_clip_outliers(
    frame: pd.DataFrame,
    transform: CleaningTransform,
    changes: dict[str, _ColumnChange],
    tally: _RecipeTally,
) -> None:
    # CL-8: lossy op. Explicit lower/upper win; otherwise clip to IQR fences.
    column = _require_column(transform)
    before = _series(frame, column)
    numeric = cast(pd.Series, pd.to_numeric(before, errors="coerce"))
    lower = transform.params.get("lower")
    upper = transform.params.get("upper")
    if lower is None or upper is None:
        finite = numeric.dropna()
        if len(finite) >= 4:
            q1 = float(finite.quantile(0.25))
            q3 = float(finite.quantile(0.75))
            iqr = q3 - q1
            if lower is None:
                lower = q1 - 1.5 * iqr
            if upper is None:
                upper = q3 + 1.5 * iqr
    after = numeric.clip(lower=lower, upper=upper)
    _record_change(changes, column, before, after, tally)
    frame[column] = after


def _apply_flag_constant_column(
    frame: pd.DataFrame,
    transform: CleaningTransform,
    warnings: list[str],
) -> None:
    # CL-8: non-destructive diagnostic. Emit a warning when the column is constant.
    column = _require_column(transform)
    series = _series(frame, column)
    if series.nunique(dropna=False) <= 1:
        warnings.append(f"constant_column:{column}")


def _record_change(
    changes: dict[str, _ColumnChange],
    column: str,
    before: pd.Series,
    after: pd.Series,
    tally: _RecipeTally | None = None,
) -> None:
    mask = _changed_mask(before, after)
    if tally is not None:
        changed_count = int(mask.sum())
        tally.cells_changed += changed_count
        if changed_count:
            tally.edited_rows.update(cast(Any, mask[mask]).index.tolist())
    existing = changes.get(column)
    examples = _examples(before, after, mask)
    if existing is None:
        changes[column] = _ColumnChange(
            before_dtype=str(before.dtype),
            after_dtype=str(after.dtype),
            before_missing=int(before.isna().sum()),
            after_missing=int(after.isna().sum()),
            changed_rows=int(mask.sum()),
            examples=examples,
        )
        return
    existing.after_dtype = str(after.dtype)
    existing.after_missing = int(after.isna().sum())
    existing.changed_rows += int(mask.sum())
    existing.examples = [*(existing.examples or []), *examples][:5]


def _require_unique_columns(frame: pd.DataFrame) -> None:
    if frame.columns.has_duplicates:
        duplicated = frame.columns[frame.columns.duplicated()].unique().tolist()
        names = ", ".join(str(name) for name in duplicated)
        raise ValueError(
            f"Cannot clean a frame with duplicate column name(s): {names}. "
            "Rename or deduplicate columns before cleaning."
        )


def _changed_mask(before: pd.Series, after: pd.Series) -> pd.Series:
    comparable = pd.DataFrame({"before": before.to_numpy(), "after": after.to_numpy()})
    both_missing = comparable["before"].isna() & comparable["after"].isna()
    changed = (comparable["before"] != comparable["after"]) & ~both_missing
    changed.index = before.index
    return cast(pd.Series, changed)


def _examples(
    before: pd.Series,
    after: pd.Series,
    mask: pd.Series,
    *,
    limit: int = 5,
) -> list[CleaningValueExample]:
    changed_indexes = list(cast(Any, mask[mask]).index)[:limit]
    return [
        CleaningValueExample(
            before=_json_scalar(before.loc[index]),
            after=_json_scalar(after.loc[index]),
        )
        for index in changed_indexes
    ]


def _json_scalar(value: Any) -> str | int | float | bool | None:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _require_column(transform: CleaningTransform) -> str:
    if transform.target_column is None:
        raise ValueError(f"{transform.type} requires target_column")
    return transform.target_column


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def _next_free_version(
    output_dir: Path,
    dataset_id: str,
    start_version: int,
    name: str,
) -> tuple[int, Path, Path]:
    """CL-2: next version whose directory does not already exist (read-only
    probe; mirrors the reservation rule in :func:`_reserve_version_dir`)."""
    dataset_dir = _safe_dataset_output_dir(output_dir, dataset_id, create=False)
    version = start_version
    while True:
        base_dir = dataset_dir / f"v{version}"
        if not base_dir.exists():
            return version, base_dir, base_dir / name
        version += 1


_VERSION_RESERVE_ATTEMPTS = 100


def _reserve_version_dir(
    output_dir: Path,
    dataset_id: str,
    start_version: int,
    name: str,
) -> tuple[int, Path, Path]:
    """C5: atomically claim the next free version directory via
    mkdir(exist_ok=False); a FileExistsError means a concurrent apply won that
    slot, so probe the next one."""
    dataset_dir = _safe_dataset_output_dir(output_dir, dataset_id, create=True)
    version = start_version
    for _ in range(_VERSION_RESERVE_ATTEMPTS):
        base_dir = dataset_dir / f"v{version}"
        try:
            base_dir.mkdir(exist_ok=False)
        except FileExistsError:
            version += 1
            continue
        if base_dir.is_symlink() or base_dir.resolve().parent != dataset_dir.resolve():
            raise RuntimeError("Cleaned version directory escaped its dataset directory.")
        return version, base_dir, base_dir / name
    raise RuntimeError(
        f"Could not reserve a cleaned version directory for {dataset_id} after "
        f"{_VERSION_RESERVE_ATTEMPTS} attempts starting at v{start_version}."
    )


def _safe_dataset_output_dir(
    output_dir: Path, dataset_id: str, *, create: bool
) -> Path:
    """Reject a symlinked dataset parent before version probing or creation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise RuntimeError("Cleaning output directory must be a real directory.")
    output_root = output_dir.resolve()
    dataset_dir = output_dir / dataset_id
    if dataset_dir.is_symlink():
        raise RuntimeError("Cleaning dataset directory cannot be a symbolic link.")
    if create:
        dataset_dir.mkdir(exist_ok=True)
    if dataset_dir.exists():
        if not dataset_dir.is_dir() or dataset_dir.resolve().parent != output_root:
            raise RuntimeError("Cleaning dataset directory escaped its output directory.")
    elif dataset_dir.resolve().parent != output_root:
        raise RuntimeError("Cleaning dataset directory escaped its output directory.")
    return dataset_dir
