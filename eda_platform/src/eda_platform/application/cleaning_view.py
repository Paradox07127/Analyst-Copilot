"""Cleaning-transparency row shaping shared by API presentation services."""

from __future__ import annotations

from eda_platform.schemas.artifacts import Artifact, DatasetProfile
from eda_platform.schemas.cleaning import CleaningRecipe


def dataset_names_from_profile_artifacts(profile_artifacts: list[Artifact]) -> dict[str, str]:
    names: dict[str, str] = {}
    for artifact in profile_artifacts:
        profile = DatasetProfile.model_validate(artifact.payload)
        names[profile.dataset_id] = profile.name
    return names


def cleaning_summary_rows(recipe_artifacts: list[Artifact]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for artifact in recipe_artifacts:
        recipe = CleaningRecipe.model_validate(artifact.payload)
        lineage = recipe.lineage
        rows.append(
            {
                "dataset": lineage.source_name if lineage else recipe.dataset_id,
                "recipe_id": recipe.recipe_id,
                "rows_before": lineage.rows_before if lineage else None,
                "rows_after": lineage.rows_after if lineage else None,
                "rows_removed": (lineage.rows_before - lineage.rows_after if lineage else None),
                "columns_before": lineage.columns_before if lineage else None,
                "columns_after": lineage.columns_after if lineage else None,
                "columns_removed": (
                    lineage.columns_before - lineage.columns_after if lineage else None
                ),
                "delete_steps": len(recipe.transforms),
                "protection_triggers": len(recipe.guardrails),
                "requires_approval": recipe.requires_approval,
            }
        )
    return rows


def cleaning_operation_rows(recipe_artifacts: list[Artifact]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for artifact in recipe_artifacts:
        recipe = CleaningRecipe.model_validate(artifact.payload)
        dataset = recipe.lineage.source_name if recipe.lineage else recipe.dataset_id
        for transform in recipe.transforms:
            rows.append(
                {
                    "dataset": dataset,
                    "operation": transform.type,
                    "column": transform.target_column or "",
                    "rows_deleted": transform.expected_impact_rows or 0,
                    "columns_deleted": 1 if transform.type == "drop_column" else 0,
                    "reason": cleaning_reason(transform.params),
                    "details": transform.description,
                }
            )
    return rows


def cleaning_guardrail_rows(recipe_artifacts: list[Artifact]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for artifact in recipe_artifacts:
        recipe = CleaningRecipe.model_validate(artifact.payload)
        dataset = recipe.lineage.source_name if recipe.lineage else recipe.dataset_id
        for guardrail in recipe.guardrails:
            rows.append(
                {
                    "dataset": dataset,
                    "code": guardrail.code,
                    "reason": guardrail.message,
                    "thresholds": format_cleaning_params(guardrail.params),
                }
            )
    return rows


def cleaning_suggestion_rows(recipe_artifacts: list[Artifact]) -> list[dict[str, str]]:
    operation_types: set[str] = set()
    guard_codes: set[str] = set()
    for artifact in recipe_artifacts:
        recipe = CleaningRecipe.model_validate(artifact.payload)
        operation_types.update(transform.type for transform in recipe.transforms)
        guard_codes.update(guardrail.code for guardrail in recipe.guardrails)

    suggestions: list[str] = []
    if "drop_column" in operation_types:
        suggestions.append(
            "Review dropped high-missing columns before modeling; some may be useful "
            "as missingness flags."
        )
    if "drop_missing_rows" in operation_types:
        suggestions.append(
            "Check whether removed missing rows are random or concentrated in a key group."
        )
    if "drop_outlier_rows" in operation_types:
        suggestions.append("Inspect outlier rows separately before treating them as errors.")
    if guard_codes:
        suggestions.append(
            "When protection triggers, consider a less aggressive threshold, "
            "imputation, or manual review."
        )
    return [{"suggestion": suggestion} for suggestion in suggestions]


def cleaning_reason(params: dict[str, object]) -> str:
    reason = params.get("reason") or params.get("method")
    return str(reason) if reason is not None else ""


def format_cleaning_params(params: dict[str, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in params.items())
