"""Permission-gated dispatch for applying a cleaning recipe."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from eda_platform.core.ids import stable_hash
from eda_platform.core.permissions import (
    PermissionTier,
    classify_action,
    pending_action_payload,
    require_permission,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.core.trace import trace_event
from eda_platform.schemas.cleaning import CleaningPreview, CleaningRecipe, transform_is_lossy
from eda_platform.tools.cleaning import (
    AppliedCleaning,
    apply_cleaning_recipe,
    preview_cleaning_recipe,
)
from eda_platform.tools.loader import LoadedDataset


@dataclass(frozen=True)
class CleaningDispatch:
    status: Literal["awaiting_approval", "applied", "refused"]
    message: str
    preview: CleaningPreview | None = None
    pending_action: dict[str, Any] | None = None
    applied: AppliedCleaning | None = None


def cleaning_action(recipe: CleaningRecipe) -> dict[str, Any]:
    """The canonical action dict whose hash an approval must match."""
    return {
        "type": "cleaning_apply",
        "dataset_id": recipe.dataset_id,
        "recipe_id": recipe.recipe_id,
        "transform_ids": [transform.transform_id for transform in recipe.transforms],
        "reversible": not recipe.requires_approval,
        "recipe_digest": stable_hash(recipe.model_dump(mode="json"), length=32),
    }


def preview_cleaning(loaded: LoadedDataset, recipe: CleaningRecipe) -> CleaningDispatch:
    """Read-only: compute the diff and the pending CONFIRM action for the UI."""
    preview = preview_cleaning_recipe(loaded, recipe)
    decision = classify_action(cleaning_action(recipe))
    return CleaningDispatch(
        status="awaiting_approval",
        message=decision.description,
        preview=preview,
        pending_action=pending_action_payload(decision),
    )


def apply_cleaning(
    loaded: LoadedDataset,
    recipe: CleaningRecipe,
    *,
    output_dir: Path | str,
    approved_hash: str | None,
    store: ArtifactStore | None = None,
    project_id: str = "default",
    session_id: str = "cleaning",
) -> CleaningDispatch:
    """Apply a recipe only if the approval hash matches its exact content."""
    action = cleaning_action(recipe)
    decision = require_permission(action, approved_hash=approved_hash)
    if decision.tier is PermissionTier.DENY or not decision.approved:
        _append_trace(store, project_id, session_id, "permission_denied", recipe, decision.feedback)
        return CleaningDispatch(status="refused", message=decision.feedback or "Cleaning denied.")

    lossy_ids = {
        transform.transform_id for transform in recipe.transforms if transform_is_lossy(transform)
    }
    applied = apply_cleaning_recipe(
        loaded,
        recipe,
        output_dir=output_dir,
        approved_lossy_transform_ids=lossy_ids,
    )
    _append_trace(
        store,
        project_id,
        session_id,
        "cleaning_applied",
        recipe,
        (
            f"v{applied.result.source_version}->v{applied.result.target_version}: "
            f"{applied.result.output_path}"
        ),
    )
    return CleaningDispatch(
        status="applied",
        message=(
            f"Applied recipe {recipe.recipe_id}: new version "
            f"v{applied.result.target_version} written."
        ),
        preview=applied.preview,
        applied=applied,
    )


def _append_trace(
    store: ArtifactStore | None,
    project_id: str,
    session_id: str,
    event_type: str,
    recipe: CleaningRecipe,
    detail: str,
) -> None:
    if store is None:
        return
    store.append_trace(
        project_id,
        trace_event(
            session_id=session_id,
            event_type=event_type,
            name="m6_cleaning_apply",
            started_at=datetime.now(UTC),
            summary={
                "recipe_id": recipe.recipe_id,
                "dataset_id": recipe.dataset_id,
                "detail": detail,
            },
        ),
    )
