"""M6.1 — the cleaning-apply CONFIRM dispatch must be permission-gated.

These are the adversarial cases the earlier audit called for: the `cleaning_apply`
CONFIRM tier had zero dispatch points, so nothing proved that applying a recipe
actually goes through the approval gate. Each test drives the real
`drivers.cleaning_apply` path.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eda_platform.core.permissions import action_hash
from eda_platform.drivers.cleaning_apply import (
    apply_cleaning,
    cleaning_action,
    preview_cleaning,
)
from eda_platform.schemas.cleaning import CleaningRecipe, CleaningTransform
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.tools.loader import LoadedDataset


def _loaded(tmp_path: Path) -> LoadedDataset:
    source = tmp_path / "customers.csv"
    frame = pd.DataFrame({"name": [" a ", "b ", "b "], "age": [1, 2, 2]})
    frame.to_csv(source, index=False)
    record = DatasetRecord(
        dataset_id="ds_customers",
        name="customers.csv",
        path=source,
        content_hash="hash",
        version=1,
    )
    return LoadedDataset(record=record, frame=frame)


def _safe_recipe() -> CleaningRecipe:
    return CleaningRecipe(
        dataset_id="ds_customers",
        source_version=1,
        recipe_id="recipe_safe",
        transforms=[
            CleaningTransform(transform_id="t_trim", type="trim_whitespace", target_column="name"),
        ],
    )


def test_apply_without_approval_is_denied(tmp_path: Path) -> None:
    loaded = _loaded(tmp_path)
    result = apply_cleaning(
        loaded, _safe_recipe(), output_dir=tmp_path / "out", approved_hash=None
    )
    assert result.status == "refused"
    assert "Tool guard rejected" in result.message
    assert not (tmp_path / "out").exists()


def test_approve_then_swap_recipe_params_is_denied(tmp_path: Path) -> None:
    loaded = _loaded(tmp_path)
    approved = _safe_recipe()
    # Same dataset, same recipe_id, same transform_id — but a different, more
    # destructive operation. Binding only transform_ids would let this through;
    # the recipe_digest makes the approval hash not match.
    swapped = CleaningRecipe(
        dataset_id="ds_customers",
        source_version=1,
        recipe_id="recipe_safe",
        transforms=[
            CleaningTransform(transform_id="t_trim", type="drop_column", target_column="age"),
        ],
    )
    approved_hash = action_hash(cleaning_action(approved))

    result = apply_cleaning(
        loaded, swapped, output_dir=tmp_path / "out", approved_hash=approved_hash
    )
    assert result.status == "refused"
    assert "hash" in result.message.lower()
    assert not (tmp_path / "out").exists()


def test_approved_apply_produces_new_version_and_leaves_source_intact(tmp_path: Path) -> None:
    loaded = _loaded(tmp_path)
    recipe = _safe_recipe()
    approved_hash = action_hash(cleaning_action(recipe))
    original_bytes = loaded.record.path.read_bytes()

    result = apply_cleaning(
        loaded, recipe, output_dir=tmp_path / "out", approved_hash=approved_hash
    )

    assert result.status == "applied"
    assert result.applied is not None
    applied = result.applied
    # New version, correct lineage back to v1.
    assert applied.result.target_version == 2
    assert applied.loaded.record.version == 2
    assert applied.loaded.record.parent_version == 1
    assert applied.loaded.record.lineage_recipe_id == "recipe_safe"
    assert applied.result.output_path.exists()
    # Original upload untouched.
    assert loaded.record.path.read_bytes() == original_bytes


def test_preview_is_read_only_and_exposes_pending_confirm(tmp_path: Path) -> None:
    loaded = _loaded(tmp_path)
    dispatch = preview_cleaning(loaded, _safe_recipe())
    assert dispatch.status == "awaiting_approval"
    assert dispatch.preview is not None
    assert dispatch.pending_action is not None
    assert dispatch.pending_action["tier"] == "confirm"
    # The hash the UI must echo back to approve.
    assert dispatch.pending_action["action_hash"] == action_hash(cleaning_action(_safe_recipe()))


def test_lossy_recipe_still_flows_through_the_confirm_gate(tmp_path: Path) -> None:
    loaded = _loaded(tmp_path)
    lossy = CleaningRecipe(
        dataset_id="ds_customers",
        source_version=1,
        recipe_id="recipe_lossy",
        transforms=[
            CleaningTransform(transform_id="t_drop", type="drop_column", target_column="age"),
        ],
    )
    # Unapproved lossy recipe is denied at the gate, not silently applied.
    denied = apply_cleaning(loaded, lossy, output_dir=tmp_path / "out", approved_hash=None)
    assert denied.status == "refused"

    # Approved lossy recipe applies and drops the column in the new version.
    approved_hash = action_hash(cleaning_action(lossy))
    applied = apply_cleaning(
        loaded, lossy, output_dir=tmp_path / "out2", approved_hash=approved_hash
    )
    assert applied.status == "applied"
    assert applied.applied is not None
    assert "age" not in applied.applied.loaded.frame.columns


@pytest.mark.parametrize("bad_hash", ["", "deadbeef", "not-the-hash"])
def test_wrong_hash_values_are_denied(tmp_path: Path, bad_hash: str) -> None:
    loaded = _loaded(tmp_path)
    result = apply_cleaning(
        loaded, _safe_recipe(), output_dir=tmp_path / "out", approved_hash=bad_hash
    )
    assert result.status == "refused"
