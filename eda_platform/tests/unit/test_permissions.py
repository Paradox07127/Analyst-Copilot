from __future__ import annotations

from eda_platform.core.permissions import (
    PermissionTier,
    action_hash,
    classify_action,
    require_permission,
)


def test_duckdb_select_is_auto_but_disguised_file_read_is_denied() -> None:
    safe = classify_action(
        {
            "type": "duckdb_select",
            "sql": "select region, sum(amount) from orders group by region",
        }
    )
    denied = classify_action(
        {
            "type": "duckdb_select",
            "sql": "select * from read_csv('/etc/passwd')",
        }
    )

    assert safe.tier is PermissionTier.AUTO
    assert denied.tier is PermissionTier.DENY
    assert "Tool guard rejected" in denied.feedback
    assert "read-only SELECT" in denied.feedback


def test_cleaning_apply_requires_bound_approval_hash() -> None:
    action = {
        "type": "cleaning_apply",
        "dataset_id": "ds_orders",
        "recipe_id": "recipe_drop_rows",
        "transform_ids": ["drop_missing_region"],
        "reversible": False,
    }

    decision = classify_action(action)

    assert decision.tier is PermissionTier.CONFIRM
    assert decision.action_hash == action_hash(action)
    assert "ds_orders" in decision.description
    assert not decision.reversible


def test_approval_hash_mismatch_blocks_action_swap() -> None:
    approved_action = {
        "type": "cleaning_apply",
        "dataset_id": "ds_orders",
        "recipe_id": "recipe_a",
        "transform_ids": ["trim_region"],
        "reversible": True,
    }
    swapped_action = {
        "type": "cleaning_apply",
        "dataset_id": "ds_orders",
        "recipe_id": "recipe_b",
        "transform_ids": ["drop_all_rows"],
        "reversible": False,
    }

    decision = require_permission(swapped_action, approved_hash=action_hash(approved_action))

    assert decision.tier is PermissionTier.DENY
    assert "approval hash" in decision.feedback.lower()
    assert decision.action_hash == action_hash(swapped_action)


def test_sandboxed_code_is_auto_but_bypass_requests_are_denied() -> None:
    safe = classify_action(
        {
            "type": "sandboxed_code",
            "code": "import json\nprint(json.dumps({'summary': 'ok', 'result_files': []}))",
            "sandboxed": True,
        }
    )
    bypass = classify_action(
        {
            "type": "sandboxed_code",
            "code": "print('x')",
            "sandboxed": False,
            "bypass_sandbox": True,
        }
    )

    assert safe.tier is PermissionTier.AUTO
    assert bypass.tier is PermissionTier.DENY
    assert "sandbox execution path" in bypass.feedback
