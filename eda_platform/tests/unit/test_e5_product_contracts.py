from __future__ import annotations

import pytest

from eda_platform.core.exploration_profiles import (
    EXPLORATION_READ_ONLY_TOOL_NAMES,
    build_read_only_exploration_toolset,
    exploration_budget_profile,
)
from eda_platform.core.exploration_tiers import (
    ANALYSIS_DEPTH_TO_EXPLORATION_TIER,
    exploration_tier_for_analysis_depth,
)


def test_analysis_depth_has_one_exploration_tier_mapping() -> None:
    assert dict(ANALYSIS_DEPTH_TO_EXPLORATION_TIER) == {
        0: "quick",
        1: "standard",
        2: "deep",
        3: "deep",
    }
    assert tuple(
        exploration_tier_for_analysis_depth(depth) for depth in range(4)
    ) == ("quick", "standard", "deep", "deep")


@pytest.mark.parametrize("analysis_depth", (-1, 4, 99))
def test_unknown_analysis_depth_does_not_silently_downgrade(
    analysis_depth: int,
) -> None:
    with pytest.raises(ValueError, match="analysis_depth"):
        exploration_tier_for_analysis_depth(analysis_depth)


def test_exploration_toolset_is_constructed_only_from_the_read_only_allowlist() -> None:
    registered = {
        "run_stat_test": "stat",
        "profile_slice": "slice",
        "apply_cleaning": "mutation",
        "write_artifact": "mutation",
        "delete_dataset": "mutation",
    }

    selected = build_read_only_exploration_toolset(registered)

    assert selected == ("stat", "slice")
    assert set(exploration_budget_profile("standard").max_tool_calls_by_kind) == set(
        EXPLORATION_READ_ONLY_TOOL_NAMES
    )
    assert {
        "apply_cleaning",
        "write_artifact",
        "delete_dataset",
    }.isdisjoint(EXPLORATION_READ_ONLY_TOOL_NAMES)


def test_every_certified_exploration_tool_can_produce_an_evidence_receipt() -> None:
    """The probe executor refuses a successful tool call without a receipt
    ("A journaled successful probe requires an EvidenceReceipt id."), so a
    receipt-less tool in the certified inventory kills any run that calls it.
    Executing each tool is the only check that cannot go stale."""
    import hashlib
    from pathlib import Path

    import pandas as pd

    from eda_platform.agents.data_tools import DataToolContext, build_data_tools
    from eda_platform.core.stat_registry import StatTestRegistry
    from eda_platform.schemas.artifacts import Artifact
    from eda_platform.schemas.datasets import DatasetRecord
    from eda_platform.tools.loader import LoadedDataset
    from eda_platform.tools.profiler import profile_dataset
    from eda_platform.tools.quality import scan_quality
    from eda_platform.tools.sql_runner import build_catalog

    def _profile_and_quality(loaded: LoadedDataset) -> list[Artifact]:
        profile = profile_dataset(
            loaded, project_id="proj_receipts", session_id="sess_receipts"
        )
        quality = scan_quality(
            profile, project_id="proj_receipts", session_id="sess_receipts"
        )
        return [profile, quality]

    frame = pd.DataFrame(
        {
            "order_date": pd.date_range("2025-01-01", periods=60).astype(str),
            "region": ["north"] * 30 + ["south"] * 30,
            "channel": ["online", "phone"] * 30,
            "units": [3 + index % 5 for index in range(60)],
            "revenue": [100.0 + index for index in range(30)]
            + [40.0 + index for index in range(30)],
            "satisfaction": [None if index % 3 else 4.0 for index in range(60)],
        }
    )
    dataset = LoadedDataset(
        record=DatasetRecord(
            dataset_id="ds_receipts",
            name="receipts.csv",
            path=Path("/data/receipts.csv"),
            content_hash=hashlib.sha256(b"receipt-contract").hexdigest(),
        ),
        frame=frame,
    )
    context = DataToolContext(
        datasets=[dataset],
        catalog=build_catalog([dataset]),
        project_id="proj_receipts",
        session_id="sess_receipts",
        store=None,
        payload_policy="schema+aggregates",
        # recommend_cleaning requires the profile and quality artifacts.
        artifacts=_profile_and_quality(dataset),
        stat_registry=StatTestRegistry(None),
    )
    registered = {tool.name: tool for tool in build_data_tools(context)}
    arguments: dict[str, dict[str, object]] = {
        "assess_join_keys": {
            "left_dataset_id": "ds_receipts",
            "right_dataset_id": "ds_receipts",
            "left_columns": ["region"],
            "right_columns": ["region"],
        },
        "screen_anomalies": {"dataset_id": "ds_receipts", "column": "revenue"},
        "run_domain_metrics": {},
        "recommend_cleaning": {"dataset_id": "ds_receipts"},
        "run_stat_test": {
            "dataset_id": "ds_receipts",
            "test_type": "independent_t_test",
            "group_column": "region",
            "value_column": "revenue",
        },
        "correlate_columns": {
            "dataset_id": "ds_receipts",
            "columns": ["revenue", "units"],
        },
        "profile_slice": {"dataset_id": "ds_receipts"},
        "analyze_time_series": {
            "dataset_id": "ds_receipts",
            "time_column": "order_date",
            "value_column": "revenue",
        },
        "diagnose_missingness": {
            "dataset_id": "ds_receipts",
            "target_column": "satisfaction",
            "group_columns": ["channel"],
        },
        "run_baseline_model": {
            "dataset_id": "ds_receipts",
            "target_column": "revenue",
        },
    }
    # run_open_analysis delegates to an injected executor that owns the receipt
    # contract, and never materializes without one, so it is not executable here.
    delegated = {"run_open_analysis"}
    covered = set(EXPLORATION_READ_ONLY_TOOL_NAMES) - delegated
    missing_fixture = covered - set(arguments)
    assert not missing_fixture, (
        "certified exploration tools without a receipt-contract fixture: "
        f"{sorted(missing_fixture)}"
    )
    for name in sorted(covered):
        tool = registered[name]
        result = tool.execute(tool.args_schema.model_validate(arguments[name]))
        assert result.receipt_artifact is not None, name
