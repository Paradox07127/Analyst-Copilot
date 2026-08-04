"""Durable-result contracts for the tools typed adjudication made release-grade.

Every typed adjudication path is a contract obligation: once a tool's receipt
can sit on an insight's evidence side, that receipt becomes release evidence
and must be reconstructible from the separately persisted primary artifact.
correlate_columns / screen_anomalies / run_baseline_model gained adjudication
on 2026-08-03 and blocked issuance of the first clean real-provider deep run
(gpt-5.6-luna seed 8) the same day.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import BaseModel

from eda_platform.agents.data_tool_result_contracts import (
    verify_data_tool_result_contract,
)
from eda_platform.agents.data_tools import (
    CorrelateColumnsArguments,
    DataToolContext,
    RunBaselineModelArguments,
    ScreenAnomaliesArguments,
    build_data_tools,
)
from eda_platform.agents.runtime import AgentToolResult
from eda_platform.agents.tool_context import ToolExecutionContext, tool_execution_scope
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.schemas.receipts import EvidenceReceipt
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.sql_runner import build_catalog


def _run_tool(
    *, frame: pd.DataFrame, tool_name: str, arguments: BaseModel
) -> tuple[EvidenceReceipt, AgentToolResult, dict[str, Any]]:
    dataset = LoadedDataset(
        record=DatasetRecord(
            dataset_id="ds_contract",
            name="contract.csv",
            path=Path("/data/contract.csv"),
            content_hash="hash-contract",
        ),
        frame=frame,
    )
    context = DataToolContext(
        datasets=[dataset],
        catalog=build_catalog([dataset]),
        project_id="project_c",
        session_id=f"run_{tool_name}",
        store=None,
        payload_policy="schema+aggregates",
    )
    tool = next(item for item in build_data_tools(context) if item.name == tool_name)
    execution = ToolExecutionContext(
        run_id=f"run_{tool_name}",
        provider_call_id=f"provider_{tool_name}",
        logical_step_id=f"step_{tool_name}",
        sequence_index=1,
    )
    with tool_execution_scope(execution):
        result = tool.execute(arguments)
    assert result.receipt_artifact is not None
    receipt = EvidenceReceipt.model_validate(result.receipt_artifact.payload)
    return receipt, result, arguments.model_dump(mode="json")


def _correlation_frame() -> pd.DataFrame:
    values = [float(index) for index in range(60)]
    return pd.DataFrame(
        {
            "x": values,
            "y": [2.0 * value + (0.4 if index % 2 else -0.4) for index, value in enumerate(values)],
            "z": [float((index * 7) % 13) for index in range(60)],
        }
    )


def test_correlate_columns_receipt_is_rebuilt_from_its_durable_artifact() -> None:
    receipt, result, arguments = _run_tool(
        frame=_correlation_frame(),
        tool_name="correlate_columns",
        arguments=CorrelateColumnsArguments(
            dataset_id="ds_contract", columns=["x", "y", "z"]
        ),
    )

    verify_data_tool_result_contract(receipt, result, arguments)


def test_a_forged_correlation_fact_is_rejected() -> None:
    """The point of the contract: a receipt may not vouch for itself."""
    receipt, result, arguments = _run_tool(
        frame=_correlation_frame(),
        tool_name="correlate_columns",
        arguments=CorrelateColumnsArguments(
            dataset_id="ds_contract", columns=["x", "y", "z"]
        ),
    )
    forged_facts = tuple(
        fact.model_copy(update={"value": 0.999999})
        if fact.fact_id.endswith(".coefficient")
        else fact
        for fact in receipt.facts
    )
    assert forged_facts != receipt.facts
    forged = receipt.model_copy(update={"facts": forged_facts})

    with pytest.raises(ValueError):
        verify_data_tool_result_contract(forged, result, arguments)


def _anomaly_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"value": [10.0 + (index % 5) * 0.1 for index in range(40)] + [500.0]}
    )


def test_screen_anomalies_receipt_is_rebuilt_from_its_durable_artifact() -> None:
    receipt, result, arguments = _run_tool(
        frame=_anomaly_frame(),
        tool_name="screen_anomalies",
        arguments=ScreenAnomaliesArguments(dataset_id="ds_contract", column="value"),
    )

    verify_data_tool_result_contract(receipt, result, arguments)


def test_a_forged_outlier_count_is_rejected() -> None:
    receipt, result, arguments = _run_tool(
        frame=_anomaly_frame(),
        tool_name="screen_anomalies",
        arguments=ScreenAnomaliesArguments(dataset_id="ds_contract", column="value"),
    )
    forged = receipt.model_copy(
        update={
            "facts": tuple(
                fact.model_copy(update={"value": 999})
                if fact.fact_id == "outlier_count"
                else fact
                for fact in receipt.facts
            )
        }
    )

    with pytest.raises(ValueError):
        verify_data_tool_result_contract(forged, result, arguments)


def _baseline_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [float(index % 17) for index in range(80)],
            "feature_b": [float((index * 3) % 11) for index in range(80)],
            "target": [float(index % 17) * 2.0 for index in range(80)],
        }
    )


def test_run_baseline_model_receipt_is_rebuilt_from_its_durable_artifact() -> None:
    receipt, result, arguments = _run_tool(
        frame=_baseline_frame(),
        tool_name="run_baseline_model",
        arguments=RunBaselineModelArguments(
            dataset_id="ds_contract", target_column="target"
        ),
    )

    verify_data_tool_result_contract(receipt, result, arguments)


def test_a_forged_model_metric_is_rejected() -> None:
    receipt, result, arguments = _run_tool(
        frame=_baseline_frame(),
        tool_name="run_baseline_model",
        arguments=RunBaselineModelArguments(
            dataset_id="ds_contract", target_column="target"
        ),
    )
    forged = receipt.model_copy(
        update={
            "facts": tuple(
                fact.model_copy(update={"value": 0.999999})
                if fact.fact_id.startswith("metric.")
                else fact
                for fact in receipt.facts
            )
        }
    )
    assert forged.facts != receipt.facts

    with pytest.raises(ValueError):
        verify_data_tool_result_contract(forged, result, arguments)
