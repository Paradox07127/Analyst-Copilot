"""diagnose_missingness runs real hypothesis tests, so they must land on the
multiplicity ledger: an attempt in the stat registry and a receipt whose
statistics carry the p-value, like run_stat_test and analyze_time_series."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from eda_platform.agents.data_tools import (
    DataToolContext,
    DiagnoseMissingnessArguments,
    build_data_tools,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.schemas.receipts import EvidenceReceipt
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.sql_runner import build_catalog


def _survey_frame() -> pd.DataFrame:
    channels = ["phone"] * 40 + ["web"] * 40
    satisfaction: list[float | None] = [None] * 30 + [float(i) for i in range(10)]
    satisfaction += [None] * 2 + [float(i) for i in range(38)]
    return pd.DataFrame(
        {
            "channel": channels,
            "satisfaction": satisfaction,
            "spend": [10.0 if value is None else 100.0 + value for value in satisfaction],
            "complete": list(range(80)),
        }
    )


def _context() -> DataToolContext:
    dataset = LoadedDataset(
        record=DatasetRecord(
            dataset_id="ds_survey",
            name="survey.csv",
            path=Path("/data/survey.csv"),
            content_hash="hash_ds_survey",
        ),
        frame=_survey_frame(),
    )
    return DataToolContext(
        datasets=[dataset],
        catalog=build_catalog([dataset]),
        project_id="project_t",
        session_id="run_t",
        store=None,
        payload_policy="schema+aggregates",
        artifacts=[],
    )


def _tool(context: DataToolContext) -> Any:
    return next(
        tool for tool in build_data_tools(context) if tool.name == "diagnose_missingness"
    )


def _last_receipt(context: DataToolContext) -> EvidenceReceipt:
    artifact: Artifact = [
        a for a in context.artifacts if a.type is ArtifactType.EVIDENCE_RECEIPT
    ][-1]
    return EvidenceReceipt.model_validate(artifact.payload)


def _diagnose(context: DataToolContext) -> None:
    _tool(context).execute(
        DiagnoseMissingnessArguments(
            dataset_id="ds_survey",
            target_column="spend",
            group_columns=["channel"],
        )
    )


def test_diagnose_missingness_registers_a_statistical_attempt() -> None:
    context = _context()
    _diagnose(context)

    receipt = _last_receipt(context)
    statistics = receipt.statistics
    assert statistics is not None
    assert statistics.p_value is not None
    assert statistics.adjusted_p_value is not None
    assert statistics.statistical_family_id is not None
    assert statistics.sequence_index == 1

    registry = context.stat_registry
    assert registry is not None
    attempts = list(registry.attempts())
    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt.family_id == statistics.statistical_family_id
    assert attempt.requested_test_type == "missingness_target_association"
    assert attempt.status == "completed"
    assert attempt.receipt_id == receipt.receipt_id
    # The E4a issuer recomputes the family from the invocation's arguments and
    # requires the declared requested_test_type to match the ledger.
    assert (
        receipt.method.parameters["requested_test_type"]
        == "missingness_target_association"
    )


def test_repeated_diagnosis_grows_the_family_count() -> None:
    context = _context()
    _diagnose(context)
    _diagnose(context)

    registry = context.stat_registry
    assert registry is not None
    attempts = list(registry.attempts())
    assert len(attempts) == 2
    assert {attempt.family_id for attempt in attempts} == {attempts[0].family_id}
    assert sorted(attempt.sequence_index for attempt in attempts) == [1, 2]
    receipt = _last_receipt(context)
    assert receipt.statistics is not None
    assert receipt.statistics.sequence_index == 2


def test_diagnosis_without_target_registers_no_attempt() -> None:
    context = _context()
    _tool(context).execute(DiagnoseMissingnessArguments(dataset_id="ds_survey"))

    receipt = _last_receipt(context)
    assert receipt.statistics is None
    registry = context.stat_registry
    assert registry is not None
    assert list(registry.attempts()) == []
