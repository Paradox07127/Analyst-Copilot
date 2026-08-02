"""E1 closure: missingness and model baselines are agent-facing receipt tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from eda_platform.agents.data_tools import (
    DataToolContext,
    DiagnoseMissingnessArguments,
    RunBaselineModelArguments,
    build_data_tools,
)
from eda_platform.core.claim_gates import run_claim_gates
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.claims import Claim, ClaimBundle
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.schemas.model_card import ModelCard
from eda_platform.schemas.receipts import EvidenceReceipt, verify_receipt_digest
from eda_platform.tools.evidence import PayloadPolicy
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.sql_runner import build_catalog


def _dataset(frame: pd.DataFrame, dataset_id: str = "ds") -> LoadedDataset:
    return LoadedDataset(
        record=DatasetRecord(
            dataset_id=dataset_id,
            name="data.csv",
            path=Path("/data/data.csv"),
            content_hash="hash_data",
        ),
        frame=frame,
    )


def _context(
    frame: pd.DataFrame,
    *,
    payload_policy: PayloadPolicy = "schema+aggregates",
) -> DataToolContext:
    dataset = _dataset(frame)
    return DataToolContext(
        datasets=[dataset],
        catalog=build_catalog([dataset]),
        project_id="project",
        session_id="session",
        store=None,
        payload_policy=payload_policy,
    )


def _tool(context: DataToolContext, name: str) -> Any:
    return next(item for item in build_data_tools(context) if item.name == name)


def _receipt(result: Any) -> EvidenceReceipt:
    artifact = result.receipt_artifact
    assert isinstance(artifact, Artifact)
    receipt = EvidenceReceipt.model_validate(artifact.payload)
    assert verify_receipt_digest(receipt)
    return receipt


def _missingness_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "channel": ["phone"] * 30 + ["web"] * 30,
            "score": [None] * 24
            + [float(index) for index in range(6)]
            + [None] * 3
            + [float(index) for index in range(27)],
            "outcome": [0.0] * 27 + [1.0] * 33,
        }
    )


def _model_frame() -> pd.DataFrame:
    rng = np.random.default_rng(12)
    signal = rng.normal(size=160)
    return pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(size=160),
            "label": (signal + rng.normal(scale=0.35, size=160) > 0).astype(int),
        }
    )


def test_e1_has_sixteen_fixed_agent_tools() -> None:
    names = [tool.name for tool in build_data_tools(_context(_model_frame()))]
    assert len(names) == len(set(names)) == 16
    assert {"diagnose_missingness", "run_baseline_model"} <= set(names)
    assert "run_open_analysis" not in names


def test_diagnose_missingness_agent_tool_emits_table_and_verified_receipt() -> None:
    context = _context(_missingness_frame())
    result = _tool(context, "diagnose_missingness").execute(
        DiagnoseMissingnessArguments(
            dataset_id="ds",
            target_column="outcome",
            group_columns=["channel"],
        )
    )

    receipt = _receipt(result)
    primary = result.artifacts[0]
    assert primary.type is ArtifactType.TABLE
    assert primary.payload["kind"] == "missingness_diagnostic"
    assert primary.payload["mnar_ruled_out"] is False
    assert receipt.tool_name == "diagnose_missingness"
    assert receipt.method.family == "missingness_diagnostic"
    assert receipt.scope.scope_resolution == "resolved"
    assert receipt.result_count == len(_missingness_frame().columns)
    facts = {fact.fact_id: fact.value for fact in receipt.facts}
    assert facts["mnar_ruled_out"] is False
    assert facts["columns_with_missing"] == 1
    assert result.receipt_artifact is not None
    assert isinstance(result.content, dict)
    assert result.content["receipt_id"] == receipt.receipt_id


def test_schema_only_hides_missingness_values_and_group_labels() -> None:
    context = _context(_missingness_frame(), payload_policy="schema_only")
    result = _tool(context, "diagnose_missingness").execute(
        DiagnoseMissingnessArguments(dataset_id="ds", group_columns=["channel"])
    )

    encoded = str(result.content)
    assert "phone" not in encoded
    assert "web" not in encoded
    assert "missing_percent" not in encoded
    assert "schema_only" in encoded


def test_run_baseline_model_agent_tool_emits_model_card_and_metric_facts() -> None:
    context = _context(_model_frame())
    result = _tool(context, "run_baseline_model").execute(
        RunBaselineModelArguments(
            dataset_id="ds",
            target_column="label",
            cv_folds=3,
        )
    )

    receipt = _receipt(result)
    primary = result.artifacts[0]
    assert primary.type is ArtifactType.MODEL_CARD
    card = ModelCard.model_validate(primary.payload)
    assert receipt.tool_name == "run_baseline_model"
    assert receipt.method.family == "ml_baseline"
    assert receipt.result_count == 1
    assert receipt.scope.scope_resolution == "resolved"
    assert card.metrics["cv_folds"] == 3.0
    facts = {fact.fact_id: fact.value for fact in receipt.facts}
    assert facts["metric.accuracy"] == card.metrics["accuracy"]
    assert facts["metric.cv_folds"] == 3.0
    assert any(name.endswith(".signed_importance") for name in facts)
    assert any(name.endswith(".importance_std") for name in facts)
    assert isinstance(result.content, dict)
    assert result.content["receipt_id"] == receipt.receipt_id


def test_real_baseline_receipt_licenses_the_matching_model_metric_claim() -> None:
    context = _context(_model_frame())
    result = _tool(context, "run_baseline_model").execute(
        RunBaselineModelArguments(
            dataset_id="ds",
            target_column="label",
            cv_folds=3,
        )
    )
    receipt = _receipt(result)
    accuracy = next(
        fact.value for fact in receipt.facts if fact.fact_id == "metric.accuracy"
    )
    bundle = ClaimBundle(
        claim_bundle_id="clb_model",
        hypothesis_id="hyp_model",
        evidence_lane="exploratory",
        claims=(
            Claim(
                claim_id="claim_model",
                claim_type="model",
                claim_text=f"Baseline model accuracy was {accuracy}.",
                support_type="direct",
                evidence_fact_ids=(f"{receipt.receipt_id}:metric.accuracy",),
            ),
        ),
    )

    report = run_claim_gates(
        bundle,
        committed_receipts={receipt.receipt_id: receipt},
        run_witness=receipt.data_state_witness,
    )
    assert report.passed, report.verdicts


def test_baseline_agent_contract_rejects_unsafe_split_combinations() -> None:
    with pytest.raises(ValidationError, match="group_column"):
        RunBaselineModelArguments(
            dataset_id="ds",
            target_column="label",
            split_policy="group",
        )
    with pytest.raises(ValidationError, match="only with"):
        RunBaselineModelArguments(
            dataset_id="ds",
            target_column="label",
            split_policy="auto",
            group_column="site",
        )

    temporal = _model_frame().assign(
        observed_at=pd.date_range("2025-01-01", periods=160, freq="D")
    )
    context = _context(temporal)
    with pytest.raises(ValueError, match="random.*temporal|temporal.*random"):
        _tool(context, "run_baseline_model").execute(
            RunBaselineModelArguments(
                dataset_id="ds",
                target_column="label",
                time_column="observed_at",
                split_policy="random",
            )
        )
