"""run_segmentation agent tool: clustering, k selection, ARI stability, receipt contract.

The two-cluster geometry is hand-checkable (exact means/medians/shares) and the
silhouette is recomputed by an independent pairwise-distance loop, so the
tool's numbers are verified, not trusted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from eda_platform.agents.data_tools import (
    DataToolContext,
    RunSegmentationArguments,
    build_data_tools,
)
from eda_platform.core.claim_language import (
    asserts_model_capability,
    implies_causation,
)
from eda_platform.core.ids import make_artifact_id
from eda_platform.core.methods import METHOD_REGISTRY
from eda_platform.core.tool_guard import ToolGuardError
from eda_platform.drivers.question_exec import _method_contract_failure
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.schemas.questions import QuestionCandidate, QuestionScore
from eda_platform.schemas.receipts import EvidenceReceipt, verify_receipt_digest
from eda_platform.schemas.segmentation import SegmentationResult
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.segmentation import (
    SEGMENTATION_LIMITATION,
    SEGMENTATION_UNSTABLE_LIMITATION,
)
from eda_platform.tools.sql_runner import build_catalog


def _dataset(name: str, frame: pd.DataFrame, dataset_id: str) -> LoadedDataset:
    return LoadedDataset(
        record=DatasetRecord(
            dataset_id=dataset_id,
            name=name,
            path=Path(f"/data/{name}"),
            content_hash="hash_" + dataset_id,
        ),
        frame=frame,
    )


def _context(frame: pd.DataFrame) -> DataToolContext:
    datasets = [_dataset("segments.csv", frame, "ds_segments")]
    return DataToolContext(
        datasets=datasets,
        catalog=build_catalog(datasets),
        project_id="project_t8b",
        session_id="run_t8b",
        store=None,
        payload_policy="schema+aggregates",
        artifacts=[],
    )


def _tool(context: DataToolContext) -> Any:
    return next(tool for tool in build_data_tools(context) if tool.name == "run_segmentation")


def _receipt(context: DataToolContext) -> EvidenceReceipt:
    artifact = [
        a for a in context.artifacts if a.type is ArtifactType.EVIDENCE_RECEIPT
    ][-1]
    return EvidenceReceipt.model_validate(artifact.payload)


def _fact(receipt: EvidenceReceipt, fact_id: str) -> Any:
    return next(fact for fact in receipt.facts if fact.fact_id == fact_id).value


def _result(context: DataToolContext) -> SegmentationResult:
    artifact = next(
        a for a in context.artifacts if a.type is ArtifactType.SEGMENTATION_RESULT
    )
    return SegmentationResult.model_validate(artifact.payload)


def _two_cluster_frame() -> pd.DataFrame:
    """Two tight, obvious clusters plus two incomplete rows to exercise dropna."""
    frame = pd.DataFrame(
        {
            "x": [0.0] * 20 + [10.0] * 20,
            "y": [1.0, 2.0] * 10 + [11.0, 12.0] * 10,
        }
    )
    return pd.concat(
        [frame, pd.DataFrame({"x": [0.0, 10.0], "y": [np.nan, np.nan]})],
        ignore_index=True,
    )


def _three_blob_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    blobs = np.vstack(
        [
            rng.normal(loc=(0, 0), scale=0.3, size=(40, 2)),
            rng.normal(loc=(10, 0), scale=0.3, size=(40, 2)),
            rng.normal(loc=(0, 10), scale=0.3, size=(40, 2)),
        ]
    )
    return pd.DataFrame(blobs, columns=["x", "y"])


def _noise_frame() -> pd.DataFrame:
    """6-d uniform noise: any 6-way partition of it is arbitrary, so refits on
    subsamples disagree and the mean ARI lands well below the 0.6 line."""
    rng = np.random.default_rng(1)
    return pd.DataFrame(rng.uniform(size=(60, 6)), columns=list("abcdef"))


def _hand_silhouette(points: np.ndarray, labels: np.ndarray) -> float:
    """Independent O(n^2) silhouette: s = (b - a) / max(a, b), averaged."""
    distances = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for index in range(len(points)):
        own = labels == labels[index]
        a = distances[index, own & (np.arange(len(points)) != index)].mean()
        b = min(
            distances[index, labels == other].mean()
            for other in set(labels)
            if other != labels[index]
        )
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores))


def test_two_obvious_clusters_are_recovered_and_hand_recomputable() -> None:
    context = _context(_two_cluster_frame())
    result = _tool(context).execute(
        RunSegmentationArguments(
            dataset_id="ds_segments", feature_columns=["x", "y"], k=2
        )
    )

    content = cast(dict[str, Any], result.content)
    segmentation = _result(context)
    assert segmentation.k == 2 and segmentation.k_selection == "explicit"
    assert segmentation.total_rows == 42 and segmentation.sample_rows == 40
    low, high = sorted(
        segmentation.clusters, key=lambda cluster: cluster.feature_summary[0].mean
    )
    assert low.size == 20 and high.size == 20
    assert low.share == 0.5 and high.share == 0.5
    by_feature_low = {item.feature: item for item in low.feature_summary}
    by_feature_high = {item.feature: item for item in high.feature_summary}
    # Hand-checkable geometry: x is 0 vs 10, y alternates 1/2 vs 11/12.
    assert by_feature_low["x"].mean == 0.0 and by_feature_high["x"].mean == 10.0
    assert by_feature_low["y"].mean == 1.5 and by_feature_high["y"].mean == 11.5
    assert by_feature_low["y"].median == 1.5 and by_feature_high["y"].median == 11.5
    # Overall means are x=5, y=6.5, so the deltas are symmetric.
    assert by_feature_low["x"].delta_vs_overall_mean == -5.0
    assert by_feature_high["x"].delta_vs_overall_mean == 5.0
    assert by_feature_low["y"].delta_vs_overall_mean == -5.0
    assert by_feature_high["y"].delta_vs_overall_mean == 5.0

    # Independent silhouette recomputation on the standardized ground truth.
    values = _two_cluster_frame().dropna().to_numpy(dtype="float64")
    standardized = (values - values.mean(axis=0)) / values.std(axis=0)
    truth = np.array([0] * 20 + [1] * 20)
    assert segmentation.silhouette == pytest.approx(
        _hand_silhouette(standardized, truth), abs=1e-6
    )

    # Clusters this separated survive every subsample refit.
    assert segmentation.ari_mean == 1.0 and segmentation.ari_min == 1.0
    assert segmentation.stability == "stable"
    assert content["stability"] == "stable"
    assert content["limitation"] == SEGMENTATION_LIMITATION


def test_silhouette_selects_three_for_three_blobs() -> None:
    context = _context(_three_blob_frame())
    _tool(context).execute(
        RunSegmentationArguments(dataset_id="ds_segments", feature_columns=["x", "y"])
    )

    segmentation = _result(context)
    assert segmentation.k == 3 and segmentation.k_selection == "silhouette"
    assert sorted(cluster.size for cluster in segmentation.clusters) == [40, 40, 40]
    assert segmentation.silhouette > 0.9
    assert segmentation.stability == "stable"
    assert segmentation.ari_mean == 1.0


def test_receipt_carries_traceable_segment_facts_and_no_p_value() -> None:
    context = _context(_two_cluster_frame())
    _tool(context).execute(
        RunSegmentationArguments(
            dataset_id="ds_segments", feature_columns=["x", "y"], k=2
        )
    )

    receipt = _receipt(context)
    assert verify_receipt_digest(receipt)
    assert receipt.tool_name == "run_segmentation"
    assert _fact(receipt, "k") == 2
    assert _fact(receipt, "stability") == "stable"
    assert _fact(receipt, "sample_rows") == 40
    assert _fact(receipt, "ari_mean") == 1.0
    assert {_fact(receipt, "cluster0.size"), _fact(receipt, "cluster1.size")} == {20}
    assert _fact(receipt, "cluster0.share") == 0.5
    # Silhouette/ARI are not hypothesis tests: no p-value on the ledger.
    assert receipt.statistics is None or receipt.statistics.p_value is None
    assert receipt.method.family == "segmentation_kmeans"
    assert receipt.method.parameters["standardization"] == "zscore"
    assert receipt.method.parameters["resamples"] == 20
    assert SEGMENTATION_LIMITATION in receipt.method.assumptions
    # A stable partition keeps its evidence standing.
    assert "hypothesis_evidence_valid" not in receipt.method.parameters
    assert not receipt.method.hypothesis_evidence_is_explicitly_invalid()
    manifest = receipt.fact_manifest
    assert manifest is not None and manifest.total_rows == 2
    assert [entry.fact_id for entry in manifest.entries] == ["cluster0", "cluster1"]


def test_unstable_noise_is_flagged_and_stripped_of_evidence_standing() -> None:
    context = _context(_noise_frame())
    result = _tool(context).execute(
        RunSegmentationArguments(
            dataset_id="ds_segments", feature_columns=list("abcdef"), k=6
        )
    )

    segmentation = _result(context)
    assert segmentation.stability == "unstable"
    assert segmentation.ari_mean < 0.6
    assert SEGMENTATION_UNSTABLE_LIMITATION in segmentation.notes

    receipt = _receipt(context)
    assert receipt.method.parameters["hypothesis_evidence_valid"] is False
    assert receipt.method.hypothesis_evidence_is_explicitly_invalid()
    assert SEGMENTATION_UNSTABLE_LIMITATION in receipt.method.warnings

    primary = next(
        a for a in context.artifacts if a.type is ArtifactType.SEGMENTATION_RESULT
    )
    assert SEGMENTATION_UNSTABLE_LIMITATION in primary.warnings
    assert primary.plain_language is not None
    assert "must not be used as evidence" in primary.plain_language
    content = cast(dict[str, Any], result.content)
    assert SEGMENTATION_UNSTABLE_LIMITATION in content["notes"]


def test_guard_rejects_bad_features_and_short_data() -> None:
    frame = pd.DataFrame(
        {
            "amount": [float(value) for value in range(40)],
            "quantity": [float(value % 7) for value in range(40)],
            "region": ["north", "south"] * 20,
            "order_id": [float(value) for value in range(40)],
            "flat": [1.0] * 40,
        }
    )
    context = _context(frame)
    tool = _tool(context)

    def run(features: list[str]) -> None:
        tool.execute(
            RunSegmentationArguments(dataset_id="ds_segments", feature_columns=features)
        )

    with pytest.raises(ToolGuardError, match="semantic type"):
        run(["amount", "region"])
    with pytest.raises(ToolGuardError, match="identifier"):
        run(["amount", "order_id"])
    with pytest.raises(ToolGuardError, match="constant"):
        run(["amount", "flat"])

    short = _context(pd.DataFrame({"a": [1.0, 2.0] * 10, "b": [3.0, 4.0] * 10}))
    with pytest.raises(ToolGuardError, match="at least 30 rows"):
        _tool(short).execute(
            RunSegmentationArguments(dataset_id="ds_segments", feature_columns=["a", "b"])
        )
    assert not [
        a for a in context.artifacts if a.type is ArtifactType.SEGMENTATION_RESULT
    ]


def test_rejects_bad_arguments() -> None:
    with pytest.raises(ValidationError):
        RunSegmentationArguments(dataset_id="ds", feature_columns=["only_one"])
    with pytest.raises(ValidationError):
        RunSegmentationArguments(dataset_id="ds", feature_columns=["x", "x"])
    with pytest.raises(ValidationError):
        RunSegmentationArguments(dataset_id="ds", feature_columns=["x", "y"], k=1)
    with pytest.raises(ValidationError):
        RunSegmentationArguments(dataset_id="ds", feature_columns=["x", "y"], k=9)
    with pytest.raises(ValidationError):
        RunSegmentationArguments(
            dataset_id="ds",
            feature_columns=["x", "y"],
            extra_field="nope",  # type: ignore[call-arg]
        )


def _segmentation_candidate() -> QuestionCandidate:
    return QuestionCandidate(
        question_id="q_segmentation",
        question_en="Which spending segments exist among these customers?",
        origin="llm",
        analysis_mode="segmentation",
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.1,
            join_risk=0.0,
            deterministic_score=0.8,
        ),
    )


def test_segmentation_answer_contract_is_satisfied_by_the_tool_output() -> None:
    context = _context(_two_cluster_frame())
    result = _tool(context).execute(
        RunSegmentationArguments(
            dataset_id="ds_segments", feature_columns=["x", "y"], k=2
        )
    )

    failure = _method_contract_failure(
        _segmentation_candidate(),
        evidence_artifacts=cast(list[Artifact], result.artifacts),
        tool_names=["run_segmentation"],
    )
    assert failure is None


def test_code_execution_result_no_longer_answers_a_segmentation_question() -> None:
    """The placeholder CodeExecutionResult mapping is retired: only the typed
    SegmentationResult artifact satisfies the segmentation answer contract."""
    payload = {
        "status": "succeeded",
        "stdout_json": {"summary": "clustered with ad-hoc code"},
        "policy_digest": "digest",
        "execution_manifest_sha256": "sha",
    }
    spoof = Artifact(
        id=make_artifact_id("code", payload),
        type=ArtifactType.CODE_EXECUTION_RESULT,
        project_id="project_t8b",
        session_id="run_t8b",
        payload=payload,
    )

    failure = _method_contract_failure(
        _segmentation_candidate(),
        evidence_artifacts=[spoof],
        tool_names=["run_segmentation"],
    )
    assert failure is not None
    assert failure.code == "method_contract_failed"
    assert "SegmentationResult" in failure.reason


def test_language_stays_descriptive_and_family_is_supported() -> None:
    context = _context(_two_cluster_frame())
    _tool(context).execute(
        RunSegmentationArguments(
            dataset_id="ds_segments", feature_columns=["x", "y"], k=2
        )
    )

    primary = next(
        a for a in context.artifacts if a.type is ArtifactType.SEGMENTATION_RESULT
    )
    tool_description = _tool(context).description
    for text in (
        SEGMENTATION_LIMITATION,
        SEGMENTATION_UNSTABLE_LIMITATION,
        cast(str, primary.plain_language),
        tool_description,
    ):
        assert not implies_causation(text)
        assert not asserts_model_capability(text)

    # The exact gate-ok wording is pinned in test_di_method_registry.
    assert METHOD_REGISTRY["segmentation"].supported is True
