"""Phase 2: type-filtered artifact loads avoid full-session JSON scans."""

from __future__ import annotations

from pathlib import Path

import pytest

from eda_platform.core.decision_coverage import assess_decision_coverage
from eda_platform.core.finding_freshness import project_run_artifacts
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.investigation_library import build_investigation_library
from eda_platform.schemas.artifacts import Artifact, ArtifactType, EvidenceRef
from eda_platform.schemas.investigations import InvestigationRecord, ValidatedFinding
from eda_platform.schemas.questions import QuestionFinding

PROJECT = "project_type_filter"
SESSION = "sess_main"


def _finding(session_id: str, inv: str) -> Artifact:
    finding = ValidatedFinding(
        finding_id=f"finding_{inv}",
        investigation_id=inv,
        question_id=f"q_{inv}",
        question=f"Question {inv}?",
        claim_class="observed",
        findings=[
            QuestionFinding(
                text=f"Answer {inv}",
                evidence=[
                    EvidenceRef(
                        kind="table",
                        artifact_id="evidence_1",
                        locator="rows[0]",
                        value=1,
                    )
                ],
            )
        ],
        evidence_support="high",
        analytical_reliability="high",
        decision_readiness="medium",
        limitations=[],
        report_eligible=True,
        report_readiness="eligible_with_limitations",
        report_readiness_reason="Supported claim.",
        source_artifact_ids=["evidence_1"],
    )
    return Artifact(
        id=f"finding_{inv}",
        type=ArtifactType.VALIDATED_FINDING,
        project_id=PROJECT,
        session_id=session_id,
        payload=finding.model_dump(mode="json"),
    )


def _record(session_id: str, inv: str) -> Artifact:
    record = InvestigationRecord(
        record_id=f"record_{inv}",
        investigation_id=inv,
        question_id=f"q_{inv}",
        status="validated",
        reason_code="test_reason_code",
        reason="Terminal outcome for the test.",
        next_action="Review the evidence.",
        finding_artifact_id=f"finding_{inv}",
    )
    return Artifact(
        id=f"record_{inv}",
        type=ArtifactType.INVESTIGATION_RECORD,
        project_id=PROJECT,
        session_id=session_id,
        payload=record.model_dump(mode="json"),
    )


def _noise_chart(session_id: str, index: int) -> Artifact:
    return Artifact(
        id=f"chart_noise_{index}",
        type=ArtifactType.CHART_SPEC,
        project_id=PROJECT,
        session_id=session_id,
        payload={
            "dataset_id": "ds",
            "title": f"noise {index}",
            "mark": "bar",
            "encoding": {},
            "data": {"values": [{"x": 1, "y": 2}]},
        },
    )


def test_list_artifacts_of_types_skips_unrelated_payload_files(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, "Type filter")
    store.start_session(PROJECT, SESSION)
    store.save_artifact(_finding(SESSION, "inv_1"))
    store.save_artifact(_record(SESSION, "inv_1"))
    for index in range(50):
        store.save_artifact(_noise_chart(SESSION, index))

    read_paths: list[str] = []
    original_read = Path.read_text

    def tracking_read(self: Path, *args: object, **kwargs: object) -> str:
        if self.suffix == ".json" and "artifacts" in self.parts:
            read_paths.append(self.name)
        return original_read(self, *args, **kwargs)

    Path.read_text = tracking_read  # type: ignore[method-assign]
    try:
        artifacts, warnings = store.list_artifacts_of_types(
            project_id=PROJECT,
            session_id=SESSION,
            artifact_types=(
                ArtifactType.VALIDATED_FINDING,
                ArtifactType.INVESTIGATION_RECORD,
            ),
        )
    finally:
        Path.read_text = original_read  # type: ignore[method-assign]

    assert warnings == []
    assert {item.type for item in artifacts} == {
        ArtifactType.VALIDATED_FINDING,
        ArtifactType.INVESTIGATION_RECORD,
    }
    assert len(artifacts) == 2
    # Indexed type filter must not open the 50 chart JSON files.
    assert not any(name.startswith("chart_noise_") for name in read_paths)
    assert any(name.startswith("finding_") for name in read_paths)
    assert any(name.startswith("record_") for name in read_paths)


def test_library_and_coverage_use_type_filter_not_full_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, "Type filter")
    store.start_session(PROJECT, SESSION)
    store.save_artifact(_finding(SESSION, "inv_1"))
    store.save_artifact(_record(SESSION, "inv_1"))
    for index in range(20):
        store.save_artifact(_noise_chart(SESSION, index))

    safe_calls = {"count": 0}
    original_safe = ArtifactStore.list_artifacts_safe

    def counting_safe(self: ArtifactStore, **kwargs: object) -> tuple[list, list]:
        safe_calls["count"] += 1
        return original_safe(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ArtifactStore, "list_artifacts_safe", counting_safe)

    library = build_investigation_library(store, project_id=PROJECT)
    assert len(library.findings) == 1
    assert safe_calls["count"] == 0

    coverage = assess_decision_coverage(store, PROJECT)
    assert coverage.validated_findings == 1
    assert safe_calls["count"] == 0

    snapshot = project_run_artifacts(store, PROJECT)
    assert len(snapshot) == 1
    assert snapshot[0][0] == SESSION
    # Freshness snapshot only loads plan/profile types — noise charts stay out.
    assert all(
        artifact.type
        in {ArtifactType.INVESTIGATION_PLAN, ArtifactType.DATASET_PROFILE}
        for _session_id, artifacts in snapshot
        for artifact in artifacts
    )
    assert safe_calls["count"] == 0
