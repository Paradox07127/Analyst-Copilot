"""Project-scoped read model for validated investigation outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.investigations import (
    InvestigationPlan,
    InvestigationRecord,
    ValidatedFinding,
)

# Project findings UI only needs these types; skip charts/profiles/etc.
_LIBRARY_ARTIFACT_TYPES = (
    ArtifactType.VALIDATED_FINDING,
    ArtifactType.INVESTIGATION_RECORD,
    ArtifactType.INVESTIGATION_PLAN,
)


@dataclass(frozen=True)
class StoredValidatedFinding:
    """One validated finding together with immutable artifact provenance."""

    artifact_id: str
    session_id: str
    created_at: datetime
    finding: ValidatedFinding


@dataclass(frozen=True)
class StoredInvestigationRecord:
    """One investigation outcome with the question resolved when available."""

    artifact_id: str
    session_id: str
    created_at: datetime
    question: str
    record: InvestigationRecord


@dataclass(frozen=True)
class InvestigationLibrary:
    """Read-only project memory for synthesis and user review."""

    findings: list[StoredValidatedFinding]
    records: list[StoredInvestigationRecord]
    warnings: list[str]


def load_investigation_library(
    *,
    workspace: str,
    project_id: str,
) -> InvestigationLibrary:
    """Load typed findings and records from every project run defensively."""
    return build_investigation_library(ArtifactStore(workspace), project_id=project_id)


def build_investigation_library(
    store: ArtifactStore,
    *,
    project_id: str,
) -> InvestigationLibrary:
    """Build the project-level findings read model from persisted artifacts."""
    candidate_findings: list[StoredValidatedFinding] = []
    records: list[StoredInvestigationRecord] = []
    warnings: list[str] = []
    for run in store.list_sessions(project_id):
        artifacts, artifact_warnings = store.list_artifacts_of_types(
            project_id=project_id,
            session_id=run.session_id,
            artifact_types=_LIBRARY_ARTIFACT_TYPES,
        )
        warnings.extend(f"{run.session_id}: {warning}" for warning in artifact_warnings)
        questions_by_id = _questions_by_id(artifacts, session_id=run.session_id, warnings=warnings)
        for artifact in artifacts:
            if artifact.type is ArtifactType.VALIDATED_FINDING:
                finding = _parse_finding(artifact, session_id=run.session_id, warnings=warnings)
                if finding is not None:
                    candidate_findings.append(
                        StoredValidatedFinding(
                            artifact_id=artifact.id,
                            session_id=run.session_id,
                            created_at=artifact.created_at,
                            finding=finding,
                        )
                    )
            elif artifact.type is ArtifactType.INVESTIGATION_RECORD:
                record = _parse_record(artifact, session_id=run.session_id, warnings=warnings)
                if record is not None:
                    records.append(
                        StoredInvestigationRecord(
                            artifact_id=artifact.id,
                            session_id=run.session_id,
                            created_at=artifact.created_at,
                            question=questions_by_id.get(record.question_id, record.question_id),
                            record=record,
                        )
                    )
    findings = _validated_findings_only(
        candidate_findings,
        records=records,
        warnings=warnings,
    )
    findings.sort(key=lambda item: (item.created_at, item.artifact_id), reverse=True)
    records.sort(key=lambda item: (item.created_at, item.artifact_id), reverse=True)
    return InvestigationLibrary(findings=findings, records=records, warnings=warnings)


def _validated_findings_only(
    candidates: list[StoredValidatedFinding],
    *,
    records: list[StoredInvestigationRecord],
    warnings: list[str],
) -> list[StoredValidatedFinding]:
    """Keep only findings backed by exactly one matching validated record."""
    records_by_finding_id: dict[str, list[StoredInvestigationRecord]] = {}
    for item in records:
        record = item.record
        if record.status == "validated" and record.finding_artifact_id:
            records_by_finding_id.setdefault(record.finding_artifact_id, []).append(item)

    findings: list[StoredValidatedFinding] = []
    for item in candidates:
        same_run_records = [
            stored
            for stored in records_by_finding_id.get(item.artifact_id, [])
            if stored.session_id == item.session_id
        ]
        if len(same_run_records) != 1:
            warnings.append(
                "excluding unverified ValidatedFinding artifact "
                f"{item.artifact_id}: expected one same-run validated investigation record"
            )
            continue
        record = same_run_records[0].record
        if (
            record.investigation_id != item.finding.investigation_id
            or record.question_id != item.finding.question_id
        ):
            warnings.append(
                "excluding mismatched ValidatedFinding artifact "
                f"{item.artifact_id}: investigation provenance does not match"
            )
            continue
        findings.append(item)
    return findings


def _questions_by_id(
    artifacts: list[Artifact],
    *,
    session_id: str,
    warnings: list[str],
) -> dict[str, str]:
    questions: dict[str, str] = {}
    for artifact in artifacts:
        if artifact.type is not ArtifactType.INVESTIGATION_PLAN:
            continue
        try:
            plan = InvestigationPlan.model_validate(artifact.payload)
        except ValueError:
            warnings.append(f"{session_id}: invalid InvestigationPlan payload {artifact.id}")
            continue
        questions[plan.question_id] = plan.question
    return questions


def _parse_finding(
    artifact: Artifact,
    *,
    session_id: str,
    warnings: list[str],
) -> ValidatedFinding | None:
    try:
        return ValidatedFinding.model_validate(artifact.payload)
    except ValueError:
        warnings.append(f"{session_id}: invalid ValidatedFinding payload {artifact.id}")
        return None


def _parse_record(
    artifact: Artifact,
    *,
    session_id: str,
    warnings: list[str],
) -> InvestigationRecord | None:
    try:
        return InvestigationRecord.model_validate(artifact.payload)
    except ValueError:
        warnings.append(f"{session_id}: invalid InvestigationRecord payload {artifact.id}")
        return None
