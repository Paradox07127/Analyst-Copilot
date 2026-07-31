from __future__ import annotations

from pydantic import ValidationError

from eda_platform.core.semantic import VerifiedAnswer
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.investigations import ValidatedFinding


def build_promotion_candidate(
    store: ArtifactStore,
    project_id: str,
    finding_artifact_id: str,
    *,
    session_id: str | None = None,
) -> VerifiedAnswer:
    """Build knowledge text solely from deterministic claim sentences."""
    exact_session_id = session_id or _unique_project_artifact_run(
        store,
        project_id=project_id,
        artifact_id=finding_artifact_id,
    )
    try:
        artifact = store.get_artifact(
            finding_artifact_id,
            project_id=project_id,
            session_id=exact_session_id,
        )
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(
            f"Finding artifact '{finding_artifact_id}' could not be loaded for promotion."
        ) from exc
    if artifact.project_id != project_id or artifact.type is not ArtifactType.VALIDATED_FINDING:
        raise ValueError(
            f"Artifact '{finding_artifact_id}' is not a ValidatedFinding in project '{project_id}'."
        )
    try:
        finding = ValidatedFinding.model_validate(artifact.payload)
    except ValidationError as exc:
        raise ValueError(
            f"Artifact '{finding_artifact_id}' does not contain a valid finding."
        ) from exc

    answer = " ".join(claim.text.strip() for claim in finding.findings if claim.text.strip())
    if not answer:
        raise ValueError("This finding has no claim sentences to promote as a verified answer.")
    return VerifiedAnswer(
        question=finding.question,
        answer=answer,
        evidence_note=_evidence_note(finding),
        verified_at=artifact.created_at,
    )


def _unique_project_artifact_run(
    store: ArtifactStore,
    *,
    project_id: str,
    artifact_id: str,
) -> str:
    try:
        row = store.artifact_index_row(
            artifact_id,
            project_id=project_id,
        )
    except ValueError as exc:
        raise ValueError(
            f"Artifact '{artifact_id}' has no unique partition in project '{project_id}'."
        ) from exc
    if row is None:
        raise ValueError(
            f"Artifact '{artifact_id}' has no unique partition in project '{project_id}'."
        )
    return str(row["session_id"])


def _evidence_note(finding: ValidatedFinding) -> str:
    source_ids = list(dict.fromkeys(finding.source_artifact_ids))
    sources = ", ".join(source_ids) if source_ids else "none recorded"
    note = f"Source artifacts: {sources}."
    if finding.limitations:
        note += f" Limitation: {finding.limitations[0]}"
    return note
