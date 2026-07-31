"""Content fingerprint for the inputs that make a decision report reusable."""

from __future__ import annotations

from collections.abc import Sequence

from eda_platform.core.ids import stable_hash
from eda_platform.schemas.artifacts import Artifact, ArtifactType

DECISION_REPORT_POLICY_VERSION = "decision-report-v1"


def decision_report_input_fingerprint(findings: Sequence[Artifact]) -> str:
    """Hash canonical finding payloads, independent of ordering and timestamps."""
    canonical: list[dict[str, object]] = []
    for artifact in sorted(findings, key=lambda item: item.id):
        if artifact.type is not ArtifactType.VALIDATED_FINDING:
            raise ValueError(f"Expected ValidatedFinding artifact: {artifact.id}")
        canonical.append({"artifact_id": artifact.id, "payload": artifact.payload})
    return stable_hash(
        {
            "fingerprint_schema_version": 1,
            "report_policy_version": DECISION_REPORT_POLICY_VERSION,
            "source_findings": canonical,
        },
        length=64,
    )
