from __future__ import annotations

from collections.abc import Iterable

from pydantic import ValidationError

from eda_platform.core.publication_fingerprint import (
    DECISION_REPORT_POLICY_VERSION,
    decision_report_input_fingerprint,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile
from eda_platform.schemas.decision_report import DecisionReport
from eda_platform.schemas.finding_freshness import (
    DecisionReportFreshness,
    FindingFreshness,
    FreshnessStatus,
)
from eda_platform.schemas.investigations import InvestigationPlan, ValidatedFinding

# Freshness only needs plans (lineage) and dataset profiles (identity compare).
_FRESHNESS_ARTIFACT_TYPES = (
    ArtifactType.INVESTIGATION_PLAN,
    ArtifactType.DATASET_PROFILE,
)


def assess_finding_freshness(
    store: ArtifactStore,
    project_id: str,
    finding_artifact_id: str,
    *,
    finding_session_id: str | None = None,
    project_runs: list[tuple[str, list[Artifact]]] | None = None,
) -> FindingFreshness:
    """Compare a finding's source snapshot with the project's current datasets.

    Pass ``project_runs`` (from :func:`project_run_artifacts`) to reuse one
    snapshot across many findings instead of re-reading every run per call.
    """
    return _assess_finding_freshness(
        store,
        project_id,
        finding_artifact_id,
        finding_session_id=finding_session_id,
        project_runs=(
            project_run_artifacts(store, project_id) if project_runs is None else project_runs
        ),
    )


def _assess_finding_freshness(
    store: ArtifactStore,
    project_id: str,
    finding_artifact_id: str,
    *,
    finding_session_id: str | None = None,
    project_runs: list[tuple[str, list[Artifact]]],
) -> FindingFreshness:
    finding_artifact = _get_project_artifact(
        store,
        project_id,
        finding_artifact_id,
        session_id=finding_session_id,
    )
    if finding_artifact is None or finding_artifact.type is not ArtifactType.VALIDATED_FINDING:
        return _unverifiable(
            finding_artifact_id,
            "The finding source could not be loaded, so its freshness cannot be verified.",
        )
    try:
        finding = ValidatedFinding.model_validate(finding_artifact.payload)
    except ValidationError:
        return _unverifiable(
            finding_artifact_id,
            "The finding source is not a valid ValidatedFinding, so its "
            "freshness cannot be verified.",
        )

    plan = _find_plan(project_runs, finding.investigation_id)
    if plan is None:
        return _unverifiable(
            finding_artifact_id,
            (
                f"No InvestigationPlan was found for investigation "
                f"'{finding.investigation_id}', so the source datasets cannot be verified."
            ),
        )

    source_artifacts = next(
        (
            artifacts
            for session_id, artifacts in project_runs
            if session_id == plan.source_session_id
        ),
        [],
    )
    recorded_profiles = _profiles_by_name(source_artifacts)
    checked_names = list(dict.fromkeys(plan.target_datasets))
    if not source_artifacts or any(name not in recorded_profiles for name in checked_names):
        missing = [name for name in checked_names if name not in recorded_profiles]
        detail = f" Dataset profiles are missing for: {', '.join(missing)}." if missing else ""
        return FindingFreshness(
            finding_artifact_id=finding_artifact_id,
            status="unverifiable",
            reasons=[f"Source session '{plan.source_session_id}' could not be verified.{detail}"],
            checked_dataset_names=checked_names,
        )

    current_profiles = _current_profiles_by_name(project_runs, checked_names)
    stale_reasons: list[str] = []
    unverifiable_reasons: list[str] = []
    for name in checked_names:
        recorded_id = recorded_profiles[name].dataset_id
        current = current_profiles.get(name)
        if current is None:
            unverifiable_reasons.append(
                f"Dataset '{name}' is no longer available in the project's current uploads."
            )
        elif current.dataset_id != recorded_id:
            stale_reasons.append(
                f"Dataset '{name}' changed from dataset_id '{recorded_id}' to "
                f"'{current.dataset_id}'; review the finding against the new upload."
            )

    for evidence_id in _evidence_artifact_ids(finding):
        if (
            _get_project_artifact(
                store,
                project_id,
                evidence_id,
                session_id=finding.source_artifact_session_ids.get(evidence_id),
            )
            is None
        ):
            unverifiable_reasons.append(
                f"Evidence artifact '{evidence_id}' no longer loads, so its claim "
                "cannot be checked."
            )

    if stale_reasons:
        return FindingFreshness(
            finding_artifact_id=finding_artifact_id,
            status="stale",
            reasons=[*stale_reasons, *unverifiable_reasons],
            checked_dataset_names=checked_names,
        )
    if unverifiable_reasons:
        return FindingFreshness(
            finding_artifact_id=finding_artifact_id,
            status="unverifiable",
            reasons=unverifiable_reasons,
            checked_dataset_names=checked_names,
        )
    return FindingFreshness(
        finding_artifact_id=finding_artifact_id,
        status="fresh",
        reasons=["The source datasets and all claim evidence still match the saved finding."],
        checked_dataset_names=checked_names,
    )


def assess_decision_report_freshness(
    store: ArtifactStore,
    project_id: str,
    report_artifact_id: str,
    *,
    report_session_id: str | None = None,
) -> DecisionReportFreshness:
    """Aggregate the existing finding-lineage checks for one decision report."""
    report_artifact = _get_project_artifact(
        store,
        project_id,
        report_artifact_id,
        session_id=report_session_id,
    )
    if report_artifact is None or report_artifact.type is not ArtifactType.DECISION_REPORT:
        return _unverifiable_report(
            report_artifact_id,
            "The decision report could not be loaded, so freshness cannot be verified.",
        )
    try:
        report = DecisionReport.model_validate(report_artifact.payload)
    except ValidationError:
        return _unverifiable_report(
            report_artifact_id,
            "The decision report payload is invalid, so freshness cannot be verified.",
        )

    source_ids = list(dict.fromkeys(report.source_finding_artifact_ids))
    missing_parents = [
        finding_id for finding_id in source_ids if finding_id not in report_artifact.parents
    ]
    if missing_parents:
        return _unverifiable_report(
            report_artifact_id,
            "Decision report lineage does not include source finding parent(s): "
            + ", ".join(missing_parents),
        )

    project_runs = project_run_artifacts(store, project_id)
    finding_statuses: dict[str, FreshnessStatus] = {}
    reasons: list[str] = []
    source_artifacts: list[Artifact] = []
    for finding_id in source_ids:
        freshness = _assess_finding_freshness(
            store,
            project_id,
            finding_id,
            finding_session_id=report.source_finding_session_ids.get(finding_id),
            project_runs=project_runs,
        )
        finding_statuses[finding_id] = freshness.status
        source_artifact = _get_project_artifact(
            store,
            project_id,
            finding_id,
            session_id=report.source_finding_session_ids.get(finding_id),
        )
        if source_artifact is not None:
            source_artifacts.append(source_artifact)
        if freshness.status != "fresh":
            reasons.extend(f"{finding_id}: {reason}" for reason in freshness.reasons)

    if any(status == "stale" for status in finding_statuses.values()):
        status: FreshnessStatus = "stale"
    elif any(status == "unverifiable" for status in finding_statuses.values()):
        status = "unverifiable"
    elif report.publication_input_fingerprint is None or report.report_policy_version is None:
        status = "unverifiable"
        reasons = [
            "This legacy decision report has no publication input fingerprint or policy "
            "version; source findings are fresh, but report integrity is unknown."
        ]
    elif report.report_policy_version != DECISION_REPORT_POLICY_VERSION:
        status = "stale"
        reasons = [
            f"Report policy changed from '{report.report_policy_version}' to "
            f"'{DECISION_REPORT_POLICY_VERSION}'."
        ]
    elif (
        decision_report_input_fingerprint(source_artifacts) != report.publication_input_fingerprint
    ):
        status = "stale"
        reasons = [
            "The canonical source-finding payloads no longer match the report's saved "
            "publication fingerprint."
        ]
    else:
        status = "fresh"
        reasons = ["Every source finding still matches its current datasets and evidence."]
    return DecisionReportFreshness(
        report_artifact_id=report_artifact_id,
        status=status,
        finding_statuses=finding_statuses,
        reasons=reasons,
    )


def _unverifiable_report(report_artifact_id: str, reason: str) -> DecisionReportFreshness:
    return DecisionReportFreshness(
        report_artifact_id=report_artifact_id,
        status="unverifiable",
        reasons=[reason],
    )


def _unverifiable(finding_artifact_id: str, reason: str) -> FindingFreshness:
    return FindingFreshness(
        finding_artifact_id=finding_artifact_id,
        status="unverifiable",
        reasons=[reason],
        checked_dataset_names=[],
    )


def _get_project_artifact(
    store: ArtifactStore,
    project_id: str,
    artifact_id: str,
    *,
    session_id: str | None = None,
) -> Artifact | None:
    try:
        artifact = store.get_artifact(
            artifact_id,
            project_id=project_id,
            session_id=session_id,
        )
    except (KeyError, OSError, ValueError):
        return None
    return artifact if artifact.project_id == project_id else None


def project_run_artifacts(
    store: ArtifactStore, project_id: str
) -> list[tuple[str, list[Artifact]]]:
    """One (session_id, artifacts) snapshot per project session for freshness.

    Only investigation plans and dataset profiles are loaded — charts, findings
    payloads, and other noise stay on disk.
    """
    snapshot: list[tuple[str, list[Artifact]]] = []
    for run in store.list_sessions(project_id):
        artifacts, _ = store.list_artifacts_of_types(
            project_id=project_id,
            session_id=run.session_id,
            artifact_types=_FRESHNESS_ARTIFACT_TYPES,
        )
        snapshot.append((run.session_id, artifacts))
    return snapshot


def _find_plan(
    project_runs: Iterable[tuple[str, list[Artifact]]],
    investigation_id: str,
) -> InvestigationPlan | None:
    for _session_id, artifacts in project_runs:
        for artifact in sorted(artifacts, key=lambda item: item.id):
            if artifact.type is not ArtifactType.INVESTIGATION_PLAN:
                continue
            try:
                plan = InvestigationPlan.model_validate(artifact.payload)
            except ValidationError:
                continue
            if plan.investigation_id == investigation_id:
                return plan
    return None


def _profiles_by_name(artifacts: Iterable[Artifact]) -> dict[str, DatasetProfile]:
    profiles: dict[str, DatasetProfile] = {}
    for artifact in sorted(artifacts, key=lambda item: item.id):
        if artifact.type is not ArtifactType.DATASET_PROFILE:
            continue
        try:
            profile = DatasetProfile.model_validate(artifact.payload)
        except ValidationError:
            continue
        profiles.setdefault(profile.name, profile)
    return profiles


def _current_profiles_by_name(
    project_runs: Iterable[tuple[str, list[Artifact]]],
    target_names: Iterable[str],
) -> dict[str, DatasetProfile]:
    remaining = set(target_names)
    current: dict[str, DatasetProfile] = {}
    for _session_id, artifacts in project_runs:
        for name, profile in _profiles_by_name(artifacts).items():
            if name in remaining:
                current[name] = profile
                remaining.remove(name)
        if not remaining:
            break
    return current


def _evidence_artifact_ids(finding: ValidatedFinding) -> list[str]:
    return sorted(
        {
            evidence.artifact_id
            for claim in finding.findings
            for evidence in claim.evidence
            if evidence.artifact_id is not None
        }
    )
