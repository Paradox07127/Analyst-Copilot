"""Bounded Role 3 synthesis over user-selected validated findings."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from eda_platform.core.config import require_absolute_workspace
from eda_platform.core.ids import make_artifact_id, stable_hash
from eda_platform.core.store import ArtifactStore
from eda_platform.core.trace import FINDINGS_DEDUPLICATED
from eda_platform.drivers.cancellation import raise_if_cancelled
from eda_platform.drivers.investigation_library import (
    InvestigationLibrary,
    StoredValidatedFinding,
    load_investigation_library,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.sessions import SessionManifest, TraceEvent
from eda_platform.schemas.synthesis import SynthesisBrief, SynthesisStoryBeat
from eda_platform.tools.interestingness import (
    DedupFinding,
    FindingCluster,
    deduplicate_findings,
)

_HYPOTHESIS_LABEL = "Hypothesis (unverified, not established by this evidence): "


@dataclass(frozen=True)
class SynthesisRunResult:
    project_id: str
    session_id: str
    artifact: Artifact
    workspace: Path


@dataclass(frozen=True)
class StoredSynthesisBrief:
    artifact_id: str
    session_id: str
    created_at: datetime
    brief: SynthesisBrief


def create_synthesis_brief(
    *,
    project_id: str,
    finding_artifact_ids: Sequence[str],
    finding_session_ids: dict[str, str] | None = None,
    workspace: Path | str,
    business_context: str = "",
    session_id: str | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> SynthesisRunResult:
    """Create a reviewable story from explicit, report-eligible selections."""
    raise_if_cancelled(cancel_check, operation="synthesis brief")
    workspace_path = require_absolute_workspace(workspace)
    library = load_investigation_library(
        workspace=str(workspace_path),
        project_id=project_id,
    )
    raise_if_cancelled(cancel_check, operation="synthesis brief")
    selected = _selected_findings(
        library,
        finding_artifact_ids,
        finding_session_ids=finding_session_ids,
    )
    store = ArtifactStore(workspace_path)
    # Quote cluster representatives while retaining supporting members for disclosure.
    clusters = _deduplicate_selected(
        selected,
        store,
        project_id=project_id,
    )
    raise_if_cancelled(cancel_check, operation="synthesis brief")
    merged_supporting = sum(len(cluster.supporting) for cluster in clusters)
    actual_session_id = session_id or _synthesis_session_id(project_id, selected, business_context)
    brief = _build_brief(
        project_id=project_id,
        selected=selected,
        library=library,
        business_context=business_context,
        merged_supporting=merged_supporting,
    )
    raise_if_cancelled(cancel_check, operation="synthesis brief")
    artifact = _brief_artifact(brief, project_id=project_id, session_id=actual_session_id)
    store.start_session(project_id, actual_session_id)
    store.write_manifest(
        SessionManifest(
            session_id=actual_session_id,
            project_id=project_id,
            input_hashes={item.artifact_id: "validated_finding" for item in selected},
            code_version="synthesis-orchestrator-v2",
            model_versions={"synthesis": "deterministic"},
        )
    )
    store.save_artifact(artifact)
    store.append_trace(
        project_id,
        TraceEvent(
            session_id=actual_session_id,
            event_type="synthesis_brief_created",
            name="synthesis_orchestrator",
            finished_at=datetime.now(UTC),
            summary={
                "selected_finding_count": len(selected),
                "source_record_count": len(brief.source_record_artifact_ids),
                "report_eligible": brief.report_eligible,
            },
        ),
    )
    store.append_trace(
        project_id,
        TraceEvent(
            session_id=actual_session_id,
            event_type=FINDINGS_DEDUPLICATED,
            name="synthesis_orchestrator",
            finished_at=datetime.now(UTC),
            summary={
                "clusters": len(clusters),
                "merged_supporting": merged_supporting,
                "selected_finding_count": len(selected),
            },
        ),
    )
    store.mark_session_status(project_id, actual_session_id, "ready_for_review")
    return SynthesisRunResult(
        project_id=project_id,
        session_id=actual_session_id,
        artifact=artifact,
        workspace=workspace_path,
    )


def load_synthesis_briefs(
    *,
    project_id: str,
    workspace: Path | str,
) -> tuple[list[StoredSynthesisBrief], list[str]]:
    """Read all valid synthesis drafts across a project, newest first."""
    store = ArtifactStore(workspace)
    briefs: list[StoredSynthesisBrief] = []
    warnings: list[str] = []
    for run in store.list_sessions(project_id):
        artifacts, artifact_warnings = store.list_artifacts_of_types(
            project_id=project_id,
            session_id=run.session_id,
            artifact_types=(ArtifactType.SYNTHESIS_BRIEF,),
        )
        warnings.extend(f"{run.session_id}: {warning}" for warning in artifact_warnings)
        for artifact in artifacts:
            try:
                brief = SynthesisBrief.model_validate(artifact.payload)
            except ValueError:
                warnings.append(f"{run.session_id}: invalid SynthesisBrief payload {artifact.id}")
                continue
            briefs.append(
                StoredSynthesisBrief(
                    artifact_id=artifact.id,
                    session_id=run.session_id,
                    created_at=artifact.created_at,
                    brief=brief,
                )
            )
    briefs.sort(key=lambda item: (item.created_at, item.artifact_id), reverse=True)
    return briefs, warnings


def _selected_findings(
    library: InvestigationLibrary,
    finding_artifact_ids: Sequence[str],
    *,
    finding_session_ids: dict[str, str] | None = None,
) -> list[StoredValidatedFinding]:
    requested_ids = list(dict.fromkeys(finding_artifact_ids))
    if not requested_ids:
        raise ValueError("Select at least one validated finding for synthesis.")
    requested_runs = finding_session_ids or {}
    unexpected = set(requested_runs).difference(requested_ids)
    if unexpected:
        raise ValueError(
            "Finding run identities were supplied for unselected artifacts: "
            + ", ".join(sorted(unexpected))
        )
    findings_by_identity = {
        (item.artifact_id, item.session_id): item for item in library.findings
    }
    selected: list[StoredValidatedFinding] = []
    missing: list[str] = []
    for artifact_id in requested_ids:
        session_id = requested_runs.get(artifact_id)
        if session_id is not None:
            item = findings_by_identity.get((artifact_id, session_id))
        else:
            matches = [
                candidate
                for candidate in library.findings
                if candidate.artifact_id == artifact_id
            ]
            item = matches[0] if len(matches) == 1 else None
        if item is None:
            missing.append(artifact_id)
        else:
            selected.append(item)
    if missing:
        raise ValueError(
            "Selected findings are unavailable or have ambiguous run identity: "
            + ", ".join(missing)
        )
    ineligible = [item.artifact_id for item in selected if not item.finding.report_eligible]
    if ineligible:
        raise ValueError(
            "Only report-eligible validated findings can enter a synthesis brief: "
            + ", ".join(ineligible)
        )
    return selected


# Structured payload keys that name the columns a source artifact is about;
# used to build cluster keys. Sentences are never parsed for columns.
_COLUMN_PAYLOAD_KEYS = ("group_column", "value_column", "column", "target_column")


def _deduplicate_selected(
    selected: Sequence[StoredValidatedFinding],
    store: ArtifactStore,
    *,
    project_id: str,
) -> list[FindingCluster]:
    """Cluster selected findings across questions and retain their representatives."""
    items: list[DedupFinding] = []
    for item in selected:
        columns = _source_columns(
            store,
            item.finding.source_artifact_ids,
            project_id=project_id,
            artifact_session_ids=item.finding.source_artifact_session_ids,
        )
        items.extend(
            DedupFinding(
                ref=f"{item.artifact_id}#{index}",
                finding=finding,
                columns=columns,
            )
            for index, finding in enumerate(item.finding.findings)
        )
    return deduplicate_findings(items)


def _source_columns(
    store: ArtifactStore,
    artifact_ids: Sequence[str],
    *,
    project_id: str,
    artifact_session_ids: dict[str, str] | None = None,
) -> list[str]:
    columns: list[str] = []
    for artifact_id in artifact_ids:
        try:
            payload = store.get_artifact(
                artifact_id,
                project_id=project_id,
                session_id=(artifact_session_ids or {}).get(artifact_id),
            ).payload
        except (KeyError, ValueError, OSError):
            continue
        for key in _COLUMN_PAYLOAD_KEYS:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                columns.append(value)
        feature_columns = payload.get("feature_columns")
        if isinstance(feature_columns, list):
            columns.extend(
                value for value in feature_columns if isinstance(value, str) and value.strip()
            )
    return columns


def _synthesis_session_id(
    project_id: str,
    selected: Sequence[StoredValidatedFinding],
    business_context: str,
) -> str:
    return "synthesis_" + stable_hash(
        {
            "project_id": project_id,
            "finding_ids": [item.artifact_id for item in selected],
            "business_context": _normalize_business_context(business_context),
        },
        length=12,
    )


def _build_brief(
    *,
    project_id: str,
    selected: Sequence[StoredValidatedFinding],
    library: InvestigationLibrary,
    business_context: str,
    merged_supporting: int = 0,
) -> SynthesisBrief:
    finding_ids = [item.artifact_id for item in selected]
    selected_question_ids = {item.finding.question_id for item in selected}
    decision_actions = _unique(
        item.finding.decision_action.strip()
        for item in selected
        if item.finding.decision_action.strip()
    )
    questions = _unique(item.finding.question for item in selected)
    headline = _headline(selected)
    limitations = _unique(
        limitation
        for item in selected
        for limitation in item.finding.limitations
        if limitation.strip()
    )
    if merged_supporting > 0:
        # Supporting members are merged into their cluster
        # representative for the story, never removed from the selection.
        limitations.append(
            f"{merged_supporting} similar finding(s) were merged into cluster "
            "representatives; only representatives are quoted as evidence."
        )
    gap_records = [
        item
        for item in library.records
        if item.record.question_id in selected_question_ids
        and item.record.status in {"needs_data", "inconclusive", "failed"}
    ]
    gaps = _unique(f"{item.question}: {item.record.next_action}" for item in gap_records)
    # decision_context is a labelled brief field, not a beat body; it still must
    # not carry unverified business_context text.
    decision_context = "The selected evidence addresses: " + "; ".join(questions)
    storyline = [
        SynthesisStoryBeat(
            title="Decision context",
            body=_context_body(questions, decision_actions),
            finding_artifact_ids=finding_ids,
        ),
        SynthesisStoryBeat(
            title="Validated evidence",
            body=_evidence_body(selected),
            finding_artifact_ids=finding_ids,
        ),
        SynthesisStoryBeat(
            title="Decision implication",
            body=_decision_body(decision_actions, questions),
            finding_artifact_ids=finding_ids,
        ),
        SynthesisStoryBeat(
            title="Limits and next validation",
            body=_limits_body(limitations, gaps),
            finding_artifact_ids=finding_ids,
        ),
    ]
    return SynthesisBrief(
        brief_id="sbrief_"
        + stable_hash(
            {
                "finding_ids": finding_ids,
                "business_context": _normalize_business_context(business_context),
            },
            length=12,
        ),
        project_id=project_id,
        business_context=business_context.strip(),
        selected_finding_artifact_ids=finding_ids,
        selected_finding_session_ids={
            item.artifact_id: item.session_id for item in selected
        },
        source_record_artifact_ids=[item.artifact_id for item in gap_records],
        decision_context=decision_context,
        headline=headline,
        storyline=storyline,
        limitations=limitations,
        investigation_gaps=gaps,
        report_eligible=True,
        report_readiness=(
            "eligible_with_limitations"
            if any(
                item.finding.report_readiness == "eligible_with_limitations" for item in selected
            )
            else "eligible"
        ),
    )


def _headline(selected: Sequence[StoredValidatedFinding]) -> str:
    # Prefer a cluster representative; fall back to any finding sentence.
    for item in selected:
        for finding in item.finding.findings:
            if finding.dedup_role != "supporting":
                return finding.text
    for item in selected:
        if item.finding.findings:
            return item.finding.findings[0].text
    return "Selected investigations produced validated evidence for review."


def _context_body(questions: list[str], decision_actions: list[str]) -> str:
    body = "The selected evidence addresses: " + "; ".join(questions) + "."
    if decision_actions:
        body += " " + _HYPOTHESIS_LABEL + "; ".join(decision_actions) + "."
    return body


def _evidence_body(selected: Sequence[StoredValidatedFinding]) -> str:
    evidence: list[str] = []
    for item in selected:
        # Quote cluster representatives only; supporting members stay in
        # the selection and are disclosed through the limitations beat.
        statements = [
            finding.text for finding in item.finding.findings if finding.dedup_role != "supporting"
        ]
        if statements:
            evidence.append(f"{item.finding.question}: " + " ".join(statements))
    return " ".join(evidence)


def _decision_body(decision_actions: list[str], questions: list[str]) -> str:
    if decision_actions:
        return (
            "Use the validated evidence when reviewing these questions: "
            + "; ".join(questions)
            + ". "
            + _HYPOTHESIS_LABEL
            + "; ".join(decision_actions)
            + "."
        )
    return "Use the validated evidence to review: " + "; ".join(questions) + "."


def _limits_body(limitations: list[str], gaps: list[str]) -> str:
    parts: list[str] = []
    if limitations:
        parts.append("Limitations: " + "; ".join(limitations))
    if gaps:
        parts.append("Open investigations: " + "; ".join(gaps))
    if not parts:
        parts.append("No additional investigation gaps were recorded for this selection.")
    return " ".join(parts)


def _brief_artifact(
    brief: SynthesisBrief,
    *,
    project_id: str,
    session_id: str,
) -> Artifact:
    payload = brief.model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("sbrief", {"session_id": session_id, "brief": payload}),
        type=ArtifactType.SYNTHESIS_BRIEF,
        project_id=project_id,
        session_id=session_id,
        parents=[*brief.selected_finding_artifact_ids, *brief.source_record_artifact_ids],
        payload=payload,
        plain_language=(
            "A user-selected decision story assembled only from validated findings and "
            "recorded investigation limits. It is not a final published report."
        ),
    )


def _normalize_business_context(business_context: str) -> str:
    return re.sub(r"\s+", " ", business_context.strip().lower())


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
