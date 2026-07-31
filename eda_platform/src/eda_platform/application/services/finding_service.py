"""Findings library use cases (§7.5 / 阶段4 slice H).

The Findings Library is project-level state (§2.7 缺口 11): listing is scoped
to the run's project and includes findings from every project run, each
annotated with its source run. Shaping reuses the existing
`investigation_library` read model and `finding_freshness` assessor; one
project-runs snapshot is built per listing and shared across all freshness
assessments.
"""

from __future__ import annotations

import logging

from eda_platform.application.dto import (
    DecisionCoverageView,
    FindingEvidenceRef,
    FindingFreshnessInfo,
    FindingStatement,
    FindingSummary,
    FindingsView,
    InvestigationLogEntry,
)
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.core.decision_coverage import assess_decision_coverage
from eda_platform.core.finding_freshness import (
    assess_finding_freshness,
    project_run_artifacts,
)
from eda_platform.core.ids import INTERNAL_SESSION_MARKER
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.investigation_library import (
    StoredValidatedFinding,
    build_investigation_library,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.investigations import ValidatedFinding

logger = logging.getLogger(__name__)

# Typed method-execution artifacts a finding may be backed by.
_METHOD_ARTIFACT_TYPES = frozenset(
    {
        ArtifactType.STAT_TEST_RESULT,
        ArtifactType.MODEL_CARD,
        ArtifactType.ANOMALY_SCREEN_RESULT,
    }
)

_ProjectRuns = list[tuple[str, list[Artifact]]]


class FindingService:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def list_findings(self, session_id: str) -> FindingsView:
        project_id = self._project_for_run(session_id)
        library = build_investigation_library(self._store, project_id=project_id)
        project_runs = project_run_artifacts(self._store, project_id)
        findings = [
            self._to_summary(
                item, project_id=project_id, viewed_session_id=session_id, project_runs=project_runs
            )
            for item in library.findings
        ]
        records = [
            InvestigationLogEntry(
                artifact_id=item.artifact_id,
                source_session_id=item.session_id,
                from_current_session=item.session_id == session_id,
                question=item.question,
                status=item.record.status,
                reason_code=item.record.reason_code,
                reason=item.record.reason,
                next_action=item.record.next_action,
            )
            for item in library.records
        ]
        return FindingsView(
            session_id=session_id,
            project_id=project_id,
            findings=findings,
            records=records,
            warnings=library.warnings,
        )

    def get_decision_coverage(self, session_id: str) -> DecisionCoverageView:
        """Coverage across every run of this run's project, computed on read —
        the DECISION_COVERAGE artifact is not written or consulted here."""
        project_id = self._project_for_run(session_id)
        coverage = assess_decision_coverage(self._store, project_id)
        return DecisionCoverageView(
            session_id=session_id,
            project_id=coverage.project_id,
            top_cards_total=coverage.top_cards_total,
            top_cards_terminal=coverage.top_cards_terminal,
            uninvestigated_high_value=list(coverage.uninvestigated_high_value),
            findings_not_eligible=coverage.findings_not_eligible,
            validated_findings=coverage.validated_findings,
            coverage_ready=coverage.coverage_ready,
            gaps=list(coverage.gaps),
        )

    def _to_summary(
        self,
        item: StoredValidatedFinding,
        *,
        project_id: str,
        viewed_session_id: str,
        project_runs: _ProjectRuns,
    ) -> FindingSummary:
        finding = item.finding
        artifact_cache: dict[tuple[str, str | None], Artifact | None] = {}
        statements = [
            FindingStatement(
                text=statement.text,
                evidence=[
                    FindingEvidenceRef(
                        kind=str(evidence.kind),
                        artifact_id=evidence.artifact_id,
                        locator=evidence.locator,
                        session_id=self._evidence_session_id(
                            evidence.artifact_id,
                            project_id,
                            finding.source_artifact_session_ids.get(
                                evidence.artifact_id or "",
                            ),
                            artifact_cache,
                        ),
                    )
                    for evidence in statement.evidence
                ],
            )
            for statement in finding.findings
        ]
        interpretation = (
            finding.interpretation
            if finding.interpretation_status == "validated" and finding.interpretation
            else None
        )
        return FindingSummary(
            artifact_id=item.artifact_id,
            source_session_id=item.session_id,
            source_session_navigable=INTERNAL_SESSION_MARKER not in item.session_id,
            from_current_session=item.session_id == viewed_session_id,
            created_at=item.created_at,
            question=finding.question,
            claim_class=finding.claim_class,
            evidence_support=finding.evidence_support,
            analytical_reliability=finding.analytical_reliability,
            decision_readiness=finding.decision_readiness,
            report_readiness=finding.report_readiness,
            report_readiness_reason=finding.report_readiness_reason,
            statements=statements,
            limitations=list(finding.limitations),
            interpretation=interpretation,
            value_hypothesis=finding.value_hypothesis or None,
            method_artifact_id=self._method_artifact_id(
                finding,
                project_id,
                artifact_cache,
            ),
            freshness=self._freshness(
                project_id,
                item.artifact_id,
                item.session_id,
                project_runs,
            ),
        )

    def _get_artifact_cached(
        self,
        artifact_id: str,
        project_id: str,
        session_id: str | None,
        cache: dict[tuple[str, str | None], Artifact | None],
    ) -> Artifact | None:
        key = (artifact_id, session_id)
        if key not in cache:
            try:
                cache[key] = self._store.get_artifact(
                    artifact_id,
                    project_id=project_id,
                    session_id=session_id,
                )
            except (KeyError, OSError, ValueError):
                cache[key] = None
        return cache[key]

    def _evidence_session_id(
        self,
        artifact_id: str | None,
        project_id: str,
        session_id: str | None,
        cache: dict[tuple[str, str | None], Artifact | None],
    ) -> str | None:
        """The run whose Artifacts page can show this evidence, if navigable."""
        if not artifact_id:
            return None
        artifact = self._get_artifact_cached(
            artifact_id,
            project_id,
            session_id,
            cache,
        )
        if artifact is None or INTERNAL_SESSION_MARKER in artifact.session_id:
            return None
        return artifact.session_id

    def _method_artifact_id(
        self,
        finding: ValidatedFinding,
        project_id: str,
        cache: dict[tuple[str, str | None], Artifact | None],
    ) -> str | None:
        candidate_ids: list[str] = list(finding.source_artifact_ids)
        for statement in finding.findings:
            for evidence in statement.evidence:
                if evidence.artifact_id:
                    candidate_ids.append(evidence.artifact_id)
        seen: set[str] = set()
        for artifact_id in candidate_ids:
            if artifact_id in seen:
                continue
            seen.add(artifact_id)
            artifact = self._get_artifact_cached(
                artifact_id,
                project_id,
                finding.source_artifact_session_ids.get(
                    artifact_id,
                ),
                cache,
            )
            if artifact is not None and artifact.type in _METHOD_ARTIFACT_TYPES:
                return artifact.id
        return None

    def _freshness(
        self,
        project_id: str,
        finding_artifact_id: str,
        finding_session_id: str,
        project_runs: _ProjectRuns,
    ) -> FindingFreshnessInfo:
        try:
            freshness = assess_finding_freshness(
                self._store,
                project_id,
                finding_artifact_id,
                finding_session_id=finding_session_id,
                project_runs=project_runs,
            )
        except Exception:  # noqa: BLE001 — a broken assessor must not break the list
            logger.warning(
                "Freshness assessment failed for finding %s in project %s",
                finding_artifact_id,
                project_id,
                exc_info=True,
            )
            return FindingFreshnessInfo(
                status="unverifiable",
                reasons=["Freshness could not be computed for this finding."],
            )
        return FindingFreshnessInfo(status=freshness.status, reasons=list(freshness.reasons))

    def _project_for_run(self, session_id: str) -> str:
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        row = self._store.get_session_index_row(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        return str(row["project_id"])
