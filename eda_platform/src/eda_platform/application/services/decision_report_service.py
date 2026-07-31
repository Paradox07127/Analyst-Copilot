"""Role-3 decision report use cases (§10.3).

Decision reports are project-level: they are persisted under the synthesis run
that produced them, not under the analysis run being viewed, so the lookup is
scoped to the run's project and the newest report wins. A project without one
is a normal ``status="none"`` response, not a 404.

The write side mirrors the two buttons on that page. Both go through the job
system rather than the request thread, and both put their work on a derived
lifecycle run (`sbsess_*`, `drsess_*`) that carries only the job: the synthesis
orchestrator mints its own `synthesis_*` run for the brief, and a decision
report inherits the brief's run, so neither artifact belongs to the job's run.
Neither kind spends an LLM — the orchestrator is deterministic and the report
driver is called with ``llm=None`` so generation stays deterministic.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from eda_platform.application.dto import (
    CandidateDecisionView,
    DecisionEvidenceRefView,
    DecisionReportFreshnessView,
    DecisionReportGenerationStarted,
    DecisionReportSectionView,
    DecisionReportView,
    DecisionStoryBeatView,
    DecisionStoryDraftStarted,
    DecisionStoryDraftView,
    DecisionStoryFindingView,
    DecisionStoryView,
    JobCreated,
    JobStatus,
    MetaInsightView,
    SCQAView,
)
from eda_platform.application.services.job_service import JobConflictError, JobService
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.application.workspace_paths import relativize_workspace_paths
from eda_platform.core.finding_freshness import (
    assess_decision_report_freshness,
    assess_finding_freshness,
    project_run_artifacts,
)
from eda_platform.core.ids import INTERNAL_SESSION_MARKER, stable_hash
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.investigation_library import (
    StoredValidatedFinding,
    build_investigation_library,
)
from eda_platform.drivers.synthesis_orchestrator import (
    StoredSynthesisBrief,
    load_synthesis_briefs,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.decision_report import DecisionReport
from eda_platform.schemas.investigations import ValidatedFinding

logger = logging.getLogger(__name__)

MAX_DECISION_REPORT_BYTES = 5 * 1024 * 1024

SYNTHESIS_JOB_KIND = "synthesis_brief_create"
DECISION_REPORT_JOB_KIND = "decision_report_generate"
SYNTHESIS_SESSION_PREFIX = "sbsess_"
DECISION_REPORT_SESSION_PREFIX = "drsess_"
# Drafting and generating both read the project's findings and write into the
# same story surface, so the busy guard treats them as one lane per source run.
DECISION_STORY_JOB_KINDS = frozenset({SYNTHESIS_JOB_KIND, DECISION_REPORT_JOB_KIND})


def generate_synthesis_session_id(source_session_id: str) -> str:
    """Derived lifecycle run for one drafting job; the suffix is deterministic
    so an idempotent retry can prove a job really came from this request."""
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    suffix = stable_hash({"synthesis_source_session_id": source_session_id}, length=6)
    return f"{SYNTHESIS_SESSION_PREFIX}{stamp}_{suffix}"


def generate_decision_report_session_id(source_session_id: str, brief_artifact_id: str) -> str:
    """Derived lifecycle run for one generation, keyed on the brief too so two
    drafts of one run do not collide in the busy guard."""
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    suffix = stable_hash(
        {"report_source_session_id": source_session_id, "brief_artifact_id": brief_artifact_id},
        length=6,
    )
    return f"{DECISION_REPORT_SESSION_PREFIX}{stamp}_{suffix}"


class DecisionStoryError(Exception):
    pass


class DecisionStoryBusyError(DecisionStoryError):
    def __init__(self, session_id: str, job_id: str) -> None:
        super().__init__(
            f"Run {session_id} has an active job ({job_id}); wait for it to finish "
            "before curating the decision story."
        )
        self.session_id = session_id
        self.job_id = job_id


class DecisionStoryNotDraftableError(DecisionStoryError):
    """The selection cannot become a brief — the driver would only raise."""


class DecisionStoryDraftNotFoundError(DecisionStoryError):
    def __init__(self, brief_artifact_id: str) -> None:
        super().__init__(f"Unknown decision story draft: {brief_artifact_id}")
        self.brief_artifact_id = brief_artifact_id


class DecisionReportReadError(DecisionStoryError):
    def __init__(self, artifact_id: str, message: str) -> None:
        super().__init__(message)
        self.artifact_id = artifact_id


class DecisionReportMissingError(DecisionReportReadError):
    def __init__(self, artifact_id: str) -> None:
        super().__init__(artifact_id, "The stored decision report file is missing.")


class DecisionReportCorruptError(DecisionReportReadError):
    def __init__(self, artifact_id: str) -> None:
        super().__init__(
            artifact_id,
            "The stored decision report is unreadable or corrupt.",
        )


class DecisionReportIdentityInvalidError(DecisionReportReadError):
    def __init__(self, artifact_id: str) -> None:
        super().__init__(
            artifact_id,
            "The stored decision report failed identity validation.",
        )


class DecisionReportTooLargeError(DecisionReportReadError):
    def __init__(self, artifact_id: str) -> None:
        super().__init__(
            artifact_id,
            "The stored decision report exceeds the read limit.",
        )


class DecisionReportUnavailableError(DecisionReportReadError):
    def __init__(self, artifact_id: str) -> None:
        super().__init__(
            artifact_id,
            "The stored decision report is temporarily unavailable.",
        )


_GATE_VERDICT_SEVERITY = {"pass": 0, "degraded": 1, "rejected": 2}

# Weakest-wins ordering for the report-level confidence label.
_RELIABILITY_SEVERITY = {"high": 0, "medium": 1, "low": 2}


class DecisionReportService:
    def __init__(self, store: ArtifactStore, jobs: JobService) -> None:
        self._store = store
        self._jobs = jobs

    def get_decision_story(self, session_id: str) -> DecisionStoryView:
        """Curation surface: report-eligible findings plus existing drafts.

        Freshness is annotated rather than filtered — stale findings stay
        selectable so the client can choose whether to surface them.
        """
        project_id = self._project_for_run(session_id)
        library = build_investigation_library(self._store, project_id=project_id)
        eligible = [item for item in library.findings if item.finding.report_eligible]
        snapshot = project_run_artifacts(self._store, project_id)
        drafts, warnings = load_synthesis_briefs(
            project_id=project_id, workspace=self._store.root
        )
        return DecisionStoryView(
            session_id=session_id,
            project_id=project_id,
            eligible_findings=[
                DecisionStoryFindingView(
                    artifact_id=item.artifact_id,
                    source_session_id=item.session_id,
                    question=item.finding.question,
                    analytical_reliability=item.finding.analytical_reliability,
                    report_readiness=item.finding.report_readiness,
                    freshness=self._finding_freshness(
                        project_id,
                        item.artifact_id,
                        item.session_id,
                        snapshot,
                    ),
                )
                for item in eligible
            ],
            drafts=[_to_draft_view(item) for item in drafts],
            warnings=[*library.warnings, *warnings],
        )

    def create_draft(
        self,
        session_id: str,
        *,
        finding_artifact_ids: list[str],
        finding_session_ids: dict[str, str] | None = None,
        business_context: str = "",
        idempotency_key: str | None = None,
    ) -> DecisionStoryDraftStarted:
        """Queue drafting a decision story from an explicit finding selection."""
        project_id = self._project_for_run(session_id)
        idempotency_content = {
            "source_session_id": session_id,
            "finding_artifact_ids": list(finding_artifact_ids),
            "finding_session_ids": dict(finding_session_ids or {}),
            "business_context": business_context,
        }
        # Same ordering as the other kinds: an idempotent replay must answer
        # before the busy guard, or a retry would 409 against its own job.
        if idempotency_key:
            existing = self._store.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                self._check_replay_matches(
                    existing, session_id=session_id, kind=SYNTHESIS_JOB_KIND
                )
                self._jobs.assert_idempotent_replay(
                    existing,
                    request_scope=session_id,
                    project_id=project_id,
                    kind=SYNTHESIS_JOB_KIND,
                    content=idempotency_content,
                )
                job = self._jobs.get_job(str(existing["job_id"]))
                return DecisionStoryDraftStarted(
                    session_id=session_id, execution_session_id=job.session_id, job=_to_created(job)
                )
        self._require_story_lane_free(project_id, session_id)
        exact_finding_session_ids = self._require_draftable(
            project_id,
            finding_artifact_ids,
            finding_session_ids=finding_session_ids,
        )
        job = self._jobs.create_synthesis_brief_job(
            generate_synthesis_session_id(session_id),
            project_id=project_id,
            source_session_id=session_id,
            finding_artifact_ids=finding_artifact_ids,
            finding_session_ids=exact_finding_session_ids,
            business_context=business_context,
            idempotency_key=idempotency_key,
            idempotency_content=idempotency_content,
        )
        return DecisionStoryDraftStarted(
            session_id=session_id, execution_session_id=job.session_id, job=_to_created(job)
        )

    def generate_report(
        self,
        session_id: str,
        *,
        brief_artifact_id: str,
        brief_session_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> DecisionReportGenerationStarted:
        """Queue generating a decision report from one persisted draft."""
        project_id = self._project_for_run(session_id)
        idempotency_content = {
            "source_session_id": session_id,
            "brief_artifact_id": brief_artifact_id,
            "brief_session_id": brief_session_id,
        }
        if idempotency_key:
            existing = self._store.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                self._check_replay_matches(
                    existing, session_id=session_id, kind=DECISION_REPORT_JOB_KIND
                )
                self._jobs.assert_idempotent_replay(
                    existing,
                    request_scope=session_id,
                    project_id=project_id,
                    kind=DECISION_REPORT_JOB_KIND,
                    content=idempotency_content,
                )
                job = self._jobs.get_job(str(existing["job_id"]))
                return DecisionReportGenerationStarted(
                    session_id=session_id,
                    brief_artifact_id=brief_artifact_id,
                    execution_session_id=job.session_id,
                    job=_to_created(job),
                )
        self._require_story_lane_free(project_id, session_id)
        exact_brief_session_id = self._require_brief(
            project_id,
            brief_artifact_id,
            brief_session_id=brief_session_id,
        )
        job = self._jobs.create_decision_report_job(
            generate_decision_report_session_id(session_id, brief_artifact_id),
            project_id=project_id,
            source_session_id=session_id,
            brief_artifact_id=brief_artifact_id,
            brief_session_id=exact_brief_session_id,
            idempotency_key=idempotency_key,
            idempotency_content=idempotency_content,
        )
        return DecisionReportGenerationStarted(
            session_id=session_id,
            brief_artifact_id=brief_artifact_id,
            execution_session_id=job.session_id,
            job=_to_created(job),
        )

    def _finding_freshness(
        self,
        project_id: str,
        artifact_id: str,
        artifact_session_id: str,
        snapshot: list[tuple[str, list[Artifact]]],
    ) -> str:
        try:
            return assess_finding_freshness(
                self._store,
                project_id,
                artifact_id,
                finding_session_id=artifact_session_id,
                project_runs=snapshot,
            ).status
        except Exception:  # noqa: BLE001 — a broken assessor must not 500 the page
            logger.warning("Finding freshness failed for %s", artifact_id, exc_info=True)
            return "unverifiable"

    def _require_draftable(
        self,
        project_id: str,
        finding_artifact_ids: list[str],
        *,
        finding_session_ids: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Fail before queueing whatever `_selected_findings` would reject, so
        an impossible selection is a 422 instead of a job that can only fail."""
        requested = list(dict.fromkeys(finding_artifact_ids))
        requested_runs = finding_session_ids or {}
        unexpected = set(requested_runs).difference(requested)
        if unexpected:
            raise DecisionStoryNotDraftableError(
                "Finding run identities were supplied for unselected artifacts: "
                + ", ".join(sorted(unexpected))
            )
        library = build_investigation_library(self._store, project_id=project_id)
        exact: dict[str, StoredValidatedFinding] = {}
        missing: list[str] = []
        for artifact_id in requested:
            session_id = requested_runs.get(artifact_id)
            matches = [
                item
                for item in library.findings
                if item.artifact_id == artifact_id
                and (session_id is None or item.session_id == session_id)
            ]
            if len(matches) != 1:
                missing.append(artifact_id)
            else:
                exact[artifact_id] = matches[0]
        if missing:
            raise DecisionStoryNotDraftableError(
                "Selected findings are unavailable or have ambiguous run identity: "
                + ", ".join(sorted(missing))
            )
        ineligible = [
            item for item in requested if not exact[item].finding.report_eligible
        ]
        if ineligible:
            raise DecisionStoryNotDraftableError(
                "Only report-eligible validated findings can enter a decision "
                "story: " + ", ".join(sorted(ineligible))
            )
        return {artifact_id: exact[artifact_id].session_id for artifact_id in requested}

    def _require_brief(
        self,
        project_id: str,
        brief_artifact_id: str,
        *,
        brief_session_id: str | None = None,
    ) -> str:
        """A generation request must name a real synthesis brief of this
        project; a wrong type or another project's id reads as not-found."""
        try:
            artifact = self._store.get_artifact(
                brief_artifact_id,
                project_id=project_id,
                session_id=brief_session_id,
            )
        except (KeyError, OSError, ValueError) as exc:
            raise DecisionStoryDraftNotFoundError(brief_artifact_id) from exc
        if artifact.type is not ArtifactType.SYNTHESIS_BRIEF:
            raise DecisionStoryDraftNotFoundError(brief_artifact_id)
        return artifact.session_id

    def _require_story_lane_free(self, project_id: str, session_id: str) -> None:
        """One decision-story job per source run at a time. Two generations
        would race on the same project-level report the reader picks by
        recency, and the source run itself must be idle too."""
        active = self._store.find_active_job_for_lane(session_id)
        if active is not None:
            raise DecisionStoryBusyError(session_id, str(active["job_id"]))
        for job in self._store.list_active_jobs():
            if (
                str(job["kind"]) in DECISION_STORY_JOB_KINDS
                and str(job["project_id"]) == project_id
                and str(job["session_id"]).rsplit("_", 1)[-1]
                in self._derived_suffixes(project_id, session_id)
            ):
                raise DecisionStoryBusyError(session_id, str(job["job_id"]))

    def _derived_suffixes(self, project_id: str, session_id: str) -> set[str]:
        """Deterministic suffixes this source run could have minted. Report ids
        also hash the brief, so every existing draft contributes one."""
        suffixes = {generate_synthesis_session_id(session_id).rsplit("_", 1)[-1]}
        drafts, _warnings = load_synthesis_briefs(
            project_id=project_id, workspace=self._store.root
        )
        suffixes |= {
            generate_decision_report_session_id(session_id, item.artifact_id).rsplit("_", 1)[-1]
            for item in drafts
        }
        return suffixes

    def _check_replay_matches(self, job_row: dict, *, session_id: str, kind: str) -> None:
        """A replay is only legitimate for this kind in this run's own project,
        derived from this very run (mirrors report generation)."""
        job_id = str(job_row["job_id"])
        conflict = JobConflictError(
            job_id,
            f"Idempotency key already used by job {job_id} from a "
            "different project, kind or source run.",
        )
        run_row = self._store.get_session_index_row(session_id)
        if run_row is None:
            raise conflict
        project_id = str(run_row["project_id"])
        if (
            str(job_row["kind"]) != kind
            or str(job_row["project_id"]) != project_id
            or str(job_row["session_id"]).rsplit("_", 1)[-1]
            not in self._derived_suffixes(project_id, session_id)
        ):
            raise conflict

    def get_decision_report(self, session_id: str) -> DecisionReportView:
        project_id = self._project_for_run(session_id)
        found = self._latest_decision_report(project_id)
        if found is None:
            return DecisionReportView(session_id=session_id, status="none")
        artifact, report = found

        freshness = self._freshness(
            project_id,
            artifact.id,
            report_session_id=artifact.session_id,
        )
        findings = self._source_findings(
            project_id,
            report.source_finding_artifact_ids,
            finding_session_ids=report.source_finding_session_ids,
        )
        return DecisionReportView(
            session_id=session_id,
            status="available",
            artifact_id=artifact.id,
            report_session_id=artifact.session_id,
            report_id=report.report_id,
            brief_id=report.brief_id,
            title=report.title,
            generated_at=artifact.created_at,
            scqa=SCQAView(
                situation=report.scqa.situation,
                complication=report.scqa.complication,
                question=report.scqa.question,
                answer=report.scqa.answer,
            ),
            sections=[
                DecisionReportSectionView(
                    title=section.title,
                    body=section.body,
                    finding_artifact_ids=list(section.finding_artifact_ids),
                )
                for section in report.sections
            ],
            limitations=list(report.limitations),
            investigation_gaps=list(report.investigation_gaps),
            meta_insight=(
                None
                if report.meta_insight is None or report.meta_insight.is_empty
                else MetaInsightView(
                    commonality_statements=list(report.meta_insight.commonality_statements),
                    exception_statements=list(report.meta_insight.exception_statements),
                )
            ),
            candidate_decisions=_candidate_decisions(findings),
            evidence_refs=self._evidence_refs(project_id, findings),
            source_finding_artifact_ids=list(report.source_finding_artifact_ids),
            granted_evidence_artifact_ids=list(report.granted_evidence_artifact_ids),
            report_readiness=report.report_readiness,
            narrative_status=report.narrative_status,
            narrative_fallback_reason=report.narrative_fallback_reason,
            freshness=freshness,
            publication_status="published",
            gate_verdict=self._gate_verdict(project_id, session_id),
            confidence_label=_confidence_label(findings),
            # Same rule as the legacy UI: only a fresh report may be exported.
            export_available=freshness.status == "fresh",
        )

    def _latest_decision_report(
        self, project_id: str
    ) -> tuple[Artifact, DecisionReport] | None:
        newest: tuple[Artifact, DecisionReport] | None = None
        for row in self._store.project_artifact_index_rows(
            project_id, ArtifactType.DECISION_REPORT.value
        ):
            if INTERNAL_SESSION_MARKER in str(row["session_id"]):
                continue
            artifact = self._read_contained(row["path"], expected_id=str(row["artifact_id"]))
            try:
                report = DecisionReport.model_validate(artifact.payload)
            except ValidationError as exc:
                raise DecisionReportCorruptError(artifact.id) from exc
            if newest is None or artifact.created_at > newest[0].created_at:
                newest = (artifact, report)
        return newest

    def _read_contained(self, path: Path, *, expected_id: str) -> Artifact:
        """Read an indexed artifact with the workspace containment, size and
        envelope-identity guards the other read services apply."""
        try:
            if not path.resolve().is_relative_to(self._store.root.resolve()):
                raise DecisionReportIdentityInvalidError(expected_id)
            if path.stat().st_size > MAX_DECISION_REPORT_BYTES:
                raise DecisionReportTooLargeError(expected_id)
            artifact = Artifact.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DecisionReportMissingError(expected_id) from exc
        except (UnicodeError, ValidationError, ValueError) as exc:
            raise DecisionReportCorruptError(expected_id) from exc
        except OSError as exc:
            raise DecisionReportUnavailableError(expected_id) from exc
        if artifact.id != expected_id or artifact.type is not ArtifactType.DECISION_REPORT:
            raise DecisionReportIdentityInvalidError(expected_id)
        return artifact.model_copy(
            update={
                "payload": relativize_workspace_paths(artifact.payload, self._store.root)
            }
        )

    def _freshness(
        self,
        project_id: str,
        artifact_id: str,
        *,
        report_session_id: str | None = None,
    ) -> DecisionReportFreshnessView:
        try:
            freshness = assess_decision_report_freshness(
                self._store,
                project_id,
                artifact_id,
                report_session_id=report_session_id,
            )
        except Exception:  # noqa: BLE001 — freshness must fail closed, not 500
            logger.warning(
                "Decision report freshness failed for %s", artifact_id, exc_info=True
            )
            return DecisionReportFreshnessView(
                status="unverifiable",
                reasons=["Decision report freshness could not be assessed."],
            )
        return DecisionReportFreshnessView(
            status=freshness.status, reasons=list(freshness.reasons)
        )

    def _source_findings(
        self,
        project_id: str,
        finding_ids: list[str],
        *,
        finding_session_ids: dict[str, str] | None = None,
    ) -> list[tuple[str, str, ValidatedFinding]]:
        findings: list[tuple[str, str, ValidatedFinding]] = []
        for finding_id in dict.fromkeys(finding_ids):
            try:
                artifact = self._store.get_artifact(
                    finding_id,
                    project_id=project_id,
                    session_id=(finding_session_ids or {}).get(finding_id),
                )
            except (KeyError, OSError, ValueError):
                continue
            if artifact.type is not ArtifactType.VALIDATED_FINDING:
                continue
            try:
                findings.append(
                    (
                        finding_id,
                        artifact.session_id,
                        ValidatedFinding.model_validate(artifact.payload),
                    )
                )
            except ValidationError:
                continue
        return findings

    def _evidence_refs(
        self,
        project_id: str,
        findings: list[tuple[str, str, ValidatedFinding]],
    ) -> list[DecisionEvidenceRefView]:
        refs: dict[str, DecisionEvidenceRefView] = {}
        for _finding_id, finding_session_id, finding in findings:
            for statement in finding.findings:
                for evidence in statement.evidence:
                    artifact_id = evidence.artifact_id
                    if not artifact_id or artifact_id in refs:
                        continue
                    refs[artifact_id] = DecisionEvidenceRefView(
                        artifact_id=artifact_id,
                        kind=str(evidence.kind),
                        locator=evidence.locator,
                        session_id=self._navigable_run(
                            artifact_id,
                            project_id,
                            session_id=finding.source_artifact_session_ids.get(
                                artifact_id,
                                finding_session_id,
                            ),
                        ),
                    )
        return list(refs.values())

    def _navigable_run(
        self,
        artifact_id: str,
        project_id: str,
        *,
        session_id: str | None = None,
    ) -> str | None:
        try:
            artifact = self._store.get_artifact(
                artifact_id,
                project_id=project_id,
                session_id=session_id,
            )
        except (KeyError, OSError, ValueError):
            return None
        return None if INTERNAL_SESSION_MARKER in artifact.session_id else artifact.session_id

    def _gate_verdict(self, project_id: str, session_id: str) -> str | None:
        """Worst report-gate verdict recorded for the viewed run.

        Standalone ReportAudit artifacts win; the audit embedded in a
        ReportBundle is only a fallback for runs that persisted no standalone
        one (the same precedence core.session_metrics uses)."""
        audits = [
            artifact.payload
            for artifact in self._store.list_indexed_artifacts(
                project_id=project_id,
                session_id=session_id,
                artifact_types=(ArtifactType.REPORT_AUDIT,),
            )
        ]
        if not audits:
            audits = [
                audit
                for artifact in self._store.list_indexed_artifacts(
                    project_id=project_id,
                    session_id=session_id,
                    artifact_types=(ArtifactType.REPORT_BUNDLE,),
                )
                for audit in [artifact.payload.get("audit")]
                if isinstance(audit, dict)
            ]
        verdicts = [
            audit["gate_verdict"]
            for audit in audits
            if isinstance(audit.get("gate_verdict"), str)
            and audit["gate_verdict"] in _GATE_VERDICT_SEVERITY
        ]
        if not verdicts:
            return None
        return max(verdicts, key=lambda verdict: _GATE_VERDICT_SEVERITY[verdict])

    def _project_for_run(self, session_id: str) -> str:
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        row = self._store.get_session_index_row(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        return str(row["project_id"])


def _to_created(job: JobStatus) -> JobCreated:
    return JobCreated(
        job_id=job.job_id,
        session_id=job.session_id,
        status=job.status,
        events_url=job.events_url,
    )


def _to_draft_view(item: StoredSynthesisBrief) -> DecisionStoryDraftView:
    """Shape one persisted synthesis brief for the curation surface."""
    brief = item.brief
    return DecisionStoryDraftView(
        artifact_id=item.artifact_id,
        session_id=item.session_id,
        created_at=item.created_at,
        brief_id=brief.brief_id,
        headline=brief.headline,
        decision_context=brief.decision_context,
        storyline=[
            DecisionStoryBeatView(title=beat.title, body=beat.body)
            for beat in brief.storyline
        ],
        limitations=list(brief.limitations),
        investigation_gaps=list(brief.investigation_gaps),
        business_context=brief.business_context,
        report_eligible=brief.report_eligible,
        report_readiness=brief.report_readiness,
        selected_finding_artifact_ids=list(brief.selected_finding_artifact_ids),
    )


def _candidate_decisions(
    findings: list[tuple[str, str, ValidatedFinding]],
) -> list[CandidateDecisionView]:
    return [
        CandidateDecisionView(
            finding_artifact_id=finding_id,
            question=finding.question,
            decision_action=finding.decision_action.strip(),
            decision_readiness=finding.decision_readiness,
            analytical_reliability=finding.analytical_reliability,
            report_readiness=finding.report_readiness,
        )
        for finding_id, _session_id, finding in findings
        if finding.decision_action.strip()
    ]


def _confidence_label(
    findings: list[tuple[str, str, ValidatedFinding]],
) -> str | None:
    ratings = [
        finding.analytical_reliability
        for _artifact_id, _session_id, finding in findings
    ]
    if not ratings:
        return None
    return max(ratings, key=lambda rating: _RELIABILITY_SEVERITY.get(rating, 1))
