"""Investigation governance use cases.

One question card becomes an investigation through four gates, each of which
this service exposes as its own request:

  build   → `investigation_plan` job; deterministic, spends no model budget
  decide  → approve / reject, bound to the plan's content fingerprint
  execute → `investigation_execute` job; runs the approved method, spends budget
  loop    → `macro_loop` job; depth >=2 only, runs follow-up rounds by itself

Every budget-spending step goes through ApprovalService twice (prepare then
consume) and lands on its own derived run, so a failure never rewrites the
source run's status. The plans themselves live on the orchestrator's own
`investigation_*` run; approvals and findings land there beside them.
"""

from __future__ import annotations

import json
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from eda_platform.application.dto import (
    InvestigationDecisionPrepared,
    InvestigationDecisionRecorded,
    InvestigationExecutionPrepared,
    InvestigationExecutionStarted,
    InvestigationGateView,
    InvestigationPlanBuildStarted,
    InvestigationPlanView,
    InvestigationsView,
    JobCreated,
    JobStatus,
    MacroLoopPrepared,
    MacroLoopRoundView,
    MacroLoopStarted,
    MacroLoopView,
)
from eda_platform.application.services.approval_service import (
    ApprovalService,
    payload_digest,
)
from eda_platform.application.services.job_service import JobConflictError, JobService
from eda_platform.application.services.question_service import latest_candidate_set
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.application.services.settings_service import (
    DEEP_INVESTIGATION_DEPTH,
    MACRO_LOOP_DEPTH,
)
from eda_platform.core.ids import INTERNAL_SESSION_MARKER, stable_hash
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.investigation_orchestrator import (
    _plan_fingerprint,
    preauthorize_macro_loop,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.investigations import (
    InvestigationApproval,
    InvestigationPlan,
    InvestigationRecord,
    ValidatedFinding,
)
from eda_platform.schemas.loop import DEPTH_PROFILES, LoopLedger

APPROVAL_KIND_DECISION = "investigation_approve"
APPROVAL_KIND_EXECUTE = "investigation_execute"
APPROVAL_KIND_MACRO_LOOP = "macro_loop"

PLAN_JOB_KIND = "investigation_plan"
EXECUTE_JOB_KIND = "investigation_execute"
MACRO_LOOP_JOB_KIND = "macro_loop"

PLAN_SESSION_PREFIX = "ipsess_"
EXECUTE_SESSION_PREFIX = "ixsess_"
MACRO_LOOP_SESSION_PREFIX = "mlsess_"

MAX_PLAN_QUESTIONS = 20
# Marker tool the orchestrator writes onto a plan that may run deep probes.
DEEP_MARKER_TOOL = "llm_probe_planner"

_EXECUTION_LANE_KINDS = frozenset({EXECUTE_JOB_KIND, MACRO_LOOP_JOB_KIND})


def _lane_of(session_id: str) -> str:
    """The lane segment of a derived run id minted below (second to last)."""
    parts = session_id.split("_")
    return parts[-2] if len(parts) >= 2 else ""


def generate_plan_build_session_id(source_session_id: str, question_ids: list[str]) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    lane = stable_hash({"plan_source_session_id": source_session_id}, length=6)
    request = stable_hash({"question_ids": sorted(set(question_ids))}, length=6)
    return f"{PLAN_SESSION_PREFIX}{stamp}_{lane}_{request}"


def plan_build_lane(source_session_id: str) -> str:
    return stable_hash({"plan_source_session_id": source_session_id}, length=6)


def generate_execute_session_id(plan_session_id: str, plan_ids: list[str]) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    request = stable_hash({"plan_ids": sorted(set(plan_ids))}, length=6)
    return f"{EXECUTE_SESSION_PREFIX}{stamp}_{execution_lane(plan_session_id)}_{request}"


def generate_macro_loop_session_id(plan_session_id: str, depth: int) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    request = stable_hash({"macro_loop": plan_session_id, "depth": depth}, length=6)
    return f"{MACRO_LOOP_SESSION_PREFIX}{stamp}_{execution_lane(plan_session_id)}_{request}"


def execution_lane(plan_session_id: str) -> str:
    """One lane per plan run: executing plans and looping over them both
    read-modify-write that run's artifacts, so they must not run concurrently."""
    return stable_hash({"plan_session_id": plan_session_id}, length=6)


class InvestigationServiceError(Exception):
    pass


class InvestigationNotFoundError(InvestigationServiceError):
    def __init__(self, plan_id: str, session_id: str) -> None:
        super().__init__(f"Investigation plan not found for run {session_id}: {plan_id}")
        self.plan_id = plan_id
        self.session_id = session_id


class InvestigationNotDecidableError(InvestigationServiceError):
    def __init__(self, plan_id: str, reason: str) -> None:
        super().__init__(f"Investigation plan {plan_id} cannot be decided: {reason}")
        self.plan_id = plan_id


class InvestigationNotExecutableError(InvestigationServiceError):
    def __init__(self, plan_id: str, reason: str) -> None:
        super().__init__(f"Investigation plan {plan_id} cannot be executed: {reason}")
        self.plan_id = plan_id


class InvestigationSourceChangedError(InvestigationServiceError):
    """The plan changed since prepare; the approval no longer binds it."""

    def __init__(self, plan_id: str) -> None:
        super().__init__(
            f"Investigation plan {plan_id} changed since the approval was "
            "prepared; prepare again and approve the fresh plan."
        )
        self.plan_id = plan_id


class InvestigationRunBusyError(InvestigationServiceError):
    def __init__(self, session_id: str, job_id: str) -> None:
        super().__init__(
            f"Run {session_id} has an active job ({job_id}); wait for it to finish "
            "before changing its investigations."
        )
        self.session_id = session_id
        self.job_id = job_id


class MacroLoopNotAuthorizedError(InvestigationServiceError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"The macro loop is not authorized: {reason}")


class InvestigationValidationError(InvestigationServiceError):
    pass


class _PlanRecord:
    """One plan artifact with everything the run knows about its lifecycle."""

    __slots__ = ("approval", "artifact", "finding", "plan", "record", "session_id")

    def __init__(self, session_id: str, artifact: Artifact, plan: InvestigationPlan) -> None:
        self.session_id = session_id
        self.artifact = artifact
        self.plan = plan
        self.approval: InvestigationApproval | None = None
        self.record: InvestigationRecord | None = None
        self.finding: ValidatedFinding | None = None

    @property
    def decided(self) -> bool:
        return self.approval is not None

    @property
    def has_outcome(self) -> bool:
        """An execution outcome exists. A rejection record is a decision, not
        an outcome, so it must not read as executed."""
        if self.finding is not None:
            return True
        return self.record is not None and self.record.status != "rejected"

    @property
    def status(self) -> str:
        if self.has_outcome:
            return "executed"
        if self.approval is not None:
            return "approved" if self.approval.decision == "approved" else "rejected"
        return "pending"


class InvestigationService:
    def __init__(
        self,
        store: ArtifactStore,
        approvals: ApprovalService,
        jobs: JobService,
    ) -> None:
        self._store = store
        self._approvals = approvals
        self._jobs = jobs

    # Read path

    def list_investigations(
        self, session_id: str, *, analysis_depth: int = 0
    ) -> InvestigationsView:
        project_id = self._project_for_run(session_id)
        records = self._plan_records(project_id, session_id)
        plan_session_ids = list(dict.fromkeys(record.session_id for record in records))
        return InvestigationsView(
            session_id=session_id,
            project_id=project_id,
            analysis_depth=analysis_depth,
            deep_investigation_enabled=analysis_depth >= DEEP_INVESTIGATION_DEPTH,
            macro_loop_authorized=analysis_depth >= MACRO_LOOP_DEPTH,
            plans=[_to_plan_view(record) for record in records],
            macro_loops=[
                view
                for plan_session_id in plan_session_ids
                for view in self._macro_loops(project_id, plan_session_id)
            ],
        )

    # Build

    def build_plans(
        self,
        session_id: str,
        *,
        question_ids: list[str],
        deep: bool = False,
        analysis_depth: int = 0,
        idempotency_key: str | None = None,
    ) -> InvestigationPlanBuildStarted:
        """Queue plan building for the selected questions.

        Deep planning follows the product rule: the explicit checkbox OR a
        thinking level of Deep or higher.
        """
        project_id = self._project_for_run(session_id)
        selected = list(dict.fromkeys(question_ids))
        deep_requested = bool(deep) or analysis_depth >= DEEP_INVESTIGATION_DEPTH
        execution_session_id = generate_plan_build_session_id(session_id, selected)
        idempotency_content = {
            "source_session_id": session_id,
            "question_ids": selected,
            "deep": deep_requested,
        }
        # Same ordering as the other job kinds: an idempotent replay must answer
        # before the busy guard, or a retry would 409 against its own job.
        if idempotency_key:
            existing = self._store.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                self._check_build_replay_matches(existing, session_id=session_id)
                self._jobs.assert_idempotent_replay(
                    existing,
                    request_scope=session_id,
                    project_id=project_id,
                    kind=PLAN_JOB_KIND,
                    content=idempotency_content,
                )
                return self._build_replayed(session_id, existing)
        if not selected:
            raise InvestigationValidationError("Select at least one question to plan.")
        if len(selected) > MAX_PLAN_QUESTIONS:
            raise InvestigationValidationError(
                f"Plan at most {MAX_PLAN_QUESTIONS} questions at a time."
            )
        self._require_source_session_idle(session_id)
        self._require_plan_lane_free(project_id, session_id)
        self._require_known_questions(project_id, session_id, selected)
        job = self._jobs.create_investigation_plan_job(
            execution_session_id,
            project_id=project_id,
            source_session_id=session_id,
            question_ids=selected,
            deep=deep_requested,
            idempotency_key=idempotency_key,
            idempotency_content=idempotency_content,
        )
        return InvestigationPlanBuildStarted(
            session_id=session_id,
            execution_session_id=job.session_id,
            question_ids=selected,
            deep=deep_requested,
            job=_to_created(job),
        )

    # Decide

    def prepare_decision(
        self, session_id: str, plan_id: str, *, decision: str, reason: str = ""
    ) -> InvestigationDecisionPrepared:
        if decision not in {"approved", "rejected"}:
            raise InvestigationValidationError(
                "decision must be 'approved' or 'rejected'."
            )
        project_id = self._project_for_run(session_id)
        record = self._require_plan(project_id, session_id, plan_id)
        self._require_decidable(record, decision)
        fingerprint = _plan_fingerprint(record.plan)
        digest, approval_token, expires_at = self._approvals.register(
            kind=APPROVAL_KIND_DECISION,
            session_id=session_id,
            project_id=project_id,
            action={
                "type": "investigation_decision",
                "source_session_id": session_id,
                "plan_id": plan_id,
                "decision": decision,
                "plan_fingerprint": fingerprint,
            },
            payload={
                "project_id": project_id,
                "source_session_id": session_id,
                "plan_session_id": record.session_id,
                "plan_id": plan_id,
                "decision": decision,
                "reason": reason,
                "plan_fingerprint": fingerprint,
            },
        )
        return InvestigationDecisionPrepared(
            session_id=session_id,
            plan_id=plan_id,
            plan_session_id=record.session_id,
            decision=decision,
            reason=reason,
            action_hash=digest,
            approval_token=approval_token,
            expires_at=expires_at,
            plan=_to_plan_view(record),
        )

    def decide(
        self,
        session_id: str,
        plan_id: str,
        *,
        decision: str,
        action_hash: str,
        approval_token: str,
    ) -> InvestigationDecisionRecorded:
        """Consume the approval and persist the driver's own decision artifact."""
        from eda_platform.drivers.investigation_orchestrator import approve_plan, reject_plan

        project_id = self._project_for_run(session_id)
        def validate(payload: dict[str, Any]) -> tuple[Any, str]:
            if str(payload.get("plan_id", "")) != plan_id:
                raise InvestigationValidationError(
                    "The approval was prepared for a different plan than the request path."
                )
            if str(payload.get("decision", "")) != decision:
                raise InvestigationValidationError(
                    "The approval was prepared for a different decision than this request."
                )
            record = self._require_plan(project_id, session_id, plan_id)
            if _plan_fingerprint(record.plan) != str(
                payload.get("plan_fingerprint", "")
            ):
                raise InvestigationSourceChangedError(plan_id)
            self._require_decidable(record, decision)
            return record, str(payload.get("reason", "")) or _default_reason(decision)

        _payload, (record, reason) = self._approvals.validate_then_consume(
            action_hash,
            kind=APPROVAL_KIND_DECISION,
            session_id=session_id,
            generation=approval_token,
            validate=validate,
        )
        with self._approvals.compensate_on_failure(action_hash, session_id=session_id):
            if decision == "approved":
                artifact = approve_plan(
                    project_id=project_id,
                    plan_session_id=record.session_id,
                    plan_id=plan_id,
                    workspace=self._store.root,
                    reason=reason,
                )
            else:
                artifact = reject_plan(
                    project_id=project_id,
                    plan_session_id=record.session_id,
                    plan_id=plan_id,
                    workspace=self._store.root,
                    reason=reason,
                )[0]
        refreshed = self._require_plan(project_id, session_id, plan_id)
        return InvestigationDecisionRecorded(
            session_id=session_id,
            plan_id=plan_id,
            decision=decision,
            approval_artifact_id=artifact.id,
            plan=_to_plan_view(refreshed),
        )

    # Execute

    def prepare_execution(
        self, session_id: str, *, plan_ids: list[str], llm: str = "env"
    ) -> InvestigationExecutionPrepared:
        project_id = self._project_for_run(session_id)
        selected = list(dict.fromkeys(plan_ids))
        if not selected:
            raise InvestigationValidationError("Select at least one approved plan.")
        records = [self._require_plan(project_id, session_id, plan_id) for plan_id in selected]
        plan_session_ids = {record.session_id for record in records}
        if len(plan_session_ids) != 1:
            raise InvestigationValidationError(
                "Selected plans must come from the same plan run."
            )
        plan_session_id = plan_session_ids.pop()
        for record in records:
            self._require_executable(record)
        self._require_source_session_idle(session_id)
        self._require_execution_lane_free(project_id, plan_session_id)
        fingerprints = {
            record.artifact.id: _plan_fingerprint(record.plan) for record in records
        }
        digest, approval_token, expires_at = self._approvals.register(
            kind=APPROVAL_KIND_EXECUTE,
            session_id=session_id,
            project_id=project_id,
            action={
                "type": "investigation_execute",
                "source_session_id": session_id,
                "plan_session_id": plan_session_id,
                "plan_fingerprints": fingerprints,
            },
            payload={
                "project_id": project_id,
                "source_session_id": session_id,
                "plan_session_id": plan_session_id,
                "plan_ids": selected,
                "plan_fingerprints": fingerprints,
                # The approval binds the LLM mode: execute runs what was
                # approved, never what the execute request claims.
                "llm": llm,
            },
        )
        return InvestigationExecutionPrepared(
            session_id=session_id,
            plan_session_id=plan_session_id,
            plan_ids=selected,
            action_hash=digest,
            approval_token=approval_token,
            expires_at=expires_at,
            llm_mode=llm,
            plans=[_to_plan_view(record) for record in records],
        )

    def execute(
        self,
        session_id: str,
        *,
        action_hash: str,
        approval_token: str,
        payload_policy: str | None = None,
        llm_env: dict[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> InvestigationExecutionStarted:
        return self._approvals.run_idempotent_producer(
            action_hash,
            session_id=session_id,
            idempotency_key=idempotency_key,
            operation=lambda deadline: self._execute_once(
                session_id,
                action_hash=action_hash,
                approval_token=approval_token,
                payload_policy=payload_policy,
                llm_env=llm_env,
                idempotency_key=idempotency_key,
                contention_deadline=deadline,
            ),
        )

    def _execute_once(
        self,
        session_id: str,
        *,
        action_hash: str,
        approval_token: str,
        payload_policy: str | None,
        llm_env: dict[str, str] | None,
        idempotency_key: str | None,
        contention_deadline: float,
    ) -> InvestigationExecutionStarted:
        project_id = self._project_for_run(session_id)
        if idempotency_key:
            existing = self._store.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                replay_payload, replay_payload_digest = (
                    self._approval_payload_for_idempotency(
                        existing, action_hash, session_id=session_id
                    )
                )
                self._jobs.assert_idempotent_replay(
                    existing,
                    request_scope=session_id,
                    project_id=project_id,
                    kind=EXECUTE_JOB_KIND,
                    content={
                        "source_session_id": session_id,
                        "action_hash": action_hash,
                        "approval_payload_digest": replay_payload_digest,
                        "payload_policy": payload_policy,
                    },
                    env=(
                        llm_env
                        if str(replay_payload.get("llm", "env")) == "env"
                        else None
                    ),
                )
                plan_session_id, plan_ids = self._check_execute_replay_matches(
                    existing, session_id=session_id, action_hash=action_hash
                )
                return self._execute_replayed(session_id, plan_session_id, plan_ids, existing)
        def validate(
            payload: dict[str, Any],
        ) -> tuple[str, list[str], dict[str, Any]]:
            plan_session_id = str(payload.get("plan_session_id", ""))
            plan_ids = [str(item) for item in payload.get("plan_ids", [])]
            approved = payload.get("plan_fingerprints")
            approved = approved if isinstance(approved, dict) else {}
            records = [self._require_plan(project_id, session_id, plan_id) for plan_id in plan_ids]
            for record in records:
                if _plan_fingerprint(record.plan) != str(
                    approved.get(record.artifact.id, "")
                ):
                    raise InvestigationSourceChangedError(record.artifact.id)
                self._require_executable(record)
            self._require_source_session_idle(session_id)
            self._require_execution_lane_free(project_id, plan_session_id)
            return plan_session_id, plan_ids, approved

        payload, validated = self._approvals.validate_then_consume(
            action_hash,
            kind=APPROVAL_KIND_EXECUTE,
            session_id=session_id,
            generation=approval_token,
            validate=validate,
            idempotency_key=idempotency_key,
            deadline=contention_deadline,
        )
        with self._approvals.compensate_on_failure(action_hash, session_id=session_id):
            plan_session_id, plan_ids, approved = validated
            idempotency_content = {
                "source_session_id": session_id,
                "action_hash": action_hash,
                "approval_payload_digest": payload_digest(payload),
                "payload_policy": payload_policy,
            }
            job = self._jobs.create_investigation_execute_job(
                generate_execute_session_id(plan_session_id, plan_ids),
                project_id=project_id,
                source_session_id=session_id,
                plan_session_id=plan_session_id,
                plan_ids=plan_ids,
                plan_fingerprints={key: str(value) for key, value in approved.items()},
                llm=str(payload.get("llm", "env")),
                payload_policy=payload_policy,
                llm_env=llm_env,
                idempotency_key=idempotency_key,
                idempotency_content=idempotency_content,
            )
        return InvestigationExecutionStarted(
            session_id=session_id,
            plan_session_id=plan_session_id,
            execution_session_id=job.session_id,
            plan_ids=plan_ids,
            job=_to_created(job),
        )

    # Macro loop

    def prepare_macro_loop(
        self, session_id: str, *, plan_session_id: str, analysis_depth: int, llm: str = "env"
    ) -> MacroLoopPrepared:
        project_id = self._project_for_run(session_id)
        depth = int(analysis_depth)
        profile = DEPTH_PROFILES.get(depth)
        if profile is None or profile.rounds == 0:
            raise MacroLoopNotAuthorizedError(
                "set the thinking level to Ultra before running follow-up rounds"
            )
        records = [
            record
            for record in self._plan_records(project_id, session_id)
            if record.session_id == plan_session_id
        ]
        if not records:
            raise InvestigationNotFoundError(plan_session_id, session_id)
        if not any(record.has_outcome for record in records):
            raise MacroLoopNotAuthorizedError(
                "the macro loop starts from this plan run's own executed findings; "
                "run the approved plans first"
            )
        self._require_source_session_idle(session_id)
        self._require_execution_lane_free(project_id, plan_session_id)
        digest, approval_token, expires_at = self._approvals.register(
            kind=APPROVAL_KIND_MACRO_LOOP,
            session_id=session_id,
            project_id=project_id,
            action={
                "type": "macro_loop",
                "source_session_id": session_id,
                "plan_session_id": plan_session_id,
                "depth": depth,
                "rounds_cap": profile.rounds,
            },
            payload={
                "project_id": project_id,
                "source_session_id": session_id,
                "plan_session_id": plan_session_id,
                "depth": depth,
                "rounds_cap": profile.rounds,
                "llm": llm,
            },
        )
        return MacroLoopPrepared(
            session_id=session_id,
            plan_session_id=plan_session_id,
            action_hash=digest,
            approval_token=approval_token,
            expires_at=expires_at,
            depth=depth,
            rounds_cap=profile.rounds,
            questions_per_round=profile.per_round_questions,
            llm_mode=llm,
        )

    def start_macro_loop(
        self,
        session_id: str,
        *,
        action_hash: str,
        approval_token: str,
        payload_policy: str | None = None,
        llm_env: dict[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> MacroLoopStarted:
        """Consume the approval, write the §8.3 pre-authorization, queue the loop.

        The pre-authorization artifact is what `run_macro_loop` verifies before
        it generates a single follow-up question, and it is written only here —
        after a human approved this exact plan run and depth.
        """
        return self._approvals.run_idempotent_producer(
            action_hash,
            session_id=session_id,
            idempotency_key=idempotency_key,
            operation=lambda deadline: self._start_macro_loop_once(
                session_id,
                action_hash=action_hash,
                approval_token=approval_token,
                payload_policy=payload_policy,
                llm_env=llm_env,
                idempotency_key=idempotency_key,
                contention_deadline=deadline,
            ),
        )

    def _start_macro_loop_once(
        self,
        session_id: str,
        *,
        action_hash: str,
        approval_token: str,
        payload_policy: str | None,
        llm_env: dict[str, str] | None,
        idempotency_key: str | None,
        contention_deadline: float,
    ) -> MacroLoopStarted:
        project_id = self._project_for_run(session_id)
        if idempotency_key:
            existing = self._store.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                replay_payload, replay_payload_digest = (
                    self._approval_payload_for_idempotency(
                        existing, action_hash, session_id=session_id
                    )
                )
                self._jobs.assert_idempotent_replay(
                    existing,
                    request_scope=session_id,
                    project_id=project_id,
                    kind=MACRO_LOOP_JOB_KIND,
                    content={
                        "source_session_id": session_id,
                        "action_hash": action_hash,
                        "approval_payload_digest": replay_payload_digest,
                        "payload_policy": payload_policy,
                    },
                    env=(
                        llm_env
                        if str(replay_payload.get("llm", "env")) == "env"
                        else None
                    ),
                )
                plan_session_id, depth = self._check_macro_replay_matches(
                    existing, session_id=session_id, action_hash=action_hash
                )
                return self._macro_replayed(session_id, plan_session_id, depth, existing)
        def validate(payload: dict[str, Any]) -> tuple[str, int, Any]:
            plan_session_id = str(payload.get("plan_session_id", ""))
            depth = int(payload.get("depth", 0))
            profile = DEPTH_PROFILES.get(depth)
            if profile is None or profile.rounds == 0:
                raise MacroLoopNotAuthorizedError(
                    "the approved depth runs no follow-up rounds"
                )
            self._require_source_session_idle(session_id)
            self._require_execution_lane_free(project_id, plan_session_id)
            return plan_session_id, depth, profile

        payload, validated = self._approvals.validate_then_consume(
            action_hash,
            kind=APPROVAL_KIND_MACRO_LOOP,
            session_id=session_id,
            generation=approval_token,
            validate=validate,
            idempotency_key=idempotency_key,
            deadline=contention_deadline,
        )
        with self._approvals.compensate_on_failure(action_hash, session_id=session_id):
            plan_session_id, depth, profile = validated
            idempotency_content = {
                "source_session_id": session_id,
                "action_hash": action_hash,
                "approval_payload_digest": payload_digest(payload),
                "payload_policy": payload_policy,
            }
            preauthorize_macro_loop(
                project_id=project_id,
                plan_session_id=plan_session_id,
                workspace=self._store.root,
                depth=depth,
            )
            job = self._jobs.create_macro_loop_job(
                generate_macro_loop_session_id(plan_session_id, depth),
                project_id=project_id,
                source_session_id=session_id,
                plan_session_id=plan_session_id,
                depth=depth,
                llm=str(payload.get("llm", "env")),
                payload_policy=payload_policy,
                llm_env=llm_env,
                idempotency_key=idempotency_key,
                idempotency_content=idempotency_content,
            )
        return MacroLoopStarted(
            session_id=session_id,
            plan_session_id=plan_session_id,
            execution_session_id=job.session_id,
            depth=depth,
            rounds_cap=profile.rounds,
            job=_to_created(job),
        )

    # Internals

    def _plan_records(self, project_id: str, session_id: str) -> list[_PlanRecord]:
        """Every plan whose source run is ``session_id``, newest plan run first."""
        records: list[_PlanRecord] = []
        by_run: dict[str, list[_PlanRecord]] = {}
        for row in self._store.project_artifact_index_rows(
            project_id, ArtifactType.INVESTIGATION_PLAN.value
        ):
            plan_session_id = str(row["session_id"])
            try:
                artifact = self._store.get_artifact(
                    str(row["artifact_id"]), project_id=project_id, session_id=plan_session_id
                )
                plan = InvestigationPlan.model_validate(artifact.payload)
            except (KeyError, OSError, ValidationError):
                continue
            if plan.source_session_id != session_id:
                continue
            record = _PlanRecord(plan_session_id, artifact, plan)
            records.append(record)
            by_run.setdefault(plan_session_id, []).append(record)
        for plan_session_id, run_records in by_run.items():
            self._attach_lifecycle(project_id, plan_session_id, run_records)
        records.sort(key=lambda item: (item.session_id, item.plan.investigation_id), reverse=True)
        return records

    def _attach_lifecycle(
        self, project_id: str, plan_session_id: str, records: list[_PlanRecord]
    ) -> None:
        by_investigation = {record.plan.investigation_id: record for record in records}
        for artifact in self._store.list_indexed_artifacts(
            project_id=project_id,
            session_id=plan_session_id,
            artifact_types=(
                ArtifactType.INVESTIGATION_APPROVAL,
                ArtifactType.INVESTIGATION_RECORD,
                ArtifactType.VALIDATED_FINDING,
            ),
        ):
            if artifact.type is ArtifactType.INVESTIGATION_APPROVAL:
                with suppress(ValidationError):
                    approval = InvestigationApproval.model_validate(artifact.payload)
                    target = by_investigation.get(approval.investigation_id)
                    if target is not None and (
                        target.approval is None
                        or approval.decided_at >= target.approval.decided_at
                    ):
                        target.approval = approval
            elif artifact.type is ArtifactType.INVESTIGATION_RECORD:
                with suppress(ValidationError):
                    record = InvestigationRecord.model_validate(artifact.payload)
                    target = by_investigation.get(record.investigation_id)
                    if target is not None:
                        target.record = record
            else:
                with suppress(ValidationError):
                    finding = ValidatedFinding.model_validate(artifact.payload)
                    target = by_investigation.get(finding.investigation_id)
                    if target is not None:
                        target.finding = finding

    def _macro_loops(self, project_id: str, plan_session_id: str) -> list[MacroLoopView]:
        views: list[MacroLoopView] = []
        for artifact in self._store.list_indexed_artifacts(
            project_id=project_id,
            session_id=plan_session_id,
            artifact_types=(ArtifactType.LOOP_LEDGER,),
        ):
            try:
                ledger = LoopLedger.model_validate(artifact.payload)
            except ValidationError:
                continue
            views.append(
                MacroLoopView(
                    plan_session_id=plan_session_id,
                    depth=ledger.depth,
                    rounds=[
                        MacroLoopRoundView(
                            round_id=row.round_id,
                            new_validated_findings=row.new_validated_findings,
                            redundant_findings=row.redundant_findings,
                            discarded_findings=row.discarded_findings,
                            executed_questions=row.executed_questions,
                            tokens=row.tokens,
                            exit_reason=row.exit_reason,
                            disposition=row.disposition,
                        )
                        for row in ledger.rounds
                    ],
                    admitted_finding_count=len(ledger.validated_finding_ids),
                    total_tokens=sum(row.tokens for row in ledger.rounds),
                    exit_reason=(
                        ledger.rounds[-1].exit_reason if ledger.rounds else ""
                    ),
                )
            )
        return views

    def _require_plan(self, project_id: str, session_id: str, plan_id: str) -> _PlanRecord:
        for record in self._plan_records(project_id, session_id):
            if record.artifact.id == plan_id:
                return record
        raise InvestigationNotFoundError(plan_id, session_id)

    def _require_decidable(self, record: _PlanRecord, decision: str) -> None:
        plan_id = record.artifact.id
        if record.has_outcome:
            raise InvestigationNotDecidableError(
                plan_id, "it already has an execution outcome"
            )
        if record.decided:
            raise InvestigationNotDecidableError(
                plan_id,
                f"it was already {record.approval.decision if record.approval else 'decided'}",
            )
        if decision == "approved" and record.plan.status != "planned":
            raise InvestigationNotDecidableError(
                plan_id, f"its plan status is '{record.plan.status}'"
            )

    def _require_executable(self, record: _PlanRecord) -> None:
        plan_id = record.artifact.id
        if record.approval is None or record.approval.decision != "approved":
            raise InvestigationNotExecutableError(plan_id, "it has no approval")
        if record.has_outcome:
            raise InvestigationNotExecutableError(
                plan_id, "it already has an execution outcome"
            )
        if record.plan.status != "planned" or not record.plan.execution_ready:
            raise InvestigationNotExecutableError(
                plan_id,
                "it is not execution-ready; revise the question card toward an "
                "executable method",
            )

    def _require_known_questions(
        self, project_id: str, session_id: str, question_ids: list[str]
    ) -> None:
        candidate_set = latest_candidate_set(self._store, project_id, session_id)
        known = (
            {candidate.question_id for candidate in candidate_set.candidates}
            if candidate_set is not None
            else set()
        )
        missing = [item for item in question_ids if item not in known]
        if missing:
            raise InvestigationNotFoundError(missing[0], session_id)

    def _require_source_session_idle(self, session_id: str) -> None:
        active = self._store.find_active_job_for_lane(session_id)
        if active is not None:
            raise InvestigationRunBusyError(session_id, str(active["job_id"]))

    def _require_plan_lane_free(self, project_id: str, session_id: str) -> None:
        lane = plan_build_lane(session_id)
        for job in self._store.list_active_jobs():
            if (
                str(job["kind"]) == PLAN_JOB_KIND
                and str(job["project_id"]) == project_id
                and _lane_of(str(job["session_id"])) == lane
            ):
                raise InvestigationRunBusyError(session_id, str(job["job_id"]))

    def _require_execution_lane_free(self, project_id: str, plan_session_id: str) -> None:
        """One execution or macro loop per plan run: both write findings and
        records into it, and the loop reads the very results the other writes."""
        active = self._store.find_active_job_for_lane(plan_session_id)
        if active is not None:
            raise InvestigationRunBusyError(plan_session_id, str(active["job_id"]))
        lane = execution_lane(plan_session_id)
        for job in self._store.list_active_jobs():
            if (
                str(job["kind"]) in _EXECUTION_LANE_KINDS
                and str(job["project_id"]) == project_id
                and _lane_of(str(job["session_id"])) == lane
            ):
                raise InvestigationRunBusyError(plan_session_id, str(job["job_id"]))

    def _check_build_replay_matches(self, job_row: dict, *, session_id: str) -> None:
        job_id = str(job_row["job_id"])
        run_row = self._store.get_session_index_row(session_id)
        if (
            run_row is None
            or str(job_row["kind"]) != PLAN_JOB_KIND
            or str(job_row["project_id"]) != str(run_row["project_id"])
            or _lane_of(str(job_row["session_id"])) != plan_build_lane(session_id)
        ):
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id} from a different "
                "project, kind or source run.",
            )

    def _build_replayed(self, session_id: str, job_row: dict) -> InvestigationPlanBuildStarted:
        job = self._jobs.get_job(str(job_row["job_id"]))
        return InvestigationPlanBuildStarted(
            session_id=session_id,
            execution_session_id=job.session_id,
            job=_to_created(job),
        )

    def _check_execute_replay_matches(
        self, job_row: dict, *, session_id: str, action_hash: str
    ) -> tuple[str, list[str]]:
        job_id = str(job_row["job_id"])
        run_row = self._store.get_session_index_row(session_id)
        if (
            run_row is None
            or str(job_row["kind"]) != EXECUTE_JOB_KIND
            or str(job_row["project_id"]) != str(run_row["project_id"])
        ):
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id} from a different project or kind.",
            )
        payload = self._consumed_payload(job_id, action_hash, session_id=session_id)
        plan_session_id = str(payload.get("plan_session_id", ""))
        plan_ids = [str(item) for item in payload.get("plan_ids", [])]
        if _lane_of(str(job_row["session_id"])) != execution_lane(plan_session_id):
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id} derived from a "
                "different plan run.",
            )
        return plan_session_id, plan_ids

    def _check_macro_replay_matches(
        self, job_row: dict, *, session_id: str, action_hash: str
    ) -> tuple[str, int]:
        job_id = str(job_row["job_id"])
        run_row = self._store.get_session_index_row(session_id)
        if (
            run_row is None
            or str(job_row["kind"]) != MACRO_LOOP_JOB_KIND
            or str(job_row["project_id"]) != str(run_row["project_id"])
        ):
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id} from a different project or kind.",
            )
        payload = self._consumed_payload(job_id, action_hash, session_id=session_id)
        plan_session_id = str(payload.get("plan_session_id", ""))
        if _lane_of(str(job_row["session_id"])) != execution_lane(plan_session_id):
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id} derived from a "
                "different plan run.",
            )
        return plan_session_id, int(payload.get("depth", 0))

    def _consumed_payload(
        self, job_id: str, action_hash: str, *, session_id: str
    ) -> dict[str, Any]:
        """The payload of an approval this very run really consumed. A replay
        that cannot prove that is a conflict, not a free pass."""
        pending = self._store.get_pending_action(action_hash, session_id=session_id)
        if pending is None or str(pending["status"]) != "consumed":
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id}, but the action "
                "hash was never consumed by this run.",
            )
        payload = json.loads(str(pending["payload_json"]))
        return payload if isinstance(payload, dict) else {}

    def _approval_payload_for_idempotency(
        self, job_row: dict, action_hash: str, *, session_id: str
    ) -> tuple[dict[str, Any], str]:
        """Read the current approval payload before status checks.

        This ordering makes a re-armed approval with different effective
        parameters a typed idempotency mismatch, while the later consumed
        check still prevents an unspent approval from replaying a job.
        """
        job_id = str(job_row["job_id"])
        pending = self._store.get_pending_action(action_hash, session_id=session_id)
        if pending is None:
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id}, but its approval "
                "payload is unavailable.",
            )
        payload = json.loads(str(pending["payload_json"]))
        payload = payload if isinstance(payload, dict) else {}
        digest = payload_digest(payload)
        if digest != str(pending["payload_digest"]):
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id}, but its approval "
                "payload failed integrity validation.",
            )
        return payload, digest

    def _execute_replayed(
        self, session_id: str, plan_session_id: str, plan_ids: list[str], job_row: dict
    ) -> InvestigationExecutionStarted:
        job = self._jobs.get_job(str(job_row["job_id"]))
        return InvestigationExecutionStarted(
            session_id=session_id,
            plan_session_id=plan_session_id,
            execution_session_id=job.session_id,
            plan_ids=plan_ids,
            job=_to_created(job),
        )

    def _macro_replayed(
        self, session_id: str, plan_session_id: str, depth: int, job_row: dict
    ) -> MacroLoopStarted:
        job = self._jobs.get_job(str(job_row["job_id"]))
        profile = DEPTH_PROFILES.get(depth)
        return MacroLoopStarted(
            session_id=session_id,
            plan_session_id=plan_session_id,
            execution_session_id=job.session_id,
            depth=depth,
            rounds_cap=profile.rounds if profile is not None else 0,
            job=_to_created(job),
        )

    def _project_for_run(self, session_id: str) -> str:
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        row = self._store.get_session_index_row(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        return str(row["project_id"])


def _default_reason(decision: str) -> str:
    return (
        "Approved from the Questions page."
        if decision == "approved"
        else "Rejected from the Questions page."
    )


def _to_plan_view(record: _PlanRecord) -> InvestigationPlanView:
    plan = record.plan
    status = record.status
    runnable = plan.status == "planned" and plan.execution_ready
    return InvestigationPlanView(
        plan_id=record.artifact.id,
        plan_session_id=record.session_id,
        investigation_id=plan.investigation_id,
        question_id=plan.question_id,
        question=plan.question,
        method_family=plan.method_family,
        method_recipe=plan.method_recipe,
        card_version=plan.card_version,
        status=status,
        plan_status=plan.status,
        execution_ready=plan.execution_ready,
        allowed_tools=list(plan.allowed_tools),
        target_datasets=list(plan.target_datasets),
        method_requirements=list(plan.method_requirements),
        validation_gates=[
            InvestigationGateView(name=gate.name, status=gate.status, reason=gate.reason)
            for gate in plan.validation_gates
        ],
        candidate_fingerprint=plan.candidate_fingerprint,
        deep_investigation=DEEP_MARKER_TOOL in plan.allowed_tools,
        decision_reason=record.approval.reason if record.approval is not None else "",
        outcome_status=record.record.status if record.record is not None else None,
        outcome_reason=record.record.reason if record.record is not None else "",
        finding_texts=(
            [item.text for item in record.finding.findings]
            if record.finding is not None
            else []
        ),
        report_readiness=(
            record.finding.report_readiness if record.finding is not None else None
        ),
        # Rejecting stays available on a not-execution-ready plan: it is the
        # only way to close one out without executing.
        can_approve=status == "pending" and plan.status == "planned",
        can_reject=status == "pending",
        can_execute=status == "approved" and runnable,
    )


def _to_created(job: JobStatus) -> JobCreated:
    return JobCreated(
        job_id=job.job_id,
        session_id=job.session_id,
        status=job.status,
        events_url=job.events_url,
    )
