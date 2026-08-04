"""Questions use cases (§7.5 / 阶段4 slice G).

List shapes the run's QuestionCandidateSet plus the latest execution outcome
per question (auto-executed results in the run itself, then derived qsess_*
batch runs). Prepare registers a server-side pending approval bound to the
candidate content; execute consumes it once and queues a `question_exec` job
that runs the existing `run_question_batch` driver on a fresh derived run.

Card editing and free-text drafting are the write side of the same page. An
edit is a deterministic in-place rewrite of one card (it bumps card_version and
re-runs feasibility), so it happens inline. Drafting calls the model, so it
goes through the same prepare → approve → job path as execution, on a derived
`qdsess_*` run.
"""

from __future__ import annotations

import json
import re
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from eda_platform.agents.question_runtime import QUESTION_AGENT_POLICY_VERSION
from eda_platform.application.dto import (
    JobCreated,
    JobStatus,
    QuestionDraftPrepared,
    QuestionDraftStarted,
    QuestionExecutionPrepared,
    QuestionExecutionStarted,
    QuestionExecutionSummary,
    QuestionSummary,
    QuestionsView,
)
from eda_platform.application.services.approval_service import (
    ApprovalService,
    payload_digest,
)
from eda_platform.application.services.job_service import JobConflictError, JobService
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.application.workbench import checkbox_disabled_for_feasibility
from eda_platform.core.ids import INTERNAL_SESSION_MARKER, stable_hash
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.card_edit import (
    EDITABLE_FIELDS,
    CardVersionConflictError,
    edit_candidate,
)
from eda_platform.drivers.question_exec import generate_batch_session_id
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.questions import QuestionCandidate, QuestionCandidateSet

APPROVAL_KIND_QUESTION = "question_execute"
APPROVAL_KIND_DRAFT = "question_draft"
DRAFT_JOB_KIND = "question_draft"
BATCH_SESSION_PREFIX = "qsess_"
DRAFT_SESSION_PREFIX = "qdsess_"
MAX_DRAFT_QUESTION_CHARS = 500
_DERIVED_SESSION_SCAN_LIMIT = 1000
# A tool-guard rejection is a paragraph: a generic first line ("Tool guard
# rejected parameters for `x`."), then "What was wrong:", "Allowed:", "How to
# fix:". The first line names the mechanism, the first "What was wrong" bullet
# names the actual cause — that is the one the card shows. Full text stays on
# the QuestionExecutionResult artifact.
_MAX_FAILURE_HEADLINE_CHARS = 240
_FAILURE_CAUSE_MARKER = "what was wrong:"
# Guard bullets read "`plan.sql` got 'internal rule name': <the readable cause>".
_GUARD_BULLET = re.compile(r"^-\s*`[^`]+`\s+got\s+'[^']*':\s*(?P<cause>.+)$")


def _failure_headline(error: object) -> str | None:
    """The most specific sentence in an execution error, for the question card."""
    if not isinstance(error, str):
        return None
    lines = [line.strip() for line in error.splitlines()]
    populated = [line for line in lines if line]
    if not populated:
        return None
    for index, line in enumerate(lines):
        if line.lower() != _FAILURE_CAUSE_MARKER:
            continue
        for candidate in lines[index + 1 :]:
            if not candidate.startswith("-"):
                break
            match = _GUARD_BULLET.match(candidate)
            cause = match.group("cause") if match else candidate.lstrip("- ").strip()
            if cause:
                return cause[:_MAX_FAILURE_HEADLINE_CHARS]
    return populated[0][:_MAX_FAILURE_HEADLINE_CHARS]


def generate_draft_session_id(source_session_id: str, question: str) -> str:
    """Derived run id for one card draft. The lane segment identifies the source
    run (one draft at a time per run, since drafts rewrite the same candidate
    set artifact); the request segment makes an idempotent retry provable."""
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    lane = stable_hash({"draft_source_session_id": source_session_id}, length=6)
    request = stable_hash({"question": question}, length=6)
    return f"{DRAFT_SESSION_PREFIX}{stamp}_{lane}_{request}"


def draft_lane(source_session_id: str) -> str:
    return stable_hash({"draft_source_session_id": source_session_id}, length=6)


def disclosure_settings_fingerprint(
    *, payload_policy: str, provider: str, model: str, base_url: str
) -> str:
    """Non-secret identity of the disclosure settings reviewed at prepare time."""
    return stable_hash(
        {
            "payload_policy": payload_policy,
            "provider": provider,
            "model": model,
            "base_url": base_url,
        },
        length=64,
    )


def latest_candidate_set(
    store: ArtifactStore, project_id: str, session_id: str
) -> QuestionCandidateSet | None:
    """Newest parseable QuestionCandidateSet for the run. Shared with the
    worker runner so both sides select the exact same set."""
    artifacts = store.list_indexed_artifacts(
        project_id=project_id,
        session_id=session_id,
        artifact_types=(ArtifactType.QUESTION_CANDIDATE_SET,),
    )
    for artifact in reversed(artifacts):
        with suppress(ValueError):
            return QuestionCandidateSet.model_validate(artifact.payload)
    return None


def candidate_fingerprint(candidate: QuestionCandidate) -> str:
    """Hash over the execution-affecting fields only: display fields like
    score must not invalidate an approval. Shared with the worker runner."""
    return stable_hash(
        {
            "question": candidate.question_en,
            "sql_template": candidate.sql_template,
            "target_datasets": list(candidate.target_datasets),
            "origin": str(candidate.origin),
            "category": candidate.value_category,
        },
        length=64,
    )


class QuestionServiceError(Exception):
    pass


class QuestionNotFoundError(QuestionServiceError):
    def __init__(self, question_id: str, session_id: str) -> None:
        super().__init__(f"Question not found in run {session_id}: {question_id}")
        self.question_id = question_id
        self.session_id = session_id


class QuestionNotExecutableError(QuestionServiceError):
    def __init__(self, question_id: str, reason: str) -> None:
        super().__init__(f"Question {question_id} cannot be executed: {reason}")
        self.question_id = question_id


class QuestionSourceChangedError(QuestionServiceError):
    """The candidate changed since prepare; the approval no longer binds it."""

    def __init__(self, question_id: str) -> None:
        super().__init__(
            f"Question {question_id} changed since the approval was prepared; "
            "prepare again and approve the fresh content."
        )
        self.question_id = question_id


class QuestionRunBusyError(QuestionServiceError):
    def __init__(self, session_id: str, job_id: str) -> None:
        super().__init__(
            f"Run {session_id} has an active job ({job_id}); wait for the analysis "
            "to finish before executing questions."
        )
        self.session_id = session_id
        self.job_id = job_id


class QuestionValidationError(QuestionServiceError):
    pass


class QuestionVersionConflictError(QuestionServiceError):
    def __init__(self, expected_version: int, current_version: int) -> None:
        super().__init__(
            f"Question card changed since it was loaded: expected version "
            f"{expected_version}, current version is {current_version}. Reload and retry."
        )
        self.expected_version = expected_version
        self.current_version = current_version


class QuestionService:
    def __init__(
        self,
        store: ArtifactStore,
        approvals: ApprovalService,
        jobs: JobService,
    ) -> None:
        self._store = store
        self._approvals = approvals
        self._jobs = jobs

    def edit_card(
        self,
        session_id: str,
        question_id: str,
        edits: dict[str, Any],
        *,
        expected_version: int,
    ) -> QuestionSummary:
        """Apply a text edit to one card and return it at its new version.

        Only framing fields are editable — the driver refuses SQL, scope and
        analysis mode, because changing those would silently change what a
        prior approval covers. Feasibility is recomputed on the edited card.
        """
        if not edits:
            raise QuestionValidationError(
                "Send at least one field to edit. Editable fields are: "
                f"{', '.join(sorted(EDITABLE_FIELDS))}."
            )
        project_id = self._project_for_run(session_id)
        # Both an edit and a draft job read-modify-write the same candidate set
        # artifact from different processes, so they share one lane.
        self._require_source_session_idle(session_id)
        self._require_draft_lane_free(project_id, session_id)
        self._require_candidate(project_id, session_id, question_id)
        try:
            edited = edit_candidate(
                self._store,
                project_id=project_id,
                session_id=session_id,
                question_id=question_id,
                expected_version=expected_version,
                edits=edits,
            )
        except CardVersionConflictError as exc:
            raise QuestionVersionConflictError(
                exc.expected_version, exc.current_version
            ) from exc
        except ValueError as exc:
            raise QuestionValidationError(str(exc)) from exc
        executions = self._latest_executions(project_id, session_id)
        return _to_summary(edited, executions.get(edited.question_id))

    def prepare_draft(
        self,
        session_id: str,
        question: str,
        *,
        llm: str = "env",
        payload_policy: str,
        disclosure_fingerprint: str,
    ) -> QuestionDraftPrepared:
        """Register an approval bound to the exact question text to draft."""
        text = question.strip()
        if not text:
            raise QuestionValidationError("A drafted card needs a question.")
        if len(text) > MAX_DRAFT_QUESTION_CHARS:
            raise QuestionValidationError(
                f"A drafted question must be at most {MAX_DRAFT_QUESTION_CHARS} characters."
            )
        project_id = self._project_for_run(session_id)
        self._require_source_session_idle(session_id)
        self._require_draft_lane_free(project_id, session_id)
        # The run must already own a candidate set: append_candidate edits that
        # artifact rather than creating one.
        if self._latest_candidate_set(project_id, session_id) is None:
            raise QuestionNotFoundError("<candidate set>", session_id)
        digest, approval_token, expires_at = self._approvals.register(
            kind=APPROVAL_KIND_DRAFT,
            session_id=session_id,
            project_id=project_id,
            action={
                "type": "question_draft",
                "source_session_id": session_id,
                "question": text,
                "llm": llm,
                "payload_policy": payload_policy,
                "disclosure_fingerprint": disclosure_fingerprint,
            },
            payload={
                "project_id": project_id,
                "source_session_id": session_id,
                # The approved text IS the content: execute drafts what was
                # reviewed, never a question the execute request supplies.
                "question": text,
                "llm": llm,
                "payload_policy": payload_policy,
                "disclosure_fingerprint": disclosure_fingerprint,
            },
        )
        return QuestionDraftPrepared(
            session_id=session_id,
            action_hash=digest,
            approval_token=approval_token,
            expires_at=expires_at,
            question=text,
            llm_mode=llm,
        )

    def draft(
        self,
        session_id: str,
        *,
        action_hash: str,
        approval_token: str,
        disclosure_fingerprint: str,
        payload_policy: str | None = None,
        llm_env: dict[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> QuestionDraftStarted:
        return self._approvals.run_idempotent_producer(
            action_hash,
            session_id=session_id,
            idempotency_key=idempotency_key,
            operation=lambda deadline: self._draft_once(
                session_id,
                action_hash=action_hash,
                approval_token=approval_token,
                disclosure_fingerprint=disclosure_fingerprint,
                payload_policy=payload_policy,
                llm_env=llm_env,
                idempotency_key=idempotency_key,
                contention_deadline=deadline,
            ),
        )

    def _draft_once(
        self,
        session_id: str,
        *,
        action_hash: str,
        approval_token: str,
        disclosure_fingerprint: str,
        payload_policy: str | None,
        llm_env: dict[str, str] | None,
        idempotency_key: str | None,
        contention_deadline: float,
    ) -> QuestionDraftStarted:
        project_id = self._project_for_run(session_id)
        # Idempotent replay must win before approval consumption, or a retried
        # draft would 409 on its own already-consumed hash (cleaning F1).
        if idempotency_key:
            existing = self._store.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                replay_payload, replay_payload_digest, _status = (
                    self._approvals.inspect_payload(action_hash, session_id=session_id)
                )
                self._jobs.assert_idempotent_replay(
                    existing,
                    request_scope=session_id,
                    project_id=project_id,
                    kind=DRAFT_JOB_KIND,
                    content={
                        "source_session_id": session_id,
                        "action_hash": action_hash,
                        "approval_payload_digest": replay_payload_digest,
                        "disclosure_fingerprint": disclosure_fingerprint,
                        "payload_policy": payload_policy,
                    },
                    env=(
                        llm_env
                        if str(replay_payload.get("llm", "env")) == "env"
                        else None
                    ),
                )
                question = self._check_draft_replay_matches(
                    existing, session_id=session_id, action_hash=action_hash
                )
                return self._draft_replayed(session_id, question, existing)
        self._require_source_session_idle(session_id)
        self._require_draft_lane_free(project_id, session_id)
        def validate(payload: dict[str, Any]) -> str:
            question = str(payload.get("question", ""))
            if not question:
                raise QuestionValidationError("The approval carries no question text.")
            if (
                payload.get("payload_policy") != payload_policy
                or payload.get("disclosure_fingerprint") != disclosure_fingerprint
            ):
                raise QuestionValidationError(
                    "Disclosure settings changed since approval; prepare the draft again."
                )
            return question

        payload, question = self._approvals.validate_then_consume(
            action_hash,
            kind=APPROVAL_KIND_DRAFT,
            session_id=session_id,
            generation=approval_token,
            validate=validate,
            idempotency_key=idempotency_key,
            deadline=contention_deadline,
        )
        with self._approvals.compensate_on_failure(action_hash, session_id=session_id):
            idempotency_content = {
                "source_session_id": session_id,
                "action_hash": action_hash,
                "approval_payload_digest": payload_digest(payload),
                "disclosure_fingerprint": disclosure_fingerprint,
                "payload_policy": payload_policy,
            }
            job = self._jobs.create_question_draft_job(
                generate_draft_session_id(session_id, question),
                project_id=project_id,
                source_session_id=session_id,
                question=question,
                llm=str(payload.get("llm", "env")),
                payload_policy=payload_policy,
                llm_env=llm_env,
                idempotency_key=idempotency_key,
                idempotency_content=idempotency_content,
            )
        return QuestionDraftStarted(
            session_id=session_id,
            execution_session_id=job.session_id,
            question=question,
            job=_to_created(job),
        )

    def _require_draft_lane_free(self, project_id: str, session_id: str) -> None:
        lane = draft_lane(session_id)
        for job in self._store.list_active_jobs():
            if (
                str(job["kind"]) == DRAFT_JOB_KIND
                and str(job["project_id"]) == project_id
                and str(job["session_id"]).split("_")[-2:-1] == [lane]
            ):
                raise QuestionRunBusyError(session_id, str(job["job_id"]))

    def _check_draft_replay_matches(
        self, job_row: dict, *, session_id: str, action_hash: str
    ) -> str:
        """The idempotency fast path must not bypass the approval checks: a
        replay is only legitimate for a question_draft job in this run's own
        project whose action hash this run really consumed. Returns the
        approved question so the replay answers with the same content."""
        job_id = str(job_row["job_id"])
        run_row = self._store.get_session_index_row(session_id)
        if (
            run_row is None
            or str(job_row["kind"]) != DRAFT_JOB_KIND
            or str(job_row["project_id"]) != str(run_row["project_id"])
            or str(job_row["session_id"]).split("_")[-2:-1] != [draft_lane(session_id)]
        ):
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id} from a different "
                "project, kind or source run.",
            )
        pending = self._store.get_pending_action(action_hash, session_id=session_id)
        if pending is None or str(pending["status"]) != "consumed":
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id}, but the action "
                "hash was never consumed by this run.",
            )
        payload = json.loads(str(pending["payload_json"]))
        return str(payload.get("question", "")) if isinstance(payload, dict) else ""

    def _draft_replayed(
        self, session_id: str, question: str, job_row: dict
    ) -> QuestionDraftStarted:
        job = self._jobs.get_job(str(job_row["job_id"]))
        return QuestionDraftStarted(
            session_id=session_id,
            execution_session_id=job.session_id,
            question=question,
            job=_to_created(job),
        )

    def list_questions(self, session_id: str) -> QuestionsView:
        project_id = self._project_for_run(session_id)
        candidate_set = self._latest_candidate_set(project_id, session_id)
        if candidate_set is None:
            return QuestionsView(session_id=session_id, questions=[])
        executions = self._latest_executions(project_id, session_id)
        questions = [
            _to_summary(candidate, executions.get(candidate.question_id))
            for candidate in sorted(
                candidate_set.candidates,
                key=lambda item: (-item.score.deterministic_score, item.question_id),
            )
        ]
        return QuestionsView(session_id=session_id, questions=questions)

    def prepare_execution(
        self, session_id: str, question_id: str, *, llm: str = "env"
    ) -> QuestionExecutionPrepared:
        project_id = self._project_for_run(session_id)
        self._require_source_session_idle(session_id)
        candidate = self._require_candidate(project_id, session_id, question_id)
        feasibility = candidate.feasibility.status if candidate.feasibility else None
        if checkbox_disabled_for_feasibility(feasibility):
            raise QuestionNotExecutableError(
                question_id, f"deterministic feasibility is '{feasibility}'"
            )
        agent_enabled = llm != "offline"
        fingerprint = candidate_fingerprint(candidate)
        digest, approval_token, expires_at = self._approvals.register(
            kind=APPROVAL_KIND_QUESTION,
            session_id=session_id,
            project_id=project_id,
            action={
                "type": "question_execute",
                "source_session_id": session_id,
                "question_id": question_id,
                "candidate_fingerprint": fingerprint,
                "execution_mode": "agent" if agent_enabled else "pipeline",
                "agent_policy": (
                    QUESTION_AGENT_POLICY_VERSION if agent_enabled else None
                ),
            },
            payload={
                "question_id": question_id,
                "question": candidate.question_en,
                "project_id": project_id,
                "source_session_id": session_id,
                "candidate_fingerprint": fingerprint,
                # The approval binds the LLM mode: execute runs what was
                # approved, never what the execute request claims.
                "llm": llm,
                "execution_mode": "agent" if agent_enabled else "pipeline",
                "agent_policy": (
                    QUESTION_AGENT_POLICY_VERSION if agent_enabled else None
                ),
            },
        )
        return QuestionExecutionPrepared(
            session_id=session_id,
            question_id=question_id,
            action_hash=digest,
            approval_token=approval_token,
            expires_at=expires_at,
            question=candidate.question_en,
            origin=str(candidate.origin),
            # Live execution is intentionally not bound to one SQL statement:
            # the approved agent policy selects tools at execution time.
            sql_preview=candidate.sql_template if not agent_enabled else None,
            target_datasets=list(candidate.target_datasets),
            uses_llm=agent_enabled,
            llm_mode=llm,
        )

    def execute(
        self,
        session_id: str,
        *,
        question_id: str,
        action_hash: str,
        approval_token: str,
        idempotency_key: str | None = None,
    ) -> QuestionExecutionStarted:
        return self._approvals.run_idempotent_producer(
            action_hash,
            session_id=session_id,
            idempotency_key=idempotency_key,
            operation=lambda deadline: self._execute_once(
                session_id,
                question_id=question_id,
                action_hash=action_hash,
                approval_token=approval_token,
                idempotency_key=idempotency_key,
                contention_deadline=deadline,
            ),
        )

    def _execute_once(
        self,
        session_id: str,
        *,
        question_id: str,
        action_hash: str,
        approval_token: str,
        idempotency_key: str | None,
        contention_deadline: float,
    ) -> QuestionExecutionStarted:
        project_id = self._project_for_run(session_id)
        # Idempotent replay must win before approval consumption, or a retried
        # execute would 409 on its own already-consumed hash (cleaning F1).
        if idempotency_key:
            existing = self._store.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                _payload, replay_payload_digest, _status = (
                    self._approvals.inspect_payload(action_hash, session_id=session_id)
                )
                self._jobs.assert_idempotent_replay(
                    existing,
                    request_scope=session_id,
                    project_id=project_id,
                    kind="question_exec",
                    content={
                        "source_session_id": session_id,
                        "question_id": question_id,
                        "action_hash": action_hash,
                        "approval_payload_digest": replay_payload_digest,
                    },
                )
                self._check_replay_matches(
                    existing,
                    session_id=session_id,
                    question_id=question_id,
                    action_hash=action_hash,
                )
                return self._replayed(session_id, question_id, existing)
        self._require_source_session_idle(session_id)
        def validate(
            payload: dict[str, Any],
        ) -> str:
            if str(payload.get("question_id", "")) != question_id:
                raise QuestionValidationError(
                    "The approval was prepared for a different question than the request path."
                )
            candidate = self._require_candidate(project_id, session_id, question_id)
            fingerprint = candidate_fingerprint(candidate)
            if fingerprint != str(payload.get("candidate_fingerprint", "")):
                raise QuestionSourceChangedError(question_id)
            return fingerprint

        payload, fingerprint = self._approvals.validate_then_consume(
            action_hash,
            kind=APPROVAL_KIND_QUESTION,
            session_id=session_id,
            generation=approval_token,
            validate=validate,
            idempotency_key=idempotency_key,
            deadline=contention_deadline,
        )
        with self._approvals.compensate_on_failure(action_hash, session_id=session_id):
            execution_session_id = generate_batch_session_id(session_id, [question_id])
            job = self._jobs.create_question_exec_job(
                execution_session_id,
                project_id=project_id,
                source_session_id=session_id,
                question_id=question_id,
                # The consumed approval owns the LLM mode (what was approved
                # is what runs); clients cannot override it at execute time.
                llm=str(payload.get("llm", "env")),
                candidate_fingerprint=fingerprint,
                idempotency_key=idempotency_key,
                idempotency_content={
                    "source_session_id": session_id,
                    "question_id": question_id,
                    "action_hash": action_hash,
                    "approval_payload_digest": payload_digest(payload),
                },
            )
        return QuestionExecutionStarted(
            session_id=session_id,
            question_id=question_id,
            execution_session_id=job.session_id,
            job=_to_created(job),
        )

    def _check_replay_matches(
        self, job_row: dict, *, session_id: str, question_id: str, action_hash: str
    ) -> None:
        """The idempotency fast path must not bypass the approval checks:
        replay is only legitimate for a question_exec job in this run's own
        project whose action hash this run really consumed."""
        job_id = str(job_row["job_id"])
        run_row = self._store.get_session_index_row(session_id)
        if (
            run_row is None
            or str(job_row["kind"]) != "question_exec"
            or str(job_row["project_id"]) != str(run_row["project_id"])
        ):
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id} from a different "
                "project or kind.",
            )
        pending = self._store.get_pending_action(action_hash, session_id=session_id)
        if pending is None or str(pending["status"]) != "consumed":
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id}, but the action "
                "hash was never consumed by this run.",
            )
        payload = json.loads(str(pending["payload_json"]))
        if not isinstance(payload, dict) or str(payload.get("question_id", "")) != question_id:
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id} for a different question.",
            )
        # The existing job must derive from THIS source run: with the same
        # question_id in two runs of one project, replaying run1's key against
        # run2 would silently return run1's job (review C).
        derived_session_id = str(job_row["session_id"])
        manifest = None
        with suppress(OSError, ValueError):
            manifest = self._store.read_manifest(str(job_row["project_id"]), derived_session_id)
        if manifest is not None:
            source_matches = manifest.source_session_id == session_id
        else:
            # Queued job, manifest not written yet: the batch-run id suffix is
            # a deterministic hash over (source_session_id, question_ids).
            expected_suffix = generate_batch_session_id(
                session_id, [question_id]
            ).rsplit("_", 1)[-1]
            source_matches = derived_session_id.endswith(f"_{expected_suffix}")
        if not source_matches:
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id} derived from a "
                "different source run.",
            )

    def _require_source_session_idle(self, session_id: str) -> None:
        """Refuse to prepare/execute questions while the source run itself is
        still being analysed — its candidate set could change mid-flight."""
        active = self._store.find_active_job_for_lane(session_id)
        if active is not None:
            raise QuestionRunBusyError(session_id, str(active["job_id"]))

    def _replayed(
        self, session_id: str, question_id: str, job_row: dict
    ) -> QuestionExecutionStarted:
        job = self._jobs.get_job(str(job_row["job_id"]))
        return QuestionExecutionStarted(
            session_id=session_id,
            question_id=question_id,
            execution_session_id=job.session_id,
            job=_to_created(job),
        )

    def _project_for_run(self, session_id: str) -> str:
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        row = self._store.get_session_index_row(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        return str(row["project_id"])

    def _latest_candidate_set(
        self, project_id: str, session_id: str
    ) -> QuestionCandidateSet | None:
        return latest_candidate_set(self._store, project_id, session_id)

    def _require_candidate(
        self, project_id: str, session_id: str, question_id: str
    ) -> QuestionCandidate:
        candidate_set = self._latest_candidate_set(project_id, session_id)
        if candidate_set is None:
            raise QuestionNotFoundError(question_id, session_id)
        for candidate in candidate_set.candidates:
            if candidate.question_id == question_id:
                return candidate
        raise QuestionNotFoundError(question_id, session_id)

    def _latest_executions(
        self, project_id: str, session_id: str
    ) -> dict[str, QuestionExecutionSummary]:
        """Latest QuestionExecutionResult per question: the run's own
        auto-executed results first, then derived batch runs oldest→newest so
        the newest execution wins by overwrite."""
        executions: dict[str, QuestionExecutionSummary] = {}
        for execution_session_id in (session_id, *self._derived_batch_runs(project_id, session_id)):
            for artifact in self._store.list_indexed_artifacts(
                project_id=project_id,
                session_id=execution_session_id,
                artifact_types=(ArtifactType.QUESTION_EXECUTION_RESULT,),
            ):
                payload = artifact.payload
                question_id = str(payload.get("question_id", ""))
                if not question_id:
                    continue
                status = str(payload.get("status", "failed"))
                outcome = payload.get("outcome") or (
                    "answered" if status == "succeeded" else "failed"
                )
                findings = payload.get("findings")
                abstention = payload.get("abstention_code")
                executions[question_id] = QuestionExecutionSummary(
                    outcome=str(outcome),
                    status=status,
                    findings_count=len(findings) if isinstance(findings, list) else 0,
                    qexec_artifact_id=artifact.id,
                    execution_session_id=execution_session_id,
                    abstention_code=str(abstention) if abstention else None,
                    failure_reason=_failure_headline(payload.get("error")),
                )
        return executions

    def _derived_batch_runs(self, project_id: str, session_id: str) -> list[str]:
        """qsess_* runs whose manifest points back at this run, oldest first."""
        rows = self._store.query_session_index_rows(project_id, limit=_DERIVED_SESSION_SCAN_LIMIT)
        derived: list[str] = []
        # query_session_index_rows returns newest-first; reverse for oldest-first.
        for row in reversed(rows):
            candidate_session_id = str(row["session_id"])
            if not candidate_session_id.startswith(BATCH_SESSION_PREFIX):
                continue
            with suppress(OSError, ValueError):
                manifest = self._store.read_manifest(project_id, candidate_session_id)
                if manifest is not None and manifest.source_session_id == session_id:
                    derived.append(candidate_session_id)
        return derived


def _to_summary(
    candidate: QuestionCandidate, execution: QuestionExecutionSummary | None
) -> QuestionSummary:
    feasibility = candidate.feasibility.status if candidate.feasibility else None
    return QuestionSummary(
        question_id=candidate.question_id,
        question=candidate.question_en,
        origin=str(candidate.origin),
        analysis_mode=candidate.analysis_mode,
        value_category=candidate.value_category,
        feasibility_status=feasibility,
        proposed_action=candidate.proposed_action,
        priority=candidate.score.deterministic_score,
        exploratory=candidate.exploratory,
        target_datasets=list(candidate.target_datasets),
        business_decision=candidate.business_decision,
        executable=not checkbox_disabled_for_feasibility(feasibility),
        execution=execution,
        card_version=candidate.card_version,
        value_hypothesis=candidate.value_hypothesis,
        success_criterion=candidate.success_criterion,
        data_signal=candidate.data_signal,
        priority_rationale=candidate.priority_rationale,
        risks=list(candidate.risks),
        data_requirements=list(candidate.data_requirements),
    )


def _to_created(job: JobStatus) -> JobCreated:
    return JobCreated(
        job_id=job.job_id,
        session_id=job.session_id,
        status=job.status,
        events_url=job.events_url,
    )
