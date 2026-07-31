"""Run bounded, read-only follow-up probes with deterministic result validation."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from uuid import uuid4

import duckdb
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from eda_platform.agents.interpretation import validate_interpretation_text
from eda_platform.core.budget import BudgetExceeded
from eda_platform.core.ids import stable_hash
from eda_platform.core.llm import LLMClient, is_offline_client
from eda_platform.core.llm_ledger import logical_llm_call
from eda_platform.core.loop_journal import (
    JsonlLoopJournal,
    LoopTransitionError,
    make_loop_call_id,
    make_loop_probe_id,
    make_loop_step_id,
)
from eda_platform.core.query import (
    DuckDBQueryEngine,
    QueryTimeout,
    SqlBindingError,
    UnsafeQueryError,
    validate_select_statement,
)
from eda_platform.core.trace import (
    LOOP_FAILURE_HISTORY_INJECTED,
    PROBE_REPEATED_REJECTED,
    trace_event,
)
from eda_platform.drivers import question_exec
from eda_platform.schemas.artifacts import Artifact
from eda_platform.schemas.deep_investigation import (
    DeepInvestigationResult,
    LoopExitReason,
    LoopStepRecord,
)
from eda_platform.schemas.investigations import InvestigationPlan
from eda_platform.schemas.questions import QuestionCandidate, QuestionFinding, QuestionScore
from eda_platform.schemas.sessions import TraceEvent
from eda_platform.tools.sql_runner import SqlCatalog, run_sql

_TASK = "di5_bounded_probe"

# Identity for each guarded in-memory probe result.
_LOOP_PROJECT = "deep_loop"

# Extract referenced tables while excluding CTE self-references.
_TABLE_RE = re.compile(r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_.\"]*)", re.IGNORECASE)
_CTE_RE = re.compile(r"(?:\bwith\b|,)\s+([A-Za-z_][A-Za-z0-9_]*)\s+as\s*\(", re.IGNORECASE)

_REPEAT_EXIT_THRESHOLD = 2
_FAILURE_HISTORY_LIMIT = 5
_FAILURE_ENTRY_MAX_CHARS = 160

_FAILURE_HISTORY_NOTE = (
    "The following probes failed. Do not repeat the same path; choose a different direction "
    "or conclude."
)

TraceSink = Callable[[TraceEvent], None]


def _probe_fingerprint(sql: str) -> str:
    """Return a stable fingerprint after normalizing cosmetic SQL differences."""
    normalized = re.sub(r"\s+", " ", sql.strip().rstrip(";").strip()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _failure_entry(fingerprint: str, reason: str) -> str:
    """One compressed failure-history line: action fingerprint + one-line reason."""
    one_line = " ".join(reason.split())
    return f"[fp:{fingerprint}] {one_line}"[:_FAILURE_ENTRY_MAX_CHARS]


class _ProbeDecision(BaseModel):
    """Strict parse target for a probe-or-conclude decision."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["probe", "conclude"]
    purpose: str = Field(default="")
    sql: str = Field(default="")
    rationale: str = Field(default="")

    @model_validator(mode="after")
    def validate_action_payload(self) -> _ProbeDecision:
        if self.action == "probe":
            if not self.purpose.strip():
                raise ValueError("A probe decision requires a non-empty purpose.")
            if not self.sql.strip():
                raise ValueError("A probe decision requires non-empty SQL.")
        elif not self.rationale.strip():
            raise ValueError("A conclude decision requires a non-empty rationale.")
        return self


class _PersistedProbeRecord(BaseModel):
    """Recoverable probe output committed before the journal marks completion."""

    investigation_id: str
    probe_id: str
    iteration: int = Field(ge=0)
    step: LoopStepRecord
    sql_artifact: Artifact | None = None

    @model_validator(mode="after")
    def validate_artifact_binding(self) -> _PersistedProbeRecord:
        expected = self.step.result_artifact_id
        actual = self.sql_artifact.id if self.sql_artifact is not None else None
        if expected != actual:
            raise ValueError("probe record result artifact identity does not match its step.")
        return self


def run_bounded_loop(
    llm: LLMClient,
    *,
    plan: InvestigationPlan,
    primary_findings: list[QuestionFinding],
    query_engine: DuckDBQueryEngine,
    catalog: SqlCatalog,
    max_steps: int = 3,
    llm_call_cap: int = 8,
    trace_sink: TraceSink | None = None,
    journal: JsonlLoopJournal | None = None,
) -> DeepInvestigationResult:
    """Run a bounded probe loop and return its typed transcript and exit reason."""
    if journal is not None:
        return _run_journaled_loop(
            llm,
            plan=plan,
            primary_findings=primary_findings,
            query_engine=query_engine,
            catalog=catalog,
            max_steps=max_steps,
            llm_call_cap=llm_call_cap,
            trace_sink=trace_sink,
            journal=journal,
        )
    if is_offline_client(llm):
        return _build_result(
            plan,
            max_steps=max_steps,
            llm_call_cap=llm_call_cap,
            steps=[],
            exit_reason="offline",
            llm_calls_used=0,
            probe_errors=0,
            conclusion_note="",
        )

    allowed_tables = set(catalog.relations.values())
    steps: list[LoopStepRecord] = []
    accumulated: list[QuestionFinding] = list(primary_findings)
    llm_calls_used = 0
    probe_errors = 0
    consecutive_errors = 0
    probe_count = 0
    conclusion_note = ""
    exit_reason: LoopExitReason = "step_cap_reached"
    # Track repeated probes and recent failures for the next planning call.
    fingerprint_first_step: dict[str, int] = {}
    repeat_rejections = 0
    failure_history: list[str] = []
    repeat_notice = ""

    while True:
        # Hard ceilings are checked before every planning call: a probe slot cap
        # and an absolute LLM-call cap. Both are typed exits, not exceptions.
        if probe_count >= max_steps:
            exit_reason = "step_cap_reached"
            break
        if llm_calls_used >= llm_call_cap:
            exit_reason = "llm_cap_reached"
            break

        payload = _build_payload(
            plan,
            primary_findings=primary_findings,
            steps=steps,
            allowed_tables=allowed_tables,
            probes_remaining=max_steps - probe_count,
            llm_calls_remaining=llm_call_cap - llm_calls_used,
            failure_history=failure_history,
            repeat_notice=repeat_notice,
        )
        # The repeat notice is one-shot: it has now been injected (and the
        # rejected step itself stays visible in ``prior_probes``).
        repeat_notice = ""
        if failure_history and trace_sink is not None:
            injected = failure_history[-_FAILURE_HISTORY_LIMIT:]
            trace_sink(
                trace_event(
                    session_id=plan.source_session_id,
                    event_type=LOOP_FAILURE_HISTORY_INJECTED,
                    name="investigation_loop",
                    summary={
                        "investigation_id": plan.investigation_id,
                        "entries": len(injected),
                        "chars": sum(len(entry) for entry in injected),
                    },
                )
            )

        # One structured call with a single bounded retry. EVERY invocation
        # (including the retry) counts against ``llm_call_cap``; a cap hit reached
        # mid-retry ends the loop as ``llm_cap_reached``.
        decision: _ProbeDecision | None = None
        validation_feedback = ""
        for _attempt in range(2):
            if llm_calls_used >= llm_call_cap:
                break
            llm_calls_used += 1
            attempt_payload = payload
            if validation_feedback:
                attempt_payload = {
                    **payload,
                    "decision_validation_error": validation_feedback,
                    "retry_instruction": (
                        "Return exactly one valid probe or conclude decision matching the schema."
                    ),
                }
            try:
                decision = llm.structured(
                    task=_TASK,
                    schema=_ProbeDecision,
                    payload=attempt_payload,
                )
                break
            except BudgetExceeded:
                raise
            except (ValidationError, RuntimeError, ValueError) as exc:
                decision = None
                validation_feedback = f"{type(exc).__name__}: {str(exc)[:240]}"
        if decision is None:
            if llm_calls_used >= llm_call_cap:
                exit_reason = "llm_cap_reached"
                break
            # Both attempts failed to parse but budget remains: fall through to a
            # fresh planning iteration (which will eventually exhaust the cap).
            continue

        if decision.action == "conclude":
            ok, _reason = validate_interpretation_text(decision.rationale, accumulated)
            conclusion_note = decision.rationale.strip() if ok else ""
            steps.append(
                LoopStepRecord(
                    step_index=len(steps),
                    action="conclude",
                    purpose=decision.purpose.strip() or "Concluded the bounded investigation.",
                    sql="",
                    findings=[],
                    status="succeeded",
                )
            )
            exit_reason = "concluded"
            break

        # The decision model admits only "probe" or "conclude".
        purpose = decision.purpose.strip() or "Follow-up probe within the approved scope."
        sql = decision.sql.strip()
        step_index = len(steps)

        # Reject repeated probes without execution and feed the failure back.
        fingerprint = _probe_fingerprint(sql)
        first_step = fingerprint_first_step.get(fingerprint)
        if first_step is not None:
            repeat_rejections += 1
            repeat_notice = (
                f"Probe rejected without execution: identical to the probe of step "
                f"{first_step} after normalization "
                f"(duplicate of step {first_step}; choose a different direction or conclude)."
            )
            failure_history.append(
                _failure_entry(fingerprint, f"repeated probe (duplicate of step {first_step})")
            )
            steps.append(
                LoopStepRecord(
                    step_index=step_index,
                    action="probe",
                    purpose=purpose,
                    sql=sql,
                    findings=[],
                    status="skipped",
                    error=repeat_notice,
                )
            )
            if trace_sink is not None:
                trace_sink(
                    trace_event(
                        session_id=plan.source_session_id,
                        event_type=PROBE_REPEATED_REJECTED,
                        name="investigation_loop",
                        summary={
                            "investigation_id": plan.investigation_id,
                            "step_index": step_index,
                            "duplicate_of_step": first_step,
                            "fingerprint": fingerprint,
                            "repeat_rejections": repeat_rejections,
                        },
                    )
                )
            if repeat_rejections >= _REPEAT_EXIT_THRESHOLD:
                exit_reason = "repeated_action"
                break
            continue
        fingerprint_first_step[fingerprint] = step_index

        # Scope + read-only guards run BEFORE any execution, so an out-of-scope or
        # unsafe probe is recorded as a failed step that never touched the engine.
        violation = _guard_probe(sql, allowed_tables, query_engine)
        if violation is not None:
            failure_history.append(_failure_entry(fingerprint, violation))
            steps.append(
                LoopStepRecord(
                    step_index=step_index,
                    action="probe",
                    purpose=purpose,
                    sql=sql,
                    findings=[],
                    status="failed",
                    error=violation,
                )
            )
            probe_errors += 1
            consecutive_errors += 1
            probe_count += 1
            if consecutive_errors >= 2:
                exit_reason = "probe_error_cap_reached"
                break
            continue

        try:
            sql_artifact = run_sql(
                catalog, sql, project_id=_LOOP_PROJECT, session_id=plan.source_session_id
            )
        except (
            UnsafeQueryError,
            QueryTimeout,
            SqlBindingError,
            duckdb.Error,
            ValueError,
            RuntimeError,
        ) as exc:
            error_text = f"{type(exc).__name__}: {str(exc)[:200]}"
            failure_history.append(_failure_entry(fingerprint, error_text))
            steps.append(
                LoopStepRecord(
                    step_index=step_index,
                    action="probe",
                    purpose=purpose,
                    sql=sql,
                    findings=[],
                    status="failed",
                    error=error_text,
                )
            )
            probe_errors += 1
            consecutive_errors += 1
            probe_count += 1
            if consecutive_errors >= 2:
                exit_reason = "probe_error_cap_reached"
                break
            continue

        findings = _reduce_probe_findings(purpose, plan, sql_artifact, step_index)
        accumulated.extend(findings)
        steps.append(
            LoopStepRecord(
                step_index=step_index,
                action="probe",
                purpose=purpose,
                sql=sql,
                result_artifact_id=sql_artifact.id,
                findings=findings,
                status="succeeded",
            )
        )
        consecutive_errors = 0
        probe_count += 1

    return _build_result(
        plan,
        max_steps=max_steps,
        llm_call_cap=llm_call_cap,
        steps=steps,
        exit_reason=exit_reason,
        llm_calls_used=llm_calls_used,
        probe_errors=probe_errors,
        conclusion_note=conclusion_note,
    )


def _run_journaled_loop(
    llm: LLMClient,
    *,
    plan: InvestigationPlan,
    primary_findings: list[QuestionFinding],
    query_engine: DuckDBQueryEngine,
    catalog: SqlCatalog,
    max_steps: int,
    llm_call_cap: int,
    trace_sink: TraceSink | None,
    journal: JsonlLoopJournal,
) -> DeepInvestigationResult:
    """Execute or resume the loop with the append-only journal as source of truth."""
    state = journal.rebuild()
    if state is None:
        raise LoopTransitionError("journal must be initialized before loop execution.")
    if state.investigation_id != plan.investigation_id:
        raise LoopTransitionError("journal investigation does not match the approved plan.")
    if state.max_steps != max_steps or state.llm_call_cap != llm_call_cap:
        raise LoopTransitionError("journal hard caps do not match loop arguments.")
    if state.status in {"concluded", "budget_exhausted"}:
        return _load_terminal_result(journal, state.final_draft_ref)
    if state.status != "running":
        raise LoopTransitionError(
            f"cannot safely resume terminal loop status {state.status!r}."
        )
    if is_offline_client(llm):
        return _commit_terminal_result(
            journal,
            _build_result(
                plan,
                max_steps=max_steps,
                llm_call_cap=llm_call_cap,
                steps=[],
                exit_reason="offline",
                llm_calls_used=0,
                probe_errors=0,
                conclusion_note="",
            ),
        )

    journal.claim_attempt()
    state = journal.rebuild()
    assert state is not None
    if state.pending_call_id is not None:
        journal.append_new(
            "loop_call_uncertain",
            call_id=state.pending_call_id,
            error=(
                "A provider call was pending at resume and no provider idempotency "
                "guarantee is available; refusing to repeat it."
            ),
        )
        raise LoopTransitionError("pending provider call is uncertain; resume failed closed.")

    # A probe is deterministic and read-only. If its adjacent record was already
    # committed, finish only the journal transition; otherwise execute it once
    # from the completed typed decision.
    if state.pending_probe_id is not None:
        _recover_pending_probe(
            journal,
            state.pending_probe_id,
            plan=plan,
            query_engine=query_engine,
            catalog=catalog,
        )

    while True:
        state = journal.rebuild()
        assert state is not None
        steps, probe_records = _load_completed_probe_steps(journal)
        accumulated = [
            *primary_findings,
            *(finding for step in steps for finding in step.findings),
        ]
        if len(steps) >= 2 and all(step.status == "failed" for step in steps[-2:]):
            return _commit_terminal_result(
                journal,
                _build_result(
                    plan,
                    max_steps=max_steps,
                    llm_call_cap=llm_call_cap,
                    steps=steps,
                    exit_reason="probe_error_cap_reached",
                    llm_calls_used=state.llm_calls_settled,
                    probe_errors=sum(step.status == "failed" for step in steps),
                    conclusion_note="",
                ),
            )
        if len(steps) >= 2 and all(step.status == "skipped" for step in steps[-2:]):
            return _commit_terminal_result(
                journal,
                _build_result(
                    plan,
                    max_steps=max_steps,
                    llm_call_cap=llm_call_cap,
                    steps=steps,
                    exit_reason="repeated_action",
                    llm_calls_used=state.llm_calls_settled,
                    probe_errors=sum(step.status == "failed" for step in steps),
                    conclusion_note="",
                ),
            )
        if state.remaining_probe_budget <= 0:
            return _commit_terminal_result(
                journal,
                _build_result(
                    plan,
                    max_steps=max_steps,
                    llm_call_cap=llm_call_cap,
                    steps=steps,
                    exit_reason="step_cap_reached",
                    llm_calls_used=state.llm_calls_settled,
                    probe_errors=sum(step.status == "failed" for step in steps),
                    conclusion_note="",
                ),
            )
        if state.remaining_call_budget <= 0:
            return _commit_terminal_result(
                journal,
                _build_result(
                    plan,
                    max_steps=max_steps,
                    llm_call_cap=llm_call_cap,
                    steps=steps,
                    exit_reason="llm_cap_reached",
                    llm_calls_used=state.llm_calls_settled,
                    probe_errors=sum(step.status == "failed" for step in steps),
                    conclusion_note="",
                ),
            )

        iteration = state.next_iteration
        decision = _completed_decision(journal, iteration)
        if decision is None:
            payload = _build_payload(
                plan,
                primary_findings=primary_findings,
                steps=steps,
                allowed_tables=set(catalog.relations.values()),
                probes_remaining=state.remaining_probe_budget,
                llm_calls_remaining=state.remaining_call_budget,
                failure_history=[
                    _failure_entry(_probe_fingerprint(step.sql), step.error)
                    for step in steps
                    if step.error
                ][-_FAILURE_HISTORY_LIMIT:],
                repeat_notice="",
            )
            call_id = make_loop_call_id(plan.investigation_id, iteration)
            journal.append_new(
                "decision_call_started",
                iteration=iteration,
                call_id=call_id,
            )
            try:
                with logical_llm_call(call_id):
                    decision = llm.structured(
                        task=_TASK,
                        schema=_ProbeDecision,
                        payload=payload,
                    )
            except BudgetExceeded as exc:
                error = f"{type(exc).__name__}: {str(exc)[:300]}"
                if getattr(exc, "stage", None) == "settlement":
                    journal.append_new(
                        "loop_call_uncertain",
                        iteration=iteration,
                        call_id=call_id,
                        error=error,
                    )
                    raise LoopTransitionError(
                        "provider call exceeded budget during settlement; refusing retry."
                    ) from exc
                journal.append_new(
                    "decision_call_rejected",
                    iteration=iteration,
                    call_id=call_id,
                    error=error,
                )
                return _commit_terminal_result(
                    journal,
                    _build_result(
                        plan,
                        max_steps=max_steps,
                        llm_call_cap=llm_call_cap,
                        steps=steps,
                        exit_reason="budget_exhausted",
                        llm_calls_used=state.llm_calls_settled,
                        probe_errors=sum(step.status == "failed" for step in steps),
                        conclusion_note="",
                    ),
                    terminal_event="loop_budget_exhausted",
                )
            except Exception as exc:
                journal.append_new(
                    "loop_call_uncertain",
                    iteration=iteration,
                    call_id=call_id,
                    error=(
                        f"{type(exc).__name__}: provider completion state is unknown "
                        "without an idempotency guarantee"
                    ),
                )
                raise LoopTransitionError(
                    "provider call failed with uncertain completion; refusing retry."
                ) from exc
            journal.append_new(
                "decision_call_completed",
                iteration=iteration,
                call_id=call_id,
                step_id=make_loop_step_id("decision", call_id),
                response_hash=stable_hash(decision.model_dump(mode="json"), length=32),
                typed_decision=decision.model_dump(mode="json"),
            )

        if decision.action == "conclude":
            ok, _reason = validate_interpretation_text(decision.rationale, accumulated)
            conclusion_note = decision.rationale.strip() if ok else ""
            concluding_step = LoopStepRecord(
                step_index=len(steps),
                action="conclude",
                purpose=decision.purpose.strip() or "Concluded the bounded investigation.",
                sql="",
                findings=[],
                status="succeeded",
            )
            latest = journal.rebuild()
            assert latest is not None
            return _commit_terminal_result(
                journal,
                _build_result(
                    plan,
                    max_steps=max_steps,
                    llm_call_cap=llm_call_cap,
                    steps=[*steps, concluding_step],
                    exit_reason="concluded",
                    llm_calls_used=latest.llm_calls_settled,
                    probe_errors=sum(step.status == "failed" for step in steps),
                    conclusion_note=conclusion_note,
                ),
            )

        fingerprint = _probe_fingerprint(decision.sql)
        prior_step = next(
            (
                record.step.step_index
                for record in probe_records
                if _probe_fingerprint(record.step.sql) == fingerprint
            ),
            None,
        )
        probe_id = make_loop_probe_id(plan.investigation_id, iteration, fingerprint)
        journal.append_new(
            "probe_started",
            iteration=iteration,
            probe_id=probe_id,
            probe_fingerprint=fingerprint,
        )
        if prior_step is not None:
            message = (
                "Probe rejected without execution: identical to the probe of step "
                f"{prior_step} after normalization."
            )
            step = LoopStepRecord(
                step_index=len(steps),
                action="probe",
                purpose=decision.purpose,
                sql=decision.sql,
                findings=[],
                status="skipped",
                error=message,
            )
            if trace_sink is not None:
                trace_sink(
                    trace_event(
                        session_id=plan.source_session_id,
                        event_type=PROBE_REPEATED_REJECTED,
                        name="investigation_loop",
                        summary={
                            "investigation_id": plan.investigation_id,
                            "step_index": step.step_index,
                            "duplicate_of_step": prior_step,
                            "fingerprint": fingerprint,
                        },
                    )
                )
            _commit_probe_record(journal, probe_id, iteration, step, None)
            continue

        _execute_and_commit_probe(
            journal,
            probe_id,
            iteration,
            decision,
            step_index=len(steps),
            plan=plan,
            query_engine=query_engine,
            catalog=catalog,
        )


def _execute_and_commit_probe(
    journal: JsonlLoopJournal,
    probe_id: str,
    iteration: int,
    decision: _ProbeDecision,
    *,
    step_index: int,
    plan: InvestigationPlan,
    query_engine: DuckDBQueryEngine,
    catalog: SqlCatalog,
) -> None:
    purpose = decision.purpose.strip()
    sql = decision.sql.strip()
    violation = _guard_probe(sql, set(catalog.relations.values()), query_engine)
    if violation is not None:
        _commit_probe_record(
            journal,
            probe_id,
            iteration,
            LoopStepRecord(
                step_index=step_index,
                action="probe",
                purpose=purpose,
                sql=sql,
                findings=[],
                status="failed",
                error=violation,
            ),
            None,
        )
        return
    try:
        sql_artifact = run_sql(
            catalog,
            sql,
            project_id=_LOOP_PROJECT,
            session_id=plan.source_session_id,
        )
    except (
        UnsafeQueryError,
        QueryTimeout,
        SqlBindingError,
        duckdb.Error,
        ValueError,
        RuntimeError,
    ) as exc:
        error_text = f"{type(exc).__name__}: {str(exc)[:200]}"
        _commit_probe_record(
            journal,
            probe_id,
            iteration,
            LoopStepRecord(
                step_index=step_index,
                action="probe",
                purpose=purpose,
                sql=sql,
                findings=[],
                status="failed",
                error=error_text,
            ),
            None,
        )
        return

    findings = _reduce_probe_findings(purpose, plan, sql_artifact, step_index)
    _commit_probe_record(
        journal,
        probe_id,
        iteration,
        LoopStepRecord(
            step_index=step_index,
            action="probe",
            purpose=purpose,
            sql=sql,
            result_artifact_id=sql_artifact.id,
            findings=findings,
            status="succeeded",
        ),
        sql_artifact,
    )


def _recover_pending_probe(
    journal: JsonlLoopJournal,
    probe_id: str,
    *,
    plan: InvestigationPlan,
    query_engine: DuckDBQueryEngine,
    catalog: SqlCatalog,
) -> None:
    state = journal.rebuild()
    assert state is not None
    started = next(
        (
            event
            for event in reversed(journal.events())
            if event.event_type == "probe_started" and event.probe_id == probe_id
        ),
        None,
    )
    if started is None or started.iteration is None:
        raise LoopTransitionError("pending probe has no recoverable start event.")
    artifact_ref = state.step_artifact_refs.get(probe_id)
    if artifact_ref is not None:
        _load_probe_record(journal, artifact_ref, expected_probe_id=probe_id)
        journal.append_new(
            "probe_completed",
            iteration=started.iteration,
            probe_id=probe_id,
            step_id=make_loop_step_id("probe", probe_id),
        )
        return
    adjacent_path = journal.path.parent / f"{probe_id}.result.json"
    if adjacent_path.exists():
        _load_probe_record(journal, adjacent_path.name, expected_probe_id=probe_id)
        journal.append_new(
            "artifact_committed",
            iteration=started.iteration,
            probe_id=probe_id,
            artifact_ref=adjacent_path.name,
        )
        journal.append_new(
            "probe_completed",
            iteration=started.iteration,
            probe_id=probe_id,
            step_id=make_loop_step_id("probe", probe_id),
        )
        return
    decision = _completed_decision(journal, started.iteration)
    if decision is None or decision.action != "probe":
        raise LoopTransitionError("pending probe has no completed typed decision.")
    completed_steps, _records = _load_completed_probe_steps(journal)
    _execute_and_commit_probe(
        journal,
        probe_id,
        started.iteration,
        decision,
        step_index=len(completed_steps),
        plan=plan,
        query_engine=query_engine,
        catalog=catalog,
    )


def _commit_probe_record(
    journal: JsonlLoopJournal,
    probe_id: str,
    iteration: int,
    step: LoopStepRecord,
    sql_artifact: Artifact | None,
) -> None:
    state = journal.rebuild()
    assert state is not None
    record = _PersistedProbeRecord(
        investigation_id=state.investigation_id,
        probe_id=probe_id,
        iteration=iteration,
        step=step,
        sql_artifact=sql_artifact,
    )
    path = journal.path.parent / f"{probe_id}.result.json"
    _write_model_atomic(journal, path, record)
    journal.append_new(
        "artifact_committed",
        iteration=iteration,
        probe_id=probe_id,
        artifact_ref=path.name,
    )
    journal.append_new(
        "probe_completed",
        iteration=iteration,
        probe_id=probe_id,
        step_id=make_loop_step_id("probe", probe_id),
    )


def _load_completed_probe_steps(
    journal: JsonlLoopJournal,
) -> tuple[list[LoopStepRecord], list[_PersistedProbeRecord]]:
    events = journal.events()
    completed_probe_ids = {
        event.probe_id
        for event in events
        if event.event_type == "probe_completed" and event.probe_id is not None
    }
    refs = {
        event.probe_id: event.artifact_ref
        for event in events
        if event.event_type == "artifact_committed"
        and event.probe_id is not None
        and event.artifact_ref is not None
    }
    records = [
        _load_probe_record(journal, refs[probe_id], expected_probe_id=probe_id)
        for probe_id in completed_probe_ids
        if probe_id in refs
    ]
    if len(records) != len(completed_probe_ids):
        raise LoopTransitionError("completed probe is missing its committed result reference.")
    records.sort(key=lambda record: record.step.step_index)
    return [record.step for record in records], records


def _load_probe_record(
    journal: JsonlLoopJournal,
    artifact_ref: str,
    *,
    expected_probe_id: str,
) -> _PersistedProbeRecord:
    path = _safe_journal_ref(journal, artifact_ref)
    try:
        record = _PersistedProbeRecord.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise LoopTransitionError(
            f"committed probe result {artifact_ref!r} is unavailable or invalid."
        ) from exc
    state = journal.rebuild()
    if (
        state is None
        or record.investigation_id != state.investigation_id
        or record.probe_id != expected_probe_id
    ):
        raise LoopTransitionError(
            f"committed probe result {artifact_ref!r} has the wrong identity."
        )
    return record


def _completed_decision(
    journal: JsonlLoopJournal,
    iteration: int,
) -> _ProbeDecision | None:
    event = next(
        (
            event
            for event in reversed(journal.events())
            if event.event_type == "decision_call_completed"
            and event.iteration == iteration
        ),
        None,
    )
    if event is None:
        return None
    if event.typed_decision is None:
        raise LoopTransitionError("completed decision is missing its typed payload.")
    try:
        return _ProbeDecision.model_validate(event.typed_decision)
    except ValidationError as exc:
        raise LoopTransitionError("completed decision payload is invalid.") from exc


def _commit_terminal_result(
    journal: JsonlLoopJournal,
    result: DeepInvestigationResult,
    *,
    terminal_event: Literal["loop_concluded", "loop_budget_exhausted"] = "loop_concluded",
) -> DeepInvestigationResult:
    path = journal.path.parent / "terminal-result.json"
    _write_model_atomic(journal, path, result)
    state = journal.append_new(terminal_event, final_draft_ref=path.name)
    journal.write_snapshot(state)
    return result


def _load_terminal_result(
    journal: JsonlLoopJournal,
    result_ref: str | None,
) -> DeepInvestigationResult:
    if result_ref is None:
        raise LoopTransitionError("terminal journal is missing its result reference.")
    path = _safe_journal_ref(journal, result_ref)
    try:
        return DeepInvestigationResult.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise LoopTransitionError(
            "terminal result reference is unavailable or invalid."
        ) from exc


def _safe_journal_ref(journal: JsonlLoopJournal, artifact_ref: str) -> Path:
    """Resolve only flat, journal-adjacent references created by this loop."""
    relative = Path(artifact_ref)
    if relative.is_absolute() or relative.name != artifact_ref:
        raise LoopTransitionError(f"unsafe journal artifact reference {artifact_ref!r}.")
    base = journal.path.parent.resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise LoopTransitionError(f"unsafe journal artifact reference {artifact_ref!r}.")
    return path


def _write_model_atomic(
    journal: JsonlLoopJournal,
    path: Path,
    model: BaseModel,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = model.model_dump_json(indent=2).encode("utf-8")
    with journal.fenced_side_effect() as attempt_epoch:
        temporary = path.with_name(
            f".{path.name}.epoch-{attempt_epoch}.{uuid4().hex}.tmp"
        )
        try:
            with temporary.open("xb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _guard_probe(
    sql: str,
    allowed_tables: set[str],
    query_engine: DuckDBQueryEngine,
) -> str | None:
    """Return a rejection reason after scope, read-only, and binding checks."""
    if not sql:
        return "empty probe SQL"
    out_of_scope = sorted(_referenced_tables(sql) - allowed_tables - _cte_names(sql))
    if out_of_scope:
        return (
            "probe references tables outside the approved scope: "
            + ", ".join(out_of_scope)
        )
    try:
        validate_select_statement(sql)
        query_engine.dry_run(sql)
    except (UnsafeQueryError, SqlBindingError) as exc:
        return f"{type(exc).__name__}: {str(exc)[:200]}"
    return None


def _referenced_tables(sql: str) -> set[str]:
    return {_bare_identifier(match) for match in _TABLE_RE.findall(sql)}


def _cte_names(sql: str) -> set[str]:
    return {name.lower() for name in _CTE_RE.findall(sql)}


def _bare_identifier(token: str) -> str:
    # Normalize optional schema qualification and quoting.
    return token.split(".")[-1].strip('"').lower()


def _reduce_probe_findings(
    purpose: str,
    plan: InvestigationPlan,
    sql_artifact: object,
    index: int,
) -> list[QuestionFinding]:
    """Reduce probe SQL results through the shared deterministic finding reducer."""
    candidate = QuestionCandidate(
        question_id=f"probe_{plan.investigation_id}_{index}",
        question_en=_strip_number_tokens(purpose),
        origin="llm",
        target_datasets=list(plan.target_datasets),
        score=QuestionScore(
            data_availability=0.0,
            statistical_signal=0.0,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.0,
        ),
    )
    return question_exec._findings(candidate, sql_artifact)  # type: ignore[arg-type]


_NUMBER_TOKEN_PATTERN = re.compile(r"(?<![\w.-])-?\d+(?:\.\d+)?%?")


def _strip_number_tokens(text: str) -> str:
    """Replace numeric tokens in LLM-authored probe text with a neutral marker."""
    stripped = _NUMBER_TOKEN_PATTERN.sub("[n]", text).strip()
    return stripped or "Follow-up probe"


def _build_payload(
    plan: InvestigationPlan,
    *,
    primary_findings: list[QuestionFinding],
    steps: list[LoopStepRecord],
    allowed_tables: set[str],
    probes_remaining: int,
    llm_calls_remaining: int,
    failure_history: list[str] | None = None,
    repeat_notice: str = "",
) -> dict:
    prior_probes = [
        {
            "purpose": step.purpose,
            "sql": step.sql,
            "status": step.status,
            "findings": [finding.text for finding in step.findings],
            "error": step.error,
        }
        for step in steps
        if step.action == "probe"
    ]
    payload: dict = {
        "instructions": (
            "You are extending an already-approved investigation with a few "
            "read-only follow-up probes. Decide ONE action and return it as JSON. "
            "action='probe' proposes the next probe: set 'purpose' (what you want "
            "to learn) and 'sql' (a single read-only SELECT). action='conclude' "
            "stops the loop: set 'rationale' (a short, business-language summary of "
            "what the findings mean). Hard rules: (1) SQL must be a single read-only "
            "SELECT and may reference ONLY the tables in allowed_tables; (2) do NOT "
            "invent numbers in the rationale — cite only figures already present in "
            "the findings; (3) make no causal claims; keep the rationale to at most "
            "three sentences. Prefer to conclude once the follow-ups add nothing new."
        ),
        "question": plan.question,
        "method_recipe": plan.method_recipe,
        "allowed_tables": sorted(allowed_tables),
        "target_datasets": list(plan.target_datasets),
        "primary_findings": [finding.text for finding in primary_findings],
        "prior_probes": prior_probes,
        "probes_remaining": probes_remaining,
        "llm_calls_remaining": llm_calls_remaining,
    }
    # Include only the most recent compressed failures.
    if failure_history:
        payload["failed_probes_note"] = _FAILURE_HISTORY_NOTE
        payload["failed_probes"] = list(failure_history[-_FAILURE_HISTORY_LIMIT:])
    if repeat_notice:
        payload["repeated_probe_notice"] = repeat_notice
    return payload


def _build_result(
    plan: InvestigationPlan,
    *,
    max_steps: int,
    llm_call_cap: int,
    steps: list[LoopStepRecord],
    exit_reason: LoopExitReason,
    llm_calls_used: int,
    probe_errors: int,
    conclusion_note: str,
) -> DeepInvestigationResult:
    return DeepInvestigationResult(
        investigation_id=plan.investigation_id,
        question_id=plan.question_id,
        max_steps=max_steps,
        llm_call_cap=llm_call_cap,
        steps=steps,
        exit_reason=exit_reason,
        llm_calls_used=llm_calls_used,
        probe_errors=probe_errors,
        conclusion_note=conclusion_note,
    )
