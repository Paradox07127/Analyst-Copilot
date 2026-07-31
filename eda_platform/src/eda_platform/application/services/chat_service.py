"""Chat use cases (§7.5 / 阶段5 slice J).

Transport shape: `send_message` accepts the turn (202) and runs it on a worker
thread; the client follows `GET .../chat/stream?message_id=` for progress. The
kernel driver `run_chat_turn` is synchronous and exposes no token callback, so
there is no token-level streaming: what streams is the driver's own trace
events (intent, plan, SQL, validation, LLM usage) plus one final message frame.
Those arrive live because the driver writes them through the ArtifactStore it
was handed — this service hands it a subclass that mirrors every append into
the session buffer.

Plan approval: a plan with `needs_approval` never executes on the first turn.
It is registered with ApprovalService (kind `chat_plan`) and surfaced as a
`plan.pending` frame; approving re-enters the driver via its `approved_plan`
path, which is the driver's own authorization boundary.

The session buffer is per-process and in memory, so turns fall into three
states across a restart: completed turns are durable (they land in the JSONL
transcript); turns still executing are lost with their stream; and a turn that
stopped at `awaiting_approval` is recoverable — its pending_actions row and its
transcript line both survive, and `list_pending_plans` reads the same durable
approval token so the client can finish it without rotating state on GET.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from eda_platform.application.dto import (
    ChatMessageAccepted,
    ChatMessagePage,
    ChatMessageView,
    ChatPendingPlan,
    ChatPendingPlanList,
    ChatPlanRejected,
    ChatStreamEvent,
)
from eda_platform.application.services.approval_service import (
    ApprovalExpiredError,
    ApprovalNotFoundError,
    ApprovalService,
    payload_digest,
)
from eda_platform.application.services.session_service import (
    InvalidCursorError,
    SessionNotFoundError,
)
from eda_platform.application.services.settings_service import EffectiveSettings
from eda_platform.core.bounded_pagination import (
    MAX_JSONL_RECORD_BYTES,
    JsonlPageIndex,
    decode_bound_cursor,
    encode_bound_cursor,
)
from eda_platform.core.budget import SessionBudgetPolicy
from eda_platform.core.env import load_llm_settings_from_env_file
from eda_platform.core.ids import INTERNAL_SESSION_MARKER
from eda_platform.core.llm import LLMClient, LLMSettings, OfflineLLMClient, create_llm_client
from eda_platform.core.llm_ledger import (
    BUDGET_EVENT_TYPES,
    meter_llm_client,
    restore_run_budget_state,
)
from eda_platform.core.permissions import analysis_plan_action
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.chat import ChatMessage
from eda_platform.schemas.plans import AnalysisPlan
from eda_platform.schemas.sessions import TraceEvent
from eda_platform.tools.evidence import PayloadPolicy

logger = logging.getLogger(__name__)

APPROVAL_KIND_CHAT_PLAN = "chat_plan"
DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 200
MAX_MESSAGE_CHARS = 4000
MAX_TRACE_VALUE_CHARS = 4000
MAX_RETAINED_SESSIONS = 32
# Mirrors the web client's own tool-call window (MAX_TOOL_CALLS): a runaway
# turn must not grow one session's buffer without bound.
MAX_SESSION_EVENTS = 100

TERMINAL_EVENT_TYPES = frozenset({"message.completed", "plan.pending", "turn.failed"})

# Driver trace events worth showing in a chat thread: what it decided, what it
# ran, and what the validator said. Bookkeeping events stay out of the UI.
_STREAMED_TRACE_TYPES = frozenset(
    {
        "agent_intent",
        "agent_plan",
        "agent_completed",
        "agent_limit_reached",
        "tool_started",
        "tool_completed",
        "tool_failed",
        "validator_result",
        "tool_guard_rejected",
        "permission_denied",
        "code_agent_attempt",
        "chat_turn_failed",
        "llm_call",
    }
)


class ChatServiceError(Exception):
    pass


class ChatValidationError(ChatServiceError):
    pass


class ChatRunBusyError(ChatServiceError):
    def __init__(self, session_id: str) -> None:
        super().__init__(
            f"Run {session_id} already has a chat turn in flight; wait for it to finish."
        )
        self.session_id = session_id


class ChatMessageNotFoundError(ChatServiceError):
    def __init__(self, message_id: str) -> None:
        super().__init__(f"No in-flight or recent chat turn with id {message_id}.")
        self.message_id = message_id


@dataclass
class _PendingPlan:
    plan_id: str
    action_hash: str
    approval_token: str
    expires_at: datetime
    plan: AnalysisPlan

    def frame(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "action_hash": self.action_hash,
            "approval_token": self.approval_token,
            "expires_at": self.expires_at.isoformat(),
            "question": self.plan.question,
            "method": self.plan.method,
            "sql": self.plan.sql,
            "dataset_names": list(self.plan.dataset_names),
            "estimated_scan": self.plan.estimated_scan,
        }

    def view(self) -> ChatPendingPlan:
        return ChatPendingPlan(
            plan_id=self.plan_id,
            action_hash=self.action_hash,
            approval_token=self.approval_token,
            expires_at=self.expires_at,
            question=self.plan.question,
            method=self.plan.method,
            sql=self.plan.sql,
            dataset_names=list(self.plan.dataset_names),
            estimated_scan=self.plan.estimated_scan,
        )


@dataclass
class ChatStreamPage:
    events: list[ChatStreamEvent]
    done: bool
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class _EffectiveChatSettings:
    """Request-time snapshot retained only by the in-flight turn."""

    llm: LLMSettings
    payload_policy: PayloadPolicy


@dataclass
class _TurnSession:
    message_id: str
    session_id: str
    project_id: str
    events: list[ChatStreamEvent] = field(default_factory=list)
    done: bool = False
    truncated: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    next_seq: int = 1

    def append(self, event_type: str, data: dict[str, Any]) -> None:
        with self.lock:
            event = ChatStreamEvent(
                seq=self.next_seq,
                session_id=self.session_id,
                message_id=self.message_id,
                type=event_type,
                data=data,
            )
            self.next_seq += 1
            self.events.append(event)
            if event_type in TERMINAL_EVENT_TYPES:
                self.done = True
            self._trim_locked()

    def after(self, seq: int) -> ChatStreamPage:
        with self.lock:
            return ChatStreamPage(
                events=[event for event in self.events if event.seq > seq],
                done=self.done,
                truncated=self.truncated,
            )

    def _trim_locked(self) -> None:
        """Drop the oldest non-terminal events past the cap. Seq numbers stay
        monotonic, so a live reader's cursor keeps working; only a reconnect
        that resumes from before the window loses the dropped frames."""
        while len(self.events) > MAX_SESSION_EVENTS:
            index = next(
                (
                    position
                    for position, event in enumerate(self.events)
                    if event.type not in TERMINAL_EVENT_TYPES
                ),
                None,
            )
            if index is None:
                return
            del self.events[index]
            self.truncated = True


class _TurnTracingStore(ArtifactStore):
    """ArtifactStore that mirrors every trace append into a chat session.

    Subclassing is how mid-turn tool visibility is obtained without editing the
    driver: `run_chat_turn` writes its trace through whatever store it is given.
    """

    def __init__(self, root: Path, session: _TurnSession) -> None:
        # The API's own store already opened this workspace; re-running the
        # schema setup on every turn buys nothing.
        super().__init__(root, init_db=False)
        self._session = session

    def append_trace(self, project_id: str, event: TraceEvent) -> Path:
        path = super().append_trace(project_id, event)
        if event.event_type in _STREAMED_TRACE_TYPES:
            self._session.append(
                "tool.call",
                {
                    "trace_type": event.event_type,
                    "name": event.name,
                    "summary": _truncate_summary(event.summary),
                },
            )
        return path


class _ApprovedReentryLLM:
    """`approved_plan` re-entry skips routing and planning, so the driver never
    calls the model; this stands in for the required parameter and fails loudly
    if that ever stops being true."""

    def structured(self, *, task: str, schema: Any, payload: dict) -> Any:
        raise AssertionError("LLM must not be called on approved_plan re-entry.")


class ChatService:
    def __init__(
        self,
        store: ArtifactStore,
        approvals: ApprovalService,
        *,
        budget_policy: SessionBudgetPolicy | None = None,
    ) -> None:
        self._store = store
        self._approvals = approvals
        self._budget_policy = budget_policy or SessionBudgetPolicy()
        self._sessions: OrderedDict[str, _TurnSession] = OrderedDict()
        self._sessions_lock = threading.Lock()

    def list_messages(
        self,
        session_id: str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        cursor: str | None = None,
    ) -> ChatMessagePage:
        """Newest-last page of the transcript, paging backwards from `cursor`.

        Only the page's own lines are parsed: the full transcript never becomes
        a response body, and older turns cost nothing until requested.
        """
        limit = max(1, min(limit, MAX_PAGE_LIMIT))
        project_id = self._project_for_run(session_id)
        index = JsonlPageIndex(self._store.db_path, self._store.root)
        path = self._transcript_path(project_id, session_id)
        scope = f"chat:{project_id}:{session_id}"
        current_version = index.file_source_version(path)
        end = (
            decode_bound_cursor(
                cursor,
                scope=scope,
                source_version=current_version,
            )
            if cursor is not None
            else None
        )
        state = index.ensure(
            path,
            accept=lambda _line: True,
        )
        if cursor is not None and state.source_version != current_version:
            raise InvalidCursorError
        current_version = state.source_version
        end = state.valid_count if end is None else end
        if end > state.valid_count:
            raise InvalidCursorError
        page = index.page(state, start=end, limit=limit, reverse=True)
        start = page[0].ordinal if page else 0
        messages: list[ChatMessageView] = []
        for record in page:
            if record.oversized:
                messages.append(_omitted_message(record.ordinal))
                continue
            view = _parse_message(
                record.payload.decode("utf-8", errors="replace").strip(),
                record.ordinal,
            )
            if view is not None:
                messages.append(view)
        return ChatMessagePage(
            session_id=session_id,
            messages=messages,
            next_cursor=(
                encode_bound_cursor(
                    start,
                    scope=scope,
                    source_version=current_version,
                )
                if start > 0
                else None
            ),
            total=state.valid_count,
        )

    def send_message(
        self,
        session_id: str,
        *,
        text: str,
        llm: str = "env",
        effective_settings: EffectiveSettings | None = None,
    ) -> ChatMessageAccepted:
        text = text.strip()
        if not text:
            raise ChatValidationError("A chat message must not be empty.")
        if len(text) > MAX_MESSAGE_CHARS:
            raise ChatValidationError(
                f"A chat message must be at most {MAX_MESSAGE_CHARS} characters."
            )
        if llm not in {"env", "offline"}:
            raise ChatValidationError("llm must be 'env' or 'offline'.")
        effective = _freeze_effective_settings(effective_settings)
        project_id = self._project_for_run(session_id)
        session = self._reserve_session(session_id, project_id)
        try:
            self._append_message(
                project_id, session_id, ChatMessage(role="user", content=text)
            )
        except Exception:
            self._discard_session(session)
            raise
        self._spawn(
            session,
            lambda: self._run_turn(
                session,
                message=text,
                llm_mode=llm,
                effective=effective,
                approved=None,
            ),
        )
        return _accepted(session)

    def approve_plan(
        self, session_id: str, plan_id: str, *, action_hash: str, approval_token: str
    ) -> ChatMessageAccepted:
        project_id = self._project_for_run(session_id)
        session = self._reserve_session(session_id, project_id)
        try:
            plan, question = self._consume_plan(
                session_id, plan_id, action_hash, approval_token
            )

            def run_approved() -> None:
                self._run_turn(
                    session,
                    message=question,
                    llm_mode="offline",
                    effective=_freeze_effective_settings(None),
                    approved=(plan, action_hash),
                )
                if any(
                    event.type == "turn.failed"
                    for event in session.after(0).events
                ):
                    self._store.restore_pending_action(action_hash, session_id=session_id)

            with self._approvals.compensate_on_failure(action_hash, session_id=session_id):
                self._spawn(session, run_approved)
        except Exception:
            self._discard_session(session)
            raise
        return _accepted(session)

    def reject_plan(
        self, session_id: str, plan_id: str, *, action_hash: str, approval_token: str
    ) -> ChatPlanRejected:
        project_id = self._project_for_run(session_id)
        # Consume the approval even though nothing runs: the token must not stay
        # usable after the user said no.
        self._consume_plan(session_id, plan_id, action_hash, approval_token)
        with self._approvals.compensate_on_failure(action_hash, session_id=session_id):
            self._append_message(
                project_id,
                session_id,
                ChatMessage(
                    role="assistant",
                    content="Cancelled the pending analysis plan.",
                    status="refused",
                ),
            )
        return ChatPlanRejected(session_id=session_id, plan_id=plan_id)

    def list_pending_plans(self, session_id: str) -> ChatPendingPlanList:
        """Return persisted pending plans without mutating approval state."""

        self._project_for_run(session_id)
        now = datetime.now(UTC).isoformat()
        plans: list[ChatPendingPlan] = []
        for row in self._store.list_pending_actions(
            session_id=session_id, kind=APPROVAL_KIND_CHAT_PLAN
        ):
            if str(row["expires_at"]) <= now:
                continue
            payload = _load_payload(row["payload_json"])
            if payload_digest(payload) != str(row["payload_digest"]):
                raise ApprovalNotFoundError(str(row["action_hash"]))
            plan_id = str(payload.get("plan_id") or "")
            try:
                plan = AnalysisPlan.model_validate(payload.get("plan"))
            except ValidationError:
                continue
            if not plan_id:
                continue
            digest = str(row["action_hash"])
            token = str(row["generation"])
            try:
                expires_at = datetime.fromisoformat(str(row["expires_at"]))
            except ValueError:
                raise ApprovalExpiredError(digest) from None
            if not token:
                raise ApprovalExpiredError(digest)
            plans.append(
                _PendingPlan(
                    plan_id=plan_id,
                    action_hash=digest,
                    approval_token=token,
                    expires_at=expires_at,
                    plan=plan,
                ).view()
            )
        return ChatPendingPlanList(session_id=session_id, plans=plans)

    def events_after(self, session_id: str, message_id: str, after_seq: int) -> ChatStreamPage:
        session = self._session(session_id, message_id)
        return session.after(after_seq)

    def require_session(self, session_id: str, message_id: str) -> None:
        self._session(session_id, message_id)

    def _consume_plan(
        self, session_id: str, plan_id: str, action_hash: str, approval_token: str
    ) -> tuple[AnalysisPlan, str]:
        def validate(payload: dict[str, Any]) -> tuple[AnalysisPlan, str]:
            if str(payload.get("plan_id", "")) != plan_id:
                raise ChatValidationError(
                    "The approval was prepared for a different plan than the request path."
                )
            try:
                plan = AnalysisPlan.model_validate(payload.get("plan"))
            except ValidationError as exc:
                raise ChatValidationError(
                    "The approved plan is no longer readable."
                ) from exc
            return plan, str(payload.get("question") or plan.question)

        _payload, validated = self._approvals.validate_then_consume(
            action_hash,
            kind=APPROVAL_KIND_CHAT_PLAN,
            session_id=session_id,
            generation=approval_token,
            validate=validate,
        )
        return validated

    def _run_turn(
        self,
        session: _TurnSession,
        *,
        message: str,
        llm_mode: str,
        effective: _EffectiveChatSettings,
        approved: tuple[AnalysisPlan, str] | None,
    ) -> None:
        # Local import: the driver pulls the full agent stack, which must not be
        # paid at API import time (same rationale as worker/runner).
        from eda_platform.core.session_loader import load_run
        from eda_platform.drivers.chat import run_chat_turn

        session.append(
            "turn.started",
            {"stage": "loading_datasets", "approved": approved is not None},
        )
        try:
            loaded = load_run(
                session.project_id, session.session_id, workspace=self._store.root
            )
            result = loaded.result
            if result is None or not result.loaded_datasets:
                session.append(
                    "turn.failed",
                    {
                        "code": "datasets_unavailable",
                        "message": (
                            "This run's source data could not be reloaded, so chat "
                            "cannot plan against it. Re-run the analysis first."
                        ),
                    },
                )
                return
            session.append(
                "progress",
                {"stage": "executing" if approved is not None else "planning"},
            )
            traced = _TurnTracingStore(self._store.root, session)
            if approved is not None:
                llm: Any = _ApprovedReentryLLM()
            else:
                raw_llm = _build_llm(llm_mode, effective.llm)
                earliest = self._store.earliest_trace_started_at(
                    project_id=session.project_id, session_id=session.session_id
                )
                restored_budget = restore_run_budget_state(
                    self._budget_policy,
                    self._store.list_trace_events(
                        project_id=session.project_id,
                        session_id=session.session_id,
                        event_types=BUDGET_EVENT_TYPES,
                    ),
                    run_started_at=(
                        None if earliest is None else datetime.fromisoformat(earliest)
                    ),
                )

                def emit_usage(event: TraceEvent) -> None:
                    traced.append_trace(session.project_id, event)

                llm = meter_llm_client(
                    raw_llm,
                    session_id=session.session_id,
                    emit=emit_usage,
                    budget=restored_budget,
                    session_dir=self._store.session_dir(
                        session.project_id, session.session_id
                    ),
                )
            turn = run_chat_turn(
                message,
                datasets=result.loaded_datasets,
                project_id=session.project_id,
                session_id=session.session_id,
                llm=llm,  # type: ignore[arg-type]
                artifacts=result.artifacts,
                store=traced,
                approved_plan=approved[0] if approved else None,
                approved_action_hash=approved[1] if approved else None,
                payload_policy=effective.payload_policy,
            )
        except Exception as exc:  # last-resort guard: never leak a traceback
            logger.exception("Chat turn %s failed", session.message_id)
            session.append(
                "turn.failed",
                {"code": type(exc).__name__, "message": _short(str(exc))},
            )
            return

        artifact_refs = [artifact.id for artifact in turn.artifacts]
        awaiting = turn.status == "awaiting_approval" and turn.plan is not None
        # Register before persisting: the transcript line carries the plan's
        # identity so a client that lost its stream can find it again.
        pending = (
            self._register_pending_plan(session, turn.plan, turn.artifacts, llm_mode)
            if awaiting and turn.plan is not None
            else None
        )
        try:
            self._append_message(
                session.project_id,
                session.session_id,
                ChatMessage(
                    role="assistant",
                    content=turn.message,
                    status=turn.status,
                    sql=turn.sql,
                    artifact_refs=artifact_refs,
                    plan_id=pending.plan_id if pending else None,
                    action_hash=pending.action_hash if pending else None,
                    expires_at=pending.expires_at if pending else None,
                ),
            )
        except Exception as exc:
            logger.exception("Chat transcript write failed for %s", session.message_id)
            session.append(
                "turn.failed",
                {
                    "code": "transcript_write_failed",
                    "message": (
                        "The answer was produced but could not be written to this "
                        f"run's transcript: {_short(str(exc))}"
                    ),
                },
            )
            return

        if awaiting:
            if pending is None:
                session.append(
                    "turn.failed",
                    {
                        "code": "plan_unavailable",
                        "message": "The plan awaiting approval could not be identified.",
                    },
                )
            else:
                session.append("plan.pending", pending.frame())
            return
        session.append(
            "message.completed",
            {
                "role": "assistant",
                "content": turn.message,
                "status": turn.status,
                "sql": turn.sql,
                "artifact_refs": artifact_refs,
                "validation": (
                    turn.validation.model_dump(mode="json") if turn.validation else None
                ),
            },
        )

    def _register_pending_plan(
        self,
        session: _TurnSession,
        plan: AnalysisPlan,
        artifacts: list[Any],
        llm_mode: str,
    ) -> _PendingPlan | None:
        plan_id = next(
            (
                artifact.id
                for artifact in artifacts
                if artifact.type is ArtifactType.CHAT_TURN_PLAN
            ),
            "",
        )
        if not plan_id:
            return None
        digest, token, expires_at = self._approvals.register(
            kind=APPROVAL_KIND_CHAT_PLAN,
            session_id=session.session_id,
            project_id=session.project_id,
            action=analysis_plan_action(plan),
            payload={
                "plan_id": plan_id,
                "plan": plan.model_dump(mode="json"),
                "question": plan.question,
                "llm": llm_mode,
            },
        )
        return _PendingPlan(
            plan_id=plan_id,
            action_hash=digest,
            approval_token=token,
            expires_at=expires_at,
            plan=plan,
        )

    def _spawn(self, session: _TurnSession, target: Any) -> None:
        def guarded() -> None:
            try:
                target()
            finally:
                if not session.done:
                    session.append(
                        "turn.failed",
                        {"code": "turn_aborted", "message": "The chat turn ended early."},
                    )

        threading.Thread(
            target=guarded, name=f"chat-turn-{session.message_id}", daemon=True
        ).start()

    def _reserve_session(self, session_id: str, project_id: str) -> _TurnSession:
        """Claim the run's single turn slot and register its session atomically.

        Checking for a live turn and registering the new one must happen under
        one acquisition: with the check and the registration separated, two
        concurrent sends both pass the check and both start a turn, and the
        first one's stream is never handed to anybody.
        """
        session = _TurnSession(
            message_id=uuid.uuid4().hex, session_id=session_id, project_id=project_id
        )
        with self._sessions_lock:
            for existing in self._sessions.values():
                if existing.session_id == session_id and not existing.done:
                    raise ChatRunBusyError(session_id)
            self._sessions[session.message_id] = session
            while len(self._sessions) > MAX_RETAINED_SESSIONS:
                _, evicted = self._sessions.popitem(last=False)
                if not evicted.done:
                    # Never evict a live turn: put it back and drop the next one.
                    self._sessions[evicted.message_id] = evicted
                    self._sessions.move_to_end(evicted.message_id)
                    break
        return session

    def _discard_session(self, session: _TurnSession) -> None:
        """Release a reservation whose turn never started, so the run does not
        stay busy after a failed transcript write or a rejected approval."""
        with self._sessions_lock:
            self._sessions.pop(session.message_id, None)

    def _session(self, session_id: str, message_id: str) -> _TurnSession:
        with self._sessions_lock:
            session = self._sessions.get(message_id)
        if session is None or session.session_id != session_id:
            raise ChatMessageNotFoundError(message_id)
        return session

    def _append_message(self, project_id: str, session_id: str, message: ChatMessage) -> None:
        from eda_platform.drivers.chat import append_chat_message

        append_chat_message(self._store, project_id, session_id, message)

    def _transcript_path(self, project_id: str, session_id: str) -> Path:
        """Mirrors drivers.chat._chat_session_path — one session per run. The
        driver owns writes; reads live here so pagination stays lazy."""
        return self._store.project_dir(project_id) / "chat" / f"{session_id}.jsonl"

    def _project_for_run(self, session_id: str) -> str:
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        row = self._store.get_session_index_row(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        return str(row["project_id"])


def _accepted(session: _TurnSession) -> ChatMessageAccepted:
    return ChatMessageAccepted(
        session_id=session.session_id,
        message_id=session.message_id,
        stream_url=(
            f"/api/v1/sessions/{session.session_id}/chat/stream?message_id={session.message_id}"
        ),
    )


def _freeze_effective_settings(
    effective: EffectiveSettings | None,
) -> _EffectiveChatSettings:
    if effective is None:
        settings = load_llm_settings_from_env_file()
        policy: PayloadPolicy = "schema+aggregates"
    else:
        settings = effective.llm
        policy = cast(PayloadPolicy, effective.payload_policy)
    return _EffectiveChatSettings(
        llm=settings.model_copy(deep=True),
        payload_policy=policy,
    )


def _build_llm(mode: str, settings: LLMSettings) -> LLMClient:
    """Build from the request-time settings snapshot, never mutable process env."""
    if mode == "offline":
        return OfflineLLMClient()
    return create_llm_client(settings)


def _omitted_message(seq: int) -> ChatMessageView:
    """Stands in for a transcript line the index refused to read."""
    return ChatMessageView(
        seq=seq,
        role="assistant",
        content=(
            f"This message is too large to display "
            f"(over {MAX_JSONL_RECORD_BYTES // (1024 * 1024)} MiB)."
        ),
        status="omitted",
    )


def _parse_message(line: str, seq: int) -> ChatMessageView | None:
    try:
        message = ChatMessage.model_validate_json(line)
    except ValidationError:
        return None
    return ChatMessageView(
        seq=seq,
        role=message.role,
        content=message.content,
        status=message.status,
        sql=message.sql,
        artifact_refs=list(message.artifact_refs),
        created_at=message.created_at,
    )


def _truncate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _short(value, MAX_TRACE_VALUE_CHARS) if isinstance(value, str) else value
        for key, value in summary.items()
    }


def _short(text: str, limit: int = 300) -> str:
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _load_payload(raw: Any) -> dict[str, Any]:
    try:
        payload = json.loads(str(raw))
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}
