"""Durable chat admission, delivery and monotonic event cursors."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from typing import Any

from eda_platform.core.store import ArtifactStore


class ChatAdmissionConflict(RuntimeError):
    pass


class ChatExecutionRepository:
    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    def admit(
        self,
        project: str,
        session: str,
        turn: str,
        payload: dict[str, Any],
        *,
        approval: tuple[str, str, str] | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self.store._session_write_transaction(project, session) as conn:
            if approval is not None:
                action, generation, digest = approval
                changed = conn.execute(
                    "UPDATE pending_actions SET status='consumed', consumed_idempotency_key=? "
                    "WHERE project_id=? AND session_id=? AND action_hash=? AND generation=? "
                    "AND payload_digest=? AND status='pending' AND expires_at>?",
                    (turn, project, session, action, generation, digest, now),
                )
                if changed.rowcount != 1:
                    raise ChatAdmissionConflict("Approval is no longer pending.")
            conn.execute(
                "INSERT INTO chat_executions"
                "(project_id,session_id,turn_id,payload,status,updated_at) "
                "VALUES(?,?,?,?,'queued',?)",
                (project, session, turn, json.dumps(payload, ensure_ascii=False), now),
            )

    def get(self, project: str, session: str, turn: str) -> dict[str, Any] | None:
        with closing(self.store._connect()) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM chat_executions WHERE project_id=? AND session_id=? AND turn_id=?",
                (project, session, turn),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        return result

    def unfinished(self, project: str, session: str) -> list[dict[str, Any]]:
        with closing(self.store._connect()) as conn:
            rows = conn.execute(
                "SELECT turn_id FROM chat_executions WHERE project_id=? AND session_id=? "
                "AND status NOT IN ('delivered','cancelled') ORDER BY updated_at DESC LIMIT 20",
                (project, session),
            ).fetchall()
        return [record for row in rows if (record := self.get(project, session, row[0]))]

    def begin(self, project: str, session: str, turn: str) -> int:
        with self.store._session_write_transaction(project, session) as conn:
            conn.execute(
                "UPDATE chat_executions SET attempt=attempt+1,status='running',updated_at=? "
                "WHERE project_id=? AND session_id=? AND turn_id=?",
                (datetime.now(UTC).isoformat(), project, session, turn),
            )
            row = conn.execute(
                "SELECT attempt FROM chat_executions "
                "WHERE project_id=? AND session_id=? AND turn_id=?",
                (project, session, turn),
            ).fetchone()
        return int(row[0]) if row else 1

    def finish(self, project: str, session: str, turn: str, status: str) -> None:
        with self.store._session_write_transaction(project, session) as conn:
            conn.execute(
                "UPDATE chat_executions SET status=?,updated_at=? "
                "WHERE project_id=? AND session_id=? AND turn_id=?",
                (status, datetime.now(UTC).isoformat(), project, session, turn),
            )

    def append_event(
        self,
        project: str,
        session: str,
        turn: str,
        event_type: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        with self.store._session_write_transaction(project, session) as conn:
            row = conn.execute(
                "SELECT coalesce(max(seq),0)+1 FROM chat_execution_events "
                "WHERE project_id=? AND session_id=? AND turn_id=?",
                (project, session, turn),
            ).fetchone()
            event = dict(
                seq=int(row[0]), session_id=session, message_id=turn, type=event_type, data=data
            )
            conn.execute(
                "INSERT INTO chat_execution_events VALUES(?,?,?,?,?)",
                (project, session, turn, event["seq"], json.dumps(event, ensure_ascii=False)),
            )
            if event_type in {"message.completed", "plan.pending"}:
                status = "cancelled" if data.get("status") == "cancelled" else "delivered"
                conn.execute(
                    "UPDATE chat_executions SET status=?,updated_at=? "
                    "WHERE project_id=? AND session_id=? AND turn_id=?",
                    (status, datetime.now(UTC).isoformat(), project, session, turn),
                )
        return event

    def events(self, project: str, session: str, turn: str, after: int) -> list[dict[str, Any]]:
        with closing(self.store._connect()) as conn:
            rows = conn.execute(
                "SELECT payload FROM chat_execution_events WHERE project_id=? AND session_id=? "
                "AND turn_id=? AND seq>? ORDER BY seq LIMIT 100",
                (project, session, turn, after),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def cursor(self, project: str, session: str, turn: str) -> int:
        with closing(self.store._connect()) as conn:
            return int(
                conn.execute(
                    "SELECT coalesce(max(seq),0) FROM chat_execution_events "
                    "WHERE project_id=? AND session_id=? AND turn_id=?",
                    (project, session, turn),
                ).fetchone()[0]
            )
