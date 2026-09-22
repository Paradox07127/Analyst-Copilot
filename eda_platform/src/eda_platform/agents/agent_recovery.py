"""Read-only discovery and explicit delivery acknowledgement for durable chat turns.

Discovery never invokes a graph or opens SQLite in write mode. Resume remains a
user action and uses the original execution identity through the normal driver.
"""

from __future__ import annotations

import heapq
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.func import entrypoint

from eda_platform.core.graph_execution import GraphPersistence

CHAT_GRAPH_DEFINITION = "chat_agent_tool_loop"
ResumeMode = Literal["tool", "structured", "approved"]
_GRAPH_KINDS: dict[ResumeMode, tuple[str, str]] = {
    "approved": ("chat-approved:", "chat-approved-functional-v1"),
    "tool": ("chat:", CHAT_GRAPH_DEFINITION),
    "structured": ("chat-structured:", "chat-structured-functional-v1"),
}
_MESSAGE_ID = re.compile(r"[0-9a-f]{32}\Z")
_GRAPH_FILE = re.compile(r"[0-9a-f]{32}\.sqlite\Z")
MAX_RECOVERABLE_TURNS = 20


@dataclass(frozen=True)
class RecoverableTurn:
    message_id: str
    mode: ResumeMode
    question: str
    updated_at: datetime
    status: Literal["interrupted", "awaiting_delivery", "blocked"]
    reason: str | None = None


def _graph_path(directory: Path, message_id: str, mode: ResumeMode) -> Path | None:
    if not _MESSAGE_ID.fullmatch(message_id):
        return None
    path = GraphPersistence(directory, _GRAPH_KINDS[mode][0] + message_id).path
    if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
        return None
    return path


def read_recoverable_turn(directory: Path, message_id: str) -> RecoverableTurn | None:
    # A structured fallback can follow a provider's rejected native-tool graph;
    # it is then the active execution for this logical turn.
    for mode in ("approved", "structured", "tool"):
        path = _graph_path(directory, message_id, mode)
        if path is not None:
            turn = _read_turn(path, message_id=message_id)
            if turn is not None:
                return turn
    return None


def _read_turn(path: Path, *, message_id: str | None = None) -> RecoverableTurn | None:
    from eda_platform.agents.runtime import build_agent_graph

    try:
        with closing(
            sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1)
        ) as connection:
            row = connection.execute(
                "SELECT execution_id, definition, delivered FROM execution_binding WHERE id=1"
            ).fetchone()
            if row is None or row[2]:
                return None
            mode: ResumeMode | None = None
            for kind, (_, definition) in _GRAPH_KINDS.items():
                if row[1] == definition:
                    mode = kind
                    break
            if mode is None:
                return None
            prefix = _GRAPH_KINDS[mode][0]
            execution_id = str(row[0])
            if not execution_id.startswith(prefix):
                return None
            found_id = execution_id.removeprefix(prefix)
            if not _MESSAGE_ID.fullmatch(found_id) or (message_id and message_id != found_id):
                return None
            if GraphPersistence(path.parent.parent, execution_id).path != path:
                return None
            serde = JsonPlusSerializer(pickle_fallback=False, allowed_msgpack_modules=None)
            saver = SqliteSaver(connection, serde=serde)
            saver.jsonplus_serde = serde
            saver.is_setup = True  # Read-only: skip SqliteSaver's CREATE TABLE/PRAGMA setup.
            config: RunnableConfig = {"configurable": {"thread_id": execution_id}}
            if mode == "tool":
                graph = build_agent_graph().compile(checkpointer=saver)
                snapshot = graph.get_state(config)
                if not snapshot.values:
                    return None
                messages = snapshot.values.get("messages", [])
                if len(messages) < 2 or messages[1].get("role") != "user":
                    return None
                question = messages[1].get("content")
                # The newer structured workflow supersedes a rejected tool
                # request in the same turn; avoid duplicate recovery cards.
                structured = _graph_path(path.parent.parent, found_id, "structured")
                if structured is not None and _read_turn(structured) is not None:
                    return None
            else:

                @entrypoint(checkpointer=saver)
                def model_workflow(_: dict) -> dict:
                    raise AssertionError("Recovery discovery cannot execute a workflow.")

                snapshot = model_workflow.get_state(config)
                if snapshot.created_at is None:
                    return None
                checkpoint = saver.get_tuple(config)
                initial = (
                    checkpoint.checkpoint.get("channel_values", {}).get("__start__", {})
                    if checkpoint
                    else {}
                )
                question = initial.get("message") if isinstance(initial, dict) else None
                if not question and isinstance(snapshot.values, dict):
                    question = snapshot.values.get("intent", {}).get("raw_message")
            if not isinstance(question, str) or not question.strip():
                return None
            uncertain = _unresolved_execution_tree(path.parent, execution_id)
            return RecoverableTurn(
                message_id=found_id,
                mode=mode,
                question=question[:4000],
                updated_at=datetime.fromisoformat(snapshot.created_at)
                if snapshot.created_at
                else datetime.fromtimestamp(path.stat().st_mtime, UTC),
                status="blocked"
                if uncertain
                else ("interrupted" if snapshot.next else "awaiting_delivery"),
                reason=(
                    "The previous model request has an unknown outcome. Send a new question "
                    "to start a separate execution."
                    if uncertain
                    else None
                ),
            )
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError, AttributeError):
        # A corrupt/incomplete/unrelated graph must not break transcript reads.
        return None


def list_recoverable_turns(directory: Path) -> list[RecoverableTurn]:
    graphs = directory / "graphs"
    if not graphs.is_dir() or graphs.is_symlink():
        return []

    def candidates():
        try:
            for path in graphs.iterdir():
                if not _GRAPH_FILE.fullmatch(path.name) or path.is_symlink():
                    continue
                turn = _read_turn(path)
                if turn is not None:
                    yield turn
        except OSError:
            # Session deletion can race a read-only recovery listing.
            return

    # Bound response and memory without hiding an old chat behind newer EDA
    # graph files. Non-chat metadata is rejected before loading checkpoints.
    return heapq.nlargest(MAX_RECOVERABLE_TURNS, candidates(), key=lambda turn: turn.updated_at)


def mark_turn_delivered(directory: Path, message_id: str) -> None:
    """Acknowledge only after the assistant transcript line has been persisted."""
    for mode, (prefix, definition) in _GRAPH_KINDS.items():
        path = _graph_path(directory, message_id, mode)
        if path is None:
            continue  # Deterministic/approved-plan turns may have no graph.
        with closing(
            sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=30)
        ) as connection:
            with connection:
                connection.execute(
                    "UPDATE execution_binding SET delivered=1 "
                    "WHERE id=1 AND execution_id=? AND definition=?",
                    (prefix + message_id, definition),
                )


def _unresolved_execution_tree(directory: Path, execution_id: str) -> bool:
    bindings: dict[str, tuple[str | None, bool]] = {}
    for path in directory.glob("*.sqlite"):
        if path.is_symlink() or not _GRAPH_FILE.fullmatch(path.name):
            continue
        try:
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1)) as conn:
                columns = {r[1] for r in conn.execute("PRAGMA table_info(execution_binding)")}
                parent = "parent_execution_id" if "parent_execution_id" in columns else "NULL"
                row = conn.execute(
                    f"SELECT execution_id, {parent} FROM execution_binding WHERE id=1"
                ).fetchone()
                if row is None:
                    continue
                effect_columns = {
                    r[1] for r in conn.execute("PRAGMA table_info(execution_effects)")
                }
                predicate = (
                    "kind != 'retry_safe'" if "kind" in effect_columns else "key LIKE 'model:%'"
                )
                pending = conn.execute(
                    f"SELECT 1 FROM execution_effects WHERE result IS NULL AND {predicate} LIMIT 1"
                ).fetchone()
                bindings[row[0]] = (row[1], pending is not None)
        except sqlite3.Error:
            continue
    pending_ids, visited = [execution_id], set()
    while pending_ids:
        current = pending_ids.pop()
        if current in visited:
            continue
        visited.add(current)
        if bindings.get(current, (None, False))[1]:
            return True
        pending_ids.extend(child for child, (parent, _) in bindings.items() if parent == current)
    return False
