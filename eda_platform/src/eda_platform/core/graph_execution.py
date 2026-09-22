"""Local LangGraph persistence and replay-safe effect commit boundaries."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import tracing_context

from eda_platform.core.effect_policy import durable_model_request
from eda_platform.core.file_lock import exclusive_lock
from eda_platform.core.fs import BINARY_FLAG, fsync_directory
from eda_platform.core.graph_trace import GraphTraceHandler
from eda_platform.core.ids import stable_hash
from eda_platform.core.trace_correlation import (
    current_trace_execution,
    emit_execution_event,
    trace_execution_scope,
)

GRAPH_VERSION = "langgraph-v1"
_RUNTIME_VERSION = version("langgraph")
_SQLITE_VERSION = version("langgraph-checkpoint-sqlite")


class GraphIdentityError(RuntimeError):
    pass


class GraphEffectUncertain(RuntimeError):
    pass


@dataclass(frozen=True)
class GraphPersistence:
    directory: Path
    execution_id: str
    fingerprint: str = ""

    @property
    def path(self) -> Path:
        return self.directory / "graphs" / f"{stable_hash(self.execution_id, length=32)}.sqlite"


@dataclass
class GraphExecution:
    saver: SqliteSaver | None
    config: RunnableConfig
    path: Path | None = None

    def effect(
        self,
        key: str,
        request: dict[str, Any],
        call: Callable[[], dict[str, Any]],
        *,
        retry_pending: bool = False,
    ) -> dict[str, Any]:
        """Commit effects before checkpointing; retry only idempotent local services."""
        if self.path is None:
            return call()
        digest = stable_hash(request, length=32)
        with _connection(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT digest, result FROM execution_effects WHERE key = ?", (key,)
            ).fetchone()
            if row is not None:
                if row[0] != digest:
                    raise GraphIdentityError("An execution effect changed its request on resume.")
                if row[1] is None and not retry_pending:
                    raise GraphEffectUncertain(
                        "The previous effect outcome is unknown; start a new execution."
                    )
                if row[1] is not None:
                    emit_execution_event(
                        "graph.effect_reused", key, {"digest": digest}, effect_id=key
                    )
                    return json.loads(row[1])
            else:
                connection.execute(
                    "INSERT INTO execution_effects(key, digest, kind) VALUES (?, ?, ?)",
                    (key, digest, "retry_safe" if retry_pending else "guarded"),
                )
        with trace_execution_scope(effect_id=key):
            emit_execution_event("graph.effect_started", key, {"digest": digest})
            result = call()
        body = json.dumps(result, ensure_ascii=False, allow_nan=False)
        with _connection(self.path) as connection:
            connection.execute("UPDATE execution_effects SET result = ? WHERE key = ?", (body, key))
        emit_execution_event("graph.effect_completed", key, {"digest": digest}, effect_id=key)
        return result

    def model_effect(
        self,
        key: str,
        request: dict[str, Any],
        call: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        token = durable_model_request.set(self.path is not None)
        try:
            return self.effect(key, request, call)
        finally:
            durable_model_request.reset(token)


@contextmanager
def _connection(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path, timeout=30, check_same_thread=False)
    try:
        connection.execute("PRAGMA synchronous=FULL")
        with connection:
            yield connection
    finally:
        connection.close()


@contextmanager
def _open_graph_execution(
    persistence: GraphPersistence | None,
    *,
    definition: str,
    inputs: dict[str, Any],
    recursion_limit: int = 1000,
    checkpoint_types: tuple[type, ...] = (),
) -> Iterator[GraphExecution]:
    if persistence is None:
        yield GraphExecution(
            None, {"recursion_limit": recursion_limit, "callbacks": [GraphTraceHandler()]}
        )
        return
    path = persistence.path
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or path.parent.is_symlink():
        raise GraphIdentityError("Graph storage cannot be a symbolic link.")
    descriptor = os.open(
        path.with_suffix(".lock"),
        os.O_CREAT | os.O_RDWR | BINARY_FLAG | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with exclusive_lock(descriptor):
            try:
                database_fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR | BINARY_FLAG, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(database_fd)
                fsync_directory(path.parent)
        with exclusive_lock(descriptor), _connection(path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS execution_binding "
                "(id INTEGER PRIMARY KEY CHECK(id = 1), digest TEXT NOT NULL, "
                "execution_id TEXT NOT NULL, definition TEXT NOT NULL, "
                "delivered INTEGER NOT NULL DEFAULT 0)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS execution_effects "
                "(key TEXT PRIMARY KEY, digest TEXT NOT NULL, result TEXT)"
            )
            if "kind" not in {
                r[1] for r in connection.execute("PRAGMA table_info(execution_effects)")
            }:
                connection.execute(
                    "ALTER TABLE execution_effects ADD COLUMN kind TEXT NOT NULL DEFAULT 'guarded'"
                )
            if "parent_execution_id" not in {
                r[1] for r in connection.execute("PRAGMA table_info(execution_binding)")
            }:
                connection.execute(
                    "ALTER TABLE execution_binding ADD COLUMN parent_execution_id TEXT"
                )
            digest = stable_hash(
                {
                    "version": GRAPH_VERSION,
                    "langgraph": _RUNTIME_VERSION,
                    "sqlite_checkpointer": _SQLITE_VERSION,
                    "definition": definition,
                    "inputs": inputs,
                    "fingerprint": persistence.fingerprint,
                },
                length=32,
            )
            row = connection.execute("SELECT digest FROM execution_binding WHERE id = 1").fetchone()
            if row is not None and row[0] != digest:
                raise GraphIdentityError(
                    "Execution inputs or graph definition changed; start anew."
                )
            connection.execute(
                "INSERT OR IGNORE INTO execution_binding(id, digest, execution_id, definition) "
                "VALUES (1, ?, ?, ?)",
                (digest, persistence.execution_id, definition),
            )
            connection.execute(
                "UPDATE execution_binding SET parent_execution_id=? WHERE id=1",
                (current_trace_execution().parent_execution_id,),
            )
            connection.commit()
            serde = JsonPlusSerializer(
                pickle_fallback=False,
                allowed_msgpack_modules=checkpoint_types or None,
            )
            saver = SqliteSaver(connection, serde=serde)
            saver.jsonplus_serde = serde
            saver.setup()
            yield GraphExecution(
                saver,
                {
                    "configurable": {"thread_id": persistence.execution_id},
                    "recursion_limit": recursion_limit,
                    "callbacks": [GraphTraceHandler()],
                },
                path,
            )
    finally:
        os.close(descriptor)


@contextmanager
def graph_execution(
    persistence: GraphPersistence | None,
    *,
    definition: str,
    inputs: dict[str, Any],
    recursion_limit: int = 1000,
    checkpoint_types: tuple[type, ...] = (),
) -> Iterator[GraphExecution]:
    # Product tracing uses the existing redacted OTel adapter. Importing an
    # orchestration dependency must not implicitly export graph payloads.
    previous = current_trace_execution()
    execution_id = persistence.execution_id if persistence else definition
    with (
        trace_execution_scope(execution_id=execution_id, parent_execution_id=previous.execution_id),
        tracing_context(enabled=False),
        _open_graph_execution(
            persistence,
            definition=definition,
            inputs=inputs,
            recursion_limit=recursion_limit,
            checkpoint_types=checkpoint_types,
        ) as execution,
    ):
        emit_execution_event("graph.started", definition)
        try:
            yield execution
        except BaseException as exc:
            emit_execution_event(
                "graph.interrupted", definition, {"error_type": type(exc).__name__}
            )
            raise
        else:
            emit_execution_event("graph.finished", definition)
