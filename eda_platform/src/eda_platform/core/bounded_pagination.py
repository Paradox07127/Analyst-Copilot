"""Bounded, source-bound pagination primitives for file-backed API resources."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class InvalidCursorError(Exception):
    """A pagination cursor is malformed, or bound to a source that has changed.

    Defined here rather than in the run service so that core pagination stays
    importable without the application layer; ``session_service`` re-exports it for
    the callers and the API error mapping that already reference it there.
    """

    def __init__(self) -> None:
        super().__init__("Invalid pagination cursor.")


_CURSOR_VERSION = 1
MAX_CURSOR_CHARS = 2048
MAX_JSONL_RECORD_BYTES = 1 * 1024 * 1024


def semantic_scope(project_id: str, session_id: str) -> str:
    return f"semantic:{project_id}:{session_id}"


def skills_scope(project_id: str, session_id: str) -> str:
    return f"skills:{project_id}:{session_id}"


def run_resource_scopes(project_id: str, session_id: str) -> tuple[str, ...]:
    """Every ResourcePageIndex scope a run owns, for deletion to purge.

    Declared next to the builders so a new run-scoped projection cannot be
    added without the deletion path seeing it.
    """
    return (semantic_scope(project_id, session_id), skills_scope(project_id, session_id))


def jsonl_path_key(relative_path: str | PurePosixPath) -> str:
    """Canonical form of a workspace-relative JSONL key.

    Always POSIX separators, on every platform: the key is a database value
    shared between the index that writes it and run deletion that purges by it.
    Letting the host separator leak in would silently split the two on Windows.
    """
    return PurePosixPath(relative_path).as_posix()


def source_token(*parts: object) -> str:
    encoded = json.dumps(
        parts, ensure_ascii=False, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def encode_bound_cursor(
    position: int,
    *,
    scope: str,
    source_version: str,
) -> str:
    payload = {
        "v": _CURSOR_VERSION,
        "p": position,
        "s": scope,
        "r": source_version,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_bound_cursor(
    cursor: str,
    *,
    scope: str,
    source_version: str,
) -> int:
    if len(cursor) > MAX_CURSOR_CHARS:
        raise InvalidCursorError
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
        payload = json.loads(raw)
    except ValueError as exc:  # covers UnicodeError, binascii, JSONDecodeError
        raise InvalidCursorError from exc
    if (
        not isinstance(payload, dict)
        or payload.get("v") != _CURSOR_VERSION
        or payload.get("s") != scope
        or payload.get("r") != source_version
        or type(payload.get("p")) is not int
        or payload["p"] < 0
    ):
        raise InvalidCursorError
    return int(payload["p"])


def encode_bound_key_cursor(
    key: str,
    *,
    scope: str,
    source_version: str,
) -> str:
    payload = {
        "v": _CURSOR_VERSION,
        "k": key,
        "s": scope,
        "r": source_version,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_bound_key_cursor(
    cursor: str,
    *,
    scope: str,
    source_version: str,
) -> str:
    if len(cursor) > MAX_CURSOR_CHARS:
        raise InvalidCursorError
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
        payload = json.loads(raw)
    except ValueError as exc:  # covers UnicodeError, binascii, JSONDecodeError
        raise InvalidCursorError from exc
    if (
        not isinstance(payload, dict)
        or payload.get("v") != _CURSOR_VERSION
        or payload.get("s") != scope
        or payload.get("r") != source_version
        or not isinstance(payload.get("k"), str)
    ):
        raise InvalidCursorError
    return str(payload["k"])


@dataclass(frozen=True, slots=True)
class JsonlIndexState:
    path_key: str
    source_version: str
    valid_count: int


@dataclass(frozen=True, slots=True)
class JsonlIndexedRecord:
    ordinal: int
    payload: bytes
    # Indexed but never read: the record holds its ordinal so counts and cursors
    # stay right, and the caller decides how to present the gap.
    oversized: bool = False


class JsonlPageIndex:
    """SQLite byte-offset index for bounded JSONL page reads.

    Rebuilding an index is the one migration/recovery pass over a legacy file.
    Once the source identity is indexed, page reads seek directly to at most
    ``limit + 1`` capped records. Appends extend the index from its last byte;
    truncation or replacement rebuilds it.
    """

    def __init__(
        self,
        db_path: Path,
        workspace: Path,
        *,
        max_record_bytes: int = MAX_JSONL_RECORD_BYTES,
    ) -> None:
        self._db_path = db_path
        self._workspace = workspace.resolve()
        self._max_record_bytes = max_record_bytes
        self._ensure_schema()

    def ensure(
        self,
        path: Path,
        *,
        accept: Callable[[bytes], bool],
    ) -> JsonlIndexState:
        resolved = path.resolve()
        if not resolved.is_relative_to(self._workspace):
            raise ValueError("Indexed JSONL path escaped the workspace.")
        path_key = jsonl_path_key(resolved.relative_to(self._workspace).as_posix())
        try:
            stat = resolved.stat()
        except FileNotFoundError:
            self._replace_index(path_key, resolved, None, accept)
            return JsonlIndexState(path_key, source_token("missing"), 0)
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                select device, inode, source_size, source_mtime_ns,
                       indexed_bytes, valid_count, tail_digest
                from jsonl_page_sources where path_key = ?
                """,
                (path_key,),
            ).fetchone()
        if row is None:
            self._replace_index(path_key, resolved, stat, accept)
        else:
            same_file = (
                int(row[0]) == int(stat.st_dev)
                and int(row[1]) == int(stat.st_ino)
            )
            unchanged = (
                same_file
                and int(row[2]) == stat.st_size
                and int(row[3]) == stat.st_mtime_ns
            )
            if not unchanged:
                can_extend = (
                    same_file
                    and int(row[4]) <= stat.st_size
                    and int(row[2]) <= stat.st_size
                    and str(row[6]) == _tail_digest(resolved, int(row[4]))
                )
                if can_extend:
                    self._extend_index(
                        path_key,
                        resolved,
                        start_byte=int(row[4]),
                        start_ordinal=int(row[5]),
                        accept=accept,
                    )
                else:
                    self._replace_index(path_key, resolved, stat, accept)
        with closing(self._connect()) as conn:
            current = conn.execute(
                """
                select device, inode, source_size, source_mtime_ns, valid_count
                from jsonl_page_sources where path_key = ?
                """,
                (path_key,),
            ).fetchone()
        if current is None:
            return JsonlIndexState(path_key, source_token("missing"), 0)
        version = source_token(*(int(value) for value in current[:4]))
        return JsonlIndexState(path_key, version, int(current[4]))

    @staticmethod
    def file_source_version(path: Path) -> str:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return source_token("missing")
        return source_token(
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )

    def page(
        self,
        state: JsonlIndexState,
        *,
        start: int,
        limit: int,
        reverse: bool = False,
    ) -> list[JsonlIndexedRecord]:
        if reverse:
            sql = """
                select ordinal, byte_offset, byte_length
                from jsonl_page_entries
                where path_key = ? and ordinal < ?
                order by ordinal desc limit ?
            """
            params = (state.path_key, start, limit)
        else:
            sql = """
                select ordinal, byte_offset, byte_length
                from jsonl_page_entries
                where path_key = ? and ordinal >= ?
                order by ordinal limit ?
            """
            params = (state.path_key, start, limit)
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        path = self._workspace / PurePosixPath(state.path_key)
        records: list[JsonlIndexedRecord] = []
        try:
            with path.open("rb") as handle:
                for ordinal, offset, length in rows:
                    if int(length) > self._max_record_bytes:
                        records.append(
                            JsonlIndexedRecord(int(ordinal), b"", oversized=True)
                        )
                        continue
                    handle.seek(int(offset))
                    payload = handle.read(int(length))
                    records.append(JsonlIndexedRecord(int(ordinal), payload))
        except OSError:
            return []
        if reverse:
            records.reverse()
        return records

    def _replace_index(
        self,
        path_key: str,
        path: Path,
        stat: os.stat_result | None,
        accept: Callable[[bytes], bool],
    ) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "delete from jsonl_page_entries where path_key = ?", (path_key,)
            )
            conn.execute("delete from jsonl_page_sources where path_key = ?", (path_key,))
        if stat is None:
            return
        self._extend_index(
            path_key,
            path,
            start_byte=0,
            start_ordinal=0,
            accept=accept,
        )

    def _extend_index(
        self,
        path_key: str,
        path: Path,
        *,
        start_byte: int,
        start_ordinal: int,
        accept: Callable[[bytes], bool],
    ) -> None:
        entries: list[tuple[str, int, int, int]] = []
        ordinal = start_ordinal
        indexed_bytes = start_byte
        try:
            with path.open("rb") as handle:
                handle.seek(start_byte)
                while True:
                    offset = handle.tell()
                    raw = handle.readline(self._max_record_bytes + 2)
                    if not raw:
                        indexed_bytes = handle.tell()
                        break
                    if len(raw) > self._max_record_bytes + 1:
                        while raw and not raw.endswith(b"\n"):
                            raw = handle.readline(self._max_record_bytes + 2)
                        if not raw.endswith(b"\n"):
                            # Same rule as a short partial record: an unterminated
                            # tail may still be growing, so publish nothing.
                            indexed_bytes = offset
                            break
                        indexed_bytes = handle.tell()
                        # Index the true length so page() reports it as oversized
                        # rather than silently shifting every later ordinal.
                        entries.append(
                            (path_key, ordinal, offset, indexed_bytes - offset)
                        )
                        ordinal += 1
                        continue
                    if not raw.endswith(b"\n"):
                        # A concurrently appended partial record is retried on
                        # the next ensure call; do not publish its byte offset.
                        indexed_bytes = offset
                        break
                    stripped = raw.strip()
                    indexed_bytes = handle.tell()
                    if stripped and accept(stripped):
                        entries.append((path_key, ordinal, offset, len(raw)))
                        ordinal += 1
            stat = path.stat()
        except OSError:
            return
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                """
                insert or replace into jsonl_page_entries(
                    path_key, ordinal, byte_offset, byte_length
                ) values(?, ?, ?, ?)
                """,
                entries,
            )
            conn.execute(
                """
                insert into jsonl_page_sources(
                    path_key, device, inode, source_size, source_mtime_ns,
                    indexed_bytes, valid_count, tail_digest
                ) values(?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(path_key) do update set
                    device=excluded.device,
                    inode=excluded.inode,
                    source_size=excluded.source_size,
                    source_mtime_ns=excluded.source_mtime_ns,
                    indexed_bytes=excluded.indexed_bytes,
                    valid_count=excluded.valid_count,
                    tail_digest=excluded.tail_digest
                """,
                (
                    path_key,
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                    indexed_bytes,
                    ordinal,
                    _tail_digest(path, indexed_bytes),
                ),
            )

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.executescript(
                """
                create table if not exists jsonl_page_sources (
                    path_key text primary key,
                    device integer not null,
                    inode integer not null,
                    source_size integer not null,
                    source_mtime_ns integer not null,
                    indexed_bytes integer not null,
                    valid_count integer not null,
                    tail_digest text not null default ''
                );
                create table if not exists jsonl_page_entries (
                    path_key text not null,
                    ordinal integer not null,
                    byte_offset integer not null,
                    byte_length integer not null,
                    primary key(path_key, ordinal)
                );
                """
            )
            columns = {
                str(row[1])
                for row in conn.execute("pragma table_info(jsonl_page_sources)")
            }
            if "tail_digest" not in columns:
                conn.execute(
                    "alter table jsonl_page_sources "
                    "add column tail_digest text not null default ''"
                )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=5.0)


def _tail_digest(path: Path, end: int, window: int = 4096) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, end - window))
            payload = handle.read(min(window, end))
    except OSError:
        return ""
    return hashlib.sha256(payload).hexdigest()


class ResourcePageIndex:
    """Versioned SQLite projection for legacy monolithic JSON resources."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._ensure_schema()

    def is_current(self, scope: str, source_version: str) -> bool:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "select source_version from resource_page_sources where scope = ?",
                (scope,),
            ).fetchone()
        return row is not None and str(row[0]) == source_version

    def replace(
        self,
        scope: str,
        source_version: str,
        collections: dict[str, list[str]],
    ) -> None:
        with closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                conn.execute("delete from resource_page_entries where scope = ?", (scope,))
                for collection, payloads in collections.items():
                    conn.executemany(
                        """
                        insert into resource_page_entries(
                            scope, collection, ordinal, payload_json
                        ) values(?, ?, ?, ?)
                        """,
                        (
                            (scope, collection, ordinal, payload)
                            for ordinal, payload in enumerate(payloads)
                        ),
                    )
                conn.execute(
                    """
                    insert into resource_page_sources(scope, source_version)
                    values(?, ?)
                    on conflict(scope) do update set
                        source_version=excluded.source_version
                    """,
                    (scope, source_version),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def page(
        self,
        scope: str,
        source_version: str,
        collection: str,
        *,
        offset: int,
        limit: int,
    ) -> list[str]:
        if not self.is_current(scope, source_version):
            raise InvalidCursorError
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select payload_json from resource_page_entries
                where scope = ? and collection = ? and ordinal >= ?
                order by ordinal limit ?
                """,
                (scope, collection, offset, limit),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.executescript(
                """
                create table if not exists resource_page_sources (
                    scope text primary key,
                    source_version text not null
                );
                create table if not exists resource_page_entries (
                    scope text not null,
                    collection text not null,
                    ordinal integer not null,
                    payload_json text not null,
                    primary key(scope, collection, ordinal)
                );
                create index if not exists idx_resource_page_lookup
                    on resource_page_entries(scope, collection, ordinal);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=5.0)
