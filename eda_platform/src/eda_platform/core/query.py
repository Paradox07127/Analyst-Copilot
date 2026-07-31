from __future__ import annotations

import math
import re
from collections.abc import Sequence
from contextlib import closing
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path

import duckdb
import pandas as pd

from eda_platform.core.cancellation import CancellationToken, current_cancellation_token

# Defense-in-depth beyond DuckDB's parser and ``enable_external_access=False``.
# The scanner below applies these only to SQL code, never to string literals,
# quoted identifiers, or comments.
_BLOCKED_WORDS = frozenset(
    {
        "read_csv",
        "read_csv_auto",
        "read_parquet",
        "read_json",
        "read_json_auto",
        "read_text",
        "read_blob",
        # Dynamic table functions re-parse a string as SQL/a relation name.
        # Skipping literal contents is safe only when these indirection seams
        # are unavailable to user SQL.
        "query",
        "query_table",
        "copy",
        "install",
        "load",
        "attach",
        "detach",
        "export",
        "import",
        "insert",
        "update",
        "delete",
        "drop",
        "create",
        "alter",
        "truncate",
        "pragma",
        "call",
        "set",
        "reset",
    }
)
_INDIRECT_OR_EXTERNAL_FUNCTIONS = frozenset(
    {
        "read_csv",
        "read_csv_auto",
        "read_parquet",
        "read_json",
        "read_json_auto",
        "read_text",
        "read_blob",
        "query",
        "query_table",
    }
)
_DOLLAR_QUOTE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")


class UnsafeQueryError(ValueError):
    """Raised when a query is not a safe, single, read-only SELECT."""


class SqlBindingError(ValueError):
    """Raised when DuckDB cannot bind a syntactically safe SELECT."""


class QueryTimeout(TimeoutError):
    """Raised when a safe query exceeds the configured wall-clock timeout."""


class DuckDBQueryEngine:
    def __init__(self, *, max_rows: int = 10_000) -> None:
        self.max_rows = max_rows
        self._connection = duckdb.connect(
            config={
                "enable_external_access": False,
                "allow_unsigned_extensions": False,
            }
        )

    def register_frame(self, name: str, frame: pd.DataFrame) -> None:
        """Register an in-memory frame as a queryable view (no filesystem access)."""
        self._connection.register(_safe_relation_name(name), frame)

    def register_csv(self, name: str, path: Path | str, frame: pd.DataFrame | None = None) -> None:
        """Register a dataset."""
        loaded = frame if frame is not None else pd.read_csv(Path(path))
        self.register_frame(name, loaded)

    def execute_select(
        self,
        sql: str,
        *,
        cancellation: CancellationToken | None = None,
    ) -> pd.DataFrame:
        cancellation = cancellation or current_cancellation_token()
        self._validate(sql)
        if cancellation is not None:
            cancellation.checkpoint()
        try:
            if cancellation is None:
                relation = self._connection.sql(sql)
                return relation.limit(self.max_rows).df()
            with cancellation.interrupt_on_cancel(self.interrupt):
                relation = self._connection.sql(sql)
                result = relation.limit(self.max_rows).df()
        except duckdb.Error:
            if cancellation is not None:
                cancellation.checkpoint()
            raise
        cancellation.checkpoint()
        return result

    def dry_run(self, sql: str) -> None:
        statement = validate_select_statement(sql)
        try:
            self._connection.execute(f"EXPLAIN {statement}").fetchall()
        except duckdb.Error as exc:
            raise SqlBindingError(f"SQL binding failed: {exc}") from exc

    def interrupt(self) -> None:
        self._connection.interrupt()

    def _validate(self, sql: str) -> None:
        validate_select_statement(sql)


def validate_select_statement(sql: str) -> str:
    statement = sql.strip().rstrip(";").strip()
    if not statement:
        raise UnsafeQueryError("Empty query.")
    tokens = _sql_code_tokens(statement)
    if (
        not tokens
        or tokens[0][0] != "word"
        or tokens[0][1].casefold() not in {"select", "with"}
    ):
        raise UnsafeQueryError("Only SELECT/WITH queries are allowed.")
    for index, (kind, value) in enumerate(tokens):
        normalized = value.casefold()
        if kind == "word" and normalized in _BLOCKED_WORDS:
            raise UnsafeQueryError(f"Query uses a blocked keyword/function: {value}")
        next_is_call = (
            index + 1 < len(tokens)
            and tokens[index + 1] == ("punctuation", "(")
        )
        if next_is_call and (
            normalized.startswith("read_")
            or normalized in _INDIRECT_OR_EXTERNAL_FUNCTIONS
        ):
            raise UnsafeQueryError(f"Query uses a blocked keyword/function: {value}")
    # Lexically reject known unsafe code before invoking DuckDB's statement
    # extractor. Some non-SELECT statements (notably IMPORT DATABASE) perform
    # filesystem discovery during extraction, so parser-first validation is
    # not a side-effect-free safety boundary.
    try:
        parsed = duckdb.extract_statements(statement)
    except duckdb.Error as exc:
        raise UnsafeQueryError(f"SQL could not be parsed safely: {exc}") from exc
    if len(parsed) != 1:
        raise UnsafeQueryError("Exactly one SQL statement is required.")
    if parsed[0].type != duckdb.StatementType.SELECT:
        raise UnsafeQueryError("Only SELECT/WITH queries are allowed.")
    return statement


def _sql_code_tokens(statement: str) -> list[tuple[str, str]]:
    """Return code tokens while skipping DuckDB lexical literal/comment regions.

    DuckDB's own parser remains authoritative for statement count, statement
    type, and malformed syntax. This small scanner has one narrower job:
    prevent the defense-in-depth blacklist from inspecting data literals and
    comments. It supports DuckDB's doubled quotes, ``E'...'`` escapes, tagged
    dollar quotes, line comments, and nested block comments.
    """

    tokens: list[tuple[str, str]] = []
    index = 0
    length = len(statement)
    while index < length:
        if statement.startswith("--", index):
            newline = statement.find("\n", index + 2)
            index = length if newline == -1 else newline + 1
            continue
        if statement.startswith("/*", index):
            index = _end_of_block_comment(statement, index)
            continue
        char = statement[index]
        if char == "'":
            escaped = (
                index > 0
                and statement[index - 1] in {"e", "E"}
                and (index < 2 or not _is_word_char(statement[index - 2]))
            )
            index = _end_of_sql_quote(statement, index, "'", backslash_escapes=escaped)
            continue
        if char == '"':
            end = _end_of_sql_quote(statement, index, '"', backslash_escapes=False)
            value = statement[index + 1 : end - 1].replace('""', '"')
            tokens.append(("quoted_identifier", value))
            index = end
            continue
        if char == "$":
            delimiter = _DOLLAR_QUOTE.match(statement, index)
            if delimiter is not None:
                marker = delimiter.group(0)
                end = statement.find(marker, delimiter.end())
                if end == -1:
                    raise UnsafeQueryError("Unterminated dollar-quoted string.")
                index = end + len(marker)
                continue
        if char.isalpha() or char == "_":
            end = index + 1
            while end < length and _is_word_char(statement[end]):
                end += 1
            tokens.append(("word", statement[index:end]))
            index = end
            continue
        if not char.isspace():
            tokens.append(("punctuation", char))
        index += 1
    return tokens


def _end_of_sql_quote(
    statement: str,
    start: int,
    quote: str,
    *,
    backslash_escapes: bool,
) -> int:
    index = start + 1
    while index < len(statement):
        if backslash_escapes and statement[index] == "\\":
            index += 2
            continue
        if statement[index] == quote:
            if index + 1 < len(statement) and statement[index + 1] == quote:
                index += 2
                continue
            return index + 1
        index += 1
    raise UnsafeQueryError(f"Unterminated {quote}-quoted SQL region.")


def _end_of_block_comment(statement: str, start: int) -> int:
    depth = 1
    index = start + 2
    while index < len(statement):
        if statement.startswith("/*", index):
            depth += 1
            index += 2
        elif statement.startswith("*/", index):
            depth -= 1
            index += 2
            if depth == 0:
                return index
        else:
            index += 1
    raise UnsafeQueryError("Unterminated SQL block comment.")


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char in {"_", "$"}


def _safe_relation_name(value: str) -> str:
    """Reduce a dataset name to a safe DuckDB relation identifier."""
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", value.strip())
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned


class TrustedPathError(ValueError):
    """Raised when a server-side file query targets a path outside the allow-list."""


class TrustedFileQueryEngine:
    """Server-side-only DuckDB connection for scanning dataset files lazily.

    Separate from ``DuckDBQueryEngine`` on purpose (§7.4 two-connection model):
    user SQL never reaches this connection, and this connection's file access is
    fenced twice — DuckDB's own ``allowed_directories`` allow-list (locked at
    construction) plus an explicit containment check on every path. All SQL here
    is a server-side template; paths are bound as prepared-statement parameters.
    """

    def __init__(self, allowed_directories: Sequence[Path | str]) -> None:
        roots = tuple(Path(entry).resolve() for entry in allowed_directories)
        if not roots:
            raise ValueError("TrustedFileQueryEngine requires at least one allowed directory.")
        self._allowed = roots
        # memory_limit must be set here: lock_configuration below freezes it,
        # and an unbounded COPY on a 1GB upload was measured at ~4.5x RSS.
        connection = duckdb.connect(config={"memory_limit": "1GB"})
        # DuckDB >= 1.2 (verified against 1.5.4): ``allowed_directories`` only
        # takes effect once ``enable_external_access`` is disabled afterwards,
        # and neither can be set together in the connect() config — the pair
        # must be applied in this order on a live connection, then locked so
        # nothing can re-enable full filesystem access later.
        quoted = ", ".join("'" + str(root).replace("'", "''") + "'" for root in roots)
        connection.execute(f"set allowed_directories = [{quoted}]")
        connection.execute("set enable_external_access = false")
        connection.execute("set lock_configuration = true")
        self._connection = connection

    @property
    def allowed_directories(self) -> tuple[Path, ...]:
        return self._allowed

    def describe_file(self, path: Path | str) -> list[tuple[str, str]]:
        """Return (column, duckdb_type) pairs by reading only the file header/sample."""
        target = self._checked(path)
        reader = _reader_for(target)
        # Per-call cursor: DuckDB result sets live on the executing connection,
        # so sharing self._connection across FastAPI's threadpool returned one
        # request's rows to another. cursor() inherits the allow-list and the
        # config lock (verified) while isolating results per call.
        with closing(self._connection.cursor()) as cursor:
            rows = cursor.execute(f"describe select * from {reader}(?)", [target]).fetchall()
        return [(str(row[0]), str(row[1])) for row in rows]

    def preview_file(
        self, path: Path | str, *, limit: int, offset: int = 0
    ) -> tuple[list[str], list[list[object]]]:
        """Paged scan of a CSV/Parquet file; values converted to JSON-safe types."""
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        target = self._checked(path)
        reader = _reader_for(target)
        with closing(self._connection.cursor()) as cursor:
            result = cursor.execute(
                f"select * from {reader}(?) limit ? offset ?", [target, limit, offset]
            )
            columns = [str(item[0]) for item in result.description or []]
            rows = [[json_safe_value(value) for value in row] for row in result.fetchall()]
        return columns, rows

    def copy_csv_to_parquet(self, source: Path | str, destination: Path | str) -> Path:
        """Materialise a Parquet copy of a CSV; both paths must be allow-listed."""
        checked_source = self._checked(source)
        checked_destination = Path(self._checked(destination))
        checked_destination.parent.mkdir(parents=True, exist_ok=True)
        # COPY targets cannot be bound parameters; the path passed containment
        # above, so quote-escape it into the statement.
        escaped = str(checked_destination).replace("'", "''")
        with closing(self._connection.cursor()) as cursor:
            cursor.execute(
                f"copy (select * from read_csv(?)) to '{escaped}' (format parquet)",
                [checked_source],
            )
        return checked_destination

    def _checked(self, path: Path | str) -> str:
        resolved = Path(path).resolve()
        if not any(resolved.is_relative_to(root) for root in self._allowed):
            raise TrustedPathError(f"Path outside allowed directories: {path}")
        return str(resolved)


def _reader_for(target: str) -> str:
    return "read_parquet" if target.lower().endswith(".parquet") else "read_csv"


def json_safe_value(value: object) -> object:
    """Coerce a DuckDB scalar to something JSON-serializable (no frames, no NaN)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
