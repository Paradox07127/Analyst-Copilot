from __future__ import annotations

import pytest

from eda_platform.core.query import UnsafeQueryError, validate_select_statement


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM orders WHERE region <> 'Gift Set'",
        "SELECT 'reset load install attach copy pragma read_csv' AS note",
        "SELECT 'query(read_csv(''safe.csv'')) and query_table' AS note",
        "SELECT 'it''s a reset' AS note",
        r"SELECT E'it\'s a reset' AS note",
        "SELECT $$reset; load; read_csv('/etc/passwd')$$ AS note",
        "SELECT $body$copy; pragma; attach$body$ AS note",
        'SELECT "reset", "Gift Set" FROM '
        '(SELECT 1 AS "reset", 2 AS "Gift Set") AS values_',
        "-- RESET; LOAD read_csv('/etc/passwd')\nSELECT 1",
        "/* COPY /* nested RESET */ ATTACH */ SELECT 1",
        "SELECT 1 /* reset */ + 2 -- load\n",
        "WITH labels(value) AS (VALUES ('Gift Set')) SELECT value FROM labels",
    ],
)
def test_readonly_validator_ignores_blocked_words_outside_sql_code(sql: str) -> None:
    assert validate_select_statement(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SET threads = 1",
        "RESET threads",
        "LOAD json",
        "INSTALL json",
        "ATTACH ':memory:' AS other",
        "DETACH other",
        "COPY (SELECT 1) TO '/tmp/result.csv'",
        "PRAGMA version",
        "CALL checkpoint()",
        "EXPORT DATABASE '/tmp/export'",
        "IMPORT DATABASE '/tmp/export'",
        "INSERT INTO orders VALUES (1)",
        "UPDATE orders SET amount = 1",
        "DELETE FROM orders",
        "CREATE TABLE copied AS SELECT 1",
        "DROP TABLE orders",
        "ALTER TABLE orders ADD COLUMN note VARCHAR",
        "TRUNCATE orders",
        "VALUES (1)",
        "FROM orders",
    ],
)
def test_readonly_validator_rejects_non_select_statement_types(sql: str) -> None:
    with pytest.raises(UnsafeQueryError):
        validate_select_statement(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT * FROM main.read_parquet('/tmp/data.parquet')",
        "SELECT * FROM read_ndjson('/tmp/data.ndjson')",
        "SELECT * FROM read_ndjson_auto('/tmp/data.ndjson')",
        "SELECT * FROM read_xlsx('/tmp/data.xlsx')",
        'SELECT * FROM "read_xlsx" /* extension function */ ('
        "'/tmp/data.xlsx'"
        ")",
        'SELECT * FROM "read_json"("/tmp/data.json")',
        "WITH leaked AS (SELECT * FROM read_blob('/etc/passwd')) SELECT * FROM leaked",
        "SELECT * FROM query("
        "'SELECT * FROM read_csv_auto(''/tmp/data.csv'')'"
        ")",
        "SELECT * FROM query /* dynamic SQL */ ("
        "'SELECT * FROM read_parquet(''/tmp/data.parquet'')'"
        ")",
        'SELECT * FROM "query"('
        "'SELECT * FROM read_json_auto(''/tmp/data.json'')'"
        ")",
        "SELECT * FROM query_table('orders')",
        'SELECT * FROM "query_table" /* dynamic relation */ ('
        "'orders'"
        ")",
    ],
)
def test_readonly_validator_rejects_external_or_dynamic_access_functions(sql: str) -> None:
    with pytest.raises(UnsafeQueryError):
        validate_select_statement(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; SELECT 2",
        "SELECT '; RESET threads' AS safe; RESET threads",
        "SELECT $$; LOAD json$$ AS safe; LOAD json",
        "SELECT 1 /* ; RESET */; -- disguise\nRESET threads",
    ],
)
def test_readonly_validator_rejects_multiple_real_statements(sql: str) -> None:
    with pytest.raises(UnsafeQueryError):
        validate_select_statement(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 'unterminated",
        'SELECT "unterminated',
        "SELECT $tag$unterminated",
        "SELECT 1 /* unterminated",
    ],
)
def test_readonly_validator_fails_closed_on_malformed_lexical_regions(sql: str) -> None:
    with pytest.raises(UnsafeQueryError):
        validate_select_statement(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "IMPORT DATABASE '/tmp/must-not-be-inspected'",
        "SET threads = 1",
        "SELECT * FROM read_ndjson('/tmp/must-not-be-opened')",
    ],
)
def test_known_unsafe_code_is_rejected_before_duckdb_parser(
    sql: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def parser_must_not_run(statement: str) -> object:
        raise AssertionError(f"parser unexpectedly received {statement!r}")

    monkeypatch.setattr("eda_platform.core.query.duckdb.extract_statements", parser_must_not_run)
    with pytest.raises(UnsafeQueryError):
        validate_select_statement(sql)
