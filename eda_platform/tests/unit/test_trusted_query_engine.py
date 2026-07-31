"""TrustedFileQueryEngine: allow-list containment, DuckDB-level lockdown, and
the user-SQL engine staying fully file-blind."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event

import duckdb
import pandas as pd
import pytest

from eda_platform.core.cancellation import (
    CancellationContext,
    CancellationRequested,
    DurableCancellationRecord,
    StorageBackedCancellationToken,
    cancellation_scope,
)
from eda_platform.core.query import (
    DuckDBQueryEngine,
    TrustedFileQueryEngine,
    TrustedPathError,
    UnsafeQueryError,
    json_safe_value,
)


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    allowed = tmp_path / "allowed"
    denied = tmp_path / "denied"
    allowed.mkdir()
    denied.mkdir()
    (allowed / "a.csv").write_text("id,amount,when\n1,1.5,2026-01-01\n2,,2026-01-02\n")
    (denied / "b.csv").write_text("x\n1\n")
    return allowed, denied


@pytest.fixture
def engine(dirs: tuple[Path, Path]) -> TrustedFileQueryEngine:
    return TrustedFileQueryEngine([dirs[0]])


def test_describe_and_preview_inside_allowlist(
    engine: TrustedFileQueryEngine, dirs: tuple[Path, Path]
) -> None:
    allowed, _ = dirs
    described = engine.describe_file(allowed / "a.csv")
    assert [name for name, _ in described] == ["id", "amount", "when"]
    columns, rows = engine.preview_file(allowed / "a.csv", limit=10)
    assert columns == ["id", "amount", "when"]
    # JSON-safe values: ints/floats stay numeric, blank -> None, date -> ISO str.
    assert rows[0] == [1, 1.5, "2026-01-01"]
    assert rows[1][1] is None


def test_preview_offset_pages(engine: TrustedFileQueryEngine, dirs: tuple[Path, Path]) -> None:
    allowed, _ = dirs
    _, rows = engine.preview_file(allowed / "a.csv", limit=1, offset=1)
    assert len(rows) == 1
    assert rows[0][0] == 2


def test_path_outside_allowlist_rejected(
    engine: TrustedFileQueryEngine, dirs: tuple[Path, Path]
) -> None:
    _, denied = dirs
    with pytest.raises(TrustedPathError):
        engine.describe_file(denied / "b.csv")
    with pytest.raises(TrustedPathError):
        engine.preview_file(denied / "b.csv", limit=5)
    with pytest.raises(TrustedPathError):
        engine.copy_csv_to_parquet(dirs[0] / "a.csv", denied / "out.parquet")


def test_traversal_out_of_allowlist_rejected(
    engine: TrustedFileQueryEngine, dirs: tuple[Path, Path]
) -> None:
    allowed, _ = dirs
    with pytest.raises(TrustedPathError):
        engine.describe_file(allowed / ".." / "denied" / "b.csv")


def test_duckdb_layer_blocks_even_without_containment_check(
    engine: TrustedFileQueryEngine, dirs: tuple[Path, Path]
) -> None:
    """Guard-has-teeth: bypass the Python containment check and hit DuckDB raw.

    Control pair — the same raw statement succeeds for an allow-listed file, so
    a pass here cannot come from the query being broken."""
    allowed, denied = dirs
    raw = engine._connection  # noqa: SLF001 - deliberate bypass for the probe
    assert raw.execute("select count(*) from read_csv(?)", [str(allowed / "a.csv")]).fetchone() == (
        2,
    )
    with pytest.raises(duckdb.PermissionException):
        raw.execute("select * from read_csv(?)", [str(denied / "b.csv")]).fetchall()


def test_configuration_is_locked(engine: TrustedFileQueryEngine) -> None:
    with pytest.raises(duckdb.Error):
        engine._connection.execute("set enable_external_access = true")  # noqa: SLF001


def test_copy_csv_to_parquet_roundtrip(
    engine: TrustedFileQueryEngine, dirs: tuple[Path, Path]
) -> None:
    allowed, _ = dirs
    destination = engine.copy_csv_to_parquet(allowed / "a.csv", allowed / "pq" / "a.parquet")
    assert destination.is_file()
    columns, rows = engine.preview_file(destination, limit=5)
    assert columns == ["id", "amount", "when"]
    assert len(rows) == 2


def test_requires_at_least_one_directory() -> None:
    with pytest.raises(ValueError):
        TrustedFileQueryEngine([])


def test_user_sql_engine_still_blocks_file_reads(dirs: tuple[Path, Path]) -> None:
    allowed, _ = dirs
    user_engine = DuckDBQueryEngine()
    with pytest.raises(UnsafeQueryError):
        user_engine.execute_select(f"select * from read_csv('{allowed / 'a.csv'}')")


def test_user_sql_engine_interrupts_active_duckdb_query() -> None:
    engine = DuckDBQueryEngine()
    started = Event()
    interrupted = Event()

    class BlockingRelation:
        def limit(self, _max_rows: int) -> BlockingRelation:
            return self

        def df(self) -> pd.DataFrame:
            assert interrupted.wait(timeout=5)
            return pd.DataFrame({"value": [1]})

    class InterruptibleConnection:
        def sql(self, _sql: str) -> BlockingRelation:
            started.set()
            return BlockingRelation()

        def interrupt(self) -> None:
            interrupted.set()

    class ObservedCancellation(CancellationContext):
        @contextmanager
        def interrupt_on_cancel(self, callback: Callable[[], None]) -> Iterator[None]:
            with super().interrupt_on_cancel(callback):
                yield

    cancellation = ObservedCancellation()
    engine._connection = InterruptibleConnection()  # type: ignore[assignment]  # noqa: SLF001

    def execute() -> None:
        engine.execute_select(
            "select 1",
            cancellation=cancellation,
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(execute)
        assert started.wait(timeout=2)
        assert cancellation.request_cancel("stop query")
        with pytest.raises(CancellationRequested, match="stop query"):
            future.result(timeout=5)


def test_durable_cancel_flag_interrupts_already_blocked_duckdb_query() -> None:
    engine = DuckDBQueryEngine()
    started = Event()
    interrupted = Event()
    durable_cancelled = False

    class BlockingRelation:
        def limit(self, _max_rows: int) -> BlockingRelation:
            return self

        def df(self) -> pd.DataFrame:
            started.set()
            assert interrupted.wait(timeout=5)
            return pd.DataFrame({"value": [1]})

    class InterruptibleConnection:
        def sql(self, _sql: str) -> BlockingRelation:
            return BlockingRelation()

        def interrupt(self) -> None:
            interrupted.set()

    def read(_job_id: str) -> DurableCancellationRecord:
        return DurableCancellationRecord(
            job_id="job_query",
            generation=4,
            owner="worker-a",
            cancel_requested=durable_cancelled,
            reason="durable stop",
        )

    cancellation = StorageBackedCancellationToken(
        job_id="job_query",
        generation=4,
        owner="worker-a",
        reader=read,
    )
    engine._connection = InterruptibleConnection()  # type: ignore[assignment]  # noqa: SLF001

    def execute_in_scope() -> pd.DataFrame:
        with cancellation_scope(cancellation):
            return engine.execute_select("select 1")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(execute_in_scope)
        assert started.wait(timeout=2)
        durable_cancelled = True
        with pytest.raises(CancellationRequested, match="durable stop"):
            future.result(timeout=5)

    assert interrupted.is_set()


def test_json_safe_value_edge_cases() -> None:
    assert json_safe_value(float("nan")) is None
    assert json_safe_value(float("inf")) is None
    assert json_safe_value(b"\xff") == "�"
    assert json_safe_value(True) is True
