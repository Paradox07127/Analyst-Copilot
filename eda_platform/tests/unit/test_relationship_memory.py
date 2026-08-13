"""Relationship discovery must reuse catalog tables, not pin every frame.

2026-08-12 review (F2): `discover_relationship_candidates` unconditionally
registered `dataset.frame` for every dataset into the engine. DuckDB holds a
reference to each registered DataFrame until the connection dies, so eager
discovery re-created the "all tables resident at once" peak that the
one-table frame pool exists to prevent — and the same relation name shadowed
the file-backed catalog table built by `build_catalog`.
"""

from __future__ import annotations

from pathlib import Path

from eda_platform.core.query import DuckDBQueryEngine
from eda_platform.tools.loader import load_csv
from eda_platform.tools.relationship_discovery import discover_relationship_candidates


def _write_csv(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _datasets(tmp_path: Path):
    orders = _write_csv(
        tmp_path,
        "orders.csv",
        "order_id,customer_id\n" + "\n".join(f"o{i},c{i % 5}" for i in range(20)),
    )
    customers = _write_csv(
        tmp_path,
        "customers.csv",
        "customer_id,region\n" + "\n".join(f"c{i},r{i % 2}" for i in range(5)),
    )
    return [
        load_csv(orders, dataset_id="ds_orders"),
        load_csv(customers, dataset_id="ds_customers"),
    ]


def test_has_relation_sees_registered_frames_and_trusted_tables(tmp_path: Path) -> None:
    datasets = _datasets(tmp_path)
    engine = DuckDBQueryEngine()
    engine.register_frame("ds_orders", datasets[0].frame)
    assert engine.has_relation("ds_orders")
    assert not engine.has_relation("ds_missing")

    trusted = DuckDBQueryEngine(
        database=tmp_path / "catalog.duckdb", trusted_csv_ingest=True
    )
    trusted.register_trusted_csv("ds_customers", tmp_path / "customers.csv")
    trusted.seal()
    assert trusted.has_relation("ds_customers")
    assert not trusted.has_relation("ds_orders")


def test_discovery_does_not_reregister_catalog_relations(tmp_path: Path) -> None:
    datasets = _datasets(tmp_path)

    class SpyEngine(DuckDBQueryEngine):
        def __init__(self) -> None:
            super().__init__()
            self.register_calls: list[str] = []

        def register_frame(self, name: str, frame) -> None:  # noqa: ANN001
            self.register_calls.append(name)
            super().register_frame(name, frame)

    engine = SpyEngine()
    # Simulate the file-backed catalog: both relations already exist.
    engine.register_frame("ds_orders", datasets[0].frame)
    engine.register_frame("ds_customers", datasets[1].frame)
    engine.register_calls.clear()

    candidates = discover_relationship_candidates(datasets, engine)

    # The catalog tables are used as-is; no frame is pinned a second time.
    assert engine.register_calls == []
    assert any(
        candidate.pair.left_columns == ["customer_id"]
        and candidate.pair.right_columns == ["customer_id"]
        for candidate in candidates.candidates
    )


def test_discovery_still_registers_on_a_fresh_engine(tmp_path: Path) -> None:
    datasets = _datasets(tmp_path)
    candidates = discover_relationship_candidates(datasets, None)
    assert candidates.candidates  # standalone use keeps working
