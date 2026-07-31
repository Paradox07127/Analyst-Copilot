"""Review round 3 F1: the shared trusted connection must not leak results
across threads — DuckDB result sets live on the executing cursor."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eda_platform.core.query import TrustedFileQueryEngine


def _write_csv(path: Path, prefix: str, rows: int = 50) -> None:
    lines = ["value"] + [f"{prefix}{i}" for i in range(rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_concurrent_previews_never_cross_datasets(tmp_path: Path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_csv(a, "AAA")
    _write_csv(b, "BBB")
    engine = TrustedFileQueryEngine([tmp_path])

    def probe(path: Path, prefix: str) -> list[str]:
        errors: list[str] = []
        for _ in range(40):
            _, rows = engine.preview_file(path, limit=10, offset=0)
            if not rows:
                errors.append("empty result")
            elif not all(str(row[0]).startswith(prefix) for row in rows):
                errors.append(f"foreign rows: {rows[:2]}")
        return errors

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(probe, path, prefix)
            for path, prefix in ((a, "AAA"), (b, "BBB")) * 4
        ]
        all_errors = [error for future in futures for error in future.result()]
    assert all_errors == []


def test_concurrent_describe_matches_schema(tmp_path: Path) -> None:
    one, two = tmp_path / "one.csv", tmp_path / "two.csv"
    one.write_text("x\n1\n", encoding="utf-8")
    two.write_text("p,q,r\n1,2,3\n", encoding="utf-8")
    engine = TrustedFileQueryEngine([tmp_path])

    def probe(path: Path, expected_cols: int) -> int:
        mismatches = 0
        for _ in range(30):
            if len(engine.describe_file(path)) != expected_cols:
                mismatches += 1
        return mismatches

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(probe, path, cols) for path, cols in ((one, 1), (two, 3)) * 4
        ]
        assert sum(future.result() for future in futures) == 0
