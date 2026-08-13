from __future__ import annotations

from pathlib import Path

import eda_platform.tools.loader as loader_module
from eda_platform.tools.loader import DatasetFramePool, defer_csv, load_csv


def test_load_csv_sniffs_without_path_read_bytes(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("id,value\n1,10\n2,20\n", encoding="utf-8")

    def reject_read_bytes(_path: Path) -> bytes:
        raise AssertionError("load_csv must not materialize the complete file for sniffing")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)

    loaded = load_csv(csv_path)

    assert loaded.frame.to_dict("records") == [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
    ]


def test_load_csv_reuses_precomputed_content_hash(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("id,value\n1,10\n", encoding="utf-8")

    def reject_hash(_path: Path) -> str:
        raise AssertionError("a supplied content hash must not be recomputed")

    monkeypatch.setattr(loader_module, "hash_file", reject_hash)

    loaded = load_csv(csv_path, dataset_id="ds_known", content_hash="known_hash")

    assert loaded.record.dataset_id == "ds_known"
    assert loaded.record.content_hash == "known_hash"


def test_dataset_source_pool_retains_at_most_one_table(
    tmp_path: Path, monkeypatch
) -> None:
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    first_path.write_text("id,value\n1,10\n", encoding="utf-8")
    second_path.write_text("id,value\n2,20\n", encoding="utf-8")
    pool = DatasetFramePool()
    first = defer_csv(first_path, frame_pool=pool)
    second = defer_csv(second_path, frame_pool=pool)
    real_load_csv = loader_module.load_csv
    calls: list[str] = []

    def recording_load_csv(path, **kwargs):
        calls.append(Path(path).name)
        return real_load_csv(path, **kwargs)

    monkeypatch.setattr(loader_module, "load_csv", recording_load_csv)

    assert first.frame.iloc[0]["value"] == 10
    assert first.frame.iloc[0]["value"] == 10
    assert second.frame.iloc[0]["value"] == 20
    assert first.frame.iloc[0]["value"] == 10
    assert calls == ["first.csv", "second.csv", "first.csv"]
    pool.clear()
    assert first.frame.iloc[0]["value"] == 10
    assert calls[-1] == "first.csv"
