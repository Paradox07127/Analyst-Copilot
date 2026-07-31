from pathlib import Path

from eda_platform.core.bounded_pagination import (
    MAX_JSONL_RECORD_BYTES,
    JsonlPageIndex,
)
from eda_platform.core.store import ArtifactStore


def test_jsonl_index_caps_one_record_but_keeps_its_ordinal(tmp_path: Path) -> None:
    """A capped record is reported, not dropped: skipping it silently shifted
    every later ordinal and under-reported the total. Its bytes are still never
    read, which is what the cap exists for."""
    store = ArtifactStore(tmp_path)
    path = tmp_path / "capture.jsonl"
    path.write_bytes(
        b"x" * (MAX_JSONL_RECORD_BYTES + 1)
        + b"\n"
        + b'{"task":"bounded"}\n'
    )
    index = JsonlPageIndex(store.db_path, store.root)
    state = index.ensure(path, accept=lambda payload: payload.startswith(b"{"))

    assert state.valid_count == 2
    page = index.page(state, start=0, limit=2)
    assert [
        (record.ordinal, record.oversized, record.payload.strip()) for record in page
    ] == [
        (0, True, b""),
        (1, False, b'{"task":"bounded"}'),
    ]


def test_jsonl_index_rebuilds_when_same_inode_is_overwritten(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    path = tmp_path / "capture.jsonl"
    path.write_bytes(b"old-0\nold-1\n")
    index = JsonlPageIndex(store.db_path, store.root)
    old = index.ensure(path, accept=lambda _payload: True)
    assert [row.payload.strip() for row in index.page(old, start=0, limit=2)] == [
        b"old-0",
        b"old-1",
    ]

    # write_bytes truncates/reuses the same inode on normal filesystems. The
    # tail fingerprint must distinguish this from an append.
    path.write_bytes(b"new-0\nnew-1\n")
    new = index.ensure(path, accept=lambda _payload: True)
    assert new.source_version != old.source_version
    assert [row.payload.strip() for row in index.page(new, start=0, limit=2)] == [
        b"new-0",
        b"new-1",
    ]
