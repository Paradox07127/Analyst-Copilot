from __future__ import annotations

import json
import multiprocessing
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest

from eda_platform.core.storage_operations import (
    ReplacementFile,
    ResourceDigestMismatchError,
    ResourceOperationInProgressError,
    ResourceTarget,
    StorageOperationBlockedError,
    StorageOperationJournal,
    UnsafeStoragePathError,
    canonical_digest,
    composite_digest,
    missing_resource_digest,
    quarantine_relative_path,
    staging_relative_path,
)
from eda_platform.core.store import ArtifactStore


class InjectedStorageFault(RuntimeError):
    pass


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    artifact_store = ArtifactStore(tmp_path / "workspace")
    _install_storage_operation_schema(artifact_store.db_path)
    return artifact_store


def _install_storage_operation_schema(db_path: Path) -> None:
    """The production schema belongs to ArtifactStore's migration steward."""
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            create table if not exists resource_heads (
                resource_kind text not null,
                project_id text not null,
                resource_key text not null,
                relative_path text not null,
                version integer not null,
                content_digest text not null,
                updated_at text not null,
                primary key(resource_kind, project_id, resource_key)
            );

            create table if not exists storage_operations (
                op_id text primary key,
                op_kind text not null,
                resource_kind text not null,
                project_id text not null,
                resource_key text not null,
                expected_version integer,
                target_version integer,
                base_digest text,
                target_digest text,
                request_key text,
                state text not null,
                error_code text,
                error_message text,
                created_at text not null,
                updated_at text not null
            );

            create table if not exists storage_operation_items (
                op_id text not null,
                ordinal integer not null,
                mode text not null,
                source_relpath text not null,
                work_relpath text not null,
                base_digest text,
                target_digest text,
                payload blob,
                required integer not null default 1,
                primary key(op_id, ordinal)
            );

            create unique index if not exists idx_storage_operations_active_target
            on storage_operations(resource_kind, project_id, resource_key)
            where state in ('prepared', 'fs_applied', 'db_committed', 'blocked');

            create unique index if not exists idx_storage_operations_request
            on storage_operations(op_kind, request_key)
            where request_key is not null;

            create index if not exists idx_storage_operation_items_op
            on storage_operation_items(op_id, ordinal);
            """
        )


def _write_json(store: ArtifactStore, relative_path: str, payload: object) -> Path:
    path = store.root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _board_target() -> ResourceTarget:
    return ResourceTarget("board", "demo", "investigation")


def _bootstrap_board(
    store: ArtifactStore, *, payload: object | None = None
) -> StorageOperationJournal:
    journal = StorageOperationJournal(store)
    path = "projects/demo/boards/investigation.json"
    if payload is not None:
        _write_json(store, path, payload)
    journal.bootstrap_head(
        _board_target(),
        primary_relative_path=path,
        version=0,
    )
    return journal


def _operation_row(store: ArtifactStore) -> sqlite3.Row:
    with sqlite3.connect(store.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "select * from storage_operations order by created_at desc limit 1"
        ).fetchone()
    assert row is not None
    return row


def _recover_in_process(
    workspace: str,
    op_id: str,
    ready: Any,
    start: Any,
    results: Any,
) -> None:
    ready.put("ready")
    start.wait(5)

    def widen_staging_window(
        stage: str, _op_id: str, _ordinal: int | None
    ) -> None:
        if stage == "before_stage_write":
            time.sleep(0.15)

    try:
        head = StorageOperationJournal(
            ArtifactStore(workspace, init_db=False),
            fault_hook=widen_staging_window,
        ).recover_replace(op_id)
        results.put(("ok", head.version))
    except Exception as exc:  # pragma: no cover - asserted in the parent process
        results.put(("error", type(exc).__name__))


def test_canonical_digest_ignores_json_format_and_key_order() -> None:
    assert canonical_digest('{"b": 2, "a": [1, 3]}') == canonical_digest(
        {"a": [1, 3], "b": 2}
    )


def test_missing_resource_digest_is_path_canonical_and_requires_unique_paths() -> None:
    assert missing_resource_digest(("z.json", "a.json")) == missing_resource_digest(
        ("a.json", "z.json")
    )
    with pytest.raises(ValueError, match="unique"):
        missing_resource_digest(("a.json", "a.json"))
    with pytest.raises(ValueError, match="sequence"):
        missing_resource_digest("a.json")


def test_deterministic_control_paths_are_workspace_relative() -> None:
    assert staging_relative_path("sop_abc", 2) == (
        ".storage-operations/staging/sop_abc/0002.tmp"
    )
    assert quarantine_relative_path("sop_abc", 2, "projects/demo/sessions/run_a") == (
        ".storage-operations/quarantine/sop_abc/0002-run_a"
    )


def test_board_single_file_replace_commits_head_and_file(store: ArtifactStore) -> None:
    journal = _bootstrap_board(store)
    target_payload = {
        "board_id": "investigation",
        "version": 1,
        "columns": [],
        "cards": [],
    }

    head = journal.replace_resource(
        _board_target(),
        expected_version=0,
        replacements=[
            ReplacementFile(
                "projects/demo/boards/investigation.json",
                json.dumps(target_payload, indent=2),
            )
        ],
    )

    assert head.version == 1
    assert head.content_digest == _operation_row(store)["target_digest"]
    assert json.loads(
        (store.root / "projects/demo/boards/investigation.json").read_text(
            encoding="utf-8"
        )
    ) == target_payload
    assert _operation_row(store)["state"] == "done"


def test_bootstrap_fails_closed_when_file_changes_before_transactional_recheck(
    store: ArtifactStore,
) -> None:
    path = "projects/demo/boards/investigation.json"
    _write_json(store, path, {"board_id": "investigation", "version": 0})

    def mutate(stage: str, _op_id: str, _ordinal: int | None) -> None:
        if stage == "before_bootstrap_recheck":
            _write_json(store, path, {"board_id": "changed", "version": 1})

    with pytest.raises(ResourceDigestMismatchError, match="changed"):
        StorageOperationJournal(store, fault_hook=mutate).bootstrap_head(
            _board_target(),
            primary_relative_path=path,
            version=0,
        )

    assert StorageOperationJournal(store).get_head(_board_target()) is None
    assert json.loads((store.root / path).read_text())["board_id"] == "changed"


def test_bootstrap_rejects_files_changed_after_caller_preread(
    store: ArtifactStore,
) -> None:
    path = "projects/demo/boards/investigation.json"
    original = {"board_id": "investigation", "version": 0}
    _write_json(store, path, original)
    expected = composite_digest({path: canonical_digest(original)})
    _write_json(store, path, {"board_id": "changed", "version": 9})

    with pytest.raises(ResourceDigestMismatchError, match="pre-read"):
        StorageOperationJournal(store).bootstrap_head(
            _board_target(),
            primary_relative_path=path,
            version=0,
            expected_content_digest=expected,
        )

    assert StorageOperationJournal(store).get_head(_board_target()) is None


def test_two_connections_cannot_reserve_same_resource_generation(
    store: ArtifactStore,
) -> None:
    _bootstrap_board(store)
    barrier = Barrier(2)

    def reserve(card_id: str) -> str:
        barrier.wait()
        journal = StorageOperationJournal(ArtifactStore(store.root, init_db=False))
        try:
            journal.reserve_replace(
                _board_target(),
                expected_version=0,
                replacements=[
                    ReplacementFile(
                        "projects/demo/boards/investigation.json",
                        {
                            "board_id": "investigation",
                            "version": 1,
                            "columns": [],
                            "cards": [{"card_id": card_id}],
                        },
                    )
                ],
            )
        except ResourceOperationInProgressError:
            return "conflict"
        return "reserved"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(reserve, ("card_a", "card_b")))

    assert outcomes == ["conflict", "reserved"]
    with sqlite3.connect(store.db_path) as conn:
        count = conn.execute(
            """
            select count(*) from storage_operations
            where state in ('prepared', 'fs_applied', 'db_committed', 'blocked')
            """
        ).fetchone()
    assert count == (1,)


def test_journal_connection_enforces_item_foreign_key_and_parent_cascade(
    store: ArtifactStore,
) -> None:
    journal = _bootstrap_board(store)
    with journal._connect() as conn:
        assert int(conn.execute("pragma foreign_keys").fetchone()[0]) == 1
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            conn.execute(
                """
                insert into storage_operation_items(
                    op_id, ordinal, mode, source_relpath, work_relpath
                ) values(
                    'sop_missing', 0, 'replace_file',
                    'projects/demo/boards/investigation.json',
                    '.storage-operations/staging/sop_missing/0000.tmp'
                )
                """
            )

    reservation = journal.reserve_replace(
        _board_target(),
        expected_version=0,
        replacements=[
            ReplacementFile(
                "projects/demo/boards/investigation.json",
                {"board_id": "investigation", "version": 1},
            )
        ],
    )
    with journal._connect() as conn:
        item_count = conn.execute(
            "select count(*) from storage_operation_items where op_id = ?",
            (reservation.op_id,),
        ).fetchone()
        assert int(item_count[0]) == 1
        conn.execute(
            "delete from storage_operations where op_id = ?",
            (reservation.op_id,),
        )
        item_count = conn.execute(
            "select count(*) from storage_operation_items where op_id = ?",
            (reservation.op_id,),
        ).fetchone()
        assert int(item_count[0]) == 0


def test_crash_after_reserve_recovers_from_base_file(store: ArtifactStore) -> None:
    def fault(stage: str, _op_id: str, _ordinal: int | None) -> None:
        if stage == "after_reserve":
            raise InjectedStorageFault(stage)

    _bootstrap_board(store)
    crashing = StorageOperationJournal(store, fault_hook=fault)
    with pytest.raises(InjectedStorageFault, match="after_reserve"):
        crashing.replace_resource(
            _board_target(),
            expected_version=0,
            replacements=[
                ReplacementFile(
                    "projects/demo/boards/investigation.json",
                    {"board_id": "investigation", "version": 1},
                )
            ],
        )
    row = _operation_row(store)
    assert row["state"] == "prepared"
    assert not (store.root / "projects/demo/boards/investigation.json").exists()

    head = StorageOperationJournal(store).recover_replace(str(row["op_id"]))

    assert head.version == 1
    assert _operation_row(store)["state"] == "done"
    assert json.loads(
        (store.root / "projects/demo/boards/investigation.json").read_text()
    )["version"] == 1


def test_two_processes_recover_one_prepared_operation_without_staging_race(
    store: ArtifactStore,
) -> None:
    path = "projects/demo/boards/investigation.json"
    journal = _bootstrap_board(
        store, payload={"board_id": "investigation", "version": 0}
    )
    reservation = journal.reserve_replace(
        _board_target(),
        expected_version=0,
        replacements=[
            ReplacementFile(path, {"board_id": "investigation", "version": 1})
        ],
    )
    # spawn, not fork: fork does not exist on Windows, and Python 3.14 warns
    # about forking this multi-threaded test process.
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_recover_in_process,
            args=(str(store.root), reservation.op_id, ready, start, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    assert [ready.get(timeout=5) for _ in processes] == ["ready", "ready"]
    start.set()
    for process in processes:
        process.join(timeout=10)

    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(results.get(timeout=5) for _ in processes) == [
        ("ok", 1),
        ("ok", 1),
    ]
    assert _operation_row(store)["state"] == "done"
    assert not (
        store.root / staging_relative_path(reservation.op_id, 0)
    ).exists()


def test_crash_after_replace_before_state_recovers_without_rewriting_base(
    store: ArtifactStore,
) -> None:
    fired = False

    def fault(stage: str, _op_id: str, _ordinal: int | None) -> None:
        nonlocal fired
        if stage == "after_replace" and not fired:
            fired = True
            raise InjectedStorageFault(stage)

    _bootstrap_board(store, payload={"board_id": "investigation", "version": 0})
    with pytest.raises(InjectedStorageFault, match="after_replace"):
        StorageOperationJournal(store, fault_hook=fault).replace_resource(
            _board_target(),
            expected_version=0,
            replacements=[
                ReplacementFile(
                    "projects/demo/boards/investigation.json",
                    {"board_id": "investigation", "version": 1},
                )
            ],
        )
    row = _operation_row(store)
    assert row["state"] == "prepared"
    assert json.loads(
        (store.root / "projects/demo/boards/investigation.json").read_text()
    )["version"] == 1

    head = StorageOperationJournal(store).recover_replace(str(row["op_id"]))
    assert head.version == 1
    assert _operation_row(store)["state"] == "done"


def test_semantic_multi_file_partial_apply_recovers_both_files(
    store: ArtifactStore,
) -> None:
    seeds_path = "projects/demo/semantic/seeds.json"
    versions_path = "projects/demo/semantic/versions.json"
    _write_json(store, seeds_path, {"version": 1, "verified_answers": []})
    _write_json(store, versions_path, {"seeds": 0})
    target = ResourceTarget("semantic_seeds", "demo", "seeds")
    journal = StorageOperationJournal(store)
    journal.bootstrap_head(
        target,
        primary_relative_path=seeds_path,
        version=0,
        tracked_relative_paths=(seeds_path, versions_path),
    )
    fired = False

    def fault(stage: str, _op_id: str, _ordinal: int | None) -> None:
        nonlocal fired
        if stage == "after_replace" and not fired:
            fired = True
            raise InjectedStorageFault(stage)

    replacements = [
        ReplacementFile(
            seeds_path,
            {
                "version": 1,
                "verified_answers": [{"question": "Revenue?", "answer": "$10"}],
            },
        ),
        ReplacementFile(versions_path, {"seeds": 1}),
    ]
    with pytest.raises(InjectedStorageFault, match="after_replace"):
        StorageOperationJournal(store, fault_hook=fault).replace_resource(
            target,
            expected_version=0,
            replacements=replacements,
        )
    row = _operation_row(store)
    assert row["state"] == "prepared"

    head = journal.recover_replace(str(row["op_id"]))

    assert head.version == 1
    assert json.loads((store.root / versions_path).read_text()) == {"seeds": 1}
    answers = json.loads((store.root / seeds_path).read_text())["verified_answers"]
    assert answers == [{"question": "Revenue?", "answer": "$10"}]
    assert _operation_row(store)["state"] == "done"


def test_finalize_fault_leaves_fs_applied_operation_recoverable(
    store: ArtifactStore,
) -> None:
    def fault(stage: str, _op_id: str, _ordinal: int | None) -> None:
        if stage == "before_finalize":
            raise InjectedStorageFault(stage)

    _bootstrap_board(store)
    with pytest.raises(InjectedStorageFault, match="before_finalize"):
        StorageOperationJournal(store, fault_hook=fault).replace_resource(
            _board_target(),
            expected_version=0,
            replacements=[
                ReplacementFile(
                    "projects/demo/boards/investigation.json",
                    {"board_id": "investigation", "version": 1},
                )
            ],
        )

    row = _operation_row(store)
    assert row["state"] == "fs_applied"
    assert StorageOperationJournal(store).get_head(_board_target()).version == 0  # type: ignore[union-attr]

    recovered = StorageOperationJournal(store).recover_replace(str(row["op_id"]))
    assert recovered.version == 1


def test_request_key_replay_reports_original_generation_after_later_write(
    store: ArtifactStore,
) -> None:
    path = "projects/demo/boards/investigation.json"
    journal = _bootstrap_board(store)
    first_payload = {"board_id": "investigation", "version": 1, "cards": ["first"]}
    first = journal.replace_resource(
        _board_target(),
        expected_version=0,
        replacements=[ReplacementFile(path, first_payload)],
        request_key="board-request-1",
    )
    second = journal.replace_resource(
        _board_target(),
        expected_version=1,
        replacements=[
            ReplacementFile(
                path,
                {"board_id": "investigation", "version": 2, "cards": ["second"]},
            )
        ],
        request_key="board-request-2",
    )

    replay = journal.replace_resource(
        _board_target(),
        expected_version=0,
        replacements=[ReplacementFile(path, first_payload)],
        request_key="board-request-1",
    )

    assert first.version == replay.version == 1
    assert replay.content_digest == first.content_digest
    assert second.version == 2
    assert json.loads((store.root / path).read_text())["version"] == 2


def test_public_replay_record_preserves_original_paths_and_payloads(
    store: ArtifactStore,
) -> None:
    path = "projects/demo/boards/investigation.json"
    journal = _bootstrap_board(store)
    payload = '{"version": 1, "board_id": "investigation"}'
    head = journal.replace_resource(
        _board_target(),
        expected_version=0,
        replacements=[ReplacementFile(path, payload)],
        request_key="inspect-replay",
    )

    replay = journal.get_replacement_replay("inspect-replay")

    assert replay is not None
    assert replay.target == _board_target()
    assert replay.expected_version == 0
    assert replay.target_version == head.version == 1
    assert replay.state == "done"
    assert [(item.relative_path, item.payload) for item in replay.items] == [
        (path, payload.encode())
    ]


def test_unknown_digest_blocks_and_never_advances_head(store: ArtifactStore) -> None:
    path = "projects/demo/boards/investigation.json"
    _bootstrap_board(store, payload={"board_id": "investigation", "version": 0})
    journal = StorageOperationJournal(store)
    reservation = journal.reserve_replace(
        _board_target(),
        expected_version=0,
        replacements=[
            ReplacementFile(path, {"board_id": "investigation", "version": 1})
        ],
    )
    _write_json(store, path, {"board_id": "intruder", "version": 99})

    with pytest.raises(StorageOperationBlockedError, match="unknown"):
        journal.apply_replace(reservation.op_id)

    assert _operation_row(store)["state"] == "blocked"
    head = journal.get_head(_board_target())
    assert head is not None and head.version == 0
    assert json.loads((store.root / path).read_text())["version"] == 99
    with pytest.raises(StorageOperationBlockedError):
        journal.recover_replace(reservation.op_id)


@pytest.mark.parametrize(
    "relative_path",
    [
        "../outside.json",
        "/tmp/outside.json",
        "projects/demo/../../outside.json",
        r"projects\\demo\\board.json",
        ".storage-operations/staging/attacker/0000.tmp",
    ],
)
def test_unsafe_replacement_paths_are_rejected(
    store: ArtifactStore, relative_path: str
) -> None:
    _bootstrap_board(store)
    with pytest.raises(UnsafeStoragePathError):
        StorageOperationJournal(store).reserve_replace(
            _board_target(),
            expected_version=0,
            replacements=[ReplacementFile(relative_path, {})],
        )


def test_symlink_parent_is_rejected_without_touching_external_file(
    store: ArtifactStore, tmp_path: Path
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    external_file = external / "board.json"
    external_file.write_text('{"safe": true}', encoding="utf-8")
    link = store.root / "linked"
    link.symlink_to(external, target_is_directory=True)

    target = ResourceTarget("board", "demo", "linked")
    journal = StorageOperationJournal(store)
    with pytest.raises(UnsafeStoragePathError):
        journal.bootstrap_head(
            target,
            primary_relative_path="linked/board.json",
            version=0,
        )
    assert json.loads(external_file.read_text()) == {"safe": True}


def test_malformed_external_json_blocks_reserved_operation(store: ArtifactStore) -> None:
    path = "projects/demo/boards/investigation.json"
    _bootstrap_board(store, payload={"board_id": "investigation", "version": 0})
    journal = StorageOperationJournal(store)
    reservation = journal.reserve_replace(
        _board_target(),
        expected_version=0,
        replacements=[
            ReplacementFile(path, {"board_id": "investigation", "version": 1})
        ],
    )
    (store.root / path).write_text("{malformed", encoding="utf-8")

    with pytest.raises(StorageOperationBlockedError, match="unknown content"):
        journal.apply_replace(reservation.op_id)

    assert _operation_row(store)["state"] == "blocked"
    head = journal.get_head(_board_target())
    assert head is not None and head.version == 0
