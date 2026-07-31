"""Versioned semantic seeds repository over the cross-media storage journal."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from eda_platform.core.semantic import FieldMeaning, SemanticSeeds
from eda_platform.core.semantic_resources import (
    SemanticResourceStateError,
    SemanticSeedsRepository,
    load_semantic_seeds_safe,
)
from eda_platform.core.storage_operations import (
    ReplacementFile,
    ResourceDigestMismatchError,
    ResourceOperationInProgressError,
    ResourceTarget,
    ResourceVersionConflictError,
    StorageOperationJournal,
    StorageRequestKeyConflictError,
)
from eda_platform.core.store import ArtifactStore


def _seeds(label: str) -> SemanticSeeds:
    return SemanticSeeds(
        field_meanings=[
            FieldMeaning(dataset="orders", column="region", meaning=label)
        ]
    )


def test_legacy_bootstrap_uses_missing_seed_version_zero(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    semantic_dir = store.project_dir("demo") / "semantic"
    semantic_dir.mkdir(parents=True)
    (semantic_dir / "seeds.json").write_text(
        _seeds("legacy").model_dump_json(),
        encoding="utf-8",
    )

    snapshot = SemanticSeedsRepository(store, "demo").read()

    assert snapshot.version == 0
    assert snapshot.seeds.field_meanings[0].meaning == "legacy"
    assert len(snapshot.content_digest) == 64
    with sqlite3.connect(store.db_path) as conn:
        head = conn.execute(
            """
            select version, relative_path, content_digest from resource_heads
            where resource_kind = 'semantic_seeds'
              and project_id = 'demo' and resource_key = 'seeds'
            """
        ).fetchone()
    assert head == (
        0,
        "projects/demo/semantic/seeds.json",
        snapshot.content_digest,
    )


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        ("seeds.json", "{broken"),
        ("versions.json", '{"seeds": "zero"}'),
    ],
)
def test_corrupt_legacy_semantic_files_fail_closed(
    tmp_path: Path,
    filename: str,
    payload: str,
) -> None:
    store = ArtifactStore(tmp_path)
    semantic_dir = store.project_dir("demo") / "semantic"
    semantic_dir.mkdir(parents=True)
    (semantic_dir / filename).write_text(payload, encoding="utf-8")

    with pytest.raises(SemanticResourceStateError):
        SemanticSeedsRepository(store, "demo").read()


def test_two_connections_same_version_allow_exactly_one_replace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    first = SemanticSeedsRepository(ArtifactStore(workspace), "demo")
    second = SemanticSeedsRepository(ArtifactStore(workspace), "demo")
    assert first.read().version == 0
    barrier = Barrier(2)

    def replace(repository: SemanticSeedsRepository, label: str) -> str:
        barrier.wait()
        try:
            repository.replace_seeds(
                expected_version=0,
                new_seeds=_seeds(label),
                request_key=f"request-{label}",
            )
        except (ResourceVersionConflictError, ResourceOperationInProgressError):
            return "conflict"
        return "committed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(
            pool.map(
                lambda args: replace(*args),
                ((first, "first"), (second, "second")),
            )
        )

    assert outcomes == ["committed", "conflict"]
    snapshot = SemanticSeedsRepository(ArtifactStore(workspace), "demo").read()
    assert snapshot.version == 1
    assert snapshot.seeds.field_meanings[0].meaning in {"first", "second"}


@pytest.mark.parametrize(
    "fault_stage",
    ["after_reserve", "after_replace", "after_apply_before_state"],
)
def test_replace_recovers_after_crash_boundaries(
    tmp_path: Path,
    fault_stage: str,
) -> None:
    store = ArtifactStore(tmp_path)
    assert SemanticSeedsRepository(store, "demo").read().version == 0

    def fail_at_boundary(stage: str, _op_id: str, _ordinal: int | None) -> None:
        if stage == fault_stage:
            raise RuntimeError("injected crash")

    crashing = SemanticSeedsRepository(
        store,
        "demo",
        journal=StorageOperationJournal(store, fault_hook=fail_at_boundary),
    )
    with pytest.raises(RuntimeError, match="injected crash"):
        crashing.replace_seeds(
            expected_version=0,
            new_seeds=_seeds("recovered"),
            request_key="recover-request",
        )

    recovered = SemanticSeedsRepository(ArtifactStore(tmp_path), "demo").read()
    assert recovered.version == 1
    assert recovered.seeds.field_meanings[0].meaning == "recovered"


def test_runtime_safe_loader_recovers_prepared_generation(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    SemanticSeedsRepository(store, "demo").read()

    def fail_after_reserve(
        stage: str, _op_id: str, _ordinal: int | None
    ) -> None:
        if stage == "after_reserve":
            raise RuntimeError("prepared crash")

    with pytest.raises(RuntimeError, match="prepared crash"):
        SemanticSeedsRepository(
            store,
            "demo",
            journal=StorageOperationJournal(store, fault_hook=fail_after_reserve),
        ).replace_seeds(
            expected_version=0,
            new_seeds=_seeds("runtime recovered"),
            request_key="runtime-prepared",
        )

    loaded = load_semantic_seeds_safe(store, "demo")
    assert loaded is not None
    assert loaded.field_meanings[0].meaning == "runtime recovered"


def test_runtime_safe_loader_fails_open_for_corrupt_generation(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    SemanticSeedsRepository(store, "demo").read()
    seeds_path = store.project_dir("demo") / "semantic" / "seeds.json"
    seeds_path.parent.mkdir(parents=True, exist_ok=True)
    seeds_path.write_text("{broken", encoding="utf-8")

    assert load_semantic_seeds_safe(store, "demo") is None


def test_out_of_band_semantic_write_fails_closed(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    repository = SemanticSeedsRepository(store, "demo")
    repository.read()
    seeds_path = store.project_dir("demo") / "semantic" / "seeds.json"
    seeds_path.parent.mkdir(parents=True, exist_ok=True)
    seeds_path.write_text(_seeds("tampered").model_dump_json(), encoding="utf-8")

    with pytest.raises(ResourceDigestMismatchError):
        repository.read()


def test_request_key_replays_one_promotion_write(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    repository = SemanticSeedsRepository(store, "demo")
    repository.read()
    promoted = _seeds("promoted")

    first = repository.replace_seeds(
        expected_version=0,
        new_seeds=promoted,
        request_key="promotion-finding-1",
    )
    replay = repository.replace_seeds(
        expected_version=0,
        new_seeds=promoted,
        request_key="promotion-finding-1",
    )

    assert replay.version == first.version == 1
    assert replay.content_digest == first.content_digest
    with sqlite3.connect(store.db_path) as conn:
        count = conn.execute(
            """
            select count(*) from storage_operations
            where op_kind = 'replace_resource'
              and resource_kind = 'semantic_seeds'
              and project_id = 'demo'
            """
        ).fetchone()
    assert count == (1,)


def test_snapshot_does_not_follow_later_disk_generation(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    repository = SemanticSeedsRepository(store, "demo")
    before = repository.read()

    after = repository.replace_seeds(
        expected_version=0,
        new_seeds=_seeds("new generation"),
        request_key="edit-1",
    )

    assert before.version == 0
    assert before.seeds.field_meanings == []
    assert after.version == 1
    assert after.seeds.field_meanings[0].meaning == "new generation"
    versions = json.loads(
        (store.project_dir("demo") / "semantic" / "versions.json").read_text(
            encoding="utf-8"
        )
    )
    assert versions["seeds"] == 1


def test_same_raw_request_key_is_isolated_by_project(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    first = SemanticSeedsRepository(store, "project-a")
    second = SemanticSeedsRepository(store, "project-b")
    first.read()
    second.read()

    first_snapshot = first.replace_seeds(
        expected_version=0,
        new_seeds=_seeds("first project"),
        request_key="shared-raw-key",
    )
    second_snapshot = second.replace_seeds(
        expected_version=0,
        new_seeds=_seeds("second project"),
        request_key="shared-raw-key",
    )

    assert first_snapshot.seeds.field_meanings[0].meaning == "first project"
    assert second_snapshot.seeds.field_meanings[0].meaning == "second project"
    with sqlite3.connect(store.db_path) as conn:
        rows = conn.execute(
            """
            select project_id, request_key from storage_operations
            where resource_kind = 'semantic_seeds'
            order by project_id
            """
        ).fetchall()
    assert [row[0] for row in rows] == ["project-a", "project-b"]
    assert rows[0][1] != rows[1][1]
    assert all(str(row[1]).startswith("semantic-seeds:") for row in rows)


def test_replay_after_later_generation_returns_original_operation_snapshot(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    repository = SemanticSeedsRepository(store, "demo")
    repository.read()
    original = repository.replace_seeds(
        expected_version=0,
        new_seeds=_seeds("generation one"),
        request_key="original-request",
    )
    seeds_path = "projects/demo/semantic/seeds.json"
    versions_path = "projects/demo/semantic/versions.json"
    proposals_path = "projects/demo/semantic/meaning_proposals.json"
    StorageOperationJournal(store).replace_resource(
        ResourceTarget("semantic_seeds", "demo", "seeds"),
        expected_version=1,
        replacements=(
            ReplacementFile(
                seeds_path,
                _seeds("generation two").model_dump(mode="json"),
            ),
            ReplacementFile(versions_path, {"seeds": 2, "taxonomy": 99}),
            ReplacementFile(
                proposals_path,
                original.proposals.model_dump(mode="json"),
            ),
        ),
        request_key="later-generation",
    )

    replay = repository.replace_seeds(
        expected_version=0,
        new_seeds=_seeds("generation one"),
        request_key="original-request",
    )

    assert replay == original
    assert replay.version == 1
    assert replay.seeds.field_meanings[0].meaning == "generation one"
    assert SemanticSeedsRepository(store, "demo").read().version == 2
    current_versions = json.loads((store.root / versions_path).read_text())
    assert current_versions == {"seeds": 2, "taxonomy": 99}


def test_same_request_key_with_different_seed_semantics_conflicts_after_v2(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    repository = SemanticSeedsRepository(store, "demo")
    repository.read()
    repository.replace_seeds(
        expected_version=0,
        new_seeds=_seeds("original"),
        request_key="semantic-replay-key",
    )
    repository.replace_seeds(
        expected_version=1,
        new_seeds=_seeds("later"),
        request_key="later-key",
    )

    with pytest.raises(StorageRequestKeyConflictError, match="different seeds"):
        repository.replace_seeds(
            expected_version=0,
            new_seeds=_seeds("different"),
            request_key="semantic-replay-key",
        )


def test_same_active_request_recovers_before_returning_original_snapshot(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    assert SemanticSeedsRepository(store, "demo").read().version == 0

    def crash_after_reserve(
        stage: str, _op_id: str, _ordinal: int | None
    ) -> None:
        if stage == "after_reserve":
            raise RuntimeError("reserved crash")

    crashing = SemanticSeedsRepository(
        store,
        "demo",
        journal=StorageOperationJournal(store, fault_hook=crash_after_reserve),
    )
    with pytest.raises(RuntimeError, match="reserved crash"):
        crashing.replace_seeds(
            expected_version=0,
            new_seeds=_seeds("recover me"),
            request_key="active-replay",
        )

    replay = SemanticSeedsRepository(store, "demo").replace_seeds(
        expected_version=0,
        new_seeds=_seeds("recover me"),
        request_key="active-replay",
    )

    assert replay.version == 1
    assert replay.seeds.field_meanings[0].meaning == "recover me"


def test_nonzero_versions_file_cannot_hide_missing_seeds(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    semantic_dir = store.project_dir("demo") / "semantic"
    semantic_dir.mkdir(parents=True)
    (semantic_dir / "versions.json").write_text(
        '{"seeds": 3}',
        encoding="utf-8",
    )

    with pytest.raises(SemanticResourceStateError, match="missing"):
        SemanticSeedsRepository(store, "demo").read()
