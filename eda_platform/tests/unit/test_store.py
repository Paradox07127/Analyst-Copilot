import json
import sqlite3
from datetime import UTC, datetime

import pytest

from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.sessions import SessionManifest, TraceEvent


def test_store_saves_artifact_payload_and_sqlite_index(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "run_demo")
    artifact = Artifact(
        id="prof_abc12345",
        type=ArtifactType.DATASET_PROFILE,
        project_id="project_demo",
        session_id="run_demo",
        created_at=datetime(2026, 7, 3, tzinfo=UTC),
        payload={"dataset_id": "ds_orders", "rows": 2},
    )

    store.save_artifact(artifact)

    loaded = store.get_artifact("prof_abc12345")
    payload_path = tmp_path / "projects/project_demo/sessions/run_demo/artifacts/prof_abc12345.json"
    conn = sqlite3.connect(tmp_path / "state.sqlite")
    row = conn.execute(
        "select artifact_id, artifact_type, project_id, session_id from artifacts"
    ).fetchone()

    assert loaded.payload["rows"] == 2
    assert json.loads(payload_path.read_text())["id"] == "prof_abc12345"
    assert row == ("prof_abc12345", "DatasetProfile", "project_demo", "run_demo")


def test_store_writes_manifest_and_trace(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "run_demo")
    manifest = SessionManifest(
        session_id="run_demo",
        project_id="project_demo",
        input_hashes={"orders.csv": "abc"},
        code_version="unknown",
    )
    trace = TraceEvent(
        session_id="run_demo",
        event_type="step_completed",
        name="profile_dataset",
        started_at=datetime(2026, 7, 3, tzinfo=UTC),
        finished_at=datetime(2026, 7, 3, tzinfo=UTC),
    )

    store.write_manifest(manifest)
    store.append_trace("project_demo", trace)

    manifest_path = tmp_path / "projects/project_demo/sessions/run_demo/manifest.json"
    trace_path = tmp_path / "projects/project_demo/sessions/run_demo/trace.jsonl"

    assert json.loads(manifest_path.read_text())["session_id"] == "run_demo"
    assert store.read_manifest("project_demo", "run_demo") == manifest
    assert json.loads(trace_path.read_text().splitlines()[0])["name"] == "profile_dataset"


def test_store_lists_artifacts_by_run(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "run_demo")
    store.save_artifact(
        Artifact(
            id="report_abc12345",
            type=ArtifactType.MARKDOWN_REPORT,
            project_id="project_demo",
            session_id="run_demo",
            payload={"markdown": "# Report"},
        )
    )

    artifacts = store.list_artifacts(project_id="project_demo", session_id="run_demo")

    assert [artifact.id for artifact in artifacts] == ["report_abc12345"]


@pytest.mark.parametrize(
    "project_id",
    [
        "",
        ".",
        "..",
        "parent/child",
        r"parent\child",
        "cafe\u0301",
    ],
)
def test_project_dir_rejects_noncanonical_or_multisegment_ids(
    tmp_path, project_id: str
) -> None:
    store = ArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="canonical non-empty path segment"):
        store.project_dir(project_id)
    with pytest.raises(ValueError, match="canonical non-empty path segment"):
        store.session_dir(project_id, "run_demo")


@pytest.mark.parametrize("project_id", ["Project Alpha", "acme.inc", "v1.2 draft"])
def test_project_dir_preserves_legal_spaces_and_dots(tmp_path, project_id: str) -> None:
    store = ArtifactStore(tmp_path)

    assert store.project_dir(project_id) == tmp_path / "projects" / project_id


def test_store_lists_trace_events_by_project_and_run_in_order(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "run_demo")
    store.start_session("project_demo", "other_run")
    first = TraceEvent(
        session_id="run_demo",
        event_type="step_started",
        name="profile_dataset",
        summary={"index": 1},
    )
    second = TraceEvent(
        session_id="run_demo",
        event_type="step_completed",
        name="profile_dataset",
        summary={"index": 2},
    )
    other = TraceEvent(
        session_id="other_run",
        event_type="step_completed",
        name="scan_quality",
        summary={"index": 3},
    )

    store.append_trace("project_demo", first)
    store.append_trace("project_demo", other)
    store.append_trace("project_demo", second)

    events = store.list_trace_events(project_id="project_demo", session_id="run_demo")

    assert [(event.event_type, event.name) for event in events] == [
        ("step_started", "profile_dataset"),
        ("step_completed", "profile_dataset"),
    ]
    assert [event.summary["index"] for event in events] == [1, 2]


def _partition_artifact(project_id: str, session_id: str, rows: int) -> Artifact:
    return Artifact(
        id="prof_shared01",
        type=ArtifactType.DATASET_PROFILE,
        project_id=project_id,
        session_id=session_id,
        payload={"dataset_id": "ds_orders", "name": "orders.csv", "rows": rows},
    )


def _two_partition_store(tmp_path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    for project, run, rows in (("proj_a", "run_a", 1), ("proj_b", "run_b", 2)):
        store.ensure_project(project, name=project)
        store.start_session(project, run)
        store.save_artifact(_partition_artifact(project, run, rows))
    return store


def test_same_artifact_id_across_partitions_does_not_steal_rows(tmp_path) -> None:
    """Regression (slice-E review F4): artifact_id is content-derived, so the
    same dataset in two projects shares one id. Saving it in project B must
    not remove project A's index row."""
    store = _two_partition_store(tmp_path)

    a_rows = store.list_artifacts(project_id="proj_a", session_id="run_a")
    b_rows = store.list_artifacts(project_id="proj_b", session_id="run_b")

    assert [(a.project_id, a.session_id) for a in a_rows] == [("proj_a", "run_a")]
    assert [(a.project_id, a.session_id) for a in b_rows] == [("proj_b", "run_b")]
    assert store.artifact_type_counts("proj_a", "run_a") == {"DatasetProfile": 1}
    assert store.artifact_type_counts("proj_b", "run_b") == {"DatasetProfile": 1}


def test_get_artifact_scoped_and_global_ambiguous(tmp_path) -> None:
    store = _two_partition_store(tmp_path)

    scoped = store.get_artifact("prof_shared01", project_id="proj_a", session_id="run_a")
    assert (scoped.project_id, scoped.session_id) == ("proj_a", "run_a")

    with pytest.raises(ValueError, match="ambiguous artifact identity"):
        store.get_artifact("prof_shared01")
    with pytest.raises(ValueError, match="ambiguous artifact identity"):
        store.artifact_index_row("prof_shared01")
    scoped_row = store.artifact_index_row("prof_shared01", project_id="proj_a", session_id="run_a")
    assert scoped_row is not None
    assert (scoped_row["project_id"], scoped_row["session_id"]) == ("proj_a", "run_a")


def test_init_db_migrates_legacy_single_pk_artifacts_table(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("proj_a", name="proj_a")
    store.start_session("proj_a", "run_a")
    path = store.save_artifact(_partition_artifact("proj_a", "run_a", 1))
    rel = str(path.resolve().relative_to(store.root.resolve()))
    # Rebuild the artifacts table with the legacy single-column PK schema.
    with sqlite3.connect(tmp_path / "state.sqlite") as conn:
        conn.executescript(
            """
            drop table artifacts;
            create table artifacts (
                artifact_id text primary key,
                artifact_type text not null,
                project_id text not null,
                session_id text not null,
                path text not null
            );
            """
        )
        conn.execute(
            "insert into artifacts values (?, ?, ?, ?, ?)",
            ("prof_shared01", "DatasetProfile", "proj_a", "run_a", rel),
        )

    reopened = ArtifactStore(tmp_path)

    with sqlite3.connect(tmp_path / "state.sqlite") as conn:
        pk_cols = {
            row[1]: row[5] for row in conn.execute("pragma table_info(artifacts)") if row[5] > 0
        }
        indexes = {row[1] for row in conn.execute("pragma index_list(artifacts)")}
        count = conn.execute("select count(*) from artifacts").fetchone()[0]
    assert pk_cols == {"artifact_id": 1, "project_id": 2, "session_id": 3}
    assert "idx_artifacts_run_type" in indexes
    assert "idx_artifacts_run_order" in indexes
    assert count == 1  # legacy data survived the rebuild
    assert reopened.get_artifact("prof_shared01").project_id == "proj_a"

    # The rebuilt table accepts the cross-partition sibling row.
    reopened.ensure_project("proj_b", name="proj_b")
    reopened.start_session("proj_b", "run_b")
    reopened.save_artifact(_partition_artifact("proj_b", "run_b", 2))
    assert len(reopened.list_artifacts(project_id="proj_a", session_id="run_a")) == 1
    assert len(reopened.list_artifacts(project_id="proj_b", session_id="run_b")) == 1


def test_unfiltered_artifact_pagination_uses_run_order_index(tmp_path) -> None:
    store = ArtifactStore(tmp_path)

    with sqlite3.connect(store.db_path) as conn:
        plan = conn.execute(
            """
            explain query plan
            select rowid, artifact_id, artifact_type, path from artifacts
            where project_id = ? and session_id = ?
            order by rowid limit ?
            """,
            ("project_demo", "run_demo", 51),
        ).fetchall()

    descriptions = [str(row[3]) for row in plan]
    assert any("idx_artifacts_run_order" in description for description in descriptions)
    assert all("TEMP B-TREE" not in description for description in descriptions)
