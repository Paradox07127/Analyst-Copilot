from __future__ import annotations

from pathlib import Path

from eda_platform.application.services.session_family_service import SessionFamilyService
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.sessions import SessionManifest


def _run(
    store: ArtifactStore,
    project_id: str,
    session_id: str,
    *,
    source_session_id: str | None = None,
) -> None:
    store.start_session(project_id, session_id)
    store.write_manifest(
        SessionManifest(
            session_id=session_id,
            project_id=project_id,
            input_hashes={"orders.csv": "hash"},
            code_version="test",
            model_versions={"analysis": "model-a"},
            seed=42,
            source_session_id=source_session_id,
        )
    )
    store.mark_session_status(project_id, session_id, "completed")


def test_family_collects_machinery_but_not_what_if_variants(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    _run(store, "demo", "root")
    _run(store, "demo", "qsess_child", source_session_id="root")
    _run(store, "demo", "step__internal", source_session_id="qsess_child")
    _run(store, "demo", "fksess_lifecycle", source_session_id="root")
    _run(store, "demo", "run_variant", source_session_id="root")

    family = SessionFamilyService(store).collect("demo", "root")

    assert family.session_ids == ("root", "qsess_child", "step__internal")
    assert family.warnings == ()


def test_lineage_relations_and_broken_or_cyclic_sources(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    _run(store, "demo", "parent")
    _run(store, "demo", "left", source_session_id="parent")
    _run(store, "demo", "right", source_session_id="parent")
    service = SessionFamilyService(store)

    siblings = service.lineage("demo", "left", "right")
    assert siblings.relation == "siblings"
    assert siblings.common_ancestor_session_id == "parent"
    assert siblings.left_path == ["left", "parent"]
    assert service.lineage("demo", "parent", "left").relation == "direct_parent"

    _run(store, "demo", "broken", source_session_id="missing")
    broken = service.lineage("demo", "broken", "right")
    assert broken.relation == "unknown"
    assert any("broken lineage" in warning for warning in broken.warnings)

    _run(store, "demo", "cycle_a", source_session_id="cycle_b")
    _run(store, "demo", "cycle_b", source_session_id="cycle_a")
    cyclic = service.lineage("demo", "cycle_a", "right")
    assert cyclic.relation == "unknown"
    assert any("cycle" in warning for warning in cyclic.warnings)


def test_a_cycle_closing_through_a_root_id_is_reported_not_silently_skipped(
    tmp_path: Path,
) -> None:
    """The family filter used to run first, so a cycle back to an ordinary root
    was dropped by the prefix check before anything noticed it was a cycle. The
    walk still terminated, but it terminated without saying why."""
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    _run(store, "demo", "root", source_session_id="qsess_child")
    _run(store, "demo", "qsess_child", source_session_id="root")

    family = SessionFamilyService(store).collect("demo", "root")

    assert family.session_ids == ("root", "qsess_child")
    assert any("cycle" in warning for warning in family.warnings)
