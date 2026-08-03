from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from eda_platform.core.exploration_shadow_store import (
    ShadowExplorationStore,
    ShadowPathViolationError,
    ShadowProjectionConflictError,
    shadow_run_root,
    validate_shadow_run_path,
)
from eda_platform.schemas.exploration_shadow import (
    ShadowExplorationProjection,
    validate_exploration_id,
)
from eda_platform.schemas.insights import InsightProof, InsightRecord


def _insight(**overrides: object) -> InsightRecord:
    receipt_id = "rcpt_support"
    values: dict[str, object] = {
        "insight_id": "ins_1",
        "hypothesis_id": "hyp_1",
        "family": "Diagnostic",
        "status": "new",
        "trust_level": "supported",
        "claim_bundle_id": "clb_1",
        "supporting_receipt_ids": (receipt_id,),
        "proof": (
            InsightProof(
                receipt_id=receipt_id,
                fact_ids=("fact_1",),
                comparison="supports",
            ),
        ),
        "created_round": 1,
        "last_updated_round": 1,
    }
    values.update(overrides)
    return InsightRecord.model_validate(values)


def _projection(**overrides: object) -> ShadowExplorationProjection:
    values: dict[str, object] = {
        "exploration_id": "expl_abc",
        "last_seq": 4,
        "status": "running",
        "policy_fingerprint": "xplcy_abc",
        "data_state_witness": "dsw_abc",
        "coverage_completed": ("region_difference",),
        "coverage_unexplored": ("missingness_mechanism", "spike_day"),
    }
    values.update(overrides)
    return ShadowExplorationProjection.model_validate(values)


def _terminal_projection(**overrides: object) -> ShadowExplorationProjection:
    values: dict[str, object] = {
        "status": "stopped",
        "stop_reason": "completed",
        "insight_records": (_insight(),),
    }
    values.update(overrides)
    return _projection(**values)


def test_shadow_projection_is_isolated_from_product_artifacts(tmp_path: Path) -> None:
    store = ShadowExplorationStore(tmp_path.resolve())
    path = store.project(_terminal_projection())

    assert path == tmp_path / "exploration-eval/expl_abc/projection.json"
    restored = store.read("expl_abc")
    assert restored is not None
    assert restored.user_visible is False
    assert restored.production_artifact_ids == ()
    assert isinstance(restored.insight_records[0], InsightRecord)
    assert isinstance(restored.insight_records[0].proof[0], InsightProof)
    assert not (tmp_path / "artifacts").exists()


def test_projection_seq_must_advance_because_journal_is_authoritative(tmp_path: Path) -> None:
    store = ShadowExplorationStore(tmp_path.resolve())
    store.project(_projection(last_seq=4))
    with pytest.raises(ShadowProjectionConflictError, match="must advance"):
        store.project(_projection(last_seq=4))
    with pytest.raises(ShadowProjectionConflictError, match="must advance"):
        store.project(_projection(last_seq=3))
    store.project(_projection(last_seq=5))
    restored = store.read("expl_abc")
    assert restored is not None
    assert restored.last_seq == 5


def test_concurrent_writers_cannot_both_publish_the_same_seq(tmp_path: Path) -> None:
    store = ShadowExplorationStore(tmp_path.resolve())
    store.project(_projection(last_seq=4))

    def publish() -> str:
        try:
            store.project(_projection(last_seq=5))
        except ShadowProjectionConflictError:
            return "conflict"
        return "written"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(lambda _index: publish(), range(2)))

    assert outcomes == ["conflict", "written"]
    restored = store.read("expl_abc")
    assert restored is not None
    assert restored.last_seq == 5


@pytest.mark.parametrize(
    "exploration_id",
    (
        "",
        ".",
        "..",
        "../escape",
        "/absolute",
        "nested/run",
        r"nested\run",
        r"C:\absolute",
        "expl space",
        "a" * 129,
    ),
)
def test_exploration_id_validation_is_reusable_at_every_entry_point(
    tmp_path: Path, exploration_id: str
) -> None:
    store = ShadowExplorationStore(tmp_path.resolve())
    with pytest.raises(ValueError, match="exploration_id"):
        validate_exploration_id(exploration_id)
    with pytest.raises(ValueError, match="exploration_id"):
        store.path_for(exploration_id)
    with pytest.raises(ValidationError, match="exploration_id"):
        _projection(exploration_id=exploration_id)


def test_run_path_validator_rejects_relative_parent_and_external_paths(
    tmp_path: Path,
) -> None:
    run_root = shadow_run_root(tmp_path, "expl_abc")
    assert run_root == tmp_path / "exploration-eval/expl_abc"
    assert (
        validate_shadow_run_path(
            tmp_path, "expl_abc", run_root / "journal.jsonl"
        )
        == run_root / "journal.jsonl"
    )

    with pytest.raises(ShadowPathViolationError, match="absolute"):
        validate_shadow_run_path(tmp_path, "expl_abc", "journal.jsonl")
    with pytest.raises(ShadowPathViolationError, match=r"\.\."):
        validate_shadow_run_path(
            tmp_path, "expl_abc", run_root / ".." / "other" / "journal.jsonl"
        )
    with pytest.raises(ShadowPathViolationError, match="contained"):
        validate_shadow_run_path(tmp_path, "expl_abc", tmp_path / "journal.jsonl")


@pytest.mark.parametrize("symlink_location", ("shadow_root", "run_root"))
def test_store_rejects_preexisting_directory_symlinks(
    tmp_path: Path, symlink_location: str
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    shadow_root = tmp_path / "exploration-eval"
    if symlink_location == "shadow_root":
        shadow_root.symlink_to(outside, target_is_directory=True)
    else:
        shadow_root.mkdir()
        (shadow_root / "expl_abc").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ShadowPathViolationError, match="symlink"):
        ShadowExplorationStore(tmp_path).project(_projection())
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("symlink_location", ("projection", "lock"))
def test_store_rejects_preexisting_file_symlinks(
    tmp_path: Path, symlink_location: str
) -> None:
    run_root = tmp_path / "exploration-eval/expl_abc"
    run_root.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("unchanged", encoding="utf-8")
    leaf = (
        run_root / "projection.json"
        if symlink_location == "projection"
        else run_root / "projection.json.lock"
    )
    leaf.symlink_to(outside)

    with pytest.raises(ShadowPathViolationError, match="symlink"):
        ShadowExplorationStore(tmp_path).project(_projection())
    assert outside.read_text(encoding="utf-8") == "unchanged"


def test_shadow_schema_accepts_only_terminal_typed_insight_projections() -> None:
    stopped = _terminal_projection()
    assert stopped.insight_records == (_insight(),)

    with pytest.raises(ValidationError, match="only after.*stopped"):
        _projection(insight_records=(_insight(),))
    with pytest.raises(ValidationError):
        _terminal_projection(insight_records=({"insight_id": "untyped"},))
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _projection(payload={"api_key": "must-not-leak"})


def test_shadow_schema_requires_machine_checkable_proof_projection() -> None:
    mismatched = _insight(
        proof=(
            InsightProof(
                receipt_id="rcpt_other",
                fact_ids=("fact_1",),
                comparison="supports",
            ),
        )
    )
    with pytest.raises(ValidationError, match="uncited receipt"):
        _terminal_projection(insight_records=(mismatched,))


def test_shadow_schema_forbids_any_product_publication_reference() -> None:
    with pytest.raises(ValidationError):
        _projection(production_artifact_ids=("art_1",))
    with pytest.raises(ValidationError):
        _projection(user_visible=True)


def test_stop_reason_only_exists_for_terminal_projection() -> None:
    with pytest.raises(ValidationError, match="stop_reason"):
        _projection(stop_reason="failed")
    stopped = _projection(status="stopped", stop_reason="completed")
    assert stopped.stop_reason == "completed"
