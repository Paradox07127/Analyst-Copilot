"""Semantic API slice: run-scoped read of the project semantic layer, seeds
editing under optimistic locking (409 `version_conflict`), idempotent join
confirm/revoke (confirmed is never downgraded), and proposal review."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from semantic_test_helpers import load_seeds, save_meaning_proposals, save_seeds

from eda_platform.api.main import create_app
from eda_platform.application.services.semantic_service import SemanticService
from eda_platform.core.column_roles import ColumnRole, ColumnRoleName, ColumnRoleSet
from eda_platform.core.meaning_proposals import (
    MeaningProposal,
    MeaningProposals,
)
from eda_platform.core.semantic import (
    ColumnRoleSeed,
    EntityNote,
    FieldMeaning,
    JoinWhitelist,
    JoinWhitelistEntry,
    MetricDefinition,
    SemanticSeeds,
    VerifiedAnswer,
    VerifiedRelation,
    load_join_whitelist,
    save_join_whitelist,
)
from eda_platform.core.semantic_resources import SemanticSeedsRepository
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile

PROJECT = "proj_semantic"
RUN = "run_semantic"
ORDERS = "orders.csv"
CUSTOMERS = "customers.csv"
SEEDED_VERIFIED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _profile(name: str, dataset_id: str) -> DatasetProfile:
    return DatasetProfile(
        dataset_id=dataset_id,
        name=name,
        rows=2,
        columns=1,
        column_names=["amount"],
        dtypes={"amount": "int64"},
        missing_values={"amount": 0},
        missing_percent={"amount": 0.0},
        numeric_columns=["amount"],
        categorical_columns=[],
    )


def _join_entry(
    label_columns: tuple[str, str],
    *,
    status: str,
    validation_verified: bool,
    left_id: str | None = "ds_orders_1",
    right_id: str | None = "ds_customers_1",
) -> JoinWhitelistEntry:
    return JoinWhitelistEntry(
        left_dataset=ORDERS,
        left_dataset_id=left_id,
        left_columns=[label_columns[0]],
        right_dataset=CUSTOMERS,
        right_dataset_id=right_id,
        right_columns=[label_columns[1]],
        cardinality="many_to_one",
        validation_verified=validation_verified,
        status=status,  # type: ignore[arg-type]
    )


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, "Semantic project")
    store.start_session(PROJECT, RUN)
    for name, dataset_id in ((ORDERS, "ds_orders_1"), (CUSTOMERS, "ds_customers_1")):
        store.save_artifact(
            Artifact(
                id=f"profile_{dataset_id}",
                type=ArtifactType.DATASET_PROFILE,
                project_id=PROJECT,
                session_id=RUN,
                payload=_profile(name, dataset_id).model_dump(),
            )
        )
    store.save_artifact(
        Artifact(
            id="roles_orders",
            type=ArtifactType.COLUMN_ROLE_SET,
            project_id=PROJECT,
            session_id=RUN,
            payload=ColumnRoleSet(
                dataset=ORDERS,
                roles=[
                    ColumnRole(
                        column="amount",
                        role=ColumnRoleName.MEASURE,
                        confidence=0.9,
                        provenance="inferred",
                    )
                ],
            ).model_dump(mode="json"),
        )
    )
    project_dir = store.project_dir(PROJECT)
    save_seeds(
        project_dir,
        SemanticSeeds(
            field_meanings=[
                FieldMeaning(
                    dataset=ORDERS,
                    column="amount",
                    meaning="Order value before refunds.",
                    unit="USD",
                )
            ],
            metric_definitions=[
                MetricDefinition(
                    name="Active user",
                    definition="A user with at least one session in 28 days.",
                    formula="count(distinct user_id)",
                )
            ],
            entity_notes=[
                EntityNote(name="customer", note="One row per billing account.")
            ],
            verified_answers=[
                VerifiedAnswer(
                    question="What was Q3 revenue?",
                    answer="$4.2M.",
                    verified_at=SEEDED_VERIFIED_AT,
                )
            ],
        ),
    )
    save_join_whitelist(
        project_dir,
        JoinWhitelist(
            entries=[
                _join_entry(("customer_id", "customer_id"), status="proposed",
                            validation_verified=True),
                _join_entry(("cust_ref", "customer_id"), status="auto_confirmed",
                            validation_verified=True),
                _join_entry(("guess_id", "customer_id"), status="proposed",
                            validation_verified=False),
                _join_entry(("account_id", "customer_id"), status="confirmed",
                            validation_verified=True),
            ]
        ),
    )
    save_meaning_proposals(
        project_dir,
        MeaningProposals(
            proposals=[
                MeaningProposal(
                    dataset=ORDERS,
                    column="amount",
                    meaning="Gross order amount.",
                    unit_guess="USD",
                    confidence="verified",
                ),
                MeaningProposal(
                    dataset=CUSTOMERS,
                    column="amount",
                    meaning="Customer lifetime value guess.",
                ),
            ]
        ),
    )
    return tmp_path


@pytest.fixture()
def client(workspace: Path) -> TestClient:
    return TestClient(create_app(workspace))


def _view(client: TestClient) -> dict:
    response = client.get(f"/api/v1/sessions/{RUN}/semantic")
    assert response.status_code == 200, response.text
    return response.json()


def _put_seeds(client: TestClient, expected_version: int, field_meanings: list[dict]):
    return client.put(
        f"/api/v1/sessions/{RUN}/semantic/seeds",
        json={"expected_version": expected_version, "field_meanings": field_meanings},
    )


def test_view_shapes_seeds_roles_whitelist_and_pending_proposals(
    client: TestClient,
) -> None:
    body = _view(client)
    assert body["session_id"] == RUN
    assert body["project_id"] == PROJECT
    # Pre-existing seeds file without a sidecar reads as version 0.
    assert body["seeds_version"] == 0
    assert body["field_meanings"] == [
        {
            "dataset": ORDERS,
            "column": "amount",
            "meaning": "Order value before refunds.",
            "unit": "USD",
            "aliases": [],
        }
    ]
    assert body["column_roles"] == [
        {"dataset": ORDERS, "column": "amount", "role": "measure", "confidence": 0.9}
    ]
    statuses = {entry["label"]: entry["status"] for entry in body["join_whitelist"]}
    assert statuses == {
        f"{ORDERS}.customer_id -> {CUSTOMERS}.customer_id": "proposed",
        f"{ORDERS}.cust_ref -> {CUSTOMERS}.customer_id": "auto_confirmed",
        f"{ORDERS}.guess_id -> {CUSTOMERS}.customer_id": "proposed",
        f"{ORDERS}.account_id -> {CUSTOMERS}.customer_id": "confirmed",
    }
    freshness = {
        entry["label"]: entry["freshness"] for entry in body["join_whitelist"]
    }
    assert freshness[f"{ORDERS}.customer_id -> {CUSTOMERS}.customer_id"] == "fresh"
    assert freshness[f"{ORDERS}.guess_id -> {CUSTOMERS}.customer_id"] == "unverifiable"
    assert [(p["dataset"], p["column"]) for p in body["proposals"]] == [
        (ORDERS, "amount"),
        (CUSTOMERS, "amount"),
    ]


def test_semantic_view_is_paginated(client: TestClient) -> None:
    first = client.get(
        f"/api/v1/sessions/{RUN}/semantic", params={"limit": 1}
    ).json()
    assert all(
        len(first[name]) <= 1
        for name in (
            "field_meanings",
            "metric_definitions",
            "entity_notes",
            "verified_answers",
            "verified_relations",
            "column_roles",
            "join_whitelist",
            "proposals",
        )
    )
    assert first["next_cursor"]
    second = client.get(
        f"/api/v1/sessions/{RUN}/semantic",
        params={"limit": 1, "cursor": first["next_cursor"]},
    ).json()
    assert second["join_whitelist"] != first["join_whitelist"]


def test_indexed_semantic_page_does_not_reparse_snapshot(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = client.get(
        f"/api/v1/sessions/{RUN}/semantic", params={"limit": 1}
    ).json()
    assert first["next_cursor"]

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("an indexed page must not parse the semantic JSON")

    monkeypatch.setattr(SemanticSeedsRepository, "read", forbidden)
    second = client.get(
        f"/api/v1/sessions/{RUN}/semantic",
        params={"limit": 1, "cursor": first["next_cursor"]},
    )
    assert second.status_code == 200


def test_semantic_cursor_rejects_source_change_before_reparse(
    client: TestClient,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = client.get(
        f"/api/v1/sessions/{RUN}/semantic", params={"limit": 1}
    ).json()
    path = ArtifactStore(workspace).project_dir(PROJECT) / "semantic" / "seeds.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a stale cursor must fail before parsing semantic JSON")

    monkeypatch.setattr(SemanticSeedsRepository, "read", forbidden)
    stale = client.get(
        f"/api/v1/sessions/{RUN}/semantic",
        params={"limit": 1, "cursor": first["next_cursor"]},
    )
    assert stale.status_code == 400
    assert stale.json()["error"]["code"] == "invalid_cursor"


def test_semantic_cursor_is_bound_to_run(
    client: TestClient, workspace: Path
) -> None:
    cursor = client.get(
        f"/api/v1/sessions/{RUN}/semantic", params={"limit": 1}
    ).json()["next_cursor"]
    ArtifactStore(workspace).start_session(PROJECT, "run_semantic_other")
    replay = client.get(
        "/api/v1/sessions/run_semantic_other/semantic",
        params={"limit": 1, "cursor": cursor},
    )
    assert replay.status_code == 400
    assert replay.json()["error"]["code"] == "invalid_cursor"


def test_seeds_update_bumps_version_and_persists(
    client: TestClient, workspace: Path
) -> None:
    updated = _put_seeds(
        client,
        0,
        [
            {
                "dataset": ORDERS,
                "column": "amount",
                "meaning": "Net order value.",
                "unit": "EUR",
                "aliases": ["order_value"],
            }
        ],
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["version"] == 1
    seeds = load_seeds(ArtifactStore(workspace).project_dir(PROJECT))
    assert seeds.field_meanings[0].meaning == "Net order value."
    assert seeds.field_meanings[0].unit == "EUR"
    assert seeds.field_meanings[0].aliases == ["order_value"]
    # Sidecar file carries the counter; the seeds schema version is untouched.
    versions_path = (
        ArtifactStore(workspace).project_dir(PROJECT) / "semantic" / "versions.json"
    )
    assert json.loads(versions_path.read_text(encoding="utf-8")) == {"seeds": 1}
    assert seeds.version == 1  # schema tag, not the edit counter

    second = _put_seeds(client, 1, [])
    assert second.status_code == 200
    assert second.json()["version"] == 2
    assert load_seeds(ArtifactStore(workspace).project_dir(PROJECT)).field_meanings == []


def test_seeds_update_with_stale_version_is_409(client: TestClient) -> None:
    assert _put_seeds(client, 0, []).status_code == 200
    stale = _put_seeds(client, 0, [])
    assert stale.status_code == 409
    body = stale.json()
    assert body["error"]["code"] == "version_conflict"
    assert "current version is 1" in body["error"]["message"]
    # The view reports the current version so the client can reload and retry.
    assert _view(client)["seeds_version"] == 1


def test_seeds_update_requires_complete_rows(client: TestClient) -> None:
    response = _put_seeds(
        client, 0, [{"dataset": ORDERS, "column": "amount", "meaning": "   "}]
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "semantic_invalid"


def _confirm(
    client: TestClient, label: str, *, expected_version: int | None = None
):
    version = (
        _view(client)["seeds_version"]
        if expected_version is None
        else expected_version
    )
    return client.post(
        f"/api/v1/sessions/{RUN}/semantic/joins/confirm",
        json={"label": label, "expected_version": version},
    )


def _revoke(
    client: TestClient, label: str, *, expected_version: int | None = None
):
    version = (
        _view(client)["seeds_version"]
        if expected_version is None
        else expected_version
    )
    return client.post(
        f"/api/v1/sessions/{RUN}/semantic/joins/revoke",
        json={"label": label, "expected_version": version},
    )


def test_join_confirm_is_idempotent(client: TestClient) -> None:
    label = f"{ORDERS}.customer_id -> {CUSTOMERS}.customer_id"
    first = _confirm(client, label)
    assert first.status_code == 200, first.text
    assert first.json()["status"] == "confirmed"
    replay = _confirm(client, label)
    assert replay.status_code == 200
    assert replay.json()["status"] == "confirmed"


def test_join_confirm_rejects_a_stale_semantic_version(client: TestClient) -> None:
    label = f"{ORDERS}.customer_id -> {CUSTOMERS}.customer_id"
    assert _put_seeds(client, 0, []).status_code == 200
    stale = _confirm(client, label, expected_version=0)

    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "version_conflict"
    entry = next(item for item in _view(client)["join_whitelist"] if item["label"] == label)
    assert entry["status"] == "proposed"


def test_relationship_confirm_rolls_back_whitelist_when_seed_commit_fails(
    client: TestClient,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label = f"{ORDERS}.customer_id -> {CUSTOMERS}.customer_id"

    def fail_replace(*args: object, **kwargs: object) -> None:
        raise OSError("semantic journal unavailable")

    monkeypatch.setattr(SemanticSeedsRepository, "replace_seeds", fail_replace)
    with pytest.raises(OSError, match="semantic journal unavailable"):
        cast(Any, client.app).state.semantic_service.confirm_join_and_sink_relation(
            RUN,
            label=label,
            left=f"{ORDERS}.customer_id",
            right=f"{CUSTOMERS}.customer_id",
            cardinality="many_to_one",
            source_session_id=RUN,
            expected_version=0,
        )

    project_dir = ArtifactStore(workspace).project_dir(PROJECT)
    entry = load_join_whitelist(project_dir).entry(label)
    assert entry is not None and entry.status == "proposed"
    assert load_seeds(project_dir).verified_relations == []


def test_relationship_confirm_recovers_a_commit_then_response_failure(
    client: TestClient,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label = f"{ORDERS}.customer_id -> {CUSTOMERS}.customer_id"
    original_replace = SemanticSeedsRepository.replace_seeds

    def commit_then_fail(
        repository: SemanticSeedsRepository, **kwargs: object
    ):
        original_replace(repository, **kwargs)  # type: ignore[arg-type]
        raise OSError("response path failed")

    monkeypatch.setattr(SemanticSeedsRepository, "replace_seeds", commit_then_fail)
    result = cast(Any, client.app).state.semantic_service.confirm_join_and_sink_relation(
        RUN,
        label=label,
        left=f"{ORDERS}.customer_id",
        right=f"{CUSTOMERS}.customer_id",
        cardinality="many_to_one",
        source_session_id=RUN,
        expected_version=0,
    )

    assert result.status == "confirmed"
    project_dir = ArtifactStore(workspace).project_dir(PROJECT)
    entry = load_join_whitelist(project_dir).entry(label)
    assert entry is not None and entry.status == "confirmed"
    relations = load_seeds(project_dir).verified_relations
    assert [(item.left, item.right) for item in relations] == [
        (f"{ORDERS}.customer_id", f"{CUSTOMERS}.customer_id")
    ]


def test_join_confirm_requires_validation(client: TestClient) -> None:
    response = _confirm(client, f"{ORDERS}.guess_id -> {CUSTOMERS}.customer_id")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "join_state_invalid"


def test_join_confirm_unknown_label_is_404(client: TestClient) -> None:
    response = _confirm(client, "nope.a -> nope.b")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "join_not_found"


def test_join_revoke_auto_confirmed_is_idempotent(client: TestClient) -> None:
    label = f"{ORDERS}.cust_ref -> {CUSTOMERS}.customer_id"
    first = _revoke(client, label)
    assert first.status_code == 200, first.text
    assert first.json()["status"] == "proposed"
    replay = _revoke(client, label)
    assert replay.status_code == 200
    assert replay.json()["status"] == "proposed"


def test_join_revoke_never_downgrades_user_confirmed(client: TestClient) -> None:
    response = _revoke(client, f"{ORDERS}.account_id -> {CUSTOMERS}.customer_id")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "join_state_invalid"


def _accept(client: TestClient, dataset: str, column: str, **extra):
    if "expected_version" not in extra:
        extra["expected_version"] = _view(client)["seeds_version"]
    return client.post(
        f"/api/v1/sessions/{RUN}/semantic/proposals/accept",
        json={"dataset": dataset, "column": column, **extra},
    )


def test_proposal_accept_writes_seed_and_is_idempotent(client: TestClient) -> None:
    first = _accept(client, CUSTOMERS, "amount")
    assert first.status_code == 200, first.text
    assert first.json()["status"] == "accepted"
    version_after_accept = first.json()["seeds_version"]
    assert version_after_accept == 1

    view = _view(client)
    meanings = {
        (item["dataset"], item["column"]): item["meaning"]
        for item in view["field_meanings"]
    }
    assert meanings[(CUSTOMERS, "amount")] == "Customer lifetime value guess."
    assert (CUSTOMERS, "amount") not in {
        (p["dataset"], p["column"]) for p in view["proposals"]
    }

    replay = _accept(client, CUSTOMERS, "amount")
    assert replay.status_code == 200
    assert replay.json()["status"] == "accepted"
    # An untouched-seed replay is idempotent and does not bump the version.
    assert replay.json()["seeds_version"] == version_after_accept


def test_proposal_reject_and_retraction(client: TestClient) -> None:
    accepted = _accept(client, CUSTOMERS, "amount")
    assert accepted.status_code == 200
    rejected = client.post(
        f"/api/v1/sessions/{RUN}/semantic/proposals/reject",
        json={
            "dataset": CUSTOMERS,
            "column": "amount",
            "expected_version": accepted.json()["seeds_version"],
        },
    )
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()["status"] == "rejected"
    # Rejecting an accepted proposal retracts its seed and bumps the version.
    assert rejected.json()["seeds_version"] == accepted.json()["seeds_version"] + 1
    view = _view(client)
    assert (CUSTOMERS, "amount") not in {
        (item["dataset"], item["column"]) for item in view["field_meanings"]
    }


def test_proposal_unknown_is_404(client: TestClient) -> None:
    response = _accept(client, "nope.csv", "nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "proposal_not_found"


def test_unknown_run_is_404(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/no_such_run/semantic")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_proposal_composite_commit_then_response_failure_is_idempotent(
    client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_replace = SemanticSeedsRepository.replace_state
    calls = 0

    def commit_then_fail(
        repository: SemanticSeedsRepository, **kwargs: object
    ):
        nonlocal calls
        snapshot = original_replace(repository, **kwargs)  # type: ignore[arg-type]
        calls += 1
        if calls == 1:
            raise OSError("response path failed")
        return snapshot

    monkeypatch.setattr(SemanticSeedsRepository, "replace_state", commit_then_fail)
    with pytest.raises(OSError, match="response path failed"):
        _accept(client, CUSTOMERS, "amount")

    store = ArtifactStore(workspace)
    committed = SemanticSeedsRepository(store, PROJECT).read()
    assert committed.version == 1
    proposal = committed.proposals.find(CUSTOMERS, "amount")
    assert proposal is not None and proposal.status == "accepted"

    retry = _accept(client, CUSTOMERS, "amount")
    assert retry.status_code == 200
    assert retry.json()["seeds_version"] == 1
    assert SemanticSeedsRepository(store, PROJECT).read().version == 1


def test_proposal_seed_failure_leaves_status_pending_and_retryable(
    client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_replace = SemanticSeedsRepository.replace_state

    def fail_replace(*args: object, **kwargs: object) -> None:
        raise OSError("seeds disk full")

    monkeypatch.setattr(SemanticSeedsRepository, "replace_state", fail_replace)
    with pytest.raises(OSError, match="seeds disk full"):
        _accept(client, CUSTOMERS, "amount")

    store = ArtifactStore(workspace)
    proposal = SemanticSeedsRepository(store, PROJECT).read().proposals.find(
        CUSTOMERS, "amount"
    )
    assert proposal is not None and proposal.status == "proposed"

    monkeypatch.setattr(SemanticSeedsRepository, "replace_state", original_replace)
    retry = _accept(client, CUSTOMERS, "amount")
    assert retry.status_code == 200
    assert retry.json()["seeds_version"] == 1


# --------------------------------------------------------------------------- #
# H2 — accepting a proposal never rolls back a hand-edited seed
# --------------------------------------------------------------------------- #
def test_unjournaled_hand_edit_after_accept_fails_closed(
    client: TestClient, workspace: Path
) -> None:
    accepted = _accept(client, CUSTOMERS, "amount")
    assert accepted.status_code == 200
    project_dir = ArtifactStore(workspace).project_dir(PROJECT)
    seeds = load_seeds(project_dir)
    edited = next(
        field
        for field in seeds.field_meanings
        if (field.dataset, field.column) == (CUSTOMERS, "amount")
    )
    edited.meaning = "Hand-tuned by the analyst."
    save_seeds(project_dir, seeds)

    replay = _accept(
        client,
        CUSTOMERS,
        "amount",
        expected_version=accepted.json()["seeds_version"],
    )
    assert replay.status_code == 500
    assert replay.json()["error"]["code"] == "semantic_state_corrupt"
    # The hand edit survives untouched.
    fields = {
        (field.dataset, field.column): field.meaning
        for field in load_seeds(project_dir).field_meanings
    }
    assert fields[(CUSTOMERS, "amount")] == "Hand-tuned by the analyst."


def test_proposal_accept_blocked_when_existing_seed_differs(client: TestClient) -> None:
    # The fixture seed for orders.amount ("Order value before refunds.") is the
    # user's own definition; accepting the machine draft must not clobber it.
    response = _accept(client, ORDERS, "amount")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "proposal_conflict"
    view = _view(client)
    meanings = {
        (item["dataset"], item["column"]): item["meaning"]
        for item in view["field_meanings"]
    }
    assert meanings[(ORDERS, "amount")] == "Order value before refunds."


def test_proposal_accept_of_rejected_proposal_is_409(client: TestClient) -> None:
    # customers.amount has no pre-existing seed, so the 409 can only come from
    # the explicit-rejection guard.
    rejected = client.post(
        f"/api/v1/sessions/{RUN}/semantic/proposals/reject",
        json={
            "dataset": CUSTOMERS,
            "column": "amount",
            "expected_version": _view(client)["seeds_version"],
        },
    )
    assert rejected.status_code == 200
    response = _accept(client, CUSTOMERS, "amount")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "proposal_conflict"


# --------------------------------------------------------------------------- #
# H5 — PUT body validation
# --------------------------------------------------------------------------- #
def test_seeds_update_rejects_overlong_fields(client: TestClient) -> None:
    response = _put_seeds(
        client,
        0,
        [{"dataset": ORDERS, "column": "amount", "meaning": "x" * 501}],
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_seeds_update_rejects_duplicate_keys(client: TestClient) -> None:
    row = {"dataset": ORDERS, "column": "amount", "meaning": "First."}
    response = _put_seeds(client, 0, [row, {**row, "meaning": "Second."}])
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "seeds_invalid"
    assert f"{ORDERS}.amount" in body["error"]["message"]


def test_seeds_update_requires_field_meanings_key(client: TestClient) -> None:
    response = client.put(
        f"/api/v1/sessions/{RUN}/semantic/seeds", json={"expected_version": 0}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


# --------------------------------------------------------------------------- #
# H4 — repository fail-closed + commit ordering
# --------------------------------------------------------------------------- #
def _corrupt_sidecar(workspace: Path) -> None:
    path = ArtifactStore(workspace).project_dir(PROJECT) / "semantic" / "versions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")


def test_corrupt_sidecar_fails_closed_with_500(
    client: TestClient, workspace: Path
) -> None:
    _corrupt_sidecar(workspace)
    read = client.get(f"/api/v1/sessions/{RUN}/semantic")
    assert read.status_code == 500
    assert read.json()["error"]["code"] == "semantic_state_corrupt"
    write = _put_seeds(client, 0, [])
    assert write.status_code == 500
    assert write.json()["error"]["code"] == "semantic_state_corrupt"


def test_repository_crash_before_commit_leaves_version_retryable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_replace = SemanticSeedsRepository.replace_seeds

    def crashing_replace(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(SemanticSeedsRepository, "replace_seeds", crashing_replace)
    # TestClient re-raises unhandled server exceptions; the crash is the point.
    with pytest.raises(OSError, match="disk full"):
        _put_seeds(client, 0, [])

    monkeypatch.setattr(SemanticSeedsRepository, "replace_seeds", original_replace)
    # No generation committed, so the same expected version remains valid.
    retry = _put_seeds(client, 0, [])
    assert retry.status_code == 200
    assert retry.json()["version"] == 1


# --------------------------------------------------------------------------- #
# H3 — out-of-band writer detection during PUT
# --------------------------------------------------------------------------- #
def test_out_of_band_seed_write_during_put_is_409(
    client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = ArtifactStore(workspace).project_dir(PROJECT)
    original_replace = SemanticSeedsRepository.replace_seeds

    def replace_with_out_of_band_write(
        repository: SemanticSeedsRepository, **kwargs: object
    ):
        # Simulate a worker process replacing seeds.json inside our window.
        save_seeds(
            project_dir,
            SemanticSeeds(
                field_meanings=[
                    FieldMeaning(dataset=ORDERS, column="amount", meaning="Worker wrote this.")
                ]
            ),
        )
        return original_replace(repository, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        SemanticSeedsRepository, "replace_seeds", replace_with_out_of_band_write
    )
    response = _put_seeds(client, 0, [])
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "version_conflict"
    # The out-of-band write was not clobbered.
    assert load_seeds(project_dir).field_meanings[0].meaning == "Worker wrote this."


# --------------------------------------------------------------------------- #
# H9 — server-side confirm gate mirrors the UI gate
# --------------------------------------------------------------------------- #
def test_join_confirm_with_stale_validation_is_409(
    client: TestClient, workspace: Path
) -> None:
    project_dir = ArtifactStore(workspace).project_dir(PROJECT)
    save_join_whitelist(
        project_dir,
        JoinWhitelist(
            entries=[
                _join_entry(
                    ("customer_id", "customer_id"),
                    status="proposed",
                    validation_verified=True,
                    left_id="ds_orders_OLD",
                )
            ]
        ),
    )
    response = _confirm(client, f"{ORDERS}.customer_id -> {CUSTOMERS}.customer_id")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "join_not_confirmable"


def test_join_confirm_many_to_many_is_409(client: TestClient, workspace: Path) -> None:
    project_dir = ArtifactStore(workspace).project_dir(PROJECT)
    entry = _join_entry(
        ("customer_id", "customer_id"), status="proposed", validation_verified=True
    )
    entry.cardinality = "many_to_many"
    save_join_whitelist(project_dir, JoinWhitelist(entries=[entry]))
    response = _confirm(client, f"{ORDERS}.customer_id -> {CUSTOMERS}.customer_id")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "join_not_confirmable"


# --------------------------------------------------------------------------- #
# H1 — join review is serialized within the process
# --------------------------------------------------------------------------- #
def test_concurrent_confirms_of_different_labels_all_land(workspace: Path) -> None:
    store = ArtifactStore(workspace)
    project_dir = store.project_dir(PROJECT)
    labels = [f"col_{index}" for index in range(12)]
    save_join_whitelist(
        project_dir,
        JoinWhitelist(
            entries=[
                _join_entry((label, "customer_id"), status="proposed", validation_verified=True)
                for label in labels
            ]
        ),
    )
    service = SemanticService(store)
    barrier = threading.Barrier(len(labels))
    errors: list[Exception] = []

    def confirm(column: str) -> None:
        barrier.wait()
        try:
            service.confirm_whitelist_join(RUN, f"{ORDERS}.{column} -> {CUSTOMERS}.customer_id")
        except Exception as exc:  # noqa: BLE001 — collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=confirm, args=(label,)) for label in labels]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    statuses = {
        entry.label(): entry.status for entry in load_join_whitelist(project_dir).entries
    }
    assert len(statuses) == 12
    assert set(statuses.values()) == {"confirmed"}


# --------------------------------------------------------------------------- #
# Metric definitions / entity notes / verified answers — the three seed classes
# via the semantic API. They share seeds.json, so they share the
# version counter; a class the PUT omits must survive untouched.
# --------------------------------------------------------------------------- #
def _seeds(workspace: Path) -> SemanticSeeds:
    return load_seeds(ArtifactStore(workspace).project_dir(PROJECT))


def _put(client: TestClient, expected_version: int, **classes: object):
    return client.put(
        f"/api/v1/sessions/{RUN}/semantic/seeds",
        json={
            "expected_version": expected_version,
            "field_meanings": [],
            **classes,
        },
    )


def test_view_exposes_the_three_hand_edited_seed_classes(client: TestClient) -> None:
    body = _view(client)
    assert body["metric_definitions"] == [
        {
            "name": "Active user",
            "definition": "A user with at least one session in 28 days.",
            "formula": "count(distinct user_id)",
            "caveats": None,
        }
    ]
    assert body["entity_notes"] == [
        {"name": "customer", "note": "One row per billing account."}
    ]
    assert body["verified_answers"] == [
        {
            "question": "What was Q3 revenue?",
            "answer": "$4.2M.",
            "evidence_note": None,
            "verified_at": SEEDED_VERIFIED_AT.isoformat().replace("+00:00", "Z"),
        }
    ]


def test_metric_entity_and_answer_round_trip_through_one_put(
    client: TestClient, workspace: Path
) -> None:
    response = _put(
        client,
        0,
        metric_definitions=[
            {
                "name": "Active user",
                "definition": "Two sessions in 28 days.",
                "formula": "count(distinct user_id)",
                "caveats": "Excludes test accounts.",
            },
            {"name": "Churn", "definition": "No session in 90 days."},
        ],
        entity_notes=[{"name": "customer", "note": "One row per person."}],
        verified_answers=[
            {
                "question": "What was Q3 revenue?",
                "answer": "$4.4M, restated.",
                "evidence_note": "Audited close.",
                "verified_at": SEEDED_VERIFIED_AT.isoformat(),
            }
        ],
    )
    assert response.status_code == 200, response.text
    assert response.json()["version"] == 1
    body = response.json()
    assert [metric["name"] for metric in body["metric_definitions"]] == [
        "Active user",
        "Churn",
    ]

    seeds = _seeds(workspace)
    assert seeds.metric_definitions[0].caveats == "Excludes test accounts."
    assert seeds.metric_definitions[1].formula is None
    assert seeds.entity_notes == [EntityNote(name="customer", note="One row per person.")]
    assert seeds.verified_answers[0].answer == "$4.4M, restated."
    # An edit round-trips the original verification date instead of restamping.
    assert seeds.verified_answers[0].verified_at == SEEDED_VERIFIED_AT


def test_a_new_verified_answer_is_stamped_by_the_server(
    client: TestClient, workspace: Path
) -> None:
    before = datetime.now(UTC)
    response = _put(
        client,
        0,
        verified_answers=[{"question": "Top region?", "answer": "EMEA."}],
    )
    assert response.status_code == 200, response.text
    stamped = _seeds(workspace).verified_answers[0].verified_at
    assert before <= stamped <= datetime.now(UTC)


def test_omitting_a_class_leaves_it_untouched(
    client: TestClient, workspace: Path
) -> None:
    """The 'omitted means clear' footgun, for the three new classes."""
    response = _put(client, 0, entity_notes=[{"name": "order", "note": "One line."}])
    assert response.status_code == 200, response.text
    seeds = _seeds(workspace)
    assert [note.name for note in seeds.entity_notes] == ["order"]
    # Untouched by a PUT that never mentioned them.
    assert [metric.name for metric in seeds.metric_definitions] == ["Active user"]
    assert [answer.question for answer in seeds.verified_answers] == [
        "What was Q3 revenue?"
    ]
    # And an explicit null is the same as omitting.
    assert _put(client, 1, metric_definitions=None).status_code == 200
    assert [metric.name for metric in _seeds(workspace).metric_definitions] == [
        "Active user"
    ]


def test_an_empty_list_clears_a_class_deliberately(
    client: TestClient, workspace: Path
) -> None:
    assert _put(client, 0, metric_definitions=[]).status_code == 200
    seeds = _seeds(workspace)
    assert seeds.metric_definitions == []
    assert [note.name for note in seeds.entity_notes] == ["customer"]


def test_the_three_classes_share_the_field_meaning_version(client: TestClient) -> None:
    assert _put(client, 0, entity_notes=[]).status_code == 200
    stale = _put_seeds(client, 0, [])
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "version_conflict"
    assert _view(client)["seeds_version"] == 1


def test_seed_classes_never_touch_relations_or_column_roles(
    client: TestClient, workspace: Path
) -> None:
    project_dir = ArtifactStore(workspace).project_dir(PROJECT)
    seeds = load_seeds(project_dir)
    seeds.column_role_seeds = [
        ColumnRoleSeed(dataset=ORDERS, column="amount", role="measure")
    ]
    seeds.verified_relations = [
        VerifiedRelation(
            left=f"{ORDERS}.customer_id",
            right=f"{CUSTOMERS}.customer_id",
            cardinality="many_to_one",
        )
    ]
    save_seeds(project_dir, seeds)
    assert _put(client, 0, metric_definitions=[], entity_notes=[]).status_code == 200
    after = load_seeds(project_dir)
    assert len(after.column_role_seeds) == 1
    assert len(after.verified_relations) == 1


@pytest.mark.parametrize(
    ("payload", "message_fragment"),
    [
        ({"metric_definitions": [{"name": "  ", "definition": "x"}]}, "metric definition"),
        ({"metric_definitions": [{"name": "m", "definition": " "}]}, "metric definition"),
        ({"entity_notes": [{"name": " ", "note": "x"}]}, "entity note"),
        ({"entity_notes": [{"name": "e", "note": "  "}]}, "entity note"),
        ({"verified_answers": [{"question": " ", "answer": "x"}]}, "verified answer"),
        ({"verified_answers": [{"question": "q", "answer": " "}]}, "verified answer"),
    ],
)
def test_blank_required_fields_are_422(
    client: TestClient, payload: dict, message_fragment: str
) -> None:
    response = _put(client, 0, **payload)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "semantic_invalid"
    assert message_fragment in response.json()["error"]["message"]


@pytest.mark.parametrize(
    ("payload", "label"),
    [
        (
            {
                "metric_definitions": [
                    {"name": "Active user", "definition": "a"},
                    {"name": " Active user ", "definition": "b"},
                ]
            },
            "metric definition names",
        ),
        (
            {"entity_notes": [{"name": "c", "note": "a"}, {"name": "c", "note": "b"}]},
            "entity note names",
        ),
        (
            {
                "verified_answers": [
                    {"question": "q", "answer": "a"},
                    {"question": "q", "answer": "b"},
                ]
            },
            "verified answer questions",
        ),
    ],
)
def test_duplicate_keys_are_422(client: TestClient, payload: dict, label: str) -> None:
    response = _put(client, 0, **payload)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "seeds_invalid"
    assert label in response.json()["error"]["message"]


def test_oversized_seed_rows_are_rejected_before_the_lock(client: TestClient) -> None:
    response = _put(
        client, 0, metric_definitions=[{"name": "x" * 201, "definition": "y"}]
    )
    assert response.status_code == 422
    # Pydantic's own length guard, not the service's — the payload never lands.
    assert response.json()["error"]["code"] == "validation_error"


def test_a_rejected_payload_never_bumps_the_version(
    client: TestClient, workspace: Path
) -> None:
    assert _put(client, 0, entity_notes=[{"name": " ", "note": "x"}]).status_code == 422
    assert _view(client)["seeds_version"] == 0
    assert [note.name for note in _seeds(workspace).entity_notes] == ["customer"]


def _seed_relations(workspace: Path, *pairs: tuple[str, str]) -> None:
    project_dir = ArtifactStore(workspace).project_dir(PROJECT)
    seeds = load_seeds(project_dir)
    seeds.verified_relations = [
        VerifiedRelation(left=left, right=right, cardinality="many_to_one")
        for left, right in pairs
    ]
    save_seeds(project_dir, seeds)


def _delete_relation(
    client: TestClient,
    left: str,
    right: str,
    session_id: str = RUN,
    *,
    expected_version: int | None = None,
):
    version = (
        _view(client)["seeds_version"]
        if expected_version is None
        else expected_version
    )
    return client.post(
        f"/api/v1/sessions/{session_id}/semantic/verified-relations/delete",
        json={"left": left, "right": right, "expected_version": version},
    )


def test_view_exposes_verified_relations(client: TestClient, workspace: Path) -> None:
    _seed_relations(workspace, (f"{ORDERS}.customer_id", f"{CUSTOMERS}.customer_id"))
    relations = _view(client)["verified_relations"]
    assert len(relations) == 1
    assert relations[0]["left"] == f"{ORDERS}.customer_id"
    assert relations[0]["right"] == f"{CUSTOMERS}.customer_id"
    assert relations[0]["cardinality"] == "many_to_one"
    assert relations[0]["confirmed_by"] == "user"


def test_delete_removes_the_named_row_not_a_position(
    client: TestClient, workspace: Path
) -> None:
    """The middle row goes; its neighbours stay. A positional delete resolved
    against a stale render would take the wrong one."""
    _seed_relations(
        workspace,
        (f"{ORDERS}.a", f"{CUSTOMERS}.a"),
        (f"{ORDERS}.b", f"{CUSTOMERS}.b"),
        (f"{ORDERS}.c", f"{CUSTOMERS}.c"),
    )
    response = _delete_relation(client, f"{ORDERS}.b", f"{CUSTOMERS}.b")
    assert response.status_code == 200, response.text
    assert [
        (item["left"], item["right"]) for item in response.json()["verified_relations"]
    ] == [
        (f"{ORDERS}.a", f"{CUSTOMERS}.a"),
        (f"{ORDERS}.c", f"{CUSTOMERS}.c"),
    ]
    assert [relation.left for relation in _seeds(workspace).verified_relations] == [
        f"{ORDERS}.a",
        f"{ORDERS}.c",
    ]


def test_delete_bumps_the_seeds_version_once_and_replays_unchanged(
    client: TestClient, workspace: Path
) -> None:
    _seed_relations(workspace, (f"{ORDERS}.a", f"{CUSTOMERS}.a"))
    first = _delete_relation(client, f"{ORDERS}.a", f"{CUSTOMERS}.a")
    assert first.status_code == 200
    assert first.json()["seeds_version"] == 1

    # Idempotent: the row is already gone, so nothing changes and the version
    # does not tick again and 409 an unrelated in-flight seeds editor.
    replay = _delete_relation(client, f"{ORDERS}.a", f"{CUSTOMERS}.a")
    assert replay.status_code == 200, replay.text
    assert replay.json()["verified_relations"] == []
    assert replay.json()["seeds_version"] == 1
    assert _view(client)["seeds_version"] == 1


def test_delete_leaves_the_other_seed_classes_untouched(
    client: TestClient, workspace: Path
) -> None:
    _seed_relations(workspace, (f"{ORDERS}.a", f"{CUSTOMERS}.a"))
    assert _delete_relation(client, f"{ORDERS}.a", f"{CUSTOMERS}.a").status_code == 200
    after = _seeds(workspace)
    assert [field.column for field in after.field_meanings] == ["amount"]
    assert [metric.name for metric in after.metric_definitions] == ["Active user"]
    assert [note.name for note in after.entity_notes] == ["customer"]


def test_delete_unknown_run_is_404(client: TestClient) -> None:
    response = _delete_relation(client, "a.b", "c.d", session_id="run_missing")
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "session_not_found"


@pytest.mark.parametrize(
    "payload",
    [
        {"left": "", "right": "c.d"},
        {"left": "a.b", "right": ""},
        {"left": "a.b"},
        {},
    ],
)
def test_delete_requires_both_identity_keys(client: TestClient, payload: dict) -> None:
    response = client.post(
        f"/api/v1/sessions/{RUN}/semantic/verified-relations/delete", json=payload
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "validation_error"


def test_semantic_cursor_survives_an_artifact_written_mid_run(
    client: TestClient, workspace: Path
) -> None:
    """A run writing profiles must not invalidate an open page: the source
    version has to be stable while the run is producing artifacts."""
    first = client.get(
        f"/api/v1/sessions/{RUN}/semantic", params={"limit": 1}
    ).json()
    assert first["next_cursor"]
    ArtifactStore(workspace).save_artifact(
        Artifact(
            id="profile_ds_extra_1",
            type=ArtifactType.DATASET_PROFILE,
            project_id=PROJECT,
            session_id=RUN,
            payload=_profile("extra.csv", "ds_extra_1").model_dump(),
        )
    )
    second = client.get(
        f"/api/v1/sessions/{RUN}/semantic",
        params={"limit": 1, "cursor": first["next_cursor"]},
    )
    assert second.status_code == 200, second.text
