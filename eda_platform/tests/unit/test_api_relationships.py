"""Relationships API vertical slice: the graph is shaped from the run's own
RELATIONSHIP_* artifacts plus the project join whitelist (no recomputation),
prepare registers a server-side approval bound to the candidate's columns, and
validate consumes it once into a `relationship_validate` job whose driver
writes the validation back onto the SOURCE run. Confirm/revoke forward to the
semantic layer. Spawned-worker pattern mirrors test_api_skills."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from job_test_helpers import run_claimed_job
from semantic_test_helpers import load_seeds

from eda_platform.api.main import create_app
from eda_platform.application.ports import JobRef
from eda_platform.application.services.approval_service import ApprovalService
from eda_platform.application.services.relationship_service import (
    RelationshipService,
    generate_discover_session_id,
    generate_validate_session_id,
    relationship_id_for,
)
from eda_platform.core.llm import OfflineLLMClient
from eda_platform.core.semantic import load_join_whitelist
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import run_auto_eda
from eda_platform.schemas.artifacts import ArtifactType

_JOB_TIMEOUT_SECONDS = 180.0
_SOURCE_RUN = "run_rel_src"
# `run_auto_eda` defers discovery by default, so this run reaches the graph with
# no RELATIONSHIP_* artifacts at all — the state the discover endpoint exists for.
_DEFERRED_RUN = "run_rel_deferred"
_SINGLE_DATASET_RUN = "run_rel_single"
# The high-confidence, id-named pair is validated (and auto-confirmed) eagerly
# during the run; the region_code pair is not id-named, so it stays a bare
# candidate and is the one this slice validates on demand.
_DEFERRED_LABEL = "customers.csv.region_code -> regions.csv.region_code"
_AUTO_CONFIRMED_LABEL = "orders.csv.customer_id -> customers.csv.customer_id"


def _customers_csv() -> str:
    regions = ["north", "south", "east", "west"]
    rows = ["customer_id,region_code,city"]
    rows.extend(f"c{index:03d},{regions[index % 4]},city{index % 7}" for index in range(40))
    return "\n".join(rows) + "\n"


def _orders_csv() -> str:
    rows = ["order_id,customer_id,amount"]
    rows.extend(f"o{index:04d},c{index % 40:03d},{50 + index % 37}" for index in range(120))
    return "\n".join(rows) + "\n"


def _regions_csv() -> str:
    rows = ["region_code,region_name"]
    rows.extend(f"{name},{name.title()} Region" for name in ("north", "south", "east", "west"))
    return "\n".join(rows) + "\n"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("relationships_api")
    store = ArtifactStore(root)
    store.ensure_project("demo", name="Demo")
    seed = root / "seed"
    seed.mkdir(parents=True, exist_ok=True)
    (seed / "customers.csv").write_text(_customers_csv(), encoding="utf-8")
    (seed / "orders.csv").write_text(_orders_csv(), encoding="utf-8")
    (seed / "regions.csv").write_text(_regions_csv(), encoding="utf-8")
    # Eager discovery in-process: the generic job route runs the default
    # on_demand policy, which produces no relationship artifacts at all.
    run_auto_eda(
        [seed / "orders.csv", seed / "customers.csv", seed / "regions.csv"],
        workspace=root,
        project_id="demo",
        session_id=_SOURCE_RUN,
        llm=OfflineLLMClient(),
        relationship_discovery="eager",
        generate_report=False,
    )
    # Default (on_demand) policy: no relationship artifacts, which is what
    # every real run looks like when the Relationships page opens.
    run_auto_eda(
        [seed / "orders.csv", seed / "customers.csv", seed / "regions.csv"],
        workspace=root,
        project_id="demo",
        session_id=_DEFERRED_RUN,
        llm=OfflineLLMClient(),
        generate_report=False,
    )
    run_auto_eda(
        [seed / "orders.csv"],
        workspace=root,
        project_id="demo",
        session_id=_SINGLE_DATASET_RUN,
        llm=OfflineLLMClient(),
        generate_report=False,
    )
    return root


@pytest.fixture(scope="module")
def app(workspace: Path) -> FastAPI:
    return create_app(workspace)


@pytest.fixture(scope="module")
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _wait_terminal(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + _JOB_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/jobs/{job_id}").json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body
        time.sleep(0.2)
    raise AssertionError(f"Job {job_id} did not reach a terminal status in time.")


def _graph(client: TestClient, session_id: str = _SOURCE_RUN) -> dict:
    response = client.get(f"/api/v1/sessions/{session_id}/relationships")
    assert response.status_code == 200, response.text
    return response.json()


def _edge(client: TestClient, label: str, session_id: str = _SOURCE_RUN) -> dict:
    edges = {edge["label"]: edge for edge in _graph(client, session_id)["edges"]}
    assert label in edges, sorted(edges)
    return edges[label]


def _prepare(client: TestClient, relationship_id: str, session_id: str = _SOURCE_RUN):
    return client.post(
        f"/api/v1/sessions/{session_id}/relationships/{relationship_id}/prepare-validate"
    )


def _prepared_ok(client: TestClient, relationship_id: str) -> dict:
    response = _prepare(client, relationship_id)
    assert response.status_code == 200, response.text
    return response.json()


def _validate(
    client: TestClient,
    relationship_id: str,
    prepared: dict,
    idempotency_key: str | None = None,
    session_id: str = _SOURCE_RUN,
):
    headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
    return client.post(
        f"/api/v1/sessions/{session_id}/relationships/{relationship_id}/validate",
        json={
            "action_hash": prepared["action_hash"],
            "approval_token": prepared["approval_token"],
        },
        headers=headers,
    )


def test_graph_shapes_nodes_and_three_edge_states(client: TestClient) -> None:
    body = _graph(client)
    assert body["discovered"] is True
    assert {node["name"] for node in body["nodes"]} == {
        "orders.csv",
        "customers.csv",
        "regions.csv",
    }
    assert all(node["source_available"] for node in body["nodes"])

    edges = {edge["label"]: edge for edge in body["edges"]}
    deferred = edges[_DEFERRED_LABEL]
    assert deferred["state"] == "candidate"
    assert deferred["verified"] is False
    assert deferred["cardinality"] is None
    assert deferred["join_status"] == "proposed"
    assert deferred["can_validate"] is True
    assert deferred["can_confirm"] is False
    assert deferred["candidate_artifact_id"].startswith("relcand_")

    auto = edges[_AUTO_CONFIRMED_LABEL]
    assert auto["state"] == "confirmed"
    assert auto["join_status"] == "auto_confirmed"
    assert auto["verified"] is True
    assert auto["cardinality"] == "many_to_one"
    assert auto["orphan_rate_left"] == 0.0
    assert auto["freshness"] == "fresh"
    assert auto["can_revoke"] is True
    assert auto["validation_artifact_id"].startswith("relval_")


def test_graph_edge_ids_are_stable_over_the_pair_label(client: TestClient) -> None:
    edge = _edge(client, _DEFERRED_LABEL)
    assert edge["relationship_id"] == relationship_id_for(_DEFERRED_LABEL)


def test_graph_unknown_run_is_404(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/run_missing/relationships")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_prepare_unknown_relationship_is_404(client: TestClient) -> None:
    response = _prepare(client, "0" * 16)
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "relationship_not_found"


def test_prepare_describes_the_pair_and_never_uses_an_llm(client: TestClient) -> None:
    prepared = _prepared_ok(client, relationship_id_for(_DEFERRED_LABEL))
    assert prepared["label"] == _DEFERRED_LABEL
    assert prepared["left_dataset"] == "customers.csv"
    assert prepared["right_dataset"] == "regions.csv"
    assert prepared["uses_llm"] is False
    assert len(prepared["action_hash"]) == 64
    assert len(prepared["approval_token"]) == 32


def test_prepare_refuses_an_already_validated_relationship(client: TestClient) -> None:
    response = _prepare(client, relationship_id_for(_AUTO_CONFIRMED_LABEL))
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "relationship_not_validatable"


def test_low_confidence_edges_are_not_offered_for_validation(client: TestClient) -> None:
    """`validate_relationships` skips anything below medium, so a Validate
    button there would queue a job that can only fail."""
    low = next(
        edge for edge in _graph(client)["edges"] if edge["confidence"] == "low"
    )
    assert low["can_validate"] is False
    response = _prepare(client, low["relationship_id"])
    assert response.status_code == 409, response.text
    body = response.json()["error"]
    assert body["code"] == "relationship_not_validatable"
    assert "low" in body["message"]


def test_semantic_kind_approval_rejected_on_relationships(
    client: TestClient, workspace: Path
) -> None:
    """A different approval kind on the same run must read as not-found."""
    relationship_id = relationship_id_for(_DEFERRED_LABEL)
    digest, token, _expires = ApprovalService(ArtifactStore(workspace)).register(
        kind="question_execute",
        session_id=_SOURCE_RUN,
        project_id="demo",
        action={"type": "question_probe"},
        payload={"relationship_id": relationship_id},
    )
    response = client.post(
        f"/api/v1/sessions/{_SOURCE_RUN}/relationships/{relationship_id}/validate",
        json={"action_hash": digest, "approval_token": token},
    )
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "approval_not_found"


def test_validate_unknown_token_is_404(client: TestClient) -> None:
    relationship_id = relationship_id_for(_DEFERRED_LABEL)
    prepared = _prepared_ok(client, relationship_id)
    response = _validate(client, relationship_id, {**prepared, "approval_token": "f" * 32})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "approval_not_found"


def test_validate_expired_approval_is_410(
    app: FastAPI, client: TestClient, workspace: Path
) -> None:
    store = ArtifactStore(workspace)
    original = app.state.relationship_service
    app.state.relationship_service = RelationshipService(
        store,
        app.state.dataset_service,
        ApprovalService(store, ttl_seconds=-1),
        app.state.job_service,
        app.state.semantic_service,
    )
    try:
        relationship_id = relationship_id_for(_DEFERRED_LABEL)
        prepared = _prepared_ok(client, relationship_id)
        response = _validate(client, relationship_id, prepared)
        assert response.status_code == 410, response.text
        assert response.json()["error"]["code"] == "approval_expired"
    finally:
        app.state.relationship_service = original


def test_validate_approval_bound_to_another_relationship_is_422(
    app: FastAPI,
    client: TestClient,
    workspace: Path,
) -> None:
    """The path must not be able to swap the relationship an approval reviewed."""
    prepared = _prepared_ok(client, relationship_id_for(_DEFERRED_LABEL))
    other = relationship_id_for(_AUTO_CONFIRMED_LABEL)
    response = _validate(client, other, prepared)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "relationship_invalid"
    row = ArtifactStore(workspace).get_pending_action(
        prepared["action_hash"], session_id=_SOURCE_RUN
    )
    assert row is not None
    assert row["status"] == "pending"
    assert row["generation"] == prepared["approval_token"]

    backend = app.state.job_service._backend
    original = backend.enqueue
    backend.enqueue = lambda command: JobRef(job_id=command.job_id, pid=None)
    try:
        retried = _validate(
            client,
            relationship_id_for(_DEFERRED_LABEL),
            prepared,
        )
    finally:
        backend.enqueue = original
    assert retried.status_code == 201, retried.text
    ArtifactStore(workspace).mark_job_status(
        retried.json()["job"]["job_id"],
        "completed",
    )


def test_validate_idempotency_key_replays_same_job(app: FastAPI, client: TestClient) -> None:
    from eda_platform.application.ports import JobRef

    relationship_id = relationship_id_for(_DEFERRED_LABEL)
    prepared = _prepared_ok(client, relationship_id)
    backend = app.state.job_service._backend
    original = backend.enqueue
    backend.enqueue = lambda command: JobRef(job_id=command.job_id, pid=None)
    key = uuid.uuid4().hex
    try:
        first = _validate(client, relationship_id, prepared, key)
        assert first.status_code == 201, first.text
        replay = _validate(client, relationship_id, prepared, key)
    finally:
        backend.enqueue = original
        # The stub left a queued job on the run; free it so later tests can
        # prepare/validate again.
        app.state.job_service._store.mark_job_status(
            first.json()["job"]["job_id"], "cancelled"
        )
    assert replay.status_code == 201, replay.text
    assert replay.json()["job"]["job_id"] == first.json()["job"]["job_id"]
    assert replay.json()["execution_session_id"] == first.json()["execution_session_id"]


def test_a_second_validation_of_the_same_run_is_refused_while_one_is_active(
    app: FastAPI, client: TestClient, workspace: Path
) -> None:
    """Two validate jobs read-modify-write the same merged validation artifact
    from separate processes, so the second must be refused, not queued."""
    store = ArtifactStore(workspace)
    relationship_id = relationship_id_for(_DEFERRED_LABEL)
    other_id = relationship_id_for(_AUTO_CONFIRMED_LABEL)
    # A queued job on a derived run of THIS source run, as a real validate
    # would have created before its worker started.
    job = store.create_job(
        job_id=f"job_{uuid.uuid4().hex[:12]}",
        session_id=generate_validate_session_id(_SOURCE_RUN, other_id),
        project_id="demo",
        kind="relationship_validate",
        idempotency_key=None,
    )
    try:
        response = _prepare(client, relationship_id)
        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == "relationship_session_busy"
    finally:
        store.mark_job_status(str(job["job_id"]), "cancelled")
    # Cleared once it settles.
    assert _prepare(client, relationship_id).status_code == 200


def test_validate_then_confirm_full_chain(client: TestClient, workspace: Path) -> None:
    """The whole slice: validate runs as a job on a derived rvsess_* run, the
    validation artifact lands on the SOURCE run, and the join becomes
    confirmable — then confirmed, in the semantic layer too."""
    relationship_id = relationship_id_for(_DEFERRED_LABEL)
    prepared = _prepared_ok(client, relationship_id)
    response = _validate(client, relationship_id, prepared, uuid.uuid4().hex)
    assert response.status_code == 201, response.text
    started = response.json()
    assert started["execution_session_id"].startswith("rvsess_")
    assert started["job"]["session_id"] == started["execution_session_id"]

    final = _wait_terminal(client, started["job"]["job_id"])
    assert final["status"] == "completed", final
    assert final["kind"] == "relationship_validate"

    # The source run keeps its own status, and gained the evidence artifact.
    assert client.get(f"/api/v1/sessions/{_SOURCE_RUN}").json()["status"] == "completed"
    edge = _edge(client, _DEFERRED_LABEL)
    assert edge["state"] == "validated"
    assert edge["verified"] is True
    assert edge["cardinality"] == "many_to_one"
    assert edge["orphan_rate_left"] == 0.0
    assert "select" in edge["verification_sql"].lower()
    assert edge["validation_artifact_id"].startswith("relval_")
    assert edge["can_validate"] is False
    assert edge["can_confirm"] is True

    confirmed = client.post(
        f"/api/v1/sessions/{_SOURCE_RUN}/relationships/{relationship_id}/confirm",
        json={
            "expected_version": client.get(
                f"/api/v1/sessions/{_SOURCE_RUN}/relationships"
            ).json()["seeds_version"]
        },
    )
    assert confirmed.status_code == 200, confirmed.text
    body = confirmed.json()
    assert body["state"] == "confirmed"
    assert body["join_status"] == "confirmed"
    assert body["can_confirm"] is False

    # Semantic side is the same store, not a parallel one.
    entry = load_join_whitelist(ArtifactStore(workspace).project_dir("demo")).entry(
        _DEFERRED_LABEL
    )
    assert entry is not None and entry.status == "confirmed"
    semantic = client.get(f"/api/v1/sessions/{_SOURCE_RUN}/semantic").json()
    joins = {item["label"]: item for item in semantic["join_whitelist"]}
    assert joins[_DEFERRED_LABEL]["status"] == "confirmed"

    # Confirming also sinks a verified relation into the semantic seeds, the
    # class value_discovery and question_agent read as join context.
    seeds = load_seeds(ArtifactStore(workspace).project_dir("demo"))
    sunk = [
        relation
        for relation in seeds.verified_relations
        if relation.left == "customers.csv.region_code"
        and relation.right == "regions.csv.region_code"
    ]
    assert len(sunk) == 1, seeds.verified_relations
    assert sunk[0].cardinality == "many_to_one"
    assert sunk[0].source_session_id == _SOURCE_RUN
    assert sunk[0].confirmed_by == "user"

    view = client.get(f"/api/v1/sessions/{_SOURCE_RUN}/semantic").json()
    assert ("customers.csv.region_code", "regions.csv.region_code") in {
        (item["left"], item["right"]) for item in view["verified_relations"]
    }


def test_validate_replayed_token_is_409_consumed(client: TestClient) -> None:
    """Runs after the full chain: the token from that prepare is consumed, and
    a fresh prepare on the now-validated pair is refused instead."""
    relationship_id = relationship_id_for(_DEFERRED_LABEL)
    response = _prepare(client, relationship_id)
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "relationship_not_validatable"


def test_confirm_refuses_to_downgrade_a_user_confirmed_join(client: TestClient) -> None:
    response = client.post(
        f"/api/v1/sessions/{_SOURCE_RUN}/relationships/"
        f"{relationship_id_for(_DEFERRED_LABEL)}/revoke",
        json={
            "expected_version": client.get(
                f"/api/v1/sessions/{_SOURCE_RUN}/relationships"
            ).json()["seeds_version"]
        },
    )
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "join_state_invalid"


def test_revoke_auto_confirmed_join_drops_it_back_to_proposed(client: TestClient) -> None:
    relationship_id = relationship_id_for(_AUTO_CONFIRMED_LABEL)
    response = client.post(
        f"/api/v1/sessions/{_SOURCE_RUN}/relationships/{relationship_id}/revoke",
        json={
            "expected_version": client.get(
                f"/api/v1/sessions/{_SOURCE_RUN}/relationships"
            ).json()["seeds_version"]
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["join_status"] == "proposed"
    assert body["state"] == "validated"
    assert body["can_confirm"] is True


def test_runner_refuses_a_relationship_that_changed_since_approval(workspace: Path) -> None:
    """Defence in depth: the worker recomputes the fingerprint before touching
    any data, and only the derived run is marked failed."""
    store = ArtifactStore(workspace)
    derived_session_id = f"rvsess_probe_{uuid.uuid4().hex[:6]}"
    job = store.create_job(
        job_id=f"job_{uuid.uuid4().hex[:12]}",
        session_id=derived_session_id,
        project_id="demo",
        kind="relationship_validate",
        idempotency_key=None,
    )
    params = {
        "source_session_id": _SOURCE_RUN,
        "relationship_id": relationship_id_for(_DEFERRED_LABEL),
        "pair_label": _DEFERRED_LABEL,
        "candidate_fingerprint": "stale" * 12,
    }
    run_claimed_job(workspace, str(job["job_id"]), json.dumps(params))
    final = store.get_job(str(job["job_id"]))
    assert final is not None
    assert final["status"] == "failed"
    assert "relationship source changed since approval" in str(final["error_message"])
    assert store.get_session_status(_SOURCE_RUN) == "completed"


def test_generic_jobs_route_rejects_relationship_validate_kind(client: TestClient) -> None:
    response = client.post(
        f"/api/v1/sessions/{_SOURCE_RUN}/jobs",
        json={"kind": "relationship_validate", "datasets": ["seed/orders.csv"]},
    )
    assert response.status_code == 422


def _discover(client: TestClient, session_id: str, idempotency_key: str | None = None):
    headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
    return client.post(f"/api/v1/sessions/{session_id}/relationships/discover", headers=headers)


def _ensure_discovered(client: TestClient, session_id: str) -> None:
    """Precondition helper: keeps the rescan test standalone instead of relying
    on an earlier test in this module having run first."""
    if _graph(client, session_id)["discovered"]:
        return
    response = _discover(client, session_id, uuid.uuid4().hex)
    assert response.status_code == 201, response.text
    final = _wait_terminal(client, response.json()["job"]["job_id"])
    assert final["status"] == "completed", final


def test_a_deferred_run_reports_no_discovery_but_still_lists_its_datasets(
    client: TestClient,
) -> None:
    body = _graph(client, _DEFERRED_RUN)
    assert body["discovered"] is False
    assert body["edges"] == []
    assert {node["name"] for node in body["nodes"]} == {
        "orders.csv",
        "customers.csv",
        "regions.csv",
    }


def test_discover_unknown_run_is_404(client: TestClient) -> None:
    response = _discover(client, "run_missing")
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "session_not_found"


def test_discover_needs_two_datasets(client: TestClient) -> None:
    response = _discover(client, _SINGLE_DATASET_RUN)
    assert response.status_code == 409, response.text
    body = response.json()["error"]
    assert body["code"] == "relationship_not_discoverable"
    assert "at least two datasets" in body["message"]


def test_discover_runs_as_a_job_and_populates_the_source_run_graph(
    client: TestClient,
) -> None:
    """The whole gap-1 slice: a run that deferred discovery gets its candidate
    set from the Relationships page, on a derived rdsess_* run."""
    response = _discover(client, _DEFERRED_RUN, uuid.uuid4().hex)
    assert response.status_code == 201, response.text
    started = response.json()
    assert started["session_id"] == _DEFERRED_RUN
    assert started["execution_session_id"].startswith("rdsess_")
    assert started["job"]["session_id"] == started["execution_session_id"]

    final = _wait_terminal(client, started["job"]["job_id"])
    assert final["status"] == "completed", final
    assert final["kind"] == "relationship_discover"

    body = _graph(client, _DEFERRED_RUN)
    assert body["discovered"] is True
    labels = {edge["label"] for edge in body["edges"]}
    assert _AUTO_CONFIRMED_LABEL in labels, sorted(labels)
    assert all(
        edge["candidate_artifact_id"].startswith("relcand_") for edge in body["edges"]
    )
    # The evidence landed on the SOURCE run, and its status is untouched.
    assert client.get(f"/api/v1/sessions/{_DEFERRED_RUN}").json()["status"] == "completed"
    artifacts = client.get(
        f"/api/v1/sessions/{_DEFERRED_RUN}/artifacts", params={"type": "RelationshipCandidateSet"}
    ).json()
    assert [item["artifact_id"] for item in artifacts["items"]] == [
        body["edges"][0]["candidate_artifact_id"]
    ]


def test_the_discovery_run_is_derived_and_stays_out_of_session_history(
    client: TestClient,
) -> None:
    runs = client.get("/api/v1/projects/demo/sessions").json()
    assert not any(item["session_id"].startswith("rdsess_") for item in runs["items"])


def _scan_count(client: TestClient, session_id: str) -> int:
    """How many times the discovery pipeline actually ran on this run. The
    driver only emits this event when it rescans; an early return emits none,
    which is exactly what separates a real re-run from a no-op."""
    page = client.get(
        f"/api/v1/sessions/{session_id}/trace",
        params={"type": "relationship_discovery_bounded", "limit": 100},
    )
    assert page.status_code == 200, page.text
    return len(page.json()["items"])


def test_rediscovery_rescans_instead_of_returning_the_existing_set(
    client: TestClient, workspace: Path
) -> None:
    """Re-run discovery must really rescan: the driver's on-demand entry
    returns the existing candidate set untouched unless it is hidden from it."""
    _ensure_discovered(client, _DEFERRED_RUN)
    before_scans = _scan_count(client, _DEFERRED_RUN)
    assert before_scans >= 1

    response = _discover(client, _DEFERRED_RUN, uuid.uuid4().hex)
    assert response.status_code == 201, response.text
    started = response.json()
    final = _wait_terminal(client, started["job"]["job_id"])
    assert final["status"] == "completed", final

    assert _scan_count(client, _DEFERRED_RUN) == before_scans + 1
    # A rescan of unchanged data is content-identical, so it must not multiply
    # the candidate artifacts (the id is derived from the payload).
    store = ArtifactStore(workspace)
    assert (
        len(
            store.list_indexed_artifacts(
                project_id="demo",
                session_id=_DEFERRED_RUN,
                artifact_types=(ArtifactType.RELATIONSHIP_CANDIDATE_SET,),
            )
        )
        == 1
    )
    assert _graph(client, _DEFERRED_RUN)["discovered"] is True


def test_discover_idempotency_key_replays_same_job(
    app: FastAPI, client: TestClient
) -> None:
    from eda_platform.application.ports import JobRef

    backend = app.state.job_service._backend
    original = backend.enqueue
    backend.enqueue = lambda command: JobRef(job_id=command.job_id, pid=None)
    key = uuid.uuid4().hex
    try:
        first = _discover(client, _DEFERRED_RUN, key)
        assert first.status_code == 201, first.text
        replay = _discover(client, _DEFERRED_RUN, key)
    finally:
        backend.enqueue = original
        app.state.job_service._store.mark_job_status(
            first.json()["job"]["job_id"], "cancelled"
        )
    assert replay.status_code == 201, replay.text
    assert replay.json()["job"]["job_id"] == first.json()["job"]["job_id"]
    assert replay.json()["execution_session_id"] == first.json()["execution_session_id"]


def test_discover_and_validate_share_one_lane_per_source_run(
    app: FastAPI, client: TestClient, workspace: Path
) -> None:
    """Discovery can rewrite the candidate set a validation was approved
    against, so the two kinds must not overlap on the same source run."""
    store = ArtifactStore(workspace)
    discover_job = store.create_job(
        job_id=f"job_{uuid.uuid4().hex[:12]}",
        session_id=generate_discover_session_id(_DEFERRED_RUN),
        project_id="demo",
        kind="relationship_discover",
        idempotency_key=None,
    )
    try:
        busy = _discover(client, _DEFERRED_RUN)
        assert busy.status_code == 409, busy.text
        assert busy.json()["error"]["code"] == "relationship_session_busy"
        edge = _edge(client, _AUTO_CONFIRMED_LABEL, session_id=_DEFERRED_RUN)
        blocked = _prepare(client, edge["relationship_id"], session_id=_DEFERRED_RUN)
        assert blocked.status_code == 409, blocked.text
        assert blocked.json()["error"]["code"] == "relationship_session_busy"
    finally:
        store.mark_job_status(str(discover_job["job_id"]), "cancelled")

    validate_job = store.create_job(
        job_id=f"job_{uuid.uuid4().hex[:12]}",
        session_id=generate_validate_session_id(
            _DEFERRED_RUN, relationship_id_for(_DEFERRED_LABEL)
        ),
        project_id="demo",
        kind="relationship_validate",
        idempotency_key=None,
    )
    try:
        busy = _discover(client, _DEFERRED_RUN)
        assert busy.status_code == 409, busy.text
        assert busy.json()["error"]["code"] == "relationship_session_busy"
    finally:
        store.mark_job_status(str(validate_job["job_id"]), "cancelled")
    assert _discover(client, _DEFERRED_RUN).status_code == 201


def test_generic_jobs_route_rejects_relationship_discover_kind(client: TestClient) -> None:
    response = client.post(
        f"/api/v1/sessions/{_SOURCE_RUN}/jobs",
        json={"kind": "relationship_discover", "datasets": ["seed/orders.csv"]},
    )
    assert response.status_code == 422
