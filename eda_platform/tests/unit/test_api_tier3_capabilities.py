"""Tier-3 capabilities exposed over HTTP:
pre-cleaning before Auto EDA, promoting a finding into semantic knowledge,
generating a report on demand, and forking a what-if variant.

Every job here runs through the real spawned worker (same pattern as
test_api_relationships), offline LLM only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from data_operation_helpers import await_data_operation
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.application.services.semantic_service import SemanticService
from eda_platform.core.semantic_resources import SemanticSeedsRepository
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    DatasetProfile,
    EvidenceRef,
)
from eda_platform.schemas.investigations import (
    InvestigationPlan,
    InvestigationRecord,
    ValidatedFinding,
)
from eda_platform.schemas.questions import QuestionFinding

PROJECT = "tier3"
RUN_ID = "run_tier3"
FINDING_ID = "tier3_finding"
EVIDENCE_ID = "tier3_evidence_profile"
QUESTION = "What is the average amount?"
ANSWER = "Average amount was 14."
_JOB_TIMEOUT_SECONDS = 600.0


def _sample_csv() -> str:
    """Wide enough to survive the guards, dirty enough that pre-cleaning bites:
    `note` is 90% empty, `region` has a few gaps, `amount` has two outliers."""
    rows = ["id,region,amount,note"]
    for index in range(60):
        region = "" if index % 17 == 0 else f"r{index % 4}"
        amount = 5000 if index in {7, 23} else 10 + index % 9
        note = "seen" if index % 10 == 0 else ""
        rows.append(f"{index},{region},{amount},{note}")
    return "\n".join(rows) + "\n"


def _wait_terminal(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + _JOB_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/jobs/{job_id}").json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body
        time.sleep(0.2)
    raise AssertionError(f"Job {job_id} did not reach a terminal status in time.")


def _artifacts(store: ArtifactStore, session_id: str, kind: ArtifactType) -> list:
    return store.list_indexed_artifacts(
        project_id=PROJECT, session_id=session_id, artifact_types=(kind,)
    )


def _seed_promotable_finding(store: ArtifactStore, name: str, dataset_id: str) -> None:
    """A fresh, promotable finding: plan + finding + record, all pointing at a
    profile whose dataset_id is still the project's current upload."""
    store.save_artifact(
        Artifact(
            id=EVIDENCE_ID,
            type=ArtifactType.DATASET_PROFILE,
            project_id=PROJECT,
            session_id=RUN_ID,
            payload=DatasetProfile(
                dataset_id=dataset_id,
                name=name,
                rows=54,
                columns=3,
                column_names=["id", "region", "amount"],
                dtypes={"amount": "int64"},
                missing_values={"amount": 0},
                missing_percent={"amount": 0.0},
                numeric_columns=["amount"],
                categorical_columns=["region"],
            ).model_dump(),
        )
    )
    store.save_artifact(
        Artifact(
            id="tier3_plan",
            type=ArtifactType.INVESTIGATION_PLAN,
            project_id=PROJECT,
            session_id=RUN_ID,
            payload=InvestigationPlan(
                investigation_id="inv_tier3",
                source_session_id=RUN_ID,
                question_id="q_tier3",
                card_version=1,
                candidate_fingerprint="fp",
                question=QUESTION,
                target_datasets=[name],
                method_family="descriptive",
                method_recipe="mean amount",
                allowed_tools=["sql"],
                feasibility="ready",
                status="planned",
                status_reason="ready",
            ).model_dump(),
        )
    )
    store.save_artifact(
        Artifact(
            id=FINDING_ID,
            type=ArtifactType.VALIDATED_FINDING,
            project_id=PROJECT,
            session_id=RUN_ID,
            payload=ValidatedFinding(
                finding_id=FINDING_ID,
                investigation_id="inv_tier3",
                question_id="q_tier3",
                question=QUESTION,
                value_hypothesis="Could steer pricing.",
                claim_class="observed",
                findings=[
                    QuestionFinding(
                        text=ANSWER,
                        evidence=[
                            EvidenceRef(
                                kind="profile_field",
                                artifact_id=EVIDENCE_ID,
                                locator="numeric.amount.mean",
                                value=14,
                            )
                        ],
                    )
                ],
                evidence_support="high",
                analytical_reliability="high",
                decision_readiness="medium",
                limitations=["Single table."],
                report_eligible=True,
                report_readiness="eligible_with_limitations",
                report_readiness_reason="Descriptive claim is supported.",
                source_artifact_ids=[EVIDENCE_ID],
            ).model_dump(),
        )
    )
    store.save_artifact(
        Artifact(
            id="tier3_record",
            type=ArtifactType.INVESTIGATION_RECORD,
            project_id=PROJECT,
            session_id=RUN_ID,
            payload=InvestigationRecord(
                record_id="tier3_record",
                investigation_id="inv_tier3",
                question_id="q_tier3",
                status="validated",
                reason_code="evidence_complete",
                reason="Claim is backed by the profile.",
                next_action="Promote if still fresh.",
                finding_artifact_id=FINDING_ID,
            ).model_dump(),
        )
    )


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("tier3_api")


@pytest.fixture(scope="module")
def client(workspace: Path) -> TestClient:
    return TestClient(create_app(workspace))


@pytest.fixture(scope="module")
def store(workspace: Path) -> ArtifactStore:
    return ArtifactStore(workspace)


@pytest.fixture(scope="module")
def precleaned_run(client: TestClient, store: ArtifactStore) -> dict:
    """One real Auto EDA job started WITH pre-cleaning; every other test in this
    module reads or derives from the run it produced."""
    assert client.post("/api/v1/projects", json={"project_id": PROJECT}).status_code in (
        200,
        201,
    )
    upload = client.post(
        f"/api/v1/projects/{PROJECT}/uploads",
        files={"file": ("sales.csv", _sample_csv(), "text/csv")},
    )
    assert upload.status_code in (200, 201), upload.text
    raw_dataset_id = upload.json()["dataset"]["dataset_id"]

    created = client.post(
        f"/api/v1/sessions/{RUN_ID}/jobs",
        json={
            "kind": "auto_eda",
            "project_id": PROJECT,
            "datasets": [raw_dataset_id],
            "generate_report": False,
            "llm": "offline",
            "precleaning": {
                "clean_missing_values": True,
                "missing_threshold_percent": 70,
                "min_rows_keep_percent": 50,
                "drop_iqr_outliers": True,
            },
        },
        headers={"Idempotency-Key": "tier3-preclean"},
    )
    assert created.status_code == 201, created.text
    job = _wait_terminal(client, created.json()["job_id"])
    assert job["status"] == "completed", job

    profiles = _artifacts(store, RUN_ID, ArtifactType.DATASET_PROFILE)
    return {
        "raw_dataset_id": raw_dataset_id,
        "cleaned_dataset_id": profiles[0].payload["dataset_id"],
        "cleaned_name": profiles[0].payload["name"],
    }


# --- pre-cleaning -----------------------------------------------------------


def test_precleaning_ingests_a_new_version_and_records_the_recipe(
    precleaned_run: dict, store: ArtifactStore, workspace: Path
) -> None:
    recipes = _artifacts(store, RUN_ID, ArtifactType.CLEANING_RECIPE)
    assert recipes, "pre-cleaning must leave a CleaningRecipe as evidence"
    recipe = recipes[0].payload
    assert {transform["type"] for transform in recipe["transforms"]} == {
        "drop_column",
        "drop_missing_rows",
        "drop_outlier_rows",
    }
    lineage = recipe["lineage"]
    assert lineage["source_dataset_id"] == precleaned_run["raw_dataset_id"]
    assert lineage["rows_after"] < lineage["rows_before"]
    assert lineage["columns_after"] < lineage["columns_before"]

    # The ingested dataset is a NEW version; the raw upload rides along as the
    # lineage parent and is still on disk untouched.
    raw_profiles = _artifacts(store, RUN_ID, ArtifactType.RAW_DATASET_PROFILE)
    assert {a.payload["dataset_id"] for a in raw_profiles} == {
        precleaned_run["raw_dataset_id"]
    }
    assert precleaned_run["cleaned_dataset_id"] != precleaned_run["raw_dataset_id"]
    uploads = workspace / "projects" / PROJECT / "uploads"
    assert (uploads / precleaned_run["raw_dataset_id"]).is_dir()
    assert (uploads / precleaned_run["cleaned_dataset_id"]).is_dir()
    # The staging copies the cleaner wrote next to the source are cleaned up.
    assert not list(uploads.rglob("_precleaned"))


def test_a_run_without_precleaning_ingests_the_upload_untouched(
    client: TestClient, store: ArtifactStore, precleaned_run: dict
) -> None:
    session_id = "run_tier3_plain"
    created = client.post(
        f"/api/v1/sessions/{session_id}/jobs",
        json={
            "kind": "auto_eda",
            "project_id": PROJECT,
            "datasets": [precleaned_run["raw_dataset_id"]],
            "generate_report": False,
            "llm": "offline",
        },
        headers={"Idempotency-Key": "tier3-plain"},
    )
    assert created.status_code == 201, created.text
    assert _wait_terminal(client, created.json()["job_id"])["status"] == "completed"

    assert not _artifacts(store, session_id, ArtifactType.CLEANING_RECIPE)
    profiles = _artifacts(store, session_id, ArtifactType.DATASET_PROFILE)
    assert {a.payload["dataset_id"] for a in profiles} == {
        precleaned_run["raw_dataset_id"]
    }


# --- on-demand report generation --------------------------------------------


def test_report_is_generated_on_demand_and_lands_on_the_source_run(
    client: TestClient, precleaned_run: dict, workspace: Path
) -> None:
    assert client.get(f"/api/v1/sessions/{RUN_ID}/report").json()["status"] == "none"

    started = client.post(
        f"/api/v1/sessions/{RUN_ID}/report/generate",
        json={"llm": "offline"},
        headers={"Idempotency-Key": "tier3-report"},
    )
    assert started.status_code == 201, started.text
    body = started.json()
    assert body["regenerated"] is False
    # The job runs on its own derived run so a failed generation cannot flip the
    # source run to failed; the report itself still lands on the source run.
    assert body["execution_session_id"].startswith("rpsess_")
    assert _wait_terminal(client, body["job"]["job_id"])["status"] == "completed"

    report_md = workspace / "projects" / PROJECT / "sessions" / RUN_ID / "report" / "report.md"
    assert report_md.is_file() and report_md.stat().st_size > 0
    after = client.get(f"/api/v1/sessions/{RUN_ID}/report").json()
    assert after["status"] != "none" and after["markdown"]


def test_regenerating_an_existing_report_is_flagged_as_a_replacement(
    client: TestClient,
) -> None:
    started = client.post(
        f"/api/v1/sessions/{RUN_ID}/report/generate",
        json={"llm": "offline"},
        headers={"Idempotency-Key": "tier3-report-again"},
    )
    assert started.status_code == 201, started.text
    assert started.json()["regenerated"] is True
    assert _wait_terminal(client, started.json()["job"]["job_id"])["status"] == "completed"


def test_report_generation_replays_on_the_same_idempotency_key(
    client: TestClient, precleaned_run: dict,
) -> None:
    first = client.post(
        f"/api/v1/sessions/{RUN_ID}/report/generate",
        json={"llm": "offline"},
        headers={"Idempotency-Key": "tier3-report-replay"},
    )
    assert first.status_code == 201, first.text
    _wait_terminal(client, first.json()["job"]["job_id"])
    replay = client.post(
        f"/api/v1/sessions/{RUN_ID}/report/generate",
        json={"llm": "offline"},
        headers={"Idempotency-Key": "tier3-report-replay"},
    )
    assert replay.status_code == 201, replay.text
    assert replay.json()["job"]["job_id"] == first.json()["job"]["job_id"]


def test_report_generation_idempotency_key_binds_llm_mode(
    client: TestClient, precleaned_run: dict,
) -> None:
    key = "tier3-report-content-bound"
    first = client.post(
        f"/api/v1/sessions/{RUN_ID}/report/generate",
        json={"llm": "offline"},
        headers={"Idempotency-Key": key},
    )
    assert first.status_code == 201, first.text

    changed = client.post(
        f"/api/v1/sessions/{RUN_ID}/report/generate",
        json={"llm": "env"},
        headers={"Idempotency-Key": key},
    )
    assert changed.status_code == 422, changed.text
    assert changed.json()["error"]["code"] == "idempotency_key_reused"
    assert _wait_terminal(client, first.json()["job"]["job_id"])["status"] == "completed"


def test_report_generation_404s_for_an_unknown_run(client: TestClient) -> None:
    response = client.post("/api/v1/sessions/run_missing/report/generate", json={})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


# --- what-if fork -----------------------------------------------------------


def test_fork_derives_a_new_run_whose_manifest_points_back(
    client: TestClient, store: ArtifactStore, precleaned_run: dict
) -> None:
    started = client.post(
        f"/api/v1/sessions/{RUN_ID}/fork",
        json={"decision": "ml_target", "ml_target_column": "amount", "llm": "offline"},
        headers={"Idempotency-Key": "tier3-fork"},
    )
    assert started.status_code == 201, started.text
    body = started.json()
    assert body["decision"] == "ML target → amount"
    assert body["execution_session_id"].startswith("fksess_")
    assert _wait_terminal(client, body["job"]["job_id"])["status"] == "completed"

    # The forked run mints its own id inside the driver; the only channel back
    # to the client is the job's `session.forked` trace event.
    events = client.get(
        f"/api/v1/sessions/{body['execution_session_id']}/trace?type=session.forked"
    ).json()
    assert events["items"], "the fork job must report the run it created"
    forked_session_id = events["items"][0]["summary"]["forked_session_id"]
    assert forked_session_id != RUN_ID

    manifest = store.read_manifest(PROJECT, forked_session_id)
    assert manifest is not None and manifest.source_session_id == RUN_ID
    detail = client.get(f"/api/v1/sessions/{forked_session_id}").json()
    assert detail["source_session_id"] == RUN_ID


def test_a_dataset_fork_needs_at_least_one_dataset(client: TestClient) -> None:
    response = client.post(
        f"/api/v1/sessions/{RUN_ID}/fork",
        json={"decision": "dataset", "datasets": [], "llm": "offline"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "session_fork_invalid"


def test_fork_404s_for_an_unknown_run(client: TestClient) -> None:
    response = client.post(
        "/api/v1/sessions/run_missing/fork",
        json={"decision": "ml_target", "ml_target_column": None},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


# --- knowledge promotion ----------------------------------------------------


@pytest.fixture(scope="module")
def promotable(store: ArtifactStore, precleaned_run: dict) -> None:
    _seed_promotable_finding(
        store, precleaned_run["cleaned_name"], precleaned_run["cleaned_dataset_id"]
    )


def _prepare(client: TestClient, finding_id: str = FINDING_ID):
    return client.post(f"/api/v1/sessions/{RUN_ID}/findings/{finding_id}/prepare-promote")


def _promote(client: TestClient, prepared: dict, finding_id: str = FINDING_ID):
    return client.post(
        f"/api/v1/sessions/{RUN_ID}/findings/{finding_id}/promote",
        json={
            "action_hash": prepared["action_hash"],
            "approval_token": prepared["approval_token"],
        },
    )


def _seeds_snapshot(workspace: Path) -> str:
    path = workspace / "projects" / PROJECT / "semantic" / "seeds.json"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def test_prepare_previews_the_exact_knowledge_without_writing_it(
    client: TestClient, promotable: None, workspace: Path
) -> None:
    before = _seeds_snapshot(workspace)
    prepared = _prepare(client)
    assert prepared.status_code == 200, prepared.text
    body = prepared.json()
    assert body["question"] == QUESTION
    assert body["answer"] == ANSWER
    assert EVIDENCE_ID in body["evidence_note"]
    # Nothing is written until the approval is consumed.
    assert _seeds_snapshot(workspace) == before


def test_promotion_writes_a_verified_answer_into_the_project_seeds(
    client: TestClient, promotable: None, workspace: Path
) -> None:
    prepared = _prepare(client).json()
    promoted = _promote(client, prepared)
    assert promoted.status_code == 201, promoted.text
    assert promoted.json()["verified_answer_count"] == 1

    # Promoting the same question again replaces its answer rather than
    # appending a second one, which is why the second prepare says so.
    assert _prepare(client).json()["replaces_existing"] is True

    seeds = json.loads(
        (workspace / "projects" / PROJECT / "semantic" / "seeds.json").read_text(
            encoding="utf-8"
        )
    )
    answers = seeds["verified_answers"]
    assert [answer["question"] for answer in answers] == [QUESTION]
    assert answers[0]["answer"] == ANSWER


def test_replaying_same_promotion_generation_returns_persisted_success(
    client: TestClient, promotable: None
) -> None:
    prepared = _prepare(client).json()
    first = _promote(client, prepared)
    assert first.status_code == 201
    replay = _promote(client, prepared)
    assert replay.status_code == 201
    assert replay.json() == first.json()


def test_promotion_commit_then_response_failure_replays_without_double_version(
    client: TestClient,
    promotable: None,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare(client).json()
    original_promote = SemanticService.promote_approved_answer
    calls = 0

    def commit_then_fail(
        service: SemanticService,
        *args: object,
        **kwargs: object,
    ):
        nonlocal calls
        snapshot = original_promote(service, *args, **kwargs)  # type: ignore[arg-type]
        calls += 1
        if calls == 1:
            raise OSError("response path failed")
        return snapshot

    monkeypatch.setattr(
        SemanticService, "promote_approved_answer", commit_then_fail
    )
    with pytest.raises(OSError, match="response path failed"):
        _promote(client, prepared)

    repository = SemanticSeedsRepository(ArtifactStore(workspace), PROJECT)
    committed_version = repository.read().version
    retry = _promote(client, prepared)
    assert retry.status_code == 201, retry.text
    assert repository.read().version == committed_version


def test_an_approval_of_another_kind_cannot_open_a_promotion(
    client: TestClient, promotable: None, precleaned_run: dict
) -> None:
    """Same run, same pending_actions table, different kind: the promotion path
    must not even admit the row exists."""
    cleaning = client.post(
        f"/api/v1/sessions/{RUN_ID}/cleaning/preview",
        json={
            "dataset_id": precleaned_run["cleaned_dataset_id"],
            "drop_duplicate_rows": True,
        },
    )
    result, job = await_data_operation(
        client,
        cleaning,
        "cleaning-preview-result",
    )
    assert job["status"] == "completed"
    assert result is not None
    cross = _promote(client, result.json())
    assert cross.status_code == 404
    assert cross.json()["error"]["code"] == "approval_not_found"


def test_a_promotion_approval_is_bound_to_the_finding_in_the_path(
    client: TestClient, promotable: None
) -> None:
    prepared = _prepare(client).json()
    other = client.post(
        f"/api/v1/sessions/{RUN_ID}/findings/some_other_finding/promote",
        json={
            "action_hash": prepared["action_hash"],
            "approval_token": prepared["approval_token"],
        },
    )
    assert other.status_code == 422
    assert other.json()["error"]["code"] == "promotion_invalid"


def test_promotion_404s_for_a_finding_that_does_not_exist(
    client: TestClient, promotable: None
) -> None:
    response = _prepare(client, finding_id="not_a_finding")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "finding_not_found"


def test_a_non_finding_artifact_is_not_promotable(
    client: TestClient, promotable: None, store: ArtifactStore
) -> None:
    """The evidence profile the finding points at is a real artifact of this
    project — promoting it must still 404, not leak its payload."""
    response = _prepare(client, finding_id=EVIDENCE_ID)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "finding_not_found"
