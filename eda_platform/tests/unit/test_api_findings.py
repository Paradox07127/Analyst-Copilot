"""Findings API slice: the project-level library viewed from one run, with
source-run annotations, evidence references, and freshness."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
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

PROJECT = "proj_findings"
LIB_RUN = "run_lib"
CURRENT_RUN = "run_current"
DATASET = "orders.csv"


def _profile_artifact(session_id: str, dataset_id: str) -> Artifact:
    profile = DatasetProfile(
        dataset_id=dataset_id,
        name=DATASET,
        rows=2,
        columns=1,
        column_names=["amount"],
        dtypes={"amount": "int64"},
        missing_values={"amount": 0},
        missing_percent={"amount": 0.0},
        numeric_columns=["amount"],
        categorical_columns=[],
    )
    return Artifact(
        id=f"profile_{session_id}",
        type=ArtifactType.DATASET_PROFILE,
        project_id=PROJECT,
        session_id=session_id,
        payload=profile.model_dump(),
    )


def _plan_artifact(session_id: str, investigation_id: str, question: str) -> Artifact:
    plan = InvestigationPlan(
        investigation_id=investigation_id,
        source_session_id=session_id,
        question_id=f"q_{investigation_id}",
        card_version=1,
        candidate_fingerprint="fingerprint",
        question=question,
        target_datasets=[DATASET],
        method_family="descriptive",
        method_recipe="average amount",
        allowed_tools=["sql"],
        feasibility="ready",
        status="planned",
        status_reason="Ready to execute.",
    )
    return Artifact(
        id=f"plan_{investigation_id}",
        type=ArtifactType.INVESTIGATION_PLAN,
        project_id=PROJECT,
        session_id=session_id,
        payload=plan.model_dump(),
    )


def _finding_artifact(
    session_id: str, investigation_id: str, question: str, evidence_id: str
) -> Artifact:
    finding = ValidatedFinding(
        finding_id=f"finding_{investigation_id}",
        investigation_id=investigation_id,
        question_id=f"q_{investigation_id}",
        question=question,
        value_hypothesis="Could increase profit.",
        claim_class="observed",
        findings=[
            QuestionFinding(
                text="Average order value was $42.",
                evidence=[
                    EvidenceRef(
                        kind="table",
                        artifact_id=evidence_id,
                        locator="rows[0].average",
                        value=42,
                    )
                ],
            )
        ],
        evidence_support="high",
        analytical_reliability="high",
        decision_readiness="medium",
        limitations=["Refunds were not identified."],
        report_eligible=True,
        report_readiness="eligible_with_limitations",
        report_readiness_reason="The descriptive claim is supported.",
        source_artifact_ids=[evidence_id],
    )
    return Artifact(
        id=f"finding_{investigation_id}",
        type=ArtifactType.VALIDATED_FINDING,
        project_id=PROJECT,
        session_id=session_id,
        payload=finding.model_dump(),
    )


def _record_artifact(
    session_id: str,
    investigation_id: str,
    *,
    status: str = "validated",
    finding_artifact_id: str | None = None,
) -> Artifact:
    record = InvestigationRecord(
        record_id=f"record_{investigation_id}",
        investigation_id=investigation_id,
        question_id=f"q_{investigation_id}",
        status=status,  # type: ignore[arg-type]
        reason_code="test_reason_code",
        reason="Terminal outcome for the test.",
        next_action="Review the evidence.",
        finding_artifact_id=finding_artifact_id,
    )
    return Artifact(
        id=f"record_{investigation_id}",
        type=ArtifactType.INVESTIGATION_RECORD,
        project_id=PROJECT,
        session_id=session_id,
        payload=record.model_dump(),
    )


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    root: Path = tmp_path_factory.mktemp("findings_api")
    store = ArtifactStore(root)
    store.ensure_project(PROJECT, "Findings project")
    store.start_session(PROJECT, LIB_RUN)
    store.start_session(PROJECT, CURRENT_RUN)

    # Library run: a fully verifiable (fresh) finding + its validated record.
    store.save_artifact(_profile_artifact(LIB_RUN, "ds_orders_v1"))
    store.save_artifact(_plan_artifact(LIB_RUN, "inv_lib", "What was average order value?"))
    store.save_artifact(
        _finding_artifact(
            LIB_RUN, "inv_lib", "What was average order value?", f"profile_{LIB_RUN}"
        )
    )
    store.save_artifact(
        _record_artifact(LIB_RUN, "inv_lib", finding_artifact_id="finding_inv_lib")
    )
    # And one inconclusive record for the investigation log.
    store.save_artifact(_plan_artifact(LIB_RUN, "inv_open", "Is churn predictable?"))
    store.save_artifact(_record_artifact(LIB_RUN, "inv_open", status="inconclusive"))

    # Current run: a finding without a plan → freshness is unverifiable.
    store.save_artifact(
        _finding_artifact(
            CURRENT_RUN, "inv_cur", "Do regions differ?", f"profile_{LIB_RUN}"
        )
    )
    store.save_artifact(
        _record_artifact(CURRENT_RUN, "inv_cur", finding_artifact_id="finding_inv_cur")
    )
    return TestClient(create_app(root))


def _findings_by_id(body: dict) -> dict[str, dict]:
    return {item["artifact_id"]: item for item in body["findings"]}


def test_findings_are_project_scoped_with_source_annotations(client: TestClient) -> None:
    response = client.get(f"/api/v1/sessions/{CURRENT_RUN}/findings")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_id"] == CURRENT_RUN
    assert body["project_id"] == PROJECT
    findings = _findings_by_id(body)
    assert set(findings) == {"finding_inv_lib", "finding_inv_cur"}

    lib = findings["finding_inv_lib"]
    assert lib["source_session_id"] == LIB_RUN
    assert lib["from_current_session"] is False
    assert lib["question"] == "What was average order value?"
    assert lib["claim_class"] == "observed"
    assert lib["report_readiness"] == "eligible_with_limitations"
    assert lib["statements"][0]["text"] == "Average order value was $42."
    assert lib["statements"][0]["evidence"][0]["artifact_id"] == f"profile_{LIB_RUN}"
    assert lib["freshness"]["status"] == "fresh"

    current = findings["finding_inv_cur"]
    assert current["from_current_session"] is True
    assert current["freshness"]["status"] == "unverifiable"
    assert current["freshness"]["reasons"]


def test_findings_view_from_library_run_marks_its_own_finding(client: TestClient) -> None:
    body = client.get(f"/api/v1/sessions/{LIB_RUN}/findings").json()
    findings = _findings_by_id(body)
    assert findings["finding_inv_lib"]["from_current_session"] is True
    assert findings["finding_inv_cur"]["from_current_session"] is False


def test_investigation_log_resolves_questions(client: TestClient) -> None:
    body = client.get(f"/api/v1/sessions/{CURRENT_RUN}/findings").json()
    records = {item["artifact_id"]: item for item in body["records"]}
    open_record = records["record_inv_open"]
    assert open_record["status"] == "inconclusive"
    assert open_record["question"] == "Is churn predictable?"
    assert open_record["reason_code"] == "test_reason_code"
    assert open_record["source_session_id"] == LIB_RUN


def test_unknown_and_internal_runs_are_404(client: TestClient) -> None:
    response = client.get("/api/v1/sessions/no_such_run/findings")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"
    response = client.get("/api/v1/sessions/run__internal_probe/findings")
    assert response.status_code == 404


def test_evidence_refs_carry_their_source_run(client: TestClient) -> None:
    body = client.get(f"/api/v1/sessions/{CURRENT_RUN}/findings").json()
    findings = _findings_by_id(body)
    # Evidence artifacts live in LIB_RUN even when the finding is elsewhere;
    # the ref names that run so the UI can link into the right Artifacts page.
    for finding_id in ("finding_inv_lib", "finding_inv_cur"):
        evidence = findings[finding_id]["statements"][0]["evidence"][0]
        assert evidence["artifact_id"] == f"profile_{LIB_RUN}"
        assert evidence["session_id"] == LIB_RUN
    assert findings["finding_inv_lib"]["source_session_navigable"] is True


def test_internal_source_run_is_not_navigable(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    root = tmp_path_factory.mktemp("findings_internal")
    store = ArtifactStore(root)
    store.ensure_project(PROJECT, "Findings project")
    internal_run = "run_probe__internal_x"
    view_run = "run_view"
    store.start_session(PROJECT, internal_run)
    store.start_session(PROJECT, view_run)
    store.save_artifact(_profile_artifact(internal_run, "ds_internal"))
    store.save_artifact(
        _finding_artifact(
            internal_run, "inv_int", "Internal probe?", f"profile_{internal_run}"
        )
    )
    store.save_artifact(
        _record_artifact(internal_run, "inv_int", finding_artifact_id="finding_inv_int")
    )
    client = TestClient(create_app(root))

    body = client.get(f"/api/v1/sessions/{view_run}/findings").json()
    finding = _findings_by_id(body)["finding_inv_int"]
    # The finding stays listed, but nothing about it is linkable: there is no
    # page for internal runs, and its evidence lives in one.
    assert finding["source_session_id"] == internal_run
    assert finding["source_session_navigable"] is False
    assert finding["statements"][0]["evidence"][0]["session_id"] is None


def test_freshness_snapshot_is_shared_across_findings(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H7: one project-runs snapshot per listing, not one per finding."""
    root = tmp_path_factory.mktemp("findings_perf")
    store = ArtifactStore(root)
    store.ensure_project(PROJECT, "Findings project")
    session_ids = [f"run_{index:02d}" for index in range(20)]
    for session_id in session_ids:
        store.start_session(PROJECT, session_id)
    for index in range(40):
        inv = f"inv_{index:02d}"
        store.save_artifact(
            _finding_artifact(session_ids[0], inv, f"Question {index}?", f"missing_{index}")
        )
        store.save_artifact(
            _record_artifact(session_ids[0], inv, finding_artifact_id=f"finding_{inv}")
        )
    client = TestClient(create_app(root))

    typed_calls = {"count": 0}
    full_scan_calls = {"count": 0}
    original_typed = ArtifactStore.list_artifacts_of_types
    original_safe = ArtifactStore.list_artifacts_safe

    def counting_typed(
        self: ArtifactStore, **kwargs: object
    ) -> tuple[list, list]:
        typed_calls["count"] += 1
        return original_typed(self, **kwargs)  # type: ignore[arg-type]

    def counting_safe(self: ArtifactStore, **kwargs: object) -> tuple[list, list]:
        full_scan_calls["count"] += 1
        return original_safe(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ArtifactStore, "list_artifacts_of_types", counting_typed)
    monkeypatch.setattr(ArtifactStore, "list_artifacts_safe", counting_safe)
    response = client.get(f"/api/v1/sessions/{session_ids[0]}/findings")
    assert response.status_code == 200, response.text
    assert len(response.json()["findings"]) == 40
    # 20 sessions for the library read model + 20 for the shared freshness
    # snapshot. Before the shared snapshot each of the 40 findings rebuilt
    # its own: 20 + 40 × 20 = 820 full scans. Indexed paths must not fall
    # back to the full disk scan.
    assert typed_calls["count"] == 40
    assert full_scan_calls["count"] == 0
