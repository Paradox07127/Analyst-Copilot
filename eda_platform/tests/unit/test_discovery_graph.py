from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from eda_platform.core.kernel import SessionContext
from eda_platform.core.llm import MalformedProviderResponseError
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers import auto_eda, card_edit
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.questions import QuestionCandidateSet
from eda_platform.schemas.value_discovery import ValueMap
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.worker import runner


class WorkerExit(BaseException):
    pass


class Provider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def structured[T: BaseModel](self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append(task)
        if "semantic" in task:
            return schema.model_validate({"entity": "Sales", "columns": []})
        if self.calls.count(task) == 1:
            raise MalformedProviderResponseError("invalid question response")
        return schema.model_validate({"questions": [{
            "question_en": "Which region has the most revenue?",
            "target_datasets": ["sales.csv"],
            "llm_business_relevance": 0.8, "llm_actionability": 0.9,
        }]})

    def text(self, *, task: str, payload: dict) -> str:
        return "Sales Overview"

    def last_usage(self) -> None:
        return None


def _source(tmp_path: Path) -> tuple[ArtifactStore, Any, Artifact]:
    source = tmp_path / "sales.csv"
    source.write_text("region,revenue\nEast,10\nWest,20\n")
    dataset = load_csv(source, dataset_id="ds-sales")
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("project", name="Project")
    store.start_session("project", "source")
    profile = profile_dataset(dataset, project_id="project", session_id="source")
    store.save_artifact(profile)
    store.save_artifact(Artifact(
        id="value-map", type=ArtifactType.VALUE_MAP, project_id="project",
        session_id="source", payload=ValueMap().model_dump(mode="json"),
    ))
    return store, dataset, profile


def test_discovery_adopts_bootstrap_and_each_schema_repair_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, dataset, profile = _source(tmp_path)

    def run(provider: Provider) -> list[Artifact]:
        step = auto_eda.DiscoverQuestionsStep(
            [dataset], profile_artifact_ids=[profile.id], quality_artifact_ids=[],
            quality_context_artifact_ids=[], analysis_artifact_ids=[],
            relationship_artifact_ids=[], value_map_artifact_id="value-map", llm=provider,
            business_context="Sales", payload_policy="schema+aggregates",
        )
        return step.run(SessionContext(project_id="project", session_id="source", store=store))

    original = auto_eda.discover_question_candidates

    def crash(*args: Any, **kwargs: Any) -> Any:
        raise WorkerExit

    monkeypatch.setattr(auto_eda, "discover_question_candidates", crash)
    first = Provider()
    with pytest.raises(WorkerExit):
        run(first)
    assert len(first.calls) == 3
    monkeypatch.setattr(auto_eda, "discover_question_candidates", original)
    resumed = Provider()
    artifacts = run(resumed)
    assert resumed.calls == []
    candidate_set = next(a for a in artifacts if a.type == ArtifactType.QUESTION_CANDIDATE_SET)
    assert candidate_set.payload["llm_route_skipped"] is False


def test_manual_draft_restart_adopts_saved_repaired_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _, _ = _source(tmp_path)
    store.save_artifact(Artifact(
        id="questions", type=ArtifactType.QUESTION_CANDIDATE_SET, project_id="project",
        session_id="source", payload=QuestionCandidateSet(candidates=[]).model_dump(mode="json"),
    ))
    job = {"project_id": "project", "session_id": "draft-session", "job_id": "draft-job"}
    params = {"source_session_id": "source", "question": "Which region has most revenue?"}
    first = Provider()
    monkeypatch.setattr(runner, "_build_llm", lambda _: first)
    monkeypatch.setattr(runner, "emit_job_event", lambda *args, **kwargs: None)
    original = card_edit.append_candidate

    def crash(*args: Any, **kwargs: Any) -> Any:
        raise WorkerExit

    monkeypatch.setattr(card_edit, "append_candidate", crash)
    with pytest.raises(WorkerExit):
        runner._run_question_draft_job(store, str(store.root), job, params)
    assert len(first.calls) == 2
    resumed = Provider()
    monkeypatch.setattr(runner, "_build_llm", lambda _: resumed)
    monkeypatch.setattr(card_edit, "append_candidate", original)
    runner._run_question_draft_job(store, str(store.root), job, params)
    assert resumed.calls == []
    saved = store.get_artifact("questions", project_id="project", session_id="source")
    assert len(saved.payload["candidates"]) == 1


def _step(store: ArtifactStore, dataset: Any, profile: Artifact, provider: Provider,
          relationships: list[str] | None = None) -> list[Artifact]:
    return auto_eda.DiscoverQuestionsStep(
        [dataset], profile_artifact_ids=[profile.id], quality_artifact_ids=[],
        quality_context_artifact_ids=[], analysis_artifact_ids=[],
        relationship_artifact_ids=relationships or [], value_map_artifact_id="value-map",
        llm=provider, business_context="Sales", payload_policy="schema+aggregates",
    ).run(SessionContext(project_id="project", session_id="source", store=store))


def _join(status: str) -> Any:
    from eda_platform.core.semantic import JoinWhitelistEntry

    return JoinWhitelistEntry.model_validate({
        "left_dataset": "sales.csv", "right_dataset": "sales.csv",
        "left_dataset_id": "ds-sales", "right_dataset_id": "ds-sales",
        "left_columns": ["region"], "right_columns": ["region"],
        "cardinality": "one_to_one", "validation_verified": True,
        "status": status, "confirmed_by": "auto" if status == "auto_confirmed" else "user",
    })


@pytest.mark.parametrize("changed", ["seeds", "docs", "skills", "joins"])
def test_discovery_rejects_context_drift_even_after_graph_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    from eda_platform.core.graph_execution import GraphIdentityError
    from eda_platform.core.semantic import FieldMeaning, JoinWhitelist, SemanticSeeds
    from eda_platform.core.support_docs import SupportDoc

    store, dataset, profile = _source(tmp_path)
    seeds = SemanticSeeds()
    docs: list[SupportDoc] = []
    skills = ""
    whitelist = JoinWhitelist()
    monkeypatch.setattr(auto_eda, "load_semantic_seeds_safe",
                        lambda *_: seeds.model_copy(deep=True))
    monkeypatch.setattr(auto_eda, "load_support_docs", lambda *_: list(docs))
    monkeypatch.setattr(auto_eda, "catalog_block", lambda *_: skills)
    monkeypatch.setattr(auto_eda, "_load_join_whitelist_safe",
                        lambda *_, **__: whitelist.model_copy(deep=True))
    _step(store, dataset, profile, Provider())
    if changed == "seeds":
        seeds.field_meanings.append(
            FieldMeaning(dataset="sales.csv", column="revenue", meaning="Net")
        )
    elif changed == "docs":
        docs.append(SupportDoc(name="dictionary.md", text="Revenue means net sales."))
    elif changed == "skills":
        skills = "- Validated net revenue analysis"
    else:
        whitelist.entries.append(_join("confirmed"))
    resumed = Provider()
    with pytest.raises(GraphIdentityError):
        _step(store, dataset, profile, resumed)
    assert resumed.calls == []


@pytest.mark.parametrize("revoke", [False, True])
def test_discovery_own_join_write_resumes_but_user_revocation_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, revoke: bool,
) -> None:
    from eda_platform.core.graph_execution import GraphIdentityError
    from eda_platform.core.semantic import load_join_whitelist, save_join_whitelist
    from eda_platform.schemas.relations import RelationshipCandidateSet

    store, dataset, profile = _source(tmp_path)
    store.save_artifact(Artifact(
        id="relations", type=ArtifactType.RELATIONSHIP_CANDIDATE_SET, project_id="project",
        session_id="source", payload=RelationshipCandidateSet().model_dump(mode="json"),
    ))
    monkeypatch.setattr(auto_eda, "propose_join_candidates",
                        lambda *_, **__: [_join("auto_confirmed")])
    original = auto_eda.discover_question_candidates

    def crash(*_: Any, **__: Any) -> Any:
        raise WorkerExit

    monkeypatch.setattr(auto_eda, "discover_question_candidates", crash)
    first = Provider()
    with pytest.raises(WorkerExit):
        _step(store, dataset, profile, first, ["relations"])
    whitelist = load_join_whitelist(store.project_dir("project"))
    assert whitelist.entries[0].status == "auto_confirmed"
    if revoke:
        whitelist.entries[0].status = "proposed"
        whitelist.entries[0].confirmed_by = ""
        save_join_whitelist(store.project_dir("project"), whitelist)
    monkeypatch.setattr(auto_eda, "discover_question_candidates", original)
    resumed = Provider()
    if revoke:
        with pytest.raises(GraphIdentityError):
            _step(store, dataset, profile, resumed, ["relations"])
    else:
        assert _step(store, dataset, profile, resumed, ["relations"])
    assert resumed.calls == []


@pytest.mark.parametrize("changed", ["seeds", "docs", "skills", "joins"])
def test_outer_pipeline_cache_does_not_hide_discovery_context_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    from eda_platform.core.graph_execution import GraphIdentityError
    from eda_platform.core.kernel import run_pipeline
    from eda_platform.core.semantic import FieldMeaning, JoinWhitelist, SemanticSeeds
    from eda_platform.core.support_docs import SupportDoc

    store, dataset, profile = _source(tmp_path)
    seeds = SemanticSeeds()
    docs: list[SupportDoc] = []
    skills = ""
    whitelist = JoinWhitelist()
    monkeypatch.setattr(
        auto_eda, "load_semantic_seeds_safe", lambda *_: seeds.model_copy(deep=True)
    )
    monkeypatch.setattr(auto_eda, "load_support_docs", lambda *_: list(docs))
    monkeypatch.setattr(auto_eda, "catalog_block", lambda *_: skills)
    monkeypatch.setattr(auto_eda, "_load_join_whitelist_safe",
                        lambda *_, **__: whitelist.model_copy(deep=True))

    def invoke(provider: Provider) -> list[Artifact]:
        step = auto_eda.DiscoverQuestionsStep(
            [dataset], profile_artifact_ids=[profile.id], quality_artifact_ids=[],
            quality_context_artifact_ids=[], analysis_artifact_ids=[],
            relationship_artifact_ids=[], value_map_artifact_id="value-map",
            llm=provider, business_context="Sales", payload_policy="schema+aggregates",
        )
        return run_pipeline([step], SessionContext(
            project_id="project", session_id="source", store=store
        )).artifacts

    assert invoke(Provider())
    unchanged = Provider()
    assert invoke(unchanged)
    assert unchanged.calls == []
    if changed == "seeds":
        seeds.field_meanings.append(
            FieldMeaning(dataset="sales.csv", column="revenue", meaning="Net")
        )
    elif changed == "docs":
        docs.append(SupportDoc(name="dictionary.md", text="Revenue means net sales."))
    elif changed == "skills":
        skills = "- Validated net revenue analysis"
    else:
        whitelist.entries.append(_join("confirmed"))
    resumed = Provider()
    with pytest.raises(GraphIdentityError):
        invoke(resumed)
    assert resumed.calls == []
