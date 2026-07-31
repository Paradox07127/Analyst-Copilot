"""Fresh-run driver for deterministic workflow evaluation cases."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from eda_platform.core.config import require_absolute_workspace
from eda_platform.core.llm import OfflineLLMClient
from eda_platform.core.session_metrics import persist_run_metrics
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import run_auto_eda
from eda_platform.drivers.question_exec import run_question_batch
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.questions import (
    QuestionCandidate,
    QuestionCandidateSet,
    QuestionScore,
)
from eda_platform.schemas.workflow_eval import WorkflowEvalProbe, WorkflowEvalSpec
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.relationship_discovery import _quote_identifier, _relation_name

_SOURCE_SUPPORT_TYPES = {
    ArtifactType.DATASET_PROFILE,
    ArtifactType.QUALITY_ISSUE_SET,
    ArtifactType.QUALITY_CONTEXT_SET,
    ArtifactType.COLUMN_ROLE_SET,
}


def run_fresh_workflow_eval_case(
    spec: WorkflowEvalSpec,
    *,
    input_dir: Path,
    workspace: Path,
    repeat: int,
) -> list[list[Artifact]]:
    """Execute a case repeatedly with the offline client and return eval inputs."""
    workspace = require_absolute_workspace(workspace)
    files = [input_dir / filename for filename in spec.input_files]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise ValueError("Missing case input(s): " + ", ".join(str(path) for path in missing))
    runs: list[list[Artifact]] = []
    for index in range(1, repeat + 1):
        run_workspace = workspace / f"run_{index:02d}"
        result = run_auto_eda(
            files,
            workspace=run_workspace,
            project_id=f"workflow_eval_{spec.name}",
            business_context=spec.business_context,
            llm=OfflineLLMClient(),
        )
        store = ArtifactStore(run_workspace)
        if not spec.probe_questions:
            runs.append(
                store.list_artifacts(
                    project_id=result.project_id,
                    session_id=result.session_id,
                )
            )
            continue
        source_artifacts = store.list_artifacts(
            project_id=result.project_id,
            session_id=result.session_id,
        )
        candidate_artifact = next(
            artifact
            for artifact in source_artifacts
            if artifact.type is ArtifactType.QUESTION_CANDIDATE_SET
        )
        candidate_set = QuestionCandidateSet.model_validate(candidate_artifact.payload)
        candidates = [
            _probe_candidate(probe, result.loaded_datasets) for probe in spec.probe_questions
        ]
        candidate_set.candidates.extend(candidates)
        candidate_artifact.payload = candidate_set.model_dump(mode="json")
        store.save_artifact(candidate_artifact)
        batch_session_id = f"{result.session_id}_eval_probe"
        batch = run_question_batch(
            project_id=result.project_id,
            source_session_id=result.session_id,
            question_ids=[candidate.question_id for candidate in candidates],
            workspace=run_workspace,
            llm=OfflineLLMClient(),
            session_id=batch_session_id,
            business_context=spec.business_context,
        )
        persist_run_metrics(store, result.project_id, batch.session_id)
        batch_artifacts = store.list_artifacts(
            project_id=result.project_id,
            session_id=batch.session_id,
        )
        support_artifacts = [
            artifact for artifact in source_artifacts if artifact.type in _SOURCE_SUPPORT_TYPES
        ]
        runs.append([*support_artifacts, *batch_artifacts])
    return runs


def _probe_candidate(
    probe: WorkflowEvalProbe, loaded_datasets: Sequence[LoadedDataset]
) -> QuestionCandidate:
    dataset = next(
        (dataset for dataset in loaded_datasets if dataset.record.name == probe.dataset_file),
        None,
    )
    if dataset is None:
        raise ValueError(
            f"Probe {probe.question_id!r} references missing dataset {probe.dataset_file!r}."
        )
    relation = _quote_identifier(_relation_name(dataset.record.dataset_id))
    sql = probe.sql_template.replace("{{relation}}", relation)
    return QuestionCandidate(
        question_id=probe.question_id,
        question_en=probe.question,
        origin="template",
        template_id=(
            "domain_metric" if probe.answer_contract.kind == "metric" else "workflow_eval_probe"
        ),
        answer_contract=probe.answer_contract,
        metric_id=probe.answer_contract.metric_id,
        produced_units=probe.produced_units,
        target_datasets=[probe.dataset_file],
        sql_template=sql,
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.75,
        ),
        status="approved",
    )
