"""Card editing."""

from __future__ import annotations

from typing import Any

from eda_platform.core.methods import MethodGateContext, evaluate_feasibility
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile
from eda_platform.schemas.questions import QuestionCandidate, QuestionCandidateSet

# Fields that can change without altering the analysis.
_TEXT_FIELDS: frozenset[str] = frozenset(
    {
        "question_en",
        "business_decision",
        "value_hypothesis",
        "success_criterion",
        "priority_rationale",
        "data_signal",
    }
)
_LIST_FIELDS: frozenset[str] = frozenset({"risks", "data_requirements"})
EDITABLE_FIELDS: frozenset[str] = _TEXT_FIELDS | _LIST_FIELDS

# These fields define the analysis and require a new card when changed.
_FORBIDDEN_FIELDS: frozenset[str] = frozenset({"sql_template", "target_datasets", "analysis_mode"})


class CardVersionConflictError(ValueError):
    def __init__(self, expected_version: int, current_version: int) -> None:
        super().__init__(
            f"Question card changed since it was loaded: expected version "
            f"{expected_version}, current version is {current_version}."
        )
        self.expected_version = expected_version
        self.current_version = current_version


def _normalize_edits(edits: dict[str, Any]) -> dict[str, Any]:
    unknown = set(edits) - EDITABLE_FIELDS
    if unknown:
        forbidden = sorted(unknown & _FORBIDDEN_FIELDS)
        if forbidden:
            raise ValueError(
                f"Cannot edit {forbidden}: these fields define what analysis runs "
                "(the SQL/scope/method), not how the card is framed. Changing them "
                "here would silently change what gets executed without re-running "
                "feasibility on the new scope. To change the analysis itself, "
                "propose a new card instead of editing this one. Editable fields "
                f"are: {sorted(EDITABLE_FIELDS)}."
            )
        raise ValueError(
            f"Unknown field(s) for a card edit: {sorted(unknown)}. Editable "
            f"fields are: {sorted(EDITABLE_FIELDS)}."
        )

    normalized: dict[str, Any] = {}
    for key, value in edits.items():
        if key in _TEXT_FIELDS:
            if not isinstance(value, str):
                raise ValueError(f"Field {key!r} must be a string; got {type(value).__name__}.")
            normalized[key] = value
        else:  # key in _LIST_FIELDS
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"Field {key!r} must be a list of strings; got {value!r}.")
            normalized[key] = list(value)
    return normalized


def _find_candidate_set_artifact(artifacts: list[Artifact]) -> Artifact:
    for artifact in artifacts:
        if artifact.type is ArtifactType.QUESTION_CANDIDATE_SET:
            return artifact
    raise ValueError("This run has no QuestionCandidateSet artifact to edit.")


def _profiles_for_feasibility(artifacts: list[Artifact]) -> list[DatasetProfile]:
    return [
        DatasetProfile.model_validate(artifact.payload)
        for artifact in artifacts
        if artifact.type is ArtifactType.DATASET_PROFILE
    ]


def edit_candidate(
    store: ArtifactStore,
    *,
    project_id: str,
    session_id: str,
    question_id: str,
    expected_version: int,
    edits: dict[str, str | list[str]],
) -> QuestionCandidate:
    """Apply a text edit to one candidate card and persist the new version."""
    normalized_edits = _normalize_edits(dict(edits))

    artifacts = store.list_artifacts(project_id=project_id, session_id=session_id)
    candidate_set_artifact = _find_candidate_set_artifact(artifacts)
    edited_result: QuestionCandidate | None = None

    def mutate(current_artifact: Artifact) -> Artifact:
        nonlocal edited_result
        # Re-read inside the fence: feasibility must reflect the profiles the
        # committed card is graded against, not a pre-lock snapshot.
        profiles = _profiles_for_feasibility(
            store.list_artifacts(project_id=project_id, session_id=session_id)
        )
        candidate_set = QuestionCandidateSet.model_validate(current_artifact.payload)
        index = next(
            (
                i
                for i, candidate in enumerate(candidate_set.candidates)
                if candidate.question_id == question_id
            ),
            None,
        )
        if index is None:
            raise ValueError(
                f"No candidate with question_id={question_id!r} in run {session_id!r}."
            )
        candidate = candidate_set.candidates[index]
        if candidate.card_version != expected_version:
            raise CardVersionConflictError(expected_version, candidate.card_version)
        candidate_payload = candidate.model_dump(mode="json")
        candidate_payload.update(normalized_edits)
        candidate_payload["card_version"] = candidate.card_version + 1
        edited = QuestionCandidate.model_validate(candidate_payload)
        feasibility = evaluate_feasibility(
            MethodGateContext(
                profiles=profiles,
                target_datasets=edited.target_datasets,
                analysis_mode=edited.analysis_mode,
                target_column=None,
            )
        )
        edited = edited.model_copy(
            update={
                "feasibility": feasibility,
                "candidate_methods": (
                    edited.candidate_methods
                    or ([feasibility.method_id] if feasibility.method_id is not None else [])
                ),
            }
        )
        new_candidates = list(candidate_set.candidates)
        new_candidates[index] = edited
        updated_candidate_set = candidate_set.model_copy(
            update={"candidates": new_candidates}
        )
        edited_result = edited
        return current_artifact.model_copy(
            update={"payload": updated_candidate_set.model_dump(mode="json")}
        )

    store.mutate_artifact(
        project_id=project_id,
        session_id=session_id,
        artifact_id=candidate_set_artifact.id,
        mutate=mutate,
    )
    if edited_result is None:  # not an assert: -O would strip the guard
        raise RuntimeError("Card mutation completed without producing a candidate.")
    return edited_result


def append_candidate(
    store: ArtifactStore,
    *,
    project_id: str,
    session_id: str,
    candidate: QuestionCandidate,
) -> QuestionCandidate:
    """Append a reviewable candidate to the run's existing candidate set."""
    artifacts = store.list_artifacts(project_id=project_id, session_id=session_id)
    candidate_set_artifact = _find_candidate_set_artifact(artifacts)
    candidate_set = QuestionCandidateSet.model_validate(candidate_set_artifact.payload)
    existing = {item.question_id: item for item in candidate_set.candidates}
    existing[candidate.question_id] = candidate
    updated_set = candidate_set.model_copy(update={"candidates": list(existing.values())})
    store.save_artifact(
        candidate_set_artifact.model_copy(update={"payload": updated_set.model_dump(mode="json")})
    )
    return candidate
