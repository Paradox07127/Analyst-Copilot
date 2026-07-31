"""F-016 independent-review counterexamples for durable semantic commands."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eda_platform.application.services.approval_service import ApprovalService
from eda_platform.application.services.promotion_service import (
    PromotionService,
    PromotionSourceChangedError,
)
from eda_platform.application.services.semantic_service import (
    SemanticProposalConflictError,
    SemanticService,
)
from eda_platform.core.meaning_proposals import (
    MeaningProposal,
    MeaningProposals,
    apply_proposal_to_seeds,
)
from eda_platform.core.semantic import VerifiedAnswer
from eda_platform.core.semantic_resources import SemanticSeedsRepository
from eda_platform.core.storage_operations import (
    ResourceVersionConflictError,
    StorageOperationJournal,
    canonical_digest,
)
from eda_platform.core.store import ArtifactStore

PROJECT = "durable-semantic"
RUN = "run-durable-semantic"
FINDING = "finding-durable"
FIXED_AT = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


def _store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, "Durable semantic")
    store.start_session(PROJECT, RUN)
    return store


def _answer(*, evidence: str = "Source artifacts: evidence-a.") -> VerifiedAnswer:
    return VerifiedAnswer(
        question="What changed?",
        answer="Revenue increased.",
        evidence_note=evidence,
        verified_at=FIXED_AT,
    )


def _approval(
    store: ArtifactStore,
    answer: VerifiedAnswer,
) -> tuple[str, str, dict[str, object]]:
    payload: dict[str, object] = {
        "finding_id": FINDING,
        "project_id": PROJECT,
        "source_session_id": RUN,
        "verified_answer": answer.model_dump(mode="json"),
    }
    action_hash, generation, _expires = ApprovalService(store).register(
        kind="knowledge_promote",
        session_id=RUN,
        project_id=PROJECT,
        action={
            "type": "knowledge_promote",
            "source_session_id": RUN,
            "finding_id": FINDING,
            **answer.model_dump(mode="json"),
        },
        payload=payload,
    )
    return action_hash, generation, payload


def _execute(
    repository: SemanticSeedsRepository,
    *,
    action_hash: str,
    generation: str,
    payload: dict[str, object],
    answer: VerifiedAnswer,
):
    request_digest = canonical_digest(
        {
            "action_hash": action_hash,
            "generation": generation,
            "approval_payload": payload,
        }
    )
    return repository.execute_approved_answer(
        action_hash=action_hash,
        session_id=RUN,
        generation=generation,
        approval_kind="knowledge_promote",
        approval_payload=payload,
        candidate=answer,
        request_digest=request_digest,
    )


def test_crash_after_atomic_reservation_recovers_same_generation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    SemanticSeedsRepository(store, PROJECT).read()
    answer = _answer()
    action_hash, generation, payload = _approval(store, answer)

    def crash(stage: str, _op_id: str, _ordinal: int | None) -> None:
        if stage == "after_reserve":
            raise RuntimeError("crash after durable reservation")

    crashing = SemanticSeedsRepository(
        store,
        PROJECT,
        journal=StorageOperationJournal(store, fault_hook=crash),
    )
    with pytest.raises(RuntimeError, match="durable reservation"):
        _execute(
            crashing,
            action_hash=action_hash,
            generation=generation,
            payload=payload,
            answer=answer,
        )

    pending = store.get_pending_action(action_hash, session_id=RUN)
    assert pending is not None and pending["status"] == "executing"
    recovered = _execute(
        SemanticSeedsRepository(store, PROJECT),
        action_hash=action_hash,
        generation=generation,
        payload=payload,
        answer=answer,
    )
    assert recovered.version == 1
    assert recovered.seeds.verified_answers == [answer]
    pending = store.get_pending_action(action_hash, session_id=RUN)
    assert pending is not None and pending["status"] == "consumed"


def test_new_approval_generation_never_replays_old_operation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    repository = SemanticSeedsRepository(store, PROJECT)
    repository.read()
    answer = _answer()
    action_hash, generation_one, payload_one = _approval(store, answer)
    assert _execute(
        repository,
        action_hash=action_hash,
        generation=generation_one,
        payload=payload_one,
        answer=answer,
    ).version == 1

    current = repository.read()
    current.seeds.verified_answers = []
    repository.replace_state(
        expected_version=current.version,
        new_seeds=current.seeds,
        new_proposals=current.proposals,
        request_key="intervening-user-edit",
    )
    _, generation_two, payload_two = _approval(store, answer)
    assert generation_two != generation_one
    promoted = _execute(
        repository,
        action_hash=action_hash,
        generation=generation_two,
        payload=payload_two,
        answer=answer,
    )
    assert promoted.version == 3
    assert repository.read().seeds.verified_answers == [answer]


def test_proposal_retry_after_intervening_edit_is_not_false_success(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    repository = SemanticSeedsRepository(store, PROJECT)
    initial = repository.read()
    proposal = MeaningProposal(
        dataset="orders.csv",
        column="amount",
        meaning="Order value.",
        unit_guess="USD",
        revision="proposal-revision-1",
    )
    repository.replace_state(
        expected_version=initial.version,
        new_seeds=initial.seeds,
        new_proposals=MeaningProposals(proposals=[proposal]),
        request_key="seed-proposal",
    )
    service = SemanticService(store)
    accepted = service.accept_meaning_proposal(
        RUN, dataset="orders.csv", column="amount"
    )
    assert accepted.seeds_version == 2

    intervening = repository.read()
    intervening.seeds.field_meanings = []
    repository.replace_state(
        expected_version=intervening.version,
        new_seeds=intervening.seeds,
        new_proposals=intervening.proposals,
        request_key="analyst-removes-field",
    )
    with pytest.raises(SemanticProposalConflictError, match="no longer matches"):
        service.accept_meaning_proposal(
            RUN, dataset="orders.csv", column="amount"
        )


def test_stale_proposal_writer_cannot_overwrite_other_process_acceptance(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    repository = SemanticSeedsRepository(store, PROJECT)
    initial = repository.read()
    repository.replace_state(
        expected_version=initial.version,
        new_seeds=initial.seeds,
        new_proposals=MeaningProposals(
            proposals=[
                MeaningProposal(
                    dataset="orders.csv",
                    column="a",
                    meaning="A",
                    revision="rev-a",
                ),
                MeaningProposal(
                    dataset="orders.csv",
                    column="b",
                    meaning="B",
                    revision="rev-b",
                ),
            ]
        ),
        request_key="two-proposals",
    )
    first = SemanticSeedsRepository(store, PROJECT).read()
    stale = SemanticSeedsRepository(store, PROJECT).read()
    proposal_a = first.proposals.find("orders.csv", "a")
    assert proposal_a is not None
    apply_proposal_to_seeds(first.seeds, proposal_a)
    proposal_a.status = "accepted"
    SemanticSeedsRepository(store, PROJECT).replace_state(
        expected_version=first.version,
        new_seeds=first.seeds,
        new_proposals=first.proposals,
        request_key="accept-a",
    )

    proposal_b = stale.proposals.find("orders.csv", "b")
    assert proposal_b is not None
    apply_proposal_to_seeds(stale.seeds, proposal_b)
    proposal_b.status = "accepted"
    with pytest.raises(ResourceVersionConflictError):
        SemanticSeedsRepository(store, PROJECT).replace_state(
            expected_version=stale.version,
            new_seeds=stale.seeds,
            new_proposals=stale.proposals,
            request_key="stale-accept-b",
        )
    persisted_a = (
        SemanticSeedsRepository(store, PROJECT)
        .read()
        .proposals.find("orders.csv", "a")
    )
    assert persisted_a is not None and persisted_a.status == "accepted"


def test_promotion_rejects_evidence_note_swap_after_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    semantic = SemanticService(store)
    promotion = PromotionService(store, ApprovalService(store), semantic)
    monkeypatch.setattr(
        promotion,
        "_require_promotable",
        lambda *_args: _answer(evidence="Source artifacts: evidence-a."),
    )
    prepared = promotion.prepare(RUN, FINDING)
    monkeypatch.setattr(
        promotion,
        "_require_promotable",
        lambda *_args: _answer(evidence="Source artifacts: evidence-b."),
    )
    with pytest.raises(PromotionSourceChangedError):
        promotion.promote(
            RUN,
            finding_id=FINDING,
            action_hash=prepared.action_hash,
            approval_token=prepared.approval_token,
        )


def test_old_generation_compensation_cannot_reopen_new_generation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    SemanticSeedsRepository(store, PROJECT).read()
    answer = _answer()
    action_hash, generation_one, payload_one = _approval(store, answer)

    def crash(stage: str, _op_id: str, _ordinal: int | None) -> None:
        if stage == "after_reserve":
            raise RuntimeError("leave prepared")

    repository = SemanticSeedsRepository(
        store,
        PROJECT,
        journal=StorageOperationJournal(store, fault_hook=crash),
    )
    with pytest.raises(RuntimeError, match="leave prepared"):
        _execute(
            repository,
            action_hash=action_hash,
            generation=generation_one,
            payload=payload_one,
            answer=answer,
        )
    with sqlite3.connect(store.db_path) as conn:
        op_id = str(
            conn.execute(
                """
                select op_id from semantic_commands
                where action_hash = ? and session_id = ?
                  and approval_generation = ?
                """,
                (action_hash, RUN, generation_one),
            ).fetchone()[0]
        )

    _, generation_two, _payload_two = _approval(store, answer)
    assert repository.compensate_approved_answer(
        action_hash=action_hash,
        session_id=RUN,
        generation=generation_one,
        op_id=op_id,
    )
    pending = store.get_pending_action(action_hash, session_id=RUN)
    assert pending is not None
    assert pending["generation"] == generation_two
    assert pending["status"] == "pending"
