"""Knowledge promotion use case (§7.5): turn a validated, still-fresh finding
into a `VerifiedAnswer` in the project's semantic seeds.

This writes the semantic layer, so it follows the questions prepare → approve →
execute shape: prepare returns the exact text that would be stored plus a
one-time approval, and promote consumes that approval before touching seeds.
The seeds replacement is committed through SemanticService's cross-process
repository transaction.
"""

from __future__ import annotations

from datetime import UTC, datetime

from eda_platform.application.dto import KnowledgePromoted, KnowledgePromotionPrepared
from eda_platform.application.services.approval_service import (
    ApprovalExpiredError,
    ApprovalNotFoundError,
    ApprovalService,
)
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.application.services.semantic_service import SemanticService
from eda_platform.core.finding_freshness import assess_finding_freshness
from eda_platform.core.ids import INTERNAL_SESSION_MARKER
from eda_platform.core.semantic import VerifiedAnswer
from eda_platform.core.storage_operations import canonical_digest
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.knowledge_promotion import build_promotion_candidate
from eda_platform.schemas.artifacts import ArtifactType

APPROVAL_KIND_PROMOTION = "knowledge_promote"


class PromotionServiceError(Exception):
    pass


class PromotionFindingNotFoundError(PromotionServiceError):
    def __init__(self, finding_id: str, session_id: str) -> None:
        super().__init__(
            f"Finding not found in the project behind run {session_id}: {finding_id}"
        )
        self.finding_id = finding_id
        self.session_id = session_id


class PromotionNotAllowedError(PromotionServiceError):
    """The finding exists but is not promotable (stale, or nothing to say)."""

    def __init__(self, finding_id: str, reason: str) -> None:
        super().__init__(f"Finding {finding_id} cannot be promoted: {reason}")
        self.finding_id = finding_id


class PromotionSourceChangedError(PromotionServiceError):
    def __init__(self, finding_id: str) -> None:
        super().__init__(
            f"Finding {finding_id} changed since the approval was prepared; "
            "prepare again and approve the fresh content."
        )
        self.finding_id = finding_id


class PromotionValidationError(PromotionServiceError):
    pass


class PromotionService:
    def __init__(
        self,
        store: ArtifactStore,
        approvals: ApprovalService,
        semantic: SemanticService,
    ) -> None:
        self._store = store
        self._approvals = approvals
        self._semantic = semantic

    def prepare(self, session_id: str, finding_id: str) -> KnowledgePromotionPrepared:
        project_id = self._project_for_run(session_id)
        candidate = self._require_promotable(project_id, session_id, finding_id)
        seeds = self._semantic.read_project_snapshot(project_id).seeds
        replaces = any(
            existing.question == candidate.question for existing in seeds.verified_answers
        )
        digest, approval_token, expires_at = self._approvals.register(
            kind=APPROVAL_KIND_PROMOTION,
            session_id=session_id,
            project_id=project_id,
            action={
                "type": "knowledge_promote",
                "source_session_id": session_id,
                "finding_id": finding_id,
                "question": candidate.question,
                "answer": candidate.answer,
                "evidence_note": candidate.evidence_note,
                "verified_at": candidate.verified_at.isoformat(),
            },
            payload={
                "finding_id": finding_id,
                "project_id": project_id,
                "source_session_id": session_id,
                # The approval binds the text: promote refuses if the finding
                # has been rewritten since the user read this preview.
                "verified_answer": candidate.model_dump(mode="json"),
            },
        )
        return KnowledgePromotionPrepared(
            session_id=session_id,
            finding_id=finding_id,
            action_hash=digest,
            approval_token=approval_token,
            expires_at=expires_at,
            question=candidate.question,
            answer=candidate.answer,
            evidence_note=candidate.evidence_note or "",
            replaces_existing=replaces,
        )

    def promote(
        self,
        session_id: str,
        *,
        finding_id: str,
        action_hash: str,
        approval_token: str,
    ) -> KnowledgePromoted:
        project_id = self._project_for_run(session_id)
        payload, _payload_digest, status = self._approvals.inspect_payload(
            action_hash, session_id=session_id
        )
        row = self._store.get_pending_action(action_hash, session_id=session_id)
        if (
            row is None
            or str(row["kind"]) != APPROVAL_KIND_PROMOTION
            or str(row["generation"]) != approval_token
        ):
            raise ApprovalNotFoundError(action_hash)
        if (
            str(row["status"]) == "expired"
            or str(row["expires_at"]) <= datetime.now(UTC).isoformat()
        ):
            self._store.expire_pending_action(
                action_hash,
                session_id=session_id,
                now=datetime.now(UTC).isoformat(),
            )
            raise ApprovalExpiredError(action_hash)
        if str(payload.get("finding_id", "")) != finding_id:
            raise PromotionValidationError(
                "The approval was prepared for a different finding than the request path."
            )
        frozen_payload = payload.get("verified_answer")
        if not isinstance(frozen_payload, dict):
            raise PromotionValidationError(
                "The approval does not contain a complete verified answer."
            )
        try:
            approved = VerifiedAnswer.model_validate(frozen_payload)
        except ValueError as exc:
            raise PromotionValidationError(
                "The approval contains an invalid verified answer."
            ) from exc
        if status == "pending":
            candidate = self._require_promotable(project_id, session_id, finding_id)
            if (
                candidate.question != approved.question
                or candidate.answer != approved.answer
                or candidate.evidence_note != approved.evidence_note
                or candidate.verified_at != approved.verified_at
            ):
                raise PromotionSourceChangedError(finding_id)
        request_digest = canonical_digest(
            {
                "action_hash": action_hash,
                "generation": approval_token,
                "approval_payload": payload,
            }
        )
        snapshot = self._semantic.promote_approved_answer(
            project_id,
            session_id=session_id,
            action_hash=action_hash,
            generation=approval_token,
            approval_payload=payload,
            candidate=approved,
            request_digest=request_digest,
        )
        return KnowledgePromoted(
            session_id=session_id,
            finding_id=finding_id,
            question=approved.question,
            answer=approved.answer,
            verified_answer_count=len(snapshot.seeds.verified_answers),
        )

    def _require_promotable(
        self, project_id: str, session_id: str, finding_id: str
    ) -> VerifiedAnswer:
        """The VerifiedAnswer this finding would become, or the reason it cannot.

        Freshness is re-checked on prepare and execute. The prepare step must
        refuse up front rather than hand out an approval that can only fail.
        `assess_finding_freshness` reports a missing artifact as
        "unverifiable", so existence is checked separately to keep 404 and 409
        apart.
        """
        try:
            artifact = self._store.get_artifact(
                finding_id,
                project_id=project_id,
                session_id=session_id,
            )
        except (KeyError, OSError, ValueError) as exc:
            raise PromotionFindingNotFoundError(finding_id, session_id) from exc
        if artifact.type is not ArtifactType.VALIDATED_FINDING:
            raise PromotionFindingNotFoundError(finding_id, session_id)
        freshness = assess_finding_freshness(
            self._store,
            project_id,
            finding_id,
            finding_session_id=session_id,
        )
        if freshness.status != "fresh":
            raise PromotionNotAllowedError(
                finding_id,
                f"it is {freshness.status}. {' '.join(freshness.reasons)}".strip(),
            )
        try:
            return build_promotion_candidate(
                self._store,
                project_id,
                finding_id,
                session_id=session_id,
            )
        except ValueError as exc:
            raise PromotionNotAllowedError(finding_id, str(exc)) from exc

    def _project_for_run(self, session_id: str) -> str:
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        row = self._store.get_session_index_row(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        return str(row["project_id"])
