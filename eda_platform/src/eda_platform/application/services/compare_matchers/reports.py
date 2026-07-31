from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .canonical import (
    canonical_evidence_signature,
    canonical_json,
    normalize_text,
    versioned_key,
)
from .generic import (
    MatchConfidence,
    MatchResult,
    MatchStatus,
    RuleMatch,
    deterministic_one_to_one_match,
)

REPORT_SECTION_MATCHER_VERSION = "report-section-v1"
REPORT_CLAIM_MATCHER_VERSION = "report-claim-v1"


@dataclass(frozen=True)
class ReportSectionComparable:
    record_id: str
    title: str
    required_key: str = ""
    """The required-section slot this fills, empty for a free-form section.

    Sections are identified by title everywhere else in the codebase, which
    makes any title edit look like one section removed and another added. The
    slot is the identity the plan's tier 1 asks for; the title is tier 2.
    """
    comparison_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReportClaimComparable:
    record_id: str
    text: str
    section_title: str = ""
    claim_id: str = ""
    evidence_signature: tuple[tuple[str, str], ...] = ()
    comparison_payload: dict[str, Any] = field(default_factory=dict)


def match_report_sections(
    left: list[ReportSectionComparable],
    right: list[ReportSectionComparable],
) -> list[MatchResult[ReportSectionComparable]]:
    return deterministic_one_to_one_match(
        left,
        right,
        identity=lambda section: section.record_id,
        candidate=_section_candidate,
        equivalent=lambda first, second: canonical_json(_section_content(first))
        == canonical_json(_section_content(second)),
        version=REPORT_SECTION_MATCHER_VERSION,
    )


def match_report_claims(
    left: list[ReportClaimComparable],
    right: list[ReportClaimComparable],
) -> list[MatchResult[ReportClaimComparable]]:
    return deterministic_one_to_one_match(
        left,
        right,
        identity=lambda claim: claim.record_id,
        candidate=_claim_candidate,
        equivalent=lambda first, second: canonical_json(_claim_content(first))
        == canonical_json(_claim_content(second)),
        version=REPORT_CLAIM_MATCHER_VERSION,
    )


def _section_candidate(
    left: ReportSectionComparable,
    right: ReportSectionComparable,
) -> RuleMatch | None:
    required = left.required_key.strip()
    other_required = right.required_key.strip()
    if required and other_required:
        # Both sides name a slot, so the slot decides and the title never gets
        # a say: two different required sections that happen to share wording
        # are two sections, not one that changed.
        if required != other_required:
            return None
        return RuleMatch(
            priority=0,
            score=1.0,
            match_key=versioned_key("report-section-required", required),
            reason="same required report section",
            confidence=MatchConfidence.HIGH,
            status=MatchStatus.STRONG,
        )
    title = normalize_text(left.title)
    if title and title == normalize_text(right.title):
        return RuleMatch(
            priority=1,
            score=1.0,
            match_key=versioned_key("report-section-title", title),
            reason="same normalized report section title",
            confidence=MatchConfidence.HIGH,
            status=MatchStatus.STRONG,
        )
    return None


def _claim_candidate(
    left: ReportClaimComparable,
    right: ReportClaimComparable,
) -> RuleMatch | None:
    section_matches = normalize_text(left.section_title) == normalize_text(
        right.section_title
    )
    if (
        section_matches
        and left.claim_id
        and left.claim_id == right.claim_id
    ):
        return RuleMatch(
            priority=0,
            score=1.0,
            match_key=versioned_key(
                "report-claim-id",
                (normalize_text(left.section_title), left.claim_id),
            ),
            reason="same claim identity in matched report section",
            confidence=MatchConfidence.EXACT,
            status=MatchStatus.EXACT,
        )

    evidence = canonical_evidence_signature(left.evidence_signature)
    normalized_text = normalize_text(left.text)
    if (
        section_matches
        and evidence
        and evidence == canonical_evidence_signature(right.evidence_signature)
        and normalized_text
        and normalized_text == normalize_text(right.text)
    ):
        return RuleMatch(
            priority=1,
            score=1.0,
            match_key=versioned_key(
                "report-claim-evidence",
                (normalize_text(left.section_title), evidence, normalized_text),
            ),
            reason="same normalized claim text and evidence signature",
            confidence=MatchConfidence.HIGH,
            status=MatchStatus.STRONG,
        )
    return None


def _section_content(section: ReportSectionComparable) -> object:
    return {
        "title": normalize_text(section.title),
        "payload": section.comparison_payload,
    }


def _claim_content(claim: ReportClaimComparable) -> object:
    return {
        "text": claim.text,
        "section_title": normalize_text(claim.section_title),
        "evidence_signature": canonical_evidence_signature(claim.evidence_signature),
        "payload": claim.comparison_payload,
    }
