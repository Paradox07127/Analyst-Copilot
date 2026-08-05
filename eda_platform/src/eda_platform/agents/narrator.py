"""Connective prose over claims that already passed the report's gates.

The report was a bullet list because every sentence in it has to be defensible,
and only typed claims are. This layer buys readability without giving that up:
it may reorder and join the claims it is shown, and a paragraph carrying any
figure those claims do not already state is thrown away. Citations are appended
from the claims' own evidence, so the model never authors an artifact id.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from eda_platform.core.llm import LLMClient, is_offline_client
from eda_platform.schemas.reports import ReportBundle, ReportClaim, ReportSection
from eda_platform.tools.exporter import (
    claim_number_signature,
    neutralize_markdown_inline,
)

_TASK = "report_section_narrative"
# One bullet needs no connective prose; narrating it only restates it.
_MIN_CLAIMS_TO_NARRATE = 2
_MAX_CLAIMS_IN_PAYLOAD = 12
_MAX_NARRATIVE_CHARS = 900
_MAX_CITATIONS = 6

_INSTRUCTIONS = (
    "Write two or three sentences that connect the findings below into a "
    "single passage a business reader can follow. "
    "Do NOT introduce any number, percentage, date, or magnitude that is not "
    "already written in one of the findings -- do not average them, round "
    "them, total them, or infer a rate from them. Reusing a figure verbatim is "
    "expected; producing a new one is not. "
    "Do not write artifact ids, column ids, or citations; they are attached "
    "afterwards. "
    "Say what the findings mean together and where they disagree. If they say "
    "nothing in common, say that plainly rather than manufacturing a theme. "
    "List in cited_claim_ids the ids of the findings the passage draws on."
)


class _NarrativeDraft(BaseModel):
    text: str = Field(default="")
    cited_claim_ids: list[str] = Field(default_factory=list)


def borrowed_numbers_only(prose: str, claims: Sequence[ReportClaim]) -> bool:
    """Whether every figure in ``prose`` already appears in one of ``claims``."""
    allowed: set[str] = set()
    for claim in claims:
        allowed |= claim_number_signature(claim.text)
    return claim_number_signature(prose) <= allowed


@dataclass(frozen=True, slots=True)
class NarrationOutcome:
    """A bullet-only report has three possible causes and they look identical.

    Distinguishing "nothing was asked" from "everything was refused" is the
    first question to ask of a live run, so the counts are reported, not just
    the successes.
    """

    written: int = 0
    rejected: int = 0
    skipped: int = 0

    @property
    def attempted(self) -> int:
        return self.written + self.rejected


def narrate_report(bundle: ReportBundle, *, llm: LLMClient | None) -> NarrationOutcome:
    """Fill in each eligible section's narrative."""
    if llm is None or is_offline_client(llm):
        return NarrationOutcome(skipped=len(bundle.sections))
    written = rejected = skipped = 0
    for section in bundle.sections:
        if not _is_narratable(section):
            skipped += 1
            continue
        narrative = _narrate_section(section, llm=llm)
        if narrative is None:
            rejected += 1
            continue
        section.narrative = narrative
        written += 1
    return NarrationOutcome(written=written, rejected=rejected, skipped=skipped)


def _eligible_claims(section: ReportSection) -> list[ReportClaim]:
    return [claim for claim in section.claims if claim.id and claim.text.strip()]


def _is_narratable(section: ReportSection) -> bool:
    return len(_eligible_claims(section)) >= _MIN_CLAIMS_TO_NARRATE


def _narrate_section(section: ReportSection, *, llm: LLMClient) -> str | None:
    claims = _eligible_claims(section)
    shown = claims[:_MAX_CLAIMS_IN_PAYLOAD]
    payload = {
        "instructions": _INSTRUCTIONS,
        "section_title": section.title,
        "claims": [{"id": claim.id, "text": claim.text} for claim in shown],
    }
    try:
        draft = llm.structured(task=_TASK, schema=_NarrativeDraft, payload=payload)
    except Exception:
        # A report that renders as bullets is the working product; a narration
        # failure must never cost the reader the section.
        return None
    # A client that answers with some other shape has not answered.
    if not isinstance(draft, _NarrativeDraft):
        return None
    text = " ".join(draft.text.split())
    if not text or len(text) > _MAX_NARRATIVE_CHARS:
        return None
    by_id = {claim.id: claim for claim in shown}
    cited = list(dict.fromkeys(draft.cited_claim_ids))
    # An id the model invented means it was not reading the claims it was
    # given, which disqualifies the prose as well as the citation.
    if not cited or any(claim_id not in by_id for claim_id in cited):
        return None
    if not borrowed_numbers_only(text, shown):
        return None
    # The prose is model output going into a markdown document whose headings
    # drive the table of contents and whose code spans become evidence buttons.
    # Only the citation we build ourselves is allowed to carry either.
    citation = _citation_suffix([by_id[claim_id] for claim_id in cited])
    return f"{_as_plain_paragraph(text)}{citation}"


# The prose is collapsed to one line, so only its first character can still open
# a markdown block: a heading, a quote, or a list item.
_BLOCK_STARTERS = ("#", ">", "-", "*", "+", "=", "|")


def _as_plain_paragraph(text: str) -> str:
    """Model prose that cannot open a block or forge an inline span.

    ``neutralize_markdown_inline`` covers the inline syntax; a narrative is
    rendered as its own paragraph, which the inline pass has no reason to guard.
    """
    inline_safe = neutralize_markdown_inline(text)
    if inline_safe.startswith(_BLOCK_STARTERS):
        return f"\\{inline_safe}"
    return inline_safe


def _citation_suffix(claims: Sequence[ReportClaim]) -> str:
    """Evidence ids taken from the cited claims, in the order they were cited."""
    artifact_ids: list[str] = []
    for claim in claims:
        for ref in claim.evidence:
            if ref.artifact_id and ref.artifact_id not in artifact_ids:
                artifact_ids.append(ref.artifact_id)
    if not artifact_ids:
        return ""
    shown = artifact_ids[:_MAX_CITATIONS]
    listed = ", ".join(f"`{artifact_id}`" for artifact_id in shown)
    return f" (evidence: {listed})"
