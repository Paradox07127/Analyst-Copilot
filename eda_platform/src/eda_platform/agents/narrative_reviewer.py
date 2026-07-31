from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from eda_platform.core.llm import LLMClient, is_offline_client
from eda_platform.schemas.artifacts import SqlResult
from eda_platform.schemas.reports import ReportBundle
from eda_platform.tools.evidence import EvidencePack

_BUSINESS_NARRATIVE_SECTIONS: tuple[str, ...] = (
    "Executive Summary",
    "Business Findings",
    "Business Recommendations",
)


@dataclass(frozen=True)
class NarrativeReviewEvent:
    section_title: str
    status: Literal["reviewed", "skipped", "reverted"]
    reason: str = ""


@dataclass(frozen=True)
class NarrativeReviewResult:
    """Outcome of a narrative review pass without mutating the input bundle."""

    bundle: ReportBundle
    events: list[NarrativeReviewEvent] = field(default_factory=list)
    llm_calls: int = 0


def review_narrative(
    bundle: ReportBundle,
    *,
    evidence_pack: EvidencePack,
    llm: LLMClient | None,
    sql_results: dict[str, SqlResult] | None = None,
    target_sections: Iterable[str] | None = None,
) -> NarrativeReviewResult:
    """Keep section bodies structural; factual prose belongs in typed claims."""
    if llm is None or is_offline_client(llm):
        return NarrativeReviewResult(bundle=bundle, events=[], llm_calls=0)

    _ = evidence_pack, sql_results
    targets = (
        set(target_sections)
        if target_sections is not None
        else set(_BUSINESS_NARRATIVE_SECTIONS)
    )
    working = bundle.model_copy(deep=True)
    events = [
        NarrativeReviewEvent(
            section.title,
            "skipped",
            "section body is structural; factual prose belongs in typed claims",
        )
        for section in working.sections
        if section.title in targets and section.body
    ]
    return NarrativeReviewResult(bundle=working, events=events, llm_calls=0)
