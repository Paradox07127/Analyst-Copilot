"""Conservative extractive composition over report claims that passed the gates.

The model selects and orders whole claims. Exact source binding preserves each
claim's entity, quantity, scope and qualifiers without pretending a numeric
regex can establish arbitrary paraphrase entailment. Invalid passages leave the
original claims visible; richer synthesis requires a typed conclusion contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from eda_platform.agents.report_graph import (
    ReportWorkflow,
    model_binding,
    raise_recorded_error,
    run_report_workflow,
)
from eda_platform.core.cancellation import CancellationError
from eda_platform.core.graph_execution import (
    GraphEffectUncertain,
    GraphIdentityError,
    GraphPersistence,
)
from eda_platform.core.llm import LLMClient, LLMResultMetadata, is_offline_client
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
    "Compose a short evidence-preserving passage by selecting and ordering complete "
    "claim texts supplied below. Copy each selected claim verbatim, including its "
    "qualifiers, units, scope and limitations; separate claims with a space. "
    "Do not paraphrase, split a claim, add transitions, change numbers, or introduce "
    "an explanation, comparison, recommendation or causal relationship. "
    "Use only claims relevant to this section and include every selected claim id "
    "in cited_claim_ids, in the same order. Do not cite claims you did not copy. "
    "Return empty text and no ids if no concise passage fits these constraints. "
    "This is a conservative extractive composition, not a new analysis."
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
    # One entry per discarded draft: {"section": title, "reason": code}. Purely
    # observability — the caller turns these into a trace event.
    discards: list[dict[str, str]] = field(default_factory=list)

    llm_calls: list[LLMResultMetadata] = field(default_factory=list)
    llm_events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def attempted(self) -> int:
        return self.written + self.rejected


def narrate_report(
    bundle: ReportBundle, *, llm: LLMClient | None,
    workflow: ReportWorkflow | None = None,
    persistence: GraphPersistence | None = None,
) -> NarrationOutcome:
    """Fill eligible sections through individually checkpointed narrative tasks."""
    if workflow is None:
        def run(active: ReportWorkflow) -> dict[str, Any]:
            result = narrate_report(bundle, llm=llm, workflow=active)
            return {
                "outcome": {**asdict(result), "llm_calls": [
                    item.model_dump(mode="json") for item in result.llm_calls
                ]},
                "narratives": [section.narrative for section in bundle.sections],
            }

        saved = run_report_workflow(
            run, persistence=persistence,
            inputs={
                "bundle": bundle.model_dump(
                    mode="json", exclude={"sections": {"__all__": {"narrative"}}},
                ),
                "model": model_binding(llm),
            },
            definition="report-narration-functional-v1",
        )
        for section, narrative in zip(bundle.sections, saved["narratives"], strict=True):
            section.narrative = narrative
        saved["outcome"]["llm_calls"] = [
            LLMResultMetadata.model_validate(item) for item in saved["outcome"]["llm_calls"]
        ]
        return NarrationOutcome(**saved["outcome"])
    if llm is None or is_offline_client(llm):
        return NarrationOutcome(skipped=len(bundle.sections))
    written = rejected = skipped = 0
    discards: list[dict[str, str]] = []
    llm_calls: list[LLMResultMetadata] = []
    llm_events: list[dict[str, Any]] = []
    for section_index, section in enumerate(bundle.sections):
        if not _is_narratable(section):
            skipped += 1
            continue
        def narrate(
            current: ReportSection = section, index: int = section_index,
        ) -> dict[str, Any]:
            records: list[dict[str, Any]] = []
            narrative, reason = _narrate_section(
                current, llm=llm, workflow=workflow,
                effect_key=f"narrative:{index}", records=records,
            )
            return {"narrative": narrative, "reason": reason, "records": records}

        saved = workflow.step("narrate_section", narrate)
        narrative, discard_reason = saved["narrative"], saved["reason"]
        for record in saved["records"]:
            if record["usage"] is not None:
                llm_calls.append(LLMResultMetadata.model_validate(record["usage"]))
            error = record.get("error", {})
            llm_events.append({
                "task": _TASK, "status": "error" if error else "success",
                "attempt": section_index + 1, "operation_id": record["operation_id"],
                "started_at": record["started_at"],
                "finished_at": record["finished_at"], "usage": record["usage"],
                "error_type": error.get("type", ""), "error": error.get("message", ""),
            })
        if narrative is None:
            rejected += 1
            discards.append(
                {"section": section.title, "reason": discard_reason or "unknown"}
            )
            continue
        section.narrative = narrative
        written += 1
    return NarrationOutcome(
        written=written, rejected=rejected, skipped=skipped, discards=discards,
        llm_calls=llm_calls, llm_events=llm_events
    )


def _eligible_claims(section: ReportSection) -> list[ReportClaim]:
    return [claim for claim in section.claims if claim.id and claim.text.strip()]


def _is_narratable(section: ReportSection) -> bool:
    return len(_eligible_claims(section)) >= _MIN_CLAIMS_TO_NARRATE


def _narrate_section(
    section: ReportSection, *, llm: LLMClient,
    workflow: ReportWorkflow, effect_key: str, records: list[dict[str, Any]],
) -> tuple[str | None, str | None]:
    """The section's narrative, or (None, reason) naming why the draft died."""
    claims = _eligible_claims(section)
    shown = claims[:_MAX_CLAIMS_IN_PAYLOAD]
    payload = {
        "instructions": _INSTRUCTIONS,
        "section_title": section.title,
        "claims": [{"id": claim.id, "text": claim.text} for claim in shown],
    }
    try:
        result = workflow.structured(
            key=effect_key, llm=llm, task_name=_TASK, schema=_NarrativeDraft, payload=payload,
        )
        records.append(result)
        if result.get("error"):
            raise_recorded_error(result["error"])
        if not result["valid_shape"]:
            return None, "invalid_response_shape"
        draft = _NarrativeDraft.model_validate(result["draft"])
    except (CancellationError, GraphIdentityError, GraphEffectUncertain):
        raise
    except Exception:
        # A report that renders as bullets is the working product; a narration
        # failure must never cost the reader the section.
        return None, "llm_error"
    # A client that answers with some other shape has not answered.
    if not isinstance(draft, _NarrativeDraft):
        return None, "invalid_response_shape"
    text = " ".join(draft.text.split())
    if not text or len(text) > _MAX_NARRATIVE_CHARS:
        return None, "empty_or_oversized_text"
    by_id = {claim.id: claim for claim in shown}
    cited = list(dict.fromkeys(draft.cited_claim_ids))
    # An id the model invented means it was not reading the claims it was
    # given, which disqualifies the prose as well as the citation.
    if not cited or any(claim_id not in by_id for claim_id in cited):
        return None, "uncited_or_unknown_claim"
    cited_claims = [by_id[claim_id] for claim_id in cited]
    if not borrowed_numbers_only(text, cited_claims):
        return None, "unverifiable_figure"
    # Exact whole-claim composition binds entities, predicates, values, units,
    # scope and caveats to their cited source. This deliberately rejects valid
    # paraphrases too: number/token heuristics cannot prove arbitrary entailment.
    expected = " ".join(" ".join(claim.text.split()) for claim in cited_claims)
    if text != expected:
        return None, "unsupported_rewrite"
    # The prose is model output going into a markdown document whose headings
    # drive the table of contents and whose code spans become evidence buttons.
    # Only the citation we build ourselves is allowed to carry either.
    citation = _citation_suffix([by_id[claim_id] for claim_id in cited])
    return f"{_as_plain_paragraph(text)}{citation}", None


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
