"""Analysis macro-loop artifacts (L1 foundations, design doc 2026-07-23 §2/§5.2).

The follow-up generator emits FollowUpProposalSet (LLM may explicitly conclude,
the InsightPilot bottom-symbol pattern); LoopLedger is the append-only run-level
state that drives dedup, keep/discard, and termination. Persistence is owned by
the L2/L3 orchestrator — these are pure data models.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Distinct from the di5 micro-loop's LoopExitReason (schemas/deep_investigation.py).
MacroLoopExitReason = Literal[
    "continue",
    "concluded",
    "no_new_information",
    "round_cap",
    "budget_cap",
    "crash",
]

RoundDisposition = Literal["keep", "discard", "crash"]


class LoopDepthProfile(BaseModel):
    """Macro-loop budget for one analysis depth tier (design §3)."""

    rounds: int = Field(ge=0)
    per_round_questions: int = Field(ge=0)


# Depth 0 = single-pass pipeline, depth 1 = di5 micro-loop only: no macro rounds.
# Round/question quotas are the design §3 estimates; recalibrate against live runs
# when depth >=2 ships (design §7.1).
DEPTH_PROFILES: dict[int, LoopDepthProfile] = {
    0: LoopDepthProfile(rounds=0, per_round_questions=0),
    1: LoopDepthProfile(rounds=0, per_round_questions=0),
    2: LoopDepthProfile(rounds=1, per_round_questions=4),
    3: LoopDepthProfile(rounds=3, per_round_questions=4),
}


class FollowUpProposal(BaseModel):
    question_text: str
    rationale: str = ""
    parent_finding_id: str = ""
    priority_hint: Literal["high", "medium", "low"] = "medium"


class FollowUpProposalSet(BaseModel):
    schema_version: int = 1
    round_id: int
    concluded: bool = False  # LLM explicitly chose the bottom symbol
    conclusion_reason: str = ""
    proposals: list[FollowUpProposal] = Field(default_factory=list)


class LoopRoundRecord(BaseModel):
    """One row of the round ledger (results.tsv isomorph, design §5.2)."""

    round_id: int
    new_validated_findings: int = 0
    redundant_findings: int = 0
    # Results rejected by the validation bridge (design §8.2), as opposed to
    # redundant_findings which counts fingerprint duplicates.
    discarded_findings: int = 0
    executed_questions: int = 0
    tokens: int = 0
    exit_reason: MacroLoopExitReason = "continue"
    disposition: RoundDisposition = "discard"


class LoopLedger(BaseModel):
    schema_version: int = 1
    depth: int = 0
    finding_fingerprints: list[str] = Field(default_factory=list)
    question_fingerprints: list[str] = Field(default_factory=list)
    validated_finding_ids: list[str] = Field(default_factory=list)
    rounds: list[LoopRoundRecord] = Field(default_factory=list)
