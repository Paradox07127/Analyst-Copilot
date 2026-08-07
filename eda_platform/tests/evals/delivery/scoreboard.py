"""Delivery scoreboard: how much of what a run computed reached the reader.

Eval-0 scores what the platform concludes -- planted defects found, forbidden
claims avoided. Nothing scored the path from "the SQL returned correct rows" to
"the reader sees a correct sentence", which is where every defect of the
2026-08-04..06 rounds lived: questions killed by a guard with no retry, answers
discarded by the finding binder, a section suppressed by one surviving claim.

Needs no ground truth, so it runs on any stored session, real dataset included.
`SessionMetrics` already carries the counts; what it does not carry is *why* a
question failed, which is what distinguishes a guard that misread the SQL from a
model that could not write it.

    python eda_platform/tests/evals/delivery/scoreboard.py [--workspace DIR] [--json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_WORKSPACE = Path("eda_platform/workspace")

# Ordered: the first match wins, so specific markers precede generic ones.
_FAILURE_MARKERS: tuple[tuple[str, str], ...] = (
    ("no declared required_relations", "guard:join_scope"),
    ("not a confirmed join", "guard:join_whitelist"),
    ("produced invalid sql after retry", "planner:unrepaired"),
    ("llm client is required", "config:offline"),
    ("did not include sql_template", "template:missing_sql"),
)


def classify_failure(payload: dict[str, Any]) -> str:
    """Why one question did not reach the reader, as a pipeline-stage label."""
    code = payload.get("abstention_code")
    if code:
        return f"abstain:{code}"
    error = str(payload.get("error") or "").lower()
    for marker, label in _FAILURE_MARKERS:
        if marker in error:
            return label
    exception, separator, _ = error.partition(":")
    if separator and exception and " " not in exception:
        return f"runtime:{exception}"
    return "unclassified"


@dataclass
class DeliveryScore:
    project: str
    session: str
    status: str | None = None
    # Funnel: every selected question, what became of it, and what got printed.
    selected: int = 0
    answered: int = 0
    findings: int = 0
    claims: int = 0
    failures: dict[str, int] = field(default_factory=dict)
    # Sections whose skeleton was published with nothing under it.
    empty_sections: list[str] = field(default_factory=list)
    claims_without_evidence: int = 0
    interpretation_validated: int = 0
    interpretation_fallbacks: int = 0
    gate_verdict: str | None = None
    validation_attempts: int = 0
    report_eligible_findings: int | None = None

    @property
    def answer_rate(self) -> float:
        return self.answered / self.selected if self.selected else 0.0


def _artifacts(session_dir: Path) -> list[dict[str, Any]]:
    found = []
    for path in sorted((session_dir / "artifacts").glob("*.json")):
        try:
            found.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return found


def _validation_attempts(session_dir: Path) -> int:
    trace = session_dir / "trace.jsonl"
    if not trace.exists():
        return 0
    attempts = 0
    with trace.open(encoding="utf-8") as handle:
        for line in handle:
            if '"report_validation"' not in line:
                continue
            try:
                attempts = max(attempts, int(json.loads(line)["summary"].get("attempt", 0)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return attempts


def score_session(session_dir: Path, *, project: str) -> DeliveryScore:
    score = DeliveryScore(project=project, session=session_dir.name)
    failures: Counter[str] = Counter()
    for artifact in _artifacts(session_dir):
        payload = artifact.get("payload") or {}
        kind = artifact.get("type")
        if kind == "QuestionExecutionResult":
            score.selected += 1
            if payload.get("status") == "succeeded":
                score.answered += 1
                score.findings += len(payload.get("findings") or [])
            else:
                failures[classify_failure(payload)] += 1
        elif kind == "ReportBundle":
            score.status = payload.get("status")
            for section in payload.get("sections") or []:
                claims = section.get("claims") or []
                score.claims += len(claims)
                score.claims_without_evidence += sum(
                    1 for claim in claims if not claim.get("evidence")
                )
                if not claims:
                    score.empty_sections.append(str(section.get("title")))
        elif kind == "SessionMetrics":
            score.interpretation_validated = payload.get("interpretation_validated", 0)
            score.interpretation_fallbacks = payload.get("interpretation_fallbacks", 0)
            score.gate_verdict = payload.get("report_gate_verdict")
            score.report_eligible_findings = payload.get("report_eligible_findings")
    score.failures = dict(failures.most_common())
    score.validation_attempts = _validation_attempts(session_dir)
    return score


def score_workspace(workspace: Path, *, min_questions: int = 1) -> list[DeliveryScore]:
    scores = []
    for session_dir in sorted(workspace.glob("projects/*/sessions/*")):
        if not session_dir.is_dir():
            continue
        score = score_session(session_dir, project=session_dir.parent.parent.name)
        if score.selected >= min_questions:
            scores.append(score)
    return scores


def format_table(scores: list[DeliveryScore]) -> str:
    header = (
        f"{'project':<16}{'session':<14}{'sel':>4}{'ans':>4}{'rate':>7}"
        f"{'find':>6}{'clm':>5}{'noev':>6}{'empty':>7}{'fb':>4}  gate"
    )
    lines = [header, "-" * len(header)]
    for score in scores:
        lines.append(
            f"{score.project[:15]:<16}{score.session[-13:]:<14}"
            f"{score.selected:>4}{score.answered:>4}{score.answer_rate:>7.0%}"
            f"{score.findings:>6}{score.claims:>5}{score.claims_without_evidence:>6}"
            f"{len(score.empty_sections):>7}{score.interpretation_fallbacks:>4}"
            f"  {score.gate_verdict or '-'}"
        )
    totals = Counter()
    for score in scores:
        totals.update(score.failures)
    selected = sum(score.selected for score in scores)
    answered = sum(score.answered for score in scores)
    lines.append("")
    lines.append(f"selected {selected}, answered {answered}")
    lines.append("questions that did not reach the reader:")
    for label, count in totals.most_common():
        lines.append(f"  {count:>3}  {label}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--min-questions", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="emit records instead of a table")
    args = parser.parse_args()

    scores = score_workspace(args.workspace, min_questions=args.min_questions)
    if args.json:
        print(json.dumps([asdict(score) for score in scores], indent=1))
    else:
        print(format_table(scores))


if __name__ == "__main__":
    main()
