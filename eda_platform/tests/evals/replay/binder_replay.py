"""Replay every stored SQL result through the current finding binder.

The binder turns rows into the sentence a reader actually gets, it is fully
deterministic, and it is where the 2026-08-04..06 defects lived: a label built
from the wrong column, a ranking read off a group size, an unranked result
discarded, a timestamp that had to parse as a number. None of that needs an LLM
to reproduce -- only the rows, which every run already persisted.

So the regression corpus is the run history itself, and it grows for free. Each
stored `(QuestionCandidate, SqlResult)` goes back through `_findings`; the text
that comes out is compared against the text that was published at the time.

A diff is not a failure. Changing the binder is supposed to change sentences;
the point is that no sentence changes without someone reading it. Review the
diff, then freeze it with --write-golden so the next change shows only itself.

    python eda_platform/tests/evals/replay/binder_replay.py [--workspace DIR]
    python eda_platform/tests/evals/replay/binder_replay.py --write-golden
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from os.path import commonprefix
from pathlib import Path
from typing import Any

from eda_platform.drivers.question_exec import _findings
from eda_platform.schemas.artifacts import Artifact
from eda_platform.schemas.questions import QuestionCandidate

DEFAULT_WORKSPACE = Path("eda_platform/workspace")
# Keyed by local session id, so the golden belongs beside the other run-derived
# outputs rather than in the tree. It moves in once a shared corpus is frozen.
DEFAULT_GOLDEN = Path("output/replay/binder_golden.json")

# The agent route writes its own answer text when the binder finds nothing, so
# its findings are not a binder output and replaying them proves nothing.
_REPLAYABLE_MODE = "pipeline"


@dataclass(frozen=True)
class ReplayCase:
    session: str
    question_id: str
    question: str
    stored: list[str]
    replayed: list[str]
    error: str | None = None

    @property
    def verdict(self) -> str:
        if self.error is not None:
            return "error"
        if self.stored == self.replayed:
            return "same"
        if not self.replayed:
            return "vanished"
        if not self.stored:
            return "appeared"
        return "changed"

    def key(self) -> str:
        return f"{self.session}:{self.question_id}"


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _candidates(session_dir: Path) -> dict[str, QuestionCandidate]:
    for path in sorted((session_dir / "artifacts").glob("*.json")):
        record = _load(path)
        if record is None or record.get("type") != "QuestionCandidateSet":
            continue
        found = {}
        for raw in record["payload"].get("candidates") or []:
            try:
                candidate = QuestionCandidate.model_validate(raw)
            except ValueError:
                continue
            found[candidate.question_id] = candidate
        return found
    return {}


def replay_session(session_dir: Path) -> list[ReplayCase]:
    candidates = _candidates(session_dir)
    if not candidates:
        return []
    cases = []
    for path in sorted((session_dir / "artifacts").glob("qexec_*.json")):
        record = _load(path)
        if record is None:
            continue
        payload = record["payload"]
        candidate = candidates.get(str(payload.get("question_id")))
        sql_id = payload.get("sql_result_artifact_id")
        if (
            candidate is None
            or not sql_id
            or payload.get("status") != "succeeded"
            or payload.get("execution_mode") != _REPLAYABLE_MODE
        ):
            continue
        sql_record = _load(session_dir / "artifacts" / f"{sql_id}.json")
        if sql_record is None:
            continue
        stored = [str(item.get("text")) for item in payload.get("findings") or []]
        common = {
            "session": session_dir.name,
            "question_id": candidate.question_id,
            "question": candidate.question_en,
            "stored": stored,
        }
        try:
            artifact = Artifact.model_validate(sql_record)
            replayed = [finding.text for finding in _findings(candidate, artifact)]
        except (ValueError, TypeError, KeyError) as exc:
            cases.append(
                ReplayCase(**common, replayed=[], error=f"{type(exc).__name__}: {exc}"[:200])
            )
            continue
        cases.append(ReplayCase(**common, replayed=replayed))
    return cases


def replay_workspace(workspace: Path) -> list[ReplayCase]:
    cases = []
    for session_dir in sorted(workspace.glob("projects/*/sessions/*")):
        if session_dir.is_dir():
            cases.extend(replay_session(session_dir))
    return cases


def drift(cases: list[ReplayCase], golden: dict[str, list[str]]) -> list[ReplayCase]:
    """Cases whose replayed text differs from the last reviewed text.

    Falls back to the published text for a case the golden has never seen, so a
    fresh checkout reports the whole history rather than silently approving it.
    """
    return [
        case
        for case in cases
        if case.replayed != golden.get(case.key(), case.stored) or case.error is not None
    ]


def _from_divergence(
    stored: list[str], replayed: list[str], *, width: int
) -> tuple[list[str], list[str]]:
    """Both sides trimmed to where they start to differ.

    Every finding repeats the question before saying anything, so a fixed-width
    window off the front shows the same hundred characters twice and hides the
    change underneath them.
    """
    if len(stored) != 1 or len(replayed) != 1:
        return ([text[:width] for text in stored], [text[:width] for text in replayed])
    left, right = stored[0], replayed[0]
    start = max(0, len(commonprefix([left, right])) - 20)
    lead = "…" if start else ""
    return ([lead + left[start : start + width]], [lead + right[start : start + width]])


def format_report(cases: list[ReplayCase], drifted: list[ReplayCase], *, limit: int) -> str:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.verdict] = counts.get(case.verdict, 0) + 1
    lines = [
        f"replayed {len(cases)} stored results through the current binder",
        "  " + ", ".join(f"{name} {count}" for name, count in sorted(counts.items())),
        f"needing review: {len(drifted)}",
    ]
    for case in drifted[:limit]:
        lines.append("")
        lines.append(f"--- {case.key()}  [{case.verdict}]")
        lines.append(f"    Q: {case.question[:100]}")
        if case.error is not None:
            lines.append(f"    !! {case.error}")
        stored, replayed = _from_divergence(case.stored, case.replayed, width=150)
        for text in stored:
            lines.append(f"    - {text}")
        for text in replayed:
            lines.append(f"    + {text}")
    if len(drifted) > limit:
        lines.append("")
        lines.append(f"... {len(drifted) - limit} more, raise --limit to see them")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--write-golden",
        action="store_true",
        help="record the current text as reviewed; do this only after reading the diff",
    )
    args = parser.parse_args()

    cases = replay_workspace(args.workspace)
    golden = _load(args.golden) or {} if args.golden.exists() else {}
    drifted = drift(cases, golden)

    if args.write_golden:
        args.golden.parent.mkdir(parents=True, exist_ok=True)
        args.golden.write_text(
            json.dumps({case.key(): case.replayed for case in cases}, indent=1, sort_keys=True),
            encoding="utf-8",
        )
        print(f"froze {len(cases)} cases into {args.golden}")
        return
    print(format_report(cases, drifted, limit=args.limit))


if __name__ == "__main__":
    main()
