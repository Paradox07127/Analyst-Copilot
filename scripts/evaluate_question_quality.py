#!/usr/bin/env python
"""Export blind question-review requests or grade imported reviews; no network calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eda_platform" / "src"))

from eda_platform.tools.question_quality_eval import (  # noqa: E402
    ReviewSubmission,
    evaluate_question_reviews,
    prepare_question_reviews,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    assets = REPO_ROOT / "eda_platform/tests/evals/question_quality"
    parser.add_argument("--calibration", type=Path, default=assets / "judge_calibration.json")
    parser.add_argument("--rubric", type=Path, default=assets / "judge_rubric.md")
    parser.add_argument("--scores", type=Path, help="Imported ReviewSubmission JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    rubric = args.rubric.read_text(encoding="utf-8")
    if args.scores:
        submission = ReviewSubmission.model_validate_json(args.scores.read_text(encoding="utf-8"))
        result = evaluate_question_reviews(calibration, rubric, submission)
    else:
        result = prepare_question_reviews(calibration, rubric)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"output={args.output}")
    if args.scores:
        print(f"anchor_agreement_passed={result['anchor_agreement_passed']}")
        print("scope=anchor agreement only; no production release approval")
        return 0 if result["anchor_agreement_passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
