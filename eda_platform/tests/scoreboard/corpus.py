"""Frozen scoreboard corpus loader.

Replicates the exact loading chain the audit probes use, so scoreboard
numbers computed on these fixtures match the audited baseline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from eda_platform.agents.reporting import _extract_question_evidence, _register_question_artifacts
from eda_platform.schemas.artifacts import Artifact, SqlResult
from eda_platform.schemas.reports import ReportBundle
from eda_platform.tools.evidence import EvidencePack, build_evidence_pack

CORPUS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "scoreboard_corpus"


@dataclass
class CorpusRun:
    slug: str
    bundle: ReportBundle
    pack: EvidencePack
    sql_results: dict[str, SqlResult]
    artifacts: list[Artifact]


def _load_run(session_dir: Path) -> CorpusRun:
    artifacts: list[Artifact] = []
    bundle_path: Path | None = None
    for path in sorted((session_dir / "artifacts").glob("*.json")):
        if path.name.startswith("bundle_"):
            bundle_path = path
        try:
            artifacts.append(Artifact.model_validate(json.loads(path.read_text())))
        except Exception:
            continue
    if bundle_path is None:
        raise FileNotFoundError(f"no bundle_*.json in {session_dir}")

    pack = build_evidence_pack(artifacts, payload_policy="schema+aggregates")
    question_results, sql_results = _extract_question_evidence(artifacts)
    _register_question_artifacts(pack, question_results, sql_results)
    bundle = ReportBundle.model_validate(json.loads(bundle_path.read_text())["payload"])
    return CorpusRun(
        slug=session_dir.name,
        bundle=bundle,
        pack=pack,
        sql_results=sql_results,
        artifacts=artifacts,
    )


def load_corpus() -> list[CorpusRun]:
    run_dirs = sorted(p for p in CORPUS_DIR.iterdir() if p.is_dir())
    return [_load_run(session_dir) for session_dir in run_dirs]
