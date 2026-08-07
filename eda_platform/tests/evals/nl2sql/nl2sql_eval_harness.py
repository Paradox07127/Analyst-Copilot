"""Execution-accuracy harness for the NL2SQL golden set (M4 eval week / M6.1 T3).

Implements the community-standard *execution accuracy* paradigm (Spider /
BIRD): the candidate SQL and the golden SQL are each executed through the
platform's read-only DuckDB runner and the two result sets are compared for
semantic equivalence:

- row order insensitive (multiset of rows);
- column order and column *name* insensitive (name-aligned fast path, then a
  signature-pruned column-permutation search, so aliases and reordered SELECT
  lists still match);
- float tolerance via quantisation (default ``1e-6``);
- NULL semantics: SQL NULL (surfaced as ``None``/``NaN``/``NaT``) equals NULL
  and differs from every non-NULL value.

The SQL-generation side is injected as a ``SqlProvider`` callable so one
runner serves three modes:

1. offline self-check (``golden_sql_provider``: golden vs golden must be 100%);
2. perturbation / discriminative-power tests (mutated SQL must fail);
3. live LLM scoring (``chat_sql_provider`` drives the real M3 chat path:
   intent routing -> plan -> SQL), gated behind ``EDA_LIVE_LLM_TEST=1``.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Hashable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from eda_platform.agents.planner import build_plan
from eda_platform.core.llm import StructuredLLM
from eda_platform.drivers.chat import build_value_context, run_chat_turn
from eda_platform.drivers.question_exec import execute_question_candidate
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.questions import QuestionCandidate, QuestionScore
from eda_platform.tools.loader import LoadedDataset, load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.sql_runner import SqlCatalog, build_catalog

_EVAL_DIR = Path(__file__).parent
DEFAULT_GOLDEN_PATH = _EVAL_DIR / "golden_nl2sql.json"
DEFAULT_DATA_DIR = _EVAL_DIR.parents[1] / "golden" / "data"

# Beyond this many columns the permutation search is skipped (name/identity
# alignment only); golden results are all far below the cap.
_MAX_PERMUTATION_COLUMNS = 10
_MAX_MAPPINGS_TRIED = 5_000


@dataclass(frozen=True)
class GoldenNL2SQLCase:
    case_id: str
    language: str
    category: str
    question: str
    golden_sql: str
    datasets: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class EquivalenceReport:
    equivalent: bool
    reason: str


@dataclass(frozen=True)
class CaseOutcome:
    case_id: str
    passed: bool
    reason: str
    golden_sql: str
    candidate_sql: str | None
    error: str | None = None


@dataclass(frozen=True)
class ExecutionAccuracyResult:
    outcomes: list[CaseOutcome] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def passed_count(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.passed)

    @property
    def accuracy(self) -> float:
        return self.passed_count / self.total if self.outcomes else 0.0

    @property
    def failed(self) -> list[CaseOutcome]:
        return [outcome for outcome in self.outcomes if not outcome.passed]

    def to_json_payload(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed_count,
            "execution_accuracy": round(self.accuracy, 4),
            "outcomes": [
                {
                    "case_id": outcome.case_id,
                    "passed": outcome.passed,
                    "reason": outcome.reason,
                    "golden_sql": outcome.golden_sql,
                    "candidate_sql": outcome.candidate_sql,
                    "error": outcome.error,
                }
                for outcome in self.outcomes
            ],
        }

    def summary_table(self) -> str:
        lines = [f"{'case':<14} {'ok':<4} reason"]
        for outcome in self.outcomes:
            lines.append(
                f"{outcome.case_id:<14} {'yes' if outcome.passed else 'NO':<4} {outcome.reason}"
            )
        lines.append(
            f"execution accuracy: {self.passed_count}/{self.total}"
            f" = {self.accuracy:.1%}"
        )
        return "\n".join(lines)


SqlProvider = Callable[[GoldenNL2SQLCase, Sequence[LoadedDataset]], str]


def load_golden_cases(path: Path | str | None = None) -> list[GoldenNL2SQLCase]:
    golden_path = Path(path) if path is not None else DEFAULT_GOLDEN_PATH
    raw = json.loads(golden_path.read_text(encoding="utf-8"))
    cases = [
        GoldenNL2SQLCase(
            case_id=str(item["id"]),
            language=str(item["language"]),
            category=str(item["category"]),
            question=str(item["question"]),
            golden_sql=str(item["golden_sql"]),
            datasets=tuple(str(name) for name in item["datasets"]),
            notes=str(item.get("notes", "")),
        )
        for item in raw["cases"]
    ]
    return cases


def build_case_catalog(
    dataset_names: Sequence[str],
    *,
    data_dir: Path | str | None = None,
) -> tuple[SqlCatalog, list[LoadedDataset]]:
    """Load the referenced golden CSVs and register them exactly like the chat
    path does (table name = filename without extension)."""
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    loaded = [
        load_csv(directory / f"{name}.csv", dataset_id=f"ds_{name}")
        for name in dataset_names
    ]
    return build_catalog(loaded), loaded


def execute_readonly(catalog: SqlCatalog, sql: str) -> pd.DataFrame:
    """Run one SELECT through the guarded engine (raises on non-read-only SQL)."""
    return catalog.engine.execute_select(sql)


def golden_sql_provider(case: GoldenNL2SQLCase, datasets: Sequence[LoadedDataset]) -> str:
    """Self-check provider: the golden SQL is its own candidate."""
    del datasets
    return case.golden_sql


def question_sql_provider(llm: StructuredLLM) -> SqlProvider:
    """Candidate SQL from the auto-EDA question path, on the inputs it really gets.

    Both paths call the same `build_plan`, but not with the same context: chat
    passes a masked value profile and the project's semantic seeds, and this one
    passes neither -- `execute_question_candidate` takes no payload policy, so it
    cannot decide whether column values may be sent, and auto-EDA's call site
    omits the seeds its own batch path forwards.

    Scoring both providers on one question set turns that difference into a
    measured number. The call below mirrors `auto_eda.ExecuteTopQuestionsStep`
    argument for argument; adding anything here would measure a pipeline that
    does not exist.
    """

    def provider(case: GoldenNL2SQLCase, datasets: Sequence[LoadedDataset]) -> str:
        candidate = QuestionCandidate(
            question_id=f"q_{case.case_id}",
            question_en=case.question,
            origin="llm",
            target_datasets=[dataset.record.name for dataset in datasets],
            score=QuestionScore(
                data_availability=1.0,
                statistical_signal=0.5,
                quality_risk=0.0,
                join_risk=0.0,
                deterministic_score=0.6,
            ),
        )
        artifacts = execute_question_candidate(
            candidate,
            datasets=datasets,
            project_id="nl2sql_eval",
            session_id=f"eval_{case.case_id}",
            parent_ids=[],
            llm=llm,
            confirmed_joins=[],
        )
        for artifact in artifacts:
            if artifact.type is ArtifactType.SQL_RESULT:
                return str(artifact.payload["sql"])
        qexec = next(
            (a for a in artifacts if a.type is ArtifactType.QUESTION_EXECUTION_RESULT), None
        )
        reason = str(qexec.payload.get("error") if qexec is not None else "no artifact")
        raise RuntimeError(f"question path produced no SQL for {case.case_id}: {reason[:200]}")

    return provider


def plan_sql_provider(
    llm: StructuredLLM, *, with_value_context: bool = False
) -> SqlProvider:
    """Candidate SQL straight from the planner, with or without value hints.

    The two variants differ in exactly one argument, which is the one the two
    product surfaces disagree on: chat passes a masked top-5 value profile per
    column, and the auto-EDA question path passes nothing, because
    `execute_question_candidate` takes no payload policy and so cannot decide
    whether values may leave the process. Scoring both says what that argument
    is worth instead of arguing about it.
    """

    def provider(case: GoldenNL2SQLCase, datasets: Sequence[LoadedDataset]) -> str:
        catalog = build_catalog(datasets)
        columns = {
            catalog.relations[dataset.record.name]: {
                str(column) for column in dataset.frame.columns
            }
            for dataset in datasets
        }
        value_context = None
        if with_value_context:
            profiles = [
                profile_dataset(dataset, project_id="nl2sql_eval", session_id=case.case_id)
                for dataset in datasets
            ]
            value_context = build_value_context(
                datasets,
                profiles,
                catalog.relations,
                project_id="nl2sql_eval",
                session_id=case.case_id,
            )
        plan = build_plan(
            case.question,
            llm=llm,
            catalog_columns=columns,
            value_context=value_context,
            engine=catalog.engine,
        )
        return plan.sql

    return provider


def chat_sql_provider(llm: StructuredLLM) -> SqlProvider:
    """Candidate SQL from the real chat path (M3 intent -> plan -> SQL chain).

    Kept for the offline discriminative-power tests. Not used for live scoring:
    chat now explores before it answers -- three to ten statements per question,
    with the answering one not reliably last -- and execution accuracy needs a
    single candidate query to compare (2026-08-06).

    This is the live-LLM integration point: pass any client satisfying the
    ``StructuredLLM`` protocol (``create_llm_client(load_llm_settings_from_env_file())``
    for a real provider, or a scripted fake in tests).
    """

    def provider(case: GoldenNL2SQLCase, datasets: Sequence[LoadedDataset]) -> str:
        result = run_chat_turn(
            case.question,
            datasets=datasets,
            project_id="nl2sql_eval",
            session_id=f"eval_{case.case_id}",
            llm=llm,
        )
        if not result.sql:
            raise RuntimeError(
                f"chat path produced no SQL for {case.case_id}: {result.message[:200]}"
            )
        return result.sql

    return provider


def run_execution_accuracy(
    cases: Sequence[GoldenNL2SQLCase],
    sql_provider: SqlProvider,
    *,
    data_dir: Path | str | None = None,
    float_tol: float = 1e-6,
) -> ExecutionAccuracyResult:
    """Execute candidate vs golden SQL per case and score result-set equivalence.

    Candidate-side failures (provider errors, unsafe SQL, binding errors) fail
    the case but never abort the run — an eval must keep scoring.
    """
    outcomes: list[CaseOutcome] = []
    for case in cases:
        catalog, loaded = build_case_catalog(case.datasets, data_dir=data_dir)
        golden_frame = execute_readonly(catalog, case.golden_sql)
        candidate_sql: str | None = None
        try:
            candidate_sql = sql_provider(case, loaded)
            candidate_frame = execute_readonly(catalog, candidate_sql)
        except Exception as exc:  # noqa: BLE001 - eval must record, not crash
            outcomes.append(
                CaseOutcome(
                    case_id=case.case_id,
                    passed=False,
                    reason="candidate generation/execution failed",
                    golden_sql=case.golden_sql,
                    candidate_sql=candidate_sql,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        report = results_equivalent(golden_frame, candidate_frame, float_tol=float_tol)
        outcomes.append(
            CaseOutcome(
                case_id=case.case_id,
                passed=report.equivalent,
                reason=report.reason,
                golden_sql=case.golden_sql,
                candidate_sql=candidate_sql,
            )
        )
    return ExecutionAccuracyResult(outcomes=outcomes)


def results_equivalent(
    golden: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    float_tol: float = 1e-6,
) -> EquivalenceReport:
    """Decide whether two result sets denote the same answer.

    Order of checks: shape -> name-aligned fast path -> column-permutation
    search (per-column value-multiset signatures prune the search space, and
    every complete mapping is verified against the full row multiset).
    """
    if len(golden) != len(candidate):
        return EquivalenceReport(
            False, f"row count differs: golden={len(golden)} candidate={len(candidate)}"
        )
    if len(golden.columns) != len(candidate.columns):
        return EquivalenceReport(
            False,
            "column count differs: "
            f"golden={len(golden.columns)} candidate={len(candidate.columns)}",
        )
    ndigits = _tolerance_digits(float_tol)
    golden_rows = _normalized_rows(golden, ndigits)
    candidate_rows = _normalized_rows(candidate, ndigits)
    column_count = len(golden.columns)
    if len(golden) == 0:
        return EquivalenceReport(True, "both results empty with matching column count")

    identity = tuple(range(column_count))
    if _rows_match_under_mapping(golden_rows, candidate_rows, identity):
        return EquivalenceReport(True, "row multisets match in given column order")

    name_mapping = _name_based_mapping(list(golden.columns), list(candidate.columns))
    if name_mapping is not None and _rows_match_under_mapping(
        golden_rows, candidate_rows, name_mapping
    ):
        return EquivalenceReport(True, "row multisets match after column-name alignment")

    if column_count > _MAX_PERMUTATION_COLUMNS:
        return EquivalenceReport(
            False,
            f"no match by order/name and column count {column_count} exceeds "
            f"permutation-search cap {_MAX_PERMUTATION_COLUMNS}",
        )
    for mapping in _candidate_mappings(golden_rows, candidate_rows, column_count):
        if _rows_match_under_mapping(golden_rows, candidate_rows, mapping):
            return EquivalenceReport(
                True, "row multisets match under a column permutation/renaming"
            )
    return EquivalenceReport(False, _mismatch_detail(golden_rows, candidate_rows))


# --- normalisation -----------------------------------------------------------


def _tolerance_digits(float_tol: float) -> int:
    if float_tol <= 0:
        return 12
    return max(0, min(12, round(-math.log10(float_tol))))


def _normalized_rows(frame: pd.DataFrame, ndigits: int) -> list[tuple[Hashable, ...]]:
    records = frame.to_dict("records")
    columns = list(frame.columns)
    return [
        tuple(_normalize_cell(record[column], ndigits) for column in columns)
        for record in records
    ]


def _normalize_cell(value: Any, ndigits: int) -> Hashable:
    if value is None:
        return None
    if isinstance(value, (list, dict, set, tuple)):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp | datetime):
        if (value.hour, value.minute, value.second, value.microsecond) == (0, 0, 0, 0):
            return value.date().isoformat()
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    item_method = getattr(value, "item", None)  # numpy scalar -> python scalar
    if callable(item_method):
        try:
            value = item_method()
        except (TypeError, ValueError):
            pass
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float | Decimal):
        numeric = float(value)
        if math.isnan(numeric):
            return None
        return round(numeric, ndigits)
    if isinstance(value, str):
        return value
    return str(value)


# --- column alignment --------------------------------------------------------


def _rows_match_under_mapping(
    golden_rows: list[tuple[Hashable, ...]],
    candidate_rows: list[tuple[Hashable, ...]],
    mapping: tuple[int, ...],
) -> bool:
    remapped = Counter(
        tuple(row[mapping[index]] for index in range(len(mapping)))
        for row in candidate_rows
    )
    return remapped == Counter(golden_rows)


def _name_based_mapping(
    golden_columns: Sequence[Any],
    candidate_columns: Sequence[Any],
) -> tuple[int, ...] | None:
    golden_names = [str(name).strip().lower() for name in golden_columns]
    candidate_names = [str(name).strip().lower() for name in candidate_columns]
    if sorted(golden_names) != sorted(candidate_names):
        return None
    if len(set(candidate_names)) != len(candidate_names):
        return None  # duplicate names: fall through to permutation search
    positions = {name: index for index, name in enumerate(candidate_names)}
    return tuple(positions[name] for name in golden_names)


def _candidate_mappings(
    golden_rows: list[tuple[Hashable, ...]],
    candidate_rows: list[tuple[Hashable, ...]],
    column_count: int,
) -> Iterator[tuple[int, ...]]:
    """Yield column bijections whose per-column value multisets agree.

    Signature pruning keeps this tiny in practice; a hard cap bounds worst
    cases with many identical columns.
    """
    golden_signatures = [_column_signature(golden_rows, i) for i in range(column_count)]
    candidate_signatures = [
        _column_signature(candidate_rows, i) for i in range(column_count)
    ]
    compatible: list[list[int]] = [
        [
            candidate_index
            for candidate_index in range(column_count)
            if candidate_signatures[candidate_index] == golden_signatures[golden_index]
        ]
        for golden_index in range(column_count)
    ]
    order = sorted(range(column_count), key=lambda index: len(compatible[index]))
    assignment: dict[int, int] = {}
    used: set[int] = set()
    tried = 0

    def backtrack(position: int) -> Iterator[tuple[int, ...]]:
        nonlocal tried
        if tried >= _MAX_MAPPINGS_TRIED:
            return
        if position == column_count:
            tried += 1
            yield tuple(assignment[index] for index in range(column_count))
            return
        golden_index = order[position]
        for candidate_index in compatible[golden_index]:
            if candidate_index in used:
                continue
            assignment[golden_index] = candidate_index
            used.add(candidate_index)
            yield from backtrack(position + 1)
            used.discard(candidate_index)
            del assignment[golden_index]

    yield from backtrack(0)


def _column_signature(
    rows: list[tuple[Hashable, ...]], column_index: int
) -> tuple[tuple[Hashable, int], ...]:
    counted = Counter(row[column_index] for row in rows)
    return tuple(sorted(counted.items(), key=lambda item: (str(type(item[0])), str(item[0]))))


def _mismatch_detail(
    golden_rows: list[tuple[Hashable, ...]],
    candidate_rows: list[tuple[Hashable, ...]],
) -> str:
    golden_counter = Counter(golden_rows)
    candidate_counter = Counter(candidate_rows)
    only_golden = list((golden_counter - candidate_counter).elements())[:3]
    only_candidate = list((candidate_counter - golden_counter).elements())[:3]
    return (
        "no column mapping equates the row multisets; "
        f"sample rows only in golden: {only_golden}; "
        f"sample rows only in candidate: {only_candidate}"
    )
