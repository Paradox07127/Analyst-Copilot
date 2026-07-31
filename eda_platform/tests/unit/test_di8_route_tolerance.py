"""DI8-A route tolerance: lenient coercion, resolve-then-reject dataset names,
repair-style retries (never blind), partial acceptance, and explicit
yellow-light degradation metrics. All LLM calls are mocked."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from eda_platform.agents.question_agent import (
    coerce_string_list,
    propose_llm_question_candidates,
    resolve_dataset_name,
)
from eda_platform.core.session_metrics import summarize_session
from eda_platform.core.store import ArtifactStore
from eda_platform.core.tool_guard import ToolGuardError
from eda_platform.schemas.artifacts import Artifact
from eda_platform.schemas.sessions import TraceEvent
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset

T = TypeVar("T", bound=BaseModel)


class FakeQuestionLLM:
    """Same mock contract as tests/unit/test_m4_questions.py: scripted
    structured() responses; the last response repeats once exhausted."""

    def __init__(self, result: BaseModel | dict[str, Any] | list[dict[str, Any]]) -> None:
        self.results = result if isinstance(result, list) else [result]
        self.calls: list[dict[str, Any]] = []
        self.call_count = 0

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append({"task": task, "schema": schema.__name__, "payload": payload})
        index = min(self.call_count, len(self.results) - 1)
        self.call_count += 1
        result = self.results[index]
        if isinstance(result, BaseModel):
            return cast(T, result)
        return schema.model_validate(result)

    def text(self, *, task: str, payload: dict) -> str:
        return "fake"

    def last_usage(self) -> None:
        return None


def _profile(tmp_path: Path, filename: str, content: str) -> Artifact:
    csv_path = tmp_path / filename
    csv_path.write_text(content, encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id=f"ds_{csv_path.stem}")
    return profile_dataset(loaded, project_id="project_demo", session_id="run_demo")


def _sales_profile(tmp_path: Path) -> Artifact:
    return _profile(tmp_path, "sales.csv", "region,revenue\nEast,10\nWest,20\n")


def _good_question(**overrides: Any) -> dict[str, Any]:
    question: dict[str, Any] = {
        "question_en": "Which region has the most revenue?",
        "target_datasets": ["sales.csv"],
        "llm_business_relevance": 0.8,
        "llm_actionability": 0.9,
    }
    question.update(overrides)
    return question


# --------------------------------------------------------------------------- #
# 1. Lenient coercion (str -> [str], None -> [])
# --------------------------------------------------------------------------- #
def test_coerce_string_list_covers_all_branches() -> None:
    assert coerce_string_list(None) == ([], True)
    assert coerce_string_list("orders.csv") == (["orders.csv"], True)
    assert coerce_string_list("  ") == ([], True)
    assert coerce_string_list(["a", "b"]) == (["a", "b"], False)
    assert coerce_string_list(("a",)) == (["a"], False)
    assert coerce_string_list(5) == (5, False)


def test_str_and_none_list_fields_are_coerced_without_retry(tmp_path: Path) -> None:
    profile = _sales_profile(tmp_path)
    llm = FakeQuestionLLM(
        {
            "questions": [
                _good_question(
                    target_datasets="sales.csv",
                    risks=None,
                    data_requirements="sales.csv: region and revenue columns",
                )
            ]
        }
    )

    result = propose_llm_question_candidates([profile], llm=llm, max_questions=1)

    assert result.error is None
    assert len(llm.calls) == 1  # coercion repairs in place, no retry burned
    candidate = result.candidates[0]
    assert candidate.target_datasets == ["sales.csv"]
    assert candidate.risks == []
    assert candidate.data_requirements == ["sales.csv: region and revenue columns"]
    assert result.coerced_list_fields == 3
    assert result.degraded is False
    assert result.dropped_proposals == 0


# --------------------------------------------------------------------------- #
# 2. resolve-then-reject dataset name resolution (pure function branches)
# --------------------------------------------------------------------------- #
def test_resolve_dataset_name_exact_and_normalized() -> None:
    known = {"orders.csv", "product_category_name_translation.csv"}
    exact = resolve_dataset_name("orders.csv", known)
    assert (exact.resolved, exact.method) == ("orders.csv", "exact")

    case_fixed = resolve_dataset_name("Orders.CSV", known)
    assert (case_fixed.resolved, case_fixed.method) == ("orders.csv", "normalized")
    assert case_fixed.auto_fixed is True

    extension_fixed = resolve_dataset_name("orders", known)
    assert (extension_fixed.resolved, extension_fixed.method) == ("orders.csv", "normalized")

    separator_fixed = resolve_dataset_name(
        "product-category-name-translation.csv", known
    )
    assert separator_fixed.resolved == "product_category_name_translation.csv"


def test_resolve_dataset_name_strips_affixes() -> None:
    known = {"product_category_name_translation.csv", "orders.csv"}
    prefixed = resolve_dataset_name(
        "olist_product_category_name_translation.csv", known
    )
    assert prefixed.resolved == "product_category_name_translation.csv"
    assert prefixed.method == "affix"

    # The model dropped a prefix that the manifest name carries.
    known_prefixed = {"olist_orders.csv", "products.csv"}
    dropped = resolve_dataset_name("orders.csv", known_prefixed)
    assert (dropped.resolved, dropped.method) == ("olist_orders.csv", "affix")


def test_resolve_dataset_name_edit_distance_only_when_unique_and_close() -> None:
    known = {"orders.csv", "product_category_name_translation.csv"}
    typo = resolve_dataset_name("ordes.csv", known)
    assert (typo.resolved, typo.method) == ("orders.csv", "edit_distance")

    far_away = resolve_dataset_name("revenue_summary.csv", known)
    assert far_away.resolved is None and far_away.method is None

    ambiguous = resolve_dataset_name("orders.csv", {"orders_2024.csv", "orders_2025.csv"})
    assert ambiguous.resolved is None  # two affix candidates -> low confidence


def test_affix_dataset_names_are_auto_resolved_and_counted(tmp_path: Path) -> None:
    profile = _profile(
        tmp_path,
        "product_category_name_translation.csv",
        "product_category_name,product_category_name_english\nbeleza,beauty\n",
    )
    llm = FakeQuestionLLM(
        {
            "questions": [
                _good_question(
                    question_en="Which categories translate inconsistently?",
                    target_datasets=["olist_product_category_name_translation.csv"],
                    dataset_display_names={
                        "olist_product_category_name_translation.csv": "Category translation"
                    },
                )
            ]
        }
    )

    result = propose_llm_question_candidates([profile], llm=llm, max_questions=1)

    assert result.error is None
    assert len(llm.calls) == 1  # resolved, not rejected -> no retry
    candidate = result.candidates[0]
    assert candidate.target_datasets == ["product_category_name_translation.csv"]
    assert candidate.dataset_display_names == {
        "product_category_name_translation.csv": "Category translation"
    }
    assert result.resolved_dataset_names == 2  # target + display-name key
    assert result.degraded is False


# --------------------------------------------------------------------------- #
# 3. Repair-style retry: violation feedback + schema example injected, and the
#    retry payload always differs from the previous call (no blind resend).
# --------------------------------------------------------------------------- #
def test_low_confidence_dataset_rejection_lists_known_datasets_in_feedback(
    tmp_path: Path,
) -> None:
    profile = _sales_profile(tmp_path)
    guard_errors: list[ToolGuardError] = []
    llm = FakeQuestionLLM(
        [
            {"questions": [_good_question(target_datasets=["revenue_summary_2024.csv"])]},
            {"questions": [_good_question()]},
        ]
    )

    result = propose_llm_question_candidates(
        [profile],
        llm=llm,
        max_questions=1,
        on_guard_rejected=guard_errors.append,
    )

    assert result.error is None
    assert len(llm.calls) == 2
    retry_payload = llm.calls[1]["payload"]
    assert "previous_error" in retry_payload
    # Rejection feedback carries the known-dataset list (resolve-then-reject).
    assert "sales.csv" in retry_payload["previous_error"]
    assert "could not be resolved" in retry_payload["previous_error"]
    # Correct-schema example rides along with the violation feedback.
    assert retry_payload["schema_example"]["questions"][0]["llm_business_relevance"] == 0.8
    assert retry_payload["repair_attempt"] == 1
    assert guard_errors and "revenue_summary_2024.csv" in guard_errors[0].to_model_feedback()


def test_every_retry_payload_differs_from_the_previous_call(tmp_path: Path) -> None:
    profile = _sales_profile(tmp_path)
    always_bad = {"questions": [_good_question(llm_business_relevance=8)]}
    llm = FakeQuestionLLM(always_bad)

    result = propose_llm_question_candidates([profile], llm=llm, max_questions=1)

    # Retry cap is 2 (3 attempts total), then the route degrades explicitly.
    assert len(llm.calls) == 3
    payloads = [call["payload"] for call in llm.calls]
    assert "previous_error" not in payloads[0]
    assert payloads[1] != payloads[0]
    assert payloads[2] != payloads[1]  # repair_attempt differs even if feedback repeats
    assert all("previous_error" in payload for payload in payloads[1:])
    assert result.error is not None and "LLM route skipped" in result.error
    assert result.degraded is True
    assert result.candidates == []


# --------------------------------------------------------------------------- #
# 4. Partial acceptance: bad entries dropped and counted, good entries kept.
# --------------------------------------------------------------------------- #
def test_partial_acceptance_keeps_good_entries_and_counts_drops(tmp_path: Path) -> None:
    profile = _sales_profile(tmp_path)
    mixed_batch = {
        "questions": [
            _good_question(),
            _good_question(
                question_en="Broken scores question?",
                llm_business_relevance=8,  # persists across all retries
            ),
        ]
    }
    llm = FakeQuestionLLM(mixed_batch)

    result = propose_llm_question_candidates([profile], llm=llm, max_questions=5)

    assert len(llm.calls) == 3  # both retries burned trying to repair
    assert result.error is None  # route delivered -> not skipped
    assert len(result.candidates) == 1
    assert result.candidates[0].question_en == "Which region has the most revenue?"
    assert result.dropped_proposals == 1
    assert result.degraded is True  # partial delivery is an explicit yellow light


# --------------------------------------------------------------------------- #
# 5. Yellow-light metrics: trace events roll up into SessionMetrics + degraded.
# --------------------------------------------------------------------------- #
def _store_with_run(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "run_demo")
    return store


def test_run_metrics_expose_skip_and_repair_counters_as_degraded(tmp_path: Path) -> None:
    store = _store_with_run(tmp_path)
    store.append_trace(
        "project_demo",
        TraceEvent(
            session_id="run_demo",
            event_type="question_llm_skipped",
            name="m4_question_discovery",
            summary={
                "error": "LLM route skipped after retry: ...",
                "proposals_dropped": 2,
                "dataset_names_resolved": 1,
                "list_coercions": 3,
                "degraded": True,
            },
        ),
    )

    metrics = summarize_session(store, "project_demo", "run_demo")

    assert metrics.question_llm_skipped is True
    assert metrics.question_proposals_dropped == 2
    assert metrics.question_dataset_names_resolved == 1
    assert metrics.question_list_coercions == 3
    assert metrics.degraded is True


def test_run_metrics_mark_partial_acceptance_degraded_but_not_skipped(
    tmp_path: Path,
) -> None:
    store = _store_with_run(tmp_path)
    store.append_trace(
        "project_demo",
        TraceEvent(
            session_id="run_demo",
            event_type="llm_call",
            name="m4_question_discovery",
            summary={
                "candidate_count": 2,
                "proposals_dropped": 1,
                "dataset_names_resolved": 2,
                "list_coercions": 0,
                "degraded": True,
            },
        ),
    )

    metrics = summarize_session(store, "project_demo", "run_demo")

    assert metrics.question_llm_skipped is False
    assert metrics.question_proposals_dropped == 1
    assert metrics.question_dataset_names_resolved == 2
    assert metrics.degraded is True


def test_run_metrics_stay_green_when_the_route_is_healthy(tmp_path: Path) -> None:
    store = _store_with_run(tmp_path)
    store.append_trace(
        "project_demo",
        TraceEvent(
            session_id="run_demo",
            event_type="llm_call",
            name="m4_question_discovery",
            summary={
                "candidate_count": 3,
                "proposals_dropped": 0,
                "dataset_names_resolved": 1,
                "list_coercions": 0,
                "degraded": False,
            },
        ),
    )

    metrics = summarize_session(store, "project_demo", "run_demo")

    assert metrics.question_llm_skipped is False
    assert metrics.question_proposals_dropped == 0
    assert metrics.question_dataset_names_resolved == 1  # repair counted, not degraded
    assert metrics.degraded is False
