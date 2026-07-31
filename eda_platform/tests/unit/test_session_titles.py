"""Human-friendly run titles: deterministic builder, manifest round-trip,
follow-up batch titles, and the cosmetic LLM upgrade's fallback contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from eda_platform.core.kernel import SessionContext
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import _llm_session_title, _sanitize_session_title
from eda_platform.drivers.question_exec import follow_up_session_title
from eda_platform.schemas.sessions import (
    SESSION_TITLE_MAX_CHARS,
    SessionManifest,
    build_run_title,
    clip_run_title,
)


# --------------------------------------------------------------------------- #
# build_run_title (pure)
# --------------------------------------------------------------------------- #
def test_build_run_title_empty_is_empty() -> None:
    assert build_run_title([]) == ""
    assert build_run_title(["", "   "]) == ""


def test_build_run_title_single_dataset_cleans_stem() -> None:
    assert build_run_title(["creditcard.csv"]) == "Creditcard"
    assert build_run_title(["ecommerce_customers.csv"]) == "Ecommerce Customers"
    # Inner caps survive (no str.title() mangling).
    assert build_run_title(["GDP_by-region.csv"]) == "GDP By Region"


def test_build_run_title_common_prefix_collapses_to_table_count() -> None:
    names = [f"olist_{part}_dataset.csv" for part in ("orders", "items", "sellers")]
    assert build_run_title(names) == "Olist (3 tables)"


def test_build_run_title_majority_prefix_wins_over_one_odd_file() -> None:
    # One translation lookup beside many olist_* tables must not hijack the name.
    names = ["product_category_name_translation.csv"] + [
        f"olist_{part}_dataset.csv" for part in ("orders", "items", "sellers")
    ]
    assert build_run_title(names) == "Olist (4 tables)"


def test_build_run_title_mixed_names_use_first_stem_plus_more() -> None:
    assert build_run_title(["customers.csv", "orders.csv", "geo.csv"]) == (
        "Customers +2 more"
    )


def test_build_run_title_two_unrelated_names_do_not_claim_a_family() -> None:
    # 1-of-2 is not a majority: "(2 tables)" would wrongly claim a shared family.
    assert build_run_title(["customers.csv", "orders.csv"]) == "Customers +1 more"


def test_build_run_title_exact_tie_falls_back_to_plus_more() -> None:
    # 2-of-4 is a tie, not a majority.
    names = ["a_x.csv", "a_y.csv", "b_x.csv", "b_y.csv"]
    assert build_run_title(names) == "A X +3 more"


def test_build_run_title_caps_length_with_ellipsis() -> None:
    long_name = "a_very_long_dataset_name_that_keeps_going_and_going.csv"
    title = build_run_title([long_name])
    assert len(title) <= SESSION_TITLE_MAX_CHARS
    assert title.endswith("…")


def test_clip_run_title_collapses_whitespace() -> None:
    assert clip_run_title("  Fraud \n Overview  ") == "Fraud Overview"


# --------------------------------------------------------------------------- #
# Manifest round-trip -> SessionInfo.title (store)
# --------------------------------------------------------------------------- #
def test_run_info_title_round_trips_through_manifest(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("proj", name="proj")
    store.start_session("proj", "run_titled")
    store.write_manifest(
        SessionManifest(
            session_id="run_titled",
            project_id="proj",
            input_hashes={"creditcard.csv": "abc"},
            code_version="unknown",
            title="Creditcard Fraud Overview",
        )
    )
    infos = {info.session_id: info for info in store.list_sessions("proj")}
    assert infos["run_titled"].title == "Creditcard Fraud Overview"


def test_run_info_title_none_for_legacy_manifest(tmp_path: Path) -> None:
    """A manifest written before the title field existed yields title=None."""
    store = ArtifactStore(tmp_path)
    store.ensure_project("proj", name="proj")
    session_dir = store.start_session("proj", "run_legacy")
    legacy = {
        "session_id": "run_legacy",
        "project_id": "proj",
        "input_hashes": {"orders.csv": "abc"},
        "code_version": "unknown",
        "created_at": "2026-07-03T00:00:00+00:00",
    }
    (session_dir / "manifest.json").write_text(json.dumps(legacy), encoding="utf-8")
    infos = {info.session_id: info for info in store.list_sessions("proj")}
    assert infos["run_legacy"].title is None


# --------------------------------------------------------------------------- #
# Follow-up batch titles (question_exec)
# --------------------------------------------------------------------------- #
def test_follow_up_title_uses_source_manifest_title(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("proj", name="proj")
    store.start_session("proj", "run_src")
    store.write_manifest(
        SessionManifest(
            session_id="run_src",
            project_id="proj",
            input_hashes={"creditcard.csv": "abc"},
            code_version="unknown",
            title="Creditcard",
        )
    )
    title = follow_up_session_title(
        store, project_id="proj", source_session_id="run_src", question_count=3
    )
    assert title == "Follow-up: Creditcard"


def test_follow_up_title_derives_from_source_inputs_when_untitled(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("proj", name="proj")
    store.start_session("proj", "run_src")
    store.write_manifest(
        SessionManifest(
            session_id="run_src",
            project_id="proj",
            input_hashes={"olist_orders.csv": "a", "olist_items.csv": "b"},
            code_version="unknown",
        )
    )
    title = follow_up_session_title(
        store, project_id="proj", source_session_id="run_src", question_count=2
    )
    assert title == "Follow-up: Olist (2 tables)"


def test_follow_up_title_falls_back_to_question_count(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    assert (
        follow_up_session_title(
            store, project_id="proj", source_session_id="missing", question_count=1
        )
        == "Follow-up (1 question)"
    )
    assert (
        follow_up_session_title(
            store, project_id="proj", source_session_id="missing", question_count=4
        )
        == "Follow-up (4 questions)"
    )


# --------------------------------------------------------------------------- #
# LLM title upgrade: sanitize + fall back on ANY failure (auto_eda)
# --------------------------------------------------------------------------- #
T = TypeVar("T", bound=BaseModel)


class _FakeTitleLLM:
    """Minimal live-looking LLMClient double for the session_title call."""

    def __init__(self, response: str = "", error: Exception | None = None) -> None:
        self.response = response
        self.error = error

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        raise NotImplementedError

    def text(self, *, task: str, payload: dict) -> str:
        assert task == "session_title"
        if self.error is not None:
            raise self.error
        return self.response

    def last_usage(self) -> None:
        return None


def test_sanitize_run_title_strips_quotes_newlines_and_caps() -> None:
    assert _sanitize_session_title('"Fraud\nSignals Overview"\n') == (
        "Fraud Signals Overview"
    )
    long = _sanitize_session_title("Word " * 30)
    assert len(long) <= SESSION_TITLE_MAX_CHARS
    assert long.endswith("…")


def test_llm_run_title_success_writes_trace(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    ctx = SessionContext(project_id="proj", session_id="run_x", store=store)
    title = _llm_session_title(
        ctx,
        _FakeTitleLLM(response="'Credit Card Fraud Patterns'"),
        dataset_names=["creditcard.csv"],
        business_context="fraud team",
        report_artifacts=[],
    )
    assert title == "Credit Card Fraud Patterns"
    events = store.list_trace_events(project_id="proj", session_id="run_x")
    assert any(
        event.event_type == "llm_call" and event.name == "session_title"
        for event in events
    )


def test_llm_run_title_failure_returns_none_and_traces_error(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    ctx = SessionContext(project_id="proj", session_id="run_x", store=store)
    title = _llm_session_title(
        ctx,
        _FakeTitleLLM(error=RuntimeError("provider down")),
        dataset_names=["creditcard.csv"],
        business_context="",
        report_artifacts=[],
    )
    assert title is None
    events = store.list_trace_events(project_id="proj", session_id="run_x")
    assert any(
        event.event_type == "llm_error" and event.name == "session_title"
        for event in events
    )


def test_llm_run_title_empty_response_returns_none(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    ctx = SessionContext(project_id="proj", session_id="run_x", store=store)
    title = _llm_session_title(
        ctx,
        _FakeTitleLLM(response="  \n '' "),
        dataset_names=[],
        business_context="",
        report_artifacts=[],
    )
    assert title is None
