"""T13: the resource ceiling is a user setting, and a limited stop says why.

The run this covers is the real one: nine Olist tables with pre-cleaning on,
whose working-set estimate crossed the 2 GiB default and stopped the analysis
before ingest with nothing but a one-line English sentence.
"""

from __future__ import annotations

import pytest

from eda_platform.application.dto import SettingsPatch
from eda_platform.application.services.job_service import _resolved_resource_policy
from eda_platform.application.services.session_service import SessionService
from eda_platform.application.services.settings_service import (
    DEFAULT_MAX_ROWS_PER_DATASET,
    DEFAULT_MAX_WORKING_SET_BYTES,
    SettingsService,
    SettingsValidationError,
)
from eda_platform.core.llm import LLMProvider, LLMSettings
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.resource_metrics import EdaDatasetEstimate, EdaResourcePolicy
from eda_platform.schemas.sessions import SessionManifest
from eda_platform.tools.resource_preflight import (
    LIMITING_REASON_CODES,
    decide_resource_preflight,
    resource_limit_guidance,
)

GIB = 1 << 30
MIB = 1 << 20


def _estimate(name: str, *, frame_bytes: int, rows: int = 1_000) -> EdaDatasetEstimate:
    return EdaDatasetEstimate(
        name=name,
        file_bytes=frame_bytes // 2,
        columns=8,
        sample_rows=10,
        sample_complete=True,
        sample_frame_deep_bytes=frame_bytes,
        sample_serialized_bytes=max(frame_bytes // 2, 1),
        frame_expansion_ratio=2.0,
        estimated_rows=rows,
        estimated_frame_deep_bytes=frame_bytes,
        exact_rows=rows,
        exact_frame_deep_bytes=frame_bytes,
    )


def _olist_shaped_tables() -> list[EdaDatasetEstimate]:
    """Nine tables whose largest sits between the two multiplier thresholds."""
    largest = 232 * MIB
    return [_estimate("geolocation", frame_bytes=largest)] + [
        _estimate(f"table_{index}", frame_bytes=20 * MIB) for index in range(8)
    ]


def _service() -> SettingsService:
    return SettingsService(
        defaults=LLMSettings(provider=LLMProvider.OFFLINE, model="offline-deterministic")
    )


# --- the setting itself -----------------------------------------------------


def test_new_session_reports_the_shipped_resource_ceiling() -> None:
    view = _service().get_settings()
    assert view.max_working_set_bytes == DEFAULT_MAX_WORKING_SET_BYTES == 2 * GIB
    assert view.max_rows_per_dataset == DEFAULT_MAX_ROWS_PER_DATASET == 10_000_000


def test_raised_ceiling_survives_later_reads_and_reaches_a_new_run() -> None:
    service = _service()
    service.update_settings(SettingsPatch(max_working_set_bytes=6 * GIB), "browser-a")

    assert service.get_settings("browser-a").max_working_set_bytes == 6 * GIB
    # A run queued afterwards must execute against the raised ceiling, not the
    # shipped default: this is the whole point of making it settable.
    assert service.resolve("browser-a").resource_overrides == {
        "max_working_set_bytes": 6 * GIB,
        "max_rows_per_dataset": DEFAULT_MAX_ROWS_PER_DATASET,
    }


def test_ceiling_is_per_caller_and_reset_returns_the_default() -> None:
    service = _service()
    service.update_settings(SettingsPatch(max_rows_per_dataset=50_000_000), "browser-a")
    assert service.get_settings("browser-b").max_rows_per_dataset == (
        DEFAULT_MAX_ROWS_PER_DATASET
    )
    service.reset("browser-a")
    assert service.get_settings("browser-a").max_rows_per_dataset == (
        DEFAULT_MAX_ROWS_PER_DATASET
    )


@pytest.mark.parametrize(
    "patch",
    [
        SettingsPatch(max_working_set_bytes=1),
        SettingsPatch(max_working_set_bytes=1024 * GIB),
        SettingsPatch(max_rows_per_dataset=0),
        SettingsPatch(max_rows_per_dataset=10_000_000_000),
    ],
)
def test_out_of_range_ceilings_are_refused(patch: SettingsPatch) -> None:
    with pytest.raises(SettingsValidationError):
        _service().update_settings(patch)


def test_only_the_two_settable_fields_reach_the_policy() -> None:
    policy = _resolved_resource_policy(
        "limited",
        {
            "max_working_set_bytes": 4 * GIB,
            "max_rows_per_dataset": 20_000_000,
            # A browser must not be able to reshape the estimator itself.
            "active_frame_multiplier": 0.0,
            "on_exceed": "reject",
        },
    )
    assert policy.max_working_set_bytes == 4 * GIB
    assert policy.max_rows_per_dataset == 20_000_000
    assert policy.active_frame_multiplier == EdaResourcePolicy().active_frame_multiplier
    assert policy.on_exceed == "limited"


# --- the same data, before and after raising the ceiling --------------------


def test_raising_the_ceiling_turns_the_stopped_run_into_an_accepted_one() -> None:
    tables = _olist_shaped_tables()
    stopped = decide_resource_preflight(
        tables,
        policy=_resolved_resource_policy("limited"),
        precleaning_enabled=True,
    )
    assert stopped.status == "limited"
    assert "estimated_working_set_exceeded" in stopped.reason_codes

    guidance = resource_limit_guidance(stopped)
    assert guidance is not None
    raised = decide_resource_preflight(
        tables,
        policy=_resolved_resource_policy(
            "limited",
            {"max_working_set_bytes": guidance.suggested_max_working_set_bytes},
        ),
        precleaning_enabled=True,
    )
    assert raised.status == "accepted"


def test_suggested_ceiling_is_the_smallest_round_budget_that_fits() -> None:
    stopped = decide_resource_preflight(
        _olist_shaped_tables(),
        policy=EdaResourcePolicy(),
        precleaning_enabled=True,
    )
    guidance = resource_limit_guidance(stopped)
    assert guidance is not None
    assert guidance.suggested_max_working_set_bytes >= (
        stopped.estimated_working_set_bytes
    )
    assert guidance.suggested_max_working_set_bytes % (256 * MIB) == 0


# --- the structured reason --------------------------------------------------


def test_limited_decision_carries_the_numbers_behind_the_stop() -> None:
    stopped = decide_resource_preflight(
        _olist_shaped_tables(),
        policy=EdaResourcePolicy(),
        precleaning_enabled=True,
    )
    guidance = resource_limit_guidance(stopped)
    assert guidance is not None
    assert guidance.status == "limited"
    assert guidance.reason_codes == ["estimated_working_set_exceeded"]
    assert guidance.dataset_count == 9
    assert guidance.max_working_set_bytes == 2 * GIB
    assert guidance.estimated_working_set_bytes > guidance.max_working_set_bytes
    assert guidance.over_budget_bytes == (
        guidance.estimated_working_set_bytes - guidance.max_working_set_bytes
    )
    assert guidance.largest_dataset_name == "geolocation"
    assert guidance.largest_dataset_bytes == 232 * MIB
    assert guidance.largest_dataset_rows == 1_000
    assert guidance.precleaning_enabled is True


def test_turning_the_clean_off_is_offered_only_when_it_actually_fits() -> None:
    tables = _olist_shaped_tables()
    with_clean = decide_resource_preflight(
        tables, policy=EdaResourcePolicy(), precleaning_enabled=True
    )
    guidance = resource_limit_guidance(with_clean)
    assert guidance is not None
    assert guidance.disabling_precleaning_would_fit is True
    # The claim is checkable: the same tables without the clean are accepted.
    assert (
        decide_resource_preflight(
            tables, policy=EdaResourcePolicy(), precleaning_enabled=False
        ).status
        == "accepted"
    )
    assert guidance.working_set_without_precleaning_bytes <= 2 * GIB


def test_the_clean_is_not_offered_when_the_data_is_too_large_either_way() -> None:
    huge = [_estimate("huge", frame_bytes=2 * GIB)]
    guidance = resource_limit_guidance(
        decide_resource_preflight(
            huge, policy=EdaResourcePolicy(), precleaning_enabled=True
        )
    )
    assert guidance is not None
    assert guidance.disabling_precleaning_would_fit is False
    assert guidance.working_set_without_precleaning_bytes > 2 * GIB


def test_a_row_cap_stop_reports_the_row_cap_not_only_the_memory_budget() -> None:
    guidance = resource_limit_guidance(
        decide_resource_preflight(
            [_estimate("wide", frame_bytes=1 * MIB, rows=50_000_000)],
            policy=EdaResourcePolicy(max_rows_per_dataset=1_000_000),
        )
    )
    assert guidance is not None
    assert guidance.reason_codes == ["row_count_exceeded"]
    assert guidance.largest_dataset_rows == 50_000_000
    assert guidance.max_rows_per_dataset == 1_000_000
    # Turning the clean off cannot clear a row cap, so it must not be offered.
    assert guidance.disabling_precleaning_would_fit is False


def test_accepted_runs_carry_no_guidance() -> None:
    accepted = decide_resource_preflight(
        [_estimate("small", frame_bytes=1 * MIB)], policy=EdaResourcePolicy()
    )
    assert accepted.status == "accepted"
    assert resource_limit_guidance(accepted) is None


def test_worker_adjustments_are_not_presented_as_limits() -> None:
    stopped = decide_resource_preflight(
        _olist_shaped_tables(),
        requested_dataset_workers=2,
        policy=EdaResourcePolicy(),
        precleaning_enabled=True,
    )
    assert "streaming_dataset_lifecycle" in stopped.reason_codes
    guidance = resource_limit_guidance(stopped)
    assert guidance is not None
    assert "streaming_dataset_lifecycle" not in guidance.reason_codes


def test_every_limiting_code_the_decision_emits_is_a_known_one() -> None:
    """Guards the split between "a limit was hit" and "a worker was adjusted"."""
    everything_over = decide_resource_preflight(
        [
            _estimate("huge", frame_bytes=4 * GIB, rows=50_000_000),
            _estimate("second", frame_bytes=1 * MIB),
        ],
        policy=EdaResourcePolicy(
            max_dataset_count=1,
            max_columns_per_dataset=1,
            max_single_input_bytes=1,
            max_input_bytes_total=1,
            max_rows_per_dataset=1,
            max_working_set_bytes=1,
        ),
    )
    empty = decide_resource_preflight([], policy=EdaResourcePolicy())
    emitted = set(everything_over.reason_codes) | set(empty.reason_codes)
    assert emitted - {"streaming_dataset_lifecycle", "memory_budget_worker_downgrade"} == (
        LIMITING_REASON_CODES
    )


# --- the stop survives a reload --------------------------------------------


def _limited_run_with_preflight(tmp_path, decision) -> SessionService:
    from datetime import UTC, datetime

    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    store.start_session("demo", "run-1")
    store.write_manifest(
        SessionManifest(
            session_id="run-1",
            project_id="demo",
            input_hashes={"geolocation.csv": "abc"},
            code_version="auto-eda",
            created_at=datetime.now(UTC),
        )
    )
    store.save_artifact(
        Artifact(
            id="resource-preflight-1",
            type=ArtifactType.RESOURCE_PREFLIGHT,
            project_id="demo",
            session_id="run-1",
            payload=decision.model_dump(mode="json"),
        )
    )
    store.mark_session_status("demo", "run-1", "limited")
    return SessionService(store)


def test_session_detail_of_a_limited_run_explains_the_stop(tmp_path) -> None:
    decision = decide_resource_preflight(
        _olist_shaped_tables(), policy=EdaResourcePolicy(), precleaning_enabled=True
    )
    detail = _limited_run_with_preflight(tmp_path, decision).get_session_detail("run-1")

    assert detail.status == "limited"
    assert detail.resource_limit is not None
    assert detail.resource_limit.reason_codes == ["estimated_working_set_exceeded"]
    assert detail.resource_limit.largest_dataset_name == "geolocation"
    assert detail.resource_limit.disabling_precleaning_would_fit is True


def test_session_detail_of_a_normal_run_carries_no_stop(tmp_path) -> None:
    decision = decide_resource_preflight(
        [_estimate("small", frame_bytes=1 * MIB)], policy=EdaResourcePolicy()
    )
    service = _limited_run_with_preflight(tmp_path, decision)
    service._store.mark_session_status("demo", "run-1", "completed")
    assert service.get_session_detail("run-1").resource_limit is None


# --- the stop reaches the live job stream ----------------------------------


def test_completed_job_event_carries_the_stop_for_a_limited_run(tmp_path) -> None:
    from eda_platform.worker.runner import _completion_summary

    decision = decide_resource_preflight(
        _olist_shaped_tables(), policy=EdaResourcePolicy(), precleaning_enabled=True
    )
    service = _limited_run_with_preflight(tmp_path, decision)
    job = {"job_id": "job-1", "project_id": "demo", "session_id": "run-1"}

    summary = _completion_summary(service._store, job, {"llm": "offline"})
    assert summary is not None
    limit = summary["resource_limit"]
    assert limit["reason_codes"] == ["estimated_working_set_exceeded"]
    assert limit["largest_dataset_name"] == "geolocation"
    assert limit["disabling_precleaning_would_fit"] is True
    assert limit["suggested_max_working_set_bytes"] > limit["max_working_set_bytes"]


def test_completed_job_event_stays_empty_for_an_accepted_run(tmp_path) -> None:
    from eda_platform.worker.runner import _completion_summary

    decision = decide_resource_preflight(
        [_estimate("small", frame_bytes=1 * MIB)], policy=EdaResourcePolicy()
    )
    service = _limited_run_with_preflight(tmp_path, decision)
    job = {"job_id": "job-1", "project_id": "demo", "session_id": "run-1"}
    assert _completion_summary(service._store, job, {"llm": "offline"}) is None
