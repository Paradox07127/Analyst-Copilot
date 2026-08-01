from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from eda_platform.core import process_metrics
from eda_platform.schemas.resource_metrics import (
    AutoEdaResourceUsage,
    EdaDatasetEstimate,
    EdaResourcePolicy,
)
from eda_platform.tools.resource_preflight import (
    EdaResourceLimitError,
    decide_resource_preflight,
    enforce_resource_preflight,
    estimate_csv_resource,
    estimate_frame_bytes,
    estimate_working_set_bytes,
    inspect_csv_resources,
)


def _estimate(
    name: str,
    *,
    file_bytes: int = 10,
    rows: int = 10,
    columns: int = 2,
    frame_bytes: int = 100,
) -> EdaDatasetEstimate:
    return EdaDatasetEstimate(
        name=name,
        file_bytes=file_bytes,
        columns=columns,
        sample_rows=min(rows, 10),
        sample_complete=True,
        sample_frame_deep_bytes=frame_bytes,
        sample_serialized_bytes=max(file_bytes, 1),
        frame_expansion_ratio=1.0,
        estimated_rows=rows,
        estimated_frame_deep_bytes=frame_bytes,
        exact_rows=rows,
        exact_frame_deep_bytes=frame_bytes,
    )


def _policy(**updates: object) -> EdaResourcePolicy:
    return EdaResourcePolicy().model_copy(update=updates)


def test_policy_defaults_are_explicit_and_total_covers_single() -> None:
    policy = EdaResourcePolicy()
    assert policy.max_dataset_count == 20
    assert policy.max_single_input_bytes == 1 << 30
    assert policy.max_input_bytes_total == 2 << 30
    assert policy.max_columns_per_dataset == 512
    assert policy.max_rows_per_dataset == 10_000_000
    assert policy.max_working_set_bytes == 2 << 30
    assert policy.sample_rows == 10_000
    assert policy.on_exceed == "limited"
    with pytest.raises(ValidationError):
        EdaResourcePolicy(max_single_input_bytes=11, max_input_bytes_total=10)


def test_runtime_usage_defaults_keep_legacy_metrics_readable() -> None:
    usage = AutoEdaResourceUsage()
    assert usage.measurement_status == "unavailable"
    assert usage.inputs.analysis.dataset_count == 0
    assert usage.memory.peak_rss_bytes is None
    assert usage.artifacts.default_context_bytes == 0


def test_frame_estimate_clamps_ratio_and_applies_safety() -> None:
    low, low_ratio = estimate_frame_bytes(
        file_bytes=1_000,
        sample_frame_deep_bytes=1,
        sample_serialized_bytes=1_000,
        safety_factor=1.25,
    )
    high, high_ratio = estimate_frame_bytes(
        file_bytes=1_000,
        sample_frame_deep_bytes=100_000,
        sample_serialized_bytes=1,
        safety_factor=1.25,
    )
    assert low_ratio == 0.5
    assert low == 625
    assert high_ratio == 8.0
    assert high == 100_000


def test_csv_inspection_is_exact_for_a_small_complete_file(tmp_path: Path) -> None:
    source = tmp_path / "small.csv"
    source.write_text("name,value\na,1\nb,2\n", encoding="utf-8")

    estimate = estimate_csv_resource(source, sample_rows=10)

    assert estimate.file_bytes == source.stat().st_size
    assert estimate.columns == 2
    assert estimate.sample_rows == 2
    assert estimate.sample_complete is True
    assert estimate.exact_rows == 2
    assert estimate.best_rows == 2
    assert estimate.exact_frame_deep_bytes == estimate.sample_frame_deep_bytes
    assert estimate.estimated_frame_deep_bytes >= estimate.sample_frame_deep_bytes


def test_csv_inspection_handles_an_empty_file_and_checks_cancellation(tmp_path: Path) -> None:
    source = tmp_path / "empty.csv"
    source.write_bytes(b"")
    calls = 0

    def checkpoint() -> None:
        nonlocal calls
        calls += 1

    estimates = inspect_csv_resources([source], cancel_check=checkpoint)

    assert calls >= 2
    assert estimates[0].file_bytes == 0
    assert estimates[0].columns == 0
    assert estimates[0].best_rows == 0


def test_working_set_formula_accounts_for_preclean_raw_retention() -> None:
    estimates = [_estimate("a", frame_bytes=100), _estimate("b", frame_bytes=100)]
    policy = _policy(
        held_frame_multiplier=2.0,
        active_frame_multiplier=5.0,
    )
    ordinary = estimate_working_set_bytes(
        estimates,
        active_workers=1,
        baseline_peak_rss_bytes=100,
        policy=policy,
    )
    preclean = estimate_working_set_bytes(
        estimates,
        active_workers=1,
        baseline_peak_rss_bytes=100,
        policy=policy,
        precleaning_enabled=True,
    )
    assert ordinary == 1_000
    assert preclean == 1_400


def test_two_workers_downgrade_to_one_when_only_one_fits() -> None:
    estimates = [_estimate("a"), _estimate("b")]
    policy = _policy(max_working_set_bytes=1_000)

    decision = decide_resource_preflight(
        estimates,
        requested_dataset_workers=2,
        policy=policy,
    )

    assert decision.status == "accepted"
    assert decision.effective_dataset_workers == 1
    assert decision.estimated_working_set_bytes == 900
    assert decision.worker_adjustment_reason == "memory_budget_worker_downgrade"
    assert "memory_budget_worker_downgrade" in decision.reason_codes


def test_single_worker_over_budget_returns_metadata_only_limited() -> None:
    decision = decide_resource_preflight(
        [_estimate("large", frame_bytes=200)],
        policy=_policy(max_working_set_bytes=1_000),
    )
    assert decision.status == "limited"
    assert decision.compute_mode == "metadata_only"
    assert decision.reason_codes == ["estimated_working_set_exceeded"]
    assert enforce_resource_preflight(decision) is decision


def test_reject_policy_produces_typed_durable_error() -> None:
    decision = decide_resource_preflight(
        [_estimate("large", frame_bytes=200)],
        policy=_policy(max_working_set_bytes=1_000, on_exceed="reject"),
    )
    assert decision.status == "rejected"
    with pytest.raises(EdaResourceLimitError) as caught:
        enforce_resource_preflight(decision)
    assert caught.value.error_code == "eda_resource_limit_exceeded"
    assert caught.value.decision is decision


@pytest.mark.parametrize(
    ("estimates", "policy", "reason"),
    [
        (
            [_estimate("a"), _estimate("b"), _estimate("c")],
            _policy(max_dataset_count=2),
            "dataset_count_exceeded",
        ),
        (
            [_estimate("a", file_bytes=11)],
            _policy(max_single_input_bytes=10, max_input_bytes_total=20),
            "single_input_bytes_exceeded",
        ),
        (
            [_estimate("a", file_bytes=6), _estimate("b", file_bytes=6)],
            _policy(max_single_input_bytes=10, max_input_bytes_total=10),
            "input_bytes_total_exceeded",
        ),
        (
            [_estimate("a", columns=513)],
            _policy(max_columns_per_dataset=512),
            "column_count_exceeded",
        ),
        (
            [_estimate("a", rows=101)],
            _policy(max_rows_per_dataset=100),
            "row_count_exceeded",
        ),
    ],
)
def test_policy_boundaries_have_stable_reason_codes(
    estimates: list[EdaDatasetEstimate],
    policy: EdaResourcePolicy,
    reason: str,
) -> None:
    decision = decide_resource_preflight(estimates, policy=policy)
    assert decision.status == "limited"
    assert reason in decision.reason_codes


def test_exact_thresholds_are_accepted() -> None:
    policy = _policy(
        max_dataset_count=2,
        max_single_input_bytes=10,
        max_input_bytes_total=20,
        max_columns_per_dataset=2,
        max_rows_per_dataset=10,
        max_working_set_bytes=10_000,
    )
    decision = decide_resource_preflight(
        [_estimate("a"), _estimate("b")],
        requested_dataset_workers=2,
        policy=policy,
    )
    assert decision.status == "accepted"
    assert decision.effective_dataset_workers == 2


def test_empty_inputs_are_limited_and_workers_are_zero() -> None:
    decision = decide_resource_preflight([])
    assert decision.status == "limited"
    assert decision.effective_dataset_workers == 0
    assert decision.reason_codes == ["no_input_datasets"]


def test_invalid_requested_workers_fail_before_decision() -> None:
    with pytest.raises(ValueError, match="must be 1 or 2"):
        decide_resource_preflight([_estimate("a")], requested_dataset_workers=3)


def test_posix_peak_normalization_uses_platform_units() -> None:
    assert process_metrics.normalize_posix_maxrss(123, platform_name="darwin") == 123
    assert process_metrics.normalize_posix_maxrss(123, platform_name="linux") == 123 * 1024


def test_peak_rss_selects_posix_and_windows_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_metrics, "_posix_peak_maxrss", lambda: 7)
    posix = process_metrics.process_peak_rss(platform_name="linux")
    assert posix.bytes == 7 * 1024
    assert posix.method == "getrusage_ru_maxrss"

    monkeypatch.setattr(process_metrics, "_windows_peak_working_set_bytes", lambda: 99)
    windows = process_metrics.process_peak_rss(platform_name="win32")
    assert windows.bytes == 99
    assert windows.method == "get_process_memory_info_peak_working_set"


def test_peak_rss_is_typed_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_metrics, "_posix_peak_maxrss", lambda: None)
    measurement = process_metrics.process_peak_rss(platform_name="linux")
    assert measurement.bytes is None
    assert measurement.method == "unavailable"
