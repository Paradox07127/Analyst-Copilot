"""Bounded CSV inspection and deterministic Auto-EDA resource decisions."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from math import ceil
from pathlib import Path

import pandas as pd

from eda_platform.schemas.resource_metrics import (
    DatasetRole,
    EdaDatasetEstimate,
    EdaResourceLimitGuidance,
    EdaResourcePolicy,
    EdaResourcePreflight,
)
from eda_platform.tools.loader import stream_csv_chunks

_MIN_FRAME_EXPANSION_RATIO = 0.5
_MAX_FRAME_EXPANSION_RATIO = 8.0

# Reason codes that mean "a limit was exceeded", as opposed to the worker
# adjustments that also land in `reason_codes`.
LIMITING_REASON_CODES = frozenset(
    {
        "no_input_datasets",
        "dataset_count_exceeded",
        "single_input_bytes_exceeded",
        "input_bytes_total_exceeded",
        "column_count_exceeded",
        "row_count_exceeded",
        "estimated_working_set_exceeded",
    }
)

# Granularity of the memory budget a user is asked to set, so the suggestion is
# a round number in the unit the settings form uses rather than raw bytes.
_BUDGET_SUGGESTION_STEP_BYTES = 256 << 20


class EdaResourceLimitError(RuntimeError):
    """Strict-mode preflight rejection with a stable durable worker code."""

    error_code = "eda_resource_limit_exceeded"

    def __init__(self, decision: EdaResourcePreflight) -> None:
        reasons = ", ".join(decision.reason_codes) or "resource policy"
        super().__init__(f"Auto-EDA resource limit exceeded: {reasons}.")
        self.decision = decision


def estimate_frame_bytes(
    *,
    file_bytes: int,
    sample_frame_deep_bytes: int,
    sample_serialized_bytes: int,
    safety_factor: float = 1.25,
) -> tuple[int, float]:
    """Estimate full-frame deep bytes from a canonicalized bounded sample."""
    if file_bytes <= 0:
        return max(0, sample_frame_deep_bytes), 0.0
    denominator = max(1, sample_serialized_bytes)
    observed = max(0.0, sample_frame_deep_bytes / denominator)
    ratio = min(_MAX_FRAME_EXPANSION_RATIO, max(_MIN_FRAME_EXPANSION_RATIO, observed))
    estimate = max(
        sample_frame_deep_bytes,
        ceil(file_bytes * ratio * max(1.0, safety_factor)),
    )
    return estimate, ratio


def estimate_csv_resource(
    path: Path | str,
    *,
    role: DatasetRole = "analysis",
    sample_rows: int = 10_000,
    safety_factor: float = 1.25,
    cancel_check: Callable[[], object] | None = None,
) -> EdaDatasetEstimate:
    """Inspect one CSV using at most one configured parser chunk."""
    source = Path(path)
    if sample_rows < 1:
        raise ValueError("sample_rows must be >= 1")
    if cancel_check is not None:
        cancel_check()

    def first_chunk(chunks: Iterator[pd.DataFrame]) -> pd.DataFrame:
        try:
            frame = next(chunks)
        except StopIteration:
            frame = pd.DataFrame()
        if cancel_check is not None:
            cancel_check()
        return frame

    try:
        sample = stream_csv_chunks(source, first_chunk, chunksize=sample_rows)
    except pd.errors.EmptyDataError:
        sample = pd.DataFrame()
    file_bytes = source.stat().st_size
    deep_bytes = int(sample.memory_usage(index=True, deep=True).sum())
    serialized_bytes = len(sample.to_csv(index=False).encode("utf-8"))
    estimated_frame, ratio = estimate_frame_bytes(
        file_bytes=file_bytes,
        sample_frame_deep_bytes=deep_bytes,
        sample_serialized_bytes=serialized_bytes,
        safety_factor=safety_factor,
    )
    row_count = int(len(sample))
    sample_complete = row_count < sample_rows
    estimated_rows = (
        row_count
        if sample_complete
        else ceil(file_bytes * row_count / max(1, serialized_bytes))
    )
    return EdaDatasetEstimate(
        role=role,
        name=source.name,
        file_bytes=file_bytes,
        columns=int(sample.shape[1]),
        sample_rows=row_count,
        sample_complete=sample_complete,
        sample_frame_deep_bytes=deep_bytes,
        sample_serialized_bytes=serialized_bytes,
        frame_expansion_ratio=ratio,
        estimated_rows=max(row_count, estimated_rows),
        estimated_frame_deep_bytes=estimated_frame,
        exact_rows=row_count if sample_complete else None,
        exact_frame_deep_bytes=deep_bytes if sample_complete else None,
    )


def inspect_csv_resources(
    file_paths: Sequence[Path | str],
    *,
    role: DatasetRole = "analysis",
    policy: EdaResourcePolicy | None = None,
    cancel_check: Callable[[], object] | None = None,
) -> list[EdaDatasetEstimate]:
    """Inspect a sequence without retaining DataFrames or writing intermediate files."""
    effective = policy or EdaResourcePolicy()
    estimates: list[EdaDatasetEstimate] = []
    for path in file_paths:
        if cancel_check is not None:
            cancel_check()
        estimates.append(
            estimate_csv_resource(
                path,
                role=role,
                sample_rows=effective.sample_rows,
                safety_factor=effective.frame_estimate_safety_factor,
                cancel_check=cancel_check,
            )
        )
    return estimates


def estimate_working_set_bytes(
    estimates: Sequence[EdaDatasetEstimate],
    *,
    active_workers: int,
    baseline_peak_rss_bytes: int,
    policy: EdaResourcePolicy,
    precleaning_enabled: bool = False,
) -> int:
    """Estimate the peak of the sequential per-table lifecycle.

    Auto-EDA retains artifacts and lightweight source handles, not every parsed
    frame.  The retained term is therefore the largest table active in a
    lifecycle slot, while the active multiplier covers that table's profiling,
    charting and statistical temporaries.
    """
    frames = [item.best_frame_deep_bytes for item in estimates]
    retained_multiplier = 2 if precleaning_enabled else 1
    retained = max(frames, default=0) * retained_multiplier
    active = sum(sorted(frames, reverse=True)[: max(0, active_workers)])
    return ceil(
        max(0, baseline_peak_rss_bytes)
        + policy.held_frame_multiplier * retained
        + policy.active_frame_multiplier * active
    )


def decide_resource_preflight(
    estimates: Sequence[EdaDatasetEstimate],
    *,
    requested_dataset_workers: int = 1,
    baseline_peak_rss_bytes: int = 0,
    policy: EdaResourcePolicy | None = None,
    precleaning_enabled: bool = False,
) -> EdaResourcePreflight:
    """Choose exact, worker-downgraded, limited, or rejected execution."""
    if requested_dataset_workers not in {1, 2}:
        raise ValueError("requested_dataset_workers must be 1 or 2")
    effective_policy = policy or EdaResourcePolicy()
    datasets = list(estimates)
    dataset_count = len(datasets)
    # The memory-optimized driver intentionally owns one full table at a time.
    candidate_workers = min(1, requested_dataset_workers, dataset_count)
    worker_reason: str | None = None
    if candidate_workers < requested_dataset_workers and dataset_count > 0:
        worker_reason = "streaming_dataset_lifecycle"

    estimated_working_set = estimate_working_set_bytes(
        datasets,
        active_workers=candidate_workers,
        baseline_peak_rss_bytes=baseline_peak_rss_bytes,
        policy=effective_policy,
        precleaning_enabled=precleaning_enabled,
    )
    if candidate_workers > 1 and estimated_working_set > effective_policy.max_working_set_bytes:
        single_worker_estimate = estimate_working_set_bytes(
            datasets,
            active_workers=1,
            baseline_peak_rss_bytes=baseline_peak_rss_bytes,
            policy=effective_policy,
            precleaning_enabled=precleaning_enabled,
        )
        if single_worker_estimate <= effective_policy.max_working_set_bytes:
            candidate_workers = 1
            estimated_working_set = single_worker_estimate
            worker_reason = "memory_budget_worker_downgrade"

    limiting_reasons: list[str] = []
    if dataset_count == 0:
        limiting_reasons.append("no_input_datasets")
    if dataset_count > effective_policy.max_dataset_count:
        limiting_reasons.append("dataset_count_exceeded")
    if any(item.file_bytes > effective_policy.max_single_input_bytes for item in datasets):
        limiting_reasons.append("single_input_bytes_exceeded")
    if sum(item.file_bytes for item in datasets) > effective_policy.max_input_bytes_total:
        limiting_reasons.append("input_bytes_total_exceeded")
    if any(item.columns > effective_policy.max_columns_per_dataset for item in datasets):
        limiting_reasons.append("column_count_exceeded")
    if any(item.best_rows > effective_policy.max_rows_per_dataset for item in datasets):
        limiting_reasons.append("row_count_exceeded")
    if estimated_working_set > effective_policy.max_working_set_bytes:
        limiting_reasons.append("estimated_working_set_exceeded")

    status = (
        "accepted"
        if not limiting_reasons
        else "limited"
        if effective_policy.on_exceed == "limited"
        else "rejected"
    )
    reasons = list(limiting_reasons)
    if worker_reason is not None:
        reasons.append(worker_reason)
    return EdaResourcePreflight(
        status=status,
        compute_mode="streaming_exact" if status == "accepted" else "metadata_only",
        reason_codes=reasons,
        requested_dataset_workers=requested_dataset_workers,
        effective_dataset_workers=candidate_workers,
        worker_adjustment_reason=worker_reason,
        precleaning_enabled=precleaning_enabled,
        policy=effective_policy,
        input_dataset_count=dataset_count,
        input_bytes_total=sum(item.file_bytes for item in datasets),
        input_rows_estimated=sum(item.best_rows for item in datasets),
        input_columns_total=sum(item.columns for item in datasets),
        estimated_frame_deep_bytes_total=(
            sum(item.best_frame_deep_bytes for item in datasets)
            * (2 if precleaning_enabled else 1)
        ),
        baseline_peak_rss_bytes=max(0, baseline_peak_rss_bytes),
        estimated_working_set_bytes=estimated_working_set,
        datasets=datasets,
    )


def preflight_csv_resources(
    file_paths: Sequence[Path | str],
    *,
    requested_dataset_workers: int = 1,
    baseline_peak_rss_bytes: int = 0,
    policy: EdaResourcePolicy | None = None,
    precleaning_enabled: bool = False,
    cancel_check: Callable[[], object] | None = None,
) -> EdaResourcePreflight:
    """Inspect and decide in one call; rejected decisions remain returnable for persistence."""
    effective = policy or EdaResourcePolicy()
    estimates = inspect_csv_resources(
        file_paths,
        policy=effective,
        cancel_check=cancel_check,
    )
    return decide_resource_preflight(
        estimates,
        requested_dataset_workers=requested_dataset_workers,
        baseline_peak_rss_bytes=baseline_peak_rss_bytes,
        policy=effective,
        precleaning_enabled=precleaning_enabled,
    )


def resource_limit_guidance(
    decision: EdaResourcePreflight,
) -> EdaResourceLimitGuidance | None:
    """Restate a stopped decision as the numbers and the ways out.

    Returns None for an accepted run. The "turn the clean off" way out is
    computed, not guessed: the working set is re-estimated with precleaning
    disabled and offered only when that alone clears every limit.
    """
    if decision.status == "accepted":
        return None
    policy = decision.policy
    limiting = [code for code in decision.reason_codes if code in LIMITING_REASON_CODES]
    largest = max(
        decision.datasets,
        key=lambda item: item.best_frame_deep_bytes,
        default=None,
    )
    without_precleaning = (
        estimate_working_set_bytes(
            decision.datasets,
            active_workers=decision.effective_dataset_workers,
            baseline_peak_rss_bytes=decision.baseline_peak_rss_bytes,
            policy=policy,
            precleaning_enabled=False,
        )
        if decision.precleaning_enabled
        else decision.estimated_working_set_bytes
    )
    clean_off_fits = (
        decision.precleaning_enabled
        and limiting == ["estimated_working_set_exceeded"]
        and without_precleaning <= policy.max_working_set_bytes
    )
    headroom = max(decision.estimated_working_set_bytes, policy.max_working_set_bytes)
    return EdaResourceLimitGuidance(
        status=decision.status,
        reason_codes=limiting,
        dataset_count=decision.input_dataset_count,
        estimated_working_set_bytes=decision.estimated_working_set_bytes,
        max_working_set_bytes=policy.max_working_set_bytes,
        over_budget_bytes=max(
            0, decision.estimated_working_set_bytes - policy.max_working_set_bytes
        ),
        largest_dataset_name="" if largest is None else largest.name,
        largest_dataset_bytes=0 if largest is None else largest.best_frame_deep_bytes,
        largest_dataset_rows=0 if largest is None else largest.best_rows,
        max_rows_per_dataset=policy.max_rows_per_dataset,
        precleaning_enabled=decision.precleaning_enabled,
        working_set_without_precleaning_bytes=without_precleaning,
        disabling_precleaning_would_fit=clean_off_fits,
        suggested_max_working_set_bytes=(
            ceil(headroom / _BUDGET_SUGGESTION_STEP_BYTES) * _BUDGET_SUGGESTION_STEP_BYTES
        ),
    )


def enforce_resource_preflight(
    decision: EdaResourcePreflight,
) -> EdaResourcePreflight:
    """Raise only after a caller has had the opportunity to persist the decision."""
    if decision.status == "rejected":
        raise EdaResourceLimitError(decision)
    return decision
