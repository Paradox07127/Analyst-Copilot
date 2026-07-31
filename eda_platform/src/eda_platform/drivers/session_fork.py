"""Run-fork: vary one decision from a completed run and re-run."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from eda_platform.core.llm import LLMClient
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import AutoEDAResult, run_auto_eda
from eda_platform.drivers.cancellation import raise_if_cancelled
from eda_platform.schemas.sessions import TraceEvent
from eda_platform.tools.evidence import PayloadPolicy

__all__ = [
    "DatasetDecision",
    "ForkDecision",
    "MlTargetDecision",
    "fork_session",
]


@dataclass(frozen=True)
class DatasetDecision:
    """Re-run on a different input set — a new dataset or a cleaned version."""

    file_paths: Sequence[Path | str]
    label: str = ""
    kind: ClassVar[Literal["dataset"]] = "dataset"

    def summary(self) -> str:
        if self.label:
            return f"dataset → {self.label}"
        names = ", ".join(Path(path).name for path in self.file_paths)
        return f"dataset → {names or 'new input'}"


@dataclass(frozen=True)
class MlTargetDecision:
    """Change (or clear, with ``None``) the ML baseline target column."""

    ml_target_column: str | None
    kind: ClassVar[Literal["ml_target_column"]] = "ml_target_column"

    def summary(self) -> str:
        return f"ML target → {self.ml_target_column or 'none'}"


ForkDecision = DatasetDecision | MlTargetDecision


def fork_session(
    parent: AutoEDAResult,
    *,
    decision: ForkDecision,
    store: ArtifactStore,
    llm: LLMClient | None = None,
    payload_policy: PayloadPolicy = "schema+aggregates",
    ml_target_column: str | None = None,
    ml_time_column: str | None = None,
    business_context: str | None = None,
    on_trace_event: Callable[[TraceEvent], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> AutoEDAResult:
    """Fork ``parent`` into a new run that differs by exactly one ``decision``."""
    raise_if_cancelled(cancel_check, operation="run fork")
    file_paths: list[Path | str] = list(_parent_file_paths(parent))
    fork_target = ml_target_column

    if isinstance(decision, DatasetDecision):
        chosen = list(decision.file_paths)
        if not chosen:
            raise ValueError("DatasetDecision requires at least one dataset path.")
        file_paths = chosen
    elif isinstance(decision, MlTargetDecision):
        fork_target = decision.ml_target_column
    else:  # defensive: an unmodelled decision type is a caller bug, not a silent no-op
        raise TypeError(f"Unsupported fork decision: {type(decision).__name__}")

    if not file_paths:
        raise ValueError(
            "fork_session needs source datasets: the parent run has no loaded datasets "
            "and the decision did not supply new file paths."
        )

    resolved_context = business_context if business_context is not None else parent.business_context
    raise_if_cancelled(cancel_check, operation="run fork")
    result = run_auto_eda(
        file_paths,
        workspace=store.root,
        project_id=parent.project_id,
        session_id=None,
        business_context=resolved_context,
        llm=llm,
        payload_policy=payload_policy,
        ml_target_column=fork_target,
        ml_time_column=ml_time_column,
        on_trace_event=on_trace_event,
        cancel_check=cancel_check,
    )
    raise_if_cancelled(cancel_check, operation="run fork")
    return result


def _parent_file_paths(parent: AutoEDAResult) -> list[Path]:
    """The parent's ingested source files, in order, for a same-inputs re-run."""
    return [Path(dataset.record.path) for dataset in parent.loaded_datasets]
