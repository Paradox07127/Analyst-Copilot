"""Unit tests for the generalised run-fork driver (M6.2).

Each supported decision gets its own offline test: the fork runs
without an LLM, mints a fresh ``session_id`` in the same project, and the one changed
lever actually takes effect (dataset swapped or ML target modelled). Two edge tests
pin the "needs source data" contract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import eda_platform.drivers.session_fork as run_fork_driver
from eda_platform.core.kernel import SessionCancelled
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import AutoEDAResult, run_auto_eda
from eda_platform.drivers.session_fork import (
    DatasetDecision,
    MlTargetDecision,
    fork_session,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.model_card import ModelCard
from eda_platform.schemas.reports import ReportBundle

GOLDEN_DATA = Path(__file__).parents[1] / "golden" / "data"


def _parent(tmp_path: Path, *, dataset: str = "ecommerce_orders.csv") -> AutoEDAResult:
    return run_auto_eda(
        [GOLDEN_DATA / dataset],
        workspace=tmp_path / "ws",
        project_id="proj_fork",
    )


def _bundle(result: AutoEDAResult) -> ReportBundle:
    artifact = next(
        a for a in result.artifacts if a.type is ArtifactType.REPORT_BUNDLE
    )
    return ReportBundle.model_validate(artifact.payload)


def _model_cards(result: AutoEDAResult) -> list[Artifact]:
    return [a for a in result.artifacts if a.type is ArtifactType.MODEL_CARD]


def _write_churn_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "segment": ["A"] * 40 + ["B"] * 40,
            "spend": [float(i % 20) for i in range(80)],
            "visits": [i % 6 + 1 for i in range(80)],
            "churned": [1 if i % 5 in {0, 1} else 0 for i in range(80)],
        }
    ).to_csv(path, index=False)


# --- dataset decision -------------------------------------------------------


def test_fork_dataset_decision_swaps_the_input(tmp_path: Path) -> None:
    parent = _parent(tmp_path, dataset="ecommerce_orders.csv")
    store = ArtifactStore(parent.workspace)

    child = fork_session(
        parent,
        decision=DatasetDecision(file_paths=(GOLDEN_DATA / "ecommerce_customers.csv",)),
        store=store,
    )

    assert child.session_id != parent.session_id
    assert child.project_id == parent.project_id
    assert [ds.record.name for ds in child.loaded_datasets] == ["ecommerce_customers.csv"]
    # The parent is untouched: it still points at its original input.
    assert [ds.record.name for ds in parent.loaded_datasets] == ["ecommerce_orders.csv"]


def test_fork_dataset_decision_needs_no_parent_datasets(tmp_path: Path) -> None:
    # A DatasetDecision supplies its own inputs, so it forks even a source-less parent.
    empty = AutoEDAResult(
        project_id="proj_fork",
        session_id="run_empty",
        business_context="",
        artifacts=[],
        report_markdown="",
        workspace=tmp_path / "ws",
        loaded_datasets=[],
    )
    store = ArtifactStore(empty.workspace)

    child = fork_session(
        empty,
        decision=DatasetDecision(file_paths=(GOLDEN_DATA / "ecommerce_orders.csv",)),
        store=store,
    )

    assert child.project_id == "proj_fork"
    assert [ds.record.name for ds in child.loaded_datasets] == ["ecommerce_orders.csv"]


# --- ML target decision -----------------------------------------------------


def test_fork_ml_target_decision_produces_a_model_card(tmp_path: Path) -> None:
    csv_path = tmp_path / "churn.csv"
    _write_churn_csv(csv_path)
    parent = run_auto_eda([csv_path], workspace=tmp_path / "ws", project_id="proj_fork")
    store = ArtifactStore(parent.workspace)
    assert not _model_cards(parent), "parent ran without an ML target"

    child = fork_session(
        parent,
        decision=MlTargetDecision(ml_target_column="churned"),
        store=store,
    )

    cards = _model_cards(child)
    assert cards, "forking with an ML target should model it"
    assert ModelCard.model_validate(cards[0].payload).target_column == "churned"
    assert child.session_id != parent.session_id
    assert child.project_id == parent.project_id


# --- edge: a non-dataset fork needs the parent's source data ----------------


def test_fork_without_source_data_raises(tmp_path: Path) -> None:
    empty = AutoEDAResult(
        project_id="proj_fork",
        session_id="run_empty",
        business_context="",
        artifacts=[],
        report_markdown="",
        workspace=tmp_path / "ws",
        loaded_datasets=[],
    )
    store = ArtifactStore(empty.workspace)

    with pytest.raises(ValueError, match="source datasets"):
        fork_session(
            empty,
            decision=MlTargetDecision(ml_target_column="churned"),
            store=store,
        )


def test_decision_summaries_describe_the_change() -> None:
    assert "churned" in MlTargetDecision(ml_target_column="churned").summary()
    assert "none" in MlTargetDecision(ml_target_column=None).summary()
    assert "orders.csv" in DatasetDecision(file_paths=("/tmp/orders.csv",)).summary()


def test_fork_forwards_cancel_check_and_stops_after_child_driver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _parent(tmp_path)
    store = ArtifactStore(parent.workspace)
    child_returned = False
    forwarded: list[object] = []

    def cancel_check() -> bool:
        return child_returned

    def fake_run_auto_eda(*args: object, **kwargs: object) -> AutoEDAResult:
        nonlocal child_returned
        forwarded.append(kwargs.get("cancel_check"))
        child_returned = True
        return parent

    monkeypatch.setattr(run_fork_driver, "run_auto_eda", fake_run_auto_eda)

    with pytest.raises(SessionCancelled):
        fork_session(
            parent,
            decision=MlTargetDecision(ml_target_column=None),
            store=store,
            cancel_check=cancel_check,
        )

    assert forwarded == [cancel_check]
