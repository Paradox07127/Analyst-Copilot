from unittest.mock import Mock

import pytest

from eda_platform.application.dto import DatasetHandle
from eda_platform.application.services.relationship_service import (
    MAX_DISCOVERY_DATASETS,
    MAX_DISCOVERY_PAIRS,
    RelationshipNotDiscoverableError,
    RelationshipService,
)
from eda_platform.tools.relationship_discovery import _should_sample


def _handle(index: int) -> DatasetHandle:
    return DatasetHandle(
        dataset_id=f"ds_{index}",
        project_id="demo",
        display_name=f"table_{index}.csv",
        original_uri=f"projects/demo/uploads/ds_{index}/v1/table_{index}.csv",
        ingest_status="ready",
    )


def test_relationship_discovery_rejects_unbounded_dataset_pairs() -> None:
    datasets = Mock(
        list_datasets=lambda _session_id: [
            _handle(index) for index in range(MAX_DISCOVERY_DATASETS + 1)
        ]
    )
    service = RelationshipService(
        store=Mock(),
        datasets=datasets,
        approvals=Mock(),
        jobs=Mock(),
        semantic=Mock(),
    )

    with pytest.raises(RelationshipNotDiscoverableError) as caught:
        service._require_discoverable("run_many")  # noqa: SLF001

    assert str(MAX_DISCOVERY_DATASETS) in str(caught.value)
    assert str(MAX_DISCOVERY_PAIRS) in str(caught.value)


def test_relationship_overlap_samples_both_equal_large_tables() -> None:
    options = {"sample_threshold_rows": 100_000, "sample_size": 25_000}

    assert _should_sample(1_000_000, 1_000_000, **options) is True
    assert _should_sample(50_000, 1_000_000, **options) is False
