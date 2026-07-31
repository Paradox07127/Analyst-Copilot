from eda_platform.application.services.approval_service import (
    ApprovalConsumedError,
    ApprovalExpiredError,
    ApprovalNotFoundError,
)
from eda_platform.application.services.cleaning_service import (
    CleaningSourceChangedError,
    CleaningValidationError,
)
from eda_platform.application.services.dataset_service import (
    DatasetNotFoundError,
    DatasetSourceMissingError,
)
from eda_platform.application.services.insight_service import (
    CustomChartValidationError,
)
from eda_platform.worker.runner import _durable_error_code


def test_worker_persists_stable_approval_error_codes() -> None:
    assert _durable_error_code(ApprovalConsumedError("hash")) == "approval_consumed"
    assert _durable_error_code(ApprovalExpiredError("hash")) == "approval_expired"
    assert _durable_error_code(ApprovalNotFoundError("hash")) == "approval_not_found"


def test_worker_keeps_legacy_class_name_for_untyped_failures() -> None:
    assert _durable_error_code(ValueError("bad input")) == "ValueError"


def test_data_operation_failures_persist_public_error_codes() -> None:
    assert _durable_error_code(CleaningValidationError("bad options")) == (
        "cleaning_invalid"
    )
    assert _durable_error_code(CleaningSourceChangedError("dataset")) == (
        "cleaning_source_changed"
    )
    assert _durable_error_code(DatasetNotFoundError("dataset", "run")) == (
        "dataset_not_found"
    )
    assert _durable_error_code(DatasetSourceMissingError("dataset")) == (
        "dataset_source_missing"
    )
    assert _durable_error_code(CustomChartValidationError("bad chart")) == (
        "custom_chart_invalid"
    )
