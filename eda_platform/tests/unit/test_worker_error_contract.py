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
from eda_platform.worker.error_translation import durable_error_code


def test_worker_persists_stable_approval_error_codes() -> None:
    assert durable_error_code(ApprovalConsumedError("hash")) == "approval_consumed"
    assert durable_error_code(ApprovalExpiredError("hash")) == "approval_expired"
    assert durable_error_code(ApprovalNotFoundError("hash")) == "approval_not_found"


def test_worker_uses_exception_class_name_for_untyped_failures() -> None:
    assert durable_error_code(ValueError("bad input")) == "ValueError"


def test_data_operation_failures_persist_public_error_codes() -> None:
    assert durable_error_code(CleaningValidationError("bad options")) == (
        "cleaning_invalid"
    )
    assert durable_error_code(CleaningSourceChangedError("dataset")) == (
        "cleaning_source_changed"
    )
    assert durable_error_code(DatasetNotFoundError("dataset", "run")) == (
        "dataset_not_found"
    )
    assert durable_error_code(DatasetSourceMissingError("dataset")) == (
        "dataset_source_missing"
    )
    assert durable_error_code(CustomChartValidationError("bad chart")) == (
        "custom_chart_invalid"
    )
