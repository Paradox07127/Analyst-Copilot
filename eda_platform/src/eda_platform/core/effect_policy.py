"""Transport policy at a durable model-effect boundary."""

from contextvars import ContextVar

durable_model_request: ContextVar[bool] = ContextVar("durable_model_request", default=False)
