from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field, WithJsonSchema

from eda_platform.core.budget import BudgetExceeded
from eda_platform.core.llm import StructuredLLM
from eda_platform.schemas.plans import Intent, IntentKind


def _fold_param_pairs(value: Any) -> Any:
    """Fold the wire's [{name, value}] list back into a mapping.

    Providers on json_object mode still send a plain mapping, which passes through.
    Unreadable entries are skipped rather than raised: params is routing metadata,
    and failing the whole intent would discard a good classification.
    """
    if not isinstance(value, list):
        return value
    folded: dict[str, Any] = {}
    for entry in value:
        if isinstance(entry, Mapping) and isinstance(entry.get("name"), str):
            folded[entry["name"]] = entry.get("value")
    return folded


_PARAM_PAIRS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
    },
}


class RoutedIntent(Intent):
    """Wire form of `Intent`: params travel as pairs, not as a dynamic-key object.

    OpenAI's json_schema strict mode rejects `additionalProperties: true` objects
    with HTTP 400 (observed 2026-08-01 on gpt-5.6-luna), which broke every OpenAI
    model on the chat path. Runtime stays a plain dict with arbitrary keys.
    """

    params: Annotated[
        dict[str, Any],
        BeforeValidator(_fold_param_pairs),
        WithJsonSchema(_PARAM_PAIRS_SCHEMA),
    ] = Field(default_factory=dict)


STRUCTURED_OUTPUT_SCHEMAS: tuple[type[BaseModel], ...] = (RoutedIntent,)

_INTENTS: tuple[IntentKind, ...] = (
    "ask_from_artifacts",
    "new_analysis",
    "open_analysis",
    "meta_help",
    "out_of_scope",
    "refine_analysis",
)


def route_intent(
    message: str,
    *,
    llm: StructuredLLM | None = None,
    confidence_threshold: float = 0.55,
) -> Intent:
    """Classify a chat turn, falling back to deterministic rules when needed."""
    if llm is not None:
        try:
            routed = llm.structured(
                task="m3_route_intent",
                schema=RoutedIntent,
                payload={
                    "message": message,
                    "allowed_intents": list(_INTENTS),
                    "routing_policy": {
                        "ask_from_artifacts": (
                            "Questions answerable from existing profiles, quality findings, "
                            "charts, report sections, or run metadata."
                        ),
                        "new_analysis": "Requests requiring a new read-only SQL analysis.",
                        "open_analysis": (
                            "Open-ended analysis that needs custom Python beyond SQL/templates, "
                            "such as regression residual anomalies or custom re-binning. "
                            "Use conservatively; anything SQL can answer stays new_analysis."
                        ),
                        "meta_help": "Questions about what the app can do or how to use it.",
                        "out_of_scope": "Requests unrelated to the loaded datasets or app.",
                        "refine_analysis": "Follow-up edits to the immediately prior analysis.",
                    },
                },
            )
            if routed.confidence >= confidence_threshold:
                return Intent.model_validate(routed.model_dump())
        except BudgetExceeded:
            raise
        except Exception:
            pass

    return _route_by_rules(message)


def _route_by_rules(message: str) -> Intent:
    text = message.strip()
    lowered = text.lower()

    if _contains_any(
        lowered,
        (
            "what can you do",
            "how do i use",
            "how to use",
            "help",
        ),
    ):
        return Intent(kind="meta_help", confidence=0.9, raw_message=message)

    if _contains_any(
        lowered,
        (
            "weather",
            "calendar",
            "send email",
            "write an email",
            "stock price",
        ),
    ):
        return Intent(kind="out_of_scope", confidence=0.85, raw_message=message)

    if _contains_any(
        lowered,
        (
            "report",
            "artifact",
            "profile",
            "schema",
            "missing",
            "quality",
            "warning",
            "chart",
        ),
    ):
        return Intent(kind="ask_from_artifacts", confidence=0.82, raw_message=message)

    if _contains_any(
        lowered,
        (
            "regression residual",
            "residual anomaly",
            "anomalies via regression",
            "custom re-bin",
            "custom rebin",
            "custom re-binning",
            "custom binning",
            "bespoke binning",
            "python analysis",
            "open-ended analysis",
        ),
    ):
        return Intent(kind="open_analysis", confidence=0.76, raw_message=message)

    return Intent(kind="new_analysis", confidence=0.65, raw_message=message)


def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle in value for needle in needles)
