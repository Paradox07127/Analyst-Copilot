"""Generated-OpenAPI mutation policy and shared response replay/audit middleware."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from collections.abc import Awaitable, Callable
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from typing import Any, Literal

from anyio import to_thread
from fastapi import FastAPI

from eda_platform.api.middleware import Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

MutationPolicy = Literal["idempotency", "version", "intrinsic"]

# A keyed replay is the only path that must hold the whole request in memory.
# The global BodySizeLimitMiddleware is sized for uploads, so without this the
# buffer would inherit an upload-sized ceiling on ordinary JSON mutations.
MAX_REPLAY_BODY_BYTES = 1 << 20

# Commands and creates whose successful response may be replayed. Keep this
# explicit: a newly-added mutation must not acquire replay semantics by
# accident merely because it was omitted from the reviewed policy matrix.
IDEMPOTENT_OPERATIONS = frozenset(
    {
        "create_project_api_v1_projects_post",
        "delete_project_api_v1_projects__project_id__self_delete",
        "delete_upload_api_v1_projects__project_id__uploads__dataset_id__delete",
        "delete_session_api_v1_sessions__session_id__delete",
        "start_dataset_distributions_api_v1_sessions__session_id__datasets__dataset_id__distributions_post",
        "create_job_api_v1_sessions__session_id__jobs_post",
        "apply_cleaning_api_v1_sessions__session_id__cleaning_apply_post",
        "draft_question_card_api_v1_sessions__session_id__questions_post",
        "execute_question_api_v1_sessions__session_id__questions__question_id__execute_post",
        "generate_report_api_v1_sessions__session_id__report_generate_post",
        "build_custom_chart_api_v1_sessions__session_id__charts_custom_post",
        "promote_finding_api_v1_sessions__session_id__findings__finding_id__promote_post",
        "discover_relationships_api_v1_sessions__session_id__relationships_discover_post",
        "validate_relationship_api_v1_sessions__session_id__relationships__relationship_id__validate_post",
        "fork_session_api_v1_sessions__session_id__fork_post",
        "save_skill_api_v1_sessions__session_id__skills_post",
        "import_seed_skill_api_v1_sessions__session_id__skills__seed_id__import_post",
        "delete_skill_api_v1_projects__project_id__skills__skill_id__delete",
        "execute_skill_replay_api_v1_sessions__session_id__skills__skill_id__execute_post",
        "send_chat_message_api_v1_sessions__session_id__chat_messages_post",
        "approve_chat_plan_api_v1_sessions__session_id__chat_plans__plan_id__approve_post",
        "reject_chat_plan_api_v1_sessions__session_id__chat_plans__plan_id__reject_post",
        "record_client_failure_api_v1_sessions__session_id__client_failures_post",
        "create_decision_story_draft_api_v1_sessions__session_id__decision_story_drafts_post",
        "generate_decision_report_api_v1_sessions__session_id__decision_report_generate_post",
        "create_support_doc_api_v1_projects__project_id__support_docs_post",
        "delete_support_doc_api_v1_projects__project_id__support_docs__doc_id__delete",
        "build_investigation_plans_api_v1_sessions__session_id__investigations_plan_post",
        "approve_investigation_plan_api_v1_sessions__session_id__investigations__plan_id__approve_post",
        "reject_investigation_plan_api_v1_sessions__session_id__investigations__plan_id__reject_post",
        "execute_investigation_plans_api_v1_sessions__session_id__investigations_execute_post",
        "start_macro_loop_api_v1_sessions__session_id__investigations_macro_loop_post",
    }
)

# Shared resources: stale writers must be rejected instead of replayed.
VERSIONED_OPERATIONS = frozenset(
    {
        "edit_question_card_api_v1_sessions__session_id__questions__question_id__patch",
        "update_semantic_seeds_api_v1_sessions__session_id__semantic_seeds_put",
        "confirm_join_api_v1_sessions__session_id__semantic_joins_confirm_post",
        "revoke_join_api_v1_sessions__session_id__semantic_joins_revoke_post",
        "delete_verified_relation_api_v1_sessions__session_id__semantic_verified_relations_delete_post",
        "accept_proposal_api_v1_sessions__session_id__semantic_proposals_accept_post",
        "reject_proposal_api_v1_sessions__session_id__semantic_proposals_reject_post",
        "confirm_relationship_api_v1_sessions__session_id__relationships__relationship_id__confirm_post",
        "revoke_relationship_api_v1_sessions__session_id__relationships__relationship_id__revoke_post",
        "put_board_api_v1_projects__project_id__boards__board_id__put",
        "update_settings_api_v1_settings_put",
        "reset_settings_api_v1_settings_delete",
    }
)

# A versioned write can additionally make transport retries replay-safe without
# weakening its optimistic concurrency check.  The original request still
# reaches the handler exactly once and must pass expected_version; only an
# identical request carrying the same key may replay that completed response.
REPLAYABLE_VERSIONED_OPERATIONS = frozenset(
    {
        "put_board_api_v1_projects__project_id__boards__board_id__put",
    }
)

# These do not create an independently replayable mutation: prepare returns a
# content-derived approval, preview is read-only apart from that approval, and
# cancel/test are naturally convergent state/probe operations.
INTRINSIC_OPERATIONS = frozenset(
    {
        "cancel_job_api_v1_jobs__job_id__cancel_post",
        "create_upload_api_v1_projects__project_id__uploads_post",
        "preview_cleaning_api_v1_sessions__session_id__cleaning_preview_post",
        "prepare_question_execution_api_v1_sessions__session_id__questions__question_id__prepare_post",
        "prepare_question_draft_api_v1_sessions__session_id__questions_prepare_draft_post",
        "prepare_promotion_api_v1_sessions__session_id__findings__finding_id__prepare_promote_post",
        "prepare_relationship_validation_api_v1_sessions__session_id__relationships__relationship_id__prepare_validate_post",
        "prepare_skill_replay_api_v1_sessions__session_id__skills__skill_id__prepare_post",
        "test_connection_api_v1_settings_test_post",
        # A read that has to be a POST because it must bypass the cache. It
        # creates nothing, so replaying it is harmless.
        "refresh_models_api_v1_settings_models_refresh_post",
        "prepare_investigation_decision_api_v1_sessions__session_id__investigations__plan_id__prepare_decision_post",
        "prepare_investigation_execution_api_v1_sessions__session_id__investigations_prepare_execute_post",
        "prepare_macro_loop_api_v1_sessions__session_id__investigations_prepare_macro_loop_post",
        # Renaming is naturally convergent: repeating the same display name
        # does not create a second resource or require optimistic locking.
        "rename_project_api_v1_projects__project_id__patch",
        "rename_session_api_v1_sessions__session_id__patch",
        "reorder_projects_api_v1_projects_order_put",
    }
)


def mutation_policy(operation_id: str) -> MutationPolicy:
    if operation_id in IDEMPOTENT_OPERATIONS:
        return "idempotency"
    if operation_id in VERSIONED_OPERATIONS:
        return "version"
    if operation_id in INTRINSIC_OPERATIONS:
        return "intrinsic"
    raise ValueError(
        f"Mutation operation {operation_id!r} has no explicit contract policy."
    )


def _template_pattern(template: str) -> re.Pattern[str]:
    """Escape the literal segments so only ``{param}`` becomes a wildcard."""
    parts = re.split(r"(\{[^/{}]+\})", template)
    return re.compile(
        "".join(
            "[^/]+" if part.startswith("{") else re.escape(part) for part in parts
        )
    )


@dataclass(frozen=True)
class MutationRoute:
    operation_id: str
    method: str
    template: str
    policy: MutationPolicy
    pattern: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pattern", _template_pattern(self.template))

    def matches(self, method: str, path: str) -> bool:
        return method == self.method and self.pattern.fullmatch(path) is not None


class MutationContractMiddleware:
    """Content-bound replay and digest-only audit for classified mutations.

    The key remains optional for backward compatibility with local scripts.
    When supplied, it is globally bound to operation + request content and a
    completed response is replayed byte-for-byte. Concurrent duplicates get a
    typed retryable conflict instead of executing twice.
    """

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        db_path: str,
        routes: tuple[MutationRoute, ...],
    ) -> None:
        self.app = app
        self.db_path = db_path
        self.routes = routes
        self._ensure_schema()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        route = next(
            (
                item
                for item in self.routes
                if item.matches(
                    str(scope.get("method", "")).upper(),
                    str(scope.get("path", "")),
                )
            ),
            None,
        )
        if route is None:
            await self.app(scope, receive, send)
            return
        headers = {
            name.decode("latin-1").lower(): value.decode("latin-1")
            for name, value in scope.get("headers") or []
        }
        key = headers.get("idempotency-key", "").strip()
        if not (_supports_replay(route) and key):
            # Nothing to replay, so the body never has to be materialized. This
            # is also what keeps a 1 GiB upload streaming instead of buffered.
            await self._audit_streaming(scope, receive, send, route, key=key or None)
            return
        captured = await _capture_request(receive, MAX_REPLAY_BODY_BYTES)
        if captured is None:
            await _send_contract_error(
                send,
                413,
                "request_too_large",
                f"A replayable request body may not exceed {MAX_REPLAY_BODY_BYTES} bytes.",
            )
            return
        body, replay_receive = captured
        hasher = _request_hasher(route, scope)
        hasher.update(body)
        digest = hasher.hexdigest()
        try:
            replay = await to_thread.run_sync(
                partial(self._reserve, route.operation_id, key, digest)
            )
        except MutationKeyConflictError:
            await _send_contract_error(
                send,
                422,
                "idempotency_key_reused",
                "Idempotency-Key is already bound to different request content.",
            )
            return
        except MutationInProgressError:
            await _send_contract_error(
                send,
                409,
                "mutation_in_progress",
                "A request with this Idempotency-Key is still executing.",
                retry_after=1,
            )
            return
        if replay is not None:
            status, response_headers, response_body = replay
            await send(
                {
                    "type": "http.response.start",
                    "status": status,
                    "headers": response_headers,
                }
            )
            await send({"type": "http.response.body", "body": response_body})
            return
        status = 500
        response_headers: list[tuple[bytes, bytes]] = []
        chunks: list[bytes] = []

        async def capture_send(message: Message) -> None:
            nonlocal status, response_headers
            if message["type"] == "http.response.start":
                status = int(message["status"])
                response_headers = list(message.get("headers", []))
            elif message["type"] == "http.response.body":
                chunks.append(bytes(message.get("body", b"")))
            await send(message)

        try:
            await self.app(scope, replay_receive, capture_send)
        finally:
            response_body = b"".join(chunks)
            await self._finish_quietly(
                route=route,
                key=key,
                digest=digest,
                status=status,
                response_headers=response_headers,
                response_body=response_body,
                resource=str(scope.get("path", "")),
            )

    async def _audit_streaming(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        route: MutationRoute,
        *,
        key: str | None,
    ) -> None:
        """Hash the request as it streams, without materializing the body.

        Reaching this path means no response will be replayed, so neither the
        request nor the response has to be held in memory.
        """
        hasher = _request_hasher(route, scope)
        status = 500
        response_headers: list[tuple[bytes, bytes]] = []

        async def hashing_receive() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                hasher.update(bytes(message.get("body", b"")))
            return message

        async def capture_send(message: Message) -> None:
            nonlocal status, response_headers
            if message["type"] == "http.response.start":
                status = int(message["status"])
                response_headers = list(message.get("headers", []))
            await send(message)

        try:
            await self.app(scope, hashing_receive, capture_send)
        finally:
            await self._finish_quietly(
                route=route,
                key=key,
                digest=hasher.hexdigest(),
                status=status,
                response_headers=response_headers,
                response_body=b"",
                resource=str(scope.get("path", "")),
            )

    async def _finish_quietly(self, **kwargs: Any) -> None:
        """Persist the audit/replay row without letting it mask the real error.

        Both call sites run inside a ``finally``, so an exception here would
        replace whatever the application raised and point the operator at the
        audit path instead of the actual failure.
        """
        try:
            await to_thread.run_sync(partial(self._finish, **kwargs))
        except Exception:
            logger.exception(
                "Mutation audit persistence failed for %s", kwargs.get("resource")
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=5.0)

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.executescript(
                """
                create table if not exists mutation_replays (
                    operation_id text not null,
                    idempotency_key text not null,
                    request_digest text not null,
                    state text not null,
                    status_code integer,
                    response_headers blob,
                    response_body blob,
                    created_at real not null,
                    updated_at real not null,
                    primary key(operation_id, idempotency_key)
                );
                create table if not exists mutation_audit (
                    id integer primary key autoincrement,
                    operation_id text not null,
                    policy text not null,
                    resource text not null,
                    idempotency_key text,
                    request_digest text not null,
                    outcome_status integer not null,
                    occurred_at text not null
                );
                create index if not exists idx_mutation_audit_operation_time
                    on mutation_audit(operation_id, occurred_at);
                """
            )

    def _reserve(
        self, operation_id: str, key: str, digest: str
    ) -> tuple[int, list[tuple[bytes, bytes]], bytes] | None:
        now = time.time()
        with closing(self._connect()) as conn:
            conn.execute("begin immediate")
            row = conn.execute(
                """
                select request_digest, state, status_code,
                       response_headers, response_body, updated_at
                from mutation_replays
                where operation_id = ? and idempotency_key = ?
                """,
                (operation_id, key),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    insert into mutation_replays(
                        operation_id, idempotency_key, request_digest, state,
                        created_at, updated_at
                    ) values(?, ?, ?, 'executing', ?, ?)
                    """,
                    (operation_id, key, digest, now, now),
                )
                conn.commit()
                return None
            if str(row[0]) != digest:
                conn.rollback()
                raise MutationKeyConflictError()
            if str(row[1]) == "completed":
                headers = json.loads(bytes(row[3]).decode("utf-8"))
                conn.commit()
                return (
                    int(row[2]),
                    [
                        (str(name).encode("latin-1"), str(value).encode("latin-1"))
                        for name, value in headers
                    ],
                    bytes(row[4]),
                )
            if now - float(row[5]) < 300:
                conn.rollback()
                raise MutationInProgressError()
            conn.execute(
                """
                update mutation_replays set updated_at = ?
                where operation_id = ? and idempotency_key = ?
                """,
                (now, operation_id, key),
            )
            conn.commit()
            return None

    def _finish(
        self,
        *,
        route: MutationRoute,
        key: str | None,
        digest: str,
        status: int,
        response_headers: list[tuple[bytes, bytes]],
        response_body: bytes,
        resource: str,
    ) -> None:
        with closing(self._connect()) as conn:
            if _supports_replay(route) and key:
                if 200 <= status < 300:
                    safe_headers = [
                        (name.decode("latin-1"), value.decode("latin-1"))
                        for name, value in response_headers
                        if name.lower()
                        in {b"content-type", b"content-disposition", b"location"}
                    ]
                    conn.execute(
                        """
                        update mutation_replays
                        set state = 'completed', status_code = ?,
                            response_headers = ?, response_body = ?, updated_at = ?
                        where operation_id = ? and idempotency_key = ?
                          and request_digest = ?
                        """,
                        (
                            status,
                            json.dumps(safe_headers).encode(),
                            response_body,
                            time.time(),
                            route.operation_id,
                            key,
                            digest,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        delete from mutation_replays
                        where operation_id = ? and idempotency_key = ?
                          and request_digest = ? and state = 'executing'
                        """,
                        (route.operation_id, key, digest),
                    )
            conn.execute(
                """
                insert into mutation_audit(
                    operation_id, policy, resource, idempotency_key,
                    request_digest, outcome_status, occurred_at
                ) values(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    route.operation_id,
                    route.policy,
                    resource,
                    key,
                    digest,
                    status,
                    datetime.now(UTC).isoformat(),
                ),
            )
            conn.commit()


class MutationKeyConflictError(Exception):
    pass


class MutationInProgressError(Exception):
    pass


def _request_hasher(route: MutationRoute, scope: Scope) -> Any:
    hasher = hashlib.sha256()
    hasher.update(route.operation_id.encode())
    hasher.update(b"\0")
    hasher.update(str(scope.get("path", "")).encode())
    hasher.update(b"?")
    hasher.update(bytes(scope.get("query_string", b"")))
    hasher.update(b"\0")
    return hasher


def _supports_replay(route: MutationRoute) -> bool:
    return (
        route.policy == "idempotency"
        or route.operation_id in REPLAYABLE_VERSIONED_OPERATIONS
    )


def configure_mutation_contract(app: FastAPI, *, db_path: str) -> None:
    """Classify every live OpenAPI mutation, document it, then install replay."""
    routes: list[MutationRoute] = []
    unsafe = {"POST", "PUT", "PATCH", "DELETE"}
    original_openapi = app.openapi

    def contracted_openapi() -> dict[str, Any]:
        schema = original_openapi()
        for _template, path_item in schema.get("paths", {}).items():
            for method, operation in path_item.items():
                if method.upper() not in unsafe or not isinstance(operation, dict):
                    continue
                operation_id = str(operation["operationId"])
                policy = mutation_policy(operation_id)
                operation["x-mutation-policy"] = policy
                if (
                    policy != "idempotency"
                    and operation_id not in REPLAYABLE_VERSIONED_OPERATIONS
                ):
                    continue
                parameters = list(operation.get("parameters", []))
                if any(
                    item.get("in") == "header"
                    and item.get("name", "").lower() == "idempotency-key"
                    for item in parameters
                    if isinstance(item, dict)
                ):
                    continue
                parameters.append(
                    {
                        "name": "Idempotency-Key",
                        "in": "header",
                        "required": False,
                        "schema": {"type": "string", "minLength": 8, "maxLength": 200},
                        "description": (
                            "Optional for backward compatibility. When supplied, "
                            "the key is content-bound and a completed retry replays "
                            "the original response."
                        ),
                    }
                )
                operation["parameters"] = parameters
        return schema

    app.openapi = contracted_openapi  # type: ignore[method-assign]
    schema = app.openapi()
    for template, path_item in schema["paths"].items():
        for method, operation in path_item.items():
            if method.upper() not in unsafe or not isinstance(operation, dict):
                continue
            operation_id = str(operation["operationId"])
            policy = mutation_policy(operation_id)
            routes.append(
                MutationRoute(
                    operation_id=operation_id,
                    method=method.upper(),
                    template=template,
                    policy=policy,
                )
            )
    app.add_middleware(
        MutationContractMiddleware,
        db_path=db_path,
        routes=tuple(routes),
    )


async def _capture_request(
    receive: Receive, max_bytes: int
) -> tuple[bytes, Receive] | None:
    """Buffer the body for replay, or return ``None`` once it exceeds the cap.

    Chunked senders never declare a length, so the ceiling is enforced while
    accumulating rather than from Content-Length alone.
    """
    messages: list[Message] = []
    body = bytearray()
    while True:
        message = await receive()
        messages.append(message)
        if message["type"] == "http.request":
            body.extend(message.get("body", b""))
            if len(body) > max_bytes:
                return None
            if not message.get("more_body", False):
                break
        else:
            break
    index = 0

    async def replay() -> Message:
        nonlocal index
        if index < len(messages):
            message = messages[index]
            index += 1
            return message
        return {"type": "http.request", "body": b"", "more_body": False}

    return bytes(body), replay


async def _send_contract_error(
    send: Send,
    status: int,
    code: str,
    message: str,
    *,
    retry_after: int | None = None,
) -> None:
    body = json.dumps({"error": {"code": code, "message": message}}).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    if retry_after is not None:
        headers.append((b"retry-after", str(retry_after).encode("ascii")))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
