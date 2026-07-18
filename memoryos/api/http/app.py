"""HTTP 应用入口。"""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import os
import uuid
from typing import Any
from urllib.parse import parse_qs

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.api.memory_contract import validate_memory_request, validate_memory_response
from memoryos.api.retrieval_contract import parse_retrieval_options
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.application.context.orchestrator import RetrievalUnavailableError
from memoryos.application.session.events import AgentEventType, AgentHookEvent, NormalizedAgentEvent
from memoryos.contextdb.retrieval.limits import MAX_RETRIEVAL_LIMIT, MAX_TOKEN_BUDGET, bounded_int
from memoryos.core.readiness import RuntimeNotReadyError
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.security.sanitization import sanitize_error_text
from memoryos.security.trusted_context import (
    AUTHORITATIVE_FORGET,
    AUTHORITATIVE_REMEMBER,
    COMMIT_SESSION,
    HARD_ERASE_MEMORY,
    READ_CONTEXT,
    AuthenticationError,
    TrustedRequestContext,
    capabilities_from_csv,
    sanitize_ingress_messages,
    scope_keys_from_csv,
    workspace_ids_from_csv,
)


def handle(
    route: str,
    client: MemoryOSClient,
    payload: dict[str, Any],
    *,
    caller: TrustedRequestContext | None = None,
) -> dict[str, Any]:
    if route == "POST /predict":
        request_payload = payload.get("request")
        if not isinstance(request_payload, dict):
            raise ValueError("POST /predict requires object field: request")
        policies = [ActionPolicy(**item) for item in payload.get("policies", [])]
        return client.predict(PredictionRequest(**request_payload), policies).to_dict()
    if route == "POST /context/search":
        return {
            "results": client.search_context(
                _required_str(payload, "query", route),
                **_search_kwargs(payload),
                caller=caller,
            )
        }
    if route == "POST /context/assemble":
        return client.assemble_context(
            _required_str(payload, "query", route),
            options=parse_retrieval_options(payload.get("options")),
            user_id=payload.get("user_id"),
            token_budget=bounded_int(
                payload.get("token_budget"),
                default=2000,
                minimum=1,
                maximum=MAX_TOKEN_BUDGET,
                label="token_budget",
            ),
            context_types=payload.get("context_types"),
            limit=bounded_int(
                payload.get("limit"),
                default=20,
                minimum=1,
                maximum=MAX_RETRIEVAL_LIMIT,
                label="limit",
            ),
            connect_metadata=payload.get("connect_metadata"),
            search_scope=payload.get("search_scope"),
            retrieval_views=payload.get("retrieval_views"),
            project_id=str(payload.get("project_id") or ""),
            tenant_id=(str(payload["tenant_id"]) if payload.get("tenant_id") is not None else None),
            applicability_scopes=payload.get("applicability_scopes"),
            record_kinds=payload.get("record_kinds"),
            document_ids=payload.get("document_ids"),
            document_kinds=payload.get("document_kinds"),
            query_intent=payload.get("query_intent"),
            caller=caller,
        )
    if route == "POST /sessions/commit":
        result = client.commit_agent_session(
            user_id=_required_str(payload, "user_id", route),
            session_id=_required_str(payload, "session_id", route),
            messages=payload.get("messages"),
            used_contexts=payload.get("used_contexts"),
            used_skills=payload.get("used_skills"),
            tool_results=payload.get("tool_results"),
            connect_metadata=payload.get("connect_metadata"),
            async_commit=bool(payload.get("async_commit", True)),
            project_id=str(payload.get("project_id") or ""),
            session_key=str(payload.get("session_key") or ""),
            scope=payload.get("scope"),
            provenance=payload.get("provenance"),
            tenant_id=(str(payload["tenant_id"]) if payload.get("tenant_id") is not None else None),
            caller=caller,
        )
        if result is None:
            return {"status": "accepted"}
        return {
            "status": result.status,
            "task_id": result.task_id,
            "archive_uri": result.archive_uri,
            "done": result.done,
            "state": result.state.value,
            "commit_group_id": result.commit_group_id,
            "memory_committed": result.memory_committed,
            "memory_document_change_count": result.memory_document_change_count,
            "edit_proposal_count": result.edit_proposal_count,
            "edit_proposal_ids": list(result.edit_proposal_ids),
            "archive_committed": result.archive_committed,
            "session_projection_status": result.session_projection_status,
            "session_projected_count": result.session_projected_count,
            "commit_group_status": result.commit_group_status,
        }
    raise KeyError(f"Unknown route: {route}")


class MemoryOSASGI:
    def __init__(
        self,
        client: MemoryOSClient,
        *,
        api_token: str | None = None,
        trusted_context: TrustedRequestContext | None = None,
        allow_unauthenticated_local: bool = False,
        max_body_bytes: int = 2_000_000,
    ) -> None:
        self.client = client
        self.api_token = api_token
        self.allow_unauthenticated_local = allow_unauthenticated_local
        self.trusted_context = trusted_context or _trusted_context_from_env()
        if str(getattr(client, "tenant_id", "default")) != self.trusted_context.tenant_id:
            raise ValueError("HTTP client tenant does not match trusted caller tenant")
        self.max_body_bytes = max_body_bytes
        self.sessions = client.agent_session_service

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            return
        request_id = self._request_id(scope)
        try:
            caller = self._authorize(scope)
            payload = await self._payload(scope, receive)
            body = self._dispatch(
                str(scope.get("method", "GET")),
                str(scope.get("path", "")),
                payload,
                scope,
                caller,
            )
            status = 200
        except AuthenticationError as exc:
            status, body = 401, self._error("UNAUTHORIZED", exc, False, request_id)
        except PermissionError as exc:
            status, body = 403, self._error("FORBIDDEN", exc, False, request_id)
        except RetrievalUnavailableError as exc:
            status, body = 503, self._error("RETRIEVAL_UNAVAILABLE", exc, True, request_id)
        except RuntimeNotReadyError as exc:
            status, body = 503, self._error("NOT_READY", exc, True, request_id)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            status, body = 400, self._error("BAD_REQUEST", exc, False, request_id)
        except FileNotFoundError as exc:
            status, body = 404, self._error("NOT_FOUND", exc, False, request_id)
        except Exception as exc:
            status, body = 500, self._error("INTERNAL_ERROR", exc, True, request_id)
        body["request_id"] = request_id
        raw = json.dumps(body, ensure_ascii=False).encode()
        headers = [(b"content-type", b"application/json"), (b"x-request-id", request_id.encode())]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": raw})

    def _dispatch(
        self,
        method: str,
        path: str,
        payload: dict[str, Any],
        scope: dict[str, Any],
        caller: TrustedRequestContext,
    ) -> dict[str, Any]:
        if method == "GET" and path == "/health":
            return self.client.health()
        # Health is the only public endpoint available before startup recovery
        # has established a complete document and projection serving state.
        # Session event/checkpoint routes write durable staging files directly
        # through AgentSessionService, so relying only on SDK method-level
        # gates would let those routes mutate state while NOT_READY.
        self.client.readiness.require_ready()
        if method == "POST" and path == "/v1/context/search":
            return handle(
                "POST /context/search",
                self.client,
                _bound_payload(payload, caller),
                caller=caller,
            )
        if method == "POST" and path == "/v1/context/assemble":
            return handle(
                "POST /context/assemble",
                self.client,
                _bound_payload(payload, caller),
                caller=caller,
            )
        if method == "POST" and path == "/v1/sessions/events":
            caller.require(COMMIT_SESSION)
            event_payload = _bound_session_event_payload(payload, caller)
            event_name = str(event_payload.get("event_type") or event_payload.get("hook_event_name") or "after_turn")
            event = AgentHookEvent.from_payload(
                event_payload,
                adapter_id=caller.actor_id,
                hook_name=event_name,
                user_id=caller.user_id,
            ).normalize()
            caller.assert_workspace(event.project_id)
            appended = self.sessions.append_event(event)
            self.sessions.append_transcript(event)
            return {"status": "ARCHIVED", "appended": appended, "session_key": event.session_key}
        if method == "POST" and path.endswith("/checkpoint") and path.startswith("/v1/sessions/"):
            caller.require(COMMIT_SESSION)
            session_key = path.split("/")[-2]
            self._require_session_owner(self._last_event(session_key), caller)
            return self.sessions.checkpoint(session_key)
        if method == "POST" and path.endswith("/finalize") and path.startswith("/v1/sessions/"):
            caller.require(COMMIT_SESSION)
            session_key = path.split("/")[-2]
            event = self._last_event(session_key)
            self._require_session_owner(event, caller)
            commit_payload = self.sessions.commit_payload(event)
            result = handle(
                "POST /sessions/commit",
                self.client,
                {
                    **commit_payload,
                    "connect_metadata": _connect_metadata(event),
                    "async_commit": bool(payload.get("async_commit", True)),
                    "scope": {
                        **dict(commit_payload.get("scope", {}) or {}),
                        "tenant_id": caller.tenant_id,
                    },
                },
                caller=caller,
            )
            self.sessions.finalize(session_key, commit_state="COMMITTED" if result.get("done") else "QUEUED")
            return result
        if method == "POST" and path == "/v1/memories/remember":
            caller.require(AUTHORITATIVE_REMEMBER)
            request = validate_memory_request("remember", payload)
            return validate_memory_response(
                "remember",
                self.client.remember(**request, tenant_id=caller.tenant_id, caller=caller),
            )
        if method == "POST" and path == "/v1/memories/adopt":
            caller.require(AUTHORITATIVE_REMEMBER)
            request = validate_memory_request("adopt", payload)
            return validate_memory_response(
                "adopt",
                self.client.adopt_memory_document(
                    **request,
                    tenant_id=caller.tenant_id,
                    caller=caller,
                ),
            )
        if method == "POST" and path == "/v1/memories/edit":
            caller.require(AUTHORITATIVE_REMEMBER)
            request = validate_memory_request("edit", payload)
            return validate_memory_response(
                "edit",
                self.client.edit_memory_document(**request, tenant_id=caller.tenant_id, caller=caller),
            )
        if method == "POST" and path == "/v1/memories/rename":
            caller.require(AUTHORITATIVE_REMEMBER)
            request = validate_memory_request("rename", payload)
            return validate_memory_response(
                "rename",
                self.client.rename_memory_document(**request, tenant_id=caller.tenant_id, caller=caller),
            )
        if method == "POST" and path == "/v1/memories/merge/propose":
            caller.require(AUTHORITATIVE_REMEMBER)
            caller.require(AUTHORITATIVE_FORGET)
            request = validate_memory_request("merge_propose", payload)
            return validate_memory_response(
                "merge_propose",
                self.client.propose_memory_consolidation(
                    **request,
                    tenant_id=caller.tenant_id,
                    caller=caller,
                ),
            )
        if method == "POST" and path == "/v1/memories/merge":
            caller.require(AUTHORITATIVE_REMEMBER)
            caller.require(AUTHORITATIVE_FORGET)
            request = validate_memory_request("merge", payload)
            return validate_memory_response(
                "merge",
                self.client.merge_memory_documents(**request, tenant_id=caller.tenant_id, caller=caller),
            )
        if method == "POST" and path == "/v1/memories/merge/resume":
            caller.require(AUTHORITATIVE_REMEMBER)
            caller.require(AUTHORITATIVE_FORGET)
            request = validate_memory_request("merge_resume", payload)
            return validate_memory_response(
                "merge_resume",
                self.client.resume_memory_consolidation(
                    **request,
                    tenant_id=caller.tenant_id,
                    caller=caller,
                ),
            )
        if method == "POST" and path == "/v1/memories/forget":
            caller.require(AUTHORITATIVE_FORGET)
            request = validate_memory_request("forget", payload)
            if request["mode"] == "HARD_ERASE":
                caller.require(HARD_ERASE_MEMORY)
            return validate_memory_response(
                "forget",
                self.client.forget(**request, tenant_id=caller.tenant_id, caller=caller),
            )
        if method == "GET" and path == "/v1/memories/history":
            caller.require(READ_CONTEXT)
            query = parse_qs(scope.get("query_string", b"").decode())
            request = validate_memory_request(
                "history",
                {"document_uri": str(query.get("document_uri", [""])[0])},
            )
            return validate_memory_response(
                "history",
                self.client.list_memory_history(**request, tenant_id=caller.tenant_id, caller=caller),
            )
        if method == "POST" and path == "/v1/memories/restore":
            caller.require(AUTHORITATIVE_REMEMBER)
            request = validate_memory_request("restore", payload)
            return validate_memory_response(
                "restore",
                self.client.restore_memory_revision(**request, tenant_id=caller.tenant_id, caller=caller),
            )
        if method == "POST" and path == "/v1/memories/review":
            caller.require(AUTHORITATIVE_REMEMBER)
            request = validate_memory_request("review", payload)
            return validate_memory_response(
                "review",
                self.client.review_memory_edit(**request, tenant_id=caller.tenant_id, caller=caller),
            )
        if method == "POST" and path == "/v1/memories/review/preview":
            caller.require(READ_CONTEXT)
            request = validate_memory_request("review_preview", payload)
            return validate_memory_response(
                "review_preview",
                self.client.preview_memory_edit(**request, tenant_id=caller.tenant_id, caller=caller),
            )
        if method == "GET" and path == "/v1/context/read":
            query = parse_qs(scope.get("query_string", b"").decode())
            return self.client.read(
                str(query.get("uri", [""])[0]),
                layer=str(query.get("layer", ["L2"])[0]),
                caller=caller,
            )
        if method == "GET" and path.startswith("/v1/recall-traces/"):
            return self.client.recall_trace(path.rsplit("/", 1)[-1], caller=caller)
        if method == "POST" and path == "/v1/archives/search":
            caller.require(READ_CONTEXT)
            caller.assert_identity(user_id=payload.get("user_id"), tenant_id=payload.get("tenant_id"))
            return {
                "results": self.client.archive_search(
                    _required_str(payload, "query", path),
                    user_id=caller.user_id,
                    limit=bounded_int(
                        payload.get("limit"),
                        default=20,
                        minimum=1,
                        maximum=100,
                        label="limit",
                    ),
                    tenant_id=caller.tenant_id,
                    caller=caller,
                    project_id=caller.bind_read_workspace(payload.get("project_id")),
                )
            }
        if method == "GET" and path == "/v1/archives/read":
            query = parse_qs(scope.get("query_string", b"").decode())
            return self.client.archive_read(str(query.get("archive_uri", [""])[0]), caller=caller)
        raise KeyError(f"unknown route: {method} {path}")

    async def _payload(self, scope: dict[str, Any], receive: Any) -> dict[str, Any]:
        if scope.get("method") == "GET":
            return {}
        chunks = bytearray()
        while True:
            message = await receive()
            chunks.extend(message.get("body", b""))
            if len(chunks) > self.max_body_bytes:
                raise ValueError("request body too large")
            if not message.get("more_body"):
                break
        value = json.loads(bytes(chunks) or b"{}")
        if not isinstance(value, dict):
            raise ValueError("request body must be an object")
        return value

    def _authorize(self, scope: dict[str, Any]) -> TrustedRequestContext:
        headers = {key.decode().lower(): value.decode() for key, value in scope.get("headers", [])}
        if not self.api_token:
            if not self.allow_unauthenticated_local or not _scope_client_is_loopback(scope):
                raise AuthenticationError("API token is required")
            return self.trusted_context
        expected = f"Bearer {self.api_token}"
        if not hmac.compare_digest(headers.get("authorization", ""), expected):
            raise AuthenticationError("invalid API token")
        self.trusted_context.assert_identity(
            user_id=headers.get("x-memoryos-user"),
            tenant_id=headers.get("x-memoryos-tenant"),
        )
        return self.trusted_context

    def _last_event(self, session_key: str) -> NormalizedAgentEvent:
        events = self.sessions.events(session_key)
        if not events:
            raise FileNotFoundError("live session not found")
        data = dict(events[-1])
        data["event_type"] = AgentEventType(str(data["event_type"]))
        return NormalizedAgentEvent(**data)

    def _require_session_owner(
        self,
        event: NormalizedAgentEvent,
        caller: TrustedRequestContext,
    ) -> None:
        if (
            event.user_id != caller.user_id
            or event.tenant_id != caller.tenant_id
            or event.adapter_id != caller.actor_id
            or event.project_id not in caller.allowed_workspace_ids
        ):
            raise FileNotFoundError("live session not found")

    def _request_id(self, scope: dict[str, Any]) -> str:
        headers = {key.decode().lower(): value.decode() for key, value in scope.get("headers", [])}
        return headers.get("x-request-id", "")[:128] or str(uuid.uuid4())

    def _error(self, code: str, exc: Exception, retryable: bool, request_id: str) -> dict[str, Any]:
        return {
            "error": {
                "code": code,
                "message": sanitize_error_text(str(exc) or type(exc).__name__),
                "retryable": retryable,
                "request_id": request_id,
                "operation": "http_request",
            }
        }


def create_app(
    root: str | None = None,
    *,
    api_token: str | None = None,
    allow_unauthenticated_local: bool = False,
) -> MemoryOSASGI:
    resolved_root = root or os.environ.get("MEMORYOS_ROOT") or "./memory-root"
    trusted_context = _trusted_context_from_env()
    return MemoryOSASGI(
        MemoryOSClient(resolved_root, mode="server", tenant_id=trusted_context.tenant_id),
        api_token=api_token if api_token is not None else os.environ.get("MEMORYOS_API_TOKEN"),
        trusted_context=trusted_context,
        allow_unauthenticated_local=allow_unauthenticated_local,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MemoryOS HTTP server")
    parser.add_argument("--root", default=os.environ.get("MEMORYOS_ROOT", "./memory-root"))
    parser.add_argument("--host", default=os.environ.get("MEMORYOS_HTTP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MEMORYOS_HTTP_PORT", "8765")))
    args = parser.parse_args(argv)
    api_token = os.environ.get("MEMORYOS_API_TOKEN")
    allow_unauthenticated_local = False
    if not api_token:
        try:
            allow_unauthenticated_local = ipaddress.ip_address(args.host).is_loopback
        except ValueError:
            allow_unauthenticated_local = False
        if not allow_unauthenticated_local:
            raise RuntimeError("MEMORYOS_API_TOKEN is required when binding HTTP to a non-loopback host")
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install memoryos[server] to run the HTTP server") from exc
    uvicorn.run(
        create_app(
            args.root,
            api_token=api_token,
            allow_unauthenticated_local=allow_unauthenticated_local,
        ),
        host=args.host,
        port=args.port,
    )


def _search_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "options": parse_retrieval_options(payload.get("options")),
        "user_id": payload.get("user_id"),
        "context_type": payload.get("context_type"),
        "limit": bounded_int(
            payload.get("limit"),
            default=10,
            minimum=1,
            maximum=MAX_RETRIEVAL_LIMIT,
            label="limit",
        ),
        "connect_metadata": payload.get("connect_metadata"),
        "search_scope": payload.get("search_scope"),
        "retrieval_views": payload.get("retrieval_views"),
        "project_id": str(payload.get("project_id") or ""),
        "tenant_id": (str(payload["tenant_id"]) if payload.get("tenant_id") is not None else None),
        "applicability_scopes": payload.get("applicability_scopes"),
        "record_kinds": payload.get("record_kinds"),
        "document_ids": payload.get("document_ids"),
        "document_kinds": payload.get("document_kinds"),
        "query_intent": payload.get("query_intent"),
    }


def _bound_payload(payload: dict[str, Any], caller: TrustedRequestContext) -> dict[str, Any]:
    caller.require(READ_CONTEXT)
    options = parse_retrieval_options(payload.get("options"))
    caller.assert_identity(user_id=payload.get("user_id"), tenant_id=payload.get("tenant_id"))
    if options is not None:
        caller.assert_identity(user_id=options.owner_user_id, tenant_id=options.tenant_id)
    requested_project = str(payload.get("project_id") or "").strip()
    option_workspaces = options.workspace_ids if options is not None else ()
    if len(option_workspaces) > 1:
        raise PermissionError("caller must select one authorized workspace")
    if requested_project and option_workspaces and option_workspaces != (requested_project,):
        raise PermissionError("structured options conflict with project_id")
    project_id = caller.bind_read_workspace(requested_project or (option_workspaces[0] if option_workspaces else None))
    caller.assert_applicability_scopes(
        payload.get("applicability_scopes"),
        workspace_id=project_id,
    )
    if options is not None:
        caller.assert_applicability_scope_keys(
            options.metadata_filters.get("applicability_scope_keys"),
            workspace_id=project_id,
        )
    if options is not None and options.adapter_id is not None and options.adapter_id != caller.actor_id:
        raise PermissionError("caller adapter_id does not match trusted actor")
    return {
        **payload,
        "options": options,
        "user_id": caller.user_id,
        "tenant_id": caller.tenant_id,
        "project_id": project_id,
        "connect_metadata": caller.bind_agent_connect_metadata(payload.get("connect_metadata")),
    }


def _bound_session_event_payload(
    payload: dict[str, Any],
    caller: TrustedRequestContext,
) -> dict[str, Any]:
    caller.assert_identity(user_id=payload.get("user_id"), tenant_id=payload.get("tenant_id"))
    if payload.get("adapter_id") is not None and payload.get("adapter_id") != caller.actor_id:
        raise PermissionError("caller adapter_id does not match trusted actor")
    result = dict(payload)
    result.pop("transcript_path", None)
    result.pop("transcript_cursor", None)
    prompt = result.pop("prompt", None)
    user_prompt = result.pop("user_prompt", None)
    raw_input = result.pop("input", None)
    prompt = prompt or user_prompt or raw_input
    messages = list(result.get("messages") or [])
    if prompt:
        messages.append(
            {
                "id": str(result.get("event_id") or "prompt"),
                "role": "user",
                "content": str(prompt),
            }
        )
    metadata = dict(result.get("metadata", {}) or {}) if isinstance(result.get("metadata"), dict) else {}
    for key in (
        "actor_id",
        "actor_kind",
        "asserted_by",
        "authority",
        "effect_authority",
        "source_role",
        "transcript_path",
        "transcript_cursor",
    ):
        metadata.pop(key, None)
    metadata.update(
        {
            "tenant_id": caller.tenant_id,
            "ingress_actor_kind": caller.actor_kind,
            "ingress_actor_id": caller.actor_id,
        }
    )
    result.update(
        {
            "adapter_id": caller.actor_id,
            "user_id": caller.user_id,
            "tenant_id": caller.tenant_id,
            "prompt": None,
            "messages": sanitize_ingress_messages(messages, caller),
            "metadata": metadata,
        }
    )
    return result


def _connect_metadata(event: NormalizedAgentEvent) -> dict[str, Any]:
    return {
        "connect_type": "agent",
        "adapter_id": event.adapter_id,
        "run_mode": "context_reduction",
        "world_domain": "digital",
        "source_kind": "coding_agent",
        "extra": {"project_id": event.project_id},
    }


def _trusted_context_from_env() -> TrustedRequestContext:
    actor_kind = os.environ.get("MEMORYOS_ACTOR_KIND", "agent")
    user_id = os.environ.get("MEMORYOS_USER_ID", "default")
    actor_id = os.environ.get("MEMORYOS_ACTOR_ID") or (
        user_id if actor_kind == "user" else os.environ.get("MEMORYOS_ADAPTER_ID", "generic_agent")
    )
    return TrustedRequestContext(
        tenant_id=os.environ.get("MEMORYOS_TENANT_ID", "default"),
        user_id=user_id,
        actor_kind=actor_kind,
        actor_id=actor_id,
        capabilities=capabilities_from_csv(os.environ.get("MEMORYOS_HTTP_CAPABILITIES")),
        allowed_workspace_ids=workspace_ids_from_csv(os.environ.get("MEMORYOS_WORKSPACE_IDS")),
        authorized_scope_keys=scope_keys_from_csv(os.environ.get("MEMORYOS_AUTHORIZED_SCOPE_KEYS")),
    )


def _required_str(payload: dict[str, Any], key: str, route: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{route} requires non-empty string field: {key}")
    return value


def _scope_client_is_loopback(scope: dict[str, Any]) -> bool:
    client = scope.get("client")
    if not isinstance(client, tuple | list) or not client:
        return False
    host = client[0]
    if not isinstance(host, str):
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
