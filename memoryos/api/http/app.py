"""HTTP 应用入口。"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import uuid
from typing import Any
from urllib.parse import parse_qs

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.adapters.agent_hooks.events import AgentEventType, AgentHookEvent, NormalizedAgentEvent
from memoryos.adapters.agent_hooks.session_service import AgentSessionService
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.prediction.model.prediction_request import PredictionRequest


def handle(route: str, client: MemoryOSClient, payload: dict[str, Any]) -> dict[str, Any]:
    if route == "POST /predict":
        request_payload = payload.get("request")
        if not isinstance(request_payload, dict):
            raise ValueError("POST /predict requires object field: request")
        policies = [ActionPolicy(**item) for item in payload.get("policies", [])]
        return client.predict(PredictionRequest(**request_payload), policies).to_dict()
    if route == "POST /context/search":
        return {"results": client.search_context(_required_str(payload, "query", route), **_search_kwargs(payload))}
    if route == "POST /context/assemble":
        return client.assemble_context(
            _required_str(payload, "query", route),
            user_id=payload.get("user_id"),
            token_budget=int(payload.get("token_budget", 2000)),
            context_types=payload.get("context_types"),
            limit=int(payload.get("limit", 20)),
            connect_metadata=payload.get("connect_metadata"),
            search_scope=payload.get("search_scope"),
            retrieval_views=payload.get("retrieval_views"),
            project_id=str(payload.get("project_id") or ""),
            tenant_id=str(payload.get("tenant_id") or "default"),
            applicability_scopes=payload.get("applicability_scopes"),
            memory_states=payload.get("memory_states"),
            memory_types=payload.get("memory_types"),
            claim_uris=payload.get("claim_uris"),
            slot_uris=payload.get("slot_uris"),
            query_intent=payload.get("query_intent"),
        )
    if route == "POST /sessions/commit":
        result = client.commit_agent_session(
            user_id=_required_str(payload, "user_id", route),
            session_id=_required_str(payload, "session_id", route),
            messages=payload.get("messages"),
            used_contexts=payload.get("used_contexts"),
            tool_results=payload.get("tool_results"),
            connect_metadata=payload.get("connect_metadata"),
            async_commit=bool(payload.get("async_commit", True)),
            project_id=str(payload.get("project_id") or ""),
            session_key=str(payload.get("session_key") or ""),
            scope=payload.get("scope"),
            provenance=payload.get("provenance"),
        )
        if result is None:
            return {"status": "accepted"}
        return {
            "status": result.status,
            "task_id": result.task_id,
            "archive_uri": result.archive_uri,
            "done": result.done,
        }
    raise KeyError(f"Unknown route: {route}")


class MemoryOSASGI:
    def __init__(
        self, client: MemoryOSClient, *, api_token: str | None = None, max_body_bytes: int = 2_000_000
    ) -> None:
        self.client = client
        self.api_token = api_token
        self.max_body_bytes = max_body_bytes
        self.sessions = AgentSessionService(client.root)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            return
        request_id = self._request_id(scope)
        try:
            self._authorize(scope)
            payload = await self._payload(scope, receive)
            body = self._dispatch(str(scope.get("method", "GET")), str(scope.get("path", "")), payload, scope)
            status = 200
        except PermissionError as exc:
            status, body = 401, self._error("UNAUTHORIZED", exc, False, request_id)
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

    def _dispatch(self, method: str, path: str, payload: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
        if method == "GET" and path == "/health":
            return self.client.health()
        if method == "POST" and path == "/v1/context/search":
            return handle("POST /context/search", self.client, payload)
        if method == "POST" and path == "/v1/context/assemble":
            return handle("POST /context/assemble", self.client, payload)
        if method == "POST" and path == "/v1/sessions/events":
            event_name = str(payload.get("event_type") or payload.get("hook_event_name") or "after_turn")
            event = AgentHookEvent.from_payload(
                payload,
                adapter_id=str(payload.get("adapter_id") or "generic_agent"),
                hook_name=event_name,
                user_id=str(payload.get("user_id") or "default"),
            ).normalize()
            appended = self.sessions.append_event(event)
            self.sessions.append_transcript(event)
            return {"status": "ARCHIVED", "appended": appended, "session_key": event.session_key}
        if method == "POST" and path.endswith("/checkpoint") and path.startswith("/v1/sessions/"):
            return self.sessions.checkpoint(path.split("/")[-2])
        if method == "POST" and path.endswith("/finalize") and path.startswith("/v1/sessions/"):
            session_key = path.split("/")[-2]
            event = self._last_event(session_key)
            result = handle(
                "POST /sessions/commit",
                self.client,
                {
                    **self.sessions.commit_payload(event),
                    "connect_metadata": _connect_metadata(event),
                    "async_commit": bool(payload.get("async_commit", True)),
                },
            )
            self.sessions.finalize(session_key, commit_state="COMMITTED" if result.get("done") else "QUEUED")
            return result
        if method == "POST" and path == "/v1/memories/remember":
            return self.client.remember(
                user_id=_required_str(payload, "user_id", path),
                content=_required_str(payload, "content", path),
                title=str(payload.get("title") or ""),
                memory_type=str(payload.get("memory_type") or "project_decision"),
                project_id=str(payload.get("project_id") or ""),
                constraint_polarity=str(payload.get("constraint_polarity") or ""),
                condition=str(payload.get("condition") or ""),
                exception=str(payload.get("exception") or ""),
                connect_metadata=payload.get("connect_metadata"),
            )
        if method == "POST" and path == "/v1/memories/forget":
            return self.client.forget(
                user_id=_required_str(payload, "user_id", path), uri=_required_str(payload, "uri", path)
            )
        if method == "GET" and path == "/v1/memories/pending":
            query = parse_qs(scope.get("query_string", b"").decode())
            lifecycle_states = [
                item
                for value in query.get("lifecycle_state", [])
                for item in str(value).split(",")
                if item
            ]
            return {
                "results": self.client.list_pending(
                    user_id=str(query.get("user_id", [""])[0]),
                    tenant_id=str(query.get("tenant_id", ["default"])[0]),
                    lifecycle_states=lifecycle_states,
                )
            }
        if method == "POST" and path == "/v1/memories/pending/review":
            return self.client.review_pending(
                user_id=_required_str(payload, "user_id", path),
                pending_uri=_required_str(payload, "pending_uri", path),
                decision=_required_str(payload, "decision", path),
                expected_lifecycle_revision=int(payload.get("expected_lifecycle_revision", 0) or 0),
                expected_proposal_fingerprint=_required_str(
                    payload,
                    "expected_proposal_fingerprint",
                    path,
                ),
                command_id=_required_str(payload, "command_id", path),
                tenant_id=str(payload.get("tenant_id") or "default"),
                reason=str(payload.get("reason") or ""),
            )
        if method == "GET" and path == "/v1/context/read":
            query = parse_qs(scope.get("query_string", b"").decode())
            return self.client.read(str(query.get("uri", [""])[0]), layer=str(query.get("layer", ["L2"])[0]))
        if method == "GET" and path.startswith("/v1/recall-traces/"):
            return self.client.recall_trace(path.rsplit("/", 1)[-1])
        if method == "POST" and path == "/v1/archives/search":
            return {
                "results": self.client.archive_search(
                    _required_str(payload, "query", path),
                    user_id=_required_str(payload, "user_id", path),
                    limit=int(payload.get("limit", 20)),
                )
            }
        if method == "GET" and path == "/v1/archives/read":
            query = parse_qs(scope.get("query_string", b"").decode())
            return self.client.archive_read(str(query.get("archive_uri", [""])[0]))
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

    def _authorize(self, scope: dict[str, Any]) -> None:
        if not self.api_token:
            return
        headers = {key.decode().lower(): value.decode() for key, value in scope.get("headers", [])}
        expected = f"Bearer {self.api_token}"
        if not hmac.compare_digest(headers.get("authorization", ""), expected):
            raise PermissionError("invalid API token")

    def _last_event(self, session_key: str) -> NormalizedAgentEvent:
        events = self.sessions.events(session_key)
        if not events:
            raise FileNotFoundError("live session not found")
        data = dict(events[-1])
        data["event_type"] = AgentEventType(str(data["event_type"]))
        return NormalizedAgentEvent(**data)

    def _request_id(self, scope: dict[str, Any]) -> str:
        headers = {key.decode().lower(): value.decode() for key, value in scope.get("headers", [])}
        return headers.get("x-request-id", "")[:128] or str(uuid.uuid4())

    def _error(self, code: str, exc: Exception, retryable: bool, request_id: str) -> dict[str, Any]:
        return {
            "error": {
                "code": code,
                "message": str(exc)[:300],
                "retryable": retryable,
                "request_id": request_id,
                "operation": "http_request",
            }
        }


def create_app(root: str | None = None, *, api_token: str | None = None) -> MemoryOSASGI:
    resolved_root = root or os.environ.get("MEMORYOS_ROOT") or "./memory-root"
    return MemoryOSASGI(
        MemoryOSClient(resolved_root, mode="server"),
        api_token=api_token if api_token is not None else os.environ.get("MEMORYOS_API_TOKEN"),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MemoryOS HTTP server")
    parser.add_argument("--root", default=os.environ.get("MEMORYOS_ROOT", "./memory-root"))
    parser.add_argument("--host", default=os.environ.get("MEMORYOS_HTTP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MEMORYOS_HTTP_PORT", "8765")))
    args = parser.parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install memoryos[server] to run the HTTP server") from exc
    uvicorn.run(create_app(args.root), host=args.host, port=args.port)


def _search_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": payload.get("user_id"),
        "context_type": payload.get("context_type"),
        "limit": int(payload.get("limit", 10)),
        "connect_metadata": payload.get("connect_metadata"),
        "search_scope": payload.get("search_scope"),
        "retrieval_views": payload.get("retrieval_views"),
        "project_id": str(payload.get("project_id") or ""),
        "tenant_id": str(payload.get("tenant_id") or "default"),
        "applicability_scopes": payload.get("applicability_scopes"),
        "memory_states": payload.get("memory_states"),
        "memory_types": payload.get("memory_types"),
        "claim_uris": payload.get("claim_uris"),
        "slot_uris": payload.get("slot_uris"),
        "query_intent": payload.get("query_intent"),
    }


def _connect_metadata(event: NormalizedAgentEvent) -> dict[str, Any]:
    return {
        "connect_type": "agent",
        "adapter_id": event.adapter_id,
        "run_mode": "context_reduction",
        "world_domain": "digital",
        "source_kind": "coding_agent",
        "extra": {"project_id": event.project_id},
    }


def _required_str(payload: dict[str, Any], key: str, route: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{route} requires non-empty string field: {key}")
    return value
