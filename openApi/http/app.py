"""MemoryOS 本地 HTTP/ASGI 对外交付入口。

HTTP 只允许监听回环地址，使用进程配置中的唯一用户，不再实现 Token、多租户或
Capability 授权。输入限制、数据脱敏和领域校验仍由对应模块执行。
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import uuid
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs

from agent_hook.events import AgentEventType, AgentHookEvent, NormalizedAgentEvent
from foundation.identity import LocalUserContext
from foundation.readiness import RuntimeNotReadyError
from infrastructure.context.orchestrator import RetrievalUnavailableError
from infrastructure.context.retrieval.limits import MAX_RETRIEVAL_LIMIT, bounded_int
from openApi.http.config import HTTPServerConfig
from openApi.ingress import local_agent_metadata, sanitize_ingress_messages
from openApi.retrieval_contract import parse_retrieval_options
from policy.action_policy.decision.request import PredictionRequest
from policy.action_policy.model.action_policy import ActionPolicy
from sanitization import sanitize_error_text

if TYPE_CHECKING:
    from openApi.sdk.client import MemoryOSClient


def handle(
    route: str,
    client: MemoryOSClient,
    payload: dict[str, Any],
    *,
    caller: LocalUserContext | None = None,
) -> dict[str, Any]:
    """把已绑定本地用户的通用 HTTP 操作委托给进程内 SDK。"""

    if route == "POST /predict":
        request_payload = payload.get("request")
        if not isinstance(request_payload, dict):
            raise ValueError("POST /predict requires object field: request")
        policies = [ActionPolicy(**item) for item in payload.get("policies", [])]
        return client.predict(PredictionRequest(**request_payload), policies).to_dict()
    if route == "POST /context/search":
        results = client.search_context(
            _required_str(payload, "query", route),
            **_search_kwargs(payload),
            caller=caller,
        )
        response: dict[str, Any] = {"results": results}
        trace_id = str(getattr(client, "last_recall_trace_id", "") or "")
        if trace_id:
            response["trace_id"] = trace_id
        return response
    if route == "POST /context/assemble":
        return client.assemble_context(
            _required_str(payload, "query", route),
            options=parse_retrieval_options(payload.get("options")),
            user_id=payload.get("user_id"),
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
            applicability_scopes=payload.get("applicability_scopes"),
            record_kinds=payload.get("record_kinds"),
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
            "archive_committed": result.archive_committed,
            "session_projection_status": result.session_projection_status,
            "session_projected_count": result.session_projected_count,
            "commit_group_status": result.commit_group_status,
        }
    raise KeyError(f"Unknown route: {route}")


class MemoryOSASGI:
    """面向本机插件的无框架 ASGI 应用。"""

    def __init__(
        self,
        client: MemoryOSClient,
        *,
        local_context: LocalUserContext | None = None,
        max_body_bytes: int = 2_000_000,
    ) -> None:
        self.client = client
        self.local_context = local_context or _local_context_from_env()
        self.max_body_bytes = max_body_bytes
        self.sessions = client.runtime.agent.session_service

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        """完成一次本地请求的解析、分发和统一错误响应。"""

        if scope.get("type") != "http":
            return
        request_id = self._request_id(scope)
        try:
            caller = self.local_context
            payload = await self._payload(scope, receive)
            body = self._dispatch(
                str(scope.get("method", "GET")),
                str(scope.get("path", "")),
                payload,
                scope,
                caller,
            )
            status = 200
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
        caller: LocalUserContext,
    ) -> dict[str, Any]:
        """将本机请求映射到公开能力；用户身份来自进程启动配置。"""

        if method == "GET" and path == "/health":
            return self.client.health()
        # 启动恢复尚未建立完整的上下文和投影服务状态时，健康检查是唯一可用的公开端点。
        # 会话事件和检查点路由会通过 AgentSessionService 直接写入耐久暂存文件，
        # 因此只依赖 SDK 方法级门控，会让这些路由在 NOT_READY 状态下修改数据。
        self.client.runtime.readiness.require_ready()
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
            event_payload = _bound_session_event_payload(payload, caller)
            event_name = str(event_payload.get("event_type") or event_payload.get("hook_event_name") or "after_turn")
            event = AgentHookEvent.from_payload(
                event_payload,
                adapter_id=caller.actor_id,
                hook_name=event_name,
                user_id=caller.user_id,
            ).normalize()
            appended = self.sessions.append_event(event)
            self.sessions.append_transcript(event)
            return {"status": "ARCHIVED", "appended": appended, "session_key": event.session_key}
        if method == "POST" and path.endswith("/checkpoint") and path.startswith("/v1/sessions/"):
            session_key = path.split("/")[-2]
            self._require_session_owner(self._last_event(session_key), caller)
            return self.sessions.checkpoint(session_key)
        if method == "POST" and path.endswith("/finalize") and path.startswith("/v1/sessions/"):
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
                    "scope": dict(commit_payload.get("scope", {}) or {}),
                },
                caller=caller,
            )
            self.sessions.finalize(session_key, commit_state="COMMITTED" if result.get("done") else "QUEUED")
            return result
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
            if "tenant_id" in payload:
                raise ValueError("tenant_id is unavailable in local single-user mode")
            caller.assert_identity(user_id=payload.get("user_id"))
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
        caller: LocalUserContext,
    ) -> None:
        if (
            event.user_id != caller.user_id
            or event.tenant_id != caller.tenant_id
            or event.adapter_id != caller.actor_id
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
    user_id: str | None = None,
    adapter_id: str | None = None,
) -> MemoryOSASGI:
    """创建绑定一个本地用户的 ASGI 应用。"""

    # 构造 HTTP 服务时才加载进程内 SDK，导入协议模块不会提前创建运行时依赖。
    from openApi.sdk.client import MemoryOSClient

    http_config = HTTPServerConfig.from_env()
    resolved_root = root or http_config.root
    local_context = _local_context_from_env(user_id=user_id, adapter_id=adapter_id)
    return MemoryOSASGI(
        MemoryOSClient(
            resolved_root,
            mode="server",
            model_config=http_config.model,
        ),
        local_context=local_context,
    )


def run(argv: list[str] | None = None) -> None:
    """校验监听地址安全策略并启动 Uvicorn HTTP 服务。"""

    http_config = HTTPServerConfig.from_env()
    parser = argparse.ArgumentParser(description="MemoryOS HTTP server")
    parser.add_argument("--root", default=http_config.root)
    parser.add_argument("--host", default=http_config.host)
    parser.add_argument("--port", type=int, default=http_config.port)
    args = parser.parse_args(argv)
    try:
        loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = args.host.casefold() == "localhost"
    if not loopback:
        raise RuntimeError("single-user MemoryOS HTTP may only bind to a loopback host")
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install memoryos[server] to run the HTTP server") from exc
    uvicorn.run(
        create_app(args.root),
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
        "applicability_scopes": payload.get("applicability_scopes"),
        "record_kinds": payload.get("record_kinds"),
        "query_intent": payload.get("query_intent"),
    }


def _bound_payload(payload: dict[str, Any], caller: LocalUserContext) -> dict[str, Any]:
    """用进程配置绑定本地用户和工作区。"""

    options = parse_retrieval_options(payload.get("options"))
    if "tenant_id" in payload:
        raise ValueError("tenant_id is unavailable in local single-user mode")
    caller.assert_identity(user_id=payload.get("user_id"))
    if options is not None:
        caller.assert_identity(user_id=options.owner_user_id, tenant_id=options.tenant_id)
    requested_project = str(payload.get("project_id") or "").strip()
    option_workspaces = options.workspace_ids if options is not None else ()
    if len(option_workspaces) > 1:
        raise PermissionError("local query must select one workspace")
    if requested_project and option_workspaces and option_workspaces != (requested_project,):
        raise PermissionError("structured options conflict with project_id")
    project_id = caller.bind_read_workspace(requested_project or (option_workspaces[0] if option_workspaces else None))
    return {
        **payload,
        "options": options,
        "user_id": caller.user_id,
        "project_id": project_id,
        "connect_metadata": local_agent_metadata(payload.get("connect_metadata"), caller),
    }


def _bound_session_event_payload(
    payload: dict[str, Any],
    caller: LocalUserContext,
) -> dict[str, Any]:
    if "tenant_id" in payload:
        raise ValueError("tenant_id is unavailable in local single-user mode")
    caller.assert_identity(user_id=payload.get("user_id"))
    if payload.get("adapter_id") is not None and payload.get("adapter_id") != caller.actor_id:
        raise PermissionError("adapter_id does not match the configured local adapter")
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


def _local_context_from_env(
    *,
    user_id: str | None = None,
    adapter_id: str | None = None,
) -> LocalUserContext:
    return LocalUserContext(
        user_id=user_id or os.environ.get("MEMORYOS_USER_ID") or "local-user",
        adapter_id=adapter_id or os.environ.get("MEMORYOS_ADAPTER_ID") or "codex",
        workspace_id=os.environ.get("MEMORYOS_WORKSPACE_ID", ""),
    )


def _required_str(payload: dict[str, Any], key: str, route: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{route} requires non-empty string field: {key}")
    return value
