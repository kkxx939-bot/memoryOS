from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import memoryos
from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.errors import MCPErrorCode, ToolPermissionError, exception_payload, ok_payload
from memoryos.api.mcp.schemas import (
    agent_search_filter_metadata,
    connection_schema,
    normalize_action_metadata,
    normalize_agent_metadata,
    optional_bool,
    optional_int,
    optional_list,
    require_process_observation_metadata,
    required_str,
)
from memoryos.prediction.model.prediction_request import PredictionRequest


class MCPToolRouter:
    def __init__(self, client: Any, config: MCPServerConfig | None = None) -> None:
        self.client = client
        self.config = config or MCPServerConfig.from_env()

    def call(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        args = dict(arguments or {})
        try:
            if name == "memoryos_search_context":
                return self.search_context(args)
            if name == "memoryos_assemble_context":
                return self.assemble_context(args)
            if name == "memoryos_commit_session":
                return self.commit_session(args)
            if name == "memoryos_health":
                return self.health()
            if name == "memoryos_read":
                return ok_payload(self.client.read(required_str(args, "uri"), layer=str(args.get("layer") or "L2")))
            if name == "memoryos_remember":
                metadata = normalize_agent_metadata(args.get("connect_metadata"), self.config)
                return ok_payload(self.client.remember(user_id=str(args.get("user_id") or self.config.user_id), content=required_str(args, "content"), title=str(args.get("title") or ""), memory_type=str(args.get("memory_type") or "project_decision"), project_id=str(args.get("project_id") or ""), connect_metadata=metadata))
            if name == "memoryos_forget":
                return ok_payload(self.client.forget(user_id=str(args.get("user_id") or self.config.user_id), uri=required_str(args, "uri")))
            if name == "memoryos_archive_search":
                return ok_payload({"results": self.client.archive_search(required_str(args, "query"), user_id=str(args.get("user_id") or self.config.user_id), limit=optional_int(args, "limit", 20, maximum=100))})
            if name == "memoryos_archive_read":
                return ok_payload(self.client.archive_read(required_str(args, "archive_uri")))
            if name == "memoryos_recall_trace":
                return ok_payload(self.client.recall_trace(required_str(args, "trace_id")))
            if name == "memoryos_connection_schema":
                return ok_payload(connection_schema(self.config))
            if name == "memoryos_predict":
                return self.predict(args)
            if name == "memoryos_process_observation":
                return self.process_observation(args)
            return {"error": {"code": MCPErrorCode.VALIDATION_ERROR, "message": f"Unknown tool: {name}", "retryable": False, "details": {}}}
        except Exception as exc:  # Tools are a fail-safe boundary for external agents.
            return exception_payload(exc)

    def search_context(self, args: dict[str, Any]) -> dict[str, Any]:
        query = required_str(args, "query")
        limit = optional_int(args, "limit", 10, minimum=0, maximum=100)
        metadata = normalize_agent_metadata(args.get("connect_metadata"), self.config)
        filter_metadata = agent_search_filter_metadata(args.get("connect_metadata"), self.config)
        context_type = args.get("context_type")
        context_types = optional_list(args, "context_types")
        search_scope = args.get("search_scope")
        project_id = str(args.get("project_id") or "")
        retrieval_views = optional_list(args, "retrieval_views")
        requested_types = [context_type] if context_type is not None else list(context_types or [])
        if requested_types:
            contexts = []
            for requested_type in requested_types:
                contexts.extend(
                    self.client.search_context(
                        query,
                        user_id=args.get("user_id") or self.config.user_id,
                        context_type=requested_type,
                        limit=limit,
                        connect_metadata=filter_metadata,
                        search_scope=str(search_scope) if search_scope else None,
                        retrieval_views=[str(item) for item in retrieval_views or []],
                        project_id=project_id,
                    )
                )
            contexts = _dedupe_contexts(contexts)[:limit]
        else:
            contexts = self.client.search_context(
                query,
                user_id=args.get("user_id") or self.config.user_id,
                context_type=None,
                limit=limit,
                connect_metadata=filter_metadata,
                search_scope=str(search_scope) if search_scope else None,
                retrieval_views=[str(item) for item in retrieval_views or []],
                project_id=project_id,
            )
        source_uris = [str(item.get("uri", "")) for item in contexts if item.get("uri")]
        return ok_payload({"contexts": contexts, "results": contexts, "source_uris": source_uris, "metadata": {"connect": metadata}})

    def assemble_context(self, args: dict[str, Any]) -> dict[str, Any]:
        query = required_str(args, "query")
        token_budget = optional_int(args, "token_budget", self.config.token_budget, minimum=0, maximum=200_000)
        limit = optional_int(args, "limit", 20, minimum=0, maximum=200)
        context_types = optional_list(args, "context_types")
        search_scope = args.get("search_scope")
        project_id = str(args.get("project_id") or "")
        retrieval_views = optional_list(args, "retrieval_views")
        metadata = normalize_agent_metadata(args.get("connect_metadata"), self.config)
        filter_metadata = agent_search_filter_metadata(args.get("connect_metadata"), self.config)
        assembled = self.client.assemble_context(
            query,
            user_id=args.get("user_id") or self.config.user_id,
            token_budget=token_budget,
            context_types=context_types,
            limit=limit,
            connect_metadata=filter_metadata,
            search_scope=str(search_scope) if search_scope else None,
            retrieval_views=[str(item) for item in retrieval_views or []],
            project_id=project_id,
        )
        payload = {
            "packed_context": assembled.get("packed_context", ""),
            "contexts": assembled.get("contexts", []),
            "source_uris": assembled.get("source_uris", []),
            "dropped_contexts": assembled.get("dropped_contexts", []),
            "token_budget": token_budget,
            "estimated_tokens": _estimate_tokens(str(assembled.get("packed_context", ""))),
            "metadata": {"connect": metadata},
        }
        return ok_payload(payload)

    def commit_session(self, args: dict[str, Any]) -> dict[str, Any]:
        session_id = required_str(args, "session_id")
        user_id = str(args.get("user_id") or self.config.user_id)
        metadata = normalize_agent_metadata(args.get("connect_metadata"), self.config)
        result = self.client.commit_agent_session(
            user_id=user_id,
            session_id=session_id,
            messages=list(args.get("messages") or []),
            used_contexts=list(args.get("used_contexts") or []),
            tool_results=list(args.get("tool_results") or []),
            connect_metadata=metadata,
            async_commit=optional_bool(args, "async_commit", False),
            project_id=str(args.get("project_id") or ""),
            session_key=str(args.get("session_key") or ""),
            scope=dict(args.get("scope") or {}),
            provenance=dict(args.get("provenance") or {}),
        )
        result_payload = _to_payload(result) or {"status": "accepted"}
        return ok_payload({"status": result_payload.get("status", "accepted"), "result": result_payload, "metadata": {"connect": metadata}})

    def health(self) -> dict[str, Any]:
        metadata = {
            "root_configured": bool(self.config.root),
            "adapter_id": self.config.adapter_id,
        }
        health_fn = getattr(self.client, "health", None)
        raw_health = health_fn() if callable(health_fn) else {}
        health: dict[str, Any] = raw_health if isinstance(raw_health, dict) else {}
        return ok_payload(
            {
                "status": "ok",
                **health,
                "storage_ready": hasattr(self.client, "source_store"),
                "contextdb_ready": hasattr(self.client, "context_db"),
                "client_ready": self.client is not None,
                "version": memoryos.__version__,
                "metadata": metadata,
            }
        )

    def predict(self, args: dict[str, Any]) -> dict[str, Any]:
        self._ensure_action_tools_enabled()
        request_payload = args.get("request")
        if not isinstance(request_payload, dict):
            raise ValueError("requires object field: request")
        metadata = normalize_action_metadata(request_payload.get("connect_metadata") or args.get("connect_metadata"))
        request_payload = {**request_payload, "connect_metadata": metadata.to_dict()}
        result = self.client.predict(_prediction_request(request_payload), _policies(args.get("policies")))
        return ok_payload({"prediction": _to_payload(result)})

    def process_observation(self, args: dict[str, Any]) -> dict[str, Any]:
        self._ensure_action_tools_enabled()
        request_payload = args.get("request")
        if not isinstance(request_payload, dict):
            raise ValueError("requires object field: request")
        metadata = require_process_observation_metadata(request_payload.get("connect_metadata") or args.get("connect_metadata"))
        request_payload = {**request_payload, "connect_metadata": metadata.to_dict()}
        result = self.client.process_observation(
            _prediction_request(request_payload),
            _policies(args.get("policies")),
            archive_session=optional_bool(args, "archive_session", True),
            async_commit=optional_bool(args, "async_commit", True),
        )
        return ok_payload({"result": _to_payload(result)})

    def _ensure_action_tools_enabled(self) -> None:
        if not self.config.enable_action_tools:
            raise ToolPermissionError("action-capable tools are disabled; set MEMORYOS_ENABLE_ACTION_TOOLS=1")


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _to_payload(value: Any) -> Any:
    if value is None:
        return None
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _dedupe_contexts(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in sorted(contexts, key=lambda row: float(row.get("score", 0.0)), reverse=True):
        uri = str(item.get("uri", ""))
        if uri in seen:
            continue
        seen.add(uri)
        deduped.append(item)
    return deduped


def _policies(value: Any) -> list[ActionPolicy] | None:
    if not value:
        return None
    if not isinstance(value, list):
        raise ValueError("policies must be an array")
    policies = []
    for item in value:
        if isinstance(item, ActionPolicy):
            policies.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError("policies entries must be objects")
        policies.append(ActionPolicy(**item))
    return policies


def _prediction_request(payload: dict[str, Any]) -> PredictionRequest:
    try:
        return PredictionRequest(**payload)
    except TypeError as exc:
        raise ValueError("request payload does not match PredictionRequest schema") from exc
