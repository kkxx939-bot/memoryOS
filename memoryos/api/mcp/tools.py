"""MCP 工具定义。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import memoryos
from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.api.limits import MAX_RETRIEVAL_LIMIT, MAX_TOKEN_BUDGET
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
from memoryos.api.retrieval_contract import parse_retrieval_options
from memoryos.api.trusted_context import (
    AUTHORITATIVE_FORGET,
    AUTHORITATIVE_REMEMBER,
    COMMIT_SESSION,
    READ_CONTEXT,
    TrustedRequestContext,
)
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.retrieval.query_plan import (
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_TOKEN_BUDGET,
    RetrievalOptions,
    RetrievalQueryIntent,
)
from memoryos.contextdb.retrieval.query_planner import merge_retrieval_options, retrieval_options_from_legacy
from memoryos.prediction.model.prediction_request import PredictionRequest


class MCPToolRouter:
    def __init__(self, client: Any, config: MCPServerConfig | None = None) -> None:
        self.client = client
        self.config = config or MCPServerConfig.from_env()
        self.caller = self.config.trusted_context()
        client_tenant = getattr(client, "tenant_id", self.caller.tenant_id)
        if str(client_tenant) != self.caller.tenant_id:
            raise ValueError("MCP client tenant does not match trusted caller tenant")

    def call(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        args = dict(arguments or {})
        try:
            if name in {"memoryos_search", "memoryos_search_context"}:
                return self.search_context(args)
            if name in {"memoryos_assemble", "memoryos_assemble_context"}:
                return self.assemble_context(args)
            if name == "memoryos_commit_session":
                return self.commit_session(args)
            if name == "memoryos_health":
                return self.health()
            if name == "memoryos_read":
                self.caller.require(READ_CONTEXT)
                return _client_payload(
                    self.client.read(
                        required_str(args, "uri"),
                        layer=str(args.get("layer") or "L2"),
                        **self._local_caller_kwargs(),
                    )
                )
            if name == "memoryos_remember":
                self.caller.require(AUTHORITATIVE_REMEMBER)
                if self.caller.actor_kind != "user":
                    raise PermissionError("authoritative remember requires a trusted user actor")
                self._assert_identity(args)
                if "identity_fields" in args and not isinstance(args["identity_fields"], dict):
                    raise ValueError("identity_fields must be an object")
                metadata = normalize_agent_metadata(args.get("connect_metadata"), self.config)
                return _client_payload(
                    self.client.remember(
                        user_id=self.caller.user_id,
                        content=required_str(args, "content"),
                        title=str(args.get("title") or ""),
                        memory_type=str(args.get("memory_type") or "project_decision"),
                        project_id=str(args.get("project_id") or ""),
                        constraint_polarity=str(args.get("constraint_polarity") or ""),
                        condition=str(args.get("condition") or ""),
                        exception=str(args.get("exception") or ""),
                        identity_fields=(dict(args["identity_fields"]) if "identity_fields" in args else None),
                        connect_metadata=metadata,
                        tenant_id=self.caller.tenant_id,
                        **self._local_caller_kwargs(),
                    )
                )
            if name == "memoryos_list_pending":
                self.caller.require(READ_CONTEXT)
                self._assert_identity(args)
                return ok_payload(
                    {
                        "results": self.client.list_pending(
                            user_id=self.caller.user_id,
                            tenant_id=self.caller.tenant_id,
                            lifecycle_states=([str(item) for item in optional_list(args, "lifecycle_states") or []]),
                            project_id=str(args.get("project_id") or ""),
                            **self._local_caller_kwargs(),
                        )
                    }
                )
            if name == "memoryos_review_pending":
                self.caller.require(AUTHORITATIVE_REMEMBER)
                if self.caller.actor_kind != "user":
                    raise PermissionError("pending review requires a trusted user actor")
                self._assert_identity(args)
                if "corrected_proposal" in args and not isinstance(args["corrected_proposal"], dict):
                    raise ValueError("corrected_proposal must be an object")
                return _client_payload(
                    self.client.review_pending(
                        user_id=self.caller.user_id,
                        pending_uri=required_str(args, "pending_uri"),
                        decision=required_str(args, "decision"),
                        expected_lifecycle_revision=optional_int(
                            args,
                            "expected_lifecycle_revision",
                            0,
                            minimum=1,
                        ),
                        expected_proposal_fingerprint=required_str(
                            args,
                            "expected_proposal_fingerprint",
                        ),
                        command_id=required_str(args, "command_id"),
                        tenant_id=self.caller.tenant_id,
                        reason=str(args.get("reason") or ""),
                        corrected_proposal=(dict(args["corrected_proposal"]) if "corrected_proposal" in args else None),
                        **self._local_caller_kwargs(),
                    )
                )
            if name == "memoryos_forget":
                self.caller.require(AUTHORITATIVE_FORGET)
                self._assert_identity(args)
                return _client_payload(
                    self.client.forget(
                        user_id=self.caller.user_id,
                        uri=required_str(args, "uri"),
                        tenant_id=self.caller.tenant_id,
                        **self._local_caller_kwargs(),
                    )
                )
            if name == "memoryos_archive_search":
                self.caller.require(READ_CONTEXT)
                return ok_payload(
                    {
                        "results": self.client.archive_search(
                            required_str(args, "query"),
                            user_id=self._bound_user(args),
                            limit=optional_int(args, "limit", 20, maximum=100),
                            tenant_id=self._bound_tenant(args),
                            project_id=str(args.get("project_id") or ""),
                            **self._local_caller_kwargs(),
                        )
                    }
                )
            if name == "memoryos_archive_read":
                self.caller.require(READ_CONTEXT)
                archive_uri = required_str(args, "archive_uri")
                if ContextURI.parse(archive_uri).user_id != self.caller.user_id:
                    raise FileNotFoundError(archive_uri)
                return _client_payload(self.client.archive_read(archive_uri, **self._local_caller_kwargs()))
            if name == "memoryos_recall_trace":
                self.caller.require(READ_CONTEXT)
                return _client_payload(
                    self.client.recall_trace(
                        required_str(args, "trace_id"),
                        **self._local_caller_kwargs(),
                    )
                )
            if name == "memoryos_connection_schema":
                return ok_payload(connection_schema(self.config))
            if name == "memoryos_predict":
                return self.predict(args)
            if name == "memoryos_process_observation":
                return self.process_observation(args)
            return {
                "error": {
                    "code": MCPErrorCode.VALIDATION_ERROR,
                    "message": f"Unknown tool: {name}",
                    "retryable": False,
                    "details": {},
                }
            }
        except Exception as exc:  # 工具层是外部 Agent 的最后一道安全边界。
            return exception_payload(exc)

    def search_context(self, args: dict[str, Any]) -> dict[str, Any]:
        self.caller.require(READ_CONTEXT)
        query = required_str(args, "query")
        limit = optional_int(
            args,
            "limit",
            10,
            minimum=1,
            maximum=MAX_RETRIEVAL_LIMIT,
        )
        metadata = normalize_agent_metadata(args.get("connect_metadata"), self.config)
        filter_metadata = agent_search_filter_metadata(args.get("connect_metadata"), self.config)
        context_type = args.get("context_type")
        context_types = optional_list(args, "context_types")
        search_scope = args.get("search_scope")
        project_id = str(args.get("project_id") or "")
        retrieval_views = optional_list(args, "retrieval_views")
        requested_types = [context_type] if context_type is not None else list(context_types or [])
        options = self._retrieval_options(
            args,
            context_types=tuple(str(item) for item in requested_types),
            limit=limit,
        )
        contexts = self.client.search_context(
            query,
            options=options,
            user_id=self._bound_user(args),
            context_type=None,
            limit=limit,
            connect_metadata=filter_metadata,
            search_scope=str(search_scope) if search_scope else None,
            retrieval_views=[str(item) for item in retrieval_views or []],
            project_id=project_id,
            tenant_id=self._bound_tenant(args),
            applicability_scopes=[dict(item) for item in optional_list(args, "applicability_scopes") or []],
            memory_states=[str(item) for item in optional_list(args, "memory_states") or []],
            memory_types=[str(item) for item in optional_list(args, "memory_types") or []],
            claim_uris=[str(item) for item in optional_list(args, "claim_uris") or []],
            slot_uris=[str(item) for item in optional_list(args, "slot_uris") or []],
            query_intent=str(args.get("query_intent")) if args.get("query_intent") else None,
            **self._local_caller_kwargs(),
        )
        source_uris = list(
            dict.fromkeys(
                source_uri for item in contexts if (source_uri := str(item.get("source_uri") or item.get("uri") or ""))
            )
        )
        payload = {
            "contexts": contexts,
            "results": contexts,
            "source_uris": source_uris,
            "metadata": {"connect": metadata},
        }
        trace_id = str(getattr(self.client, "last_recall_trace_id", "") or "")
        if trace_id:
            payload["trace_id"] = trace_id
        return ok_payload(payload)

    def assemble_context(self, args: dict[str, Any]) -> dict[str, Any]:
        self.caller.require(READ_CONTEXT)
        query = required_str(args, "query")
        token_budget = optional_int(
            args,
            "token_budget",
            self.config.token_budget,
            minimum=1,
            maximum=MAX_TOKEN_BUDGET,
        )
        limit = optional_int(
            args,
            "limit",
            20,
            minimum=1,
            maximum=MAX_RETRIEVAL_LIMIT,
        )
        context_types = optional_list(args, "context_types")
        search_scope = args.get("search_scope")
        project_id = str(args.get("project_id") or "")
        retrieval_views = optional_list(args, "retrieval_views")
        metadata = normalize_agent_metadata(args.get("connect_metadata"), self.config)
        filter_metadata = agent_search_filter_metadata(args.get("connect_metadata"), self.config)
        options = self._retrieval_options(
            args,
            context_types=tuple(str(item) for item in context_types or []),
            limit=limit,
            token_budget=token_budget,
        )
        assembled = self.client.assemble_context(
            query,
            options=options,
            user_id=self._bound_user(args),
            token_budget=(2000 if args.get("options") is not None and "token_budget" not in args else token_budget),
            context_types=context_types,
            limit=limit,
            connect_metadata=filter_metadata,
            search_scope=str(search_scope) if search_scope else None,
            retrieval_views=[str(item) for item in retrieval_views or []],
            project_id=project_id,
            tenant_id=self._bound_tenant(args),
            applicability_scopes=[dict(item) for item in optional_list(args, "applicability_scopes") or []],
            memory_states=[str(item) for item in optional_list(args, "memory_states") or []],
            memory_types=[str(item) for item in optional_list(args, "memory_types") or []],
            claim_uris=[str(item) for item in optional_list(args, "claim_uris") or []],
            slot_uris=[str(item) for item in optional_list(args, "slot_uris") or []],
            query_intent=str(args.get("query_intent")) if args.get("query_intent") else None,
            **self._local_caller_kwargs(),
        )
        effective_budget = int(assembled.get("total_budget") or token_budget)
        payload = {
            **assembled,
            "packed_context": assembled.get("packed_context", ""),
            "contexts": assembled.get("contexts", []),
            "source_uris": assembled.get("source_uris", []),
            "dropped_contexts": assembled.get("dropped_contexts", []),
            "token_budget": effective_budget,
            "estimated_tokens": _estimate_tokens(str(assembled.get("packed_context", ""))),
            "metadata": {"connect": metadata},
        }
        return ok_payload(payload)

    def commit_session(self, args: dict[str, Any]) -> dict[str, Any]:
        self.caller.require(COMMIT_SESSION)
        session_id = required_str(args, "session_id")
        user_id = self._bound_user(args)
        tenant_id = self._bound_tenant(args)
        metadata = normalize_agent_metadata(args.get("connect_metadata"), self.config)
        scope = dict(args.get("scope") or {})
        self.caller.assert_identity(user_id=scope.get("user_id"), tenant_id=scope.get("tenant_id"))
        scope.update({"user_id": user_id, "tenant_id": tenant_id})
        result = self.client.commit_agent_session(
            user_id=user_id,
            session_id=session_id,
            messages=list(args.get("messages") or []),
            used_contexts=list(args.get("used_contexts") or []),
            used_skills=list(args.get("used_skills") or []),
            tool_results=list(args.get("tool_results") or []),
            connect_metadata=metadata,
            async_commit=optional_bool(args, "async_commit", False),
            project_id=str(args.get("project_id") or ""),
            session_key=str(args.get("session_key") or ""),
            scope=scope,
            provenance=dict(args.get("provenance") or {}),
            **self._local_caller_kwargs(),
        )
        result_payload = _to_payload(result) or {"status": "accepted"}
        if isinstance(result_payload, dict) and isinstance(result_payload.get("error"), dict):
            return result_payload
        return ok_payload(
            {
                "status": result_payload.get("status", "accepted"),
                "result": result_payload,
                "metadata": {"connect": metadata},
            }
        )

    def _assert_identity(self, args: dict[str, Any]) -> None:
        self.caller.assert_identity(user_id=args.get("user_id"), tenant_id=args.get("tenant_id"))

    def _retrieval_options(
        self,
        args: dict[str, Any],
        *,
        context_types: tuple[str, ...],
        limit: int,
        token_budget: int | None = None,
    ) -> RetrievalOptions | None:
        structured = parse_retrieval_options(args.get("options"))
        requested_project = str(args.get("project_id") or "").strip()
        option_workspaces = structured.workspace_ids if structured is not None else ()
        if structured is not None:
            self.caller.assert_identity(
                user_id=structured.owner_user_id,
                tenant_id=structured.tenant_id,
            )
            if len(option_workspaces) > 1:
                raise PermissionError("caller must select one authorized workspace")
            for workspace_id in option_workspaces:
                self.caller.assert_workspace(workspace_id)
            if requested_project and option_workspaces and option_workspaces != (requested_project,):
                raise PermissionError("structured options conflict with project_id")
            if structured.adapter_id is not None and structured.adapter_id != self.caller.actor_id:
                raise PermissionError("caller adapter_id does not match trusted actor")
            if "limit" in args and structured.final_limit != limit:
                raise ValueError("structured options conflict with legacy limit")
            if token_budget is not None and "token_budget" in args and structured.token_budget != token_budget:
                raise ValueError("structured options conflict with legacy token_budget")
            if args.get("query_intent") is not None:
                requested_intent = RetrievalQueryIntent(str(args["query_intent"]).strip().upper())
                if structured.query_intent is not requested_intent:
                    raise ValueError("structured options conflict with legacy query_intent")
        workspace_id = self.caller.bind_read_workspace(
            requested_project or (option_workspaces[0] if option_workspaces else None)
        )
        self.caller.assert_applicability_scopes(
            args.get("applicability_scopes"),
            workspace_id=workspace_id,
        )
        if structured is not None:
            self.caller.assert_applicability_scope_keys(
                structured.metadata_filters.get("applicability_scope_keys"),
                workspace_id=workspace_id,
            )
        if not context_types:
            return structured
        type_options = retrieval_options_from_legacy(
            {
                "context_types": context_types,
                "candidate_limit": max(DEFAULT_CANDIDATE_LIMIT, limit),
                "limit": limit,
                "token_budget": token_budget if token_budget is not None else DEFAULT_TOKEN_BUDGET,
                "memory_states": args.get("memory_states"),
                "query_intent": args.get("query_intent"),
            }
        )
        return type_options if structured is None else merge_retrieval_options(structured, type_options)

    def _bound_user(self, args: dict[str, Any]) -> str:
        self._assert_identity(args)
        return self.caller.user_id

    def _bound_tenant(self, args: dict[str, Any]) -> str:
        self._assert_identity(args)
        return self.caller.tenant_id

    def _local_caller_kwargs(self) -> dict[str, TrustedRequestContext]:
        if getattr(self.client, "mode", None) in {"local", "server"}:
            return {"caller": self.caller}
        return {}

    def health(self) -> dict[str, Any]:
        metadata = {
            "root_configured": bool(self.config.root),
            "adapter_id": self.config.adapter_id,
        }
        health_fn = getattr(self.client, "health", None)
        raw_health = health_fn() if callable(health_fn) else {}
        health: dict[str, Any] = raw_health if isinstance(raw_health, dict) else {}
        runtime_payload = health.get("runtime")
        runtime = dict(runtime_payload) if isinstance(runtime_payload, dict) else {}
        runtime_state = str(runtime.get("state") or "NOT_READY")
        reported_status = str(health.get("status") or "not_ready").casefold()
        ready = runtime.get("ready") is True and runtime_state == "READY" and reported_status == "ready"
        status = (
            "ok"
            if ready
            else (reported_status if reported_status in {"degraded", "not_ready"} else runtime_state.casefold())
        )
        return ok_payload(
            {
                **health,
                "status": status,
                "storage_ready": ready and hasattr(self.client, "source_store"),
                "contextdb_ready": ready and hasattr(self.client, "context_db"),
                "client_ready": ready and self.client is not None,
                "version": memoryos.__version__,
                "metadata": metadata,
            }
        )

    def predict(self, args: dict[str, Any]) -> dict[str, Any]:
        self._ensure_action_tools_enabled()
        request_payload = args.get("request")
        if not isinstance(request_payload, dict):
            raise ValueError("requires object field: request")
        required_str(request_payload, "user_id")
        self.caller.assert_identity(user_id=request_payload.get("user_id"))
        request_payload = {**request_payload, "user_id": self.caller.user_id}
        metadata = normalize_action_metadata(request_payload.get("connect_metadata") or args.get("connect_metadata"))
        request_payload = {**request_payload, "connect_metadata": metadata.to_dict()}
        result = self.client.predict(_prediction_request(request_payload), _policies(args.get("policies")))
        return ok_payload({"prediction": _to_payload(result)})

    def process_observation(self, args: dict[str, Any]) -> dict[str, Any]:
        self._ensure_action_tools_enabled()
        request_payload = args.get("request")
        if not isinstance(request_payload, dict):
            raise ValueError("requires object field: request")
        required_str(request_payload, "user_id")
        self.caller.assert_identity(user_id=request_payload.get("user_id"))
        request_payload = {**request_payload, "user_id": self.caller.user_id}
        metadata = require_process_observation_metadata(
            request_payload.get("connect_metadata") or args.get("connect_metadata")
        )
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


def _client_payload(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("error"):
        return value
    return ok_payload(value)


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
