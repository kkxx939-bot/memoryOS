"""上下文查询、组装、召回轨迹和精确读取编排。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from foundation.identity import LocalUserContext
from foundation.readiness import RuntimeReadiness
from infrastructure.context.contracts import ContextObjectReader
from infrastructure.context.exact_reader import ContextExactReader
from infrastructure.context.orchestrator import UnifiedRetrievalOrchestrator, UnifiedRetrievalResult
from infrastructure.context.query_planner import QueryPlanner, retrieval_options_from_legacy
from infrastructure.context.query_support import (
    _coerce_retrieval_options as parse_retrieval_options,
)
from infrastructure.context.query_support import (
    _compatible_scalar,
    _merge_public_retrieval_options,
    _prepare_retrieval_options,
    _requested_workspace,
    _scope_keys,
)
from infrastructure.context.retrieval.query_plan import RetrievalOptions
from infrastructure.context.trace import RecallTraceService
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.trace import RecallTraceRepository, recall_trace_root
from pre.connect import ConnectMetadata


class ContextQueryService:
    """查询用例只持有检索、读取和访问控制所需的依赖。"""

    def __init__(
        self,
        *,
        root: str,
        tenant_id: str,
        source_store: SourceStore | None,
        context_reader: ContextObjectReader,
        readiness: RuntimeReadiness | None,
        effective_tenant: Callable[[LocalUserContext | None, str | None], str],
        connect_filters_from_metadata: Callable[[dict[str, Any] | None], dict[str, str]],
        project_id_from_metadata: Callable[[dict[str, Any] | None], str],
        parse_connect_metadata: Callable[[dict[str, Any] | None], ConnectMetadata],
        retrieval_orchestrator: Callable[[], UnifiedRetrievalOrchestrator],
        require_exact_workspace: Callable[[dict[str, Any], LocalUserContext, str], None],
        require_exact_read_scope: Callable[[str, Any, LocalUserContext], None],
        set_last_recall_trace_id: Callable[[str], None],
    ) -> None:
        self.root = root
        self.tenant_id = tenant_id
        self._readiness = readiness
        self._effective_tenant = effective_tenant
        self._connect_filters_from_metadata = connect_filters_from_metadata
        self._project_id_from_metadata = project_id_from_metadata
        self._parse_connect_metadata = parse_connect_metadata
        self._retrieval_orchestrator = retrieval_orchestrator
        self._require_exact_workspace = require_exact_workspace
        self._set_last_recall_trace_id = set_last_recall_trace_id
        self._exact_reader = ContextExactReader(
            source_store=source_store,
            context_reader=context_reader,
            require_exact_read_scope=require_exact_read_scope,
        )

    def _require_ready(self) -> None:
        if self._readiness is not None:
            self._readiness.require_ready()

    def search_context(
        self,
        query: str,
        *,
        options: RetrievalOptions | Mapping[str, Any] | None = None,
        user_id: str | None = None,
        context_type: object | None = None,
        limit: int = 10,
        connect_metadata: dict[str, Any] | None = None,
        search_scope: str | None = None,
        retrieval_views: list[str] | None = None,
        project_id: str = "",
        tenant_id: str | None = None,
        applicability_scopes: list[dict[str, Any]] | None = None,
        record_kinds: list[str] | None = None,
        query_intent: str | None = None,
        caller: LocalUserContext | None = None,
    ) -> list[dict[str, Any]]:
        """按用户、工作区、状态和查询意图检索上下文。"""

        unified, _trace_id, _metadata = self._execute_query(
            query,
            options=options,
            user_id=user_id,
            context_filters={"context_type": context_type},
            limit=limit,
            default_limit=10,
            connect_metadata=connect_metadata,
            search_scope=search_scope,
            retrieval_views=retrieval_views,
            project_id=project_id,
            tenant_id=tenant_id,
            applicability_scopes=applicability_scopes,
            record_kinds=record_kinds,
            query_intent=query_intent,
            caller=caller,
        )
        return unified.search_payload()

    def assemble_context(
        self,
        query: str,
        *,
        options: RetrievalOptions | Mapping[str, Any] | None = None,
        user_id: str | None = None,
        context_types: list[object] | None = None,
        limit: int = 20,
        connect_metadata: dict[str, Any] | None = None,
        search_scope: str | None = None,
        retrieval_views: list[str] | None = None,
        project_id: str = "",
        tenant_id: str | None = None,
        applicability_scopes: list[dict[str, Any]] | None = None,
        record_kinds: list[str] | None = None,
        query_intent: str | None = None,
        caller: LocalUserContext | None = None,
    ) -> dict[str, Any]:
        """按数量上限检索并组装本次请求能看到的上下文。"""

        unified, trace_id, metadata = self._execute_query(
            query,
            options=options,
            user_id=user_id,
            context_filters={"context_types": context_types},
            limit=limit,
            default_limit=20,
            connect_metadata=connect_metadata,
            search_scope=search_scope,
            retrieval_views=retrieval_views,
            project_id=project_id,
            tenant_id=tenant_id,
            applicability_scopes=applicability_scopes,
            record_kinds=record_kinds,
            query_intent=query_intent,
            caller=caller,
        )
        result = unified.assemble_payload()
        contexts = list(result.get("contexts", []))
        return {
            **result,
            "trace_id": trace_id,
            "packed_context": "\n\n".join(str(item.get("content") or item.get("text") or "") for item in contexts),
            "source_uris": list(
                dict.fromkeys(
                    source_uri
                    for item in contexts
                    if (source_uri := str(item.get("source_uri") or item.get("uri") or ""))
                )
            ),
            "connect_metadata": metadata.to_dict(),
        }

    def _execute_query(
        self,
        query: str,
        *,
        options: RetrievalOptions | Mapping[str, Any] | None,
        user_id: str | None,
        context_filters: Mapping[str, Any],
        limit: int,
        default_limit: int,
        connect_metadata: dict[str, Any] | None,
        search_scope: str | None,
        retrieval_views: list[str] | None,
        project_id: str,
        tenant_id: str | None,
        applicability_scopes: list[dict[str, Any]] | None,
        record_kinds: list[str] | None,
        query_intent: str | None,
        caller: LocalUserContext | None,
    ) -> tuple[UnifiedRetrievalResult, str, ConnectMetadata]:
        """统一完成身份绑定、查询规划、检索执行和轨迹落盘。"""

        structured_options = parse_retrieval_options(options)
        effective_tenant = self._effective_tenant(
            caller,
            _compatible_scalar(
                tenant_id,
                structured_options.tenant_id if structured_options else None,
                "tenant_id",
            ),
        )
        self._require_ready()
        if caller is not None:
            caller.assert_identity(
                user_id=_compatible_scalar(
                    user_id,
                    structured_options.owner_user_id if structured_options else None,
                    "owner_user_id",
                ),
                tenant_id=effective_tenant,
            )
            user_id = caller.user_id
            project_id = caller.bind_read_workspace(
                _requested_workspace(project_id, structured_options.workspace_ids if structured_options else ())
            )

        metadata = self._parse_connect_metadata(connect_metadata)
        connect_filters = self._connect_filters_from_metadata(connect_metadata)
        legacy_options = retrieval_options_from_legacy(
            {
                "user_id": user_id,
                **dict(context_filters),
                "limit": limit,
                "candidate_limit": min(1000, max(50, limit * 5)),
                "metadata_filters": {"connect_filters": dict(connect_filters)},
                "search_scope": search_scope,
                "retrieval_views": retrieval_views,
                "project_id": project_id or self._project_id_from_metadata(connect_metadata),
                "adapter_id": connect_filters.get("adapter_id"),
                "tenant_id": effective_tenant,
                "applicability_scope_keys": _scope_keys(applicability_scopes) or None,
                "record_kinds": record_kinds,
                "query_intent": query_intent,
            }
        )
        effective_options = _merge_public_retrieval_options(
            structured_options,
            legacy_options,
            legacy_limit=limit,
            legacy_limit_default=default_limit,
            legacy_query_intent=query_intent,
        )
        effective_options = _prepare_retrieval_options(
            effective_options,
            caller=caller,
            project_id=project_id,
        )
        plan = QueryPlanner().build(query, options=effective_options)
        if caller is not None and caller.actor_kind == "service":
            plan = replace(plan, service_id=caller.actor_id)
        try:
            unified = self._retrieval_orchestrator().execute(plan)
        except Exception:
            self._require_ready()
            raise
        self._require_ready()
        trace_id = self._trace_service().record_unified(unified)
        self._set_last_recall_trace_id(trace_id)
        return unified, trace_id, metadata

    def recall_trace(
        self,
        trace_id: str,
        *,
        caller: LocalUserContext | None = None,
    ) -> dict[str, Any]:
        self._require_ready()
        trace = self._trace_service().read(trace_id)
        if caller is not None:
            scope = dict(trace.get("scope", {}) or {})
            if scope.get("user_id") != caller.user_id or scope.get("tenant_id") != caller.tenant_id:
                raise FileNotFoundError(trace_id)
            self._require_exact_workspace({"project_id": scope.get("project_id")}, caller, trace_id)
        return trace

    def _trace_service(self) -> RecallTraceService:
        """为当前租户组合轨迹语义服务和持久化仓库。"""

        repository = RecallTraceRepository(recall_trace_root(self.root, self.tenant_id))
        return RecallTraceService(repository)

    def read(
        self,
        uri: str,
        *,
        layer: str = "L2",
        tenant_id: str | None = None,
        caller: LocalUserContext | None = None,
    ) -> dict[str, Any]:
        tenant_id = self._effective_tenant(caller, tenant_id)
        self._require_ready()
        return self._exact_reader.read(
            uri,
            layer=layer,
            tenant_id=tenant_id,
            caller=caller,
        )


__all__ = ["ContextQueryService"]
