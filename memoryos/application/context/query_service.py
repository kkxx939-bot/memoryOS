"""Context query, assembly, trace, and exact-read orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memoryos.application.context.query_planner import QueryPlanner, retrieval_options_from_legacy
from memoryos.application.context.query_support import (
    _coerce_retrieval_options as parse_retrieval_options,
)
from memoryos.application.context.query_support import (
    _compatible_scalar,
    _merge_public_retrieval_options,
    _record_unified_recall,
    _requested_workspace,
    _scope_keys,
    _trace_root,
    _trusted_retrieval_scope,
)
from memoryos.application.context.retrieval_service import RetrievalService
from memoryos.application.service import ApplicationService
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.retrieval.query_plan import RetrievalOptions
from memoryos.memory.documents.path_policy import MemoryDocumentPathPolicy
from memoryos.security.trusted_context import READ_CONTEXT, TrustedRequestContext


class ContextQueryService(ApplicationService):
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
        document_ids: list[str] | None = None,
        document_kinds: list[str] | None = None,
        query_intent: str | None = None,
        caller: TrustedRequestContext | None = None,
    ) -> list[dict[str, Any]]:
        """按用户、工作区、状态和查询意图检索上下文。"""

        structured_options = parse_retrieval_options(options)
        tenant_id = self._effective_tenant(
            caller,
            _compatible_scalar(tenant_id, structured_options.tenant_id if structured_options else None, "tenant_id"),
        )
        self._require_ready()
        if caller is not None:
            caller.require(READ_CONTEXT)
            caller.assert_identity(
                user_id=_compatible_scalar(
                    user_id,
                    structured_options.owner_user_id if structured_options else None,
                    "owner_user_id",
                ),
                tenant_id=tenant_id,
            )
            user_id = caller.user_id
            project_id = caller.bind_read_workspace(
                _requested_workspace(project_id, structured_options.workspace_ids if structured_options else ())
            )
            caller.assert_applicability_scopes(
                applicability_scopes,
                workspace_id=project_id,
            )

        connect_filters = self._connect_filters_from_metadata(connect_metadata)
        resolved_scope_keys = _scope_keys(applicability_scopes)
        legacy_options = retrieval_options_from_legacy(
            {
                "user_id": user_id,
                "context_type": context_type,
                "limit": limit,
                "candidate_limit": min(1000, max(50, limit * 5)),
                "metadata_filters": {"connect_filters": dict(connect_filters)},
                "search_scope": search_scope,
                "retrieval_views": retrieval_views,
                "project_id": project_id or self._project_id_from_metadata(connect_metadata),
                "adapter_id": connect_filters.get("adapter_id"),
                "tenant_id": tenant_id,
                "applicability_scope_keys": resolved_scope_keys or None,
                "record_kinds": record_kinds,
                "document_ids": document_ids,
                "document_kinds": document_kinds,
                "query_intent": query_intent,
            }
        )
        effective_options = _merge_public_retrieval_options(
            structured_options,
            legacy_options,
            legacy_limit=limit,
            legacy_limit_default=10,
            legacy_query_intent=query_intent,
        )
        trusted_scope = _trusted_retrieval_scope(
            caller=caller,
            tenant_id=tenant_id,
            project_id=project_id,
            derived_scope_keys=resolved_scope_keys,
        )
        plan = QueryPlanner().build(query, options=effective_options, trusted_scope=trusted_scope)
        try:
            unified = self._retrieval_orchestrator().execute(plan)
        except Exception:
            self._require_ready()
            raise
        self._require_ready()
        trace_id = _record_unified_recall(self, unified)
        self._runtime.last_recall_trace_id = trace_id
        return unified.search_payload()

    def assemble_context(
        self,
        query: str,
        *,
        options: RetrievalOptions | Mapping[str, Any] | None = None,
        user_id: str | None = None,
        token_budget: int = 2000,
        context_types: list[object] | None = None,
        limit: int = 20,
        connect_metadata: dict[str, Any] | None = None,
        search_scope: str | None = None,
        retrieval_views: list[str] | None = None,
        project_id: str = "",
        tenant_id: str | None = None,
        applicability_scopes: list[dict[str, Any]] | None = None,
        record_kinds: list[str] | None = None,
        document_ids: list[str] | None = None,
        document_kinds: list[str] | None = None,
        query_intent: str | None = None,
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        """检索并打包本次请求能看到的上下文。"""

        structured_options = parse_retrieval_options(options)
        tenant_id = self._effective_tenant(
            caller,
            _compatible_scalar(tenant_id, structured_options.tenant_id if structured_options else None, "tenant_id"),
        )
        self._require_ready()
        if caller is not None:
            caller.require(READ_CONTEXT)
            caller.assert_identity(
                user_id=_compatible_scalar(
                    user_id,
                    structured_options.owner_user_id if structured_options else None,
                    "owner_user_id",
                ),
                tenant_id=tenant_id,
            )
            user_id = caller.user_id
            project_id = caller.bind_read_workspace(
                _requested_workspace(project_id, structured_options.workspace_ids if structured_options else ())
            )
            caller.assert_applicability_scopes(
                applicability_scopes,
                workspace_id=project_id,
            )

        metadata = self._parse_connect_metadata(connect_metadata)
        connect_filters = self._connect_filters_from_metadata(connect_metadata)
        resolved_scope_keys = _scope_keys(applicability_scopes)
        legacy_options = retrieval_options_from_legacy(
            {
                "user_id": user_id,
                "token_budget": token_budget,
                "context_types": context_types,
                "limit": limit,
                "candidate_limit": min(1000, max(50, limit * 5)),
                "metadata_filters": {"connect_filters": dict(connect_filters)},
                "search_scope": search_scope,
                "retrieval_views": retrieval_views,
                "project_id": project_id or self._project_id_from_metadata(connect_metadata),
                "adapter_id": connect_filters.get("adapter_id"),
                "tenant_id": tenant_id,
                "applicability_scope_keys": resolved_scope_keys or None,
                "record_kinds": record_kinds,
                "document_ids": document_ids,
                "document_kinds": document_kinds,
                "query_intent": query_intent,
            }
        )
        effective_options = _merge_public_retrieval_options(
            structured_options,
            legacy_options,
            legacy_limit=limit,
            legacy_limit_default=20,
            legacy_token_budget=token_budget,
            legacy_token_budget_default=2000,
            legacy_query_intent=query_intent,
        )
        trusted_scope = _trusted_retrieval_scope(
            caller=caller,
            tenant_id=tenant_id,
            project_id=project_id,
            derived_scope_keys=resolved_scope_keys,
        )
        plan = QueryPlanner().build(query, options=effective_options, trusted_scope=trusted_scope)
        try:
            unified = self._retrieval_orchestrator().execute(plan)
        except Exception:
            self._require_ready()
            raise
        self._require_ready()
        trace_id = _record_unified_recall(self, unified)
        self._runtime.last_recall_trace_id = trace_id
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

    def recall_trace(
        self,
        trace_id: str,
        *,
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        self._require_ready()
        if caller is not None:
            caller.require(READ_CONTEXT)
        trace = RetrievalService(self._context_assembler(), _trace_root(self)).read_trace(trace_id)
        if caller is not None:
            scope = dict(trace.get("scope", {}) or {})
            if scope.get("user_id") != caller.user_id or scope.get("tenant_id") != caller.tenant_id:
                raise FileNotFoundError(trace_id)
            self._require_exact_workspace({"project_id": scope.get("project_id")}, caller, trace_id)
        return trace

    def read(
        self,
        uri: str,
        *,
        layer: str = "L2",
        tenant_id: str | None = None,
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        tenant_id = self._effective_tenant(caller, tenant_id)
        self._require_ready()
        parsed = ContextURI.parse(uri)
        if caller is not None:
            caller.require(READ_CONTEXT)
        if len(parsed.segments) == 4 and parsed.segments[1:3] == ("memory", "documents"):
            owner_user_id, document_id = MemoryDocumentPathPolicy.parse_document_uri(uri)
            if caller is not None and owner_user_id != caller.user_id:
                raise FileNotFoundError(uri)
            records = self.index_store.get_catalog_by_uri(
                tenant_id=tenant_id,
                uri=uri,
                limit=2,
            )
            document_records = [
                record
                for record in records
                if record.record_kind == "memory_document" and record.document_id == document_id
            ]
            if len(document_records) != 1:
                raise FileNotFoundError(uri)
            record = document_records[0]
            if record.owner_user_id != owner_user_id:
                raise FileNotFoundError(uri)
            requested_layer = layer.upper()
            if requested_layer == "L0":
                content = record.l0_text
            elif requested_layer == "L1":
                content = record.l1_text
            elif requested_layer == "L2":
                overlay = getattr(self.context_db, "memory_document_overlay", None)
                if overlay is None:
                    raise FileNotFoundError(uri)
                view = overlay.read(
                    tenant_id=tenant_id,
                    owner_user_id=owner_user_id,
                    document_uri=uri,
                    relative_path=str(record.metadata.get("relative_path") or ""),
                    expected_source_digest=record.source_digest,
                )
                content = view.markdown
            else:
                raise FileNotFoundError(f"layer unavailable: {layer}")
            return {"object": record.to_dict(), "layer": requested_layer, "content": content}
        obj = self.context_db.read_object(uri)
        if caller is not None:
            self._require_exact_read_visibility(uri, obj, caller)
        requested_layer = layer.upper()
        layer_uri = {
            "L0": obj.layers.l0_uri,
            "L1": obj.layers.l1_uri,
            "L2": obj.layers.l2_uri or obj.uri,
        }.get(requested_layer)
        if not layer_uri:
            raise FileNotFoundError(f"layer unavailable: {layer}")
        if caller is not None:
            layer_parsed = ContextURI.parse(layer_uri)
            if layer_parsed.authority != parsed.authority or layer_parsed.user_id != parsed.user_id:
                raise FileNotFoundError(uri)
        content = self.source_store.read_content(layer_uri)
        return {"object": obj.to_dict(), "layer": requested_layer, "content": content}




__all__ = ["ContextQueryService"]
