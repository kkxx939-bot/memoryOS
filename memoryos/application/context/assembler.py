"""Application facade over the single Unified Context retrieval chain."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from memoryos.application.context.orchestrator import UnifiedRetrievalOrchestrator
from memoryos.application.context.query_planner import (
    QueryPlanner,
    TrustedRetrievalScope,
    retrieval_options_from_legacy,
)
from memoryos.application.context.reranking import Reranker
from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.retrieval.limits import MAX_RETRIEVAL_LIMIT, MAX_TOKEN_BUDGET, bounded_int
from memoryos.contextdb.retrieval.query_plan import RetrievalOptions
from memoryos.contextdb.retrieval.token_budget import HeuristicTokenCounter, TokenCounter


class ContextAssembler:
    """Preserve the old Python surface while delegating all online work once."""

    def __init__(
        self,
        context_db: ContextDB,
        *,
        reranker: Reranker | None = None,
        token_counter: TokenCounter | None = None,
        hybrid_search: HybridSearch | None = None,
    ) -> None:
        self.context_db = context_db
        self.reranker = reranker
        # Retained as a public compatibility attribute. Token selection itself
        # is owned exclusively by the Unified ContextPacker.
        self.token_counter = token_counter or HeuristicTokenCounter()
        self.hybrid_search = hybrid_search
        self.query_planner = QueryPlanner()
        self.unified_retrieval = UnifiedRetrievalOrchestrator(
            context_db,
            vector_store=getattr(hybrid_search, "vector_store", None),
            embedding_provider=getattr(hybrid_search, "embedding_provider", None),
            reranker=reranker,
            projection_store=getattr(context_db, "projection_store", None),
        )

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        context_type: object | None = None,
        limit: int = 10,
        connect_filters: dict[str, Any] | None = None,
        search_scope: str | None = None,
        retrieval_views: list[str] | None = None,
        project_id: str = "",
        adapter_id: str = "",
        tenant_id: str = "default",
        applicability_scope_keys: Sequence[str] | None = None,
        memory_states: Sequence[str] | None = None,
        memory_types: Sequence[str] | None = None,
        claim_uris: Sequence[str] | None = None,
        slot_uris: Sequence[str] | None = None,
        query_intent: str | None = None,
        options: RetrievalOptions | None = None,
    ) -> list[dict[str, Any]]:
        requested_limit = bounded_int(
            limit,
            default=10,
            minimum=0,
            maximum=MAX_RETRIEVAL_LIMIT,
            label="limit",
        )
        parsed_request_type = self._context_type(context_type)
        if requested_limit == 0 or (
            user_id is None and parsed_request_type not in {None, ContextType.RESOURCE, ContextType.SKILL}
        ):
            return []
        plan = self._unified_plan(
            query,
            options=options,
            user_id=user_id,
            context_types=(() if parsed_request_type is None else (parsed_request_type,)),
            limit=requested_limit,
            token_budget=max(512, requested_limit * 256),
            connect_filters=connect_filters,
            search_scope=search_scope,
            retrieval_views=retrieval_views,
            project_id=project_id,
            adapter_id=adapter_id,
            tenant_id=tenant_id,
            applicability_scope_keys=applicability_scope_keys,
            memory_states=memory_states,
            memory_types=memory_types,
            claim_uris=claim_uris,
            slot_uris=slot_uris,
            query_intent=query_intent,
        )
        result = self.unified_retrieval.execute(plan)
        rows = self._filter_connect(result.search_payload(), connect_filters)
        return rows[:requested_limit]

    def assemble(
        self,
        query: str,
        *,
        user_id: str | None = None,
        token_budget: int = 2000,
        context_types: Sequence[object] | None = None,
        limit: int = 20,
        connect_metadata: dict[str, Any] | None = None,
        connect_filters: dict[str, Any] | None = None,
        search_scope: str | None = None,
        retrieval_views: list[str] | None = None,
        project_id: str = "",
        adapter_id: str = "",
        tenant_id: str = "default",
        applicability_scope_keys: Sequence[str] | None = None,
        memory_states: Sequence[str] | None = None,
        memory_types: Sequence[str] | None = None,
        claim_uris: Sequence[str] | None = None,
        slot_uris: Sequence[str] | None = None,
        query_intent: str | None = None,
        options: RetrievalOptions | None = None,
    ) -> dict[str, Any]:
        limit = bounded_int(
            limit,
            default=20,
            minimum=0,
            maximum=MAX_RETRIEVAL_LIMIT,
            label="limit",
        )
        token_budget = bounded_int(
            token_budget,
            default=2000,
            minimum=0,
            maximum=MAX_TOKEN_BUDGET,
            label="token_budget",
        )
        if limit == 0 or token_budget == 0:
            return {
                "query": query,
                "token_budget": token_budget,
                "contexts": [],
                "packed_context": "",
                "source_uris": [],
                "dropped_contexts": [],
                "load_plan": [],
                "metrics": {},
                "degraded_modes": [],
                "connect_metadata": dict(connect_metadata or {}),
            }
        plan = self._unified_plan(
            query,
            options=options,
            user_id=user_id,
            context_types=context_types or (),
            limit=limit,
            token_budget=token_budget,
            connect_filters=connect_filters,
            search_scope=search_scope,
            retrieval_views=retrieval_views,
            project_id=project_id,
            adapter_id=adapter_id,
            tenant_id=tenant_id,
            applicability_scope_keys=applicability_scope_keys,
            memory_states=memory_states,
            memory_types=memory_types,
            claim_uris=claim_uris,
            slot_uris=slot_uris,
            query_intent=query_intent,
        )
        result = self.unified_retrieval.execute(plan)
        contexts = self._filter_connect(result.search_payload(), connect_filters)
        source_uris = [str(item.get("source_uri") or item.get("uri") or "") for item in contexts]
        packed_context = "\n\n".join(str(item.get("content") or item.get("text") or "") for item in contexts)
        return {
            "query": query,
            "token_budget": token_budget,
            "contexts": contexts,
            "packed_context": packed_context,
            "source_uris": source_uris,
            "dropped_contexts": [dict(item) for item in result.dropped_contexts],
            "load_plan": [dict(item) for item in result.load_plan],
            "metrics": result.metrics.to_dict(),
            "degraded_modes": list(result.degraded_modes),
            "query_plan": result.plan.to_dict(),
            "reranker_fallback": result.reranker_fallback,
            "connect_metadata": dict(connect_metadata or {}),
        }

    def _unified_plan(
        self,
        query: str,
        *,
        options: RetrievalOptions | None,
        user_id: str | None,
        context_types: Sequence[object],
        limit: int,
        token_budget: int,
        connect_filters: dict[str, Any] | None,
        search_scope: str | None,
        retrieval_views: Sequence[str] | None,
        project_id: str,
        adapter_id: str,
        tenant_id: str,
        applicability_scope_keys: Sequence[str] | None,
        memory_states: Sequence[str] | None,
        memory_types: Sequence[str] | None,
        claim_uris: Sequence[str] | None,
        slot_uris: Sequence[str] | None,
        query_intent: str | None,
    ):
        parsed_types = tuple(self._context_type(item) for item in context_types)
        normalized_types = tuple(item for item in parsed_types if item is not None)
        bounded_compat_backend = not hasattr(self.context_db, "index_store")
        if user_id is None and not bounded_compat_backend:
            public_types = (ContextType.RESOURCE, ContextType.SKILL)
            normalized_types = (
                tuple(item for item in normalized_types if item in public_types) if normalized_types else public_types
            )
        intent = query_intent or ("OPTIONS" if search_scope == "candidates" else "CURRENT")
        metadata_filters: dict[str, Any] = {}
        if connect_filters:
            metadata_filters["connect_filters"] = dict(connect_filters)
        if user_id is None and not bounded_compat_backend:
            metadata_filters["principal_absent"] = True
        if options is None:
            selected = retrieval_options_from_legacy(
                {
                    "user_id": user_id,
                    "context_types": normalized_types,
                    "tenant_id": tenant_id,
                    "project_id": project_id or None,
                    "adapter_id": adapter_id or None,
                    "search_scope": search_scope,
                    "retrieval_views": retrieval_views,
                    "claim_uris": claim_uris,
                    "slot_uris": slot_uris,
                    "memory_states": memory_states,
                    "memory_types": memory_types,
                    "applicability_scope_keys": applicability_scope_keys,
                    "query_intent": intent,
                    "candidate_limit": min(1000, max(50, limit * 5)),
                    "limit": limit,
                    "token_budget": token_budget,
                    "metadata_filters": metadata_filters,
                }
            )
        else:
            merged_metadata = {**dict(options.metadata_filters), **metadata_filters}
            selected = replace(
                options,
                context_types=options.context_types or normalized_types,
                final_limit=min(options.final_limit, limit),
                token_budget=min(options.token_budget, token_budget),
                metadata_filters=merged_metadata,
            )
        trusted = TrustedRetrievalScope(
            tenant_id=tenant_id,
            owner_user_id=user_id,
            workspace_ids=((project_id,) if project_id else None),
            adapter_id=(adapter_id or None),
        )
        return self.query_planner.build(query, options=selected, trusted_scope=trusted)

    @staticmethod
    def _context_type(context_type: object | None) -> ContextType | None:
        if context_type is None:
            return None
        if isinstance(context_type, ContextType):
            return context_type
        return ContextType(str(context_type))

    @staticmethod
    def _filter_connect(
        items: list[dict[str, Any]],
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        allowed = {"connect_type", "adapter_id", "run_mode", "world_domain", "source_kind"}
        simple_filters = {
            key: value
            for key, value in dict(filters or {}).items()
            if key in allowed and value is not None and value != ""
        }
        if not simple_filters:
            return items
        matched = []
        for item in items:
            connect = dict(item.get("metadata", {}).get("connect", {}) or {})
            views = [str(view) for view in item.get("metadata", {}).get("retrieval_views", []) or []]
            effective_filters = dict(simple_filters)
            if any(view.startswith(("project:", "user:")) for view in views):
                effective_filters.pop("adapter_id", None)
            if all(connect.get(key) == value for key, value in effective_filters.items()):
                matched.append(item)
        return matched
