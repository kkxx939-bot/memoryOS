"""上下文数据库里的上下文组装。"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from typing import Any

from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.layers.context_packer import ContextPacker
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.retrieval.token_budget import HeuristicTokenCounter, TokenCounter
from memoryos.memory.canonical.retrieval import (
    CanonicalMemoryQuery,
    CanonicalMemoryRetriever,
    CanonicalQueryIntent,
)
from memoryos.memory.retrieval_plan import MemoryRetrievalPlanner
from memoryos.providers.rerank import Reranker

logger = logging.getLogger(__name__)


def _supported_search_kwargs(function: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


class ContextAssembler:
    """先筛出有权读取的上下文，再按预算打包。"""

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
        self.token_counter = token_counter or HeuristicTokenCounter()
        self.hybrid_search = hybrid_search
        self.retrieval_planner = MemoryRetrievalPlanner()
        source_store = getattr(context_db, "source_store", None)
        index_store = getattr(context_db, "index_store", None)
        relation_store = getattr(context_db, "relation_store", None)
        self.canonical_retriever = (
            CanonicalMemoryRetriever(source_store, index_store, relation_store, hybrid_search=self.hybrid_search)
            if source_store is not None and index_store is not None
            else None
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
    ) -> list[dict[str, Any]]:
        """按给定条件查找匹配结果。"""

        parsed_type = self._context_type(context_type)
        requested_limit = max(0, limit)
        if search_scope or retrieval_views:
            results = self._search_memory_views(
                query,
                user_id=user_id,
                context_type=parsed_type,
                limit=requested_limit,
                search_scope=search_scope,
                retrieval_views=retrieval_views,
                project_id=project_id,
                adapter_id=adapter_id,
                tenant_id=tenant_id,
            )
            recalled = self._rerank(query, self._filter_connect(results, connect_filters))[:requested_limit]
            return self._merge_canonical(
                query,
                recalled,
                user_id=user_id,
                parsed_type=parsed_type,
                requested_limit=requested_limit,
                tenant_id=tenant_id,
                project_id=project_id,
                applicability_scope_keys=applicability_scope_keys,
                memory_states=memory_states,
                memory_types=memory_types,
                claim_uris=claim_uris,
                slot_uris=slot_uris,
                query_intent=query_intent or ("OPTIONS" if search_scope == "candidates" else None),
                connect_filters=connect_filters,
            )
        search_limit = max(requested_limit * 5, 50) if connect_filters and requested_limit else requested_limit
        allowed_source_uris = self._allowed_source_uris(
            user_id=user_id,
            tenant_id=tenant_id,
            context_type=parsed_type,
            project_id=project_id,
            applicability_scope_keys=applicability_scope_keys,
        )
        hits: Sequence[Any]
        if self.hybrid_search is not None:
            filters: dict[str, Any] = {"tenant_id": tenant_id}
            if allowed_source_uris is not None:
                filters["allowed_uris"] = allowed_source_uris
            if user_id:
                filters["owner_user_id"] = user_id
            if project_id:
                filters["project_id"] = project_id
            if adapter_id and search_scope == "agent_private":
                filters["adapter_id"] = adapter_id
            hits = self.hybrid_search.search(query, filters=filters, context_type=parsed_type, limit=search_limit)
        else:
            search_kwargs: dict[str, Any] = {
                "owner_user_id": user_id,
                "tenant_id": tenant_id,
                "context_type": parsed_type,
                "limit": search_limit,
            }
            if allowed_source_uris is not None:
                search_kwargs["allowed_uris"] = allowed_source_uris
            if project_id:
                search_kwargs["project_id"] = project_id
            if adapter_id and search_scope == "agent_private":
                search_kwargs["adapter_id"] = adapter_id
            hits = self.context_db.search(query, **_supported_search_kwargs(self.context_db.search, search_kwargs))
        results = [self._hit_payload(hit) for hit in hits]
        scoped = self._filter_project(results, project_id)
        recalled = self._rerank(query, self._filter_connect(scoped, connect_filters))[:requested_limit]
        return self._merge_canonical(
            query,
            recalled,
            user_id=user_id,
            parsed_type=parsed_type,
            requested_limit=requested_limit,
            tenant_id=tenant_id,
            project_id=project_id,
            applicability_scope_keys=applicability_scope_keys,
            memory_states=memory_states,
            memory_types=memory_types,
            claim_uris=claim_uris,
            slot_uris=slot_uris,
            query_intent=query_intent or ("OPTIONS" if search_scope == "candidates" else None),
            connect_filters=connect_filters,
        )

    def _allowed_source_uris(
        self,
        *,
        user_id: str | None,
        tenant_id: str,
        context_type: ContextType | None,
        project_id: str,
        applicability_scope_keys: Sequence[str] | None,
        include_candidates: bool = False,
    ) -> tuple[str, ...] | None:
        source_store = getattr(self.context_db, "source_store", None)
        if source_store is None:
            if user_id is None and context_type not in {ContextType.RESOURCE, ContextType.SKILL}:
                return ()
            return None
        available_scopes = set(str(item) for item in applicability_scope_keys or [])
        if user_id:
            available_scopes.add(f"memoryos:principal:{user_id}")
        if project_id:
            available_scopes.add(f"memoryos:workspace:{project_id}")
        allowed = []
        for obj in source_store.list_objects():
            metadata = dict(obj.metadata or {})
            reviewable_pending = bool(
                include_candidates
                and metadata.get("canonical_kind") == "pending_proposal"
                and obj.lifecycle_state
                in {LifecycleState.PENDING, LifecycleState.RETRYABLE, LifecycleState.CONFIRMED}
            )
            if obj.lifecycle_state != LifecycleState.ACTIVE and not reviewable_pending:
                continue
            if str(obj.tenant_id or "default") != tenant_id:
                continue
            if user_id is None:
                if obj.context_type not in {ContextType.RESOURCE, ContextType.SKILL} or obj.owner_user_id:
                    continue
            elif obj.owner_user_id != user_id:
                continue
            if context_type is not None and obj.context_type != context_type:
                continue
            admission = dict(metadata.get("admission", {}) or {})
            if admission.get("decision") in {"restricted", "archive_only", "reject"}:
                continue
            if admission.get("decision") == "pending" and not include_candidates:
                continue
            if (
                metadata.get("canonical_kind") == "claim"
                and metadata.get("state") != "ACTIVE"
                and not include_candidates
            ):
                continue
            scope = dict(metadata.get("scope", {}) or {})
            visibility = dict(scope.get("visibility", {}) or {})
            if visibility:
                if str(visibility.get("tenant_id", "default")) != tenant_id:
                    continue
                principals = {str(item) for item in visibility.get("allowed_principal_ids", []) or []}
                if principals and (user_id is None or user_id not in principals):
                    continue
            applicability = dict(scope.get("applicability", {}) or {})
            required_scopes = {
                f"{item.get('namespace', 'memoryos')}:{item.get('kind')}:{item.get('id')}"
                for item in applicability.get("all_of", []) or []
                if isinstance(item, dict) and item.get("kind") and item.get("id")
            }
            if required_scopes and not required_scopes.issubset(available_scopes):
                continue
            fields = dict(metadata.get("fields", {}) or {})
            item_project = str(scope.get("project_id") or fields.get("project_id") or metadata.get("project_id") or "")
            if (
                project_id
                and str(metadata.get("memory_type", "")) in {"project_rule", "project_decision", "agent_experience"}
                and item_project
                and item_project != project_id
            ):
                continue
            allowed.append(obj.uri)
        return tuple(allowed)

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
    ) -> dict[str, Any]:
        """处理 assemble 这一步。"""

        contexts: list[dict[str, Any]] = []
        if context_types:
            per_type_limit = max(1, limit)
            for context_type in context_types:
                contexts.extend(
                    self.search(
                        query,
                        user_id=user_id,
                        context_type=context_type,
                        limit=per_type_limit,
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
                )
        else:
            contexts = self.search(
                query,
                user_id=user_id,
                limit=limit,
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

        contexts = self._dedupe(contexts)[: max(0, limit)]
        max_item_tokens = max(1, min(token_budget, token_budget // max(1, min(len(contexts), 4))))
        layered = [self._select_layer(item, query, max_item_tokens) for item in contexts]
        sections = {
            "retrieved_context": [
                {
                    "uri": item["uri"],
                    "content": selected["content"],
                    "metadata": item["metadata"],
                    "layer": selected["layer"],
                    "fallback_reason": selected["reason"],
                    "token_estimate": self._estimate_tokens(selected["content"]),
                }
                for item, selected in zip(contexts, layered, strict=False)
            ]
        }
        packed = ContextPacker(total_budget=token_budget, token_counter=self.token_counter).pack(sections)
        selected = packed["slices"].get("retrieved_context", {}).get("items", [])
        source_uris = [str(item.get("uri", "")) for item in selected if item.get("uri")]
        packed_context = "\n\n".join(str(item.get("content", "")) for item in selected if item.get("content"))
        selected_uris = set(source_uris)
        selected_by_uri = {str(item.get("uri", "")): item for item in selected}
        selected_contexts = [
            {
                **item,
                "selected_layer": selected_by_uri[item["uri"]].get("layer", item.get("layer", "search")),
                "fallback_reason": selected_by_uri[item["uri"]].get("fallback_reason", ""),
            }
            for item in contexts
            if item["uri"] in selected_uris
        ]
        return {
            "query": query,
            "token_budget": token_budget,
            "contexts": selected_contexts,
            "packed_context": packed_context,
            "source_uris": source_uris,
            "dropped_contexts": packed["dropped_contexts"],
            "connect_metadata": dict(connect_metadata or {}),
        }

    def _merge_canonical(
        self,
        query: str,
        recalled: list[dict[str, Any]],
        *,
        user_id: str | None,
        parsed_type: ContextType | None,
        requested_limit: int,
        tenant_id: str,
        project_id: str,
        applicability_scope_keys: Sequence[str] | None,
        memory_states: Sequence[str] | None,
        memory_types: Sequence[str] | None,
        claim_uris: Sequence[str] | None,
        slot_uris: Sequence[str] | None,
        query_intent: str | None,
        connect_filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if (
            requested_limit <= 0
            or user_id is None
            or parsed_type not in {None, ContextType.MEMORY}
            or self.canonical_retriever is None
        ):
            return recalled[:requested_limit]
        scope_keys = list(applicability_scope_keys or [])
        if user_id:
            scope_keys.append(f"memoryos:principal:{user_id}")
        if project_id:
            scope_keys.append(f"memoryos:workspace:{project_id}")
        intent = CanonicalQueryIntent(str(query_intent).upper()) if query_intent else None
        canonical_limit = max(requested_limit * 5, 50) if connect_filters else requested_limit
        canonical = self.canonical_retriever.search(
            CanonicalMemoryQuery(
                text=query,
                tenant_id=tenant_id,
                principal_id=user_id,
                applicability_scope_keys=tuple(dict.fromkeys(scope_keys)),
                states=tuple(str(state).upper() for state in (memory_states or [])),
                memory_types=tuple(str(item) for item in (memory_types or [])),
                claim_uris=tuple(str(item) for item in (claim_uris or [])),
                slot_uris=tuple(str(item) for item in (slot_uris or [])),
                intent=intent,
                limit=canonical_limit,
            )
        )
        non_memory = [item for item in recalled if str(item.get("context_type")) != ContextType.MEMORY.value]
        candidates = []
        if intent == CanonicalQueryIntent.OPTIONS:
            candidates = [
                self._normalize_candidate_memory(item)
                for item in recalled
                if str(item.get("context_type")) == ContextType.MEMORY.value
                and (
                    dict(item.get("metadata", {}) or {}).get("memory_kind") == "memory_candidate"
                    or dict(item.get("metadata", {}) or {}).get("canonical_kind") == "pending_proposal"
                )
                and dict(dict(item.get("metadata", {}) or {}).get("admission", {}) or {}).get("decision") == "pending"
            ]
        authorized = self._filter_connect([*canonical, *candidates, *non_memory], connect_filters)
        return self._dedupe(authorized)[:requested_limit]

    def _normalize_candidate_memory(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            **item,
            "memory_state": "PENDING",
            "memory_category": "candidate",
            "retrieval_source": str(item.get("retrieval_source") or "candidate"),
        }

    def _search_memory_views(
        self,
        query: str,
        *,
        user_id: str | None,
        context_type: ContextType | None,
        limit: int,
        search_scope: str | None,
        retrieval_views: list[str] | None,
        project_id: str,
        adapter_id: str,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        if context_type is not None and context_type != ContextType.MEMORY:
            return []
        plan = self.retrieval_planner.build(
            user_id=user_id,
            adapter_id=adapter_id,
            project_id=project_id,
            search_scope=search_scope,
            retrieval_views=retrieval_views,
        )
        if not plan.retrieval_views:
            return []
        allowed_source_uris = set(
            self._allowed_source_uris(
                user_id=user_id,
                tenant_id=tenant_id,
                context_type=context_type,
                project_id=project_id,
                applicability_scope_keys=None,
                include_candidates=plan.include_candidates,
            )
            or ()
        )
        items: list[dict[str, Any]] = []
        for obj in self.context_db.source_store.list_objects():
            if obj.uri not in allowed_source_uris:
                continue
            if obj.context_type != ContextType.MEMORY:
                continue
            if str(obj.tenant_id or "default") != tenant_id:
                continue
            metadata = dict(obj.metadata or {})
            reviewable_pending = bool(
                plan.include_candidates
                and metadata.get("canonical_kind") == "pending_proposal"
                and obj.lifecycle_state
                in {LifecycleState.PENDING, LifecycleState.RETRYABLE, LifecycleState.CONFIRMED}
            )
            if obj.lifecycle_state != LifecycleState.ACTIVE and not reviewable_pending:
                continue
            if metadata.get("canonical_kind") == "claim" and metadata.get("state") != "ACTIVE":
                continue
            if user_id is not None and obj.owner_user_id != user_id:
                continue
            payload = self._object_payload(obj)
            if not self._matches_retrieval_plan(
                payload, plan.retrieval_views, include_candidates=plan.include_candidates
            ):
                continue
            payload["score"] = self._view_score(query, payload)
            if payload["score"] <= 0:
                continue
            items.append(payload)
        items.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        return items[:limit]

    def _object_payload(self, obj: Any) -> dict[str, Any]:
        try:
            text = self.context_db.source_store.read_content(obj.layers.l2_uri or obj.uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            text = str(obj.metadata.get("summary", obj.title))
        return {
            "uri": obj.uri,
            "score": 0.0,
            "context_type": obj.context_type.value,
            "title": obj.title,
            "text": text,
            "layer": "source_scan",
            "metadata": dict(obj.metadata or {}),
        }

    def _matches_retrieval_plan(
        self, item: dict[str, Any], allowed_views: list[str], *, include_candidates: bool
    ) -> bool:
        metadata = dict(item.get("metadata", {}) or {})
        admission = dict(metadata.get("admission", {}) or {})
        if metadata.get("memory_kind") == "memory_candidate" or admission.get("decision") == "pending":
            if not include_candidates:
                return False
        if admission.get("decision") in {"restricted", "archive_only", "reject"}:
            return False
        views = {str(view) for view in metadata.get("retrieval_views", []) or []}
        return bool(views & set(allowed_views))

    def _view_score(self, query: str, item: dict[str, Any]) -> float:
        terms = [term.lower() for term in str(query).split() if term.strip()]
        if not terms:
            return 0.1
        haystack = " ".join(
            [
                str(item.get("title", "")),
                str(item.get("text", "")),
                str(item.get("metadata", {})),
            ]
        ).lower()
        return sum(1.0 for term in terms if term in haystack)

    def _rerank(self, query: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.reranker is None or not items:
            return items
        try:
            return self.reranker.rerank(query, items)
        except Exception as exc:
            logger.warning(
                "reranker failed; preserving retrieval order",
                extra={"operation": "context_rerank", "error": str(exc), "retryable": True},
            )
            return items

    def _hit_payload(self, hit: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "uri": str(hit.uri),
            "score": float(hit.score),
            "context_type": str(hit.context_type),
            "title": str(getattr(hit, "title", "")),
            "text": str(getattr(hit, "title", "")),
            "layer": str(getattr(hit, "layer", "search")),
            "metadata": dict(getattr(hit, "metadata", {}) or {}),
            "retrieval_source": str(getattr(hit, "source", "lexical")),
        }
        try:
            obj = self.context_db.read_object(payload["uri"])
            payload["context_type"] = obj.context_type.value
            payload["title"] = obj.title
            payload["metadata"] = {**dict(payload["metadata"]), **dict(obj.metadata)}
            try:
                payload["text"] = self.context_db.source_store.read_content(obj.layers.l2_uri or obj.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                payload["text"] = str(obj.metadata.get("summary", obj.title))
            payload["layer_texts"] = self._layer_texts(obj)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return payload
        return payload

    def _context_type(self, context_type: object | None) -> ContextType | None:
        if context_type is None:
            return None
        if isinstance(context_type, ContextType):
            return context_type
        return ContextType(str(context_type))

    def _filter_connect(self, items: list[dict[str, Any]], filters: dict[str, Any] | None) -> list[dict[str, Any]]:
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

    def _filter_project(self, items: list[dict[str, Any]], project_id: str) -> list[dict[str, Any]]:
        if not project_id:
            return items
        scoped = []
        project_types = {"project_rule", "project_decision", "agent_experience"}
        for item in items:
            metadata = dict(item.get("metadata", {}) or {})
            memory_type = str(metadata.get("memory_type", ""))
            scope = dict(metadata.get("scope", {}) or {})
            fields = dict(metadata.get("fields", {}) or {})
            item_project = str(scope.get("project_id") or fields.get("project_id") or "")
            if memory_type in project_types and item_project != project_id:
                continue
            scoped.append(item)
        return scoped

    def _dedupe(self, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        deduped = []
        for item in sorted(contexts, key=lambda row: float(row.get("score", 0.0)), reverse=True):
            identity = str(item.get("retrieval_identity") or item.get("revision_uri") or item.get("uri", ""))
            if identity in seen:
                continue
            seen.add(identity)
            deduped.append(item)
        return deduped

    def _context_text(self, item: dict[str, Any]) -> str:
        text = str(item.get("text") or item.get("title") or "")
        title = str(item.get("title") or "")
        if title and title not in text:
            return f"{title}\n{text}"
        return text

    def _estimate_tokens(self, text: str) -> int:
        return self.token_counter.count(text)

    def _layer_texts(self, obj: Any) -> dict[str, str]:
        values: dict[str, str] = {}
        for name, uri in (("L2", obj.layers.l2_uri or obj.uri), ("L1", obj.layers.l1_uri), ("L0", obj.layers.l0_uri)):
            if not uri:
                continue
            try:
                values[name] = self.context_db.source_store.read_content(uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
        return values

    def _select_layer(self, item: dict[str, Any], query: str, max_tokens: int) -> dict[str, str]:
        layers = dict(item.get("layer_texts", {}) or {})
        metadata = dict(item.get("metadata", {}) or {})
        if metadata.get("canonical_kind") == "claim":
            expected_revision = int(item.get("source_revision") or item.get("revision") or metadata.get("revision", 0))
            layer_revisions = dict(item.get("layer_revisions", {}) or {})
            projection = dict(item.get("projection_record", {}) or {})
            consistent = bool(
                projection
                and projection.get("status") == "completed"
                and int(projection.get("source_revision", 0)) == expected_revision
                and int(projection.get("projection_revision", 0)) == expected_revision
                and layers
                and set(layer_revisions) == set(layers)
                and all(int(revision) == expected_revision for revision in layer_revisions.values())
            )
            if not consistent:
                layers = {}
        l2 = str(layers.get("L2") or self._context_text(item))
        primary_layer = (
            "L2" if layers.get("L2") else ("canonical" if metadata.get("canonical_kind") == "claim" else "L2")
        )
        primary_reason = "full_content" if primary_layer == "L2" else "canonical_projection_unavailable"
        candidates = [(primary_layer, l2, primary_reason)]
        if layers.get("L1"):
            candidates.append(("L1", str(layers["L1"]), "l2_over_budget"))
        excerpt = self._query_excerpt(l2, query, max_tokens)
        l0 = str(layers.get("L0") or "")
        query_terms = [term.casefold() for term in query.split() if term]
        l0_matches_query = bool(query_terms) and any(term in l0.casefold() for term in query_terms)
        if l0 and l0_matches_query:
            candidates.append(("L0", l0, "l1_over_budget"))
        candidates.append(("excerpt", excerpt, "abstract_over_budget"))
        if l0 and not l0_matches_query:
            candidates.append(("L0", l0, "query_excerpt_over_budget"))
        for layer, content, reason in candidates:
            if self._estimate_tokens(content) <= max_tokens:
                return {"layer": layer, "content": content, "reason": reason}
        return {"layer": "excerpt", "content": excerpt[: max(1, max_tokens * 4)], "reason": "excerpt_truncated"}

    def _query_excerpt(self, text: str, query: str, max_tokens: int) -> str:
        terms = [term.lower() for term in query.split() if term]
        lines = text.splitlines() or [text]
        ranked = sorted(enumerate(lines), key=lambda row: (-sum(term in row[1].lower() for term in terms), row[0]))
        selected = [line for _, line in ranked[: max(1, min(8, len(ranked)))]]
        return "\n".join(selected)[: max(1, max_tokens * 4)]
