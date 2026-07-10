from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.layers.context_packer import ContextPacker
from memoryos.contextdb.model.context_type import ContextType
from memoryos.memory.retrieval_plan import MemoryRetrievalPlanner
from memoryos.providers.rerank import Reranker


class ContextAssembler:
    def __init__(self, context_db: ContextDB, *, reranker: Reranker | None = None) -> None:
        self.context_db = context_db
        self.reranker = reranker
        self.retrieval_planner = MemoryRetrievalPlanner()

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
    ) -> list[dict[str, Any]]:
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
            )
            return self._rerank(query, self._filter_connect(results, connect_filters))[:requested_limit]
        search_limit = max(requested_limit * 5, 50) if connect_filters and requested_limit else requested_limit
        hits = self.context_db.search(query, owner_user_id=user_id, context_type=parsed_type, limit=search_limit)
        results = [self._hit_payload(hit) for hit in hits]
        return self._rerank(query, self._filter_connect(results, connect_filters))[:requested_limit]

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
    ) -> dict[str, Any]:
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
            )

        contexts = self._dedupe(contexts)[: max(0, limit)]
        sections = {
            "retrieved_context": [
                {
                    "uri": item["uri"],
                    "content": self._context_text(item),
                    "metadata": item["metadata"],
                    "layer": item.get("layer", "search"),
                    "token_estimate": self._estimate_tokens(self._context_text(item)),
                }
                for item in contexts
            ]
        }
        packed = ContextPacker(total_budget=token_budget).pack(sections)
        selected = packed["slices"].get("retrieved_context", {}).get("items", [])
        source_uris = [str(item.get("uri", "")) for item in selected if item.get("uri")]
        packed_context = "\n\n".join(str(item.get("content", "")) for item in selected if item.get("content"))
        selected_uris = set(source_uris)
        selected_contexts = [item for item in contexts if item["uri"] in selected_uris]
        return {
            "query": query,
            "token_budget": token_budget,
            "contexts": selected_contexts,
            "packed_context": packed_context,
            "source_uris": source_uris,
            "dropped_contexts": packed["dropped_contexts"],
            "connect_metadata": dict(connect_metadata or {}),
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
        items: list[dict[str, Any]] = []
        for obj in self.context_db.source_store.list_objects():
            if obj.context_type != ContextType.MEMORY:
                continue
            if user_id is not None and obj.owner_user_id != user_id:
                continue
            payload = self._object_payload(obj)
            if not self._matches_retrieval_plan(payload, plan.retrieval_views, include_candidates=plan.include_candidates):
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

    def _matches_retrieval_plan(self, item: dict[str, Any], allowed_views: list[str], *, include_candidates: bool) -> bool:
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
        except Exception:
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
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            pass
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
            if all(connect.get(key) == value for key, value in simple_filters.items()):
                matched.append(item)
        return matched

    def _dedupe(self, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        deduped = []
        for item in sorted(contexts, key=lambda row: float(row.get("score", 0.0)), reverse=True):
            uri = str(item.get("uri", ""))
            if uri in seen:
                continue
            seen.add(uri)
            deduped.append(item)
        return deduped

    def _context_text(self, item: dict[str, Any]) -> str:
        text = str(item.get("text") or item.get("title") or "")
        title = str(item.get("title") or "")
        if title and title not in text:
            return f"{title}\n{text}"
        return text

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
