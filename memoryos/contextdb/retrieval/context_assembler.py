from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.layers.context_packer import ContextPacker
from memoryos.contextdb.model.context_type import ContextType


class ContextAssembler:
    def __init__(self, context_db: ContextDB) -> None:
        self.context_db = context_db

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        context_type: object | None = None,
        limit: int = 10,
        connect_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        parsed_type = self._context_type(context_type)
        hits = self.context_db.search(query, owner_user_id=user_id, context_type=parsed_type, limit=limit)
        results = [self._hit_payload(hit) for hit in hits]
        return self._filter_connect(results, connect_filters)[: max(0, limit)]

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
                    )
                )
        else:
            contexts = self.search(query, user_id=user_id, limit=limit, connect_filters=connect_filters)

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
        filters = {key: value for key, value in dict(filters or {}).items() if value not in {None, ""}}
        if not filters:
            return items
        allowed = {"connect_type", "adapter_id", "run_mode", "world_domain", "source_kind"}
        simple_filters = {key: value for key, value in filters.items() if key in allowed}
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
