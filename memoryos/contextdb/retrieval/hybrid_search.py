"""上下文数据库里的混合检索。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.source_store import IndexStore, SourceStore
from memoryos.contextdb.store.vector_store import VectorStore
from memoryos.providers.embedding import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HybridHit:
    """保存 HybridHit 需要的这组数据。"""

    uri: str
    title: str
    context_type: str
    score: float
    source: str
    metadata: dict = field(default_factory=dict)


class HybridSearch:
    """合并关键词和向量结果，过滤条件始终以源数据为准。"""

    def __init__(
        self,
        index_store: IndexStore,
        vector_store: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        source_store: SourceStore | None = None,
    ) -> None:
        self.index_store = index_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.source_store = source_store

    def search(
        self,
        query: str,
        filters: dict | None = None,
        namespace: str = "",
        context_type: ContextType | None = None,
        limit: int = 10,
    ) -> list[HybridHit]:
        """按给定条件查找匹配结果。"""

        filters = dict(filters or {})
        if context_type is not None:
            filters["context_type"] = context_type.value
        index_hits = self.index_store.search(query, filters=filters, limit=limit)
        combined: dict[str, dict] = {}
        for hit in index_hits:
            combined[hit.uri] = {
                "uri": hit.uri,
                "title": hit.title,
                "context_type": hit.context_type,
                "index_score": min(1.0, float(hit.score)),
                "vector_score": None,
                "source": "index",
                "metadata": dict(hit.metadata),
            }
        if self.vector_store is not None and self.embedding_provider is not None:
            try:
                embedding = self.embedding_provider.embed(query)
                vector_limit = limit
                if self.source_store is not None and "allowed_uris" in filters:
                    vector_limit = max(limit, len(self.source_store.list_objects()))
                for vector_hit in self.vector_store.search_vector(embedding, namespace=namespace, limit=vector_limit):
                    item = self._vector_item(vector_hit.uri, vector_hit.metadata, filters, context_type)
                    if item is None:
                        continue
                    existing = combined.setdefault(
                        vector_hit.uri,
                        {
                            "uri": vector_hit.uri,
                            "title": item["title"],
                            "context_type": item["context_type"],
                            "index_score": None,
                            "vector_score": None,
                            "source": "vector",
                            "metadata": item["metadata"],
                        },
                    )
                    existing["title"] = existing.get("title") or item["title"]
                    existing["context_type"] = existing.get("context_type") or item["context_type"]
                    existing["metadata"] = {**item["metadata"], **dict(existing.get("metadata", {}))}
                    existing["vector_score"] = min(1.0, float(vector_hit.score))
                    existing["source"] = "hybrid" if existing.get("index_score") is not None else "vector"
            except Exception as exc:
                logger.warning(
                    "HybridSearch vector branch failed; falling back to lexical search: %s",
                    exc,
                    exc_info=True,
                )
        results = []
        for item in combined.values():
            index_score = item.get("index_score")
            vector_score = item.get("vector_score")
            if index_score is not None and vector_score is not None:
                score = float(index_score) * 0.55 + float(vector_score) * 0.45
            elif index_score is not None:
                score = float(index_score)
            else:
                score = float(vector_score or 0.0)
            results.append(
                HybridHit(
                    uri=str(item["uri"]),
                    title=str(item.get("title", "")),
                    context_type=str(item.get("context_type", "")),
                    score=round(score, 6),
                    source=str(item.get("source", "index")),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        results.sort(key=lambda hit: hit.score, reverse=True)
        return results[:limit]

    def _vector_item(
        self,
        uri: str,
        metadata: dict,
        filters: dict,
        context_type: ContextType | None,
    ) -> dict | None:
        metadata = dict(metadata or {})
        title = str(metadata.get("title", ""))
        hit_type = str(metadata.get("context_type", ""))
        owner_user_id = metadata.get("owner_user_id")
        tenant_id = metadata.get("tenant_id")
        lifecycle_state = metadata.get("lifecycle_state")
        if self.source_store is not None:
            try:
                obj = self.source_store.read_object(uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                obj = None
            if obj is not None:
                if obj.lifecycle_state in {LifecycleState.DELETED, LifecycleState.OBSOLETE, LifecycleState.ARCHIVED}:
                    return None
                title = obj.title
                hit_type = obj.context_type.value
                owner_user_id = obj.owner_user_id
                tenant_id = obj.tenant_id or "default"
                lifecycle_state = obj.lifecycle_state.value
                metadata = {**obj.metadata, **metadata}
        if context_type is not None and hit_type != context_type.value:
            return None
        if filters.get("context_type") and hit_type != filters["context_type"]:
            return None
        if filters.get("owner_user_id") and owner_user_id not in {None, "", filters["owner_user_id"]}:
            return None
        if filters.get("tenant_id") and str(tenant_id or "default") != str(filters["tenant_id"]):
            return None
        if "allowed_uris" in filters and uri not in set(filters.get("allowed_uris", []) or []):
            return None
        if filters.get("lifecycle_state") and lifecycle_state != filters["lifecycle_state"]:
            return None
        scope = dict(metadata.get("scope", {}) or {})
        fields = dict(metadata.get("fields", {}) or {})
        connect = dict(metadata.get("connect", {}) or {})
        admission = dict(metadata.get("admission", {}) or {})
        for filter_name, actual in (
            ("claim_state", metadata.get("state") or metadata.get("claim_state")),
            ("slot_id", metadata.get("slot_id")),
            ("memory_type", metadata.get("memory_type")),
        ):
            expected = filters.get(filter_name)
            if expected is None:
                continue
            values = set(expected) if isinstance(expected, (list, tuple, set, frozenset)) else {expected}
            if actual not in values:
                return None
        required_scopes = set(filters.get("applicability_scope_keys", []) or [])
        if required_scopes:
            applicability = dict(scope.get("applicability", {}) or {})
            actual_scopes = {
                f"{item.get('namespace', 'memoryos')}:{item.get('kind')}:{item.get('id')}"
                for item in applicability.get("all_of", []) or []
                if isinstance(item, dict) and item.get("kind") and item.get("id")
            }
            if not actual_scopes.issubset(required_scopes):
                return None
        project_id = str(scope.get("project_id") or fields.get("project_id") or "")
        if filters.get("project_id"):
            memory_type = str(metadata.get("memory_type") or "")
            if memory_type in {"project_rule", "project_decision", "agent_experience"} and project_id != str(filters["project_id"]):
                return None
        if filters.get("adapter_id") and str(connect.get("adapter_id") or metadata.get("source_adapter_id") or "") != str(filters["adapter_id"]):
            return None
        if admission.get("decision") in {"pending", "restricted", "archive_only", "reject"}:
            return None
        return {"title": title, "context_type": hit_type, "metadata": metadata}
