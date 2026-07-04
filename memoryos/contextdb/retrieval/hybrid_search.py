from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.source_store import IndexStore, SourceStore
from memoryos.contextdb.store.vector_store import VectorStore
from memoryos.providers.embedding import EmbeddingProvider


@dataclass(frozen=True)
class HybridHit:
    uri: str
    title: str
    context_type: str
    score: float
    source: str
    metadata: dict = field(default_factory=dict)


class HybridSearch:
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
                for vector_hit in self.vector_store.search_vector(embedding, namespace=namespace, limit=limit):
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
            except Exception:
                pass
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
                lifecycle_state = obj.lifecycle_state.value
                metadata = {**obj.metadata, **metadata}
        if context_type is not None and hit_type != context_type.value:
            return None
        if filters.get("context_type") and hit_type != filters["context_type"]:
            return None
        if filters.get("owner_user_id") and owner_user_id not in {None, "", filters["owner_user_id"]}:
            return None
        if filters.get("lifecycle_state") and lifecycle_state != filters["lifecycle_state"]:
            return None
        return {"title": title, "context_type": hit_type, "metadata": metadata}
