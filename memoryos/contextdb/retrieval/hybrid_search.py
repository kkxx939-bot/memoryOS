"""上下文数据库里的混合检索。"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.source_store import IndexHit, IndexStore, SourceStore
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

    DEFAULT_MIN_VECTOR_SIMILARITY = 0.20
    _OWNER_OPTIONAL_CONTEXT_TYPES = {ContextType.RESOURCE.value, ContextType.SKILL.value}

    def __init__(
        self,
        index_store: IndexStore,
        vector_store: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        source_store: SourceStore | None = None,
        min_vector_similarity: float = DEFAULT_MIN_VECTOR_SIMILARITY,
    ) -> None:
        self.index_store = index_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.source_store = source_store
        self.min_vector_similarity = self._validated_threshold(
            min_vector_similarity,
            "min_vector_similarity",
        )

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
        if not isinstance(index_hits, list | tuple):
            index_hits = []
        combined: dict[str, dict] = {}
        for hit in index_hits:
            if not isinstance(hit, IndexHit):
                continue
            if self._finite_nonnegative(hit.score) is None:
                continue
            hit_metadata = self._mapping(hit.metadata)
            retrieval_scores = self._mapping(hit_metadata.get("retrieval_scores", {}))
            item = self._vector_item(hit.uri, hit.metadata, filters, context_type)
            if item is None:
                continue
            combined[hit.uri] = {
                "uri": hit.uri,
                "title": item["title"] or hit.title,
                "context_type": item["context_type"] or hit.context_type,
                "index_score": self._bounded_score(hit.score),
                "vector_score": None,
                "source": "index",
                "metadata": item["metadata"],
                "retrieval_scores": retrieval_scores,
            }
        if self.vector_store is not None and self.embedding_provider is not None:
            try:
                embedding = self.embedding_provider.embed(query)
                vector_limit = limit
                if self.source_store is not None and "allowed_uris" in filters:
                    vector_limit = max(limit, len(self.source_store.list_objects()))
                for vector_hit in self.vector_store.search_vector(embedding, namespace=namespace, limit=vector_limit):
                    normalized_vector_score = self._bounded_score(vector_hit.score)
                    if normalized_vector_score < self.min_vector_similarity:
                        continue
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
                            "retrieval_scores": {},
                        },
                    )
                    existing["title"] = existing.get("title") or item["title"]
                    existing["context_type"] = existing.get("context_type") or item["context_type"]
                    existing["metadata"] = {**dict(existing.get("metadata", {})), **item["metadata"]}
                    existing["vector_score"] = normalized_vector_score
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
            item_vector_score = item.get("vector_score")
            if index_score is not None and item_vector_score is not None:
                score = float(index_score) * 0.55 + float(item_vector_score) * 0.45
            elif index_score is not None:
                score = float(index_score)
            else:
                score = float(item_vector_score or 0.0)
            metadata = self._mapping(item.get("metadata", {}))
            retrieval_scores = self._mapping(item.get("retrieval_scores", {}))
            retrieval_scores["lexical"] = self._bounded_score(retrieval_scores.get("lexical", 0.0))
            retrieval_scores["identity"] = self._bounded_score(retrieval_scores.get("identity", 0.0))
            retrieval_scores["vector"] = self._bounded_score(
                item_vector_score if item_vector_score is not None else retrieval_scores.get("vector", 0.0)
            )
            retrieval_scores["base_relevance"] = max(
                retrieval_scores["lexical"],
                retrieval_scores["identity"],
                retrieval_scores["vector"],
            )
            if retrieval_scores["base_relevance"] <= 0:
                continue
            retrieval_scores["score"] = score
            metadata["retrieval_scores"] = retrieval_scores
            results.append(
                HybridHit(
                    uri=str(item["uri"]),
                    title=str(item.get("title", "")),
                    context_type=str(item.get("context_type", "")),
                    score=round(score, 6),
                    source=str(item.get("source", "index")),
                    metadata=metadata,
                )
            )
        results.sort(key=lambda hit: hit.score, reverse=True)
        return results[:limit]

    def _vector_item(
        self,
        uri: str,
        metadata: Any,
        filters: dict,
        context_type: ContextType | None,
    ) -> dict | None:
        metadata = self._mapping(metadata)
        projected_revision = metadata.get("projection_source_revision")
        if projected_revision is None:
            projected_revision = metadata.get("source_revision")
        title = str(metadata.get("title", ""))
        hit_type = str(metadata.get("context_type", ""))
        owner_user_id = metadata.get("owner_user_id")
        tenant_id = metadata.get("tenant_id")
        lifecycle_state = metadata.get("lifecycle_state")
        if self.source_store is not None:
            try:
                obj = self.source_store.read_object(uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError, TypeError, ValueError):
                return None
            if obj.lifecycle_state != LifecycleState.ACTIVE:
                return None
            title = obj.title
            hit_type = obj.context_type.value
            owner_user_id = obj.owner_user_id
            tenant_id = obj.tenant_id or "default"
            lifecycle_state = obj.lifecycle_state.value
            if not isinstance(obj.metadata, Mapping):
                return None
            source_metadata = self._mapping(obj.metadata)
            metadata = {**metadata, **source_metadata}
            canonical_revision = source_metadata.get("revision")
            if (
                projected_revision is not None
                and canonical_revision is not None
                and self._revision(projected_revision) != self._revision(canonical_revision)
            ):
                return None
        if projected_revision is not None:
            normalized_revision = self._revision(projected_revision)
            if normalized_revision is None:
                return None
            metadata["projection_source_revision"] = normalized_revision
        if context_type is not None and hit_type != context_type.value:
            return None
        if filters.get("context_type") and hit_type != filters["context_type"]:
            return None
        expected_owner = filters.get("owner_user_id")
        if expected_owner:
            if hit_type in self._OWNER_OPTIONAL_CONTEXT_TYPES:
                if owner_user_id not in {None, "", expected_owner}:
                    return None
            elif owner_user_id != expected_owner:
                return None
        if filters.get("tenant_id") and str(tenant_id or "default") != str(filters["tenant_id"]):
            return None
        if "allowed_uris" in filters and uri not in set(filters.get("allowed_uris", []) or []):
            return None
        if filters.get("lifecycle_state") and lifecycle_state != filters["lifecycle_state"]:
            return None
        for field_name in ("scope", "fields", "connect", "admission"):
            nested = metadata.get(field_name)
            if nested is not None and not isinstance(nested, Mapping):
                return None
        scope = self._mapping(metadata.get("scope", {}))
        fields = self._mapping(metadata.get("fields", {}))
        connect = self._mapping(metadata.get("connect", {}))
        admission = self._mapping(metadata.get("admission", {}))
        for filter_name, actual in (
            ("claim_state", metadata.get("state") or metadata.get("claim_state")),
            ("slot_id", metadata.get("slot_id")),
            ("memory_type", metadata.get("memory_type")),
        ):
            expected = filters.get(filter_name)
            if expected is None:
                continue
            values = set(expected) if isinstance(expected, list | tuple | set | frozenset) else {expected}
            if actual not in values:
                return None
        required_scopes = set(filters.get("applicability_scope_keys", []) or [])
        if required_scopes:
            raw_applicability = scope.get("applicability", {})
            if not isinstance(raw_applicability, Mapping):
                return None
            applicability = self._mapping(raw_applicability)
            if not isinstance(applicability.get("all_of", []), list | tuple):
                return None
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
            if memory_type in {"project_rule", "project_decision", "agent_experience"} and project_id != str(
                filters["project_id"]
            ):
                return None
        if filters.get("adapter_id") and str(
            connect.get("adapter_id") or metadata.get("source_adapter_id") or ""
        ) != str(filters["adapter_id"]):
            return None
        if admission.get("decision") in {"pending", "restricted", "archive_only", "reject"}:
            return None
        return {"title": title, "context_type": hit_type, "metadata": metadata}

    def _bounded_score(self, value: Any) -> float:
        if isinstance(value, bool):
            return 0.0
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(score):
            return 0.0
        return max(0.0, min(1.0, score))

    def _mapping(self, value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    def _validated_threshold(self, value: Any, field_name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{field_name} must be a finite number between 0 and 1")
        try:
            threshold = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a finite number between 0 and 1") from exc
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{field_name} must be a finite number between 0 and 1")
        return threshold

    def _revision(self, value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            revision = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return revision if revision >= 0 else None

    def _finite_nonnegative(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        return score if math.isfinite(score) and score >= 0 else None
