"""Internal hybrid candidate primitive, not a public retrieval entrypoint.

The Unified Context orchestrator may use this component only through its
bounded compatibility adapter.  SDK, HTTP, MCP, and context assembly callers
must not call :meth:`HybridSearch.search` as a second orchestration chain.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from memoryos.contextdb.catalog import CatalogRecord
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.source_store import (
    IndexHit,
    IndexStore,
    SourceStore,
    is_canonical_memory_object,
    is_canonical_memory_uri,
)
from memoryos.contextdb.store.vector_store import VectorStore, vector_row_id
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
    """Internal lexical/vector primitive for planners and bounded adapters."""

    DEFAULT_MIN_VECTOR_SIMILARITY = 0.20
    MAX_VECTOR_OVERFETCH = 200
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
        source_snapshot: Mapping[str, Any] | None = None,
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
            item = self._vector_item(
                hit.uri,
                hit.metadata,
                filters,
                context_type,
                source_snapshot=source_snapshot,
            )
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
                if not embedding or any(not math.isfinite(float(value)) for value in embedding):
                    raise ValueError("embedding provider returned non-finite values")
                allowed_uris = tuple(dict.fromkeys(str(uri) for uri in filters.get("allowed_uris", []) or []))
                candidate_search = getattr(self.vector_store, "search_vector_candidates", None)
                degraded_mode = ""
                vector_hits: Sequence[Any]
                if "allowed_uris" in filters and callable(candidate_search):
                    candidate_ids = self._vector_candidate_ids(allowed_uris, filters)
                    raw_vector_hits = candidate_search(embedding, candidate_ids, limit=limit)
                    vector_hits = raw_vector_hits if isinstance(raw_vector_hits, Sequence) else ()
                else:
                    vector_limit = min(self.MAX_VECTOR_OVERFETCH, max(limit, limit * 4))
                    vector_hits = self.vector_store.search_vector(
                        embedding,
                        namespace=namespace,
                        limit=vector_limit,
                    )
                    if "allowed_uris" in filters:
                        degraded_mode = "bounded_vector_overfetch"
                for vector_hit in vector_hits:
                    normalized_vector_score = self._bounded_score(vector_hit.score)
                    if normalized_vector_score < self.min_vector_similarity:
                        continue
                    public_uri = self._public_vector_uri(vector_hit.uri, vector_hit.metadata)
                    if not public_uri:
                        continue
                    item = self._vector_item(
                        public_uri,
                        vector_hit.metadata,
                        filters,
                        context_type,
                        source_snapshot=source_snapshot,
                    )
                    if item is None:
                        continue
                    existing = combined.setdefault(
                        public_uri,
                        {
                            "uri": public_uri,
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
                    existing["metadata"]["vector_storage_id"] = str(vector_hit.uri)
                    if degraded_mode:
                        existing["metadata"]["vector_degraded_mode"] = degraded_mode
                    existing["vector_score"] = normalized_vector_score
                    existing["source"] = "hybrid" if existing.get("index_score") is not None else "vector"
            except Exception as exc:
                # Import lazily: HybridSearch participates in session planner
                # initialization, while the canonical visibility package also
                # imports retrieval components.  The runtime type check keeps
                # the module graph acyclic without weakening fail-closed
                # handling.
                from memoryos.memory.canonical.visibility import (
                    CommittedStateIntegrityError,
                )

                if isinstance(exc, CommittedStateIntegrityError):
                    # A vector outage may fall back to lexical search; a
                    # broken committed-state proof may never become an empty
                    # successful result.
                    raise
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

    def _vector_candidate_ids(self, allowed_uris: tuple[str, ...], filters: Mapping[str, Any]) -> tuple[str, ...]:
        """Map a bounded public allowlist to tenant-scoped vector row IDs."""

        tenant_id = str(filters.get("tenant_id") or "default")
        identifiers: list[str] = []
        getter = getattr(self.index_store, "get_catalog_by_uri", None)
        for public_uri in allowed_uris[: self.MAX_VECTOR_OVERFETCH]:
            # Preserve unique legacy rows while new projections use the
            # tenant + Catalog identity below.
            identifiers.append(public_uri)
            if not callable(getter):
                continue
            raw_records: Any = getter(public_uri, tenant_id=tenant_id, limit=16)
            if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes, bytearray)):
                raise TypeError("Catalog vector identity lookup returned an invalid result")
            for record in raw_records:
                if not isinstance(record, CatalogRecord) or record.tenant_id != tenant_id or record.uri != public_uri:
                    continue
                identifiers.append(vector_row_id(tenant_id, record.record_key))
        return tuple(dict.fromkeys(identifiers))[: self.MAX_VECTOR_OVERFETCH]

    @staticmethod
    def _public_vector_uri(storage_id: object, metadata: Any) -> str:
        values = dict(metadata) if isinstance(metadata, Mapping) else {}
        for key in ("public_uri", "uri", "claim_uri", "source_uri"):
            value = str(values.get(key) or "")
            if value.startswith("memoryos://"):
                return value
        legacy = str(storage_id)
        return legacy if legacy.startswith("memoryos://") and not legacy.startswith("memoryos-vector://") else ""

    def _vector_item(
        self,
        uri: str,
        metadata: Any,
        filters: dict,
        context_type: ContextType | None,
        *,
        source_snapshot: Mapping[str, Any] | None = None,
    ) -> dict | None:
        metadata = self._mapping(metadata)
        projected_revision = metadata.get("projection_source_revision")
        if projected_revision is None:
            projected_revision = metadata.get("source_revision")
        canonical_validation_revision = projected_revision
        current_slot_projection = str(metadata.get("record_kind") or "") == "current_slot" or str(
            metadata.get("canonical_kind") or ""
        ) == "current_slot_projection"
        if current_slot_projection:
            canonical_validation_revision = metadata.get("active_claim_revision")
            if canonical_validation_revision is None:
                canonical_validation_revision = metadata.get("canonical_revision")
            if canonical_validation_revision is None:
                return None
        title = str(metadata.get("title", ""))
        hit_type = str(metadata.get("context_type", ""))
        owner_user_id = metadata.get("owner_user_id")
        tenant_id = metadata.get("tenant_id")
        lifecycle_state = metadata.get("lifecycle_state")
        if self.source_store is not None:
            source_validation_uri = uri
            if current_slot_projection:
                source_validation_uri = str(
                    metadata.get("active_claim_uri") or metadata.get("canonical_claim_uri") or ""
                )
                if not source_validation_uri:
                    return None
            try:
                if source_snapshot is not None:
                    obj = source_snapshot.get(source_validation_uri)
                    if obj is None:
                        return None
                elif is_canonical_memory_uri(source_validation_uri):
                    from memoryos.memory.canonical.visibility import read_committed_canonical

                    obj = read_committed_canonical(self.source_store, source_validation_uri).object
                else:
                    obj = self.source_store.read_object(source_validation_uri)
                    if is_canonical_memory_object(obj):
                        from memoryos.memory.canonical.visibility import read_committed_canonical

                        obj = read_committed_canonical(self.source_store, source_validation_uri).object
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
                canonical_validation_revision is not None
                and canonical_revision is not None
                and self._revision(canonical_validation_revision) != self._revision(canonical_revision)
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
        if metadata.get("canonical_kind") in {"claim", "slot", "pending_proposal"}:
            try:
                from memoryos.memory.canonical.scope import MemoryScope

                canonical_scope = MemoryScope.from_dict(scope)
            except (KeyError, TypeError, ValueError):
                return None
            if canonical_scope.canonical_subject is None:
                return None
            if canonical_scope.visibility.tenant_id != str(tenant_id or "default"):
                return None
            if canonical_scope.authority.inferred:
                return None
            asserted_by = str(
                metadata.get("asserted_by")
                or (owner_user_id if metadata.get("canonical_kind") == "pending_proposal" else "")
                or ""
            )
            asserted_by_service = str(metadata.get("asserted_by_service") or "")
            if (canonical_scope.authority.principal_ids or canonical_scope.authority.service_ids) and not (
                asserted_by in set(canonical_scope.authority.principal_ids)
                or asserted_by_service in set(canonical_scope.authority.service_ids)
            ):
                return None
            actual_scopes = {item.key for item in canonical_scope.applicability.all_of}
        else:
            raw_applicability = scope.get("applicability", {})
            if raw_applicability is not None and not isinstance(raw_applicability, Mapping):
                return None
            applicability = self._mapping(raw_applicability or {})
            raw_scope_items = applicability.get("all_of", [])
            if not isinstance(raw_scope_items, list | tuple):
                return None
            try:
                from memoryos.memory.canonical.scope import scope_keys_from_payloads

                actual_scopes = set(scope_keys_from_payloads(raw_scope_items))
            except (KeyError, TypeError, ValueError):
                return None
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


# This module remains importable by internal planners and compatibility tests,
# but it deliberately has no public wildcard surface.  Product callers use
# RetrievalOptions -> QueryPlanner -> UnifiedRetrievalOrchestrator.
__all__: list[str] = []
