"""向量后端的有界候选召回。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from infrastructure.context.retrieval.embedding import EmbeddingProvider
from infrastructure.context.retrieval.fusion import RetrievalCandidate
from infrastructure.context.retrieval.query_plan import RetrievalQueryPlan
from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.vector import VectorHit, VectorStore, vector_capabilities, vector_row_id
from infrastructure.store.model.catalog import CatalogRecord, ServingTier
from sanitization.context_projection import ContextProjectionSanitizer


class VectorCandidateSource:
    """封装向量能力判断、查询外发清洗和 Catalog 身份复核。"""

    MAX_OVERFETCH = 200

    def __init__(
        self,
        *,
        index_store: IndexStore,
        vector_store: VectorStore | None,
        embedding_provider: EmbeddingProvider | None,
        sanitizer: ContextProjectionSanitizer,
        filters_for_plan: Callable[[RetrievalQueryPlan], dict[str, Any]],
        from_record: Callable[..., RetrievalCandidate],
        finite_score: Callable[[Any], float],
    ) -> None:
        self.index_store = index_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.sanitizer = sanitizer
        self.filters_for_plan = filters_for_plan
        self.from_record = from_record
        self.finite_score = finite_score

    def generate(
        self,
        candidates: Sequence[RetrievalCandidate],
        plan: RetrievalQueryPlan,
    ) -> tuple[tuple[RetrievalCandidate, ...], str, int, int]:
        """优先使用后端原生过滤，否则只在已授权候选集合内做向量排序。"""

        if self.vector_store is None or self.embedding_provider is None or not plan.semantic_query:
            return (), "", 0, 0
        eligible_tiers = {ServingTier.HOT.value, ServingTier.WARM.value}
        candidates = tuple(
            item
            for item in candidates
            if str(item.metadata.get("serving_tier") or "") in eligible_tiers
        )
        vector_limit = min(plan.candidate_limit, self.MAX_OVERFETCH)
        capabilities = vector_capabilities(self.vector_store)
        native_filtering = all(
            (
                capabilities.supports_metadata_filtering,
                capabilities.supports_namespace_filtering,
                capabilities.supports_time_filtering,
            )
        )
        if native_filtering:
            return self._native_filtered(plan, vector_limit=vector_limit)
        by_row_id: dict[str, list[RetrievalCandidate]] = {}
        for item in candidates:
            by_row_id.setdefault(vector_row_id(plan.tenant_id or "default", item.record_key), []).append(item)
        bounded = tuple(by_row_id)[:vector_limit]
        if not bounded:
            return (), "vector_requires_structured_candidates", 0, 0
        candidate_search = getattr(self.vector_store, "search_vector_candidates", None)
        if not callable(candidate_search):
            return (), "vector_backend_lacks_bounded_candidates", 0, 0
        try:
            embedding = self.embedding_provider.embed(self._provider_query(plan.semantic_query))
            raw_hits: Any = candidate_search(embedding, bounded, limit=vector_limit)
        except Exception as exc:
            # 向量索引允许最终一致；失败时由本地确定性分支降级承接。
            return (), f"vector_fallback:{type(exc).__name__}", 0, len(bounded)
        if (
            not isinstance(raw_hits, Sequence)
            or isinstance(raw_hits, str | bytes | bytearray)
            or any(not isinstance(hit, VectorHit) for hit in raw_hits)
        ):
            return (), "vector_fallback:InvalidResponse", 0, len(bounded)
        result: list[RetrievalCandidate] = []
        for hit in list(raw_hits)[:vector_limit]:
            for item in by_row_id.get(str(hit.uri), ()):
                result.append(item.with_branch("vector", float(hit.score), len(result) + 1))
                if len(result) >= vector_limit:
                    break
        degraded = "" if capabilities.supports_metadata_filtering else "bounded_vector_candidate_fallback"
        return tuple(result), degraded, 0, len(bounded)

    def _native_filtered(
        self,
        plan: RetrievalQueryPlan,
        *,
        vector_limit: int,
    ) -> tuple[tuple[RetrievalCandidate, ...], str, int, int]:
        """让生产后端先做可信过滤，再用 SQL Catalog 复核每个命中。"""

        filtered_search = getattr(self.vector_store, "search_vector_filtered", None)
        lister = getattr(self.index_store, "list_catalog", None)
        embedding_provider = self.embedding_provider
        if not callable(filtered_search) or not callable(lister) or embedding_provider is None:
            return (), "vector_filtered_contract_missing", 0, 0
        filters = {
            **self.filters_for_plan(plan),
            "serving_tier": (ServingTier.HOT.value, ServingTier.WARM.value),
        }
        try:
            embedding = embedding_provider.embed(self._provider_query(plan.semantic_query))
            raw_hits: Any = filtered_search(
                embedding,
                namespace=plan.tenant_id or "default",
                filters=filters,
                limit=vector_limit,
            )
        except Exception as exc:
            return (), f"vector_fallback:{type(exc).__name__}", 0, vector_limit
        if (
            not isinstance(raw_hits, Sequence)
            or isinstance(raw_hits, str | bytes | bytearray)
            or any(not isinstance(hit, VectorHit) for hit in raw_hits)
        ):
            return (), "vector_fallback:InvalidResponse", 0, vector_limit
        hits = tuple(raw_hits[:vector_limit])
        if not hits:
            return (), "", 0, vector_limit
        scores: dict[str, float] = {}
        ordered_record_keys: list[str] = []
        for hit in hits:
            metadata = dict(hit.metadata or {})
            record_key = str(metadata.get("catalog_record_key") or "")
            tenant_id = str(metadata.get("tenant_id") or "")
            expected_row_id = vector_row_id(tenant_id, record_key) if tenant_id and record_key else ""
            if not record_key or tenant_id != str(plan.tenant_id or "default") or str(hit.uri) != expected_row_id:
                continue
            if record_key not in scores:
                ordered_record_keys.append(record_key)
            scores[record_key] = max(scores.get(record_key, 0.0), self.finite_score(hit.score))
        if not ordered_record_keys:
            return (), "", 0, vector_limit
        raw_records: Any = lister(
            tenant_id=plan.tenant_id or "default",
            filters={**filters, "record_keys": tuple(ordered_record_keys)},
            limit=min(vector_limit, len(ordered_record_keys)),
        )
        records = raw_records if isinstance(raw_records, Sequence) else ()
        by_record_key = {
            record.record_key: record
            for record in records
            if isinstance(record, CatalogRecord) and record.record_key in scores
        }
        result = tuple(
            self.from_record(
                by_record_key[record_key],
                branch="vector",
                score=scores[record_key],
            ).with_branch("vector", scores[record_key], rank)
            for rank, record_key in enumerate(ordered_record_keys, start=1)
            if record_key in by_record_key
        )
        return result, "", 0, vector_limit

    def _provider_query(self, query: str) -> str:
        """只把经过失败关闭清洗的查询副本发送给外部嵌入服务。"""

        projection = self.sanitizer.sanitize(
            title="retrieval query",
            l1_text=query,
            metadata={},
            source_kind="retrieval_query",
        )
        return projection.l1_text


__all__ = ["VectorCandidateSource"]
