"""有界且具备租户隔离的 Catalog 精确查询与全文检索。"""

from __future__ import annotations

from infrastructure.store.sqlite._common import (
    _BOUNDED_FTS_OVERFETCH,
    _FTS_BM25,
    Any,
    CatalogRecord,
    IndexHit,
    Mapping,
    sqlite3,
)


class CatalogSearchOperations:
    """一次只查询一个租户，并按 record key 合并不同候选。"""

    def __init__(self, store: Any) -> None:
        self._store = store

    def search(
        self,
        query: str,
        *,
        tenant_id: str,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
    ) -> list[IndexHit]:
        return self.search_catalog(query, tenant_id=tenant_id, filters=filters, limit=limit)

    def search_catalog(
        self,
        query: str,
        *,
        tenant_id: str,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
    ) -> list[IndexHit]:
        resolved_tenant = self._store._catalog._require_tenant(tenant_id)
        normalized = self._store._catalog._tenant_filters(filters, resolved_tenant)
        bounded = self._store._bounded_limit(limit)
        candidates: dict[str, IndexHit] = {}
        for hit in self._search_metadata_exact(str(query), normalized, bounded):
            candidates[str(hit.metadata.get("catalog_record_key") or hit.uri)] = hit
        if self._store._match_query(str(query)):
            for hit in self._search_fts(str(query), normalized, bounded):
                key = str(hit.metadata.get("catalog_record_key") or hit.uri)
                current = candidates.get(key)
                if current is None or hit.score > current.score:
                    candidates[key] = hit
        return sorted(
            candidates.values(),
            key=lambda hit: (-float(hit.score), str(hit.metadata.get("catalog_record_key") or hit.uri)),
        )[:bounded]

    def _search_fts(
        self,
        query: str,
        filters: dict[str, Any],
        limit: int,
    ) -> list[IndexHit]:
        match_query = self._store._match_query(query)
        if not match_query or not self._store.fts_enabled:
            return []
        filter_sql, filter_params = self._store._base_filter_sql(filters)
        bounded_candidates = min(
            _BOUNDED_FTS_OVERFETCH,
            max(self._store._bounded_limit(limit), self._store._bounded_limit(limit) * 8),
        )
        tenant_id = str(filters["tenant_id"])
        sql = (
            f"SELECT c.*, {_FTS_BM25} AS lexical_rank FROM contexts_fts "
            "JOIN context_fts_map AS fm ON fm.fts_rowid = contexts_fts.rowid "
            "AND fm.tenant_id = contexts_fts.tenant_id "
            "AND fm.record_key = contexts_fts.record_key "
            "JOIN contexts AS c ON c.tenant_id = fm.tenant_id AND c.record_key = fm.record_key "
            "WHERE contexts_fts MATCH ? AND contexts_fts.tenant_id = ? "
            f"{filter_sql} ORDER BY lexical_rank, c.record_key LIMIT ?"
        )
        params: list[Any] = [match_query, tenant_id, *filter_params, bounded_candidates]
        with self._store._connect() as conn:
            rows = self._store._online_fetchall(conn, sql, params)
        return [
            self._hit_from_row(
                row,
                lexical=self._store._lexical_relevance(
                    query,
                    " ".join((str(row["title"]), str(row["l0_text"]), str(row["content_text"]))),
                ),
                lexical_rank=float(row["lexical_rank"]),
            )
            for row in rows
        ]

    def _search_metadata_exact(
        self,
        query: str,
        filters: dict[str, Any],
        limit: int,
    ) -> list[IndexHit]:
        raw = str(query or "").strip()
        if not raw:
            return []
        filter_sql, params = self._store._base_filter_sql(filters)
        sql = (
            "SELECT c.* FROM contexts AS c WHERE ("
            "c.record_key = ? OR c.uri = ? OR c.source_uri = ? OR c.document_id = ? "
            "OR c.block_id = ? OR c.scene_key = ? OR c.action = ? OR c.support_anchor_uri = ?) "
            f"{filter_sql} ORDER BY c.updated_at DESC, c.record_key LIMIT ?"
        )
        with self._store._connect() as conn:
            rows = self._store._online_fetchall(
                conn,
                sql,
                [raw, raw, raw, raw, raw, raw, raw, raw, *params, self._store._bounded_limit(limit)],
            )
        return [self._hit_from_row(row, identity=1.0, identity_rank=1.0) for row in rows]

    @staticmethod
    def _exact_hit_from_record(record: CatalogRecord) -> IndexHit:
        return IndexHit(
            uri=record.uri,
            score=1.0,
            context_type=record.context_type,
            title=record.title,
            layer="l1",
            metadata={
                **dict(record.metadata),
                "catalog_record_key": record.record_key,
                "tenant_id": record.tenant_id,
                "owner_user_id": record.owner_user_id,
                "workspace_id": record.workspace_id,
                "session_id": record.session_id,
                "adapter_id": record.adapter_id,
                "context_type": record.context_type,
                "source_kind": record.source_kind,
                "record_kind": record.record_kind,
                "lifecycle_state": record.lifecycle_state,
                "event_time": record.event_time,
                "transaction_time": record.transaction_time,
                "updated_at": record.updated_at,
                "document_id": record.document_id,
                "block_id": record.block_id,
                "document_kind": record.document_kind,
                "document_revision": record.document_revision,
                "projection_generation": record.projection_generation,
                "serving_tier": record.serving_tier,
                "projection_status": record.projection_status,
            },
        )

    def _hit_from_row(
        self,
        row: sqlite3.Row,
        lexical: float = 0.0,
        lexical_rank: float | None = None,
        vector: float = 0.0,
        identity: float = 0.0,
        identity_rank: float | None = None,
    ) -> IndexHit:
        components = self._score_components(
            row,
            lexical=lexical,
            lexical_rank=lexical_rank,
            vector=vector,
            identity=identity,
            identity_rank=identity_rank,
        )
        metadata = self._store._json_mapping(row["metadata_json"])
        metadata.update(
            {
                "catalog_record_key": str(row["record_key"]),
                "tenant_id": str(row["tenant_id"]),
                "owner_user_id": str(row["owner_user_id"]),
                "workspace_id": str(row["workspace_id"]),
                "session_id": str(row["session_id"]),
                "adapter_id": str(row["adapter_id"]),
                "context_type": str(row["context_type"]),
                "source_kind": str(row["source_kind"]),
                "record_kind": str(row["record_kind"]),
                "lifecycle_state": str(row["lifecycle_state"]),
                "event_time": str(row["event_time"]),
                "transaction_time": str(row["transaction_time"]),
                "updated_at": str(row["updated_at"]),
                "source_uri": str(row["source_uri"]),
                "source_digest": str(row["source_digest"]),
                "document_id": str(row["document_id"]),
                "block_id": str(row["block_id"]),
                "document_kind": str(row["document_kind"]),
                "document_revision": int(row["document_revision"]),
                "projection_generation": int(row["projection_generation"]),
                "serving_tier": str(row["serving_tier"]),
                "projection_status": str(row["projection_status"]),
                "score_components": components,
                "retrieval_scores": components,
            }
        )
        return IndexHit(
            uri=str(row["uri"]),
            score=components["score"],
            context_type=str(row["context_type"]),
            title=str(row["title"]),
            layer="l1",
            metadata=metadata,
        )

    def _score_components(
        self,
        row: sqlite3.Row,
        *,
        lexical: float = 0.0,
        lexical_rank: float | None = None,
        vector: float = 0.0,
        identity: float = 0.0,
        identity_rank: float | None = None,
    ) -> dict[str, float]:
        rank_score = 0.0 if lexical_rank is None else 1.0 / (1.0 + abs(float(lexical_rank)))
        identity_score = identity if identity_rank is None else identity_rank
        lexical_score = self._store._bounded(lexical)
        vector_score = self._store._bounded(vector)
        resolved_identity = self._store._bounded(identity_score)
        base_relevance = max(lexical_score, vector_score, resolved_identity)
        hotness = (
            self._store._bounded(row["hotness"])
            + self._store._bounded(row["semantic_hotness"])
            + self._store._bounded(row["behavior_support_hotness"])
        ) / 3.0
        return {
            "identity": resolved_identity,
            "lexical": lexical_score,
            "lexical_rank": self._store._bounded(rank_score),
            "vector": vector_score,
            "base_relevance": base_relevance,
            "hotness": hotness,
            "score": base_relevance + (0.05 * hotness if base_relevance > 0 else 0.0),
        }

    def explain_structured_query(
        self,
        *,
        tenant_id: str,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
    ) -> list[str]:
        normalized = self._store._catalog._tenant_filters(filters, str(tenant_id))
        predicate, params = self._store._base_filter_sql(normalized)
        sql = f"EXPLAIN QUERY PLAN SELECT c.record_key FROM contexts AS c WHERE 1=1 {predicate} LIMIT ?"
        with self._store._connect() as conn:
            rows = conn.execute(sql, [*params, self._store._bounded_limit(limit)]).fetchall()
        return [str(row[3]) for row in rows]


__all__ = ["CatalogSearchOperations"]
