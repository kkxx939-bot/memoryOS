"""SQLite catalog CatalogSearchOperations responsibility component."""

from __future__ import annotations

from memoryos.adapters.persistence.sqlite._common import (
    _BOUNDED_FTS_OVERFETCH,
    _FTS_RANK_CONFIG,
    _MAX_FILTER_VALUES,
    _MAX_QUERY_LIMIT,
    Any,
    CatalogCandidateBoundExceeded,
    CatalogRecord,
    CatalogRecordKind,
    IndexHit,
    Mapping,
    sqlite3,
)


class CatalogSearchOperations:
    """Own one stable subset of SQLite catalog behavior."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
        """Legacy online search; structured filters are applied before every LIMIT."""

        hits = self._store.search_catalog(query, filters=filters, limit=limit)
        deduplicated: dict[str, IndexHit] = {}
        for hit in hits:
            deduplicated.setdefault(hit.uri, hit)
        return list(deduplicated.values())[: self._store._bounded_limit(limit)]

    def list_legacy_catalog(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 100,
    ) -> list[CatalogRecord]:
        """Bounded rollback reader over the pre-unification flat Catalog shape.

        This deliberately does not use ACL-grant, closure, relation, vector,
        or validity adjuncts.  It is a conservative owner/public compatibility
        route over the same evolved ``contexts`` table, not a second Catalog.
        """

        normalized = dict(filters or {})
        bounded = self._store._bounded_limit(limit)
        filter_sql, params = self._store._legacy_filter_sql(normalized)
        sql = (
            "SELECT c.* FROM contexts AS c WHERE 1=1 "
            + filter_sql
            + " ORDER BY c.updated_at DESC, c.record_key DESC LIMIT ?"
        )
        with self._store._connect() as conn:
            rows = self._store._online_fetchall(conn, sql, [*params, bounded])
            return self._store._catalog_records_from_rows(conn, rows)

    def search_legacy_catalog(
        self,
        query: str,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
    ) -> list[IndexHit]:
        """Run the independently bounded legacy lexical read for shadow/rollback."""

        normalized = dict(filters or {})
        bounded = self._store._bounded_limit(limit)
        value = str(query).strip()
        filter_sql, params = self._store._legacy_filter_sql(normalized)
        exact_rows: list[sqlite3.Row] = []
        with self._store._connect() as conn:
            if value:
                exact_rows = self._store._online_fetchall(
                    conn,
                    "SELECT c.*, c.title AS fts_title, c.content_text AS fts_content, "
                    "'' AS fts_metadata FROM contexts AS c WHERE "
                    "(c.scene_key = ? OR c.action = ? OR c.memory_anchor_uri = ?) "
                    + filter_sql
                    + " ORDER BY c.updated_at DESC, c.record_key DESC LIMIT ?",
                    [value, value, value, *params, bounded],
                )
        merged: dict[str, IndexHit] = {
            str(row["record_key"]): self._store._hit_from_row(row, identity=1.0, identity_rank=10.0)
            for row in exact_rows
        }
        if not value or len(merged) >= bounded or not self._store.fts_enabled:
            return list(merged.values())[:bounded]
        match_query = self._store._match_query(value)
        if not match_query:
            return list(merged.values())[:bounded]
        overfetch = min(max(bounded * 4, bounded), _BOUNDED_FTS_OVERFETCH)
        sql = f"""
            SELECT c.*, contexts_fts.title AS fts_title,
                   contexts_fts.content_text AS fts_content,
                   contexts_fts.metadata_text AS fts_metadata,
                   contexts_fts.rank AS rank
            FROM contexts_fts
            CROSS JOIN contexts AS c INDEXED BY idx_contexts_record_key
              ON c.record_key = contexts_fts.record_key
            WHERE contexts_fts MATCH ? AND contexts_fts.rank MATCH ? {filter_sql}
            ORDER BY contexts_fts.rank
            LIMIT ?
        """
        with self._store._connect() as conn:
            rows = self._store._online_fetchall(
                conn,
                sql,
                [match_query, _FTS_RANK_CONFIG, *params, overfetch],
            )
        rows.sort(key=lambda row: str(row["record_key"]), reverse=True)
        rows.sort(key=lambda row: str(row["updated_at"]), reverse=True)
        rows.sort(key=lambda row: float(row["rank"]))
        for row in rows:
            haystack = " ".join((str(row["fts_title"]), str(row["fts_content"]), str(row["fts_metadata"])))
            lexical = self._store._lexical_relevance(value, haystack)
            if lexical <= 0:
                continue
            merged.setdefault(
                str(row["record_key"]),
                self._store._hit_from_row(
                    row,
                    lexical=lexical,
                    lexical_rank=self._store._lexical_match_count(value, haystack),
                ),
            )
            if len(merged) >= bounded:
                break
        return list(merged.values())[:bounded]

    def search_catalog(
        self,
        query: str,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
    ) -> list[IndexHit]:
        """Return record-key-distinct exact/FTS candidates without Python row scans."""

        normalized_filters = dict(filters or {})
        bounded_limit = self._store._bounded_limit(limit)
        exact_hits = self._store._search_metadata_exact(query, normalized_filters, bounded_limit)
        if len(exact_hits) >= bounded_limit:
            return exact_hits[:bounded_limit]
        hits: list[IndexHit] = []
        if str(query).strip():
            hits = self._store._search_fts(query, normalized_filters, bounded_limit) if self._store.fts_enabled else []
        merged: dict[str, IndexHit] = {}
        for hit in (*exact_hits, *hits):
            key = str(hit.metadata.get("catalog_record_key") or hit.uri)
            merged.setdefault(key, hit)
        return list(merged.values())[:bounded_limit]

    def explain_structured_query(self, filters: Mapping[str, Any], *, limit: int = 10) -> list[str]:
        """Expose SQLite's query plan for integration/performance acceptance tests."""

        normalized_filters = dict(filters)
        bounded_limit = self._store._bounded_limit(limit)
        filter_sql, params = self._store._base_filter_sql(
            normalized_filters,
            path_candidate_limit=bounded_limit,
        )
        from_sql, index_predicate = self._store._catalog_from_sql(normalized_filters)
        sql = f"EXPLAIN QUERY PLAN SELECT c.record_key FROM {from_sql} WHERE 1=1 {filter_sql}{index_predicate} LIMIT ?"
        with self._store._connect() as conn:
            rows = conn.execute(sql, [*params, bounded_limit]).fetchall()
        return [str(row["detail"]) for row in rows]

    def _catalog_from_sql(self, filters: Mapping[str, Any]) -> tuple[str, str]:
        """Use the Current Slot unique key for an exact serving lookup."""

        if filters.get("target_identity_uris") is not None:
            # The identity UNION below is already bounded to exact indexed
            # keys. Drive the outer ACL/type read from those record keys.
            return "contexts AS c INDEXED BY idx_contexts_record_key", ""
        if (
            filters.get("principal_owner_id") is not None
            or (filters.get("owner_user_id") == "" and filters.get("require_unscoped"))
            or filters.get("target_paths") is not None
            or filters.get("path_prefixes") is not None
        ):
            # Path candidates are already bounded inside the normalized path
            # subquery.  Drive the outer read by their record keys; otherwise
            # SQLite may prefer a tenant/time ORDER BY index and scan every
            # tenant row before testing membership in the bounded set.
            return "contexts AS c INDEXED BY idx_contexts_record_key", ""
        raw_slots = filters.get("canonical_slot_ids", filters.get("canonical_slot_id"))
        raw_kinds = filters.get("record_kinds", filters.get("record_kind"))
        slots = self._store._filter_values(raw_slots, allow_empty=True) if raw_slots is not None else ()
        kinds = self._store._filter_values(raw_kinds, allow_empty=True) if raw_kinds is not None else ()
        if len(slots) == 1 and kinds == [CatalogRecordKind.CURRENT_SLOT.value]:
            return (
                "contexts AS c INDEXED BY uq_contexts_current_slot",
                " AND c.record_kind = 'current_slot' AND c.canonical_slot_id != '' "
                "AND c.lifecycle_state NOT IN ('deleted', 'obsolete')",
            )
        return "contexts AS c", ""

    def _search_fts(self, query: str, filters: dict[str, Any], limit: int) -> list[IndexHit]:
        narrowed_filters = self._store._narrow_online_validity_filters(filters)
        if narrowed_filters is None:
            return []
        filters = narrowed_filters
        match_query = self._store._match_query(query)
        if not match_query:
            return []
        match_query = self._store._acl_bound_fts_query(match_query, filters)
        overfetch = min(max(limit * 4, limit), _BOUNDED_FTS_OVERFETCH)
        fts_filters = {**filters, "_fts_bound_candidates": True}
        filter_sql, params = self._store._base_filter_sql(
            fts_filters,
            path_candidate_limit=overfetch,
        )
        sql = f"""
            SELECT c.*, contexts_fts.title AS fts_title,
                   contexts_fts.content_text AS fts_content,
                   contexts_fts.metadata_text AS fts_metadata,
                   contexts_fts.rank AS rank
            FROM contexts_fts
            CROSS JOIN contexts c INDEXED BY idx_contexts_record_key
              ON contexts_fts.record_key = c.record_key
            WHERE contexts_fts MATCH ? AND contexts_fts.rank MATCH ? {filter_sql}
            ORDER BY contexts_fts.rank
            LIMIT ?
        """
        with self._store._connect() as conn:
            # Runtime FTS/schema/storage failures are not equivalent to a
            # successful query with zero matches.  Capability selection is
            # decided when the store is initialized; an operational failure
            # after that boundary must remain observable to the public API.
            rows = self._store._online_fetchall(
                conn,
                sql,
                [match_query, _FTS_RANK_CONFIG, *params, overfetch],
            )
        rows.sort(key=lambda row: str(row["record_key"]), reverse=True)
        rows.sort(key=lambda row: str(row["updated_at"]), reverse=True)
        rows.sort(key=lambda row: float(row["rank"]))
        hits: list[IndexHit] = []
        for row in rows:
            haystack = " ".join((str(row["fts_title"]), str(row["fts_content"]), str(row["fts_metadata"])))
            lexical = self._store._lexical_relevance(query, haystack)
            if lexical <= 0:
                continue
            hits.append(
                self._store._hit_from_row(
                    row,
                    lexical=lexical,
                    lexical_rank=self._store._lexical_match_count(query, haystack),
                )
            )
        # Preserve SQLite FTS5's deterministic BM25 order.  ``hit.score`` is
        # a normalized term-coverage component used later by Fusion; sorting
        # by it here collapses many lexical matches to the same value and
        # silently replaces BM25 relevance with URI order.
        return hits[:limit]

    def _search_metadata_exact(self, query: str, filters: dict[str, Any], limit: int) -> list[IndexHit]:
        narrowed_filters = self._store._narrow_online_validity_filters(filters)
        if narrowed_filters is None:
            return []
        filters = narrowed_filters
        value = str(query).strip()
        if not value:
            return []
        tenants = (
            self._store._filter_values(filters["tenant_id"], allow_empty=True)
            if filters.get("tenant_id") is not None
            else ["default"]
        )
        if not tenants:
            raise ValueError("exact Catalog lookup requires tenant_id")
        filters = {**filters, "tenant_id": tuple(tenants)}
        candidate_updates: dict[str, str] = {}
        branch_limit = self._store._bounded_limit(limit)
        # Exact identities are expected to be selective. Detect an excessive
        # *eligible* identity set explicitly instead of silently truncating
        # it, but only after every trusted ACL/path/time/type predicate has
        # been applied inside the branch.
        identity_limit = _MAX_QUERY_LIMIT + 1
        with self._store._connect() as conn:
            for column, index_name in (
                ("scene_key", "idx_contexts_tenant_scene_key"),
                ("action", "idx_contexts_tenant_action"),
                ("memory_anchor_uri", "idx_contexts_tenant_anchor"),
            ):
                branch_filters = {
                    **filters,
                    column: value,
                    # The exact equality index is already the selective
                    # candidate driver. Keep the direct ACL EXISTS predicate,
                    # but do not put a broader ACL Top-K in front of it.
                    "_exact_bound_candidates": True,
                }
                filter_sql, params = self._store._base_filter_sql(
                    branch_filters,
                    path_candidate_limit=min(
                        _MAX_QUERY_LIMIT,
                        max(branch_limit, identity_limit),
                    ),
                )
                rows = self._store._online_fetchall(
                    conn,
                    "SELECT c.record_key, c.updated_at FROM contexts AS c INDEXED BY "
                    + index_name
                    + " WHERE 1=1 "
                    + filter_sql
                    + " ORDER BY c.updated_at DESC, c.record_key LIMIT ?",
                    [*params, identity_limit],
                )
                if len(rows) >= identity_limit:
                    raise CatalogCandidateBoundExceeded(
                        "eligible exact identity candidates exceed the bounded online lookup"
                    )
                for row in rows:
                    key = str(row["record_key"])
                    candidate_updates[key] = max(
                        candidate_updates.get(key, ""),
                        str(row["updated_at"] or ""),
                    )
        if len(candidate_updates) > _MAX_FILTER_VALUES:
            raise CatalogCandidateBoundExceeded("eligible exact identity candidates exceed the aggregate filter bound")
        ordered_keys = sorted(candidate_updates)
        ordered_keys.sort(key=lambda key: candidate_updates[key], reverse=True)
        bounded_keys = tuple(ordered_keys[:branch_limit])
        if not bounded_keys:
            return []
        raw_records = self._store.list_catalog(
            filters={**filters, "record_keys": bounded_keys},
            limit=branch_limit,
        )
        return [self._store._exact_hit_from_record(record) for record in raw_records]

    @staticmethod
    def _exact_hit_from_record(record: CatalogRecord) -> IndexHit:
        metadata = {
            **dict(record.metadata),
            "catalog_record_key": record.record_key,
            "tenant_id": record.tenant_id,
            "owner_user_id": record.owner_user_id,
            "workspace_id": record.workspace_id,
            "workspace_shared": record.workspace_shared,
            "session_id": record.session_id,
            "adapter_id": record.adapter_id,
            "context_type": record.context_type,
            "source_kind": record.source_kind,
            "record_kind": record.record_kind,
            "lifecycle_state": record.lifecycle_state,
            "canonical_slot_id": record.canonical_slot_id,
            "canonical_slot_uri": record.canonical_slot_uri,
            "canonical_claim_id": record.canonical_claim_id,
            "canonical_claim_uri": record.canonical_claim_uri,
            "canonical_revision": record.canonical_revision,
            "canonical_state": record.canonical_state,
            "source_digest": record.source_digest,
            "event_time": record.event_time,
            "transaction_time": record.transaction_time,
            "serving_tier": record.serving_tier,
            "retrieval_scores": {
                "lexical": 0.0,
                "vector": 0.0,
                "identity": 1.0,
                "base_relevance": 1.0,
                "hotness": max(
                    record.hotness,
                    record.semantic_hotness,
                    record.behavior_support_hotness,
                ),
                "score": 10.0,
            },
        }
        return IndexHit(
            uri=record.uri,
            score=10.0,
            context_type=record.context_type,
            title=record.title,
            metadata=metadata,
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
        metadata = self._store._json_mapping(row["metadata_json"])
        # CandidateGenerator deliberately re-checks transport-independent
        # structured filters for stores that only implement the legacy search
        # protocol.  Carry the already-sanitized indexed columns in every hit
        # so that this defence-in-depth check never has to infer them from
        # optional metadata JSON (and never rejects a valid SQL-filtered row).
        metadata.update(
            {
                "catalog_record_key": str(row["record_key"]),
                "tenant_id": str(row["tenant_id"]),
                "owner_user_id": str(row["owner_user_id"]),
                "workspace_id": str(row["workspace_id"]),
                "workspace_shared": bool(row["workspace_shared"]),
                "session_id": str(row["session_id"]),
                "adapter_id": str(row["adapter_id"]),
                "context_type": str(row["context_type"]),
                "source_kind": str(row["source_kind"]),
                "record_kind": str(row["record_kind"]),
                "lifecycle_state": str(row["lifecycle_state"]),
                "canonical_slot_id": str(row["canonical_slot_id"]),
                "canonical_slot_uri": str(row["canonical_slot_uri"]),
                "canonical_claim_id": str(row["canonical_claim_id"]),
                "canonical_claim_uri": str(row["canonical_claim_uri"]),
                "canonical_revision": int(row["canonical_revision"]),
                "canonical_state": str(row["canonical_state"]),
                "source_digest": str(row["source_digest"]),
                "event_time": str(row["event_time"]),
                "transaction_time": str(row["transaction_time"]),
                "serving_tier": str(row["serving_tier"]),
            }
        )
        metadata["retrieval_scores"] = self._store._score_components(
            row,
            lexical=lexical,
            lexical_rank=lexical_rank,
            vector=vector,
            identity=identity,
            identity_rank=identity_rank,
        )
        return IndexHit(
            uri=str(row["uri"]),
            score=float(metadata["retrieval_scores"]["score"]),
            context_type=str(row["context_type"]),
            title=str(row["title"]),
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
        lexical = self._store._bounded(lexical)
        vector = self._store._bounded(vector)
        identity = self._store._bounded(identity)
        base_relevance = max(lexical, vector, identity)
        hotness = (
            self._store._bounded(row["hotness"])
            + self._store._bounded(row["semantic_hotness"])
            + self._store._bounded(row["behavior_support_hotness"])
        ) / 3.0
        ranking_relevance = max(
            self._store._finite_rank(lexical_rank if lexical_rank is not None else lexical),
            vector,
            self._store._finite_rank(identity_rank if identity_rank is not None else identity),
        )
        score = ranking_relevance + (0.05 * hotness if base_relevance > 0 else 0.0)
        return {
            "lexical": lexical,
            "vector": vector,
            "identity": identity,
            "base_relevance": base_relevance,
            "hotness": hotness,
            "score": score,
        }


__all__ = ["CatalogSearchOperations"]
