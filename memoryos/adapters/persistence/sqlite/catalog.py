"""SQLite catalog CatalogStoreOperations responsibility component."""

from __future__ import annotations

from memoryos.adapters.persistence.sqlite._common import (
    _CONTEXT_COLUMNS,
    _MAX_QUERY_LIMIT,
    _MAX_SCOPE_KEYS_PER_RECORD,
    Any,
    CatalogProjectionStatus,
    CatalogRecord,
    CatalogRecordKind,
    ContextObject,
    Mapping,
    Sequence,
    ServingTier,
    _path_ancestors,
    _PreparedCatalogRecord,
    lexical_terms,
    normalize_workspace_id,
    replace,
    sqlite3,
)


class CatalogStoreOperations:
    """Own one stable subset of SQLite catalog behavior."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def upsert_index(self, obj: ContextObject, content: str = "") -> None:
        """Project a legacy ContextObject through the same sanitized catalog writer."""

        record = CatalogRecord.from_context_object(obj, content=content)
        # Legacy callers pass the searchable L2 projection explicitly.  Keep it
        # searchable even when metadata also carries a shorter L1 summary.
        if content:
            record = replace(record, l1_text=content)
        self._store.upsert_catalog(record)

    def delete_index(self, uri: str) -> None:
        """Delete every serving record for a legacy URI without touching SourceStore."""

        with self._store._connect() as conn:
            rows = conn.execute(
                "SELECT tenant_id, record_key FROM contexts WHERE uri = ?",
                (str(uri),),
            ).fetchall()
            for row in rows:
                self._store._delete_catalog_in_transaction(
                    conn,
                    str(row["record_key"]),
                    tenant_id=str(row["tenant_id"]),
                )

    def indexed_uris(self) -> list[str]:
        with self._store._connect() as conn:
            rows = conn.execute("SELECT DISTINCT uri FROM contexts ORDER BY uri").fetchall()
        return [str(row["uri"]) for row in rows]

    def get_index_metadata(self, uri: str) -> dict[str, Any] | None:
        """Return the legacy record first, then a deterministic projection for the URI."""

        with self._store._connect() as conn:
            row = conn.execute(
                "SELECT * FROM contexts WHERE uri = ? "
                "ORDER BY CASE WHEN record_key = uri THEN 0 WHEN record_kind = 'current_slot' THEN 1 ELSE 2 END, "
                "updated_at DESC, record_key LIMIT 1",
                (str(uri),),
            ).fetchone()
        if row is None:
            return None
        metadata = self._store._json_mapping(row["metadata_json"])
        self._store._restore_internal_projection_path(metadata)
        return {
            **metadata,
            "record_key": str(row["record_key"]),
            "tenant_id": str(row["tenant_id"]),
            "owner_user_id": str(row["owner_user_id"]),
            "context_type": str(row["context_type"]),
            "claim_state": str(row["claim_state"]),
            "slot_id": str(row["slot_id"]),
            "memory_type": str(row["memory_type"]),
            "index_content_digest": (
                str(row["content_digest"])
                if self._store._content_digest(str(row["content_text"])) == str(row["stored_content_digest"])
                else self._store._content_digest(str(row["content_text"]))
            ),
        }

    def ordinary_relation_endpoint_state(
        self,
        uri: str,
        *,
        tenant_id: str,
        session_id: str = "",
    ) -> str:
        """Resolve relation endpoint liveness through durable delete barriers."""

        safe_uri = self._store._safe_reference_uri(str(uri))
        safe_session_id = str(session_id or "")
        with self._store._connect() as conn:
            blocked = conn.execute(
                "SELECT 1 FROM context_tombstones WHERE tenant_id = ? "
                "AND ((status IN ('PENDING', 'FAILED', 'CLEANING') AND uri = ?) OR ("
                "? <> '' AND status IN ('PENDING', 'FAILED', 'CLEANING', 'APPLIED') "
                "AND json_extract(payload_json, '$.record_kind') = 'session_delete_barrier' "
                "AND json_extract(payload_json, '$.session_id') = ?)) LIMIT 1",
                (str(tenant_id), safe_uri, safe_session_id, safe_session_id),
            ).fetchone()
            if blocked is not None:
                return "retired"
            row = conn.execute(
                "SELECT lifecycle_state FROM contexts WHERE tenant_id = ? "
                "AND (uri = ? OR source_uri = ? OR (? <> '' AND session_id = ?)) "
                "ORDER BY CASE WHEN lifecycle_state = 'active' THEN 0 ELSE 1 END, updated_at DESC LIMIT 1",
                (str(tenant_id), safe_uri, safe_uri, safe_session_id, safe_session_id),
            ).fetchone()
            if row is not None and str(row["lifecycle_state"]) == "active":
                return "active"
            retired = conn.execute(
                "SELECT 1 FROM context_tombstones WHERE tenant_id = ? AND uri = ? AND status = 'APPLIED' LIMIT 1",
                (str(tenant_id), safe_uri),
            ).fetchone()
            if retired is not None:
                return "retired"
        if row is None:
            return "missing"
        return "inactive"

    def clear(self) -> None:
        """Clear rebuildable serving data while retaining migration and tombstone journals."""

        with self._store._connect() as conn:
            conn.execute("DELETE FROM context_links")
            conn.execute("DELETE FROM context_acl_grants")
            conn.execute("DELETE FROM context_path_acl")
            conn.execute("DELETE FROM context_path_closure")
            conn.execute("DELETE FROM context_paths")
            conn.execute("DELETE FROM context_validity_rtree")
            conn.execute("DELETE FROM context_validity_map")
            conn.execute("DELETE FROM context_projection_state")
            conn.execute("DELETE FROM contexts")
            conn.execute("DELETE FROM contexts_fts")
            conn.execute("DELETE FROM context_fts_map")

    def begin_tenant_serving_rebuild(
        self,
        migration_name: str,
        *,
        tenant_id: str,
        batch_size: int,
        details: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Atomically gate and clear one tenant's rebuildable serving rows.

        The durable gate and destructive Catalog mutation share one SQLite
        transaction.  A process crash therefore observes either the old
        serving snapshot or an explicit BACKFILLING row that startup can
        resume; it can never observe a cleared Catalog with a COMPLETED gate.
        Tombstones, migration journals, Session frontiers and immutable Source
        evidence are intentionally retained.
        """

        if not migration_name or not tenant_id:
            raise ValueError("tenant serving rebuild requires migration_name and tenant_id")
        bounded_batch_size = int(batch_size)
        if not 1 <= bounded_batch_size <= _MAX_QUERY_LIMIT:
            raise ValueError("tenant serving rebuild batch_size must be between 1 and 1000")
        safe_details = self._store.sanitizer.sanitize_trace(dict(details))
        if not isinstance(safe_details, Mapping) or not str(safe_details.get("rebuild_epoch") or ""):
            raise ValueError("tenant serving rebuild requires a sanitized rebuild_epoch")
        now = self._store._now()
        with self._store._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM migration_state WHERE migration_name = ? AND tenant_id = ?",
                (str(migration_name), str(tenant_id)),
            ).fetchone()
            if existing is not None and str(existing["state"]) in {
                "BACKFILLING",
                "FAILED",
                "ROLLBACK",
            }:
                return self._store._row_dict(existing, json_fields=("details_json",))

            record_keys = tuple(
                str(row["record_key"])
                for row in conn.execute(
                    "SELECT record_key FROM contexts WHERE tenant_id = ? ORDER BY record_key",
                    (str(tenant_id),),
                ).fetchall()
            )
            prepared_details = {
                **dict(safe_details),
                "catalog_cleared": True,
                "cleared_records": len(record_keys),
                "phase": "VECTOR_CLEANUP",
            }
            conn.execute(
                """
                INSERT INTO migration_state(
                  migration_name, tenant_id, state, checkpoint, batch_size,
                  details_json, last_error, updated_at
                ) VALUES (?, ?, 'BACKFILLING', '', ?, ?, '', ?)
                ON CONFLICT(migration_name, tenant_id) DO UPDATE SET
                  state='BACKFILLING', checkpoint='', batch_size=excluded.batch_size,
                  details_json=excluded.details_json, last_error='', updated_at=excluded.updated_at
                """,
                (
                    str(migration_name),
                    str(tenant_id),
                    bounded_batch_size,
                    self._store._json_dump(prepared_details),
                    now,
                ),
            )
            for record_key in record_keys:
                self._store._delete_catalog_in_transaction(
                    conn,
                    record_key,
                    tenant_id=str(tenant_id),
                )
            persisted = conn.execute(
                "SELECT * FROM migration_state WHERE migration_name = ? AND tenant_id = ?",
                (str(migration_name), str(tenant_id)),
            ).fetchone()
        if persisted is None:  # pragma: no cover - the transaction either commits or raises.
            raise RuntimeError("tenant serving rebuild gate did not persist")
        return self._store._row_dict(persisted, json_fields=("details_json",))

    def rebuildable_catalog_records(
        self,
        records: Sequence[CatalogRecord],
    ) -> tuple[CatalogRecord, ...]:
        """Filter an offline rebuild batch through durable delete ownership.

        APPLIED tombstones suppress the same or an older Source revision.
        CLEANING remains a hard retry boundary because its Vector/Relation
        consumers have not reached a durable terminal state.
        """

        selected: list[CatalogRecord] = []
        with self._store._connect() as conn:
            for record in records:
                rows = conn.execute(
                    "SELECT source_revision, status FROM context_tombstones "
                    "WHERE tenant_id = ? AND (record_key = ? OR ("
                    "json_extract(payload_json, '$.record_kind') = 'session_delete_barrier' "
                    "AND json_extract(payload_json, '$.session_id') = ?)) "
                    "AND status IN ('CLEANING', 'APPLIED') ORDER BY created_at",
                    (record.tenant_id, record.record_key, record.session_id),
                ).fetchall()
                blocked = False
                for row in rows:
                    status = str(row["status"])
                    if status == "CLEANING":
                        raise RuntimeError("Catalog rebuild is blocked by in-progress tombstone cleanup")
                    tombstone_revision = int(row["source_revision"])
                    if (
                        tombstone_revision == 0
                        or record.source_revision == 0
                        or tombstone_revision >= record.source_revision
                    ):
                        blocked = True
                        break
                if not blocked:
                    selected.append(record)
        return tuple(selected)

    def upsert_catalog(self, record: CatalogRecord | Mapping[str, Any]) -> None:
        """Atomically sanitize and upsert one rebuildable catalog record."""

        self._store.upsert_catalog_batch((record,))

    def upsert_catalog_batch(self, records: Sequence[CatalogRecord | Mapping[str, Any]]) -> int:
        """Atomically project a batch; any validation, sanitization, or write error rolls it back."""

        prepared = tuple(self._store._prepare_record(self._store._coerce_record(record)) for record in records)
        if not prepared:
            return 0
        with self._store._connect() as conn:
            for item in prepared:
                self._store._upsert_prepared(conn, item)
        return len(prepared)

    def get_catalog(self, record_key: str, *, tenant_id: str | None = None) -> CatalogRecord | None:
        sql = "SELECT * FROM contexts WHERE record_key = ?"
        params: list[Any] = [str(record_key)]
        if tenant_id is not None:
            sql += " AND tenant_id = ?"
            params.append(str(tenant_id))
        with self._store._connect() as conn:
            row = conn.execute(sql, params).fetchone()
            if row is None:
                return None
            return self._store._catalog_record_from_row(conn, row)

    def get_catalog_by_uri(
        self,
        uri: str,
        *,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[CatalogRecord]:
        filters: dict[str, Any] = {"target_uris": (str(uri),)}
        filters["include_inactive"] = True
        if tenant_id is not None:
            filters["tenant_id"] = str(tenant_id)
        return self._store.list_catalog(filters=filters, limit=limit)

    def list_catalog(self, *, filters: Mapping[str, Any] | None = None, limit: int = 100) -> list[CatalogRecord]:
        narrowed_filters = self._store._narrow_online_validity_filters(dict(filters or {}))
        if narrowed_filters is None:
            return []
        normalized_filters = narrowed_filters
        bounded_limit = self._store._bounded_limit(limit)
        filter_sql, params = self._store._base_filter_sql(
            normalized_filters,
            path_candidate_limit=min(_MAX_QUERY_LIMIT, max(bounded_limit, bounded_limit * 4)),
        )
        from_sql, index_predicate = self._store._catalog_from_sql(normalized_filters)
        sql = (
            f"SELECT c.* FROM {from_sql} WHERE 1=1 {filter_sql}{index_predicate} "
            "ORDER BY c.updated_at DESC, c.record_key LIMIT ?"
        )
        with self._store._connect() as conn:
            rows = self._store._online_fetchall(conn, sql, [*params, bounded_limit])
            return self._store._catalog_records_from_rows(conn, rows)

    def list_catalog_projection_records(
        self,
        *,
        tenant_id: str,
        source_uri: str,
        projection_effect_hash: str,
        limit: int = 1_001,
    ) -> list[CatalogRecord]:
        """Read one evidence-bound projection set for offline/shadow proof.

        This exact identity lookup is intentionally separate from online
        search.  The extra row above the 1000-record proof bound lets callers
        fail closed instead of certifying a truncated projection.
        """

        bounded = int(limit)
        if not tenant_id or not source_uri or not projection_effect_hash:
            raise ValueError("projection evidence identity is required")
        if not 1 <= bounded <= 1_001:
            raise ValueError("projection proof lookup limit must be between 1 and 1001")
        with self._store._connect() as conn:
            rows = conn.execute(
                "SELECT c.* FROM contexts c WHERE c.tenant_id = ? AND c.source_uri = ? "
                "AND c.projection_effect_hash = ? ORDER BY c.record_key LIMIT ?",
                (str(tenant_id), str(source_uri), str(projection_effect_hash), bounded),
            ).fetchall()
            return self._store._catalog_records_from_rows(conn, rows)

    def scan_catalog_batch(
        self,
        *,
        after_record_key: str = "",
        filters: Mapping[str, Any] | None = None,
        limit: int = 256,
    ) -> list[CatalogRecord]:
        """Return a stable keyset-paginated batch for offline repair and GC.

        Online retrieval uses ``search_catalog``.  This administrative API is
        deliberately keyset-paginated so retention and rebuild jobs never
        materialize the full catalog or become sensitive to rows whose
        ``updated_at`` changes while a batch is processed.
        """

        bounded_limit = self._store._bounded_limit(limit)
        filter_sql, params = self._store._base_filter_sql(
            dict(filters or {}),
            path_candidate_limit=bounded_limit,
        )
        sql = f"SELECT c.* FROM contexts c WHERE c.record_key > ? {filter_sql} ORDER BY c.record_key LIMIT ?"
        with self._store._connect() as conn:
            rows = conn.execute(
                sql,
                [str(after_record_key), *params, bounded_limit],
            ).fetchall()
            return self._store._catalog_records_from_rows(conn, rows)

    def catalog_schema_version(self) -> int:
        """Return the durable SQLite schema version used by migration gates."""

        with self._store._connect() as conn:
            return int(conn.execute("PRAGMA user_version").fetchone()[0])

    def gc_orphan_paths(self, *, limit: int = 256) -> int:
        """Delete a bounded batch of paths whose rebuildable record is gone."""

        with self._store._connect() as conn:
            rows = conn.execute(
                "SELECT p.tenant_id, p.record_key, p.path FROM context_paths p "
                "LEFT JOIN contexts c ON c.record_key = p.record_key "
                "WHERE c.record_key IS NULL ORDER BY p.record_key, p.path LIMIT ?",
                (self._store._bounded_limit(limit),),
            ).fetchall()
            for row in rows:
                conn.execute(
                    "DELETE FROM context_paths WHERE tenant_id = ? AND record_key = ? AND path = ?",
                    (str(row["tenant_id"]), str(row["record_key"]), str(row["path"])),
                )
                conn.execute(
                    "DELETE FROM context_path_closure WHERE tenant_id = ? AND record_key = ? AND path = ?",
                    (str(row["tenant_id"]), str(row["record_key"]), str(row["path"])),
                )
                conn.execute(
                    "DELETE FROM context_path_acl WHERE tenant_id = ? AND record_key = ? AND path = ?",
                    (str(row["tenant_id"]), str(row["record_key"]), str(row["path"])),
                )
        return len(rows)

    def gc_applied_tombstones(self, *, updated_before: str, limit: int = 256) -> int:
        """Expire only old tombstones proven safe to forget.

        Stale tombstones did not delete the newer projection and are safe once
        aged out.  Applied tombstones remain durable by default; a projection
        owner must explicitly persist ``payload.gc_safe=true`` after proving
        that replay cannot resurrect the deleted source revision.
        """

        cutoff = self._store._coerce_timestamp(str(updated_before))
        if not cutoff:
            raise ValueError("updated_before must be an ISO-8601 timestamp")
        with self._store._connect() as conn:
            rows = conn.execute(
                "SELECT tombstone_id FROM context_tombstones "
                "WHERE updated_at < ? AND (status = 'STALE' OR "
                "(status = 'APPLIED' AND json_extract(payload_json, '$.gc_safe') = 1)) "
                "ORDER BY updated_at, tombstone_id LIMIT ?",
                (cutoff, self._store._bounded_limit(limit)),
            ).fetchall()
            for row in rows:
                conn.execute(
                    "DELETE FROM context_tombstones WHERE tombstone_id = ?",
                    (str(row["tombstone_id"]),),
                )
        return len(rows)

    def delete_catalog(self, record_key: str, *, tenant_id: str | None = None) -> bool:
        with self._store._connect() as conn:
            if tenant_id is not None:
                exists = conn.execute(
                    "SELECT 1 FROM contexts WHERE record_key = ? AND tenant_id = ?",
                    (str(record_key), str(tenant_id)),
                ).fetchone()
                if exists is None:
                    return False
            return self._store._delete_catalog_in_transaction(
                conn,
                str(record_key),
                tenant_id=tenant_id,
            )

    def _prepare_record(
        self,
        record: CatalogRecord,
        *,
        scope_keys_override: Sequence[str] | None = None,
        legacy_overrides: Mapping[str, Any] | None = None,
    ) -> _PreparedCatalogRecord:
        if record.projection_status == CatalogProjectionStatus.TOMBSTONED.value:
            raise ValueError("use apply_tombstone() for tombstoned catalog records")
        scope_keys = (
            list(scope_keys_override)
            if scope_keys_override is not None
            else self._store._scope_keys_from_metadata(record.metadata)
        )
        if len(scope_keys) > _MAX_SCOPE_KEYS_PER_RECORD:
            raise ValueError(f"Catalog scope requirements cannot exceed {_MAX_SCOPE_KEYS_PER_RECORD} keys")
        scope_signature = self._store._scope_signature(scope_keys)
        safe = record.with_sanitized_projection(self._store.sanitizer)
        safe = replace(
            safe,
            l2_uri=self._store._safe_reference_uri(safe.l2_uri),
            source_uri=self._store._safe_reference_uri(safe.source_uri),
        )
        metadata = dict(safe.metadata)
        scope = self._store._mapping(metadata.get("scope"))
        fields = self._store._mapping(metadata.get("fields"))
        connect = self._store._mapping(metadata.get("connect"))
        admission = self._store._mapping(metadata.get("admission"))
        project_id = normalize_workspace_id(
            scope.get("project_id") or fields.get("project_id") or metadata.get("project_id") or safe.workspace_id or ""
        )
        values: dict[str, Any] = {
            "record_key": safe.record_key,
            "uri": safe.uri,
            "tenant_id": safe.tenant_id,
            "owner_user_id": safe.owner_user_id,
            "project_id": project_id,
            "workspace_id": normalize_workspace_id(safe.workspace_id or project_id),
            "workspace_shared": 1 if safe.workspace_shared else 0,
            "session_id": safe.session_id,
            "adapter_id": safe.adapter_id or str(connect.get("adapter_id") or ""),
            "context_type": safe.context_type,
            "source_kind": safe.source_kind,
            "record_kind": safe.record_kind,
            "lifecycle_state": safe.lifecycle_state,
            "admission_status": str(admission.get("decision") or ""),
            "claim_state": safe.canonical_state or str(metadata.get("state") or metadata.get("claim_state") or ""),
            "slot_id": safe.canonical_slot_id or str(metadata.get("slot_id") or ""),
            "memory_type": str(metadata.get("memory_type") or ""),
            "scope_keys": self._store._json_dump(list(dict.fromkeys(str(key) for key in scope_keys))),
            "scope_signature": scope_signature,
            "parent_uri": safe.parent_uri,
            "primary_tree_path": safe.primary_tree_path,
            "path_depth": safe.path_depth,
            "created_at": safe.created_at,
            "updated_at": safe.updated_at,
            "event_time": safe.event_time,
            "ingested_at": safe.ingested_at,
            "transaction_time": safe.transaction_time,
            "valid_from": safe.valid_from,
            "valid_to": safe.valid_to,
            "title": safe.title,
            "l0_text": safe.l0_text,
            "l1_text": safe.l1_text,
            "l2_uri": safe.l2_uri,
            "source_uri": safe.source_uri,
            "source_digest": safe.source_digest,
            "source_revision": int(safe.source_revision),
            "canonical_slot_id": safe.canonical_slot_id,
            "canonical_slot_uri": safe.canonical_slot_uri,
            "canonical_claim_id": safe.canonical_claim_id,
            "canonical_claim_uri": safe.canonical_claim_uri,
            "canonical_revision": int(safe.canonical_revision),
            "canonical_state": safe.canonical_state,
            "canonical_head_digest": safe.canonical_head_digest,
            "receipt_digest": safe.receipt_digest,
            "projection_effect_hash": safe.projection_effect_hash,
            "hotness": safe.hotness,
            "semantic_hotness": safe.semantic_hotness,
            "behavior_support_hotness": safe.behavior_support_hotness,
            "serving_tier": safe.serving_tier,
            "projection_status": safe.projection_status,
            "metadata_json": self._store._json_dump(metadata),
            # The first digest proves the complete source projection without
            # retaining its potentially huge or sensitive body.  The second
            # detects tampering of the bounded sanitized serving text.
            "content_digest": self._store._content_digest(record.l1_text),
            "stored_content_digest": self._store._content_digest(safe.l1_text),
            "content_text": safe.l1_text,
            "scene_key": self._store._safe_exact_value(metadata.get("scene_key")),
            "action": self._store._safe_exact_value(metadata.get("action")),
            "memory_anchor_uri": self._store._safe_exact_value(metadata.get("memory_anchor_uri")),
        }
        for key, value in dict(legacy_overrides or {}).items():
            if key in values:
                values[key] = value
        metadata_text = self._store._safe_metadata_text(metadata)
        search_terms = " ".join(lexical_terms(" ".join((safe.title, safe.l0_text, safe.l1_text, metadata_text))))
        return _PreparedCatalogRecord(
            record=safe,
            values=values,
            scope_signature=scope_signature,
            fts_metadata_text=metadata_text,
            fts_search_terms=search_terms,
        )

    def _upsert_prepared(self, conn: sqlite3.Connection, item: _PreparedCatalogRecord) -> None:
        tombstones = conn.execute(
            "SELECT source_revision, status FROM context_tombstones "
            "WHERE tenant_id = ? AND record_key = ? AND status IN ('CLEANING', 'APPLIED')",
            (item.record.tenant_id, item.record.record_key),
        ).fetchall()
        for tombstone in tombstones:
            if str(tombstone["status"]) == "CLEANING":
                raise ValueError("catalog projection is blocked by in-progress tombstone cleanup")
            tombstone_revision = int(tombstone["source_revision"])
            if (
                tombstone_revision == 0
                or item.record.source_revision == 0
                or tombstone_revision >= item.record.source_revision
            ):
                raise ValueError("catalog projection is not newer than its applied tombstone")
        columns = ", ".join(_CONTEXT_COLUMNS)
        placeholders = ", ".join("?" for _ in _CONTEXT_COLUMNS)
        updates = ", ".join(f"{column}=excluded.{column}" for column in _CONTEXT_COLUMNS if column != "record_key")
        conn.execute(
            f"INSERT INTO contexts({columns}) VALUES ({placeholders}) ON CONFLICT(record_key) DO UPDATE SET {updates}",
            tuple(item.values[column] for column in _CONTEXT_COLUMNS),
        )
        self._store._replace_paths(conn, item.record, scope_signature=item.scope_signature)
        self._store._replace_acl_grants(conn, item.record, scope_signature=item.scope_signature)
        self._store._replace_validity(conn, item.record)
        self._store._replace_fts(conn, item)
        conn.execute(
            """
            INSERT INTO context_projection_state(
              tenant_id, record_key, source_revision, projection_status,
              projection_effect_hash, retry_count, last_error, updated_at
            ) VALUES (?, ?, ?, ?, ?, 0, '', ?)
            ON CONFLICT(tenant_id, record_key) DO UPDATE SET
              source_revision=excluded.source_revision,
              projection_status=excluded.projection_status,
              projection_effect_hash=excluded.projection_effect_hash,
              last_error='',
              updated_at=excluded.updated_at
            """,
            (
                item.record.tenant_id,
                item.record.record_key,
                item.record.source_revision,
                item.record.projection_status,
                item.record.projection_effect_hash,
                item.record.updated_at or self._store._now(),
            ),
        )

    def _replace_paths(
        self,
        conn: sqlite3.Connection,
        record: CatalogRecord,
        *,
        scope_signature: str,
    ) -> None:
        old_created = {
            str(row["path"]): str(row["created_at"])
            for row in conn.execute(
                "SELECT path, created_at FROM context_paths WHERE tenant_id = ? AND record_key = ?",
                (record.tenant_id, record.record_key),
            ).fetchall()
        }
        conn.execute(
            "DELETE FROM context_paths WHERE tenant_id = ? AND record_key = ?",
            (record.tenant_id, record.record_key),
        )
        conn.execute(
            "DELETE FROM context_path_closure WHERE tenant_id = ? AND record_key = ?",
            (record.tenant_id, record.record_key),
        )
        conn.execute(
            "DELETE FROM context_path_acl WHERE tenant_id = ? AND record_key = ?",
            (record.tenant_id, record.record_key),
        )
        now = (
            record.updated_at
            or record.transaction_time
            or record.ingested_at
            or record.created_at
            or self._store._now()
        )
        acl_grants = self._store._acl_grants_for_record(record)
        for index, path in enumerate(record.tree_paths):
            conn.execute(
                """
                INSERT INTO context_paths(
                  tenant_id, record_key, uri, owner_user_id, workspace_id, workspace_shared,
                  context_type, record_kind, canonical_slot_id, canonical_claim_id, event_time,
                  transaction_time, valid_from, valid_to,
                  path, path_kind, depth, is_primary, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.tenant_id,
                    record.record_key,
                    record.uri,
                    record.owner_user_id,
                    record.workspace_id,
                    1 if record.workspace_shared else 0,
                    record.context_type,
                    record.record_kind,
                    record.canonical_slot_id,
                    record.canonical_claim_id,
                    record.event_time,
                    record.transaction_time,
                    record.valid_from,
                    record.valid_to,
                    path,
                    "primary" if index == 0 else "secondary",
                    len(path.split("/")),
                    1 if index == 0 else 0,
                    old_created.get(path, now),
                    now,
                ),
            )
            for ancestor_path in _path_ancestors(path):
                conn.execute(
                    """
                    INSERT INTO context_path_closure(
                      tenant_id, record_key, path, ancestor_path,
                      owner_user_id, workspace_id, workspace_shared, scope_signature,
                      uri, context_type, source_kind, record_kind,
                      adapter_id, adapter_access_id, session_id,
                      canonical_slot_id, canonical_claim_id,
                      event_time, transaction_time, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.tenant_id,
                        record.record_key,
                        path,
                        ancestor_path,
                        record.owner_user_id,
                        record.workspace_id,
                        1 if record.workspace_shared else 0,
                        scope_signature,
                        record.uri,
                        record.context_type,
                        record.source_kind,
                        record.record_kind,
                        record.adapter_id,
                        self._store._adapter_access_value(record),
                        record.session_id,
                        record.canonical_slot_id,
                        record.canonical_claim_id,
                        record.event_time,
                        record.transaction_time,
                        now,
                    ),
                )
                for grant_kind, grant_id, grant_workspace_id in sorted(acl_grants):
                    conn.execute(
                        """
                        INSERT INTO context_path_acl(
                          tenant_id, record_key, path, ancestor_path,
                          grant_kind, grant_id, workspace_id, owner_user_id,
                          scope_signature, uri, context_type, source_kind, record_kind,
                          adapter_id, adapter_access_id, session_id,
                          event_time, transaction_time, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.tenant_id,
                            record.record_key,
                            path,
                            ancestor_path,
                            grant_kind,
                            grant_id,
                            grant_workspace_id,
                            record.owner_user_id,
                            scope_signature,
                            record.uri,
                            record.context_type,
                            record.source_kind,
                            record.record_kind,
                            record.adapter_id,
                            self._store._adapter_access_value(record),
                            record.session_id,
                            record.event_time,
                            record.transaction_time,
                            now,
                        ),
                    )

    def _replace_validity(self, conn: sqlite3.Connection, record: CatalogRecord) -> None:
        row = conn.execute(
            "SELECT validity_id FROM context_validity_map WHERE tenant_id = ? AND record_key = ?",
            (record.tenant_id, record.record_key),
        ).fetchone()
        if row is not None:
            conn.execute("DELETE FROM context_validity_rtree WHERE validity_id = ?", (int(row["validity_id"]),))
            conn.execute("DELETE FROM context_validity_map WHERE validity_id = ?", (int(row["validity_id"]),))
        cursor = conn.execute(
            "INSERT INTO context_validity_map(tenant_id, record_key) VALUES (?, ?)",
            (record.tenant_id, record.record_key),
        )
        raw_validity_id = cursor.lastrowid
        if raw_validity_id is None:
            raise RuntimeError("validity map insert did not return a durable rowid")
        validity_id = int(raw_validity_id)
        tenant_key = self._store._tenant_rtree_key(conn, record.tenant_id)
        valid_from = self._store._timestamp_number(record.valid_from, lower=True)
        valid_to = self._store._timestamp_number(record.valid_to, lower=False)
        conn.execute(
            "INSERT INTO context_validity_rtree("
            "validity_id, tenant_min, tenant_max, valid_from_min, valid_from_max, valid_to_min, valid_to_max"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (validity_id, tenant_key, tenant_key, valid_from, valid_from, valid_to, valid_to),
        )

    def _replace_acl_grants(
        self,
        conn: sqlite3.Connection,
        record: CatalogRecord,
        *,
        scope_signature: str,
    ) -> None:
        conn.execute(
            "DELETE FROM context_acl_grants WHERE tenant_id = ? AND record_key = ?",
            (record.tenant_id, record.record_key),
        )
        grants = self._store._acl_grants_for_record(record)
        now = (
            record.updated_at
            or record.transaction_time
            or record.ingested_at
            or record.created_at
            or self._store._now()
        )
        for grant_kind, grant_id, workspace_id in sorted(grants):
            conn.execute(
                "INSERT INTO context_acl_grants("
                "tenant_id, record_key, grant_kind, grant_id, workspace_id, "
                "scope_signature, uri, context_type, source_kind, record_kind, adapter_id, adapter_access_id, session_id, "
                "event_time, transaction_time, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.tenant_id,
                    record.record_key,
                    grant_kind,
                    grant_id,
                    workspace_id,
                    scope_signature,
                    record.uri,
                    record.context_type,
                    record.source_kind,
                    record.record_kind,
                    record.adapter_id,
                    self._store._adapter_access_value(record),
                    record.session_id,
                    record.event_time,
                    record.transaction_time,
                    now,
                ),
            )

    @staticmethod
    def _acl_grants_for_record(record: CatalogRecord) -> set[tuple[str, str, str]]:
        grants: set[tuple[str, str, str]] = set()
        scope = record.metadata.get("scope")
        visibility = scope.get("visibility") if isinstance(scope, Mapping) else None
        is_canonical = bool(
            record.context_type == "memory"
            and record.record_kind in {CatalogRecordKind.CURRENT_SLOT.value, CatalogRecordKind.CLAIM_REVISION.value}
            and record.canonical_slot_id
            and record.canonical_claim_id
        )
        valid_visibility = bool(
            is_canonical
            and isinstance(visibility, Mapping)
            and str(visibility.get("tenant_id") or "") == record.tenant_id
        )
        if valid_visibility and isinstance(visibility, Mapping):
            if str(visibility.get("tenant_id") or "") == record.tenant_id:
                for principal_id in visibility.get("allowed_principal_ids", ()) or ():
                    if isinstance(principal_id, str) and principal_id:
                        grants.add(("principal", principal_id, record.workspace_id))
                for service_id in visibility.get("allowed_service_ids", ()) or ():
                    if isinstance(service_id, str) and service_id:
                        grants.add(("service", service_id, record.workspace_id))
                tenant_public = bool(
                    visibility.get("private") is False
                    and not (visibility.get("allowed_principal_ids", ()) or ())
                    and not (visibility.get("allowed_service_ids", ()) or ())
                )
                if tenant_public and record.owner_user_id:
                    grants.add(("principal", record.owner_user_id, record.workspace_id))
                if tenant_public:
                    grants.add(("tenant", "", record.workspace_id))
        elif not is_canonical and record.owner_user_id:
            grants.add(("principal", record.owner_user_id, record.workspace_id))
        elif not is_canonical and record.context_type in {"resource", "skill"}:
            grants.add(("public", "", record.workspace_id))
        if is_canonical and record.workspace_shared and record.workspace_id:
            grants.add(("workspace", record.workspace_id, record.workspace_id))
        return grants

    @staticmethod
    def _adapter_access_value(record: CatalogRecord) -> str:
        if (
            not record.adapter_id
            or record.context_type in {"session", "resource", "skill"}
            or record.record_kind == CatalogRecordKind.CURRENT_SLOT.value
        ):
            return "*"
        return record.adapter_id

    def _replace_fts(self, conn: sqlite3.Connection, item: _PreparedCatalogRecord) -> None:
        self._store._delete_fts_record(conn, item.record.record_key)
        if item.record.serving_tier not in {ServingTier.HOT.value, ServingTier.WARM.value}:
            return
        if item.record.projection_status not in {
            CatalogProjectionStatus.PROJECTED.value,
            CatalogProjectionStatus.DEGRADED.value,
        }:
            return
        if item.record.lifecycle_state in {"deleted", "archived", "obsolete"}:
            return
        cursor = conn.execute(
            "INSERT INTO contexts_fts(record_key, uri, title, content_text, metadata_text, search_terms, acl_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                item.record.record_key,
                item.record.uri,
                item.record.title,
                item.record.l1_text,
                item.fts_metadata_text,
                item.fts_search_terms,
                self._store._fts_acl_tokens(item.record, scope_signature=item.scope_signature),
            ),
        )
        raw_fts_rowid = cursor.lastrowid
        if raw_fts_rowid is None:
            raise RuntimeError("FTS insert did not return a durable rowid")
        fts_rowid = int(raw_fts_rowid)
        if fts_rowid <= 0:
            raise RuntimeError("FTS insert did not return a durable rowid")
        conn.execute(
            "INSERT INTO context_fts_map(record_key, fts_rowid) VALUES (?, ?)",
            (item.record.record_key, fts_rowid),
        )

    def _delete_catalog_in_transaction(
        self,
        conn: sqlite3.Connection,
        record_key: str,
        *,
        tenant_id: str | None = None,
    ) -> bool:
        exists = conn.execute(
            "SELECT tenant_id FROM contexts WHERE record_key = ?",
            (record_key,),
        ).fetchone()
        if exists is not None and tenant_id is not None and str(exists["tenant_id"]) != str(tenant_id):
            raise ValueError("Catalog delete tenant does not own record_key")
        resolved_tenant = str(exists["tenant_id"]) if exists is not None else str(tenant_id or "")
        self._store._delete_fts_record(conn, record_key)
        if not resolved_tenant:
            return False
        conn.execute(
            "DELETE FROM context_acl_grants WHERE tenant_id = ? AND record_key = ?",
            (resolved_tenant, record_key),
        )
        conn.execute(
            "DELETE FROM context_path_acl WHERE tenant_id = ? AND record_key = ?",
            (resolved_tenant, record_key),
        )
        conn.execute(
            "DELETE FROM context_path_closure WHERE tenant_id = ? AND record_key = ?",
            (resolved_tenant, record_key),
        )
        conn.execute(
            "DELETE FROM context_paths WHERE tenant_id = ? AND record_key = ?",
            (resolved_tenant, record_key),
        )
        validity = conn.execute(
            "SELECT validity_id FROM context_validity_map WHERE tenant_id = ? AND record_key = ?",
            (resolved_tenant, record_key),
        ).fetchone()
        if validity is not None:
            conn.execute("DELETE FROM context_validity_rtree WHERE validity_id = ?", (int(validity["validity_id"]),))
            conn.execute("DELETE FROM context_validity_map WHERE validity_id = ?", (int(validity["validity_id"]),))
        conn.execute(
            "DELETE FROM context_links WHERE tenant_id = ? AND source_record_key = ?",
            (resolved_tenant, record_key),
        )
        conn.execute(
            "DELETE FROM context_links WHERE tenant_id = ? AND target_record_key = ?",
            (resolved_tenant, record_key),
        )
        conn.execute(
            "DELETE FROM context_projection_state WHERE tenant_id = ? AND record_key = ?",
            (resolved_tenant, record_key),
        )
        conn.execute("DELETE FROM contexts WHERE record_key = ?", (record_key,))
        return exists is not None


__all__ = ["CatalogStoreOperations"]
