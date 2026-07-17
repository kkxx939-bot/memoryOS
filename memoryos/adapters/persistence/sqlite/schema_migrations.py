"""SQLite catalog SchemaMigrationManager responsibility component."""

from __future__ import annotations

from memoryos.adapters.persistence.sqlite._common import (
    _CATALOG_SCHEMA_VERSION,
    _INVALID_SCOPE_KEY,
    _MIGRATION_BATCH_SIZE,
    _SCHEMA_UPGRADE_BOOTSTRAP_TENANT,
    _SCOPE_KEY_SCHEMA_VERSION,
    _UNIFIED_CATALOG_MIGRATION_NAME,
    Any,
    CatalogProjectionStatus,
    CatalogRecord,
    CatalogRecordKind,
    Mapping,
    Sequence,
    ServingTier,
    _PreparedCatalogRecord,
    json,
    lexical_terms,
    normalize_tree_path,
    sqlite3,
)


class SchemaMigrationManager:
    """Own one stable subset of SQLite catalog behavior."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def _record_unified_catalog_schema_upgrade(
        self,
        conn: sqlite3.Connection,
        *,
        upgraded_from_schema_version: int,
    ) -> None:
        """Persist upgrade provenance before a restart can mistake it for greenfield."""

        details = {
            "schema_version": _CATALOG_SCHEMA_VERSION,
            "upgraded_from_schema_version": int(upgraded_from_schema_version),
            "requires_backfill": True,
            "session_backfill_complete": False,
            "backfill_complete": False,
        }
        conn.execute(
            """
            INSERT INTO migration_state(
              migration_name, tenant_id, state, checkpoint, batch_size,
              details_json, last_error, updated_at
            ) VALUES (?, ?, 'SCHEMA_READY', '', ?, ?, '', ?)
            ON CONFLICT(migration_name, tenant_id) DO NOTHING
            """,
            (
                _UNIFIED_CATALOG_MIGRATION_NAME,
                _SCHEMA_UPGRADE_BOOTSTRAP_TENANT,
                _MIGRATION_BATCH_SIZE,
                self._store._json_dump(details),
                self._store._now(),
            ),
        )

    def _rebuild_legacy_contexts(self, conn: sqlite3.Connection, columns: set[str]) -> None:
        conn.execute("DROP TABLE IF EXISTS contexts_catalog_new")
        self._store._create_contexts_table(conn, "contexts_catalog_new")
        cursor = conn.execute("SELECT * FROM contexts ORDER BY uri")
        while batch := cursor.fetchmany(_MIGRATION_BATCH_SIZE):
            for row in batch:
                raw_metadata = self._store._json_mapping(row["metadata_json"] if "metadata_json" in columns else "{}")
                try:
                    scope_keys = self._store._scope_keys_from_metadata(raw_metadata)
                except (KeyError, TypeError, ValueError):
                    scope_keys = [_INVALID_SCOPE_KEY]
                record = self._store._legacy_record(row, columns, raw_metadata)
                prepared = self._store._prepare_record(
                    record,
                    scope_keys_override=scope_keys,
                    legacy_overrides={
                        "project_id": self._store._legacy_value(row, columns, "project_id"),
                        "admission_status": self._store._legacy_value(row, columns, "admission_status"),
                        "claim_state": self._store._legacy_value(row, columns, "claim_state"),
                        "slot_id": self._store._legacy_value(row, columns, "slot_id"),
                        "memory_type": self._store._legacy_value(row, columns, "memory_type"),
                    },
                )
                self._store._insert_context_row(conn, prepared.values, table_name="contexts_catalog_new")
        conn.execute("DROP TABLE contexts")
        conn.execute("ALTER TABLE contexts_catalog_new RENAME TO contexts")

    def _legacy_record(
        self,
        row: sqlite3.Row,
        columns: set[str],
        metadata: Mapping[str, Any],
    ) -> CatalogRecord:
        updated_at = self._store._coerce_timestamp(self._store._legacy_value(row, columns, "updated_at"))
        created_at = self._store._coerce_timestamp(str(metadata.get("created_at") or updated_at))
        raw_paths = metadata.get("tree_paths")
        tree_paths: tuple[str, ...] = ()
        if isinstance(raw_paths, Sequence) and not isinstance(raw_paths, str | bytes):
            try:
                tree_paths = tuple(normalize_tree_path(item) for item in raw_paths)
            except ValueError:
                tree_paths = ()
        primary = str(metadata.get("primary_tree_path") or (tree_paths[0] if tree_paths else ""))
        if primary:
            try:
                primary = normalize_tree_path(primary)
            except ValueError:
                primary = ""
                tree_paths = ()
        record_kind = str(metadata.get("record_kind") or CatalogRecordKind.CONTEXT.value)
        if record_kind not in {kind.value for kind in CatalogRecordKind}:
            record_kind = CatalogRecordKind.CONTEXT.value
        serving_tier = str(metadata.get("serving_tier") or ServingTier.HOT.value).upper()
        if serving_tier not in {tier.value for tier in ServingTier}:
            serving_tier = ServingTier.HOT.value
        projection_status = str(metadata.get("projection_status") or CatalogProjectionStatus.PROJECTED.value).upper()
        if projection_status not in {status.value for status in CatalogProjectionStatus}:
            projection_status = CatalogProjectionStatus.PROJECTED.value
        uri = self._store._legacy_value(row, columns, "uri")
        content = self._store._legacy_value(row, columns, "content_text")
        project_id = self._store._legacy_value(row, columns, "project_id")
        return CatalogRecord(
            record_key=uri,
            uri=uri,
            tenant_id=self._store._legacy_value(row, columns, "tenant_id") or "default",
            owner_user_id=self._store._legacy_value(row, columns, "owner_user_id"),
            workspace_id=str(metadata.get("workspace_id") or project_id),
            session_id=str(metadata.get("session_id") or ""),
            adapter_id=self._store._legacy_value(row, columns, "adapter_id"),
            context_type=self._store._legacy_value(row, columns, "context_type"),
            source_kind=str(metadata.get("source_kind") or "context"),
            record_kind=record_kind,
            lifecycle_state=self._store._legacy_value(row, columns, "lifecycle_state") or "active",
            primary_tree_path=primary,
            tree_paths=tree_paths,
            created_at=created_at,
            updated_at=updated_at,
            event_time=self._store._coerce_timestamp(str(metadata.get("event_time") or created_at)),
            ingested_at=self._store._coerce_timestamp(str(metadata.get("ingested_at") or created_at)),
            transaction_time=self._store._coerce_timestamp(str(metadata.get("transaction_time") or updated_at)),
            valid_from=self._store._coerce_timestamp(str(metadata.get("valid_from") or "")),
            valid_to=self._store._coerce_timestamp(str(metadata.get("valid_to") or "")),
            title=self._store._legacy_value(row, columns, "title"),
            l0_text=str(metadata.get("l0_text") or self._store._legacy_value(row, columns, "title")),
            l1_text=content,
            l2_uri=str(metadata.get("l2_uri") or uri),
            source_uri=str(metadata.get("source_uri") or uri),
            source_digest=str(metadata.get("source_digest") or self._store.sanitizer.digest(content)),
            source_revision=int(metadata.get("source_revision") or metadata.get("revision") or 0),
            canonical_slot_id=self._store._legacy_value(row, columns, "slot_id"),
            canonical_slot_uri=str(metadata.get("slot_uri") or metadata.get("canonical_slot_uri") or ""),
            canonical_claim_id=str(metadata.get("claim_id") or metadata.get("canonical_claim_id") or ""),
            canonical_claim_uri=str(metadata.get("canonical_claim_uri") or ""),
            canonical_revision=int(metadata.get("current_revision") or metadata.get("revision") or 0),
            canonical_state=self._store._legacy_value(row, columns, "claim_state"),
            canonical_head_digest=str(
                metadata.get("canonical_head_digest") or metadata.get("current_head_digest") or ""
            ),
            receipt_digest=str(metadata.get("receipt_digest") or metadata.get("current_receipt_digest") or ""),
            projection_effect_hash=str(metadata.get("projection_effect_hash") or ""),
            hotness=float(self._store._legacy_value(row, columns, "hotness") or 0.0),
            semantic_hotness=float(self._store._legacy_value(row, columns, "semantic_hotness") or 0.0),
            behavior_support_hotness=float(self._store._legacy_value(row, columns, "behavior_support_hotness") or 0.0),
            serving_tier=serving_tier,
            projection_status=projection_status,
            metadata=metadata,
        )

    def _sanitize_existing_rows(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute("SELECT record_key FROM contexts ORDER BY record_key")
        while batch := cursor.fetchmany(_MIGRATION_BATCH_SIZE):
            for key_row in batch:
                row = conn.execute(
                    "SELECT * FROM contexts WHERE record_key = ?",
                    (str(key_row["record_key"]),),
                ).fetchone()
                if row is None:
                    continue
                scope_keys = self._store._json_list(row["scope_keys"])
                record = self._store._catalog_record_from_row(conn, row)
                legacy_overrides = {
                    "project_id": str(row["project_id"]),
                    "admission_status": str(row["admission_status"]),
                    "claim_state": str(row["claim_state"]),
                    "slot_id": str(row["slot_id"]),
                    "memory_type": str(row["memory_type"]),
                }
                if str(row["content_digest"]):
                    legacy_overrides["content_digest"] = str(row["content_digest"])
                prepared = self._store._prepare_record(
                    record,
                    scope_keys_override=scope_keys,
                    legacy_overrides=legacy_overrides,
                )
                self._store._upsert_prepared(conn, prepared)

    def _migrate_scope_keys(self, conn: sqlite3.Connection) -> None:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version >= _SCOPE_KEY_SCHEMA_VERSION:
            conn.execute(
                f"UPDATE contexts SET scope_keys = '[\"{_INVALID_SCOPE_KEY}\"]' "
                "WHERE NOT json_valid(scope_keys) OR json_type(scope_keys) != 'array'"
            )
            return
        cursor = conn.execute("SELECT record_key, metadata_json FROM contexts ORDER BY record_key")
        while batch := cursor.fetchmany(_MIGRATION_BATCH_SIZE):
            for row in batch:
                try:
                    metadata = json.loads(str(row["metadata_json"] or "{}"))
                    if not isinstance(metadata, Mapping):
                        raise ValueError("metadata must be an object")
                    keys = self._store._scope_keys_from_metadata(metadata)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    keys = [_INVALID_SCOPE_KEY]
                conn.execute(
                    "UPDATE contexts SET scope_keys = ? WHERE record_key = ?",
                    (self._store._json_dump(keys), str(row["record_key"])),
                )

    def _rebuild_fts(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM contexts_fts")
        conn.execute("DELETE FROM context_fts_map")
        cursor = conn.execute("SELECT * FROM contexts ORDER BY record_key")
        while batch := cursor.fetchmany(_MIGRATION_BATCH_SIZE):
            for row in batch:
                metadata = self._store._json_mapping(row["metadata_json"])
                metadata_text = self._store._safe_metadata_text(metadata)
                item = _PreparedCatalogRecord(
                    record=self._store._catalog_record_from_row(conn, row),
                    values={},
                    scope_signature=self._store._scope_signature(self._store._json_list(row["scope_keys"])),
                    fts_metadata_text=metadata_text,
                    fts_search_terms=" ".join(
                        lexical_terms(
                            " ".join(
                                (
                                    str(row["title"]),
                                    str(row["l0_text"]),
                                    str(row["content_text"]),
                                    metadata_text,
                                )
                            )
                        )
                    ),
                )
                self._store._replace_fts(conn, item)


__all__ = ["SchemaMigrationManager"]
