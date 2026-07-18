"""SQLite-backed, tenant-safe Unified Context Catalog."""

from __future__ import annotations

import hashlib

from memoryos.adapters.persistence.sqlite._common import (
    _ONLINE_VM_STEP_LIMIT,
    Any,
    CatalogRecord,
    ContextObject,
    ContextProjectionSanitizer,
    IndexHit,
    Mapping,
    Path,
    Sequence,
    lexical_match_count,
    lexical_relevance,
    lexical_terms,
    os,
)
from memoryos.adapters.persistence.sqlite.base_filter import BaseFilterBuilder
from memoryos.adapters.persistence.sqlite.catalog import CatalogStoreOperations
from memoryos.adapters.persistence.sqlite.connection import SQLiteConnectionManager
from memoryos.adapters.persistence.sqlite.query_filters import QueryFilterBuilder
from memoryos.adapters.persistence.sqlite.schema import SchemaManager
from memoryos.adapters.persistence.sqlite.search import CatalogSearchOperations
from memoryos.adapters.persistence.sqlite.serialization import CatalogSerializer
from memoryos.adapters.persistence.sqlite.tombstones import TombstoneOperations


class SQLiteIndexStore:
    """Greenfield Catalog facade with explicit tenant ownership on every API."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.fts_enabled = True
        self.online_vm_step_limit = _ONLINE_VM_STEP_LIMIT
        self.sanitizer = ContextProjectionSanitizer()
        self._serialization = CatalogSerializer(self)
        self._connection = SQLiteConnectionManager(self)
        self._base_filter = BaseFilterBuilder(self)
        self._query_filters = QueryFilterBuilder(self)
        self._catalog = CatalogStoreOperations(self)
        self._search = CatalogSearchOperations(self)
        self._tombstones = TombstoneOperations(self)
        self._schema = SchemaManager(self)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        self._schema._init_db()
        os.chmod(self.path, 0o600)

    def __getattr__(self, name: str) -> Any:
        """Expose private component helpers without duplicating their logic."""

        for component_name in (
            "_serialization",
            "_connection",
            "_base_filter",
            "_query_filters",
            "_catalog",
            "_search",
            "_tombstones",
            "_schema",
        ):
            component = object.__getattribute__(self, component_name)
            if hasattr(type(component), name):
                return getattr(component, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def upsert_index(self, obj: ContextObject, content: str = "", *, tenant_id: str) -> None:
        self._catalog.upsert_index(obj, content, tenant_id=tenant_id)

    def delete_index(self, uri: str, *, tenant_id: str) -> None:
        self._catalog.delete_index(uri, tenant_id=tenant_id)

    def indexed_uris(self, *, tenant_id: str) -> list[str]:
        return self._catalog.indexed_uris(tenant_id=tenant_id)

    def get_index_metadata(self, uri: str, *, tenant_id: str) -> dict[str, Any] | None:
        return self._catalog.get_index_metadata(uri, tenant_id=tenant_id)

    def ordinary_relation_endpoint_state(
        self,
        uri: str,
        *,
        tenant_id: str,
        session_id: str = "",
    ) -> str:
        return self._catalog.ordinary_relation_endpoint_state(
            uri,
            tenant_id=tenant_id,
            session_id=session_id,
        )

    def clear(self, *, tenant_id: str) -> None:
        self._catalog.clear(tenant_id=tenant_id)

    def rebuildable_catalog_records(
        self,
        records: Sequence[CatalogRecord],
        *,
        tenant_id: str,
    ) -> tuple[CatalogRecord, ...]:
        return self._catalog.rebuildable_catalog_records(records, tenant_id=tenant_id)

    def search(
        self,
        query: str,
        *,
        tenant_id: str,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
    ) -> list[IndexHit]:
        return self._search.search(query, tenant_id=tenant_id, filters=filters, limit=limit)

    def upsert_catalog(
        self,
        record: CatalogRecord | Mapping[str, Any],
        *,
        tenant_id: str,
    ) -> None:
        self._catalog.upsert_catalog(record, tenant_id=tenant_id)

    def upsert_catalog_batch(
        self,
        records: Sequence[CatalogRecord | Mapping[str, Any]],
        *,
        tenant_id: str,
    ) -> int:
        return self._catalog.upsert_catalog_batch(records, tenant_id=tenant_id)

    def replace_memory_document_projection(
        self,
        document_record: CatalogRecord | Mapping[str, Any],
        block_records: Sequence[CatalogRecord | Mapping[str, Any]],
        expected_previous_generation: int | None,
        *,
        tenant_id: str,
        owner_user_id: str,
        restore_soft_deleted: bool = False,
    ) -> tuple[str, ...]:
        return self._catalog.replace_memory_document_projection(
            document_record,
            block_records,
            expected_previous_generation,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            restore_soft_deleted=restore_soft_deleted,
        )

    def get_memory_document_projection_state(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> dict[str, Any] | None:
        return self._catalog.get_memory_document_projection_state(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            document_id=document_id,
        )

    def tombstone_memory_document_projection(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        deletion_generation: int,
        deletion_event_digest: str,
        deletion_status: str,
        relative_path: str = "",
    ) -> tuple[str, ...]:
        return self._catalog.tombstone_memory_document_projection(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            document_id=document_id,
            deletion_generation=deletion_generation,
            deletion_event_digest=deletion_event_digest,
            deletion_status=deletion_status,
            relative_path=relative_path,
        )

    def get_catalog(self, record_key: str, *, tenant_id: str) -> CatalogRecord | None:
        return self._catalog.get_catalog(record_key, tenant_id=tenant_id)

    def get_catalog_by_uri(
        self,
        uri: str,
        *,
        tenant_id: str,
        limit: int = 100,
    ) -> list[CatalogRecord]:
        return self._catalog.get_catalog_by_uri(uri, tenant_id=tenant_id, limit=limit)

    def list_catalog(
        self,
        *,
        tenant_id: str,
        filters: Mapping[str, Any] | None = None,
        limit: int = 100,
    ) -> list[CatalogRecord]:
        return self._catalog.list_catalog(tenant_id=tenant_id, filters=filters, limit=limit)

    def list_catalog_projection_records(
        self,
        *,
        tenant_id: str,
        source_uri: str,
        projection_effect_hash: str,
        limit: int = 1_001,
    ) -> list[CatalogRecord]:
        return self._catalog.list_catalog_projection_records(
            tenant_id=tenant_id,
            source_uri=source_uri,
            projection_effect_hash=projection_effect_hash,
            limit=limit,
        )

    def scan_catalog_batch(
        self,
        *,
        tenant_id: str,
        after_record_key: str = "",
        filters: Mapping[str, Any] | None = None,
        limit: int = 256,
    ) -> list[CatalogRecord]:
        return self._catalog.scan_catalog_batch(
            tenant_id=tenant_id,
            after_record_key=after_record_key,
            filters=filters,
            limit=limit,
        )

    def catalog_schema_version(self) -> int:
        return self._catalog.catalog_schema_version()

    def gc_orphan_paths(self, *, tenant_id: str, limit: int = 256) -> int:
        return self._catalog.gc_orphan_paths(tenant_id=tenant_id, limit=limit)

    def gc_applied_tombstones(
        self,
        *,
        tenant_id: str,
        updated_before: str,
        limit: int = 256,
    ) -> int:
        return self._catalog.gc_applied_tombstones(
            tenant_id=tenant_id,
            updated_before=updated_before,
            limit=limit,
        )

    def search_catalog(
        self,
        query: str,
        *,
        tenant_id: str,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
    ) -> list[IndexHit]:
        return self._search.search_catalog(
            query,
            tenant_id=tenant_id,
            filters=filters,
            limit=limit,
        )

    def delete_catalog(self, record_key: str, *, tenant_id: str) -> bool:
        return self._catalog.delete_catalog(record_key, tenant_id=tenant_id)

    def apply_tombstone(
        self,
        *,
        tenant_id: str,
        record_key: str,
        reason: str,
        uri: str = "",
        source_revision: int = 0,
        tombstone_id: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._tombstones.apply_tombstone(
            tenant_id=tenant_id,
            record_key=record_key,
            reason=reason,
            uri=uri,
            source_revision=source_revision,
            tombstone_id=tombstone_id,
            payload=payload,
        )

    def enqueue_tombstone(
        self,
        *,
        tenant_id: str,
        record_key: str,
        reason: str,
        uri: str = "",
        source_revision: int = 0,
        tombstone_id: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._tombstones.enqueue_tombstone(
            tenant_id=tenant_id,
            record_key=record_key,
            reason=reason,
            uri=uri,
            source_revision=source_revision,
            tombstone_id=tombstone_id,
            payload=payload,
        )

    def get_pending_tombstones(self, *, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return self._tombstones.get_pending_tombstones(tenant_id=tenant_id, limit=limit)

    def get_pending_tombstones_for_uri(
        self,
        uri: str,
        *,
        tenant_id: str,
        after_tombstone_id: str = "",
        limit: int = 1_000,
    ) -> list[dict[str, Any]]:
        return self._tombstones.get_pending_tombstones_for_uri(
            uri,
            tenant_id=tenant_id,
            after_tombstone_id=after_tombstone_id,
            limit=limit,
        )

    def get_tombstones(
        self,
        tombstone_ids: Sequence[str],
        *,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        return self._tombstones.get_tombstones(tombstone_ids, tenant_id=tenant_id)

    def pending_tombstones(self, *, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return self.get_pending_tombstones(tenant_id=tenant_id, limit=limit)

    def mark_tombstone_applied(self, tombstone_id: str, *, tenant_id: str) -> dict[str, Any] | None:
        return self._tombstones.mark_tombstone_applied(tombstone_id, tenant_id=tenant_id)

    def begin_tombstone_cleanup(self, tombstone_id: str, *, tenant_id: str) -> dict[str, Any] | None:
        return self._tombstones.begin_tombstone_cleanup(tombstone_id, tenant_id=tenant_id)

    def finish_tombstone_cleanup(self, tombstone_id: str, *, tenant_id: str) -> dict[str, Any] | None:
        return self._tombstones.finish_tombstone_cleanup(tombstone_id, tenant_id=tenant_id)

    def mark_tombstone_cleanup_failed(
        self,
        tombstone_id: str,
        error: str,
        *,
        tenant_id: str,
    ) -> dict[str, Any] | None:
        return self._tombstones.mark_tombstone_cleanup_failed(
            tombstone_id,
            error,
            tenant_id=tenant_id,
        )

    def mark_tombstone_failed(
        self,
        tombstone_id: str,
        error: str,
        *,
        tenant_id: str,
    ) -> dict[str, Any] | None:
        return self._tombstones.mark_tombstone_failed(tombstone_id, error, tenant_id=tenant_id)

    def set_session_projection_frontier(
        self,
        *,
        tenant_id: str,
        archive_uri: str,
        owner_user_id: str = "",
        workspace_id: str = "",
        session_id: str,
        manifest_digest: str,
        status: str,
        error: str = "",
    ) -> dict[str, Any]:
        """Persist Session projection progress in the generic projection journal."""

        if not tenant_id or not archive_uri or not session_id or not status:
            raise ValueError("session projection journal identity is required")
        safe_error = str(self.sanitizer.sanitize_trace(str(error or "")))
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO context_projection_journal(
                  tenant_id, projector_kind, source_uri, owner_user_id, workspace_id,
                  source_id, source_digest, status, last_error, created_at, updated_at
                ) VALUES (?, 'session', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, projector_kind, source_uri) DO UPDATE SET
                  owner_user_id=excluded.owner_user_id,
                  workspace_id=excluded.workspace_id,
                  source_id=excluded.source_id,
                  source_digest=excluded.source_digest,
                  status=excluded.status,
                  last_error=excluded.last_error,
                  updated_at=excluded.updated_at
                """,
                (
                    str(tenant_id),
                    str(archive_uri),
                    str(owner_user_id),
                    str(workspace_id),
                    str(session_id),
                    str(manifest_digest),
                    str(status),
                    safe_error,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM context_projection_journal WHERE tenant_id = ? "
                "AND projector_kind = 'session' AND source_uri = ?",
                (str(tenant_id), str(archive_uri)),
            ).fetchone()
        if row is None:  # pragma: no cover
            raise RuntimeError("session projection journal did not persist")
        return self._row_dict(row)

    def get_session_projection_frontier_summary(
        self,
        *,
        tenant_id: str,
        owner_user_id: str | None = None,
        workspace_ids: Sequence[str] | None = None,
    ) -> dict[str, int]:
        sql = (
            "SELECT status, COUNT(*) AS count FROM context_projection_journal "
            "WHERE tenant_id = ? AND projector_kind = 'session'"
        )
        params: list[Any] = [str(tenant_id)]
        if owner_user_id is not None:
            sql += " AND owner_user_id = ?"
            params.append(str(owner_user_id))
        if workspace_ids is not None:
            values = tuple(dict.fromkeys(str(item) for item in workspace_ids))
            if not values:
                return {}
            sql += " AND workspace_id IN (" + ", ".join("?" for _ in values) + ")"
            params.extend(values)
        sql += " GROUP BY status"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def list_session_projection_frontier(
        self,
        *,
        tenant_id: str,
        statuses: Sequence[str] = ("PENDING", "FAILED"),
        after_archive_uri: str = "",
        limit: int = 256,
    ) -> list[dict[str, Any]]:
        values = tuple(dict.fromkeys(str(item) for item in statuses if str(item)))
        if not values:
            return []
        placeholders = ", ".join("?" for _ in values)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM context_projection_journal WHERE tenant_id = ? "
                "AND projector_kind = 'session' AND source_uri > ? "
                f"AND status IN ({placeholders}) ORDER BY source_uri LIMIT ?",
                (str(tenant_id), str(after_archive_uri), *values, self._bounded_limit(limit)),
            ).fetchall()
        return [self._row_dict(row) for row in rows]

    def upsert_context_link(
        self,
        *,
        tenant_id: str,
        source_record_key: str,
        source_uri: str,
        relation_type: str,
        target_uri: str,
        target_record_key: str = "",
        metadata: Mapping[str, Any] | None = None,
        link_key: str = "",
    ) -> str:
        if not tenant_id or not source_record_key or not relation_type or not target_uri:
            raise ValueError("context link identity is required")
        identity = "\x00".join(
            (
                str(tenant_id),
                str(source_record_key),
                str(relation_type),
                str(target_record_key),
                str(target_uri),
            )
        )
        resolved_link_key = str(link_key or hashlib.sha256(identity.encode("utf-8")).hexdigest())
        safe_metadata = self.sanitizer.sanitize_trace(dict(metadata or {}))
        now = self._now()
        with self._connect() as conn:
            source = conn.execute(
                "SELECT 1 FROM contexts WHERE tenant_id = ? AND record_key = ?",
                (str(tenant_id), str(source_record_key)),
            ).fetchone()
            if source is None:
                raise ValueError("context link source record is missing")
            if target_record_key:
                target = conn.execute(
                    "SELECT 1 FROM contexts WHERE tenant_id = ? AND record_key = ?",
                    (str(tenant_id), str(target_record_key)),
                ).fetchone()
                if target is None:
                    raise ValueError("context link target record is missing")
            conn.execute(
                """
                INSERT INTO context_links(
                  tenant_id, link_key, source_record_key, source_uri, relation_type,
                  target_record_key, target_uri, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, link_key) DO UPDATE SET
                  source_record_key=excluded.source_record_key,
                  source_uri=excluded.source_uri,
                  relation_type=excluded.relation_type,
                  target_record_key=excluded.target_record_key,
                  target_uri=excluded.target_uri,
                  metadata_json=excluded.metadata_json,
                  updated_at=excluded.updated_at
                """,
                (
                    str(tenant_id),
                    resolved_link_key,
                    str(source_record_key),
                    self._safe_reference_uri(str(source_uri)),
                    str(relation_type),
                    str(target_record_key),
                    self._safe_reference_uri(str(target_uri)),
                    self._json_dump(safe_metadata),
                    now,
                    now,
                ),
            )
        return resolved_link_key

    def explain_structured_query(
        self,
        *,
        tenant_id: str,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
    ) -> list[str]:
        return self._search.explain_structured_query(
            tenant_id=tenant_id,
            filters=filters,
            limit=limit,
        )


SqliteIndexStore = SQLiteIndexStore


__all__ = [
    "SQLiteIndexStore",
    "SqliteIndexStore",
    "lexical_match_count",
    "lexical_relevance",
    "lexical_terms",
]
