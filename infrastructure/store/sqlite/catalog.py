"""具备租户隔离的 SQLite Catalog 操作。"""

from __future__ import annotations

from infrastructure.store.sqlite._common import (
    _CATALOG_SCHEMA_VERSION,
    Any,
    CatalogRecord,
    CatalogRecordKind,
    ContextObject,
    Mapping,
    Sequence,
    replace,
)
from infrastructure.store.sqlite.catalog_documents import CatalogDocumentOperationsMixin
from infrastructure.store.sqlite.catalog_writes import CatalogWriteOperationsMixin

_DOCUMENT_RECORD_KINDS = frozenset(
    {
        CatalogRecordKind.MEMORY_DOCUMENT.value,
        CatalogRecordKind.MEMORY_BLOCK.value,
    }
)


class CatalogStoreOperations(CatalogDocumentOperationsMixin, CatalogWriteOperationsMixin):
    """管理事务写入和租户范围内的有界 Catalog 读取。"""

    def __init__(self, store: Any) -> None:
        self._store = store

    @staticmethod
    def _require_tenant(tenant_id: str) -> str:
        resolved = str(tenant_id or "").strip()
        if not resolved:
            raise ValueError("tenant_id is required")
        return resolved

    def upsert_index(self, obj: ContextObject, content: str = "", *, tenant_id: str) -> None:
        """把一个普通 ContextObject 投影到所属租户的 Catalog。"""

        resolved_tenant = self._require_tenant(tenant_id)
        if str(obj.tenant_id or "default") != resolved_tenant:
            raise ValueError("ContextObject tenant does not match tenant_id")
        record = CatalogRecord.from_context_object(obj, content=content)
        if content:
            record = replace(record, l1_text=content)
        self.upsert_catalog(record, tenant_id=resolved_tenant)

    def delete_index(self, uri: str, *, tenant_id: str) -> None:
        """删除一个租户内逻辑 URI 对应的全部服务记录。"""

        resolved_tenant = self._require_tenant(tenant_id)
        with self._store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT record_key, record_kind FROM contexts WHERE tenant_id = ? AND uri = ? ORDER BY record_key",
                (resolved_tenant, str(uri)),
            ).fetchall()
            if any(str(row["record_kind"]) in _DOCUMENT_RECORD_KINDS for row in rows):
                raise ValueError("memory document projections require tombstone_memory_document_projection()")
            for row in rows:
                self._delete_catalog_in_transaction(
                    conn,
                    str(row["record_key"]),
                    tenant_id=resolved_tenant,
                )

    def indexed_uris(self, *, tenant_id: str) -> list[str]:
        resolved_tenant = self._require_tenant(tenant_id)
        with self._store._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT uri FROM contexts WHERE tenant_id = ? ORDER BY uri",
                (resolved_tenant,),
            ).fetchall()
        return [str(row["uri"]) for row in rows]

    def get_index_metadata(self, uri: str, *, tenant_id: str) -> dict[str, Any] | None:
        resolved_tenant = self._require_tenant(tenant_id)
        with self._store._connect() as conn:
            row = conn.execute(
                "SELECT * FROM contexts WHERE tenant_id = ? AND uri = ? ORDER BY updated_at DESC, record_key LIMIT 1",
                (resolved_tenant, str(uri)),
            ).fetchone()
        if row is None:
            return None
        metadata = self._store._json_mapping(row["metadata_json"])
        return {
            **metadata,
            "record_key": str(row["record_key"]),
            "tenant_id": str(row["tenant_id"]),
            "owner_user_id": str(row["owner_user_id"]),
            "context_type": str(row["context_type"]),
            "document_id": str(row["document_id"]),
            "block_id": str(row["block_id"]),
            "document_kind": str(row["document_kind"]),
            "document_revision": int(row["document_revision"]),
            "projection_generation": int(row["projection_generation"]),
            "index_content_digest": self._store._content_digest(str(row["content_text"])),
        }

    def ordinary_relation_endpoint_state(
        self,
        uri: str,
        *,
        tenant_id: str,
        session_id: str = "",
    ) -> str:
        """在单个租户内判断普通关系端点是否仍然有效。"""

        resolved_tenant = self._require_tenant(tenant_id)
        safe_uri = self._store._safe_reference_uri(str(uri))
        safe_session_id = str(session_id or "")
        with self._store._connect() as conn:
            blocked = conn.execute(
                "SELECT 1 FROM context_tombstones WHERE tenant_id = ? "
                "AND status IN ('PENDING', 'FAILED', 'CLEANING') "
                "AND (uri = ? OR (? <> '' AND json_extract(payload_json, '$.session_id') = ?)) LIMIT 1",
                (resolved_tenant, safe_uri, safe_session_id, safe_session_id),
            ).fetchone()
            if blocked is not None:
                return "retired"
            row = conn.execute(
                "SELECT lifecycle_state FROM contexts WHERE tenant_id = ? "
                "AND (uri = ? OR source_uri = ? OR (? <> '' AND session_id = ?)) "
                "ORDER BY CASE WHEN lifecycle_state = 'active' THEN 0 ELSE 1 END, updated_at DESC LIMIT 1",
                (resolved_tenant, safe_uri, safe_uri, safe_session_id, safe_session_id),
            ).fetchone()
            if row is not None and str(row["lifecycle_state"]) == "active":
                return "active"
            retired = conn.execute(
                "SELECT 1 FROM context_tombstones WHERE tenant_id = ? AND uri = ? AND status = 'APPLIED' LIMIT 1",
                (resolved_tenant, safe_uri),
            ).fetchone()
        if retired is not None:
            return "retired"
        return "missing" if row is None else "inactive"

    def clear(self, *, tenant_id: str) -> None:
        """清理可重建服务记录，同时保留日志和删除屏障。"""

        resolved_tenant = self._require_tenant(tenant_id)
        with self._store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT record_key FROM contexts WHERE tenant_id = ? ORDER BY record_key",
                (resolved_tenant,),
            ).fetchall()
            for row in rows:
                self._delete_catalog_in_transaction(
                    conn,
                    str(row["record_key"]),
                    tenant_id=resolved_tenant,
                )
            conn.execute(
                """
                UPDATE memory_document_projection_state SET
                  source_digest='',
                  projection_generation=0,
                  projection_status=CASE
                    WHEN deletion_status <> '' THEN 'TOMBSTONED'
                    ELSE 'PENDING'
                  END,
                  projected_at='',
                  last_error=''
                WHERE tenant_id = ?
                """,
                (resolved_tenant,),
            )
            conn.execute(
                "UPDATE context_projection_journal SET status = 'PENDING', last_error = '', updated_at = ? "
                "WHERE tenant_id = ?",
                (self._store._now(), resolved_tenant),
            )

    def rebuildable_catalog_records(
        self,
        records: Sequence[CatalogRecord],
        *,
        tenant_id: str,
    ) -> tuple[CatalogRecord, ...]:
        """通过租户内耐久屏障过滤离线重建批次。"""

        resolved_tenant = self._require_tenant(tenant_id)
        selected: list[CatalogRecord] = []
        with self._store._connect() as conn:
            for record in records:
                if record.tenant_id != resolved_tenant:
                    raise ValueError("Catalog rebuild batch crosses tenant boundary")
                rows = conn.execute(
                    "SELECT source_revision, status FROM context_tombstones "
                    "WHERE tenant_id = ? AND record_key = ? AND status IN ('CLEANING', 'APPLIED') "
                    "ORDER BY created_at, tombstone_id",
                    (resolved_tenant, record.record_key),
                ).fetchall()
                blocked = False
                for row in rows:
                    if str(row["status"]) == "CLEANING":
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

    def upsert_catalog(
        self,
        record: CatalogRecord | Mapping[str, Any],
        *,
        tenant_id: str,
    ) -> None:
        self.upsert_catalog_batch((record,), tenant_id=tenant_id)

    def upsert_catalog_batch(
        self,
        records: Sequence[CatalogRecord | Mapping[str, Any]],
        *,
        tenant_id: str,
    ) -> int:
        """原子清洗并写入普通 Catalog 记录。"""

        resolved_tenant = self._require_tenant(tenant_id)
        coerced = tuple(self._store._coerce_record(record) for record in records)
        if len({record.record_key for record in coerced}) != len(coerced):
            raise ValueError("Catalog batch record_key values must be unique within a tenant")
        for record in coerced:
            if record.tenant_id != resolved_tenant:
                raise ValueError("Catalog record tenant does not match tenant_id")
            if record.record_kind in _DOCUMENT_RECORD_KINDS:
                raise ValueError("memory document projections require replace_memory_document_projection()")
        prepared = tuple(self._prepare_record(record) for record in coerced)
        if not prepared:
            return 0
        with self._store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for item in prepared:
                existing = conn.execute(
                    "SELECT record_kind FROM contexts WHERE tenant_id = ? AND record_key = ?",
                    (resolved_tenant, item.record.record_key),
                ).fetchone()
                if existing is not None and str(existing["record_kind"]) in _DOCUMENT_RECORD_KINDS:
                    raise ValueError("memory document projections require replace_memory_document_projection()")
            for item in prepared:
                self._upsert_prepared(conn, item)
        return len(prepared)

    def get_catalog(self, record_key: str, *, tenant_id: str) -> CatalogRecord | None:
        resolved_tenant = self._require_tenant(tenant_id)
        with self._store._connect() as conn:
            row = conn.execute(
                "SELECT * FROM contexts WHERE tenant_id = ? AND record_key = ?",
                (resolved_tenant, str(record_key)),
            ).fetchone()
            return None if row is None else self._store._catalog_record_from_row(conn, row)

    def get_catalog_by_uri(
        self,
        uri: str,
        *,
        tenant_id: str,
        limit: int = 100,
    ) -> list[CatalogRecord]:
        return self.list_catalog(
            tenant_id=tenant_id,
            filters={"target_uris": (str(uri),), "include_inactive": True},
            limit=limit,
        )

    def list_catalog(
        self,
        *,
        tenant_id: str,
        filters: Mapping[str, Any] | None = None,
        limit: int = 100,
    ) -> list[CatalogRecord]:
        resolved_tenant = self._require_tenant(tenant_id)
        normalized_filters = self._tenant_filters(filters, resolved_tenant)
        bounded_limit = self._store._bounded_limit(limit)
        filter_sql, params = self._store._base_filter_sql(normalized_filters)
        sql = f"SELECT c.* FROM contexts AS c WHERE 1=1 {filter_sql} ORDER BY c.updated_at DESC, c.record_key LIMIT ?"
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
        resolved_tenant = self._require_tenant(tenant_id)
        bounded = int(limit)
        if not source_uri or not projection_effect_hash:
            raise ValueError("projection evidence identity is required")
        if not 1 <= bounded <= 1_001:
            raise ValueError("projection proof lookup limit must be between 1 and 1001")
        with self._store._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM contexts WHERE tenant_id = ? AND source_uri = ? "
                "AND projection_effect_hash = ? ORDER BY record_key LIMIT ?",
                (resolved_tenant, str(source_uri), str(projection_effect_hash), bounded),
            ).fetchall()
            return self._store._catalog_records_from_rows(conn, rows)

    def scan_catalog_batch(
        self,
        *,
        tenant_id: str,
        after_record_key: str = "",
        filters: Mapping[str, Any] | None = None,
        limit: int = 256,
    ) -> list[CatalogRecord]:
        resolved_tenant = self._require_tenant(tenant_id)
        normalized_filters = self._tenant_filters(filters, resolved_tenant)
        bounded_limit = self._store._bounded_limit(limit)
        filter_sql, params = self._store._base_filter_sql(normalized_filters)
        sql = f"SELECT c.* FROM contexts AS c WHERE c.record_key > ? {filter_sql} ORDER BY c.record_key LIMIT ?"
        with self._store._connect() as conn:
            rows = conn.execute(sql, [str(after_record_key), *params, bounded_limit]).fetchall()
            return self._store._catalog_records_from_rows(conn, rows)

    def catalog_schema_version(self) -> int:
        with self._store._connect() as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version != _CATALOG_SCHEMA_VERSION:
            raise RuntimeError("unsupported Catalog schema version")
        return version

    def gc_orphan_paths(self, *, tenant_id: str, limit: int = 256) -> int:
        resolved_tenant = self._require_tenant(tenant_id)
        with self._store._connect() as conn:
            rows = conn.execute(
                "SELECT p.record_key, p.path FROM context_paths AS p "
                "LEFT JOIN contexts AS c ON c.tenant_id = p.tenant_id AND c.record_key = p.record_key "
                "WHERE p.tenant_id = ? AND c.record_key IS NULL "
                "ORDER BY p.record_key, p.path LIMIT ?",
                (resolved_tenant, self._store._bounded_limit(limit)),
            ).fetchall()
            for row in rows:
                params = (resolved_tenant, str(row["record_key"]), str(row["path"]))
                conn.execute(
                    "DELETE FROM context_paths WHERE tenant_id = ? AND record_key = ? AND path = ?",
                    params,
                )
                conn.execute(
                    "DELETE FROM context_path_closure WHERE tenant_id = ? AND record_key = ? AND path = ?",
                    params,
                )
                conn.execute(
                    "DELETE FROM context_path_acl WHERE tenant_id = ? AND record_key = ? AND path = ?",
                    params,
                )
        return len(rows)

    def gc_applied_tombstones(
        self,
        *,
        tenant_id: str,
        updated_before: str,
        limit: int = 256,
    ) -> int:
        resolved_tenant = self._require_tenant(tenant_id)
        cutoff = self._store._coerce_timestamp(str(updated_before))
        if not cutoff:
            raise ValueError("updated_before must be an ISO-8601 timestamp")
        with self._store._connect() as conn:
            rows = conn.execute(
                "SELECT tombstone_id FROM context_tombstones WHERE tenant_id = ? AND updated_at < ? "
                "AND (status = 'STALE' OR (status = 'APPLIED' "
                "AND json_extract(payload_json, '$.gc_safe') = 1)) "
                "ORDER BY updated_at, tombstone_id LIMIT ?",
                (resolved_tenant, cutoff, self._store._bounded_limit(limit)),
            ).fetchall()
            for row in rows:
                conn.execute(
                    "DELETE FROM context_tombstones WHERE tenant_id = ? AND tombstone_id = ?",
                    (resolved_tenant, str(row["tombstone_id"])),
                )
        return len(rows)

    def delete_catalog(self, record_key: str, *, tenant_id: str) -> bool:
        resolved_tenant = self._require_tenant(tenant_id)
        with self._store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT record_kind FROM contexts WHERE tenant_id = ? AND record_key = ?",
                (resolved_tenant, str(record_key)),
            ).fetchone()
            if existing is not None and str(existing["record_kind"]) in _DOCUMENT_RECORD_KINDS:
                raise ValueError("memory document projections require tombstone_memory_document_projection()")
            return self._delete_catalog_in_transaction(
                conn,
                str(record_key),
                tenant_id=resolved_tenant,
            )

    @staticmethod
    def _tenant_filters(filters: Mapping[str, Any] | None, tenant_id: str) -> dict[str, Any]:
        normalized = dict(filters or {})
        supplied = normalized.get("tenant_id")
        if supplied is not None:
            values = (supplied,) if isinstance(supplied, str) else tuple(supplied)
            if values != (tenant_id,):
                raise ValueError("structured filters cannot cross tenant boundary")
        normalized["tenant_id"] = tenant_id
        return normalized


__all__ = ["CatalogStoreOperations"]
