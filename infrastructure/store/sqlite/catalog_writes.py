"""SQLite Catalog 的记录规范化与派生索引写入。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from infrastructure.store.sqlite._common import (
    _CONTEXT_COLUMNS,
    _MAX_SCOPE_KEYS_PER_RECORD,
    Any,
    CatalogProjectionStatus,
    CatalogRecord,
    Mapping,
    Sequence,
    _path_ancestors,
    _PreparedCatalogRecord,
    lexical_terms,
    normalize_workspace_id,
    replace,
    sqlite3,
)

if TYPE_CHECKING:
    from infrastructure.store.sqlite.index_store import SQLiteIndexStore


class CatalogWriteOperationsMixin:
    """负责记录清洗、路径、ACL、FTS 与删除事务。"""

    _store: SQLiteIndexStore

    @staticmethod
    def _require_tenant(tenant_id: str) -> str:
        resolved = str(tenant_id or "").strip()
        if not resolved:
            raise ValueError("tenant_id is required")
        return resolved

    def _prepare_record(
        self,
        record: CatalogRecord,
        *,
        scope_keys_override: Sequence[str] | None = None,
        value_overrides: Mapping[str, Any] | None = None,
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
        project_id = normalize_workspace_id(
            scope.get("project_id") or fields.get("project_id") or metadata.get("project_id") or safe.workspace_id or ""
        )
        values: dict[str, Any] = {
            "tenant_id": safe.tenant_id,
            "record_key": safe.record_key,
            "uri": safe.uri,
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
            "title": safe.title,
            "l0_text": safe.l0_text,
            "l1_text": safe.l1_text,
            "l2_uri": safe.l2_uri,
            "source_uri": safe.source_uri,
            "source_digest": safe.source_digest,
            "source_revision": int(safe.source_revision),
            "document_id": safe.document_id,
            "block_id": safe.block_id,
            "document_kind": safe.document_kind,
            "document_revision": int(safe.document_revision),
            "projection_generation": int(safe.projection_generation),
            "projection_effect_hash": safe.projection_effect_hash,
            "hotness": safe.hotness,
            "semantic_hotness": safe.semantic_hotness,
            "behavior_support_hotness": safe.behavior_support_hotness,
            "serving_tier": safe.serving_tier,
            "projection_status": safe.projection_status,
            "metadata_json": self._store._json_dump(metadata),
            "content_digest": self._store._content_digest(record.l1_text),
            "stored_content_digest": self._store._content_digest(safe.l1_text),
            "content_text": safe.l1_text,
            "scene_key": self._store._safe_exact_value(metadata.get("scene_key")),
            "action": self._store._safe_exact_value(metadata.get("action")),
            "support_anchor_uri": self._store._safe_exact_value(metadata.get("support_anchor_uri")),
        }
        for key, value in dict(value_overrides or {}).items():
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
        updates = ", ".join(
            f"{column}=excluded.{column}" for column in _CONTEXT_COLUMNS if column not in {"tenant_id", "record_key"}
        )
        conn.execute(
            f"INSERT INTO contexts({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(tenant_id, record_key) DO UPDATE SET {updates}",
            tuple(item.values[column] for column in _CONTEXT_COLUMNS),
        )
        self._replace_paths(conn, item.record, scope_signature=item.scope_signature)
        self._replace_acl_grants(conn, item.record, scope_signature=item.scope_signature)
        self._replace_fts(conn, item)
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
        for table in ("context_path_acl", "context_path_closure", "context_paths"):
            conn.execute(
                f"DELETE FROM {table} WHERE tenant_id = ? AND record_key = ?",
                (record.tenant_id, record.record_key),
            )
        now = (
            record.updated_at
            or record.transaction_time
            or record.ingested_at
            or record.created_at
            or self._store._now()
        )
        acl_grants = self._acl_grants_for_record(record)
        for index, path in enumerate(record.tree_paths):
            conn.execute(
                """
                INSERT INTO context_paths(
                  tenant_id, record_key, uri, owner_user_id, workspace_id, workspace_shared,
                  context_type, record_kind, document_id, document_kind, event_time,
                  transaction_time, path, path_kind, depth, is_primary, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    record.document_id,
                    record.document_kind,
                    record.event_time,
                    record.transaction_time,
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
                      uri, context_type, source_kind, record_kind, adapter_id,
                      adapter_access_id, session_id, document_id, document_kind,
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
                        self._adapter_access_value(record),
                        record.session_id,
                        record.document_id,
                        record.document_kind,
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
                            self._adapter_access_value(record),
                            record.session_id,
                            record.event_time,
                            record.transaction_time,
                            now,
                        ),
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
        now = (
            record.updated_at
            or record.transaction_time
            or record.ingested_at
            or record.created_at
            or self._store._now()
        )
        for grant_kind, grant_id, workspace_id in sorted(self._acl_grants_for_record(record)):
            conn.execute(
                """
                INSERT INTO context_acl_grants(
                  tenant_id, record_key, grant_kind, grant_id, workspace_id,
                  scope_signature, uri, context_type, source_kind, record_kind,
                  adapter_id, adapter_access_id, session_id, event_time,
                  transaction_time, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
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
                    self._adapter_access_value(record),
                    record.session_id,
                    record.event_time,
                    record.transaction_time,
                    now,
                ),
            )

    @staticmethod
    def _acl_grants_for_record(record: CatalogRecord) -> set[tuple[str, str, str]]:
        grants: set[tuple[str, str, str]] = set()
        if record.owner_user_id:
            grants.add(("principal", record.owner_user_id, record.workspace_id))
        elif record.context_type in {"resource", "skill"}:
            grants.add(("public", "", record.workspace_id))
        if record.workspace_shared and record.workspace_id:
            grants.add(("workspace", record.workspace_id, record.workspace_id))
        return grants

    @staticmethod
    def _adapter_access_value(record: CatalogRecord) -> str:
        if not record.adapter_id or record.context_type in {"session", "resource", "skill"}:
            return "*"
        return record.adapter_id

    def _replace_fts(self, conn: sqlite3.Connection, item: _PreparedCatalogRecord) -> None:
        self._store._delete_fts_record(conn, item.record.tenant_id, item.record.record_key)
        if item.record.projection_status not in {
            CatalogProjectionStatus.PROJECTED.value,
            CatalogProjectionStatus.DEGRADED.value,
        }:
            return
        if item.record.lifecycle_state in {"deleted", "archived", "obsolete"}:
            return
        cursor = conn.execute(
            "INSERT INTO contexts_fts(tenant_id, record_key, uri, title, content_text, "
            "metadata_text, search_terms, acl_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item.record.tenant_id,
                item.record.record_key,
                item.record.uri,
                item.record.title,
                item.record.l1_text,
                item.fts_metadata_text,
                item.fts_search_terms,
                self._store._fts_acl_tokens(item.record, scope_signature=item.scope_signature),
            ),
        )
        if cursor.lastrowid is None or int(cursor.lastrowid) <= 0:
            raise RuntimeError("FTS insert did not return a durable rowid")
        conn.execute(
            "INSERT INTO context_fts_map(tenant_id, record_key, fts_rowid) VALUES (?, ?, ?)",
            (item.record.tenant_id, item.record.record_key, int(cursor.lastrowid)),
        )

    def _delete_catalog_in_transaction(
        self,
        conn: sqlite3.Connection,
        record_key: str,
        *,
        tenant_id: str,
    ) -> bool:
        resolved_tenant = self._require_tenant(tenant_id)
        exists = conn.execute(
            "SELECT 1 FROM contexts WHERE tenant_id = ? AND record_key = ?",
            (resolved_tenant, record_key),
        ).fetchone()
        self._store._delete_fts_record(conn, resolved_tenant, record_key)
        for table in (
            "context_acl_grants",
            "context_path_acl",
            "context_path_closure",
            "context_paths",
            "context_projection_state",
        ):
            conn.execute(
                f"DELETE FROM {table} WHERE tenant_id = ? AND record_key = ?",
                (resolved_tenant, record_key),
            )
        conn.execute(
            "DELETE FROM context_links WHERE tenant_id = ? AND (source_record_key = ? OR target_record_key = ?)",
            (resolved_tenant, record_key, record_key),
        )
        conn.execute(
            "DELETE FROM contexts WHERE tenant_id = ? AND record_key = ?",
            (resolved_tenant, record_key),
        )
        return exists is not None


__all__ = ["CatalogWriteOperationsMixin"]
