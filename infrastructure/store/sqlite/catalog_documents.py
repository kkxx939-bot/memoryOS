"""SQLite Catalog 的记忆文档投影事务。"""

from __future__ import annotations

from infrastructure.store.sqlite._common import (
    Any,
    CatalogRecord,
    CatalogRecordKind,
    Mapping,
    Sequence,
)

_DOCUMENT_RECORD_KINDS = frozenset(
    {
        CatalogRecordKind.MEMORY_DOCUMENT.value,
        CatalogRecordKind.MEMORY_BLOCK.value,
    }
)


class CatalogDocumentOperationsMixin:
    """集中处理记忆文档投影的 generation CAS、发布与墓碑事务。"""

    def get_memory_document_projection_state(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> dict[str, Any] | None:
        """读取一个文档不包含正文的服务镜像。"""

        resolved_tenant = self._require_tenant(tenant_id)
        resolved_owner = str(owner_user_id or "").strip()
        resolved_document = str(document_id or "").strip()
        if not resolved_owner or not resolved_document:
            raise ValueError("projection state requires owner_user_id and document_id")
        with self._store._connect() as conn:
            row = conn.execute(
                "SELECT tenant_id, owner_user_id, document_id, relative_path, source_digest, "
                "projection_generation, projection_status, deletion_generation, "
                "deletion_event_digest, deletion_status "
                "FROM memory_document_projection_state WHERE tenant_id = ? "
                "AND owner_user_id = ? AND document_id = ?",
                (resolved_tenant, resolved_owner, resolved_document),
            ).fetchone()
        if row is None:
            return None
        return {
            "tenant_id": str(row["tenant_id"]),
            "owner_user_id": str(row["owner_user_id"]),
            "document_id": str(row["document_id"]),
            "relative_path": str(row["relative_path"]),
            "source_digest": str(row["source_digest"]),
            "projection_generation": int(row["projection_generation"]),
            "projection_status": str(row["projection_status"]),
            "deletion_generation": int(row["deletion_generation"]),
            "deletion_event_digest": str(row["deletion_event_digest"]),
            "deletion_status": str(row["deletion_status"]),
        }

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
        """原子发布一个完整 Markdown 文档投影。

        文档记录、全部块记录、路径、ACL、FTS、投影日志和 generation CAS 状态在
        同一事务提交；返回值是本次发布后失效的记录键。
        """

        resolved_tenant = self._require_tenant(tenant_id)
        resolved_owner = str(owner_user_id or "").strip()
        if not resolved_owner:
            raise ValueError("owner_user_id is required")
        document = self._store._coerce_record(document_record)
        blocks = tuple(self._store._coerce_record(record) for record in block_records)
        if document.record_kind != CatalogRecordKind.MEMORY_DOCUMENT.value:
            raise ValueError("document_record must be a memory_document")
        if any(record.record_kind != CatalogRecordKind.MEMORY_BLOCK.value for record in blocks):
            raise ValueError("block_records must contain only memory_block records")
        projection = (document, *blocks)
        if any(record.tenant_id != resolved_tenant for record in projection):
            raise ValueError("document projection crosses tenant boundary")
        if any(record.owner_user_id != resolved_owner for record in projection):
            raise ValueError("document projection crosses owner boundary")
        if any(record.document_id != document.document_id for record in projection):
            raise ValueError("document projection contains a foreign document_id")
        if any(record.document_kind != document.document_kind for record in projection):
            raise ValueError("document projection contains a foreign document_kind")
        if any(record.projection_generation != document.projection_generation for record in projection):
            raise ValueError("document projection generation must be uniform")
        if any(record.source_digest != document.source_digest for record in projection):
            raise ValueError("document projection source digest must be uniform")
        if document.projection_generation <= 0:
            raise ValueError("projection_generation must be positive")
        record_keys = tuple(record.record_key for record in projection)
        block_ids = tuple(record.block_id for record in blocks)
        if len(record_keys) != len(set(record_keys)):
            raise ValueError("document projection record_key values must be unique")
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("document projection block_id values must be unique")
        relative_path = self._document_relative_path(document.metadata.get("relative_path"))
        prepared = tuple(self._prepare_record(record) for record in projection)

        with self._store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            placeholders = ", ".join("?" for _ in record_keys)
            collision = conn.execute(
                "SELECT record_key FROM contexts WHERE tenant_id = ? "
                f"AND record_key IN ({placeholders}) "
                "AND (owner_user_id <> ? OR document_id <> ?) LIMIT 1",
                (resolved_tenant, *record_keys, resolved_owner, document.document_id),
            ).fetchone()
            if collision is not None:
                raise ValueError("document projection record_key is owned by another document")
            state = conn.execute(
                "SELECT * FROM memory_document_projection_state "
                "WHERE tenant_id = ? AND owner_user_id = ? AND document_id = ?",
                (resolved_tenant, resolved_owner, document.document_id),
            ).fetchone()
            current_generation = int(state["projection_generation"]) if state is not None else 0
            current_digest = str(state["source_digest"]) if state is not None else ""
            expected = 0 if expected_previous_generation is None else int(expected_previous_generation)
            deletion_generation = int(state["deletion_generation"]) if state is not None else 0
            if deletion_generation and document.projection_generation <= deletion_generation:
                raise ValueError("memory document projection is not newer than its deletion barrier")
            deletion_status = str(state["deletion_status"] or "") if state is not None else ""
            if deletion_status == "HARD_ERASED":
                raise ValueError("hard-erased memory document IDs cannot be restored")
            if deletion_status == "SOFT_FORGOTTEN" and not restore_soft_deleted:
                raise ValueError("memory document projection is blocked by its deletion barrier")
            if deletion_status not in {"", "SOFT_FORGOTTEN", "HARD_ERASED"}:
                raise RuntimeError("memory document projection has an invalid deletion barrier")
            if restore_soft_deleted and deletion_status != "SOFT_FORGOTTEN":
                raise ValueError("explicit restore requires a soft-forgotten document")
            if document.projection_generation < current_generation:
                raise ValueError("memory document projection generation regressed")
            if document.projection_generation == current_generation:
                if document.source_digest != current_digest:
                    raise ValueError("memory document projection digest conflicts at the current generation")
                existing = conn.execute(
                    "SELECT record_key, record_kind, block_id, source_digest FROM contexts "
                    "WHERE tenant_id = ? AND owner_user_id = ? AND document_id = ? "
                    "AND record_kind IN (?, ?) ORDER BY record_key",
                    (
                        resolved_tenant,
                        resolved_owner,
                        document.document_id,
                        CatalogRecordKind.MEMORY_DOCUMENT.value,
                        CatalogRecordKind.MEMORY_BLOCK.value,
                    ),
                ).fetchall()
                old_shape = tuple(
                    sorted(
                        (
                            str(row["record_key"]),
                            str(row["record_kind"]),
                            str(row["block_id"]),
                            str(row["source_digest"]),
                        )
                        for row in existing
                    )
                )
                new_shape = tuple(
                    sorted(
                        (record.record_key, record.record_kind, record.block_id, record.source_digest)
                        for record in projection
                    )
                )
                if old_shape != new_shape:
                    raise ValueError("memory document projection shape conflicts at the current generation")
                return ()
            if expected != current_generation:
                raise ValueError("stale memory document projection generation")
            if document.projection_generation <= current_generation:
                raise ValueError("memory document projection generation did not advance")

            old_rows = conn.execute(
                "SELECT record_key FROM contexts WHERE tenant_id = ? AND owner_user_id = ? "
                "AND document_id = ? AND record_kind IN (?, ?) ORDER BY record_key",
                (
                    resolved_tenant,
                    resolved_owner,
                    document.document_id,
                    CatalogRecordKind.MEMORY_DOCUMENT.value,
                    CatalogRecordKind.MEMORY_BLOCK.value,
                ),
            ).fetchall()
            old_keys = tuple(str(row["record_key"]) for row in old_rows)
            for record_key in old_keys:
                self._delete_catalog_in_transaction(conn, record_key, tenant_id=resolved_tenant)
            for item in prepared:
                self._upsert_prepared(conn, item)

            now = document.updated_at or document.transaction_time or self._store._now()
            conn.execute(
                """
                INSERT INTO memory_document_projection_state(
                  tenant_id, owner_user_id, document_id, relative_path, source_digest,
                  projection_generation, projection_status, projected_at, last_error,
                  deletion_generation, deletion_event_digest, deletion_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', 0, '', '')
                ON CONFLICT(tenant_id, owner_user_id, document_id) DO UPDATE SET
                  relative_path=excluded.relative_path,
                  source_digest=excluded.source_digest,
                  projection_generation=excluded.projection_generation,
                  projection_status=excluded.projection_status,
                  projected_at=excluded.projected_at,
                  last_error=''
                """,
                (
                    resolved_tenant,
                    resolved_owner,
                    document.document_id,
                    relative_path,
                    document.source_digest,
                    document.projection_generation,
                    document.projection_status,
                    now,
                ),
            )
            if restore_soft_deleted:
                conn.execute(
                    "UPDATE memory_document_projection_state SET deletion_status = '' "
                    "WHERE tenant_id = ? AND owner_user_id = ? AND document_id = ? "
                    "AND deletion_status = 'SOFT_FORGOTTEN'",
                    (resolved_tenant, resolved_owner, document.document_id),
                )
            conn.execute(
                """
                INSERT INTO context_projection_journal(
                  tenant_id, projector_kind, source_uri, owner_user_id, workspace_id,
                  source_id, source_digest, status, last_error, created_at, updated_at
                ) VALUES (?, 'memory_document', ?, ?, ?, ?, ?, ?, '', ?, ?)
                ON CONFLICT(tenant_id, projector_kind, source_uri) DO UPDATE SET
                  owner_user_id=excluded.owner_user_id,
                  workspace_id=excluded.workspace_id,
                  source_id=excluded.source_id,
                  source_digest=excluded.source_digest,
                  status=excluded.status,
                  last_error='',
                  updated_at=excluded.updated_at
                """,
                (
                    resolved_tenant,
                    document.source_uri or document.uri,
                    resolved_owner,
                    document.workspace_id,
                    document.document_id,
                    document.source_digest,
                    document.projection_status,
                    now,
                    now,
                ),
            )
        new_keys = set(record_keys)
        return tuple(record_key for record_key in old_keys if record_key not in new_keys)

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
        """原子退役一个文档投影并发布对应耐久屏障。"""

        resolved_tenant = self._require_tenant(tenant_id)
        resolved_owner = str(owner_user_id or "").strip()
        resolved_document = str(document_id or "").strip()
        resolved_digest = str(deletion_event_digest or "").strip()
        resolved_status = str(deletion_status or "").strip().upper()
        generation = int(deletion_generation)
        if not resolved_owner or not resolved_document:
            raise ValueError("document tombstone requires owner_user_id and document_id")
        if generation <= 0:
            raise ValueError("document tombstone generation must be positive")
        if not resolved_digest or len(resolved_digest) > 256 or "\x00" in resolved_digest:
            raise ValueError("document tombstone requires a bounded deletion_event_digest")
        if resolved_status not in {"SOFT_FORGOTTEN", "HARD_ERASED"}:
            raise ValueError("document tombstone status must be SOFT_FORGOTTEN or HARD_ERASED")

        with self._store._connect() as conn:
            if resolved_status == "HARD_ERASED":
                conn.execute("PRAGMA secure_delete = ON")
            conn.execute("BEGIN IMMEDIATE")
            state = conn.execute(
                "SELECT * FROM memory_document_projection_state "
                "WHERE tenant_id = ? AND owner_user_id = ? AND document_id = ?",
                (resolved_tenant, resolved_owner, resolved_document),
            ).fetchone()
            rows = conn.execute(
                "SELECT * FROM contexts WHERE tenant_id = ? AND owner_user_id = ? "
                "AND document_id = ? AND record_kind IN (?, ?) ORDER BY record_key",
                (
                    resolved_tenant,
                    resolved_owner,
                    resolved_document,
                    CatalogRecordKind.MEMORY_DOCUMENT.value,
                    CatalogRecordKind.MEMORY_BLOCK.value,
                ),
            ).fetchall()
            serving_generation = max((int(row["projection_generation"]) for row in rows), default=0)
            current_generation = max(
                int(state["projection_generation"]) if state is not None else 0,
                serving_generation,
            )
            if generation < current_generation:
                raise ValueError("stale memory document tombstone generation")

            previous_deletion_generation = int(state["deletion_generation"]) if state is not None else 0
            previous_deletion_digest = str(state["deletion_event_digest"] or "") if state is not None else ""
            previous_deletion_status = str(state["deletion_status"] or "") if state is not None else ""
            if previous_deletion_status == "HARD_ERASED" and (
                resolved_digest != previous_deletion_digest
                or resolved_status != previous_deletion_status
                or generation < previous_deletion_generation
            ):
                raise ValueError("hard-erased memory document barrier is immutable")
            if generation < previous_deletion_generation:
                raise ValueError("stale memory document tombstone generation")
            if generation == previous_deletion_generation and previous_deletion_generation:
                if resolved_digest != previous_deletion_digest or resolved_status != previous_deletion_status:
                    raise ValueError("memory document tombstone conflicts at the current generation")

            state_relative_path = str(state["relative_path"] or "") if state is not None else ""
            supplied_relative_path = (
                self._document_relative_path(relative_path) if str(relative_path or "").strip() else ""
            )
            if state_relative_path and supplied_relative_path and state_relative_path != supplied_relative_path:
                raise ValueError("memory document tombstone relative_path does not match projection state")
            resolved_relative_path = state_relative_path or supplied_relative_path
            if not resolved_relative_path:
                for row in rows:
                    if str(row["record_kind"]) != CatalogRecordKind.MEMORY_DOCUMENT.value:
                        continue
                    metadata_relative_path = self._store._json_mapping(row["metadata_json"]).get("relative_path")
                    if metadata_relative_path:
                        resolved_relative_path = self._document_relative_path(metadata_relative_path)
                        break
            if not resolved_relative_path and resolved_status != "HARD_ERASED":
                raise ValueError("document tombstone requires relative_path when projection state is absent")
            persisted_relative_path = "" if resolved_status == "HARD_ERASED" else resolved_relative_path

            journal_row = conn.execute(
                "SELECT source_uri FROM context_projection_journal WHERE tenant_id = ? "
                "AND projector_kind = 'memory_document' AND owner_user_id = ? "
                "AND source_id = ? ORDER BY source_uri LIMIT 1",
                (resolved_tenant, resolved_owner, resolved_document),
            ).fetchone()
            document_row = next(
                (row for row in rows if str(row["record_kind"]) == CatalogRecordKind.MEMORY_DOCUMENT.value),
                None,
            )
            source_uri = (
                str(document_row["source_uri"] or document_row["uri"])
                if document_row is not None
                else str(journal_row["source_uri"] if journal_row is not None else "")
            )
            if not source_uri:
                source_uri = (
                    f"memoryos://tenants/{resolved_tenant}/users/{resolved_owner}/memory/documents/{resolved_document}"
                )
            obsolete_keys = tuple(str(row["record_key"]) for row in rows)
            for record_key in obsolete_keys:
                self._delete_catalog_in_transaction(conn, record_key, tenant_id=resolved_tenant)

            now = self._store._now()
            conn.execute(
                """
                INSERT INTO memory_document_projection_state(
                  tenant_id, owner_user_id, document_id, relative_path, source_digest,
                  projection_generation, projection_status, projected_at, last_error,
                  deletion_generation, deletion_event_digest, deletion_status
                ) VALUES (?, ?, ?, ?, '', 0, 'TOMBSTONED', ?, '', ?, ?, ?)
                ON CONFLICT(tenant_id, owner_user_id, document_id) DO UPDATE SET
                  relative_path=excluded.relative_path,
                  source_digest='',
                  projection_generation=0,
                  projection_status='TOMBSTONED',
                  projected_at=excluded.projected_at,
                  last_error='',
                  deletion_generation=excluded.deletion_generation,
                  deletion_event_digest=excluded.deletion_event_digest,
                  deletion_status=excluded.deletion_status
                """,
                (
                    resolved_tenant,
                    resolved_owner,
                    resolved_document,
                    persisted_relative_path,
                    now,
                    generation,
                    resolved_digest,
                    resolved_status,
                ),
            )
            conn.execute(
                """
                INSERT INTO context_projection_journal(
                  tenant_id, projector_kind, source_uri, owner_user_id, workspace_id,
                  source_id, source_digest, status, last_error, created_at, updated_at
                ) VALUES (?, 'memory_document', ?, ?, '', ?, ?, 'TOMBSTONED', '', ?, ?)
                ON CONFLICT(tenant_id, projector_kind, source_uri) DO UPDATE SET
                  owner_user_id=excluded.owner_user_id,
                  source_id=excluded.source_id,
                  source_digest=excluded.source_digest,
                  status='TOMBSTONED',
                  last_error='',
                  updated_at=excluded.updated_at
                """,
                (
                    resolved_tenant,
                    source_uri,
                    resolved_owner,
                    resolved_document,
                    resolved_digest,
                    now,
                    now,
                ),
            )
        if resolved_status == "HARD_ERASED":
            with self._store._connect() as conn:
                checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if checkpoint is not None and int(checkpoint[0]) != 0:
                    raise RuntimeError("Catalog WAL checkpoint is busy after hard-erasure tombstone")
        return obsolete_keys

    @staticmethod
    def _document_relative_path(value: Any) -> str:
        relative_path = str(value or "").strip()
        segments = relative_path.split("/")
        if (
            not relative_path
            or len(relative_path) > 1_000
            or relative_path.startswith("/")
            or "\\" in relative_path
            or any(segment in {"", ".", ".."} or "\x00" in segment for segment in segments)
        ):
            raise ValueError("memory document projection requires a normalized relative_path")
        return relative_path


__all__ = ["CatalogDocumentOperationsMixin"]
