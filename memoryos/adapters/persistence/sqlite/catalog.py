"""Tenant-safe operations for the greenfield SQLite Catalog."""

from __future__ import annotations

from memoryos.adapters.persistence.sqlite._common import (
    _CATALOG_SCHEMA_VERSION,
    _CONTEXT_COLUMNS,
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

_DOCUMENT_RECORD_KINDS = frozenset(
    {
        CatalogRecordKind.MEMORY_DOCUMENT.value,
        CatalogRecordKind.MEMORY_BLOCK.value,
    }
)


class CatalogStoreOperations:
    """Own transactional writes and bounded tenant-scoped Catalog reads."""

    def __init__(self, store: Any) -> None:
        self._store = store

    @staticmethod
    def _require_tenant(tenant_id: str) -> str:
        resolved = str(tenant_id or "").strip()
        if not resolved:
            raise ValueError("tenant_id is required")
        return resolved

    def upsert_index(self, obj: ContextObject, content: str = "", *, tenant_id: str) -> None:
        """Project one ordinary ContextObject into its tenant's Catalog."""

        resolved_tenant = self._require_tenant(tenant_id)
        if str(obj.tenant_id or "default") != resolved_tenant:
            raise ValueError("ContextObject tenant does not match tenant_id")
        record = CatalogRecord.from_context_object(obj, content=content)
        if content:
            record = replace(record, l1_text=content)
        self.upsert_catalog(record, tenant_id=resolved_tenant)

    def delete_index(self, uri: str, *, tenant_id: str) -> None:
        """Delete every serving record for one tenant-local logical URI."""

        resolved_tenant = self._require_tenant(tenant_id)
        with self._store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT record_key, record_kind FROM contexts "
                "WHERE tenant_id = ? AND uri = ? ORDER BY record_key",
                (resolved_tenant, str(uri)),
            ).fetchall()
            if any(str(row["record_kind"]) in _DOCUMENT_RECORD_KINDS for row in rows):
                raise ValueError(
                    "memory document projections require tombstone_memory_document_projection()"
                )
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
                "SELECT * FROM contexts WHERE tenant_id = ? AND uri = ? "
                "ORDER BY updated_at DESC, record_key LIMIT 1",
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
        """Resolve ordinary relation endpoint liveness inside one tenant."""

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
                "SELECT 1 FROM context_tombstones WHERE tenant_id = ? AND uri = ? "
                "AND status = 'APPLIED' LIMIT 1",
                (resolved_tenant, safe_uri),
            ).fetchone()
        if retired is not None:
            return "retired"
        return "missing" if row is None else "inactive"

    def clear(self, *, tenant_id: str) -> None:
        """Clear rebuildable serving rows while retaining journals and delete barriers."""

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
        """Filter an offline rebuild batch through durable tenant-local barriers."""

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
        """Atomically sanitize and upsert ordinary Catalog records."""

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
                    raise ValueError(
                        "memory document projections require replace_memory_document_projection()"
                    )
            for item in prepared:
                self._upsert_prepared(conn, item)
        return len(prepared)

    def get_memory_document_projection_state(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> dict[str, Any] | None:
        """Read the content-free serving mirror for one document."""

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
        """Atomically publish one complete Markdown document projection.

        The document row, all block rows, their paths/ACL/FTS rows, projection
        journal, and generation CAS state commit together.  Returned keys are
        records made obsolete by this publication.
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
        """Atomically retire one document projection and publish its durable fence."""

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
                if (
                    resolved_digest != previous_deletion_digest
                    or resolved_status != previous_deletion_status
                ):
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
                (
                    row
                    for row in rows
                    if str(row["record_kind"]) == CatalogRecordKind.MEMORY_DOCUMENT.value
                ),
                None,
            )
            source_uri = (
                str(document_row["source_uri"] or document_row["uri"])
                if document_row is not None
                else str(journal_row["source_uri"] if journal_row is not None else "")
            )
            if not source_uri:
                source_uri = (
                    f"memoryos://tenants/{resolved_tenant}/users/{resolved_owner}/"
                    f"memory/documents/{resolved_document}"
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
        sql = (
            "SELECT c.* FROM contexts AS c WHERE 1=1 "
            f"{filter_sql} ORDER BY c.updated_at DESC, c.record_key LIMIT ?"
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
        sql = (
            "SELECT c.* FROM contexts AS c WHERE c.record_key > ? "
            f"{filter_sql} ORDER BY c.record_key LIMIT ?"
        )
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
                raise ValueError(
                    "memory document projections require tombstone_memory_document_projection()"
                )
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
            scope.get("project_id")
            or fields.get("project_id")
            or metadata.get("project_id")
            or safe.workspace_id
            or ""
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
            f"{column}=excluded.{column}"
            for column in _CONTEXT_COLUMNS
            if column not in {"tenant_id", "record_key"}
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
        now = record.updated_at or record.transaction_time or record.ingested_at or record.created_at or self._store._now()
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
        now = record.updated_at or record.transaction_time or record.ingested_at or record.created_at or self._store._now()
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
        scope = record.metadata.get("scope")
        visibility = scope.get("visibility") if isinstance(scope, Mapping) else None
        if isinstance(visibility, Mapping) and str(visibility.get("tenant_id") or "") == record.tenant_id:
            principals = visibility.get("allowed_principal_ids", ()) or ()
            services = visibility.get("allowed_service_ids", ()) or ()
            for principal_id in principals:
                if isinstance(principal_id, str) and principal_id:
                    grants.add(("principal", principal_id, record.workspace_id))
            for service_id in services:
                if isinstance(service_id, str) and service_id:
                    grants.add(("service", service_id, record.workspace_id))
            if visibility.get("private") is False and not principals and not services:
                grants.add(("tenant", "", record.workspace_id))
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
            "DELETE FROM context_links WHERE tenant_id = ? "
            "AND (source_record_key = ? OR target_record_key = ?)",
            (resolved_tenant, record_key, record_key),
        )
        conn.execute(
            "DELETE FROM contexts WHERE tenant_id = ? AND record_key = ?",
            (resolved_tenant, record_key),
        )
        return exists is not None


__all__ = ["CatalogStoreOperations"]
