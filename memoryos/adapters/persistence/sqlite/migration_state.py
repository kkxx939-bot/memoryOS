"""SQLite catalog MigrationStateOperations responsibility component."""

from __future__ import annotations

from memoryos.adapters.persistence.sqlite._common import (
    _CATALOG_SCHEMA_VERSION,
    _GREENFIELD_CATALOG_ORIGIN_NAME,
    _MAX_FILTER_VALUES,
    _MAX_QUERY_LIMIT,
    _MIGRATION_STATES,
    _SCHEMA_UPGRADE_BOOTSTRAP_TENANT,
    Any,
    CatalogCandidateBoundExceeded,
    Mapping,
    Sequence,
    hashlib,
    normalize_workspace_id,
    sqlite3,
)


class MigrationStateOperations:
    """Own one stable subset of SQLite catalog behavior."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def set_migration_state(
        self,
        migration_name: str,
        state: str,
        checkpoint: str = "",
        details: Mapping[str, Any] | None = None,
        *,
        tenant_id: str = "",
        batch_size: int = 0,
        error: str = "",
    ) -> dict[str, Any]:
        normalized = str(state).upper()
        if normalized not in _MIGRATION_STATES:
            raise ValueError(f"unsupported migration state: {state}")
        if not migration_name:
            raise ValueError("migration_name is required")
        safe_checkpoint = str(self._store.sanitizer.sanitize_trace(str(checkpoint or "")))
        safe_details = self._store.sanitizer.sanitize_trace(dict(details or {}))
        safe_error = self._store.sanitizer.sanitize_trace(str(error or ""))
        with self._store._connect() as conn:
            conn.execute(
                """
                INSERT INTO migration_state(
                  migration_name, tenant_id, state, checkpoint, batch_size, details_json, last_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(migration_name, tenant_id) DO UPDATE SET
                  state=excluded.state,
                  checkpoint=excluded.checkpoint,
                  batch_size=excluded.batch_size,
                  details_json=excluded.details_json,
                  last_error=excluded.last_error,
                  updated_at=excluded.updated_at
                """,
                (
                    str(migration_name),
                    str(tenant_id),
                    normalized,
                    safe_checkpoint,
                    max(0, int(batch_size)),
                    self._store._json_dump(safe_details),
                    str(safe_error),
                    self._store._now(),
                ),
            )
        result = self._store.get_migration_state(migration_name, tenant_id=tenant_id)
        if result is None:  # pragma: no cover - the transaction above either commits or raises.
            raise RuntimeError("migration state write did not persist")
        return result

    def get_migration_state(self, migration_name: str, *, tenant_id: str = "") -> dict[str, Any] | None:
        with self._store._connect() as conn:
            row = conn.execute(
                "SELECT * FROM migration_state WHERE migration_name = ? AND tenant_id = ?",
                (str(migration_name), str(tenant_id)),
            ).fetchone()
        if row is None:
            return None
        return self._store._row_dict(row, json_fields=("details_json",))

    def initialize_migration_state_if_absent(
        self,
        migration_name: str,
        state: str,
        details: Mapping[str, Any] | None = None,
        *,
        tenant_id: str,
        batch_size: int = 0,
    ) -> dict[str, Any]:
        """Create a durable migration gate without overwriting live progress."""

        normalized = str(state).upper()
        if normalized not in _MIGRATION_STATES:
            raise ValueError(f"unsupported migration state: {state}")
        if not migration_name:
            raise ValueError("migration_name is required")
        if not tenant_id:
            raise ValueError("tenant_id is required")
        safe_details = self._store.sanitizer.sanitize_trace(dict(details or {}))
        with self._store._connect() as conn:
            conn.execute(
                """
                INSERT INTO migration_state(
                  migration_name, tenant_id, state, checkpoint, batch_size,
                  details_json, last_error, updated_at
                ) VALUES (?, ?, ?, '', ?, ?, '', ?)
                ON CONFLICT(migration_name, tenant_id) DO NOTHING
                """,
                (
                    str(migration_name),
                    str(tenant_id),
                    normalized,
                    max(0, int(batch_size)),
                    self._store._json_dump(safe_details),
                    self._store._now(),
                ),
            )
        result = self._store.get_migration_state(migration_name, tenant_id=tenant_id)
        if result is None:  # pragma: no cover - INSERT or an existing row must win.
            raise RuntimeError("migration state initialization did not persist")
        return result

    def record_greenfield_catalog_origin(self, *, tenant_id: str) -> dict[str, Any]:
        """Durably distinguish a safe empty first start from an unbackfilled restart."""

        return self._store.initialize_migration_state_if_absent(
            _GREENFIELD_CATALOG_ORIGIN_NAME,
            "COMPLETED",
            {
                "schema_version": _CATALOG_SCHEMA_VERSION,
                "greenfield": True,
            },
            tenant_id=tenant_id,
        )

    def has_greenfield_catalog_origin(self, *, tenant_id: str) -> bool:
        return (
            self._store.get_migration_state(
                _GREENFIELD_CATALOG_ORIGIN_NAME,
                tenant_id=tenant_id,
            )
            is not None
        )

    def bind_migration_tenant_from_schema_upgrade(
        self,
        migration_name: str,
        *,
        tenant_id: str,
        batch_size: int = 0,
    ) -> dict[str, Any] | None:
        """Atomically materialize an upgraded-database bootstrap for one tenant.

        The index database is opened before the runtime tenant migration
        coordinator is constructed.  A reserved durable row therefore records
        that the schema came from a pre-v10 database, while this method copies
        it to the concrete tenant without overwriting a concurrently advanced
        migration state.
        """

        if not migration_name:
            raise ValueError("migration_name is required")
        if not tenant_id:
            raise ValueError("tenant_id is required")
        with self._store._connect() as conn:
            conn.execute(
                """
                INSERT INTO migration_state(
                  migration_name, tenant_id, state, checkpoint, batch_size,
                  details_json, last_error, updated_at
                )
                SELECT migration_name, ?, state, checkpoint,
                       CASE WHEN ? > 0 THEN ? ELSE batch_size END,
                       details_json, last_error, ?
                FROM migration_state
                WHERE migration_name = ? AND tenant_id = ?
                ON CONFLICT(migration_name, tenant_id) DO NOTHING
                """,
                (
                    str(tenant_id),
                    int(batch_size),
                    int(batch_size),
                    self._store._now(),
                    str(migration_name),
                    _SCHEMA_UPGRADE_BOOTSTRAP_TENANT,
                ),
            )
        return self._store.get_migration_state(migration_name, tenant_id=tenant_id)

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
        """Persist the SessionArchive -> Catalog projection substate."""

        normalized = str(status).upper()
        if normalized not in {"PENDING", "PROJECTED", "FAILED", "ABANDONED"}:
            raise ValueError("unsupported Session projection frontier status")
        if not tenant_id or not archive_uri or not session_id:
            raise ValueError("Session projection frontier identity is incomplete")
        safe_error = str(self._store.sanitizer.sanitize_trace(str(error or "")))
        uri_owner = self._store._owner_from_session_archive_uri(archive_uri)
        safe_owner = str(owner_user_id or uri_owner)
        if not safe_owner or (uri_owner and safe_owner != uri_owner):
            raise ValueError("Session projection frontier owner does not match archive URI")
        safe_workspace = normalize_workspace_id(workspace_id) if workspace_id else ""
        now = self._store._now()
        with self._store._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_projection_frontier(
                  tenant_id, archive_uri, owner_user_id, workspace_id,
                  session_id, manifest_digest, status, last_error,
                  created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, archive_uri) DO UPDATE SET
                  owner_user_id=excluded.owner_user_id,
                  workspace_id=excluded.workspace_id,
                  session_id=excluded.session_id,
                  manifest_digest=excluded.manifest_digest,
                  status=excluded.status,
                  last_error=excluded.last_error,
                  updated_at=excluded.updated_at
                """,
                (
                    str(tenant_id),
                    str(archive_uri),
                    safe_owner,
                    safe_workspace,
                    str(session_id),
                    str(manifest_digest or ""),
                    normalized,
                    safe_error,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM session_projection_frontier WHERE tenant_id = ? AND archive_uri = ?",
                (str(tenant_id), str(archive_uri)),
            ).fetchone()
        if row is None:  # pragma: no cover - write transaction either persists or raises.
            raise RuntimeError("Session projection frontier write did not persist")
        return self._store._row_dict(row)

    def get_session_projection_frontier_summary(
        self,
        *,
        tenant_id: str,
        owner_user_id: str | None = None,
        workspace_ids: Sequence[str] | None = None,
    ) -> dict[str, int]:
        if not tenant_id:
            raise ValueError("tenant_id is required for Session projection frontier")
        where = ["tenant_id = ?"]
        params: list[Any] = [str(tenant_id)]
        if owner_user_id is not None:
            where.append("owner_user_id = ?")
            params.append(str(owner_user_id))
        if workspace_ids is not None:
            normalized = tuple(dict.fromkeys(normalize_workspace_id(item) if item else "" for item in workspace_ids))
            if not normalized:
                return {}
            if len(normalized) > _MAX_FILTER_VALUES:
                raise CatalogCandidateBoundExceeded("too many Session frontier workspace filters")
            where.append(f"workspace_id IN ({','.join('?' for _ in normalized)})")
            params.extend(normalized)
        with self._store._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM session_projection_frontier "
                f"WHERE {' AND '.join(where)} GROUP BY status",
                params,
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    @staticmethod
    def _owner_from_session_archive_uri(archive_uri: str) -> str:
        prefix = "memoryos://user/"
        text = str(archive_uri)
        if not text.startswith(prefix):
            return ""
        return text[len(prefix) :].split("/", 1)[0]

    def list_session_projection_frontier(
        self,
        *,
        tenant_id: str,
        statuses: Sequence[str] = ("PENDING", "FAILED"),
        after_archive_uri: str = "",
        limit: int = 256,
    ) -> list[dict[str, Any]]:
        """Read a bounded, tenant-keyset page for startup/repair replay."""

        if not tenant_id:
            raise ValueError("tenant_id is required for Session projection frontier")
        normalized = tuple(dict.fromkeys(str(item).upper() for item in statuses))
        if not normalized or any(item not in {"PENDING", "PROJECTED", "FAILED", "ABANDONED"} for item in normalized):
            raise ValueError("unsupported Session projection frontier status filter")
        placeholders = ",".join("?" for _ in normalized)
        with self._store._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM session_projection_frontier WHERE tenant_id = ? "
                f"AND status IN ({placeholders}) AND archive_uri > ? "
                "ORDER BY archive_uri LIMIT ?",
                (
                    str(tenant_id),
                    *normalized,
                    str(after_archive_uri),
                    self._store._bounded_limit(limit),
                ),
            ).fetchall()
        return [self._store._row_dict(row) for row in rows]

    def record_migration_equivalence_proof(
        self,
        migration_name: str,
        proof: Mapping[str, Any],
        *,
        tenant_id: str = "",
    ) -> dict[str, Any]:
        """Append one idempotent, payload-free source-to-Catalog proof."""

        required_text = (
            "plane",
            "source_identity_digest",
            "evidence_digest",
            "expected_digest",
            "actual_digest",
        )
        safe = self._store.sanitizer.sanitize_trace(dict(proof))
        if not isinstance(safe, Mapping) or any(not str(safe.get(key) or "") for key in required_text):
            raise ValueError("migration equivalence proof is incomplete")
        expected_count = int(safe.get("expected_count") or 0)
        actual_count = int(safe.get("actual_count") or 0)
        if not 0 <= expected_count <= 1_000 or not 0 <= actual_count <= 1_001:
            raise ValueError("migration equivalence proof count is outside its hard bound")
        matched = bool(safe.get("matched")) and not bool(safe.get("overflow"))
        with self._store._connect() as conn:
            row = conn.execute(
                "SELECT * FROM migration_state WHERE migration_name = ? AND tenant_id = ?",
                (str(migration_name), str(tenant_id)),
            ).fetchone()
            if row is None:
                raise ValueError("migration equivalence proof has no migration state")
            details = self._store._json_mapping(row["details_json"])
            state = str(row["state"])
            epoch = str(details.get("shadow_validation_epoch") or "") if state == "SHADOW_VALIDATING" else ""
            proof_identity = {
                "migration_name": migration_name,
                "tenant_id": tenant_id,
                "state": state,
                "validation_epoch": epoch,
                **{key: safe.get(key) for key in required_text},
                "expected_count": expected_count,
                "actual_count": actual_count,
                "matched": matched,
            }
            proof_id = hashlib.sha256(self._store._json_dump(proof_identity).encode("utf-8")).hexdigest()
            created_at = self._store._now()
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO migration_equivalence_journal(
                  proof_id, migration_name, tenant_id, validation_epoch, plane,
                  source_identity_digest, evidence_digest, expected_count, actual_count,
                  expected_digest, actual_digest, matched, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proof_id,
                    str(migration_name),
                    str(tenant_id),
                    epoch,
                    str(safe["plane"]),
                    str(safe["source_identity_digest"]),
                    str(safe["evidence_digest"]),
                    expected_count,
                    actual_count,
                    str(safe["expected_digest"]),
                    str(safe["actual_digest"]),
                    1 if matched else 0,
                    created_at,
                ),
            )
            inserted = cursor.rowcount == 1
            if inserted:
                details["equivalence_proof_count"] = int(details.get("equivalence_proof_count") or 0) + 1
                details["equivalence_mismatch_count"] = int(details.get("equivalence_mismatch_count") or 0) + (
                    0 if matched else 1
                )
                details["last_equivalence_proof"] = {
                    "proof_id": proof_id,
                    "plane": str(safe["plane"]),
                    "expected_count": expected_count,
                    "actual_count": actual_count,
                    "matched": matched,
                }
                if epoch:
                    summary = self._store._migration_equivalence_summary_in_transaction(
                        conn,
                        migration_name=str(migration_name),
                        tenant_id=str(tenant_id),
                        validation_epoch=epoch,
                    )
                    details["shadow_sample_count"] = summary["sample_count"]
                    details["shadow_mismatch_count"] = summary["mismatch_count"]
                conn.execute(
                    "UPDATE migration_state SET details_json = ?, updated_at = ? "
                    "WHERE migration_name = ? AND tenant_id = ?",
                    (
                        self._store._json_dump(details),
                        created_at,
                        str(migration_name),
                        str(tenant_id),
                    ),
                )
        return {
            "proof_id": proof_id,
            "inserted": inserted,
            "matched": matched,
            "validation_epoch": epoch,
        }

    def get_migration_equivalence_summary(
        self,
        migration_name: str,
        *,
        tenant_id: str = "",
        validation_epoch: str,
    ) -> dict[str, int]:
        if not validation_epoch:
            return {"sample_count": 0, "mismatch_count": 0}
        with self._store._connect() as conn:
            return self._store._migration_equivalence_summary_in_transaction(
                conn,
                migration_name=str(migration_name),
                tenant_id=str(tenant_id),
                validation_epoch=str(validation_epoch),
            )

    def list_migration_equivalence_proofs(
        self,
        migration_name: str,
        *,
        tenant_id: str = "",
        validation_epoch: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        bounded = self._store._bounded_limit(limit)
        sql = "SELECT * FROM migration_equivalence_journal WHERE migration_name = ? AND tenant_id = ?"
        params: list[Any] = [str(migration_name), str(tenant_id)]
        if validation_epoch is not None:
            sql += " AND validation_epoch = ?"
            params.append(str(validation_epoch))
        sql += " ORDER BY created_at, proof_id LIMIT ?"
        params.append(bounded)
        with self._store._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._store._row_dict(row) for row in rows]

    def record_migration_shadow_read(
        self,
        migration_name: str,
        comparison: Mapping[str, Any],
        *,
        tenant_id: str = "",
    ) -> dict[str, Any]:
        """Durably record one payload-free old/new bounded read comparison."""

        safe = self._store.sanitizer.sanitize_trace(dict(comparison))
        required = ("plan_digest", "legacy_digest", "unified_digest")
        if not isinstance(safe, Mapping) or any(not str(safe.get(key) or "") for key in required):
            raise ValueError("migration shadow read comparison is incomplete")
        legacy_count = int(safe.get("legacy_count") or 0)
        unified_count = int(safe.get("unified_count") or 0)
        overlap_count = int(safe.get("overlap_count") or 0)
        if not all(0 <= value <= _MAX_QUERY_LIMIT for value in (legacy_count, unified_count, overlap_count)):
            raise ValueError("migration shadow read count is outside its hard bound")
        if overlap_count > min(legacy_count, unified_count):
            raise ValueError("migration shadow read overlap exceeds result counts")
        matched = bool(
            legacy_count == unified_count
            and overlap_count == legacy_count
            and str(safe["legacy_digest"]) == str(safe["unified_digest"])
        )
        with self._store._connect() as conn:
            state_row = conn.execute(
                "SELECT * FROM migration_state WHERE migration_name = ? AND tenant_id = ?",
                (str(migration_name), str(tenant_id)),
            ).fetchone()
            if state_row is None or str(state_row["state"]) != "SHADOW_VALIDATING":
                raise ValueError("shadow read comparison requires SHADOW_VALIDATING state")
            details = self._store._json_mapping(state_row["details_json"])
            epoch = str(details.get("shadow_validation_epoch") or "")
            if not epoch:
                raise ValueError("shadow read comparison has no validation epoch")
            identity = {
                "migration_name": str(migration_name),
                "tenant_id": str(tenant_id),
                "validation_epoch": epoch,
                **{key: str(safe[key]) for key in required},
                "legacy_count": legacy_count,
                "unified_count": unified_count,
                "overlap_count": overlap_count,
                "matched": matched,
            }
            comparison_id = hashlib.sha256(self._store._json_dump(identity).encode("utf-8")).hexdigest()
            now = self._store._now()
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO migration_shadow_read_journal(
                  comparison_id, migration_name, tenant_id, validation_epoch,
                  plan_digest, legacy_count, unified_count, overlap_count,
                  legacy_digest, unified_digest, matched, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    comparison_id,
                    str(migration_name),
                    str(tenant_id),
                    epoch,
                    str(safe["plan_digest"]),
                    legacy_count,
                    unified_count,
                    overlap_count,
                    str(safe["legacy_digest"]),
                    str(safe["unified_digest"]),
                    1 if matched else 0,
                    now,
                ),
            )
            inserted = cursor.rowcount == 1
            if inserted:
                details["shadow_read_sample_count"] = int(details.get("shadow_read_sample_count") or 0) + 1
                details["shadow_read_mismatch_count"] = int(details.get("shadow_read_mismatch_count") or 0) + (
                    0 if matched else 1
                )
                details["last_shadow_read"] = {
                    "comparison_id": comparison_id,
                    "plan_digest": str(safe["plan_digest"]),
                    "legacy_count": legacy_count,
                    "unified_count": unified_count,
                    "overlap_count": overlap_count,
                    "matched": matched,
                }
                conn.execute(
                    "UPDATE migration_state SET details_json = ?, updated_at = ? "
                    "WHERE migration_name = ? AND tenant_id = ?",
                    (
                        self._store._json_dump(details),
                        now,
                        str(migration_name),
                        str(tenant_id),
                    ),
                )
        return {
            "comparison_id": comparison_id,
            "inserted": inserted,
            "matched": matched,
            "validation_epoch": epoch,
        }

    def get_migration_shadow_read_summary(
        self,
        migration_name: str,
        *,
        tenant_id: str = "",
        validation_epoch: str,
    ) -> dict[str, int]:
        if not validation_epoch:
            return {"sample_count": 0, "mismatch_count": 0}
        with self._store._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS sample_count, "
                "COALESCE(SUM(CASE WHEN matched = 0 THEN 1 ELSE 0 END), 0) AS mismatch_count "
                "FROM migration_shadow_read_journal WHERE migration_name = ? "
                "AND tenant_id = ? AND validation_epoch = ?",
                (str(migration_name), str(tenant_id), str(validation_epoch)),
            ).fetchone()
        return {
            "sample_count": int(row["sample_count"]),
            "mismatch_count": int(row["mismatch_count"]),
        }

    def list_migration_shadow_reads(
        self,
        migration_name: str,
        *,
        tenant_id: str = "",
        validation_epoch: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM migration_shadow_read_journal WHERE migration_name = ? AND tenant_id = ?"
        params: list[Any] = [str(migration_name), str(tenant_id)]
        if validation_epoch is not None:
            sql += " AND validation_epoch = ?"
            params.append(str(validation_epoch))
        sql += " ORDER BY created_at, comparison_id LIMIT ?"
        params.append(self._store._bounded_limit(limit))
        with self._store._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._store._row_dict(row) for row in rows]

    @staticmethod
    def _migration_equivalence_summary_in_transaction(
        conn: sqlite3.Connection,
        *,
        migration_name: str,
        tenant_id: str,
        validation_epoch: str,
    ) -> dict[str, int]:
        row = conn.execute(
            "SELECT COUNT(*) AS sample_count, "
            "COALESCE(SUM(CASE WHEN matched = 0 THEN 1 ELSE 0 END), 0) AS mismatch_count "
            "FROM migration_equivalence_journal WHERE migration_name = ? AND tenant_id = ? "
            "AND validation_epoch = ?",
            (migration_name, tenant_id, validation_epoch),
        ).fetchone()
        return {
            "sample_count": int(row["sample_count"]),
            "mismatch_count": int(row["mismatch_count"]),
        }

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
        if not all((tenant_id, source_record_key, source_uri, relation_type, target_uri)):
            raise ValueError("context link identity fields are required")
        safe_metadata = self._store.sanitizer.sanitize_trace(dict(metadata or {}))
        identity = {
            "tenant_id": tenant_id,
            "source_record_key": source_record_key,
            "relation_type": relation_type,
            "target_record_key": target_record_key,
            "target_uri": target_uri,
        }
        stable_key = link_key or self._store.sanitizer.digest(identity)
        now = self._store._now()
        with self._store._connect() as conn:
            conn.execute(
                """
                INSERT INTO context_links(
                  link_key, tenant_id, source_record_key, source_uri, relation_type,
                  target_record_key, target_uri, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(link_key) DO UPDATE SET
                  target_record_key=excluded.target_record_key,
                  target_uri=excluded.target_uri,
                  metadata_json=excluded.metadata_json,
                  updated_at=excluded.updated_at
                """,
                (
                    stable_key,
                    str(tenant_id),
                    str(source_record_key),
                    self._store._safe_reference_uri(str(source_uri)),
                    str(relation_type),
                    str(target_record_key),
                    self._store._safe_reference_uri(str(target_uri)),
                    self._store._json_dump(safe_metadata),
                    now,
                    now,
                ),
            )
        return stable_key


__all__ = ["MigrationStateOperations"]
