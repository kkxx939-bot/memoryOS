"""Tenant-safe durable Catalog tombstone operations."""

from __future__ import annotations

from memoryos.adapters.persistence.sqlite._common import (
    _MAX_FILTER_VALUES,
    Any,
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


class TombstoneOperations:
    """Own the replayable delete journal for Catalog serving rows."""

    def __init__(self, store: Any) -> None:
        self._store = store

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
        queued = self.enqueue_tombstone(
            tenant_id=tenant_id,
            record_key=record_key,
            reason=reason,
            uri=uri,
            source_revision=source_revision,
            tombstone_id=tombstone_id,
            payload=payload,
        )
        applied = self.mark_tombstone_applied(
            str(queued["tombstone_id"]),
            tenant_id=tenant_id,
        )
        if applied is None:  # pragma: no cover
            raise RuntimeError("tombstone disappeared before application")
        return applied

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
        resolved_tenant = self._store._catalog._require_tenant(tenant_id)
        if not record_key or not reason:
            raise ValueError("record_key and reason are required")
        safe_payload = self._store.sanitizer.sanitize_trace(dict(payload or {}))
        safe_reason = str(self._store.sanitizer.sanitize_trace(str(reason)))
        identity = {
            "tenant_id": resolved_tenant,
            "record_key": str(record_key),
            "reason": safe_reason,
            "source_revision": int(source_revision),
        }
        stable_id = str(tombstone_id or self._store.sanitizer.digest(identity))
        now = self._store._now()
        with self._store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM context_tombstones WHERE tenant_id = ? AND tombstone_id = ?",
                (resolved_tenant, stable_id),
            ).fetchone()
            if existing is not None:
                immutable = (
                    str(existing["record_key"]),
                    str(existing["reason"]),
                    int(existing["source_revision"]),
                )
                requested = (str(record_key), safe_reason, int(source_revision))
                if immutable != requested:
                    raise ValueError("tombstone_id is bound to a different immutable identity")
            current = conn.execute(
                "SELECT uri, record_kind FROM contexts WHERE tenant_id = ? AND record_key = ?",
                (resolved_tenant, str(record_key)),
            ).fetchone()
            if current is not None and str(current["record_kind"]) in _DOCUMENT_RECORD_KINDS:
                raise ValueError(
                    "memory document projections require tombstone_memory_document_projection()"
                )
            effective_uri = self._store._safe_reference_uri(
                str(uri or (current["uri"] if current is not None else ""))
            )
            conn.execute(
                """
                INSERT INTO context_tombstones(
                  tenant_id, tombstone_id, record_key, uri, reason, source_revision,
                  status, payload_json, created_at, updated_at, retry_count, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, 0, '')
                ON CONFLICT(tenant_id, tombstone_id) DO UPDATE SET
                  payload_json=excluded.payload_json,
                  updated_at=excluded.updated_at
                """,
                (
                    resolved_tenant,
                    stable_id,
                    str(record_key),
                    effective_uri,
                    safe_reason,
                    int(source_revision),
                    self._store._json_dump(safe_payload),
                    now,
                    now,
                ),
            )
        return {
            "tombstone_id": stable_id,
            "tenant_id": resolved_tenant,
            "record_key": str(record_key),
            "uri": effective_uri,
            "status": str(existing["status"]) if existing is not None else "PENDING",
            "source_revision": int(source_revision),
        }

    def get_pending_tombstones(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        resolved_tenant = self._store._catalog._require_tenant(tenant_id)
        with self._store._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM context_tombstones WHERE tenant_id = ? "
                "AND status IN ('PENDING', 'FAILED', 'CLEANING') "
                "ORDER BY updated_at, tombstone_id LIMIT ?",
                (resolved_tenant, self._store._bounded_limit(limit)),
            ).fetchall()
        return [self._store._row_dict(row, json_fields=("payload_json",)) for row in rows]

    def get_pending_tombstones_for_uri(
        self,
        uri: str,
        *,
        tenant_id: str,
        after_tombstone_id: str = "",
        limit: int = 1_000,
    ) -> list[dict[str, Any]]:
        resolved_tenant = self._store._catalog._require_tenant(tenant_id)
        with self._store._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM context_tombstones WHERE tenant_id = ? AND uri = ? "
                "AND status IN ('PENDING', 'FAILED', 'CLEANING') AND tombstone_id > ? "
                "ORDER BY tombstone_id LIMIT ?",
                (
                    resolved_tenant,
                    self._store._safe_reference_uri(str(uri)),
                    str(after_tombstone_id),
                    self._store._bounded_limit(limit),
                ),
            ).fetchall()
        return [self._store._row_dict(row, json_fields=("payload_json",)) for row in rows]

    def get_tombstones(
        self,
        tombstone_ids: Sequence[str],
        *,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        resolved_tenant = self._store._catalog._require_tenant(tenant_id)
        ordered_ids = tuple(dict.fromkeys(str(item) for item in tombstone_ids if str(item)))
        if not ordered_ids:
            return []
        by_id: dict[str, dict[str, Any]] = {}
        with self._store._connect() as conn:
            for offset in range(0, len(ordered_ids), _MAX_FILTER_VALUES):
                chunk = ordered_ids[offset : offset + _MAX_FILTER_VALUES]
                placeholders = ", ".join("?" for _ in chunk)
                rows = conn.execute(
                    "SELECT * FROM context_tombstones WHERE tenant_id = ? "
                    f"AND tombstone_id IN ({placeholders})",
                    (resolved_tenant, *chunk),
                ).fetchall()
                for row in rows:
                    payload = self._store._row_dict(row, json_fields=("payload_json",))
                    by_id[str(payload["tombstone_id"])] = payload
        return [by_id[tombstone_id] for tombstone_id in ordered_ids if tombstone_id in by_id]

    def pending_tombstones(self, *, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return self.get_pending_tombstones(tenant_id=tenant_id, limit=limit)

    def mark_tombstone_applied(
        self,
        tombstone_id: str,
        *,
        tenant_id: str,
    ) -> dict[str, Any] | None:
        return self._apply_or_begin(
            tombstone_id,
            tenant_id=tenant_id,
            cleanup=False,
        )

    def begin_tombstone_cleanup(
        self,
        tombstone_id: str,
        *,
        tenant_id: str,
    ) -> dict[str, Any] | None:
        return self._apply_or_begin(
            tombstone_id,
            tenant_id=tenant_id,
            cleanup=True,
        )

    def _apply_or_begin(
        self,
        tombstone_id: str,
        *,
        tenant_id: str,
        cleanup: bool,
    ) -> dict[str, Any] | None:
        resolved_tenant = self._store._catalog._require_tenant(tenant_id)
        with self._store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM context_tombstones WHERE tenant_id = ? AND tombstone_id = ?",
                (resolved_tenant, str(tombstone_id)),
            ).fetchone()
            if row is None:
                return None
            status = str(row["status"])
            terminal = {"APPLIED", "STALE"}
            if status not in terminal | {"CLEANING"}:
                current = conn.execute(
                    "SELECT record_kind, source_revision, source_digest, projection_effect_hash, updated_at "
                    "FROM contexts WHERE tenant_id = ? AND record_key = ?",
                    (resolved_tenant, str(row["record_key"])),
                ).fetchone()
                if current is not None and str(current["record_kind"]) in _DOCUMENT_RECORD_KINDS:
                    raise ValueError(
                        "memory document projections require tombstone_memory_document_projection()"
                    )
                payload = self._store._json_mapping(row["payload_json"])
                stale = bool(
                    current is not None
                    and (
                        (
                            int(row["source_revision"])
                            and int(current["source_revision"]) > int(row["source_revision"])
                        )
                        or (
                            payload.get("expected_source_digest")
                            and str(current["source_digest"]) != str(payload["expected_source_digest"])
                        )
                        or (
                            payload.get("expected_projection_effect_hash")
                            and str(current["projection_effect_hash"])
                            != str(payload["expected_projection_effect_hash"])
                        )
                        or (
                            payload.get("expected_updated_at")
                            and str(current["updated_at"]) != str(payload["expected_updated_at"])
                        )
                    )
                )
                if stale:
                    status = "STALE"
                else:
                    self._store._delete_catalog_in_transaction(
                        conn,
                        str(row["record_key"]),
                        tenant_id=resolved_tenant,
                    )
                    status = "CLEANING" if cleanup else "APPLIED"
                conn.execute(
                    "UPDATE context_tombstones SET status = ?, updated_at = ?, last_error = '' "
                    "WHERE tenant_id = ? AND tombstone_id = ?",
                    (status, self._store._now(), resolved_tenant, str(tombstone_id)),
                )
            refreshed = conn.execute(
                "SELECT * FROM context_tombstones WHERE tenant_id = ? AND tombstone_id = ?",
                (resolved_tenant, str(tombstone_id)),
            ).fetchone()
        return self._store._row_dict(refreshed, json_fields=("payload_json",)) if refreshed else None

    def finish_tombstone_cleanup(
        self,
        tombstone_id: str,
        *,
        tenant_id: str,
    ) -> dict[str, Any] | None:
        resolved_tenant = self._store._catalog._require_tenant(tenant_id)
        with self._store._connect() as conn:
            row = conn.execute(
                "SELECT status FROM context_tombstones WHERE tenant_id = ? AND tombstone_id = ?",
                (resolved_tenant, str(tombstone_id)),
            ).fetchone()
            if row is None:
                return None
            status = str(row["status"])
            if status == "CLEANING":
                conn.execute(
                    "UPDATE context_tombstones SET status = 'APPLIED', updated_at = ?, last_error = '' "
                    "WHERE tenant_id = ? AND tombstone_id = ? AND status = 'CLEANING'",
                    (self._store._now(), resolved_tenant, str(tombstone_id)),
                )
            elif status not in {"APPLIED", "STALE"}:
                raise RuntimeError(f"tombstone cleanup cannot finish from {status}")
            refreshed = conn.execute(
                "SELECT * FROM context_tombstones WHERE tenant_id = ? AND tombstone_id = ?",
                (resolved_tenant, str(tombstone_id)),
            ).fetchone()
        return self._store._row_dict(refreshed, json_fields=("payload_json",)) if refreshed else None

    def mark_tombstone_cleanup_failed(
        self,
        tombstone_id: str,
        error: str,
        *,
        tenant_id: str,
    ) -> dict[str, Any] | None:
        return self._mark_failed(
            tombstone_id,
            error,
            tenant_id=tenant_id,
            require_cleaning=True,
        )

    def mark_tombstone_failed(
        self,
        tombstone_id: str,
        error: str,
        *,
        tenant_id: str,
    ) -> dict[str, Any] | None:
        return self._mark_failed(
            tombstone_id,
            error,
            tenant_id=tenant_id,
            require_cleaning=False,
        )

    def _mark_failed(
        self,
        tombstone_id: str,
        error: str,
        *,
        tenant_id: str,
        require_cleaning: bool,
    ) -> dict[str, Any] | None:
        resolved_tenant = self._store._catalog._require_tenant(tenant_id)
        safe_error = str(self._store.sanitizer.sanitize_trace(str(error or "")))
        with self._store._connect() as conn:
            status_sql = (
                " AND status = 'CLEANING'"
                if require_cleaning
                else " AND status NOT IN ('APPLIED', 'STALE')"
            )
            next_status = "status" if require_cleaning else "'FAILED'"
            conn.execute(
                f"UPDATE context_tombstones SET status = {next_status}, retry_count = retry_count + 1, "
                "last_error = ?, updated_at = ? WHERE tenant_id = ? AND tombstone_id = ?"
                + status_sql,
                (safe_error, self._store._now(), resolved_tenant, str(tombstone_id)),
            )
            row = conn.execute(
                "SELECT * FROM context_tombstones WHERE tenant_id = ? AND tombstone_id = ?",
                (resolved_tenant, str(tombstone_id)),
            ).fetchone()
        return self._store._row_dict(row, json_fields=("payload_json",)) if row else None


__all__ = ["TombstoneOperations"]
