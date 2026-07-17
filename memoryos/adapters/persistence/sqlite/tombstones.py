"""SQLite catalog TombstoneOperations responsibility component."""

from __future__ import annotations

from memoryos.adapters.persistence.sqlite._common import (
    _MAX_FILTER_VALUES,
    Any,
    Mapping,
    Sequence,
)


class TombstoneOperations:
    """Own one stable subset of SQLite catalog behavior."""

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
        """Compatibility helper: durably enqueue, then apply one tombstone."""

        queued = self._store.enqueue_tombstone(
            tenant_id=tenant_id,
            record_key=record_key,
            reason=reason,
            uri=uri,
            source_revision=source_revision,
            tombstone_id=tombstone_id,
            payload=payload,
        )
        applied = self._store.mark_tombstone_applied(str(queued["tombstone_id"]))
        if applied is None:  # pragma: no cover - enqueue either commits or raises.
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
        """Persist a replayable projection deletion before a worker touches serving data."""

        if not tenant_id or not record_key or not reason:
            raise ValueError("tenant_id, record_key, and reason are required")
        safe_payload = self._store.sanitizer.sanitize_trace(dict(payload or {}))
        safe_reason = str(self._store.sanitizer.sanitize_trace(str(reason)))
        identity = {
            "tenant_id": str(tenant_id),
            "record_key": str(record_key),
            "reason": safe_reason,
            "source_revision": int(source_revision),
        }
        stable_id = tombstone_id or self._store.sanitizer.digest(identity)
        now = self._store._now()
        with self._store._connect() as conn:
            existing = conn.execute(
                "SELECT tenant_id, record_key, reason, source_revision, status, uri "
                "FROM context_tombstones WHERE tombstone_id = ?",
                (stable_id,),
            ).fetchone()
            if existing is not None:
                existing_identity = (
                    str(existing["tenant_id"]),
                    str(existing["record_key"]),
                    str(existing["reason"]),
                    int(existing["source_revision"]),
                )
                requested_identity = (str(tenant_id), str(record_key), safe_reason, int(source_revision))
                if existing_identity != requested_identity:
                    raise ValueError("tombstone_id is already bound to a different immutable identity")
            current = conn.execute("SELECT uri FROM contexts WHERE record_key = ?", (str(record_key),)).fetchone()
            effective_uri = self._store._safe_reference_uri(str(uri or (current["uri"] if current is not None else "")))
            conn.execute(
                """
                INSERT INTO context_tombstones(
                  tombstone_id, tenant_id, record_key, uri, reason, source_revision,
                  status, payload_json, created_at, updated_at, retry_count, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, 0, '')
                ON CONFLICT(tombstone_id) DO UPDATE SET
                  payload_json=excluded.payload_json,
                  updated_at=excluded.updated_at
                """,
                (
                    stable_id,
                    str(tenant_id),
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
            "tenant_id": str(tenant_id),
            "record_key": str(record_key),
            "uri": effective_uri,
            "status": str(existing["status"]) if existing is not None else "PENDING",
            "source_revision": int(source_revision),
        }

    def get_pending_tombstones(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._store._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM context_tombstones WHERE status IN ('PENDING', 'FAILED', 'CLEANING') "
                "ORDER BY updated_at, tombstone_id LIMIT ?",
                (self._store._bounded_limit(limit),),
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
        """Recover one delete target's exact unfinished journal without queue starvation."""

        with self._store._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM context_tombstones "
                "WHERE tenant_id = ? AND uri = ? AND status IN ('PENDING', 'FAILED', 'CLEANING') "
                "AND tombstone_id > ? ORDER BY tombstone_id LIMIT ?",
                (
                    str(tenant_id),
                    self._store._safe_reference_uri(str(uri)),
                    str(after_tombstone_id),
                    self._store._bounded_limit(limit),
                ),
            ).fetchall()
        return [self._store._row_dict(row, json_fields=("payload_json",)) for row in rows]

    def get_tombstones(self, tombstone_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Read an explicit bounded set of durable tombstones in caller order.

        Delete callers use this exact-ID path after they have durably enqueued
        every affected projection.  It avoids starvation behind unrelated
        failed journal entries and does not depend on the pending queue's
        1,000-row administrative batch limit.
        """

        ordered_ids = tuple(dict.fromkeys(str(item) for item in tombstone_ids if str(item)))
        if not ordered_ids:
            return []
        by_id: dict[str, dict[str, Any]] = {}
        with self._store._connect() as conn:
            for offset in range(0, len(ordered_ids), _MAX_FILTER_VALUES):
                chunk = ordered_ids[offset : offset + _MAX_FILTER_VALUES]
                placeholders = ", ".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT * FROM context_tombstones WHERE tombstone_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    payload = self._store._row_dict(row, json_fields=("payload_json",))
                    by_id[str(payload["tombstone_id"])] = payload
        return [by_id[tombstone_id] for tombstone_id in ordered_ids if tombstone_id in by_id]

    def pending_tombstones(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Backward-compatible alias for get_pending_tombstones()."""

        return self._store.get_pending_tombstones(limit=limit)

    def mark_tombstone_applied(self, tombstone_id: str) -> dict[str, Any] | None:
        """Idempotently apply a queued tombstone and close its durable journal row."""

        with self._store._connect() as conn:
            row = conn.execute(
                "SELECT * FROM context_tombstones WHERE tombstone_id = ?",
                (str(tombstone_id),),
            ).fetchone()
            if row is None:
                return None
            status = str(row["status"])
            if status not in {"APPLIED", "STALE"}:
                current = conn.execute(
                    "SELECT tenant_id, source_revision FROM contexts WHERE record_key = ?",
                    (str(row["record_key"]),),
                ).fetchone()
                if current is not None and str(current["tenant_id"]) != str(row["tenant_id"]):
                    raise ValueError("tombstone tenant does not own the target catalog record")
                if (
                    current is not None
                    and int(row["source_revision"])
                    and int(current["source_revision"]) > int(row["source_revision"])
                ):
                    status = "STALE"
                else:
                    self._store._delete_catalog_in_transaction(
                        conn,
                        str(row["record_key"]),
                        tenant_id=str(row["tenant_id"]),
                    )
                    status = "APPLIED"
                conn.execute(
                    "UPDATE context_tombstones SET status = ?, updated_at = ?, last_error = '' WHERE tombstone_id = ?",
                    (status, self._store._now(), str(tombstone_id)),
                )
            refreshed = conn.execute(
                "SELECT * FROM context_tombstones WHERE tombstone_id = ?",
                (str(tombstone_id),),
            ).fetchone()
        return self._store._row_dict(refreshed, json_fields=("payload_json",)) if refreshed is not None else None

    def begin_tombstone_cleanup(self, tombstone_id: str) -> dict[str, Any] | None:
        """Atomically establish deletion ownership before external cleanup.

        ``CLEANING`` is a durable, replayable intermediate state.  The Catalog
        row is removed in the same SQLite transaction that enters this state,
        so an online read can never observe a row whose external projections
        are already being retired.  A newer Catalog revision makes the
        tombstone ``STALE`` before Vector or Relation state is touched.
        """

        with self._store._connect() as conn:
            row = conn.execute(
                "SELECT * FROM context_tombstones WHERE tombstone_id = ?",
                (str(tombstone_id),),
            ).fetchone()
            if row is None:
                return None
            status = str(row["status"])
            if status not in {"APPLIED", "STALE", "CLEANING"}:
                current = conn.execute(
                    "SELECT tenant_id, source_revision, source_digest, projection_effect_hash, updated_at "
                    "FROM contexts WHERE record_key = ?",
                    (str(row["record_key"]),),
                ).fetchone()
                if current is not None and str(current["tenant_id"]) != str(row["tenant_id"]):
                    raise ValueError("tombstone tenant does not own the target catalog record")
                payload = self._store._json_mapping(row["payload_json"])
                expected_digest = str(payload.get("expected_source_digest") or "")
                expected_effect = str(payload.get("expected_projection_effect_hash") or "")
                expected_updated_at = str(payload.get("expected_updated_at") or "")
                stale = bool(
                    current is not None
                    and (
                        (int(row["source_revision"]) and int(current["source_revision"]) > int(row["source_revision"]))
                        or (expected_digest and str(current["source_digest"]) != expected_digest)
                        or (expected_effect and str(current["projection_effect_hash"]) != expected_effect)
                        or (expected_updated_at and str(current["updated_at"]) != expected_updated_at)
                    )
                )
                if stale:
                    status = "STALE"
                else:
                    self._store._delete_catalog_in_transaction(
                        conn,
                        str(row["record_key"]),
                        tenant_id=str(row["tenant_id"]),
                    )
                    status = "CLEANING"
                conn.execute(
                    "UPDATE context_tombstones SET status = ?, updated_at = ?, last_error = '' WHERE tombstone_id = ?",
                    (status, self._store._now(), str(tombstone_id)),
                )
            refreshed = conn.execute(
                "SELECT * FROM context_tombstones WHERE tombstone_id = ?",
                (str(tombstone_id),),
            ).fetchone()
        return self._store._row_dict(refreshed, json_fields=("payload_json",)) if refreshed is not None else None

    def finish_tombstone_cleanup(self, tombstone_id: str) -> dict[str, Any] | None:
        """Mark external cleanup complete without weakening terminal states."""

        with self._store._connect() as conn:
            row = conn.execute(
                "SELECT status FROM context_tombstones WHERE tombstone_id = ?",
                (str(tombstone_id),),
            ).fetchone()
            if row is None:
                return None
            status = str(row["status"])
            if status == "CLEANING":
                conn.execute(
                    "UPDATE context_tombstones SET status = 'APPLIED', updated_at = ?, last_error = '' "
                    "WHERE tombstone_id = ? AND status = 'CLEANING'",
                    (self._store._now(), str(tombstone_id)),
                )
            elif status not in {"APPLIED", "STALE"}:
                raise RuntimeError(f"tombstone cleanup cannot finish from {status}")
            refreshed = conn.execute(
                "SELECT * FROM context_tombstones WHERE tombstone_id = ?",
                (str(tombstone_id),),
            ).fetchone()
        return self._store._row_dict(refreshed, json_fields=("payload_json",)) if refreshed is not None else None

    def mark_tombstone_cleanup_failed(self, tombstone_id: str, error: str) -> dict[str, Any] | None:
        """Record an external cleanup error while retaining deletion ownership."""

        safe_error = str(self._store.sanitizer.sanitize_trace(str(error or "")))
        with self._store._connect() as conn:
            conn.execute(
                "UPDATE context_tombstones SET retry_count = retry_count + 1, last_error = ?, updated_at = ? "
                "WHERE tombstone_id = ? AND status = 'CLEANING'",
                (safe_error, self._store._now(), str(tombstone_id)),
            )
            row = conn.execute(
                "SELECT * FROM context_tombstones WHERE tombstone_id = ?",
                (str(tombstone_id),),
            ).fetchone()
        return self._store._row_dict(row, json_fields=("payload_json",)) if row is not None else None

    def mark_tombstone_failed(self, tombstone_id: str, error: str) -> dict[str, Any] | None:
        safe_error = str(self._store.sanitizer.sanitize_trace(str(error or "")))
        with self._store._connect() as conn:
            conn.execute(
                "UPDATE context_tombstones SET status = 'FAILED', retry_count = retry_count + 1, "
                "last_error = ?, updated_at = ? WHERE tombstone_id = ?",
                (safe_error, self._store._now(), str(tombstone_id)),
            )
            row = conn.execute(
                "SELECT * FROM context_tombstones WHERE tombstone_id = ?",
                (str(tombstone_id),),
            ).fetchone()
        return self._store._row_dict(row, json_fields=("payload_json",)) if row is not None else None


__all__ = ["TombstoneOperations"]
