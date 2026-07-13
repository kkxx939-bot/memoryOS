"""上下文数据库里的SQLite队列存储。"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

from memoryos.contextdb.store.source_store import (
    LeaseLostError,
    QueueIdempotencyConflictError,
    QueueJob,
    QueueLeaseIdentityError,
)
from memoryos.core.time import utc_now


class SQLiteQueueStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        self._init_db()

    def enqueue(self, job: QueueJob) -> QueueJob:
        """Create one immutable queue identity or return its existing state."""

        if job.status != "pending" or job.lease_token or job.lease_owner or job.lease_generation:
            raise ValueError("new queue jobs must be unleased and pending")
        now = utc_now()
        payload_json = self._canonical_payload(job.payload)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT * FROM queue_jobs WHERE job_id = ?", (job.job_id,)).fetchone()
            if existing is not None:
                if self._identity(existing) != (job.queue_name, job.action, job.target_uri, payload_json):
                    conn.rollback()
                    raise QueueIdempotencyConflictError(
                        f"queue job id is already bound to another payload: {job.job_id}"
                    )
                conn.commit()
                return self._row_to_job(existing)
            conn.execute(
                """
                INSERT INTO queue_jobs(job_id, queue_name, action, target_uri, payload_json, status, leased_until, retry_count, created_at, updated_at, last_error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.queue_name,
                    job.action,
                    job.target_uri,
                    payload_json,
                    "pending",
                    None,
                    0,
                    now,
                    now,
                    "",
                ),
            )
            row = conn.execute("SELECT * FROM queue_jobs WHERE job_id = ?", (job.job_id,)).fetchone()
            conn.commit()
        assert row is not None
        return self._row_to_job(row)

    def lease(
        self,
        queue_name: str,
        *,
        lease_owner: str,
        limit: int = 10,
        lease_seconds: int = 60,
        job_ids: Sequence[str] | None = None,
    ) -> list[QueueJob]:
        """Atomically select and claim only jobs owned by this write transaction."""

        if not isinstance(lease_owner, str) or not lease_owner.strip():
            raise ValueError("lease_owner must be non-empty")
        if limit <= 0:
            return []
        now = self._now_dt().isoformat()
        leased_until = (self._now_dt() + timedelta(seconds=max(1, lease_seconds))).isoformat()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            id_filter = ""
            params: list[object] = [queue_name, now]
            selected_ids = tuple(dict.fromkeys(str(item) for item in (job_ids or ()) if str(item)))
            if job_ids is not None:
                if not selected_ids:
                    conn.commit()
                    return []
                id_filter = f" AND job_id IN ({','.join('?' for _ in selected_ids)})"
                params.extend(selected_ids)
            params.append(limit)
            rows = conn.execute(
                f"""
                SELECT * FROM queue_jobs
                WHERE queue_name = ?
                  AND (status = 'pending' OR (status = 'leased' AND leased_until <= ?))
                  {id_filter}
                ORDER BY created_at
                LIMIT ?
                """,
                params,
            ).fetchall()
            jobs: list[QueueJob] = []
            for row in rows:
                token = secrets.token_urlsafe(32)
                updated = conn.execute(
                    """
                    UPDATE queue_jobs
                    SET status = 'leased', leased_until = ?, lease_token = ?,
                        lease_generation = lease_generation + 1, lease_owner = ?, updated_at = ?
                    WHERE job_id = ?
                      AND queue_name = ?
                      AND (status = 'pending' OR (status = 'leased' AND leased_until <= ?))
                    RETURNING *
                    """,
                    (leased_until, token, lease_owner, now, row["job_id"], queue_name, now),
                ).fetchone()
                if updated is not None:
                    jobs.append(self._row_to_job(updated))
            conn.commit()
        return jobs

    def ack(self, job: QueueJob) -> QueueJob:
        return self._settle(
            job,
            """
            UPDATE queue_jobs
            SET status = 'done', updated_at = ?, leased_until = NULL,
                lease_token = '', lease_owner = ''
            WHERE job_id = ? AND status = 'leased' AND lease_token = ?
              AND lease_generation = ? AND lease_owner = ? AND leased_until > ?
            RETURNING *
            """,
            (),
        )

    def fail(self, job: QueueJob, error: str) -> QueueJob:
        return self._settle(
            job,
            """
            UPDATE queue_jobs
            SET status = 'dead_letter', retry_count = retry_count + 1,
                last_error = ?, updated_at = ?, leased_until = NULL,
                lease_token = '', lease_owner = ''
            WHERE job_id = ? AND status = 'leased' AND lease_token = ?
              AND lease_generation = ? AND lease_owner = ? AND leased_until > ?
            RETURNING *
            """,
            (str(error)[:500],),
        )

    def retry(
        self,
        job: QueueJob,
        error: str,
        *,
        max_retries: int = 3,
        retryable: bool = True,
    ) -> QueueJob:
        return self._settle(
            job,
            """
            UPDATE queue_jobs
            SET status = CASE
                    WHEN ? = 1 AND retry_count + 1 < ? THEN 'pending'
                    ELSE 'dead_letter'
                END,
                retry_count = retry_count + 1, last_error = ?, updated_at = ?,
                leased_until = NULL, lease_token = '', lease_owner = ''
            WHERE job_id = ? AND status = 'leased' AND lease_token = ?
              AND lease_generation = ? AND lease_owner = ? AND leased_until > ?
            RETURNING *
            """,
            (int(bool(retryable)), max(1, int(max_retries)), str(error)[:500]),
        )

    def release(self, job: QueueJob, reason: str = "") -> QueueJob:
        """Return an unattempted owned lease without consuming retry budget."""

        return self._settle(
            job,
            """
            UPDATE queue_jobs
            SET status = 'pending', last_error = ?, updated_at = ?,
                leased_until = NULL, lease_token = '', lease_owner = ''
            WHERE job_id = ? AND status = 'leased' AND lease_token = ?
              AND lease_generation = ? AND lease_owner = ? AND leased_until > ?
            RETURNING *
            """,
            (str(reason)[:500] if reason else job.last_error,),
        )

    def quarantine(self, job: QueueJob, error: str) -> QueueJob:
        return self._settle(
            job,
            """
            UPDATE queue_jobs
            SET status = 'quarantine', retry_count = retry_count + 1,
                last_error = ?, updated_at = ?, leased_until = NULL,
                lease_token = '', lease_owner = ''
            WHERE job_id = ? AND status = 'leased' AND lease_token = ?
              AND lease_generation = ? AND lease_owner = ? AND leased_until > ?
            RETURNING *
            """,
            (str(error)[:500],),
        )

    def quarantine_identity_conflict(self, job: QueueJob, error: str) -> QueueJob:
        """Quarantine an owned lease whose immutable payload was corrupted."""

        return self._settle(
            job,
            """
            UPDATE queue_jobs
            SET status = 'quarantine', retry_count = retry_count + 1,
                last_error = ?, updated_at = ?, leased_until = NULL,
                lease_token = '', lease_owner = ''
            WHERE job_id = ? AND status = 'leased' AND lease_token = ?
              AND lease_generation = ? AND lease_owner = ? AND leased_until > ?
            RETURNING *
            """,
            (str(error)[:500],),
            verify_identity=False,
        )

    def extend(self, job: QueueJob, *, lease_seconds: int = 60) -> QueueJob:
        leased_until = (self._now_dt() + timedelta(seconds=max(1, lease_seconds))).isoformat()
        return self._settle(
            job,
            """
            UPDATE queue_jobs
            SET leased_until = ?, updated_at = ?
            WHERE job_id = ? AND status = 'leased' AND lease_token = ?
              AND lease_generation = ? AND lease_owner = ? AND leased_until > ?
            RETURNING *
            """,
            (leased_until,),
        )

    def get(self, job_id: str) -> QueueJob | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM queue_jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row is not None else None

    def recover_expired_leases(self, *, queue_name: str | None = None) -> int:
        """Return expired work to pending without consuming retry budget."""

        now = self._now_dt().isoformat()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            queue_filter = "" if queue_name is None else " AND queue_name = ?"
            parameters: tuple[object, ...] = (now,) if queue_name is None else (now, queue_name)
            cursor = conn.execute(
                f"""
                UPDATE queue_jobs
                SET status = 'pending', updated_at = ?, leased_until = NULL,
                    lease_token = '', lease_owner = ''
                WHERE status = 'leased'
                  AND (leased_until IS NULL OR leased_until <= ?)
                  {queue_filter}
                """,
                (now, *parameters),
            )
            recovered = int(cursor.rowcount)
            conn.commit()
        return recovered

    def _settle(
        self,
        job: QueueJob,
        sql: str,
        prefix: tuple[object, ...],
        *,
        verify_identity: bool = True,
    ) -> QueueJob:
        self._validate_lease(job)
        now = self._now_dt().isoformat()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT * FROM queue_jobs WHERE job_id = ?",
                (job.job_id,),
            ).fetchone()
            if verify_identity:
                try:
                    identity_changed = current is not None and self._identity(current) != (
                        job.queue_name,
                        job.action,
                        job.target_uri,
                        self._canonical_payload(job.payload),
                    )
                except QueueIdempotencyConflictError as exc:
                    conn.rollback()
                    raise QueueLeaseIdentityError(
                        f"queue immutable identity is corrupt while leased: {job.job_id}"
                    ) from exc
                if identity_changed:
                    conn.rollback()
                    raise QueueLeaseIdentityError(f"queue immutable identity changed while leased: {job.job_id}")
            row = conn.execute(
                sql,
                (
                    *prefix,
                    now,
                    job.job_id,
                    job.lease_token,
                    job.lease_generation,
                    job.lease_owner,
                    now,
                ),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise LeaseLostError(f"queue lease lost for {job.job_id} generation {job.lease_generation}")
            conn.commit()
        return self._row_to_job(row)

    def stats(self, *, queue_name: str | None = None) -> dict[str, int]:
        with self._connect() as conn:
            if queue_name is None:
                rows = conn.execute("SELECT status, COUNT(*) AS count FROM queue_jobs GROUP BY status").fetchall()
            else:
                rows = conn.execute(
                    "SELECT status, COUNT(*) AS count FROM queue_jobs WHERE queue_name = ? GROUP BY status",
                    (queue_name,),
                ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def _row_to_job(self, row: sqlite3.Row) -> QueueJob:
        return QueueJob(
            job_id=row["job_id"],
            queue_name=row["queue_name"],
            action=row["action"],
            target_uri=row["target_uri"],
            payload=json.loads(row["payload_json"] or "{}"),
            status=row["status"],
            leased_until=row["leased_until"],
            lease_token=row["lease_token"],
            lease_generation=int(row["lease_generation"]),
            lease_owner=row["lease_owner"],
            retry_count=int(row["retry_count"]),
            last_error=row["last_error"] or "",
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS queue_jobs (
                  job_id TEXT PRIMARY KEY,
                  queue_name TEXT NOT NULL,
                  action TEXT NOT NULL,
                  target_uri TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  leased_until TEXT,
                  retry_count INTEGER NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  last_error TEXT NOT NULL DEFAULT ''
                )
                """
            )
            columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(queue_jobs)")}
            migrations = {
                "lease_token": "TEXT NOT NULL DEFAULT ''",
                "lease_generation": "INTEGER NOT NULL DEFAULT 0",
                "lease_owner": "TEXT NOT NULL DEFAULT ''",
            }
            for column, declaration in migrations.items():
                if column not in columns:
                    conn.execute(f"ALTER TABLE queue_jobs ADD COLUMN {column} {declaration}")
            conn.execute("UPDATE queue_jobs SET status = 'dead_letter' WHERE status = 'failed'")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS queue_jobs_claim_idx "
                "ON queue_jobs(queue_name, status, leased_until, created_at)"
            )
            conn.commit()
        os.chmod(self.path, 0o600)

    def _validate_lease(self, job: QueueJob) -> None:
        if (
            job.status != "leased"
            or not job.lease_token
            or not job.lease_owner
            or job.lease_generation < 1
            or not job.leased_until
        ):
            raise LeaseLostError(f"queue job has no valid lease proof: {job.job_id}")

    def _canonical_payload(self, payload: object) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _identity(self, row: sqlite3.Row) -> tuple[str, str, str, str]:
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except json.JSONDecodeError as exc:
            raise QueueIdempotencyConflictError(f"existing queue payload is corrupt: {row['job_id']}") from exc
        return (
            str(row["queue_name"]),
            str(row["action"]),
            str(row["target_uri"]),
            self._canonical_payload(payload),
        )

    def _now_dt(self):
        from datetime import datetime, timezone

        return datetime.now(timezone.utc)


SqliteQueueStore = SQLiteQueueStore

__all__ = ["SQLiteQueueStore", "SqliteQueueStore"]
