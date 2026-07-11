"""上下文数据库里的SQLite队列存储。"""

from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
from pathlib import Path

from memoryos.contextdb.store.source_store import QueueJob
from memoryos.core.time import utc_now


class SQLiteQueueStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def enqueue(self, job: QueueJob) -> None:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO queue_jobs(job_id, queue_name, action, target_uri, payload_json, status, leased_until, retry_count, created_at, updated_at, last_error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO NOTHING
                """,
                (
                    job.job_id,
                    job.queue_name,
                    job.action,
                    job.target_uri,
                    json.dumps(job.payload, ensure_ascii=False),
                    job.status,
                    job.leased_until,
                    job.retry_count,
                    now,
                    now,
                    job.last_error,
                ),
            )

    def lease(self, queue_name: str, limit: int = 10, lease_seconds: int = 60) -> list[QueueJob]:
        now = utc_now()
        leased_until = (self._now_dt() + timedelta(seconds=max(1, lease_seconds))).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM queue_jobs
                WHERE queue_name = ?
                  AND (status = 'pending' OR (status = 'leased' AND leased_until <= ?))
                ORDER BY created_at
                LIMIT ?
                """,
                (queue_name, now, limit),
            ).fetchall()
            jobs = [self._row_to_job(row, status="leased", leased_until=leased_until) for row in rows]
            for job in jobs:
                conn.execute(
                    "UPDATE queue_jobs SET status = 'leased', leased_until = ?, updated_at = ? WHERE job_id = ? AND status IN ('pending', 'leased')",
                    (leased_until, now, job.job_id),
                )
        return jobs

    def ack(self, job_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE queue_jobs SET status = 'done', updated_at = ?, leased_until = NULL WHERE job_id = ?", (utc_now(), job_id))

    def fail(self, job_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE queue_jobs SET status = 'failed', retry_count = retry_count + 1, last_error = ?, updated_at = ?, leased_until = NULL WHERE job_id = ?",
                (error, utc_now(), job_id),
            )

    def retry(self, job_id: str, error: str, *, max_retries: int = 3, retryable: bool = True) -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT retry_count FROM queue_jobs WHERE job_id = ?", (job_id,)).fetchone()
            retry_count = int(row["retry_count"] if row else 0) + 1
            status = "pending" if retryable and retry_count < max_retries else "dead_letter"
            conn.execute(
                "UPDATE queue_jobs SET status = ?, retry_count = ?, last_error = ?, updated_at = ?, leased_until = NULL WHERE job_id = ?",
                (status, retry_count, error[:500], utc_now(), job_id),
            )
        return status

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS count FROM queue_jobs GROUP BY status").fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def _row_to_job(self, row: sqlite3.Row, status: str | None = None, leased_until: str | None = None) -> QueueJob:
        return QueueJob(
            job_id=row["job_id"],
            queue_name=row["queue_name"],
            action=row["action"],
            target_uri=row["target_uri"],
            payload=json.loads(row["payload_json"] or "{}"),
            status=status or row["status"],
            leased_until=leased_until if leased_until is not None else row["leased_until"],
            retry_count=int(row["retry_count"]),
            last_error=row["last_error"] or "",
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
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

    def _now_dt(self):
        from datetime import datetime, timezone

        return datetime.now(timezone.utc)


SqliteQueueStore = SQLiteQueueStore

__all__ = ["SQLiteQueueStore", "SqliteQueueStore"]
