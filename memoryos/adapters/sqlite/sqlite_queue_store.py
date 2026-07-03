from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from memoryos.contextdb.store.source_store import QueueJob


class SqliteQueueStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def enqueue(self, job: QueueJob) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO queue_jobs(job_id, queue_name, action, target_uri, payload, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.queue_name,
                    job.action,
                    job.target_uri,
                    json.dumps(job.payload, ensure_ascii=False),
                    job.status,
                ),
            )

    def lease(self, queue_name: str, limit: int = 10) -> list[QueueJob]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM queue_jobs WHERE queue_name = ? AND status = 'pending' ORDER BY rowid LIMIT ?",
                (queue_name, limit),
            ).fetchall()
            ids = [row["job_id"] for row in rows]
            for job_id in ids:
                conn.execute("UPDATE queue_jobs SET status = 'leased' WHERE job_id = ?", (job_id,))
        return [
            QueueJob(
                job_id=row["job_id"],
                queue_name=row["queue_name"],
                action=row["action"],
                target_uri=row["target_uri"],
                payload=json.loads(row["payload"] or "{}"),
                status="leased",
            )
            for row in rows
        ]

    def ack(self, job_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE queue_jobs SET status = 'done' WHERE job_id = ?", (job_id,))

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
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )


__all__ = ["SqliteQueueStore"]
