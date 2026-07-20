"""SQLite 队列的连接、Schema 校验与行映射辅助能力。"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from infrastructure.store.contracts.queue import (
    LeaseLostError,
    QueueIdempotencyConflictError,
    QueueJob,
)

_QUEUE_TABLE_LAYOUT = (
    ("job_id", "TEXT", 0, None, 1),
    ("queue_name", "TEXT", 1, None, 0),
    ("action", "TEXT", 1, None, 0),
    ("target_uri", "TEXT", 1, None, 0),
    ("payload_json", "TEXT", 1, None, 0),
    ("tenant_id", "TEXT", 1, "''", 0),
    ("owner_user_id", "TEXT", 1, "''", 0),
    ("workspace_id", "TEXT", 1, "''", 0),
    ("status", "TEXT", 1, None, 0),
    ("leased_until", "TEXT", 0, None, 0),
    ("lease_token", "TEXT", 1, "''", 0),
    ("lease_generation", "INTEGER", 1, "0", 0),
    ("lease_owner", "TEXT", 1, "''", 0),
    ("retry_count", "INTEGER", 1, "0", 0),
    ("created_at", "TEXT", 1, None, 0),
    ("updated_at", "TEXT", 1, None, 0),
    ("last_error", "TEXT", 1, "''", 0),
)
_QUEUE_STATUSES = ("pending", "leased", "done", "dead_letter", "quarantine")


class QueueStoreSupportMixin:
    """封装队列存储的基础设施细节，业务状态转换留在主存储类。"""

    # 由 SQLiteQueueStore 在初始化时绑定。
    path: Path

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
            existing = conn.execute("SELECT type FROM sqlite_master WHERE name = 'queue_jobs'").fetchone()
            if existing is None:
                self._create_queue_table(conn)
            elif str(existing["type"]) != "table":
                raise RuntimeError("unsupported QueueStore layout; reset the greenfield runtime")
            self._require_exact_queue_layout(conn)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS queue_jobs_claim_idx "
                "ON queue_jobs(queue_name, status, leased_until, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS queue_jobs_target_status_idx ON queue_jobs(queue_name, target_uri, status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS queue_jobs_scope_status_idx "
                "ON queue_jobs(queue_name, tenant_id, owner_user_id, workspace_id, status)"
            )
            conn.commit()
        os.chmod(self.path, 0o600)

    @staticmethod
    def _create_queue_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE queue_jobs (
              job_id TEXT PRIMARY KEY,
              queue_name TEXT NOT NULL,
              action TEXT NOT NULL,
              target_uri TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              tenant_id TEXT NOT NULL DEFAULT '',
              owner_user_id TEXT NOT NULL DEFAULT '',
              workspace_id TEXT NOT NULL DEFAULT '',
              status TEXT NOT NULL,
              leased_until TEXT,
              lease_token TEXT NOT NULL DEFAULT '',
              lease_generation INTEGER NOT NULL DEFAULT 0,
              lease_owner TEXT NOT NULL DEFAULT '',
              retry_count INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              last_error TEXT NOT NULL DEFAULT ''
            )
            """
        )

    @staticmethod
    def _require_exact_queue_layout(conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(queue_jobs)").fetchall()
        layout = tuple(
            (
                str(row["name"]),
                str(row["type"]).upper(),
                int(row["notnull"]),
                row["dflt_value"],
                int(row["pk"]),
            )
            for row in rows
        )
        if layout != _QUEUE_TABLE_LAYOUT:
            raise RuntimeError("unsupported QueueStore layout; reset the greenfield runtime")
        placeholders = ",".join("?" for _ in _QUEUE_STATUSES)
        invalid = conn.execute(
            f"SELECT job_id FROM queue_jobs WHERE tenant_id = '' OR status NOT IN ({placeholders}) LIMIT 1",
            _QUEUE_STATUSES,
        ).fetchone()
        if invalid is not None:
            raise RuntimeError("unsupported QueueStore data; reset the greenfield runtime")

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

    @staticmethod
    def _job_scope(job: QueueJob) -> tuple[str, str, str]:
        payload = dict(job.payload or {})
        tenant_id = str(payload.get("tenant_id") or "default")
        owner_user_id = str(payload.get("owner_user_id") or "")
        if not owner_user_id and job.target_uri.startswith("memoryos://user/"):
            candidate = job.target_uri.removeprefix("memoryos://user/").split("/", 1)[0]
            if candidate and not candidate.startswith("subject_"):
                owner_user_id = candidate
        return tenant_id, owner_user_id, str(payload.get("workspace_id") or "")

    def _now_dt(self):
        from datetime import datetime, timezone

        return datetime.now(timezone.utc)


__all__ = ["QueueStoreSupportMixin"]
