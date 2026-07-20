"""上下文数据库里的SQLite队列存储。"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

from infrastructure.store.sqlite.queue_support import QueueStoreSupportMixin
from infrastructure.store.contracts.queue import (
    LeaseLostError,
    QueueIdempotencyConflictError,
    QueueJob,
    QueueLeaseIdentityError,
)
from foundation.clock import utc_now


class SQLiteQueueStore(QueueStoreSupportMixin):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        self._init_db()

    def enqueue(self, job: QueueJob) -> QueueJob:
        """创建一个不可变队列身份，已存在时返回原有状态。"""

        if job.status != "pending" or job.lease_token or job.lease_owner or job.lease_generation:
            raise ValueError("new queue jobs must be unleased and pending")
        now = utc_now()
        payload_json = self._canonical_payload(job.payload)
        tenant_id, owner_user_id, workspace_id = self._job_scope(job)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT * FROM queue_jobs WHERE job_id = ?", (job.job_id,)).fetchone()
            if existing is not None:
                if self._identity(existing) != (job.queue_name, job.action, job.target_uri, payload_json):
                    conn.rollback()
                    raise QueueIdempotencyConflictError(
                        f"queue job id is already bound to another payload: {job.job_id}"
                    )
                conn.execute(
                    "UPDATE queue_jobs SET tenant_id = ?, owner_user_id = ?, workspace_id = ? WHERE job_id = ?",
                    (tenant_id, owner_user_id, workspace_id, job.job_id),
                )
                row = conn.execute("SELECT * FROM queue_jobs WHERE job_id = ?", (job.job_id,)).fetchone()
                conn.commit()
                assert row is not None
                return self._row_to_job(row)
            conn.execute(
                """
                INSERT INTO queue_jobs(
                  job_id, queue_name, action, target_uri, payload_json,
                  tenant_id, owner_user_id, workspace_id,
                  status, leased_until, retry_count, created_at, updated_at, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.queue_name,
                    job.action,
                    job.target_uri,
                    payload_json,
                    tenant_id,
                    owner_user_id,
                    workspace_id,
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

    def purge_target_jobs(
        self,
        *,
        queue_name: str,
        target_uri: str,
        tenant_id: str,
        owner_user_id: str,
    ) -> int:
        """物理清理一个已硬删除文档目标对应的全部过期任务。"""

        values = tuple(str(value or "").strip() for value in (queue_name, target_uri, tenant_id, owner_user_id))
        if not all(values):
            raise ValueError("queue target purge requires exact queue, URI, tenant and owner")
        queue, target, tenant, owner = values
        with self._connect() as conn:
            conn.execute("PRAGMA secure_delete = ON")
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT tenant_id, owner_user_id, payload_json FROM queue_jobs WHERE queue_name = ? AND target_uri = ?",
                (queue, target),
            ).fetchall()
            for row in rows:
                payload = json.loads(str(row["payload_json"] or "{}"))
                if (
                    str(row["tenant_id"] or "") != tenant
                    or str(row["owner_user_id"] or "") != owner
                    or str(payload.get("tenant_id") or "") != tenant
                    or str(payload.get("owner_user_id") or "") != owner
                ):
                    conn.rollback()
                    raise QueueLeaseIdentityError("queue target purge encountered a cross-scope job")
            cursor = conn.execute(
                "DELETE FROM queue_jobs WHERE queue_name = ? AND target_uri = ? "
                "AND tenant_id = ? AND owner_user_id = ?",
                (queue, target, tenant, owner),
            )
            conn.commit()
            removed = max(0, int(cursor.rowcount))
        with self._connect() as conn:
            conn.execute("PRAGMA secure_delete = ON")
            conn.execute("VACUUM")
        return removed

    def lease(
        self,
        queue_name: str,
        *,
        lease_owner: str,
        limit: int = 10,
        lease_seconds: int = 60,
        job_ids: Sequence[str] | None = None,
    ) -> list[QueueJob]:
        """原子选择并租用归当前写事务所有的任务。"""

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
        """归还尚未执行的自有租约，不消耗重试次数。"""

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
        """隔离不可变载荷已经损坏的自有租约。"""

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
        """把过期任务恢复为等待状态，不消耗重试次数。"""

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

    def stats_for_target_prefix(self, *, queue_name: str, target_uri_prefix: str) -> dict[str, int]:
        if not queue_name or not target_uri_prefix:
            raise ValueError("queue_name and target_uri_prefix are required")
        upper = f"{target_uri_prefix}\uffff"
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM queue_jobs "
                "WHERE queue_name = ? AND target_uri >= ? AND target_uri < ? GROUP BY status",
                (queue_name, target_uri_prefix, upper),
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def stats_for_scope(
        self,
        *,
        queue_name: str,
        tenant_id: str,
        owner_user_id: str,
        workspace_ids: Sequence[str] | None = None,
    ) -> dict[str, int]:
        if not queue_name or not tenant_id or not owner_user_id:
            raise ValueError("queue_name, tenant_id, and owner_user_id are required")
        workspace_sql = ""
        scoped_params: list[object] = [queue_name, tenant_id, owner_user_id]
        query_scoped = True
        if workspace_ids is not None:
            workspaces = tuple(dict.fromkeys(str(item) for item in workspace_ids))
            if not workspaces:
                query_scoped = False
            else:
                workspace_sql = f" AND workspace_id IN ({','.join('?' for _ in workspaces)})"
                scoped_params.extend(workspaces)
        blocking_statuses = ("pending", "leased", "dead_letter", "quarantine")
        with self._connect() as conn:
            scoped_rows = (
                conn.execute(
                    "SELECT status, COUNT(*) AS count FROM queue_jobs "
                    "INDEXED BY queue_jobs_scope_status_idx "
                    "WHERE queue_name = ? AND tenant_id = ? AND owner_user_id = ?" + workspace_sql + " GROUP BY status",
                    scoped_params,
                ).fetchall()
                if query_scoped
                else ()
            )
            # 旧式 subject hash 任务无法精确归属 Owner 或 Workspace，因此只在所属
            # Tenant 内保守阻塞全部主体。这里必须保持独立索引查询；合并 OR 条件会
            # 让 SQLite 错误选择覆盖整个队列的租用索引。
            unresolved_rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM queue_jobs "
                "INDEXED BY queue_jobs_scope_status_idx "
                "WHERE queue_name = ? AND tenant_id = ? AND owner_user_id = '' "
                "AND status IN (?, ?, ?, ?) GROUP BY status",
                (queue_name, tenant_id, *blocking_statuses),
            ).fetchall()
        result: dict[str, int] = {}
        for row in (*scoped_rows, *unresolved_rows):
            status = str(row["status"])
            result[status] = result.get(status, 0) + int(row["count"])
        return result


SqliteQueueStore = SQLiteQueueStore

__all__ = ["SQLiteQueueStore", "SqliteQueueStore"]
