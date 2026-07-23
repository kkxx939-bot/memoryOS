"""租赁并重放基于精确 SessionArchive 的提交任务。"""

from __future__ import annotations

import os
import uuid
from typing import Any

from foundation.identity import require_internal_job_namespace
from foundation.readiness import require_session_service_ready, session_service_is_ready
from infrastructure.store.contracts.queue import QueueJob
from infrastructure.store.session.archive_errors import SessionArchiveError
from infrastructure.store.session.commit_group import CommitGroupIntegrityError
from pre.session import SessionArchive
from runtime.session.commit_service import DerivedConsumerError, SessionCommitService


class SessionCommitWorker:
    """先恢复提交组，再结算与不可变归档完全一致的 ``commit`` 任务。"""

    def __init__(self, service: SessionCommitService, *, worker_id: str | None = None) -> None:
        self.service = service
        self.worker_id = worker_id or f"session-commit:{os.getpid()}:{uuid.uuid4().hex}"

    def process_archive(self, archive: SessionArchive) -> dict[str, object]:
        require_session_service_ready(self.service)
        result = self.service.async_commit(archive)
        if not result.done:
            raise RuntimeError("Session commit did not complete")
        return {
            "task_id": result.task_id,
            "status": result.status,
            "done": result.done,
        }

    def process_pending(
        self,
        *,
        batch_size: int = 10,
        lease_seconds: int = 60,
        max_retries: int = 3,
    ) -> dict[str, object]:
        # 提交组恢复和队列租赁之前都必须检查就绪状态；未就绪运行时不能
        # 生成派生投影或修改耐久任务。
        require_session_service_ready(self.service)
        committed = failed = dead_letter = recovered = deferred = 0
        released: list[str] = []
        self.service.commit_group_store.recover_expired_consumers()
        self.service.commit_group_store.recover_abandoned_leases()

        for group in self.service.resumable_commit_groups(limit=batch_size):
            require_session_service_ready(self.service)
            # 过期或废弃租约已经在上方释放；仍处于 RUNNING 的尝试属于存活
            # Worker，当前进程不能与其并发执行。
            if any(item.status == "running" for item in group.consumers.values()):
                deferred += 1
                continue
            try:
                archive = self.service.archive_store.read_archive_at_manifest(
                    group.archive_uri,
                    group.manifest_digest,
                    tenant_id=group.tenant_id,
                )
                result = self.service.resume_startup_commit_group(
                    archive,
                    group_id=group.group_id,
                )
                if not result.done:
                    raise RuntimeError("recovered Session group is incomplete")
                recovered += 1
            except Exception:
                failed += 1
            if not session_service_is_ready(self.service):
                return {
                    "claimed": 0,
                    "committed": committed,
                    "failed": failed,
                    "dead_letter": dead_letter,
                    "recovered": recovered,
                    "deferred": deferred,
                    "released": released,
                    "status": "not_ready",
                }

        # 恢复动作可能在不抛异常的情况下切换就绪状态，因此切换后禁止继续
        # 获取独立队列租约。
        require_session_service_ready(self.service)
        jobs = self.service.queue_store.lease(
            "commit",
            lease_owner=self.worker_id,
            limit=batch_size,
            lease_seconds=lease_seconds,
        )
        for position, job in enumerate(jobs):
            try:
                require_session_service_ready(self.service)
            except RuntimeError:
                if session_service_is_ready(self.service):
                    raise
                released.extend(self._release_unattempted(jobs[position:]))
                break

            try:
                archive = self._archive_for_job(job)
                result = self.service.async_commit(archive)
                if result.done:
                    if not session_service_is_ready(self.service):
                        failed += 1
                        released.extend(self._release_unattempted(jobs[position:]))
                        break
                    self.service.queue_store.ack(job)
                    committed += 1
                    continue
                if not session_service_is_ready(self.service):
                    failed += 1
                    released.extend(self._release_unattempted(jobs[position:]))
                    break
                status = self._retry(
                    job,
                    RuntimeError(result.status),
                    max_retries=max_retries,
                    retryable=self._result_retryable(result.commit_group_status),
                )
            except DerivedConsumerError as exc:
                if not session_service_is_ready(self.service):
                    failed += 1
                    released.extend(self._release_unattempted(jobs[position:]))
                    break
                status = self._retry(job, exc, max_retries=max_retries, retryable=exc.retryable)
            except (
                CommitGroupIntegrityError,
                SessionArchiveError,
                PermissionError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                if not session_service_is_ready(self.service):
                    failed += 1
                    released.extend(self._release_unattempted(jobs[position:]))
                    break
                status = self._retry(job, exc, max_retries=max_retries, retryable=False)
            except OSError as exc:
                if not session_service_is_ready(self.service):
                    failed += 1
                    released.extend(self._release_unattempted(jobs[position:]))
                    break
                status = self._retry(job, exc, max_retries=max_retries, retryable=True)
            except Exception as exc:
                if not session_service_is_ready(self.service):
                    failed += 1
                    released.extend(self._release_unattempted(jobs[position:]))
                    break
                status = self._retry(
                    job,
                    exc,
                    max_retries=max_retries,
                    retryable=self._exception_retryable(exc),
                )
            failed += 1
            dead_letter += int(status == "dead_letter")

        summary: dict[str, object] = {
            "claimed": len(jobs),
            "committed": committed,
            "failed": failed,
            "dead_letter": dead_letter,
            "recovered": recovered,
            "deferred": deferred,
        }
        if released:
            summary["released"] = released
        if not session_service_is_ready(self.service):
            summary["status"] = "not_ready"
        return summary

    def _archive_for_job(self, job: QueueJob) -> SessionArchive:
        expected_keys = {
            "user_id",
            "session_id",
            "tenant_id",
            "archive_digest",
            "manifest_digest",
        }
        if job.queue_name != "commit" or job.action != "async_session_commit":
            raise ValueError("queued Session commit action is unsupported")
        if set(job.payload) != expected_keys:
            raise ValueError("queued Session commit payload has unsupported fields")
        tenant_id = require_internal_job_namespace(job.payload)
        if self.service.archive_store.tenant_id != tenant_id:
            raise ValueError("Session archive store is not bound to the local storage namespace")
        manifest_digest = self._required_string(job.payload, "manifest_digest")
        archive = self.service.archive_store.read_archive_at_manifest(
            job.target_uri,
            manifest_digest,
            tenant_id=tenant_id,
        )
        durable = {
            "user_id": archive.user_id,
            "session_id": archive.session_id,
            "tenant_id": tenant_id,
            "archive_digest": archive.archive_digest,
            "manifest_digest": archive.manifest_digest,
        }
        claimed = {key: self._required_string(job.payload, key) for key in expected_keys}
        if (
            claimed != durable
            or job.job_id != archive.task_id
            or job.target_uri != archive.archive_uri
        ):
            raise ValueError("queued Session commit identity differs from the durable archive")
        return archive

    def _release_unattempted(self, jobs: list[QueueJob]) -> list[str]:
        """丢失就绪状态后释放整批任务，同时保持原重试计数。"""

        released: list[str] = []
        for job in jobs:
            settled = self.service.queue_store.release(job)
            if (
                settled.status != "pending"
                or settled.retry_count != job.retry_count
                or settled.lease_token
                or settled.lease_owner
            ):
                raise RuntimeError("Session commit batch release did not preserve queue state")
            released.append(job.job_id)
        return released

    @staticmethod
    def _required_string(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise TypeError(f"queued Session commit {key} must be a non-empty string")
        return value

    @staticmethod
    def _result_retryable(payload: dict[str, Any]) -> bool:
        consumers = payload.get("consumers")
        if not isinstance(consumers, dict):
            return False
        incomplete: list[dict[str, Any]] = []
        for item in consumers.values():
            if not isinstance(item, dict):
                return False
            if item.get("status") != "completed":
                incomplete.append(item)
        return bool(incomplete) and all(item.get("retryable") is True for item in incomplete)

    @staticmethod
    def _exception_retryable(exc: Exception) -> bool:
        explicit = getattr(exc, "retryable", None)
        return explicit if isinstance(explicit, bool) else False

    def _retry(self, job: QueueJob, exc: Exception, *, max_retries: int, retryable: bool) -> str:
        settled = self.service.queue_store.retry(
            job,
            exc.__class__.__name__,
            max_retries=max_retries,
            retryable=retryable,
        )
        return settled.status


__all__ = ["SessionCommitWorker"]
