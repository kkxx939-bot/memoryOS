"""In-memory QueueStore adapter."""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from memoryos.contextdb.store.queue_store import (
    LeaseLostError,
    QueueIdempotencyConflictError,
    QueueJob,
    QueueLeaseIdentityError,
)


class InMemoryQueueStore:
    def __init__(self) -> None:
        self.jobs: dict[str, QueueJob] = {}
        self._guard = threading.RLock()

    def enqueue(self, job: QueueJob) -> QueueJob:
        if job.status != "pending" or job.lease_token or job.lease_owner or job.lease_generation:
            raise ValueError("new queue jobs must be unleased and pending")
        with self._guard:
            existing = self.jobs.get(job.job_id)
            if existing is not None:
                if self._identity(existing) != self._identity(job):
                    raise QueueIdempotencyConflictError(
                        f"queue job id is already bound to another payload: {job.job_id}"
                    )
                return existing
            pending = QueueJob(
                job_id=job.job_id,
                queue_name=job.queue_name,
                action=job.action,
                target_uri=job.target_uri,
                payload=dict(job.payload),
            )
            self.jobs[job.job_id] = pending
            return pending

    def lease(
        self,
        queue_name: str,
        *,
        lease_owner: str,
        limit: int = 10,
        lease_seconds: int = 60,
        job_ids: Sequence[str] | None = None,
    ) -> list[QueueJob]:
        if not isinstance(lease_owner, str) or not lease_owner.strip():
            raise ValueError("lease_owner must be non-empty")
        if limit <= 0:
            return []
        now = datetime.now(timezone.utc)
        leased_until = (now + timedelta(seconds=max(1, lease_seconds))).isoformat()
        allowed = set(job_ids) if job_ids is not None else None
        leased: list[QueueJob] = []
        with self._guard:
            for job in self.jobs.values():
                expired = job.status == "leased" and self._expired(job, now)
                if (
                    job.queue_name == queue_name
                    and (allowed is None or job.job_id in allowed)
                    and (job.status == "pending" or expired)
                ):
                    claimed = QueueJob(
                        **{
                            **job.__dict__,
                            "status": "leased",
                            "leased_until": leased_until,
                            "lease_token": uuid.uuid4().hex,
                            "lease_generation": job.lease_generation + 1,
                            "lease_owner": lease_owner,
                        }
                    )
                    self.jobs[job.job_id] = claimed
                    leased.append(claimed)
                if len(leased) >= limit:
                    break
        return leased

    def ack(self, job: QueueJob) -> QueueJob:
        return self._settle(job, status="done")

    def fail(self, job: QueueJob, error: str) -> QueueJob:
        return self._settle(
            job,
            status="dead_letter",
            retry_count=job.retry_count + 1,
            last_error=str(error)[:500],
        )

    def retry(
        self,
        job: QueueJob,
        error: str,
        *,
        max_retries: int = 3,
        retryable: bool = True,
    ) -> QueueJob:
        retry_count = job.retry_count + 1
        status = "pending" if retryable and retry_count < max_retries else "dead_letter"
        return self._settle(
            job,
            status=status,
            retry_count=retry_count,
            last_error=str(error)[:500],
        )

    def release(self, job: QueueJob, reason: str = "") -> QueueJob:
        """Return an unattempted owned lease without consuming retry budget."""

        return self._settle(
            job,
            status="pending",
            last_error=str(reason)[:500] if reason else job.last_error,
        )

    def quarantine(self, job: QueueJob, error: str) -> QueueJob:
        return self._settle(
            job,
            status="quarantine",
            retry_count=job.retry_count + 1,
            last_error=str(error)[:500],
        )

    def quarantine_identity_conflict(self, job: QueueJob, error: str) -> QueueJob:
        """Quarantine an owned lease whose immutable payload was corrupted."""

        return self._settle(
            job,
            status="quarantine",
            retry_count=job.retry_count + 1,
            last_error=str(error)[:500],
            verify_identity=False,
        )

    def extend(self, job: QueueJob, *, lease_seconds: int = 60) -> QueueJob:
        with self._guard:
            current = self._owned(job)
            extended = QueueJob(
                **{
                    **current.__dict__,
                    "leased_until": (datetime.now(timezone.utc) + timedelta(seconds=max(1, lease_seconds))).isoformat(),
                }
            )
            self.jobs[job.job_id] = extended
        return extended

    def get(self, job_id: str) -> QueueJob | None:
        with self._guard:
            return self.jobs.get(job_id)

    def recover_expired_leases(self, *, queue_name: str | None = None) -> int:
        recovered = 0
        now = datetime.now(timezone.utc)
        with self._guard:
            for job_id, job in tuple(self.jobs.items()):
                if (
                    job.status != "leased"
                    or (queue_name is not None and job.queue_name != queue_name)
                    or not self._expired(job, now)
                ):
                    continue
                self.jobs[job_id] = QueueJob(
                    **{
                        **job.__dict__,
                        "status": "pending",
                        "leased_until": None,
                        "lease_token": "",
                        "lease_owner": "",
                    }
                )
                recovered += 1
        return recovered

    def stats(self, *, queue_name: str | None = None) -> dict[str, int]:
        result: dict[str, int] = {}
        with self._guard:
            for job in self.jobs.values():
                if queue_name is not None and job.queue_name != queue_name:
                    continue
                result[job.status] = result.get(job.status, 0) + 1
        return result

    def stats_for_target_prefix(self, *, queue_name: str, target_uri_prefix: str) -> dict[str, int]:
        result: dict[str, int] = {}
        with self._guard:
            for job in self.jobs.values():
                if job.queue_name != queue_name or not job.target_uri.startswith(target_uri_prefix):
                    continue
                result[job.status] = result.get(job.status, 0) + 1
        return result

    def stats_for_scope(
        self,
        *,
        queue_name: str,
        tenant_id: str,
        owner_user_id: str,
        workspace_ids: Sequence[str] | None = None,
    ) -> dict[str, int]:
        allowed_workspaces = None if workspace_ids is None else {str(item) for item in workspace_ids}
        result: dict[str, int] = {}
        with self._guard:
            for job in self.jobs.values():
                if job.queue_name != queue_name:
                    continue
                job_tenant, job_owner, job_workspace = self._job_scope(job)
                scope_matches = (
                    job_tenant == tenant_id
                    and job_owner == owner_user_id
                    and (allowed_workspaces is None or job_workspace in allowed_workspaces)
                )
                # A pre-scope unresolved job cannot be attributed safely. It
                # blocks scoped CURRENT reads inside its own Tenant until
                # replay drains it, but must not affect another Tenant.
                unknown_unresolved = job_tenant == tenant_id and not job_owner and job.status in {
                    "pending",
                    "leased",
                    "dead_letter",
                    "quarantine",
                }
                if not scope_matches and not unknown_unresolved:
                    continue
                result[job.status] = result.get(job.status, 0) + 1
        return result

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

    def _settle(
        self,
        job: QueueJob,
        *,
        status: str,
        retry_count: int | None = None,
        last_error: str | None = None,
        verify_identity: bool = True,
    ) -> QueueJob:
        with self._guard:
            current = self._owned(job, verify_identity=verify_identity)
            settled = QueueJob(
                **{
                    **current.__dict__,
                    "status": status,
                    "leased_until": None,
                    "lease_token": "",
                    "lease_owner": "",
                    "retry_count": current.retry_count if retry_count is None else retry_count,
                    "last_error": current.last_error if last_error is None else last_error,
                }
            )
            self.jobs[job.job_id] = settled
            return settled

    def _owned(self, job: QueueJob, *, verify_identity: bool = True) -> QueueJob:
        current = self.jobs.get(job.job_id)
        now = datetime.now(timezone.utc)
        if (
            current is None
            or current.status != "leased"
            or current.lease_token != job.lease_token
            or current.lease_generation != job.lease_generation
            or current.lease_owner != job.lease_owner
            or self._expired(current, now)
        ):
            raise LeaseLostError(f"queue lease lost for {job.job_id} generation {job.lease_generation}")
        if verify_identity and self._identity(current) != self._identity(job):
            raise QueueLeaseIdentityError(f"queue immutable identity changed while leased: {job.job_id}")
        return current

    def _expired(self, job: QueueJob, now: datetime) -> bool:
        if not job.leased_until:
            return True
        try:
            expires = datetime.fromisoformat(job.leased_until.replace("Z", "+00:00"))
        except ValueError:
            return True
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return expires.astimezone(timezone.utc) <= now

    def _identity(self, job: QueueJob) -> tuple[str, str, str, str]:
        payload = json.dumps(job.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return job.queue_name, job.action, job.target_uri, payload
