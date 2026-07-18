"""ContextDB queue storage protocol and lease models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class QueueJob:
    job_id: str
    queue_name: str
    action: str
    target_uri: str
    payload: dict = field(default_factory=dict)
    status: str = "pending"
    leased_until: str | None = None
    lease_token: str = ""
    lease_generation: int = 0
    lease_owner: str = ""
    retry_count: int = 0
    last_error: str = ""


class LeaseLostError(RuntimeError):
    """A worker no longer owns the queue lease it is settling."""


class QueueLeaseIdentityError(LeaseLostError):
    """A leased job's immutable identity changed in storage."""


class QueueIdempotencyConflictError(ValueError):
    """A job id was reused for a different immutable identity."""


class QueueStore(Protocol):
    def enqueue(self, job: QueueJob) -> QueueJob: ...

    def lease(
        self,
        queue_name: str,
        *,
        lease_owner: str,
        limit: int = 10,
        lease_seconds: int = 60,
        job_ids: Sequence[str] | None = None,
    ) -> list[QueueJob]: ...

    def ack(self, job: QueueJob) -> QueueJob: ...

    def fail(self, job: QueueJob, error: str) -> QueueJob: ...

    def retry(self, job: QueueJob, error: str, *, max_retries: int = 3, retryable: bool = True) -> QueueJob: ...

    def release(self, job: QueueJob, reason: str = "") -> QueueJob: ...

    def quarantine(self, job: QueueJob, error: str) -> QueueJob: ...

    def quarantine_identity_conflict(self, job: QueueJob, error: str) -> QueueJob: ...

    def extend(self, job: QueueJob, *, lease_seconds: int = 60) -> QueueJob: ...

    def get(self, job_id: str) -> QueueJob | None: ...

    def purge_target_jobs(
        self,
        *,
        queue_name: str,
        target_uri: str,
        tenant_id: str,
        owner_user_id: str,
    ) -> int: ...

    def recover_expired_leases(self, *, queue_name: str | None = None) -> int: ...

    def stats(self, *, queue_name: str | None = None) -> dict[str, int]: ...

    def stats_for_target_prefix(self, *, queue_name: str, target_uri_prefix: str) -> dict[str, int]: ...

    def stats_for_scope(
        self,
        *,
        queue_name: str,
        tenant_id: str,
        owner_user_id: str,
        workspace_ids: Sequence[str] | None = None,
    ) -> dict[str, int]: ...


__all__ = [
    "LeaseLostError",
    "QueueIdempotencyConflictError",
    "QueueJob",
    "QueueLeaseIdentityError",
    "QueueStore",
]
