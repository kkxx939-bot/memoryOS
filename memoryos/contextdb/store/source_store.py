"""上下文数据库里的源数据存储。"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Protocol

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation


@dataclass(frozen=True)
class IndexHit:
    uri: str
    score: float
    context_type: str
    title: str = ""
    layer: str = "l0"
    metadata: dict = field(default_factory=dict)


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
    """Raised when a queue worker no longer owns the leased job it is settling."""


class QueueIdempotencyConflictError(ValueError):
    """Raised when one queue job id is reused for a different immutable identity."""


@dataclass(frozen=True)
class LockToken:
    lock_key: str
    token: str
    fence: int = 0


class LockLostError(TimeoutError):
    """Raised when a writer no longer owns the lease it was issued."""


class SourceStore(Protocol):
    def read_object(self, uri: str) -> ContextObject: ...

    def write_object(self, obj: ContextObject, content: str | bytes = "") -> None: ...

    def list_objects(self) -> list[ContextObject]: ...

    def read_content(self, uri: str) -> str: ...

    def write_content(self, uri: str, content: str | bytes) -> None: ...

    def soft_delete(self, uri: str, reason: str) -> None: ...

    def delete_object(self, uri: str) -> None:
        """处理 delete object 这一步。"""
        ...


class IndexStore(Protocol):
    def upsert_index(self, obj: ContextObject, content: str = "") -> None: ...

    def delete_index(self, uri: str) -> None: ...

    def indexed_uris(self) -> list[str]: ...

    def clear(self) -> None: ...

    def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]: ...


class RelationStore(Protocol):
    def add_relation(self, relation: ContextRelation) -> None: ...

    def relations_of(
        self,
        uri: str,
        *,
        tenant_id: str | None = None,
        owner_user_id: str | None = None,
    ) -> list[ContextRelation]: ...

    def delete_relation(self, source_uri: str, relation_type: str, target_uri: str) -> None: ...


class QueueStore(Protocol):
    def enqueue(self, job: QueueJob) -> QueueJob:
        """处理 enqueue 这一步。"""
        ...

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

    def retry(
        self,
        job: QueueJob,
        error: str,
        *,
        max_retries: int = 3,
        retryable: bool = True,
    ) -> QueueJob: ...

    def quarantine(self, job: QueueJob, error: str) -> QueueJob: ...

    def get(self, job_id: str) -> QueueJob | None: ...

    def stats(self) -> dict[str, int]: ...


class LockStore(Protocol):
    def acquire(self, lock_key: str, ttl_seconds: int = 30) -> LockToken: ...

    def renew(self, token: LockToken, ttl_seconds: int = 30) -> LockToken: ...

    def assert_owned(self, token: LockToken) -> None: ...

    def fenced(
        self,
        tokens: Sequence[LockToken],
        ttl_seconds: int = 30,
    ) -> AbstractContextManager[None]: ...

    def release(self, token: LockToken) -> None: ...
