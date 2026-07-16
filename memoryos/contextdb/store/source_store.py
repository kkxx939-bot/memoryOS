"""上下文数据库里的源数据存储。"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Protocol

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation

CANONICAL_MEMORY_KINDS = frozenset({"slot", "claim", "pending_proposal"})
CANONICAL_MEMORY_SCHEMA_VERSIONS = frozenset({"canonical_memory_v2", "canonical_pending_proposal_v1"})


def is_canonical_memory_uri(uri: str) -> bool:
    return "/memories/canonical/" in uri or "/memories/pending/" in uri


def is_canonical_memory_object(obj: ContextObject) -> bool:
    return (
        str(dict(obj.metadata or {}).get("canonical_kind") or "") in CANONICAL_MEMORY_KINDS
        or obj.schema_version in CANONICAL_MEMORY_SCHEMA_VERSIONS
        or is_canonical_memory_uri(obj.uri)
    )


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


class QueueLeaseIdentityError(LeaseLostError):
    """Raised when a leased queue job's immutable identity changes in storage."""


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

    def get_index_metadata(self, uri: str) -> dict | None: ...

    def ordinary_relation_endpoint_state(
        self,
        uri: str,
        *,
        tenant_id: str,
        session_id: str = "",
    ) -> str:
        """Return active, inactive, retired, or missing for relation gating."""
        ...


class RelationStore(Protocol):
    def add_relation(self, relation: ContextRelation) -> None: ...

    def relations_of(
        self,
        uri: str,
        *,
        tenant_id: str | None = None,
        owner_user_id: str | None = None,
        limit: int | None = None,
    ) -> list[ContextRelation]: ...

    def delete_relation(
        self,
        source_uri: str,
        relation_type: str,
        target_uri: str,
        *,
        tenant_id: str | None = None,
    ) -> None: ...

    def delete_projection_relations(
        self,
        uri: str,
        *,
        tenant_id: str,
        catalog_record_key: str,
        limit: int,
    ) -> int:
        """Delete one bounded batch owned by a Catalog projection."""
        ...

    def delete_uri_relations(
        self,
        uri: str,
        *,
        tenant_id: str,
        limit: int,
    ) -> int:
        """Delete one bounded tenant-owned batch when Catalog ownership is gone."""
        ...

    def clear_ordinary_relations(self, *, tenant_id: str, limit: int) -> int:
        """Delete one bounded tenant batch whose Source is non-canonical."""
        ...

    def reconcile_ordinary_relations(
        self,
        relations: Sequence[ContextRelation],
        *,
        tenant_id: str,
    ) -> dict[str, int]:
        """Idempotently publish one bounded Source-derived ordinary batch."""
        ...

    def all_relations(self) -> list[ContextRelation]: ...


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

    def release(self, job: QueueJob, reason: str = "") -> QueueJob: ...

    def quarantine(self, job: QueueJob, error: str) -> QueueJob: ...

    def quarantine_identity_conflict(self, job: QueueJob, error: str) -> QueueJob: ...

    def extend(self, job: QueueJob, *, lease_seconds: int = 60) -> QueueJob: ...

    def get(self, job_id: str) -> QueueJob | None: ...

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
