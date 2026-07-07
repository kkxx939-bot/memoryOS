from __future__ import annotations

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
    retry_count: int = 0
    last_error: str = ""


@dataclass(frozen=True)
class LockToken:
    lock_key: str
    token: str


class SourceStore(Protocol):
    def read_object(self, uri: str) -> ContextObject: ...

    def write_object(self, obj: ContextObject, content: str | bytes = "") -> None: ...

    def list_objects(self) -> list[ContextObject]: ...

    def read_content(self, uri: str) -> str: ...

    def write_content(self, uri: str, content: str | bytes) -> None: ...

    def soft_delete(self, uri: str, reason: str) -> None: ...


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
    def enqueue(self, job: QueueJob) -> None:
        """Insert or replace by job_id; repeated enqueue of the same job_id must not duplicate work."""
        ...

    def lease(self, queue_name: str, limit: int = 10) -> list[QueueJob]: ...

    def ack(self, job_id: str) -> None: ...

    def fail(self, job_id: str, error: str) -> None: ...


class LockStore(Protocol):
    def acquire(self, lock_key: str, ttl_seconds: int = 30) -> LockToken: ...

    def release(self, token: LockToken) -> None: ...
