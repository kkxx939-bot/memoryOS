"""Authoritative SourceStore protocol for ordinary Context objects."""

from __future__ import annotations

from typing import Protocol

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.index_store import IndexHit as IndexHit
from memoryos.contextdb.store.index_store import IndexStore as IndexStore
from memoryos.contextdb.store.lock_store import LockLostError as LockLostError
from memoryos.contextdb.store.lock_store import LockStore as LockStore
from memoryos.contextdb.store.lock_store import LockToken as LockToken
from memoryos.contextdb.store.queue_store import LeaseLostError as LeaseLostError
from memoryos.contextdb.store.queue_store import QueueIdempotencyConflictError as QueueIdempotencyConflictError
from memoryos.contextdb.store.queue_store import QueueJob as QueueJob
from memoryos.contextdb.store.queue_store import QueueLeaseIdentityError as QueueLeaseIdentityError
from memoryos.contextdb.store.queue_store import QueueStore as QueueStore
from memoryos.contextdb.store.relation_store import RelationStore as RelationStore


class SourceStore(Protocol):
    def read_object(self, uri: str) -> ContextObject: ...

    def write_object(self, obj: ContextObject, content: str | bytes = "") -> None: ...

    def list_objects(self) -> list[ContextObject]: ...

    def read_content(self, uri: str) -> str: ...

    def write_content(self, uri: str, content: str | bytes) -> None: ...

    def soft_delete(self, uri: str, reason: str) -> None: ...

    def delete_object(self, uri: str) -> None: ...


__all__ = [
    "IndexHit",
    "IndexStore",
    "LeaseLostError",
    "LockLostError",
    "LockStore",
    "LockToken",
    "QueueIdempotencyConflictError",
    "QueueJob",
    "QueueLeaseIdentityError",
    "QueueStore",
    "RelationStore",
    "SourceStore",
]
