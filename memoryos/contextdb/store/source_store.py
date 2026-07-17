"""ContextDB authoritative source protocol with historical public exports."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, Protocol

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

# Annotation-only declarations keep static public API tooling aware of the
# historical names while runtime access remains lazy through ``__getattr__``.
CANONICAL_MEMORY_KINDS: frozenset[str]
CANONICAL_MEMORY_SCHEMA_VERSIONS: frozenset[str]
is_canonical_memory_object: Callable[[ContextObject], bool]
is_canonical_memory_uri: Callable[[str], bool]


class SourceStore(Protocol):
    def read_object(self, uri: str) -> ContextObject: ...

    def write_object(self, obj: ContextObject, content: str | bytes = "") -> None: ...

    def list_objects(self) -> list[ContextObject]: ...

    def read_content(self, uri: str) -> str: ...

    def write_content(self, uri: str, content: str | bytes) -> None: ...

    def soft_delete(self, uri: str, reason: str) -> None: ...

    def delete_object(self, uri: str) -> None: ...


_MEMORY_COMPAT_EXPORTS = frozenset(
    {
        "CANONICAL_MEMORY_KINDS",
        "CANONICAL_MEMORY_SCHEMA_VERSIONS",
        "is_canonical_memory_object",
        "is_canonical_memory_uri",
    }
)


def __getattr__(name: str) -> Any:
    """Resolve historical Memory classification exports without an eager edge."""

    if name not in _MEMORY_COMPAT_EXPORTS:
        raise AttributeError(name)
    classification = import_module("memoryos.memory.integration.classification")
    value = getattr(classification, name)
    globals()[name] = value
    return value

__all__ = [
    "CANONICAL_MEMORY_KINDS",
    "CANONICAL_MEMORY_SCHEMA_VERSIONS",
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
    "is_canonical_memory_object",
    "is_canonical_memory_uri",
]
