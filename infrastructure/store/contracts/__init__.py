"""集中导出存储协议；具体实现位于 ``infrastructure.store`` 的子目录。"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from infrastructure.store.contracts.index import CatalogStore as CatalogStore
    from infrastructure.store.contracts.index import IndexHit as IndexHit
    from infrastructure.store.contracts.index import IndexStore as IndexStore
    from infrastructure.store.contracts.index import (
        MemoryDocumentProjectionStore as MemoryDocumentProjectionStore,
    )
    from infrastructure.store.contracts.lock import LockStore as LockStore
    from infrastructure.store.contracts.lock import LockToken as LockToken
    from infrastructure.store.contracts.queue import QueueJob as QueueJob
    from infrastructure.store.contracts.queue import QueueStore as QueueStore
    from infrastructure.store.contracts.relation import RelationStore as RelationStore
    from infrastructure.store.contracts.session_archive import SessionArchiveStore as SessionArchiveStore
    from infrastructure.store.contracts.source import SourceStore as SourceStore
    from infrastructure.store.contracts.vector import VectorHit as VectorHit
    from infrastructure.store.contracts.vector import VectorStore as VectorStore

_EXPORTS = {
    "CatalogStore": ("infrastructure.store.contracts.index", "CatalogStore"),
    "IndexHit": ("infrastructure.store.contracts.index", "IndexHit"),
    "IndexStore": ("infrastructure.store.contracts.index", "IndexStore"),
    "MemoryDocumentProjectionStore": (
        "infrastructure.store.contracts.index",
        "MemoryDocumentProjectionStore",
    ),
    "LockStore": ("infrastructure.store.contracts.lock", "LockStore"),
    "LockToken": ("infrastructure.store.contracts.lock", "LockToken"),
    "QueueJob": ("infrastructure.store.contracts.queue", "QueueJob"),
    "QueueStore": ("infrastructure.store.contracts.queue", "QueueStore"),
    "RelationStore": ("infrastructure.store.contracts.relation", "RelationStore"),
    "SessionArchiveStore": ("infrastructure.store.contracts.session_archive", "SessionArchiveStore"),
    "SourceStore": ("infrastructure.store.contracts.source", "SourceStore"),
    "VectorHit": ("infrastructure.store.contracts.vector", "VectorHit"),
    "VectorStore": ("infrastructure.store.contracts.vector", "VectorStore"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
