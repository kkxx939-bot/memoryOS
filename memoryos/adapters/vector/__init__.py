"""Stable vector-adapter exports from their implementation owners."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_PUBLIC_ATTRS = {
    "ChromaStore": ("memoryos.adapters.vector.chroma", "ChromaStore"),
    "InMemoryVectorStore": (
        "memoryos.adapters.vector.in_memory",
        "InMemoryVectorStore",
    ),
    "LocalVectorStore": ("memoryos.adapters.vector.in_memory", "LocalVectorStore"),
    "MilvusStore": ("memoryos.adapters.vector.milvus", "MilvusStore"),
    "QdrantStore": ("memoryos.adapters.vector.qdrant", "QdrantStore"),
}

if TYPE_CHECKING:
    from memoryos.adapters.vector.chroma import ChromaStore
    from memoryos.adapters.vector.in_memory import InMemoryVectorStore, LocalVectorStore
    from memoryos.adapters.vector.milvus import MilvusStore
    from memoryos.adapters.vector.qdrant import QdrantStore

__all__ = [
    "ChromaStore",
    "InMemoryVectorStore",
    "LocalVectorStore",
    "MilvusStore",
    "QdrantStore",
]


def __getattr__(name: str) -> Any:
    target = _PUBLIC_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
